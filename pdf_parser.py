"""
Deterministic PDF Bank Statement Parser

This module provides robust, deterministic extraction of transactions from bank statement PDFs.
It uses PyMuPDF/pdfplumber for text extraction and implements a state machine for multi-line
transaction merging. LLM is used only as a fallback for problematic sections.

Transaction Sign Logic:
The parser uses a multi-layered approach to determine transaction signs (debit/credit):
1. Column semantics: Detects separate debit/credit columns from headers and uses positional
   heuristics to assign amounts to the correct column.
2. Keyword inference: Falls back to analyzing transaction descriptions for keywords like
   "deposit", "purchase", "fee", etc. to infer sign.
3. Balance delta reconciliation: If balance information is available, validates that
   prev_balance + amount ≈ current_balance, and corrects signs if needed.
4. Validation: Flags inconsistencies (e.g., debit value with positive amount) for debugging.
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a detected transaction section in the PDF."""
    start_pos: int
    end_pos: int
    header: str
    text: str
    account_type: str = "unknown"


@dataclass
class Transaction:
    """Represents a parsed transaction."""
    account_type: str
    post_date: str  # ISO YYYY-MM-DD
    description: str
    amount: float  # signed: debits negative, credits positive
    balance: Optional[float] = None
    raw_line_block: str = ""
    # Raw parsed fields for debugging/validation
    debit_value: Optional[float] = None
    credit_value: Optional[float] = None
    balance_value: Optional[float] = None


def extract_text_pages(pdf_path: str) -> List[str]:
    """
    Extract text from each page of a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of page text strings, preserving page order
    """
    pages_text = []
    
    # Try PyMuPDF first (faster)
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                pages_text.append(text)
            doc.close()
            logger.info(f"Extracted {len(pages_text)} pages using PyMuPDF")
            return pages_text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}, trying pdfplumber")
    
    # Fallback to pdfplumber
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
            logger.info(f"Extracted {len(pages_text)} pages using pdfplumber")
            return pages_text
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    raise Exception("Neither PyMuPDF nor pdfplumber is available. Please install one: pip install PyMuPDF pdfplumber")


def detect_sections(pages_text: List[str]) -> List[Section]:
    """
    Detect transaction sections in the concatenated page text.
    
    Looks for section headers like:
    - ACTIVITY, TRANSACTIONS, ACCOUNT ACTIVITY
    - CHECKING ACTIVITY, SAVINGS ACTIVITY, CREDIT CARD ACTIVITY
    - TRANSACTION DETAIL
    
    Sections end at markers like:
    - Total, Totals, Ending Balance, Closing Balance
    - Summary, or the next section header
    
    Args:
        pages_text: List of page text strings
        
    Returns:
        List of detected Section objects
    """
    # Concatenate pages with page markers
    full_text = ""
    page_boundaries = []
    for i, page_text in enumerate(pages_text):
        page_marker = f"\n---PAGE_{i+1}---\n"
        page_boundaries.append(len(full_text))
        full_text += page_marker + page_text + "\n"
    
    sections = []
    
    # Section start markers (case-insensitive)
    start_patterns = [
        r'(?i)\b(ACTIVITY|TRANSACTIONS|ACCOUNT\s+ACTIVITY|CHECKING\s+ACTIVITY|'
        r'SAVINGS\s+ACTIVITY|CREDIT\s+CARD\s+ACTIVITY|TRANSACTION\s+DETAIL)\b'
    ]
    
    # Section end markers
    end_patterns = [
        r'(?i)\b(TOTAL|TOTALS|ENDING\s+BALANCE|CLOSING\s+BALANCE|SUMMARY)\b',
        r'(?i)^\s*(ACTIVITY|TRANSACTIONS|ACCOUNT\s+ACTIVITY)',  # Next section
    ]
    
    # Find all section starts
    start_matches = []
    for pattern in start_patterns:
        for match in re.finditer(pattern, full_text):
            start_matches.append((match.start(), match.group()))
    
    # Sort by position
    start_matches.sort(key=lambda x: x[0])
    
    # Determine account type from header
    def get_account_type(header: str) -> str:
        header_lower = header.lower()
        if 'checking' in header_lower:
            return 'checking'
        elif 'savings' in header_lower:
            return 'savings'
        elif 'credit' in header_lower or 'card' in header_lower:
            return 'credit'
        else:
            return 'unknown'
    
    # For each section start, find the end
    for i, (start_pos, header) in enumerate(start_matches):
        # Find the end: next section start or end marker
        end_pos = len(full_text)
        
        # Check for next section start
        if i + 1 < len(start_matches):
            end_pos = min(end_pos, start_matches[i + 1][0])
        
        # Check for end markers before next section
        section_text = full_text[start_pos:end_pos]
        for end_pattern in end_patterns:
            end_match = re.search(end_pattern, section_text)
            if end_match:
                # End at the marker (but include it in the section for context)
                end_pos = start_pos + end_match.start()
                break
        
        # Extract section text
        section_text = full_text[start_pos:end_pos]
        account_type = get_account_type(header)
        
        sections.append(Section(
            start_pos=start_pos,
            end_pos=end_pos,
            header=header,
            text=section_text,
            account_type=account_type
        ))
    
    # If no sections found, treat entire document as one section
    if not sections:
        sections.append(Section(
            start_pos=0,
            end_pos=len(full_text),
            header="UNKNOWN",
            text=full_text,
            account_type="unknown"
        ))
    
    logger.info(f"Detected {len(sections)} transaction sections")
    return sections


def detect_column_semantics(section_text: str) -> Dict[str, str]:
    """
    Detect column semantics from section headers.
    
    Looks for column headers that indicate debit, credit, or balance columns.
    Also detects column order for positional heuristics.
    
    Args:
        section_text: Text content of the section (first ~50 lines for headers)
        
    Returns:
        Dictionary mapping column types to header labels: 
        {"debit": "DEBIT", "credit": "CREDIT", "balance": "BALANCE"}
        May also include "column_order" with list of detected column positions.
        Returns empty dict if no clear column structure found.
    """
    lines = section_text.split('\n')[:50]  # Check first 50 lines for headers
    column_map = {}
    
    # Patterns for debit columns
    debit_patterns = [
        r'(?i)\b(DEBIT|DEBITS|WITHDRAWAL|WITHDRAWALS|PAYMENT|PAYMENTS|'
        r'AMOUNT\s+SUBTRACTED|SUBTRACTED|CHARGES?|OUTGOING|'
        r'PAY|PAID|SPENT|EXPENSE)\b'
    ]
    
    # Patterns for credit columns
    credit_patterns = [
        r'(?i)\b(CREDIT|CREDITS|DEPOSIT|DEPOSITS|AMOUNT\s+ADDED|ADDED|'
        r'REFUND|REFUNDS|INTEREST|INCOMING|RECEIVED|INCOME)\b'
    ]
    
    # Patterns for balance columns
    balance_patterns = [
        r'(?i)\b(BALANCE|BAL|RUNNING\s+BALANCE|CURRENT\s+BALANCE|'
        r'ENDING\s+BALANCE|CLOSING\s+BALANCE)\b'
    ]
    
    # Look for header lines (usually contain multiple column names)
    header_line = None
    for line in lines:
        line_upper = line.upper()
        has_debit = False
        has_credit = False
        has_balance = False
        
        # Check for debit column
        for pattern in debit_patterns:
            if re.search(pattern, line):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    column_map["debit"] = match.group(1).upper()
                    has_debit = True
                    break
        
        # Check for credit column
        for pattern in credit_patterns:
            if re.search(pattern, line):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    column_map["credit"] = match.group(1).upper()
                    has_credit = True
                    break
        
        # Check for balance column
        for pattern in balance_patterns:
            if re.search(pattern, line):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    column_map["balance"] = match.group(1).upper()
                    has_balance = True
                    break
        
        # If we found a header line with multiple column types, store it for positional analysis
        if (has_debit or has_credit or has_balance) and not header_line:
            header_line = line
    
    # If we found separate debit/credit columns, also try to determine column order
    if "debit" in column_map and "credit" in column_map and header_line:
        # Find positions of debit and credit in the header
        debit_pos = header_line.upper().find(column_map["debit"])
        credit_pos = header_line.upper().find(column_map["credit"])
        
        if debit_pos >= 0 and credit_pos >= 0:
            column_map["debit_position"] = debit_pos
            column_map["credit_position"] = credit_pos
    
    return column_map


def parse_section(section_text: str, account_type: str = "unknown") -> List[Transaction]:
    """
    Parse a transaction section using a state machine for multi-line merging.
    
    State machine:
    - Start a new transaction when a date-start line is found
    - Append subsequent lines until next date-start or end marker
    - Extract date, description, amount, balance from merged block
    
    Args:
        section_text: Text content of the section
        account_type: Type of account (checking/savings/credit/unknown)
        
    Returns:
        List of parsed Transaction objects
    """
    # Detect column semantics from headers
    column_semantics = detect_column_semantics(section_text)
    
    transactions = []
    lines = section_text.split('\n')
    
    # Date patterns: MM/DD, MM/DD/YYYY, DD/MM, DD/MM/YYYY, YYYY-MM-DD, month-name formats
    date_patterns = [
        r'^\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY, DD/MM/YYYY
        r'^\s*(\d{1,2}[/-]\d{1,2})',  # MM/DD, DD/MM
        r'^\s*(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'^\s*([A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4})',  # Jan 15, 2024 or January 15 2024
        r'^\s*(\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})',  # 15 Jan 2024
    ]
    
    # Column header patterns to skip
    header_patterns = [
        r'(?i)^\s*(DATE|POSTED|TRANSACTION|DESCRIPTION|AMOUNT|DEBIT|CREDIT|BALANCE)',
        r'(?i)^\s*---',
    ]
    
    current_block = []
    current_date = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip header lines
        if any(re.match(pattern, line) for pattern in header_patterns):
            continue
        
        # Check if this line starts with a date
        is_date_start = False
        date_match = None
        for pattern in date_patterns:
            match = re.match(pattern, line)
            if match:
                is_date_start = True
                date_match = match.group(1)
                break
        
        if is_date_start:
            # Save previous block if it exists
            if current_block and current_date:
                txn = _parse_transaction_block(
                    '\n'.join(current_block),
                    current_date,
                    account_type,
                    column_semantics
                )
                if txn:
                    transactions.append(txn)
            
            # Start new block
            current_block = [line]
            current_date = date_match
        else:
            # Append to current block
            if current_block:
                current_block.append(line)
    
    # Process final block
    if current_block and current_date:
        txn = _parse_transaction_block(
            '\n'.join(current_block),
            current_date,
            account_type,
            column_semantics
        )
        if txn:
            transactions.append(txn)
    
    # Apply balance delta reconciliation if we have balances
    if len(transactions) > 1:
        transactions = _reconcile_balance_deltas(transactions)
    
    return transactions


def _infer_sign_from_keywords(description: str, amount_abs: float) -> int:
    """
    Infer transaction sign from keywords in description.
    
    Args:
        description: Transaction description text
        amount_abs: Absolute value of the amount
        
    Returns:
        1 for credit (positive), -1 for debit (negative)
    """
    desc_lower = description.lower()
    
    # Credit keywords (positive amounts)
    credit_keywords = [
        'deposit', 'refund', 'interest', 'dividend', 'salary', 'payroll',
        'transfer in', 'credit', 'payment received', 'reimbursement',
        'cash back', 'reward', 'bonus'
    ]
    
    # Debit keywords (negative amounts)
    debit_keywords = [
        'purchase', 'payment', 'fee', 'charge', 'withdrawal', 'atm',
        'transfer out', 'debit', 'bill pay', 'subscription', 'rent',
        'mortgage', 'loan payment', 'insurance'
    ]
    
    credit_score = sum(1 for kw in credit_keywords if kw in desc_lower)
    debit_score = sum(1 for kw in debit_keywords if kw in desc_lower)
    
    if credit_score > debit_score:
        return 1
    elif debit_score > credit_score:
        return -1
    else:
        # Default: assume debit for expenses (negative)
        return -1


def _parse_transaction_block(block: str, date_str: str, account_type: str, 
                             column_semantics: Dict[str, str] = None) -> Optional[Transaction]:
    """
    Parse a single transaction block (merged multi-line text) into a Transaction.
    
    Uses column semantics when available, with fallbacks to keyword inference and balance reconciliation.
    
    Args:
        block: Multi-line text block for one transaction
        date_str: Date string found at the start
        account_type: Account type
        column_semantics: Dictionary mapping column types to header labels
        
    Returns:
        Transaction object or None if parsing fails
    """
    if column_semantics is None:
        column_semantics = {}
    
    lines = [l.strip() for l in block.split('\n') if l.strip()]
    if not lines:
        return None
    
    # Parse date
    post_date = _parse_date(date_str)
    if not post_date:
        return None
    
    # Extract all numeric amounts (look for currency patterns)
    amount_pattern = r'[\$]?\(?(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\)?'
    amounts = re.findall(amount_pattern, block)
    
    # Filter out dates that match amount pattern
    date_amount_pattern = r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$'
    amounts = [a for a in amounts if not re.match(date_amount_pattern, a.replace(',', '').replace('$', ''))]
    
    if not amounts:
        return None
    
    # Parse all amounts to floats (absolute values)
    parsed_amounts = []
    amount_positions = []
    for amt_str in amounts:
        try:
            clean = amt_str.replace('$', '').replace(',', '').strip()
            # Remove parentheses but don't assume sign yet
            if clean.startswith('(') and clean.endswith(')'):
                clean = clean[1:-1]
            value = abs(float(clean))
            parsed_amounts.append(value)
            pos = block.rfind(amt_str)
            amount_positions.append((pos, amt_str, value))
        except:
            continue
    
    if not parsed_amounts:
        return None
    
    # Sort by position (left to right)
    amount_positions.sort(key=lambda x: x[0])
    
    # Extract description (everything except date and amounts)
    description = block
    # Remove date
    description = re.sub(r'^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*', '', description, flags=re.MULTILINE)
    description = re.sub(r'^\s*\d{4}-\d{2}-\d{2}\s*', '', description, flags=re.MULTILINE)
    # Remove amounts
    for amt_str in amounts:
        description = description.replace(amt_str, '')
    # Clean up
    description = re.sub(r'\s+', ' ', description).strip()
    description = description[:200]  # Limit length
    
    if not description:
        description = "Unknown Transaction"
    
    # Determine transaction amount and sign using column semantics
    debit_value = None
    credit_value = None
    balance_value = None
    transaction_amount = None
    
    # Strategy 1: Use column semantics if detected
    if "debit" in column_semantics and "credit" in column_semantics:
        # Separate debit/credit columns detected
        if len(parsed_amounts) >= 2:
            # Use positional heuristics if available
            if "debit_position" in column_semantics and "credit_position" in column_semantics:
                # Map amounts to positions in the block
                amount_with_positions = []
                for amt_str, amt_val in zip(amounts, parsed_amounts):
                    pos = block.find(amt_str)
                    if pos >= 0:
                        amount_with_positions.append((pos, amt_val, amt_str))
                
                amount_with_positions.sort(key=lambda x: x[0])  # Sort by position
                
                # Determine which amount is in debit column vs credit column
                debit_pos = column_semantics["debit_position"]
                credit_pos = column_semantics["credit_position"]
                
                for pos, amt_val, amt_str in amount_with_positions:
                    # Heuristic: amount closer to debit column header is debit
                    dist_to_debit = abs(pos - debit_pos)
                    dist_to_credit = abs(pos - credit_pos)
                    
                    if dist_to_debit < dist_to_credit:
                        if debit_value is None:
                            debit_value = amt_val
                    else:
                        if credit_value is None:
                            credit_value = amt_val
            else:
                # No positional info: assume first is debit, second is credit
                debit_value = parsed_amounts[0] if parsed_amounts[0] > 0 else None
                credit_value = parsed_amounts[1] if len(parsed_amounts) > 1 and parsed_amounts[1] > 0 else None
            
            # Determine final amount from debit/credit values
            if debit_value and not credit_value:
                transaction_amount = -debit_value
            elif credit_value and not debit_value:
                transaction_amount = credit_value
            elif debit_value and credit_value:
                # Both present: use the non-zero one
                if debit_value > 0 and credit_value == 0:
                    transaction_amount = -debit_value
                elif credit_value > 0 and debit_value == 0:
                    transaction_amount = credit_value
                else:
                    # Both non-zero: use the larger one, sign based on which column it came from
                    # This is a heuristic - in practice, only one should be non-zero
                    if debit_value >= credit_value:
                        transaction_amount = -debit_value
                    else:
                        transaction_amount = credit_value
            else:
                # Neither found: fallback to keyword inference
                sign = _infer_sign_from_keywords(description, max(parsed_amounts))
                transaction_amount = sign * max(parsed_amounts)
                if sign < 0:
                    debit_value = max(parsed_amounts)
                else:
                    credit_value = max(parsed_amounts)
        elif len(parsed_amounts) == 1:
            # Only one amount: need to determine if it's debit or credit
            # Use keyword inference as fallback
            sign = _infer_sign_from_keywords(description, parsed_amounts[0])
            transaction_amount = sign * parsed_amounts[0]
            if sign < 0:
                debit_value = parsed_amounts[0]
            else:
                credit_value = parsed_amounts[0]
    
    # Strategy 2: If we have separate columns but only one amount, use positional heuristics
    elif ("debit" in column_semantics or "credit" in column_semantics) and len(parsed_amounts) == 1:
        # Single amount with column structure: infer from column type
        if "debit" in column_semantics:
            transaction_amount = -parsed_amounts[0]
            debit_value = parsed_amounts[0]
        elif "credit" in column_semantics:
            transaction_amount = parsed_amounts[0]
            credit_value = parsed_amounts[0]
    
    # Strategy 3: Fallback to keyword inference
    else:
        # No clear column structure: use keyword inference
        sign = _infer_sign_from_keywords(description, max(parsed_amounts))
        transaction_amount = sign * max(parsed_amounts)
        if sign < 0:
            debit_value = max(parsed_amounts)
        else:
            credit_value = max(parsed_amounts)
    
    # Extract balance if present (usually the rightmost amount)
    if len(amount_positions) >= 2:
        # Rightmost amount is often balance
        balance_str = amount_positions[-1][1]
        try:
            clean = balance_str.replace('$', '').replace(',', '').strip()
            if clean.startswith('(') and clean.endswith(')'):
                clean = '-' + clean[1:-1]
            balance_value = float(clean)
            balance = balance_value
        except:
            balance = None
    else:
        balance = None
    
    if transaction_amount is None:
        return None
    
    # Validation: flag inconsistencies
    if debit_value and transaction_amount > 0:
        logger.debug(f"Warning: Debit value found but amount is positive: {description[:50]}")
    if credit_value and transaction_amount < 0:
        logger.debug(f"Warning: Credit value found but amount is negative: {description[:50]}")
    
    return Transaction(
        account_type=account_type,
        post_date=post_date,
        description=description,
        amount=transaction_amount,
        balance=balance,
        raw_line_block=block,
        debit_value=debit_value,
        credit_value=credit_value,
        balance_value=balance_value
    )


def _reconcile_balance_deltas(transactions: List[Transaction]) -> List[Transaction]:
    """
    Reconcile transaction signs using balance deltas.
    
    For each transaction, verify that: prev_balance + amount_signed ≈ current_balance
    If not, try swapping the sign and pick the one that reconciles.
    
    Args:
        transactions: List of transactions with balances
        
    Returns:
        List of transactions with corrected signs if needed
    """
    if len(transactions) < 2:
        return transactions
    
    reconciled = []
    prev_balance = None
    tolerance = 0.01
    
    for i, txn in enumerate(transactions):
        if txn.balance is None:
            # No balance available, keep as-is
            reconciled.append(txn)
            continue
        
        if prev_balance is None:
            # First transaction: can't reconcile, but store balance
            reconciled.append(txn)
            prev_balance = txn.balance
            continue
        
        # Calculate expected balance with current sign
        expected_balance = prev_balance + txn.amount
        
        # Check if it reconciles
        if abs(expected_balance - txn.balance) <= tolerance:
            # Reconciles: keep current sign
            reconciled.append(txn)
            prev_balance = txn.balance
        else:
            # Doesn't reconcile: try swapping sign
            swapped_amount = -txn.amount
            swapped_expected = prev_balance + swapped_amount
            
            if abs(swapped_expected - txn.balance) <= tolerance:
                # Swapped sign reconciles: use it
                logger.debug(f"Balance reconciliation: swapped sign for transaction {i}: "
                           f"{txn.amount} -> {swapped_amount}")
                
                # Create corrected transaction
                corrected_txn = Transaction(
                    account_type=txn.account_type,
                    post_date=txn.post_date,
                    description=txn.description,
                    amount=swapped_amount,
                    balance=txn.balance,
                    raw_line_block=txn.raw_line_block,
                    debit_value=txn.credit_value,  # Swap
                    credit_value=txn.debit_value,  # Swap
                    balance_value=txn.balance_value
                )
                reconciled.append(corrected_txn)
                prev_balance = txn.balance
            else:
                # Neither reconciles: keep original but log warning
                logger.warning(f"Balance reconciliation failed for transaction {i}: "
                             f"prev={prev_balance}, amount={txn.amount}, "
                             f"expected={expected_balance}, actual={txn.balance}")
                reconciled.append(txn)
                prev_balance = txn.balance
    
    return reconciled


def _parse_date(date_str: str) -> Optional[str]:
    """
    Parse a date string into ISO format YYYY-MM-DD.
    
    Handles various formats:
    - MM/DD/YYYY, DD/MM/YYYY
    - MM/DD, DD/MM (assumes current year)
    - YYYY-MM-DD
    - Month name formats
    
    Args:
        date_str: Date string to parse
        
    Returns:
        ISO date string (YYYY-MM-DD) or None if parsing fails
    """
    date_str = date_str.strip()
    
    # Try MM/DD/YYYY or DD/MM/YYYY
    match = re.match(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', date_str)
    if match:
        part1, part2, part3 = match.groups()
        year = int(part3)
        if year < 100:
            year += 2000 if year < 50 else 1900
        
        # Heuristic: if part1 > 12, it's DD/MM format
        if int(part1) > 12:
            day, month = int(part1), int(part2)
        else:
            # Assume MM/DD (US format)
            month, day = int(part1), int(part2)
        
        try:
            dt = datetime(year, month, day)
            return dt.strftime('%Y-%m-%d')
        except:
            pass
    
    # Try YYYY-MM-DD
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if match:
        year, month, day = map(int, match.groups())
        try:
            dt = datetime(year, month, day)
            return dt.strftime('%Y-%m-%d')
        except:
            pass
    
    # Try MM/DD (assume current year)
    match = re.match(r'(\d{1,2})[/-](\d{1,2})$', date_str)
    if match:
        part1, part2 = match.groups()
        if int(part1) > 12:
            day, month = int(part1), int(part2)
        else:
            month, day = int(part1), int(part2)
        
        # Use current year
        year = datetime.now().year
        try:
            dt = datetime(year, month, day)
            return dt.strftime('%Y-%m-%d')
        except:
            pass
    
    # Try month name formats
    month_names = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    for month_name, month_num in month_names.items():
        pattern = rf'(?i)({month_name}[a-z]*)\s+(\d{{1,2}}),?\s+(\d{{4}})'
        match = re.match(pattern, date_str)
        if match:
            day, year = int(match.group(2)), int(match.group(3))
            try:
                dt = datetime(year, month_num, day)
                return dt.strftime('%Y-%m-%d')
            except:
                pass
    
    return None


def validate(transactions: List[Transaction], debug_info: Dict) -> Tuple[bool, str]:
    """
    Validate extracted transactions and provide diagnostics.
    
    Args:
        transactions: List of parsed transactions
        debug_info: Dictionary with debug information (sections, text samples, etc.)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(transactions) < 5:
        error_msg = f"Too few transactions extracted: {len(transactions)} (expected at least 5)\n\n"
        
        # Add diagnostics
        error_msg += f"Sections detected: {debug_info.get('num_sections', 0)}\n"
        error_msg += f"Date-start matches found: {len(debug_info.get('date_matches', []))}\n"
        
        if 'text_sample' in debug_info:
            error_msg += f"\nFirst 2000 chars of extracted text:\n{debug_info['text_sample'][:2000]}\n"
        
        if 'date_matches' in debug_info and debug_info['date_matches']:
            error_msg += f"\nFirst 20 date-start matches:\n"
            for i, match in enumerate(debug_info['date_matches'][:20]):
                error_msg += f"  {i+1}. {match}\n"
        
        return False, error_msg
    
    # Check for duplicates (fuzzy match on date+amount+description)
    seen = set()
    duplicates = []
    for txn in transactions:
        key = (txn.post_date, round(txn.amount, 2), txn.description[:50])
        if key in seen:
            duplicates.append(txn)
        seen.add(key)
    
    if len(duplicates) > len(transactions) * 0.1:  # More than 10% duplicates
        return False, f"Too many duplicate transactions detected: {len(duplicates)}/{len(transactions)}"
    
    return True, ""


def llm_repair(block_text: str, client, max_tokens: int = 2500) -> List[Transaction]:
    """
    Use LLM to repair/parse a problematic transaction block.
    
    This is a fallback only - should be called for specific problematic sections,
    not the entire document.
    
    Args:
        block_text: Text block that failed deterministic parsing (max ~1500-2500 tokens)
        client: OpenAI client
        max_tokens: Maximum tokens to send to LLM
        
    Returns:
        List of Transaction objects
    """
    # Truncate if too long
    if len(block_text) > max_tokens * 4:  # Rough char-to-token estimate
        block_text = block_text[:max_tokens * 4]
    
    prompt = f"""Extract all financial transactions from this bank statement text block.

For each transaction, return:
- post_date (ISO YYYY-MM-DD format)
- description (single normalized line, merchant/transaction name)
- amount (signed float: debits negative, credits positive)
- balance (optional, if available)

Return ONLY a valid JSON array of transaction objects. Do not include any explanatory text.

Format:
[
  {{
    "post_date": "YYYY-MM-DD",
    "description": "merchant name or transaction description",
    "amount": -123.45,
    "balance": 1000.00
  }}
]

Text block:
{block_text}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial data extractor. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks
        if "```json" in result:
            result = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
            if result:
                result = result.group(1).strip()
        elif "```" in result:
            result = re.search(r'```\s*(.*?)\s*```', result, re.DOTALL)
            if result:
                result = result.group(1).strip()
        
        # Parse JSON
        data = json.loads(result)
        if not isinstance(data, list):
            data = data.get("transactions", [])
        
        # Convert to Transaction objects
        transactions = []
        for item in data:
            try:
                post_date = item.get("post_date", "")
                if not post_date:
                    continue
                
                amount = float(item.get("amount", 0))
                if amount == 0:
                    continue
                
                balance_val = item.get("balance")
                transactions.append(Transaction(
                    account_type="unknown",
                    post_date=post_date,
                    description=item.get("description", "Unknown")[:200],
                    amount=amount,
                    balance=balance_val,
                    raw_line_block=block_text[:500],  # Store snippet for validation
                    debit_value=-amount if amount < 0 else None,
                    credit_value=amount if amount > 0 else None,
                    balance_value=balance_val
                ))
            except:
                continue
        
        return transactions
        
    except Exception as e:
        logger.error(f"LLM repair failed: {e}")
        return []


def parse_pdf_deterministic(pdf_path: str) -> Tuple[List[Transaction], Dict]:
    """
    Main entry point for deterministic PDF parsing.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (transactions, debug_info)
    """
    debug_info = {
        "num_sections": 0,
        "date_matches": [],
        "text_sample": ""
    }
    
    try:
        # Extract text from pages
        pages_text = extract_text_pages(pdf_path)
        full_text = "\n".join(pages_text)
        debug_info["text_sample"] = full_text[:2000]
        
        # Detect sections
        sections = detect_sections(pages_text)
        debug_info["num_sections"] = len(sections)
        
        # Collect date matches for diagnostics
        date_pattern = r'^\s*(\d{1,2}[/-]\d{1,2}[/-]?\d{0,4})'
        for line in full_text.split('\n')[:100]:  # First 100 lines
            match = re.match(date_pattern, line.strip())
            if match:
                debug_info["date_matches"].append(line.strip()[:100])
        
        # Parse each section
        all_transactions = []
        for section in sections:
            transactions = parse_section(section.text, section.account_type)
            all_transactions.extend(transactions)
        
        return all_transactions, debug_info
        
    except Exception as e:
        logger.error(f"Deterministic parsing failed: {e}")
        raise
