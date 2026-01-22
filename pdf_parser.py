"""
PDF Bank Statement Parser

Parses bank statement PDFs and extracts transaction data.
Uses pdfplumber for structured table extraction with regex fallback.
Supports OpenAI API as an enhanced parsing option.
"""

import re
import json
import pdfplumber
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from io import BytesIO


def extract_merchant_guess(description: str) -> str:
    """
    Simple heuristic to extract merchant name from transaction description.
    Removes common prefixes and takes the first merchant-like token sequence.
    """
    if not description or pd.isna(description):
        return "Unknown"
    
    desc = str(description).strip().upper()
    
    # Common prefixes to remove
    prefixes_to_remove = [
        r"^DEBIT CARD PURCHASE\s*",
        r"^ACH\s*",
        r"^ZELLE\s*",
        r"^VENMO\s*",
        r"^PAYPAL\s*",
        r"^CHECK\s*\d+\s*",
        r"^WIRE\s*",
        r"^ONLINE TRANSFER\s*",
        r"^ATM\s*",
        r"^POS\s*",
        r"^PURCHASE\s*",
        r"^PAYMENT\s*",
    ]
    
    for prefix in prefixes_to_remove:
        desc = re.sub(prefix, "", desc, flags=re.IGNORECASE)
    
    # Remove common suffixes (account numbers, reference numbers)
    desc = re.sub(r"\s+\d{4,}.*$", "", desc)  # Remove trailing long numbers
    desc = re.sub(r"\s+REF\s+.*$", "", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\s+ID\s+.*$", "", desc, flags=re.IGNORECASE)
    
    # Take first meaningful words (2-4 words typically)
    tokens = desc.split()
    if not tokens:
        return "Unknown"
    
    # Filter out very short tokens and common words
    meaningful = [t for t in tokens[:4] if len(t) > 2 and t not in ["THE", "AND", "FOR", "TO", "OF"]]
    
    if meaningful:
        return " ".join(meaningful).title()
    
    return tokens[0].title() if tokens else "Unknown"


def parse_date(date_str: str) -> Optional[str]:
    """
    Attempt to parse various date formats and return ISO format string.
    Returns None if parsing fails.
    """
    if not date_str or pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Common date formats
    formats = [
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y-%m-%d",
        "%m/%d/%y",
        "%m-%d-%y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    # Try regex for dates like "01/15/24"
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", date_str)
    if match:
        m, d, y = match.groups()
        if len(y) == 2:
            y = "20" + y
        try:
            dt = datetime(int(y), int(m), int(d))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    
    return None


def extract_amount(amount_str: str) -> Tuple[Optional[float], str]:
    """
    Extract amount from string and infer direction.
    Returns (amount, direction) where direction is 'income' or 'expense'.
    """
    if not amount_str or pd.isna(amount_str):
        return None, "expense"
    
    amount_str = str(amount_str).strip()
    
    # Remove currency symbols and commas
    amount_str = re.sub(r"[$,\s]", "", amount_str)
    
    # Check for parentheses (often indicates negative)
    if "(" in amount_str and ")" in amount_str:
        amount_str = "-" + re.sub(r"[()]", "", amount_str)
    
    try:
        amount = float(amount_str)
        direction = "expense" if amount < 0 else "income"
        return abs(amount), direction
    except (ValueError, TypeError):
        return None, "expense"


def extract_from_table(pdf: pdfplumber.PDF) -> List[Dict]:
    """
    Extract transactions from structured tables in PDF.
    Returns list of transaction dictionaries.
    """
    transactions = []
    
    for page in pdf.pages:
        tables = page.extract_tables()
        
        for table in tables:
            if not table or len(table) < 2:
                continue
            
            # Try to identify header row
            header_row_idx = 0
            for i, row in enumerate(table[:3]):
                if row and any(cell and isinstance(cell, str) and 
                              any(keyword in cell.upper() for keyword in 
                                  ["DATE", "DESCRIPTION", "AMOUNT", "DEBIT", "CREDIT", "BALANCE"]) 
                              for cell in row if cell):
                    header_row_idx = i
                    break
            
            # Find date, description, and amount columns
            header = table[header_row_idx] if header_row_idx < len(table) else []
            if not header:
                continue
            
            date_col = None
            desc_col = None
            amount_col = None
            debit_col = None
            credit_col = None
            
            for idx, cell in enumerate(header):
                if not cell:
                    continue
                cell_upper = str(cell).upper()
                if "DATE" in cell_upper:
                    date_col = idx
                elif "DESCRIPTION" in cell_upper or "DESC" in cell_upper or "MEMO" in cell_upper:
                    desc_col = idx
                elif "AMOUNT" in cell_upper and "DEBIT" not in cell_upper and "CREDIT" not in cell_upper:
                    amount_col = idx
                elif "DEBIT" in cell_upper:
                    debit_col = idx
                elif "CREDIT" in cell_upper:
                    credit_col = idx
            
            # Process data rows
            # Get max column index needed, handling case where all are None
            col_indices = [idx for idx in [date_col, desc_col, amount_col, debit_col, credit_col] if idx is not None]
            max_col_needed = max(col_indices) if col_indices else -1
            
            for row in table[header_row_idx + 1:]:
                if not row or (max_col_needed >= 0 and len(row) <= max_col_needed):
                    continue
                
                # Extract date
                posted_date = None
                if date_col is not None and date_col < len(row):
                    date_str = str(row[date_col]).strip() if row[date_col] else ""
                    posted_date = parse_date(date_str)
                
                # Extract description
                description_raw = ""
                if desc_col is not None and desc_col < len(row):
                    description_raw = str(row[desc_col]).strip() if row[desc_col] else ""
                
                # Extract amount
                amount = None
                direction = "expense"
                
                if debit_col is not None and debit_col < len(row) and row[debit_col]:
                    amount, _ = extract_amount(str(row[debit_col]))
                    direction = "expense"
                elif credit_col is not None and credit_col < len(row) and row[credit_col]:
                    amount, _ = extract_amount(str(row[credit_col]))
                    direction = "income"
                elif amount_col is not None and amount_col < len(row) and row[amount_col]:
                    amount, direction = extract_amount(str(row[amount_col]))
                
                # Only add if we have essential fields
                if posted_date and description_raw and amount is not None:
                    transactions.append({
                        "posted_date": posted_date,
                        "description_raw": description_raw,
                        "merchant_guess": extract_merchant_guess(description_raw),
                        "amount": amount,
                        "direction": direction,
                        "source": "table",
                        "confidence": 0.9
                    })
    
    return transactions


def extract_from_text(pdf: pdfplumber.PDF) -> List[Dict]:
    """
    Fallback: Extract transactions from unstructured text using regex.
    Returns list of transaction dictionaries.
    """
    transactions = []
    
    # Patterns to match transaction lines
    # Format: Date Description Amount
    date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    amount_pattern = r"([$]?\s*-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
    
    # Combined pattern: date, description, amount
    transaction_pattern = re.compile(
        rf"{date_pattern}\s+(.+?)\s+{amount_pattern}",
        re.IGNORECASE | re.MULTILINE
    )
    
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
        
        # Split into lines and look for transaction patterns
        lines = text.split("\n")
        for line in lines:
            # Try to match transaction pattern
            match = transaction_pattern.search(line)
            if match:
                date_str, description, amount_str = match.groups()
                
                posted_date = parse_date(date_str)
                if not posted_date:
                    continue
                
                description_raw = description.strip()
                amount, direction = extract_amount(amount_str)
                
                if amount is not None and description_raw:
                    transactions.append({
                        "posted_date": posted_date,
                        "description_raw": description_raw,
                        "merchant_guess": extract_merchant_guess(description_raw),
                        "amount": amount,
                        "direction": direction,
                        "source": "text_regex",
                        "confidence": 0.6
                    })
    
    return transactions


def detect_statement_period(pdf: pdfplumber.PDF) -> Optional[str]:
    """
    Attempt to detect statement period from PDF text.
    Returns period string if found, None otherwise.
    """
    period_patterns = [
        r"STATEMENT PERIOD[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+TO\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"PERIOD[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+TO\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+TO\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    ]
    
    for page in pdf.pages[:3]:  # Check first 3 pages
        text = page.extract_text()
        if not text:
            continue
        
        for pattern in period_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_date = parse_date(match.group(1))
                end_date = parse_date(match.group(2))
                if start_date and end_date:
                    return f"{start_date} to {end_date}"
    
    return None


def extract_with_openai(file_bytes: bytes, api_key: str) -> List[Dict]:
    """
    Use OpenAI API to extract transactions from PDF.
    Converts PDF to text first, then uses GPT to parse transactions.
    """
    try:
        import openai
        
        # Create OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # First, extract text from PDF
        pdf_file = BytesIO(file_bytes)
        pdf = pdfplumber.open(pdf_file)
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        pdf.close()
        
        if not full_text:
            return []
        
        # Use OpenAI to parse transactions
        prompt = """Extract all financial transactions from the following bank statement text.
Return a JSON array of transactions, where each transaction has:
- posted_date: YYYY-MM-DD format
- description_raw: the full transaction description
- amount: positive number (absolute value)
- direction: "income" or "expense"

Only return valid JSON, no other text. Example format:
[
  {"posted_date": "2024-01-15", "description_raw": "DEBIT CARD PURCHASE AMAZON.COM", "amount": 45.99, "direction": "expense"},
  {"posted_date": "2024-01-16", "description_raw": "DIRECT DEPOSIT SALARY", "amount": 2500.00, "direction": "income"}
]

Bank statement text:
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant. Extract transactions from bank statements and return only valid JSON."},
                {"role": "user", "content": prompt + full_text[:15000]}  # Limit text to avoid token limits
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            # Split by ``` and take the middle part
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                # Remove "json" prefix if present
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
        
        # Parse JSON response
        transactions_data = json.loads(content)
        
        # Normalize to our format
        transactions = []
        for tx in transactions_data:
            if "posted_date" in tx and "description_raw" in tx and "amount" in tx:
                transactions.append({
                    "posted_date": tx["posted_date"],
                    "description_raw": tx["description_raw"],
                    "merchant_guess": extract_merchant_guess(tx["description_raw"]),
                    "amount": float(tx["amount"]),
                    "direction": tx.get("direction", "expense"),
                    "source": "openai",
                    "confidence": 0.95
                })
        
        return transactions
        
    except ImportError:
        raise Exception("OpenAI library not installed. Install with: pip install openai")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse OpenAI response as JSON: {str(e)}")
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")


def parse_bank_statement_pdf(file_bytes: bytes, openai_api_key: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to parse bank statement PDF.
    
    Args:
        file_bytes: PDF file content as bytes
        openai_api_key: Optional OpenAI API key for enhanced parsing
    
    Returns:
        Tuple of (transactions_df, summary_dict)
        - transactions_df: DataFrame with columns: posted_date, description_raw, 
          merchant_guess, amount, direction, source, confidence
        - summary_dict: Dictionary with summary statistics
    """
    transactions = []
    
    # Try OpenAI first if API key is provided
    if openai_api_key:
        try:
            transactions = extract_with_openai(file_bytes, openai_api_key)
            if transactions:
                # OpenAI extraction successful, proceed with results
                pass
            else:
                # Fall back to standard parsing
                openai_api_key = None
        except Exception as e:
            # If OpenAI fails, fall back to standard parsing
            print(f"OpenAI parsing failed, falling back to standard parsing: {str(e)}")
            openai_api_key = None
    
    # Standard parsing (if OpenAI not used or failed)
    if not transactions:
        try:
            pdf_file = BytesIO(file_bytes)
            pdf = pdfplumber.open(pdf_file)
            
            # Try structured table extraction first
            transactions = extract_from_table(pdf)
            
            # If no transactions found, try text extraction
            if not transactions:
                transactions = extract_from_text(pdf)
            
            pdf.close()
            
        except Exception as e:
            raise Exception(f"Failed to parse PDF: {str(e)}")
    
    if not transactions:
        raise Exception("No transactions found in PDF. Please check if the PDF contains transaction data.")
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Remove duplicates (same date, description, amount)
    df = df.drop_duplicates(subset=["posted_date", "description_raw", "amount"], keep="first")
    
    # Sort by date
    df = df.sort_values("posted_date").reset_index(drop=True)
    
    # Calculate summary statistics
    expenses = df[df["direction"] == "expense"]["amount"].sum()
    income = df[df["direction"] == "income"]["amount"].sum()
    net_flow = income - expenses
    
    # Top 10 merchants by absolute spend
    merchant_spend = df.groupby("merchant_guess")["amount"].sum().abs().sort_values(ascending=False)
    top_merchants = merchant_spend.head(10).to_dict()
    
    # Spend by category (all Uncategorized for now)
    category_spend = df.groupby("direction")["amount"].sum().to_dict()
    
    # Detect statement period
    pdf_file = BytesIO(file_bytes)
    pdf = pdfplumber.open(pdf_file)
    statement_period = detect_statement_period(pdf)
    pdf.close()
    
    summary = {
        "statement_period": statement_period,
        "num_transactions": len(df),
        "total_expenses": expenses,
        "total_income": income,
        "net_flow": net_flow,
        "top_merchants": top_merchants,
        "category_spend": category_spend,
    }
    
    return df, summary
