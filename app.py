import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from openai import OpenAI
import io
import re
import json
import tempfile
import os
from typing import List, Dict, Optional

# Try to import PDF libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    PyPDF2 = None

# ============================================================================
# API Keys Configuration
# ============================================================================
# OpenAI API Key - Replace with your actual key
OPENAI_API_KEY = "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"  # Get your key from https://platform.openai.com/api-keys

# Page configuration
st.set_page_config(
    page_title="Finance Assistant MVP v4",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "financial_profile" not in st.session_state:
    st.session_state.financial_profile = ""

# Fixed expense categories
EXPENSE_CATEGORIES = [
    "Housing",
    "Groceries",
    "Dining",
    "Transportation",
    "Shopping",
    "Utilities",
    "Travel",
    "Entertainment",
    "Health",
    "Transfers"
]

# ============================================================================
# Helper Functions
# ============================================================================

def validate_openai_key() -> bool:
    """Check if OpenAI API key is set."""
    return bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here" and OPENAI_API_KEY.strip())


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client if API key is available."""
    if not validate_openai_key():
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from PDF file using PyPDF2.
    Returns extracted text as string.
    """
    if not PDF_AVAILABLE:
        st.error("❌ PyPDF2 is not installed. Please install it with: pip install PyPDF2")
        return ""
    
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {str(e)}")
        return ""


def extract_transactions_with_openai(client: OpenAI, pdf_text: str) -> List[Dict]:
    """
    Use OpenAI API to extract transactions from PDF text.
    Returns list of transaction dictionaries.
    """
    if not pdf_text or len(pdf_text.strip()) < 50:
        return []
    
    try:
        # Truncate text if too long (OpenAI has token limits)
        # Keep first 8000 characters which should be enough for most statements
        text_sample = pdf_text[:8000] if len(pdf_text) > 8000 else pdf_text
        
        prompt = f"""You are a financial assistant that extracts transactions from bank statement text.

Extract all financial transactions from the following bank statement text. For each transaction, identify:
- posted_date (in YYYY-MM-DD format)
- description_raw (the transaction description/merchant name)
- merchant_guess (shortened merchant name, max 50 chars)
- amount (as a positive number)
- direction ("expense" if money going out, "income" if money coming in)

Return a JSON object with a "transactions" array. Each transaction should have these exact fields:
{{
  "posted_date": "YYYY-MM-DD",
  "description_raw": "full description",
  "merchant_guess": "short merchant name",
  "amount": 123.45,
  "direction": "expense" or "income"
}}

Bank statement text:
{text_sample}

Return the JSON object with a "transactions" array:"""

        # Try with JSON mode first (for supported models)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini for better JSON parsing
                messages=[
                    {"role": "system", "content": "You are a financial assistant that extracts transactions from bank statements. Always return valid JSON. Return a JSON object with a 'transactions' array containing all transactions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            # Fallback to regular mode if JSON mode not supported
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial assistant that extracts transactions from bank statements. Always return valid JSON arrays."},
                    {"role": "user", "content": prompt + "\n\nIMPORTANT: Return ONLY a JSON array, no other text."}
                ],
                temperature=0.1,
                max_tokens=2000
            )
        
        result = response.choices[0].message.content
        
        # Parse JSON response
        try:
            # Clean the response - remove markdown code blocks if present
            result_clean = result.strip()
            if result_clean.startswith("```json"):
                result_clean = result_clean[7:]
            if result_clean.startswith("```"):
                result_clean = result_clean[3:]
            if result_clean.endswith("```"):
                result_clean = result_clean[:-3]
            result_clean = result_clean.strip()
            
            # Parse JSON
            parsed = json.loads(result_clean)
            
            # Handle different response structures
            if isinstance(parsed, dict):
                # Look for transactions array in the response
                if "transactions" in parsed:
                    transactions_data = parsed["transactions"]
                elif "data" in parsed:
                    transactions_data = parsed["data"]
                elif "items" in parsed:
                    transactions_data = parsed["items"]
                else:
                    # Check if any value is a list
                    for key, value in parsed.items():
                        if isinstance(value, list):
                            transactions_data = value
                            break
                    else:
                        # No list found, return empty
                        transactions_data = []
            elif isinstance(parsed, list):
                transactions_data = parsed
            else:
                transactions_data = []
        except json.JSONDecodeError as e:
            # Try to extract JSON from text if not pure JSON
            json_match = re.search(r'\{[^{}]*"transactions"[^{}]*\[.*?\]', result, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\[.*?\]', result, re.DOTALL)
            
            if json_match:
                try:
                    transactions_data = json.loads(json_match.group())
                    if isinstance(transactions_data, dict) and "transactions" in transactions_data:
                        transactions_data = transactions_data["transactions"]
                except:
                    st.warning("⚠️ Could not parse OpenAI response as JSON. Using fallback extraction.")
                    return extract_transactions_from_text(pdf_text)
            else:
                st.warning("⚠️ Could not parse OpenAI response as JSON. Using fallback extraction.")
                return extract_transactions_from_text(pdf_text)
        
        # Convert to our transaction format
        transactions = []
        for txn in transactions_data:
            try:
                # Normalize date format
                posted_date = txn.get("posted_date", "")
                if not posted_date:
                    continue
                
                # Ensure amount is positive
                amount = abs(float(txn.get("amount", 0)))
                if amount == 0:
                    continue
                
                direction = txn.get("direction", "expense").lower()
                if direction not in ["expense", "income"]:
                    # Infer from amount sign if present in original
                    direction = "expense"
                
                transactions.append({
                    "posted_date": posted_date,
                    "description_raw": txn.get("description_raw", "Unknown"),
                    "merchant_guess": txn.get("merchant_guess", txn.get("description_raw", "Unknown")[:50]),
                    "amount": round(amount, 2),
                    "direction": direction,
                    "category": "Uncategorized",
                    "source": "statement_pdf",
                    "confidence": 0.9  # High confidence for AI-extracted transactions
                })
            except Exception as e:
                continue
        
        return transactions
        
    except Exception as e:
        st.error(f"❌ Error extracting transactions with OpenAI: {str(e)}")
        st.info("💡 Falling back to pattern-based extraction...")
        return extract_transactions_from_text(pdf_text)


def extract_transactions_from_text(parsed_text: str) -> List[Dict]:
    """
    Extract transaction data from parsed PDF text.
    This is a basic implementation - in production, you'd use more sophisticated parsing.
    """
    transactions = []
    
    if not parsed_text:
        return transactions
    
    # Common patterns for bank statements
    # Date patterns: MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, M/D/YY
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    
    # Amount patterns: $123.45, -$123.45, 123.45, -123.45, (123.45) for negatives
    amount_pattern = r'[\$]?\(?(-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\)?'
    
    lines = parsed_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:  # Skip very short lines
            continue
        
        # Try to find date and amount in the line
        dates = re.findall(date_pattern, line)
        amounts = re.findall(amount_pattern, line)
        
        # Filter amounts - look for reasonable transaction amounts (between $0.01 and $999,999)
        valid_amounts = []
        for amt_str in amounts:
            try:
                amt_clean = amt_str.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                amt_val = float(amt_clean)
                if 0.01 <= abs(amt_val) <= 999999:
                    valid_amounts.append((amt_str, amt_val))
            except:
                continue
        
        if dates and valid_amounts:
            try:
                # Get the first date and the most likely amount (usually the last or largest)
                date_str = dates[0]
                
                # Prefer the last amount (often the balance/transaction amount)
                amount_str, amount = valid_amounts[-1]
                
                # Handle negative amounts in parentheses
                if '(' in amount_str:
                    amount = -abs(amount)
                
                # Determine direction
                direction = "expense" if amount < 0 else "income"
                amount = abs(amount)
                
                # Extract description (everything except date and amount patterns)
                description = line
                # Remove date
                description = re.sub(date_pattern, '', description)
                # Remove amount patterns
                description = re.sub(amount_pattern, '', description)
                # Clean up extra spaces and special chars
                description = re.sub(r'\s+', ' ', description).strip()
                description = description.strip('|').strip()  # Remove table separators
                
                if not description or len(description) < 3:
                    continue
                
                # Try to parse date
                posted_date = date_str
                try:
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            month, day, year = parts
                            if len(year) == 2:
                                year = '20' + year
                            posted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    elif '-' in date_str:
                        # Already in YYYY-MM-DD or similar format
                        parts = date_str.split('-')
                        if len(parts) == 3 and len(parts[0]) == 4:
                            posted_date = date_str
                except:
                    pass
                
                # Merchant guess (first few meaningful words of description)
                words = description.split()
                merchant_guess = ' '.join(words[:3]) if words else "Unknown"
                if len(merchant_guess) > 50:
                    merchant_guess = merchant_guess[:47] + "..."
                
                # Confidence scoring
                # Higher confidence if we have clear tabular structure (multiple aligned elements)
                confidence = 0.9 if '|' in line or '\t' in line else 0.6
                
                transactions.append({
                    "posted_date": posted_date,
                    "description_raw": description,
                    "merchant_guess": merchant_guess,
                    "amount": round(amount, 2),
                    "direction": direction,
                    "category": "Uncategorized",
                    "source": "statement_pdf",
                    "confidence": confidence
                })
            except Exception as e:
                continue
    
    # Remove duplicates (same date, amount, and similar description)
    seen = set()
    unique_transactions = []
    for txn in transactions:
        key = (txn['posted_date'], txn['amount'], txn['description_raw'][:20])
        if key not in seen:
            seen.add(key)
            unique_transactions.append(txn)
    
    return unique_transactions


def categorize_transaction_openai(client: OpenAI, description: str, amount: float) -> tuple:
    """
    Use OpenAI to categorize a transaction.
    Returns (category, rationale).
    """
    try:
        prompt = f"""You are a financial assistant. Categorize this transaction into one of these categories:
{', '.join(EXPENSE_CATEGORIES)}

Transaction: {description}
Amount: ${amount:.2f}

Respond in this exact format:
CATEGORY: [category name]
RATIONALE: [brief explanation]

If the transaction doesn't fit any category well, use "Transfers"."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant that categorizes transactions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        result = response.choices[0].message.content
        
        # Parse response
        category = "Uncategorized"
        rationale = ""
        
        if "CATEGORY:" in result:
            category = result.split("CATEGORY:")[1].split("\n")[0].strip()
            if category not in EXPENSE_CATEGORIES:
                category = "Uncategorized"
        
        if "RATIONALE:" in result:
            rationale = result.split("RATIONALE:")[1].strip()
        
        return category, rationale
    except Exception as e:
        st.error(f"Error categorizing transaction: {str(e)}")
        return "Uncategorized", ""


def generate_financial_profile(client: OpenAI, df: pd.DataFrame) -> str:
    """
    Generate a financial profile using OpenAI based on transaction data.
    """
    if df.empty:
        return "No transaction data available."
    
    # Summarize transactions
    total_expenses = df[df['direction'] == 'expense']['amount'].sum()
    total_income = df[df['direction'] == 'income']['amount'].sum()
    
    spend_by_category = df[df['direction'] == 'expense'].groupby('category')['amount'].sum().to_dict()
    top_merchants = df[df['direction'] == 'expense'].groupby('merchant_guess')['amount'].sum().nlargest(5).to_dict()
    
    summary = f"""
Financial Summary:
- Total Expenses: ${total_expenses:,.2f}
- Total Income: ${total_income:,.2f}
- Net: ${total_income - total_expenses:,.2f}

Spending by Category:
{chr(10).join([f"  {cat}: ${amt:,.2f}" for cat, amt in spend_by_category.items()])}

Top Merchants:
{chr(10).join([f"  {merchant}: ${amt:,.2f}" for merchant, amt in top_merchants.items()])}
"""
    
    try:
        prompt = f"""Based on this financial summary, provide:
1. 2-3 notable spending patterns you observe
2. 2-3 practical, non-judgmental suggestions for improvement

Keep the tone practical and supportive.

{summary}"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial advisor that provides practical, non-judgmental advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating profile: {str(e)}"


def get_chatbot_response(client: OpenAI, user_message: str, df: pd.DataFrame, profile: str) -> str:
    """
    Generate chatbot response with context from transactions and profile.
    """
    # Create context summary
    if df.empty:
        context = "No transaction data available."
    else:
        total_expenses = df[df['direction'] == 'expense']['amount'].sum()
        total_income = df[df['direction'] == 'income']['amount'].sum()
        spend_by_category = df[df['direction'] == 'expense'].groupby('category')['amount'].sum().to_dict()
        
        context = f"""
Financial Context:
- Total Expenses: ${total_expenses:,.2f}
- Total Income: ${total_income:,.2f}
- Spending by Category: {', '.join([f"{k}: ${v:,.2f}" for k, v in spend_by_category.items()])}

Financial Profile:
{profile if profile else "No profile generated yet."}
"""
    
    try:
        # Build messages with system prompt and context
        messages = [
            {"role": "system", "content": "You are a helpful financial assistant. Answer questions based on the provided financial context. Be practical and supportive."}
        ]
        
        # Add context as first user message
        messages.append({"role": "user", "content": f"Here is my financial information:\n{context}\n\nPlease use this information to answer my questions."})
        
        # Add chat history (last 4 exchanges = 8 messages max)
        for msg in st.session_state.chat_history[-8:]:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


# ============================================================================
# Page Functions
# ============================================================================

def home_page():
    """Home page with overview and API key setup."""
    st.title("💰 Finance Assistant MVP")
    st.markdown("Welcome to your personal finance assistant. Upload bank statements, categorize expenses, and get financial insights.")
    
    st.divider()
    
    # API Key Status
    st.subheader("🔑 OpenAI API Key Status")
    if validate_openai_key():
        st.success("✅ OpenAI API key is configured and ready to use!")
    else:
        st.warning("⚠️ OpenAI API key not set. Please set `OPENAI_API_KEY` in the code (line ~12) to use categorization and chatbot features.")
        st.info("💡 Get your API key from: https://platform.openai.com/api-keys")
    
    st.divider()
    
    # Quick Stats
    if not st.session_state.transactions.empty:
        st.subheader("📊 Quick Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        df = st.session_state.transactions
        total_expenses = df[df['direction'] == 'expense']['amount'].sum()
        total_income = df[df['direction'] == 'income']['amount'].sum()
        transaction_count = len(df)
        categories_used = df['category'].nunique()
        
        with col1:
            st.metric("Total Expenses", f"${total_expenses:,.2f}")
        with col2:
            st.metric("Total Income", f"${total_income:,.2f}")
        with col3:
            st.metric("Transactions", transaction_count)
        with col4:
            st.metric("Categories", categories_used)
    else:
        st.info("👆 Upload a bank statement to get started!")


def upload_statement_page():
    """Page for uploading and parsing bank statement PDFs."""
    st.title("📄 Upload Bank Statement")
    
    if not validate_openai_key():
        st.warning("⚠️ Please set your OpenAI API key in the code (OPENAI_API_KEY constant) to enable PDF parsing.")
        return
    
    if not PDF_AVAILABLE:
        st.warning("⚠️ PyPDF2 is not installed. Installing PDF parsing library...")
        st.info("💡 Please run: pip install PyPDF2")
        st.stop()
    
    uploaded_file = st.file_uploader(
        "Upload Bank Statement PDF",
        type=["pdf"],
        help="Upload a PDF bank statement. The system will use OpenAI to extract transactions automatically."
    )
    
    if uploaded_file:
        st.info("📝 Processing PDF with OpenAI... This may take a moment.")
        
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        
        if not pdf_text or len(pdf_text.strip()) < 50:
            st.error("❌ Could not extract text from PDF. Please check the file format.")
            return
        
        st.success("✅ Text extracted from PDF!")
        
        # Show text preview (optional)
        with st.expander("View Extracted Text (Preview)"):
            st.text(pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text)
        
        # Extract transactions using OpenAI
        client = get_openai_client()
        with st.spinner("🤖 Using OpenAI to extract transactions..."):
            transactions = extract_transactions_with_openai(client, pdf_text)
        
        if transactions:
            st.success(f"✅ Extracted {len(transactions)} transactions using OpenAI!")
            
            # Convert to DataFrame
            new_df = pd.DataFrame(transactions)
            
            # Categorize transactions
            if validate_openai_key():
                st.info("🤖 Categorizing transactions with AI...")
                
                progress_bar = st.progress(0)
                for idx, row in new_df.iterrows():
                    category, rationale = categorize_transaction_openai(
                        client,
                        row['description_raw'],
                        row['amount']
                    )
                    new_df.at[idx, 'category'] = category
                    progress_bar.progress((idx + 1) / len(new_df))
                
                st.success("✅ Transactions categorized!")
            
            # Update session state
            if st.session_state.transactions.empty:
                st.session_state.transactions = new_df
            else:
                st.session_state.transactions = pd.concat(
                    [st.session_state.transactions, new_df],
                    ignore_index=True
                )
            
            st.success(f"✅ Added {len(new_df)} transactions to your data!")
            
            # Show preview
            st.subheader("📋 Extracted Transactions")
            st.dataframe(new_df, use_container_width=True)
        else:
            st.warning("⚠️ No transactions found in the PDF. The PDF might not contain recognizable transaction data.")


def dashboard_page():
    """Dashboard with charts and transaction table."""
    st.title("📊 Dashboard & Financial Profile")
    
    if st.session_state.transactions.empty:
        st.info("👆 Upload a bank statement first to see your dashboard!")
        return
    
    df = st.session_state.transactions.copy()
    
    # Generate financial profile if not already generated
    if not st.session_state.financial_profile and validate_openai_key():
        if st.button("Generate Financial Profile"):
            with st.spinner("Generating financial profile..."):
                client = get_openai_client()
                st.session_state.financial_profile = generate_financial_profile(client, df)
    
    # Financial Profile Section
    if st.session_state.financial_profile:
        st.subheader("💡 Financial Profile & Insights")
        st.markdown(st.session_state.financial_profile)
        st.divider()
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_expenses = df[df['direction'] == 'expense']['amount'].sum()
    total_income = df[df['direction'] == 'income']['amount'].sum()
    net = total_income - total_expenses
    transaction_count = len(df)
    
    with col1:
        st.metric("Total Expenses", f"${total_expenses:,.2f}")
    with col2:
        st.metric("Total Income", f"${total_income:,.2f}")
    with col3:
        st.metric("Net", f"${net:,.2f}", delta=f"{net/total_income*100:.1f}%" if total_income > 0 else "0%")
    with col4:
        st.metric("Transactions", transaction_count)
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Spending by Category")
        expense_df = df[df['direction'] == 'expense']
        if not expense_df.empty:
            category_totals = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            fig = px.pie(
                values=category_totals.values,
                names=category_totals.index,
                title="Expense Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expense data to display.")
    
    with col2:
        st.subheader("📈 Top Categories")
        expense_df = df[df['direction'] == 'expense']
        if not expense_df.empty:
            category_totals = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=category_totals.values,
                y=category_totals.index,
                orientation='h',
                title="Top Expense Categories",
                labels={'x': 'Amount ($)', 'y': 'Category'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expense data to display.")
    
    # Top Merchants
    st.subheader("🏪 Top Merchants")
    expense_df = df[df['direction'] == 'expense']
    if not expense_df.empty:
        merchant_totals = expense_df.groupby('merchant_guess')['amount'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=merchant_totals.index,
            y=merchant_totals.values,
            title="Top Merchants by Spending",
            labels={'x': 'Merchant', 'y': 'Amount ($)'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No expense data to display.")
    
    st.divider()
    
    # Transaction Table
    st.subheader("📋 All Transactions")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        direction_filter = st.multiselect(
            "Filter by Direction",
            options=['expense', 'income'],
            default=['expense', 'income']
        )
    with col2:
        category_filter = st.multiselect(
            "Filter by Category",
            options=df['category'].unique().tolist(),
            default=df['category'].unique().tolist()
        )
    with col3:
        min_amount = st.number_input("Min Amount", min_value=0.0, value=0.0)
    
    # Apply filters
    filtered_df = df[
        (df['direction'].isin(direction_filter)) &
        (df['category'].isin(category_filter)) &
        (df['amount'] >= min_amount)
    ].copy()
    
    # Display editable table
    st.markdown("**Edit categories:** Select a transaction row and use the dropdown below to update its category.")
    
    display_df = filtered_df[['posted_date', 'description_raw', 'merchant_guess', 'amount', 'direction', 'category', 'confidence']].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Category editing interface
    if not filtered_df.empty:
        st.markdown("---")
        st.subheader("✏️ Edit Transaction Category")
        
        # Get transaction index for editing
        transaction_indices = filtered_df.index.tolist()
        selected_idx = st.selectbox(
            "Select transaction to edit (by description)",
            options=range(len(transaction_indices)),
            format_func=lambda x: f"{filtered_df.iloc[x]['description_raw'][:50]}... (${filtered_df.iloc[x]['amount']:.2f})"
        )
        
        if selected_idx is not None:
            actual_idx = transaction_indices[selected_idx]
            current_category = filtered_df.iloc[selected_idx]['category']
            
            col1, col2 = st.columns([3, 1])
            with col1:
                new_category = st.selectbox(
                    "New Category",
                    options=EXPENSE_CATEGORIES + ["Uncategorized"],
                    index=EXPENSE_CATEGORIES.index(current_category) if current_category in EXPENSE_CATEGORIES else len(EXPENSE_CATEGORIES)
                )
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("💾 Update Category"):
                    # Update in the main dataframe
                    st.session_state.transactions.at[actual_idx, 'category'] = new_category
                    st.success(f"✅ Updated category to '{new_category}'")
                    st.rerun()
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Transactions (CSV)",
        data=csv,
        file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def chat_assistant_page():
    """Chatbot interface for financial recommendations."""
    st.title("💬 Chat Assistant")
    
    if not validate_openai_key():
        st.warning("⚠️ Please set your OpenAI API key in the code (OPENAI_API_KEY constant) to use the chat assistant.")
        return
    
    if st.session_state.transactions.empty:
        st.info("👆 Upload a bank statement first to get personalized financial advice!")
        return
    
    st.markdown("Ask me questions about your finances! For example:")
    st.markdown("- 'Where am I overspending?'")
    st.markdown("- 'How can I reduce my monthly expenses?'")
    st.markdown("- 'What category should I focus on improving?'")
    
    st.divider()
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your finances...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                client = get_openai_client()
                response = get_chatbot_response(
                    client,
                    user_input,
                    st.session_state.transactions,
                    st.session_state.financial_profile
                )
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================================
# Main App
# ============================================================================

def main():
    # Sidebar Navigation
    st.sidebar.title("💰 Finance Assistant")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Upload Statement", "Dashboard", "Chat Assistant"],
        label_visibility="collapsed"
    )
    
    # Route to appropriate page
    if page == "Home":
        home_page()
    elif page == "Upload Statement":
        upload_statement_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Chat Assistant":
        chat_assistant_page()


if __name__ == "__main__":
    main()
