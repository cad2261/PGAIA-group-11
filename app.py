import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

# Set global Plotly default template to light theme
pio.templates.default = "plotly_white"
from openai import OpenAI
import io
import re
import json
import tempfile
import os
from typing import List, Dict, Optional
from store_locator import get_nearby_grocery_stores, filter_stores_by_name

# No external PDF libraries needed - using OpenAI API directly

# ============================================================================
# API Keys Configuration
# ============================================================================
# OpenAI API Key - Check Streamlit secrets first, then fallback to hardcoded value
def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or fallback to hardcoded value."""
    try:
        # Try to get from Streamlit secrets
        if hasattr(st, 'secrets') and 'openai' in st.secrets and 'api_key' in st.secrets.openai:
            return st.secrets.openai.api_key
        # Also try direct access pattern
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets.OPENAI_API_KEY
    except Exception:
        pass
    
    # Fallback to hardcoded value (for local development)
    return "OPENAI_API_KEY"

OPENAI_API_KEY = get_openai_api_key()

# Page configuration
# Try to use custom icon from assets folder, fallback to emoji
import os
icon_paths = [
    os.path.join("assets", "logo.ico"),
    os.path.join("assets", "logo.png"),
    os.path.join("assets", "Logo.ico"),
    os.path.join("assets", "Logo.png"),
    os.path.join("assets", "icon.ico"),
    os.path.join("assets", "icon.png"),
]
page_icon = "💰"  # Default emoji
for icon_path in icon_paths:
    if os.path.exists(icon_path):
        page_icon = icon_path
        break

st.set_page_config(
    page_title="Accountable AI v1",
    page_icon=page_icon,
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
if "pending_transactions" not in st.session_state:
    st.session_state.pending_transactions = pd.DataFrame()
if "show_upload_complete" not in st.session_state:
    st.session_state.show_upload_complete = False
if "review_index" not in st.session_state:
    st.session_state.review_index = 0
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "page_navigated" not in st.session_state:
    st.session_state.page_navigated = False
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "age_range": None,
            "employment_status": None,
            "occupation": None,
            "location_country": None,
            "location_city": None,
            "location_zip": None,
            "profile_completed": False
        }
if "show_questionnaire" not in st.session_state:
    st.session_state.show_questionnaire = False
if "last_location" not in st.session_state:
    st.session_state.last_location = {"country": None, "city": None, "state": None}

# Fixed expense categories
EXPENSE_CATEGORIES = [
    "Housing",
    "Utilities",
    "Food & Dining",
    "Transportation",
    "Shopping",
    "Health",
    "Education",
    "Entertainment",
    "Travel",
    "Subscriptions",
    "Fitness & Personal Care",
    "Miscellaneous"
]

# Income is just a single category (no subcategories for now)
INCOME_CATEGORY = "Income"

# ============================================================================
# Helper Functions
# ============================================================================

def apply_plotly_light_theme(fig):
    """Apply light theme to Plotly charts: white background, dark text, light gridlines."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111111"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E6E6E6", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E6E6E6", zeroline=False)
    return fig


def validate_openai_key() -> bool:
    """Check if OpenAI API key is set."""
    return bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here" and OPENAI_API_KEY.strip())


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client if API key is available."""
    if not validate_openai_key():
        return None
    # Get fresh key from secrets
    key = get_openai_api_key()
    return OpenAI(api_key=key)


def extract_transactions_from_pdf_with_openai(client: OpenAI, pdf_file) -> List[Dict]:
    """
    Use OpenAI API to extract transactions directly from PDF file.
    Uploads PDF to OpenAI and uses Assistants API to extract transactions.
    Returns list of transaction dictionaries.
    """
    try:
        # Reset file pointer and read PDF bytes
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        
        # Save to temporary file for upload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Upload PDF file to OpenAI
            with open(tmp_path, 'rb') as f:
                uploaded_file = client.files.create(
                    file=f,
                    purpose='assistants'
                )
            
            file_id = uploaded_file.id
            
            # Use Assistants API to extract transactions
            prompt = """You are a financial assistant that extracts transactions from bank statement PDFs.

Extract all financial transactions from this bank statement PDF. For each transaction, identify:
- posted_date (in YYYY-MM-DD format)
- description_raw (the transaction description/merchant name)
- merchant_guess (shortened merchant name, max 50 chars)
- amount (as a positive number)
- direction ("expense" if money going out, "income" if money coming in)

CRITICAL: You MUST return ONLY a valid JSON object with a "transactions" array. Do not include any explanatory text, markdown, or code blocks. Return ONLY the raw JSON.

The JSON format must be exactly:
{
  "transactions": [
    {
      "posted_date": "YYYY-MM-DD",
      "description_raw": "full description",
      "merchant_guess": "short merchant name",
      "amount": 123.45,
      "direction": "expense"
    }
  ]
}

Return ONLY the JSON object, nothing else."""
            
            # Create assistant
            assistant = client.beta.assistants.create(
                name="Transaction Extractor",
                instructions=prompt,
                model="gpt-4o",
                tools=[{"type": "code_interpreter"}]
            )
            
            # Create thread
            thread = client.beta.threads.create()
            
            # Create message with file attachment using attachments parameter
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content="Extract all transactions from the uploaded PDF file. Return ONLY a valid JSON object with a 'transactions' array. Do not include any text, markdown, or explanations - just the raw JSON.",
                attachments=[{
                    "file_id": file_id,
                    "tools": [{"type": "code_interpreter"}]
                }]
            )
            
            # Run assistant
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            import time
            max_wait = 60  # 60 seconds max
            waited = 0
            while run.status in ['queued', 'in_progress'] and waited < max_wait:
                time.sleep(2)
                waited += 2
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status != 'completed':
                raise Exception(f"Run failed with status: {run.status}")
            
            # Get response
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            result = messages.data[0].content[0].text.value
            
            # Clean up
            try:
                client.beta.assistants.delete(assistant.id)
                client.files.delete(file_id)
            except:
                pass
            
            # Parse JSON from result
            result_clean = result.strip()
            
            # Remove markdown code blocks if present
            if "```json" in result_clean:
                # Extract content between ```json and ```
                json_match = re.search(r'```json\s*(.*?)\s*```', result_clean, re.DOTALL)
                if json_match:
                    result_clean = json_match.group(1).strip()
            elif "```" in result_clean:
                # Extract content between ``` and ```
                json_match = re.search(r'```\s*(.*?)\s*```', result_clean, re.DOTALL)
                if json_match:
                    result_clean = json_match.group(1).strip()
            
            # Try to find JSON object in the text
            parsed = None
            json_errors = []
            
            # First, try to parse the entire cleaned result
            try:
                parsed = json.loads(result_clean)
            except json.JSONDecodeError as e:
                json_errors.append(f"Direct parse failed: {str(e)}")
                
                # Try to find JSON object with transactions
                patterns = [
                    r'\{[^{}]*"transactions"\s*:\s*\[.*?\]\s*\}',  # { "transactions": [...] }
                    r'\{.*?"transactions".*?\}',  # Any object with transactions
                    r'\{[\s\S]*"transactions"[\s\S]*\}',  # More flexible
                ]
                
                for pattern in patterns:
                    json_match = re.search(pattern, result_clean, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            break
                        except json.JSONDecodeError as e2:
                            json_errors.append(f"Pattern match failed: {str(e2)}")
                            continue
                
                # If still no match, try to find any JSON object
                if parsed is None:
                    json_match = re.search(r'\{.*?\}', result_clean, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                        except:
                            pass
            
            if parsed is None:
                # Show what we got for debugging
                st.warning(f"⚠️ Could not parse JSON from assistant response.")
                with st.expander("View Assistant Response (for debugging)"):
                    st.text(result[:2000] if len(result) > 2000 else result)
                if json_errors:
                    st.debug(f"JSON parsing errors: {json_errors[-1]}")
                raise Exception("Could not parse JSON from assistant response")
            
            # Extract transactions array
            if isinstance(parsed, dict):
                if "transactions" in parsed:
                    transactions_data = parsed["transactions"]
                elif "data" in parsed:
                    transactions_data = parsed["data"]
                else:
                    transactions_data = []
            elif isinstance(parsed, list):
                transactions_data = parsed
            else:
                transactions_data = []
            
            # Convert to transaction format
            transactions = []
            for txn in transactions_data:
                try:
                    posted_date = txn.get("posted_date", "")
                    if not posted_date:
                        continue
                    amount = abs(float(txn.get("amount", 0)))
                    if amount == 0:
                        continue
                    direction = txn.get("direction", "expense").lower()
                    if direction not in ["expense", "income"]:
                        direction = "expense"
                    
                    transactions.append({
                        "posted_date": posted_date,
                        "description_raw": txn.get("description_raw", "Unknown"),
                        "merchant_guess": txn.get("merchant_guess", txn.get("description_raw", "Unknown")[:50]),
                        "amount": round(amount, 2),
                        "direction": direction,
                        "category": "Uncategorized",
                        "source": "statement_pdf",
                        "confidence": 0.9
                    })
                except:
                    continue
            
            return transactions
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Error extracting transactions with OpenAI: {error_msg}")
        return []


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


def categorize_transaction_openai(client: OpenAI, description: str, amount: float, direction: str = "expense") -> tuple:
    """
    Use OpenAI to categorize a transaction.
    Returns (category, rationale).
    For income transactions, just returns "Income". For expenses, categorizes into expense categories.
    """
    try:
        # If it's income, just return "Income" (no subcategories)
        if direction.lower() == "income":
            return INCOME_CATEGORY, "Income transaction"
        
        # For expenses, categorize into expense categories
        prompt = f"""You are a financial assistant. Categorize this expense transaction into one of these categories:
{', '.join(EXPENSE_CATEGORIES)}

Transaction: {description}
Amount: ${amount:.2f}
Type: Expense

Respond in this exact format:
CATEGORY: [category name]
RATIONALE: [brief explanation]

If the transaction doesn't fit any category well, use "Miscellaneous"."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant that categorizes expense transactions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        result = response.choices[0].message.content
        
        # Parse response
        category = "Miscellaneous"
        rationale = ""
        
        if "CATEGORY:" in result:
            category = result.split("CATEGORY:")[1].split("\n")[0].strip()
            # Validate category is in expense categories
            if category not in EXPENSE_CATEGORIES:
                category = "Miscellaneous"
        
        if "RATIONALE:" in result:
            rationale = result.split("RATIONALE:")[1].strip()
        
        return category, rationale
    except Exception as e:
        st.error(f"Error categorizing transaction: {str(e)}")
        return "Miscellaneous" if direction.lower() == "expense" else INCOME_CATEGORY, ""


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
        # Build profile context if available
        profile = st.session_state.user_profile
        profile_context = ""
        
        if profile.get("age_range"):
            profile_context += f"Age range: {profile['age_range']}. "
        if profile.get("employment_status"):
            profile_context += f"Employment: {profile['employment_status']}. "
        if profile.get("occupation"):
            profile_context += f"Occupation: {profile['occupation']}. "
        if profile.get("location_country"):
            location = profile['location_country']
            if profile.get("location_city"):
                location = f"{profile['location_city']}, {location}"
            profile_context += f"Location: {location}. "
        
        # Always include profile context in recommendations
        if not profile_context:
            profile_context = "No profile information provided."
        
        prompt = f"""Based on this financial summary and user profile, provide personalized recommendations:

User Profile: {profile_context}

Financial Summary:
{summary}

IMPORTANT: Always tailor your recommendations based on the user's profile information:
- Age range: Adjust spending benchmarks and savings goals accordingly
- Employment status: Consider income stability and budgeting advice
- Location: Account for cost of living differences
- Occupation: Consider industry-specific financial patterns

Provide:
1. 2-3 notable spending patterns you observe (personalized to their profile)
2. 2-3 practical, non-judgmental suggestions for improvement (tailored to their situation)

Keep the tone practical and supportive, and always reference their profile context when making recommendations."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial advisor that provides practical, non-judgmental advice. Always personalize recommendations based on user profile information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating profile: {str(e)}"


def detect_grocery_store_intent(user_message: str) -> tuple:
    """
    Detect if user is asking about grocery stores.
    Returns (intent_type, store_name, location_from_message)
    intent_type: 'grocery_store_recommendations', 'store_presence_check', or None
    """
    message_lower = user_message.lower()
    
    # Keywords for grocery store recommendations
    recommendation_keywords = [
        "store", "stores", "grocery", "groceries", "supermarket", "supermarkets",
        "where to shop", "cheaper", "save", "savings", "prioritize", "recommend",
        "options", "nearby", "local"
    ]
    
    # Keywords for store presence check
    presence_keywords = [
        "is", "in", "near", "available", "have", "there", "exist"
    ]
    
    has_recommendation = any(kw in message_lower for kw in recommendation_keywords)
    has_presence = any(kw in message_lower for kw in presence_keywords)
    
    # Extract store name (common grocery chains)
    common_stores = ["aldi", "walmart", "target", "kroger", "safeway", "whole foods", 
                     "trader joe", "costco", "sam's club", "publix", "wegmans",
                     "food lion", "stop & shop", "giant", "shoprite", "heb"]
    
    store_name = None
    for store in common_stores:
        if store in message_lower:
            store_name = store.title()
            break
    
    # Extract location from message (simple pattern: "in CITY" or "in CITY, STATE")
    location_from_message = None
    location_match = re.search(r'\bin\s+([A-Za-z\s]+(?:,\s*[A-Z]{2})?)', user_message, re.IGNORECASE)
    if location_match:
        location_from_message = location_match.group(1).strip()
    
    # Determine intent
    if has_presence and store_name:
        return ("store_presence_check", store_name, location_from_message)
    elif has_recommendation:
        return ("grocery_store_recommendations", store_name, location_from_message)
    
    return (None, None, None)


def get_user_location() -> Dict[str, Optional[str]]:
    """
    Get user location from profile or session state.
    Returns dict with country, city, state, zip keys.
    """
    profile = st.session_state.user_profile
    
    location = {
        "country": profile.get("location_country"),
        "city": profile.get("location_city"),
        "state": None,  # We don't capture state separately, but can parse from city if needed
        "zip": profile.get("location_zip")
    }
    
    # If location is in session state (from previous message), use it
    if "last_location" in st.session_state:
        location = st.session_state.last_location
    
    return location


def validate_store_response(response: str, stores: List[Dict]) -> str:
    """
    Validate that LLM response doesn't hallucinate store names.
    If it mentions stores not in the results, replace with safe fallback.
    """
    if not stores:
        return response  # No stores to validate against
    
    # Extract store names from results
    valid_store_names = {store["name"].lower() for store in stores}
    
    # Check if response mentions stores not in results
    words = re.findall(r'\b[A-Z][a-z]+\b', response)
    mentioned_stores = [w for w in words if w.lower() in valid_store_names or 
                       any(store.lower() in w.lower() for store in ["aldi", "walmart", "target", "kroger", "safeway", "whole foods"])]
    
    # If response seems to mention stores but we have results, it's probably fine
    # Only flag if response claims stores exist when we have no results
    if not stores and any(store_kw in response.lower() for store_kw in ["store", "supermarket", "grocery"]):
        return "I couldn't verify nearby stores via live lookup. Please try widening the search radius or provide a ZIP code."
    
    return response


def get_chatbot_response(client: OpenAI, user_message: str, df: pd.DataFrame, profile: str) -> tuple:
    """
    Generate chatbot response with context from transactions and profile.
    Returns (response_text, store_results) where store_results is None or a list of stores.
    """
    # Check for grocery store intent
    intent, store_name, location_from_message = detect_grocery_store_intent(user_message)
    
    store_results = None
    location_used = None
    
    if intent:
        # Get location
        location = get_user_location()
        
        # If location found in message, parse it
        if location_from_message:
            # Try to parse "City, State" or just "City"
            parts = [p.strip() for p in location_from_message.split(",")]
            if len(parts) >= 1:
                location["city"] = parts[0]
            if len(parts) >= 2 and len(parts[1]) == 2:
                location["state"] = parts[1]
            st.session_state.last_location = location
        
        # Only proceed if we have at least country or city
        if location.get("country") or location.get("city"):
            # Fetch stores
            store_results = get_nearby_grocery_stores(location, radius_km=10)
            
            # Filter by store name if specified
            if store_name and store_results:
                store_results = filter_stores_by_name(store_results, store_name)
            
            location_used = location
    
    # Create context summary
    if df.empty:
        context = "No transaction data available."
    else:
        total_expenses = df[df['direction'] == 'expense']['amount'].sum()
        total_income = df[df['direction'] == 'income']['amount'].sum()
        spend_by_category = df[df['direction'] == 'expense'].groupby('category')['amount'].sum().to_dict()
        
        # Always include user profile context (even if empty)
        user_profile = st.session_state.user_profile
        profile_info = "\nUser Profile: "
        profile_parts = []
        
        if user_profile.get("age_range"):
            profile_parts.append(f"Age {user_profile['age_range']}")
        if user_profile.get("employment_status"):
            profile_parts.append(f"Employment: {user_profile['employment_status']}")
        if user_profile.get("occupation"):
            profile_parts.append(f"Occupation: {user_profile['occupation']}")
        if user_profile.get("location_country"):
            loc = user_profile['location_country']
            if user_profile.get("location_city"):
                loc = f"{user_profile['location_city']}, {loc}"
            profile_parts.append(f"Location: {loc}")
        
        if profile_parts:
            profile_info += ", ".join(profile_parts) + "."
        else:
            profile_info += "No profile information available."
        
        context = f"""
Financial Context:
- Total Expenses: ${total_expenses:,.2f}
- Total Income: ${total_income:,.2f}
- Spending by Category: {', '.join([f"{k}: ${v:,.2f}" for k, v in spend_by_category.items()])}
{profile_info}

Financial Profile:
{profile if profile else "No profile generated yet."}

IMPORTANT: Always use the user profile information above to personalize all recommendations. Consider their age, employment status, occupation, and location when providing financial advice."""
    
    try:
        # Build system prompt with anti-hallucination guardrails and profile emphasis
        system_prompt = """You are a helpful financial assistant. Answer questions based on the provided financial context and user profile. Be practical and supportive.

CRITICAL REQUIREMENTS:
1. ALWAYS personalize recommendations using the user profile information (age, employment, occupation, location)
2. When answering questions about grocery stores or store locations:
   - NEVER assume a store exists in a location without verification
   - If store availability is needed, you MUST use the store_locator results provided
   - If store_locator results are provided, use ONLY those stores in your answer
   - Do not mention stores not in the provided list
   - If no stores are found, say "I couldn't verify nearby stores via live lookup" and suggest widening radius or providing ZIP code
   - Always include specific addresses when listing stores
3. For all financial recommendations:
   - Reference the user's age range when discussing savings goals or spending benchmarks
   - Consider their employment status when giving budgeting advice
   - Account for their location's cost of living when making spending comparisons
   - Use their occupation to provide industry-relevant insights"""
        
        # Build messages with system prompt and context
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add context as first user message
        messages.append({"role": "user", "content": f"Here is my financial information:\n{context}\n\nPlease use this information to answer my questions."})
        
        # Add store results if available
        if store_results is not None:
            if store_results:
                stores_text = "\n".join([
                    f"- {store['name']}: {store['address']} ({store['distance_km']} km away)"
                    for store in store_results
                ])
                
                if intent == "store_presence_check":
                    messages.append({
                        "role": "user",
                        "content": f"User asked: '{user_message}'\n\nStore lookup results for {location_used.get('city', location_used.get('country', 'location'))}:\n{stores_text}\n\nAnswer based ONLY on these results. If the store was not found, say so clearly."
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"User asked about grocery stores to save money.\n\nNearby stores found (within 10 km):\n{stores_text}\n\nRecommend stores from this list only. Include addresses. Focus on stores known for lower prices."
                    })
            else:
                loc_str = location_used.get('city') or location_used.get('country') or 'your location'
                messages.append({
                    "role": "user",
                    "content": f"User asked about grocery stores.\n\nStore lookup returned no results for {loc_str}. Tell the user you couldn't verify nearby stores and suggest widening the search radius or providing a ZIP code."
                })
        
        # Add chat history (last 4 exchanges = 8 messages max)
        for msg in st.session_state.chat_history[-8:]:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        
        # Validate response doesn't hallucinate stores
        response_text = validate_store_response(response_text, store_results or [])
        
        return (response_text, store_results)
    except Exception as e:
        return (f"Error generating response: {str(e)}", None)


# ============================================================================
# Page Functions
# ============================================================================

def home_page():
    """Home page with overview and API key setup."""
    # Display logo in header if available
    import os
    logo_paths = [
        os.path.join("assets", "logo.svg"),
        os.path.join("assets", "logo.png"),
        os.path.join("assets", "Logo.svg"),
        os.path.join("assets", "Logo.png"),
    ]
    
    logo_found = False
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            st.image(logo_path, width=250)
            logo_found = True
            break
    
    if not logo_found:
        st.title("💰 Accountable AI v1")
    st.markdown("Welcome to your personal finance assistant. Upload bank statements, categorize expenses, and get financial insights.")
    
    st.divider()
    
    # Profile Status
    profile = st.session_state.user_profile
    if not profile.get("profile_completed"):
        st.info("👤 **Complete your profile** to get personalized financial insights! You can do this from Settings or click the button in the sidebar.")
    else:
        with st.expander("👤 View Profile", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if profile.get("age_range"):
                    st.write(f"**Age Range:** {profile['age_range']}")
                if profile.get("employment_status"):
                    st.write(f"**Employment:** {profile['employment_status']}")
            with col2:
                if profile.get("occupation"):
                    st.write(f"**Occupation:** {profile['occupation']}")
                location = None
                if profile.get("location_city") and profile.get("location_country"):
                    location = f"{profile['location_city']}, {profile['location_country']}"
                    if profile.get("location_zip"):
                        location = f"{location} {profile['location_zip']}"
                elif profile.get("location_country"):
                    location = profile['location_country']
                    if profile.get("location_zip"):
                        location = f"{location} {profile['location_zip']}"
                elif profile.get("location_zip"):
                    location = profile['location_zip']
                if location:
                    st.write(f"**Location:** {location}")
            if st.button("✏️ Edit Profile", use_container_width=True):
                st.session_state.show_questionnaire = True
                st.rerun()
    
    st.divider()
    
    # API Key Status
    st.subheader("🔑 OpenAI API Key Status")
    if validate_openai_key():
        # Check if key is from secrets or hardcoded
        key_source = "code"
        try:
            if hasattr(st, 'secrets') and (('openai' in st.secrets and 'api_key' in st.secrets.openai) or 'OPENAI_API_KEY' in st.secrets):
                key_source = "Streamlit secrets"
        except:
            pass
        st.success(f"✅ OpenAI API key is configured and ready to use! (from {key_source})")
    else:
        st.warning("⚠️ OpenAI API key not set.")
        st.info("💡 **Set your API key in one of these ways:**")
        st.markdown("""
        1. **Streamlit Secrets** (recommended for production):
           - Create `.streamlit/secrets.toml` file with:
           ```toml
           [openai]
           api_key = "sk-your-key-here"
           ```
        2. **Code** (for local development):
           - Update `OPENAI_API_KEY` in the code
        """)
        st.info("Get your API key from: https://platform.openai.com/api-keys")
    
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
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("👆 Upload a bank statement to get started!", use_container_width=True, type="primary"):
                # Navigate to Upload Statement tab
                st.session_state.page = "Upload Statement"
                st.session_state.page_navigated = True
                st.rerun()
                st.rerun()


def parse_csv_transactions(csv_file) -> List[Dict]:
    """
    Parse CSV file and extract transactions.
    Tries to automatically detect date, description, and amount columns.
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        if df.empty:
            return []
        
        # Show CSV preview
        with st.expander("View CSV Preview"):
            st.dataframe(df.head(), use_container_width=True)
        
        # Try to identify columns automatically
        date_col = None
        desc_col = None
        amount_col = None
        
        # Look for common column names
        for col in df.columns:
            col_lower = col.lower()
            if not date_col and any(x in col_lower for x in ['date', 'posted', 'transaction_date']):
                date_col = col
            if not desc_col and any(x in col_lower for x in ['description', 'merchant', 'payee', 'details', 'memo', 'note']):
                desc_col = col
            if not amount_col and any(x in col_lower for x in ['amount', 'value', 'debit', 'credit', 'balance']):
                amount_col = col
        
        # If not found, use first few columns as guesses
        if not date_col and len(df.columns) > 0:
            date_col = df.columns[0]
        if not desc_col and len(df.columns) > 1:
            desc_col = df.columns[1]
        if not amount_col and len(df.columns) > 2:
            amount_col = df.columns[2]
        
        # Let user select columns if auto-detection unclear
        if not all([date_col, desc_col, amount_col]):
            st.info("💡 Please select the columns for your CSV file:")
            col1, col2, col3 = st.columns(3)
            with col1:
                date_col = st.selectbox("Date Column", options=df.columns.tolist(), index=0 if date_col else None)
            with col2:
                desc_col = st.selectbox("Description Column", options=df.columns.tolist(), index=1 if desc_col else None)
            with col3:
                amount_col = st.selectbox("Amount Column", options=df.columns.tolist(), index=2 if amount_col else None)
        
        # Extract transactions
        transactions = []
        for idx, row in df.iterrows():
            try:
                # Get date
                date_val = str(row[date_col]) if date_col else ""
                # Try to parse date
                try:
                    if pd.notna(date_val):
                        # Try common date formats
                        if '/' in date_val:
                            parts = date_val.split('/')
                            if len(parts) == 3:
                                month, day, year = parts
                                if len(year) == 2:
                                    year = '20' + year
                                posted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            else:
                                posted_date = date_val
                        elif '-' in date_val:
                            posted_date = date_val
                        else:
                            posted_date = date_val
                    else:
                        continue
                except:
                    posted_date = date_val
                
                # Get description
                description = str(row[desc_col]) if desc_col and pd.notna(row[desc_col]) else "Unknown"
                
                # Get amount
                try:
                    amount = abs(float(row[amount_col])) if amount_col and pd.notna(row[amount_col]) else 0
                except:
                    amount = 0
                
                if amount == 0:
                    continue
                
                # Determine direction - positive amounts are usually income
                direction = "expense"
                if amount_col:
                    try:
                        amt_val = float(row[amount_col])
                        if amt_val > 0:
                            # Check column name for hints
                            if 'credit' in str(amount_col).lower() or 'deposit' in str(amount_col).lower() or 'income' in str(amount_col).lower():
                                direction = "income"
                            elif 'debit' in str(amount_col).lower() or 'withdrawal' in str(amount_col).lower() or 'expense' in str(amount_col).lower():
                                direction = "expense"
                            else:
                                # Default: positive amounts are usually income
                                direction = "income"
                        else:
                            # Negative amounts are expenses
                            direction = "expense"
                    except:
                        pass
                else:
                    # If no amount column, default to expense
                    direction = "expense"
                
                # Merchant guess
                merchant_guess = description[:50] if len(description) > 50 else description
                
                transactions.append({
                    "posted_date": posted_date,
                    "description_raw": description,
                    "merchant_guess": merchant_guess,
                    "amount": round(amount, 2),
                    "direction": direction,
                    "category": "Uncategorized",
                    "source": "statement_csv",
                    "confidence": 0.9
                })
            except Exception as e:
                continue
        
        return transactions
        
    except Exception as e:
        st.error(f"❌ Error parsing CSV: {str(e)}")
        return []


def show_upload_complete_screen():
    """Show 'all done' screen after successfully reviewing all transactions."""
    # Get the transactions that were just added (from the main transactions dataframe)
    df = st.session_state.transactions.copy()
    
    # If we have recent transactions, show summary of those
    # Otherwise show overall summary
    if df.empty:
        st.info("No transactions to display.")
        st.session_state.show_upload_complete = False
        return
    
    st.markdown("---")
    
    # Success message with icon
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">✅</h1>
            <h2 style="color: #4CAF50; margin-bottom: 1rem;">All Done!</h2>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
                Successfully processed your bank statement
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary statistics
    st.markdown("### 📊 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_txns = len(df)
        st.metric("Total Transactions", total_txns)
    
    with col2:
        expenses = len(df[df['direction'] == 'expense'])
        st.metric("Expenses", expenses)
    
    with col3:
        income = len(df[df['direction'] == 'income'])
        st.metric("Income", income)
    
    with col4:
        total_amount = df['amount'].sum()
        st.metric("Total Amount", f"${total_amount:,.2f}")
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📊 View Dashboard", type="primary", use_container_width=True):
            st.session_state.show_upload_complete = False
            st.session_state.page = "Dashboard"
            st.session_state.page_navigated = True
            st.rerun()
    
    with col2:
        if st.button("🔄 Upload Another", use_container_width=True):
            st.session_state.show_upload_complete = False
            st.session_state.pending_transactions = pd.DataFrame()
            st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
            st.rerun()
    
    st.markdown("---")
    
    # Preview of transactions
    with st.expander("👀 Preview Recent Transactions", expanded=False):
        preview_df = df.tail(10)[['posted_date', 'description_raw', 'merchant_guess', 'amount', 'direction', 'category']]
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        if len(df) > 10:
            st.caption(f"Showing last 10 of {len(df)} total transactions")


def upload_statement_page():
    """Page for uploading and parsing bank statements (PDF or CSV)."""
    st.title("📄 Upload Bank Statement")
    
    # Show "all done" screen if review was just completed
    if st.session_state.get("show_upload_complete", False):
        show_upload_complete_screen()
        return
    
    # If we have pending transactions, show classifier first and reset uploader
    if not st.session_state.pending_transactions.empty:
        show_transaction_review_ui()
        return
    
    # Use key so uploader resets after processing (avoids re-processing on rerun)
    upload_key = st.session_state.get("upload_key", 0)
    uploaded_file = st.file_uploader(
        "Upload Bank Statement (PDF or CSV)",
        type=["pdf", "csv"],
        help="Upload a PDF or CSV bank statement. PDFs will be processed with OpenAI, CSVs will be parsed directly.",
        key=f"upload_{upload_key}"
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == "pdf":
            if not validate_openai_key():
                st.warning("⚠️ Please set your OpenAI API key in the code (OPENAI_API_KEY constant) to enable PDF parsing.")
                return
            
            st.info("📝 Processing PDF with OpenAI... This may take a moment.")
            
            # Extract transactions using OpenAI directly from PDF
            client = get_openai_client()
            with st.spinner("🤖 Using OpenAI to extract transactions from PDF (this may take 30-60 seconds)..."):
                transactions = extract_transactions_from_pdf_with_openai(client, uploaded_file)
        
        elif file_type == "csv":
            st.info("📝 Processing CSV file...")
            with st.spinner("Parsing CSV file..."):
                transactions = parse_csv_transactions(uploaded_file)
        
        else:
            st.error("❌ Unsupported file type. Please upload a PDF or CSV file.")
            return
        
        if transactions:
            st.success(f"✅ Extracted {len(transactions)} transactions!")
            
            # Convert to DataFrame
            new_df = pd.DataFrame(transactions)
            
            # Auto-categorize transactions if OpenAI key is available (as suggestions)
            if validate_openai_key():
                client = get_openai_client()
                st.info("🤖 Generating category suggestions with AI...")
                
                progress_bar = st.progress(0)
                for idx, row in new_df.iterrows():
                    category, rationale = categorize_transaction_openai(
                        client,
                        row['description_raw'],
                        row['amount'],
                        row.get('direction', 'expense')
                    )
                    new_df.at[idx, 'category'] = category
                    progress_bar.progress((idx + 1) / len(new_df))
                
                st.success("✅ Category suggestions generated!")
            else:
                st.info("💡 Set your OpenAI API key to get automatic category suggestions.")
            
            # Store in pending transactions, bump upload key to reset uploader
            st.session_state.pending_transactions = new_df
            st.session_state.review_index = 0
            st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
            st.rerun()
        else:
            # No transactions found
            if file_type == "pdf":
                st.warning("⚠️ No transactions found in the PDF. The PDF might not contain recognizable transaction data.")
            elif file_type == "csv":
                st.warning("⚠️ No transactions found in the CSV. Please check the file format.")


def clean_merchant_name(merchant: str, description: str = "") -> str:
    """
    Clean up merchant name by removing common prefixes/suffixes and extra whitespace.
    Fallback function when OpenAI API is not available.
    """
    if not merchant:
        merchant = description
    
    # Remove common prefixes
    prefixes = ["PAYMENT TO ", "PAYMENT - ", "DEBIT CARD PURCHASE ", "ACH DEBIT ", 
                "ACH CREDIT ", "ONLINE PAYMENT ", "ELECTRONIC PAYMENT ", "CHECK ",
                "AUTOMATIC PAYMENT ", "RECURRING PAYMENT ", "BILL PAY "]
    for prefix in prefixes:
        if merchant.upper().startswith(prefix):
            merchant = merchant[len(prefix):].strip()
    
    # Remove common suffixes
    suffixes = [" #", " -", " *", " POS", " ONLINE", " MOBILE"]
    for suffix in suffixes:
        if suffix in merchant.upper():
            idx = merchant.upper().find(suffix)
            merchant = merchant[:idx].strip()
    
    # Remove extra whitespace and clean up
    merchant = ' '.join(merchant.split())
    
    # Truncate if too long
    if len(merchant) > 50:
        merchant = merchant[:47] + "..."
    
    return merchant.strip() if merchant.strip() else "Unknown"


def clean_merchant_name_openai(client: OpenAI, merchant: str, description: str = "") -> str:
    """
    Use OpenAI API to clean and standardize merchant names.
    Returns a clean, standardized merchant name (e.g., "Amazon" instead of "AMAZON.COM #1234").
    Falls back to simple cleaning if API call fails.
    """
    if not merchant:
        merchant = description
    
    if not merchant or merchant.strip() == "Unknown":
        return "Unknown"
    
    # Check cache first to avoid repeated API calls
    cache_key = f"merchant_clean_{merchant}_{description}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    try:
        prompt = f"""You are a financial assistant. Clean and standardize this merchant name from a bank transaction.

Raw merchant/description: "{merchant}"
Full transaction description: "{description}"

Return ONLY a clean, standardized merchant name. Examples:
- "AMAZON.COM #1234" → "Amazon"
- "DEBIT CARD PURCHASE STARBUCKS STORE" → "Starbucks"
- "PAYMENT TO WALMART SUPERCENTER" → "Walmart"
- "ACH DEBIT NETFLIX.COM" → "Netflix"

Rules:
- Remove payment prefixes (PAYMENT TO, DEBIT CARD PURCHASE, ACH DEBIT, etc.)
- Remove transaction IDs, reference numbers, and suffixes (#, POS, ONLINE, etc.)
- Return the actual business/merchant name only
- Keep it short (max 30 characters)
- Use proper capitalization (e.g., "Amazon", "Walmart", "Starbucks")
- If unclear, return the most likely merchant name

Return ONLY the cleaned merchant name, nothing else."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant that standardizes merchant names from bank transactions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        cleaned = response.choices[0].message.content.strip()
        
        # Remove quotes if present
        cleaned = cleaned.strip('"\'')
        
        # Truncate if too long
        if len(cleaned) > 50:
            cleaned = cleaned[:47] + "..."
        
        # Cache the result
        st.session_state[cache_key] = cleaned
        return cleaned if cleaned else clean_merchant_name(merchant, description)
        
    except Exception as e:
        # Fall back to simple cleaning if API fails
        return clean_merchant_name(merchant, description)


def show_transaction_review_ui():
    """Show UI for reviewing and approving transaction categories."""
    pending_df = st.session_state.pending_transactions.copy()
    current_idx = st.session_state.review_index
    
    if current_idx >= len(pending_df):
        # All transactions reviewed, clean any remaining merchant names with OpenAI before saving
        client = get_openai_client()
        if client and validate_openai_key():
            for idx in pending_df.index:
                original_merchant = pending_df.at[idx, 'merchant_guess']
                description = pending_df.at[idx, 'description_raw']
                cleaned = clean_merchant_name_openai(client, original_merchant, description)
                pending_df.at[idx, 'merchant_guess'] = cleaned
        
        # All transactions reviewed, add remaining to database
        if not pending_df.empty:
            if st.session_state.transactions.empty:
                st.session_state.transactions = pending_df
            else:
                st.session_state.transactions = pd.concat(
                    [st.session_state.transactions, pending_df],
                    ignore_index=True
                )
        
        # Clear pending transactions and show success screen
        st.session_state.pending_transactions = pd.DataFrame()
        st.session_state.review_index = 0
        st.session_state.show_upload_complete = True
        st.rerun()
        return
    
    st.divider()
    st.subheader("📝 Review & Categorize Transactions")
    
    # Progress indicator
    progress_pct = (current_idx / len(pending_df)) * 100
    st.progress(progress_pct / 100)
    st.caption(f"Reviewing transaction {current_idx + 1} of {len(pending_df)}")
    
    # Get current transaction
    current_txn = pending_df.iloc[current_idx]
    
    # Display transaction details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Transaction Details")
        st.markdown(f"**Date:** {current_txn['posted_date']}")
        st.markdown(f"**Description:** {current_txn['description_raw']}")
        st.markdown(f"**Merchant:** {current_txn['merchant_guess']}")
        st.markdown(f"**Amount:** ${current_txn['amount']:,.2f}")
        st.markdown(f"**Type:** {current_txn['direction'].title()}")
    
    with col2:
        st.markdown("### Transaction Type")
        current_direction = current_txn.get('direction', 'expense').lower()
        is_income = current_direction == 'income'
        
        # Allow user to change direction
        new_direction = st.radio(
            "Transaction Type:",
            ["Income", "Expense"],
            index=1 if current_direction == 'expense' else 0,
            horizontal=True
        )
        new_direction_lower = new_direction.lower()
        
        st.markdown("### Current Category")
        current_category = current_txn.get('category', 'Uncategorized')
        if current_category and current_category != 'Uncategorized':
            st.info(f"💡 Suggested: **{current_category}**")
        else:
            st.info("No suggestion")
    
    st.divider()
    
    # Category selection UI
    st.markdown("### Select Category")
    
    # Determine which categories to show based on transaction direction
    is_income = new_direction_lower == 'income'
    if is_income:
        # Income just uses "Income" category
        available_categories = [INCOME_CATEGORY]
        default_category = INCOME_CATEGORY
    else:
        available_categories = EXPENSE_CATEGORIES
        default_category = "Miscellaneous"
    
    # Option 1: Select from predefined categories
    category_option = st.radio(
        "Category Selection Method:",
        ["Select from predefined", "Type custom category", "Skip (leave blank)"],
        horizontal=True
    )
    
    selected_category = None
    
    if category_option == "Select from predefined":
        # Show appropriate categories based on transaction type
        if is_income:
            # For income, just show "Income" option
            category_options = ["", INCOME_CATEGORY, "Uncategorized"]
            current_index = 1 if current_category == INCOME_CATEGORY else (2 if current_category == "Uncategorized" else 0)
        else:
            # For expenses, show expense categories
            category_options = [""] + available_categories + ["Uncategorized"]
            if current_category in available_categories:
                current_index = available_categories.index(current_category) + 1
            else:
                current_index = 0
        
        selected_category = st.selectbox(
            f"Choose a category ({'Income' if is_income else 'Expense'}):",
            options=category_options,
            index=current_index,
            label_visibility="visible"
        )
        if selected_category == "":
            selected_category = None
    
    elif category_option == "Type custom category":
        # Allow typing custom category
        custom_category = st.text_input(
            "Enter custom category:",
            value="" if current_category in EXPENSE_CATEGORIES or current_category == "Uncategorized" else current_category,
            placeholder="e.g., Pet Care, Home Improvement"
        )
        selected_category = custom_category.strip() if custom_category.strip() else None
    
    else:  # Skip - leave blank
        selected_category = ""  # Will be set to empty string, can be left blank
    
    st.divider()
    
    # Merchant name editing UI
    st.markdown("### Edit Merchant Name")
    
    # Get original merchant_guess from transaction (this is what we'll keep if user skips)
    original_merchant_guess = current_txn.get('merchant_guess', '')
    description_raw = current_txn.get('description_raw', '')
    
    # Clean up the merchant name using OpenAI API for better suggestion
    cleaned_merchant = None
    client = get_openai_client()
    if client and validate_openai_key():
        with st.spinner("Cleaning merchant name..."):
            cleaned_merchant = clean_merchant_name_openai(client, original_merchant_guess, description_raw)
    else:
        # Fallback to simple cleaning if OpenAI is not available
        cleaned_merchant = clean_merchant_name(original_merchant_guess, description_raw)
    
    # Show suggested merchant name (cleaned up version)
    st.info(f"💡 Suggested: **{cleaned_merchant}**")
    
    # Option 1: Keep suggested merchant
    merchant_option = st.radio(
        "Merchant Name:",
        ["Keep suggested", "Edit merchant name", "Skip (leave as is)"],
        horizontal=True
    )
    
    selected_merchant = None
    
    if merchant_option == "Keep suggested":
        selected_merchant = cleaned_merchant
    
    elif merchant_option == "Edit merchant name":
        # Allow typing custom merchant name
        custom_merchant = st.text_input(
            "Enter merchant name:",
            value=cleaned_merchant,
            placeholder="e.g., Amazon, Walmart, Starbucks",
            help="Enter a clean, short merchant name (max 50 characters)"
        )
        selected_merchant = custom_merchant.strip() if custom_merchant.strip() else cleaned_merchant
        # Truncate if too long
        if len(selected_merchant) > 50:
            selected_merchant = selected_merchant[:47] + "..."
    
    else:  # Skip - leave as is
        selected_merchant = original_merchant_guess  # Keep the original merchant_guess value from the transaction
    
    # Action buttons
    st.markdown('<div class="txn-actions">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Approve & Next", type="primary", use_container_width=True, key="txn_approve_next"):
            # Update direction, category, and merchant
            pending_df.at[pending_df.index[current_idx], 'direction'] = new_direction_lower
            final_category = selected_category.strip() if selected_category and selected_category.strip() else ""
            # If income and no category selected, default to "Income"
            if new_direction_lower == 'income' and not final_category:
                final_category = INCOME_CATEGORY
            pending_df.at[pending_df.index[current_idx], 'category'] = final_category
            # Update merchant name
            if selected_merchant:
                pending_df.at[pending_df.index[current_idx], 'merchant_guess'] = selected_merchant
            else:
                # If merchant was skipped, still try to clean it with OpenAI API
                client = get_openai_client()
                if client and validate_openai_key():
                    original_merchant = pending_df.at[pending_df.index[current_idx], 'merchant_guess']
                    description = pending_df.at[pending_df.index[current_idx], 'description_raw']
                    cleaned = clean_merchant_name_openai(client, original_merchant, description)
                    pending_df.at[pending_df.index[current_idx], 'merchant_guess'] = cleaned
            st.session_state.pending_transactions = pending_df
            st.session_state.review_index = current_idx + 1
            st.rerun()
    
    with col2:
        if st.button("⏭️ Skip This", use_container_width=True, key="txn_skip_one"):
            # Leave category as is, but still clean merchant name with OpenAI if available
            client = get_openai_client()
            if client and validate_openai_key():
                original_merchant = pending_df.at[pending_df.index[current_idx], 'merchant_guess']
                description = pending_df.at[pending_df.index[current_idx], 'description_raw']
                cleaned = clean_merchant_name_openai(client, original_merchant, description)
                pending_df.at[pending_df.index[current_idx], 'merchant_guess'] = cleaned
                st.session_state.pending_transactions = pending_df
            st.session_state.review_index = current_idx + 1
            st.rerun()
    
    with col3:
        if st.button("⏭️⏭️ Skip All Remaining", use_container_width=True, key="txn_skip_all"):
            # Clean all remaining merchant names with OpenAI before saving
            client = get_openai_client()
            if client and validate_openai_key():
                with st.spinner("Cleaning merchant names for remaining transactions..."):
                    for idx in pending_df.index:
                        original_merchant = pending_df.at[idx, 'merchant_guess']
                        description = pending_df.at[idx, 'description_raw']
                        cleaned = clean_merchant_name_openai(client, original_merchant, description)
                        pending_df.at[idx, 'merchant_guess'] = cleaned
            
            # Add all remaining transactions with current categories
            if st.session_state.transactions.empty:
                st.session_state.transactions = pending_df
            else:
                st.session_state.transactions = pd.concat(
                    [st.session_state.transactions, pending_df],
                    ignore_index=True
                )
            # Clear pending transactions and show success screen
            st.session_state.pending_transactions = pd.DataFrame()
            st.session_state.review_index = 0
            st.session_state.show_upload_complete = True
            st.rerun()
    
    with col4:
        if st.button("❌ Cancel", use_container_width=True, key="txn_cancel"):
            # Clear pending transactions
            st.session_state.pending_transactions = pd.DataFrame()
            st.session_state.review_index = 0
            st.info("Upload cancelled. No transactions were added.")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show preview of remaining transactions
    with st.expander(f"📋 Preview Remaining Transactions ({len(pending_df) - current_idx - 1} left)"):
        remaining_df = pending_df.iloc[current_idx + 1:].head(10)
        if not remaining_df.empty:
            st.dataframe(
                remaining_df[['posted_date', 'description_raw', 'amount', 'direction', 'category']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No more transactions to review.")


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
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#000000"),
                title=dict(font=dict(color="#000000")),
            )
            fig.update_traces(
                textfont=dict(color="#000000"),
                textposition='auto',
                textinfo='label+percent'
            )
            fig.update_layout(legend=dict(font=dict(color="#000000")))
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
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#000000"),
                title=dict(font=dict(color="#000000")),
            )
            fig.update_xaxes(
                tickfont=dict(color="#000000"),
                title=dict(font=dict(color="#000000"))
            )
            fig.update_yaxes(
                tickfont=dict(color="#000000"),
                title=dict(font=dict(color="#000000")),
                tickmode='linear'
            )
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
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#000000"),
            title=dict(font=dict(color="#000000")),
        )
        fig.update_xaxes(tickfont=dict(color="#000000"), title=dict(font=dict(color="#000000")))
        fig.update_yaxes(tickfont=dict(color="#000000"), title=dict(font=dict(color="#000000")))
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
            selected_txn = filtered_df.iloc[selected_idx]
            current_category = selected_txn['category']
            is_income = selected_txn.get('direction', 'expense').lower() == 'income'
            if is_income:
                available_categories = [INCOME_CATEGORY]
            else:
                available_categories = EXPENSE_CATEGORIES
            
            col1, col2 = st.columns([3, 1])
            with col1:
                # Determine default index
                if is_income:
                    category_options = [INCOME_CATEGORY, "Uncategorized"]
                    default_index = 0 if current_category == INCOME_CATEGORY else 1
                else:
                    category_options = available_categories + ["Uncategorized"]
                    if current_category in available_categories:
                        default_index = available_categories.index(current_category)
                    else:
                        default_index = len(available_categories)  # Uncategorized
                
                new_category = st.selectbox(
                    f"New Category ({'Income' if is_income else 'Expense'}):",
                    options=category_options,
                    index=default_index
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
        st.warning("⚠️ Please set your OpenAI API key in Streamlit secrets or in the code to use the chat assistant.")
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
        
        # Check if this is a grocery store query
        intent, _, _ = detect_grocery_store_intent(user_input)
        is_store_query = intent is not None
        
        # Get assistant response
        with st.chat_message("assistant"):
            if is_store_query:
                with st.spinner("🔍 Looking up nearby stores..."):
                    client = get_openai_client()
                    response, store_results = get_chatbot_response(
                        client,
                        user_input,
                        st.session_state.transactions,
                        st.session_state.financial_profile
                    )
            else:
                with st.spinner("Thinking..."):
                    client = get_openai_client()
                    response, store_results = get_chatbot_response(
                        client,
                        user_input,
                        st.session_state.transactions,
                        st.session_state.financial_profile
                    )
            
            st.write(response)
            
            # Show store results in expandable section if available
            if store_results and len(store_results) > 0:
                with st.expander(f"📍 Store Locations ({len(store_results)} found)", expanded=False):
                    st.markdown("**Top options near you (within 10 km):**")
                    for i, store in enumerate(store_results[:5], 1):
                        st.markdown(f"{i}. **{store['name']}**")
                        st.markdown(f"   {store['address']}")
                        st.markdown(f"   Distance: {store['distance_km']} km")
                        if i < len(store_results[:5]):
                            st.markdown("---")
                    if len(store_results) > 5:
                        st.markdown(f"\n*... and {len(store_results) - 5} more stores*")
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================================
# Main App
# ============================================================================

def show_profile_questionnaire():
    """Show the user profile questionnaire."""
    st.title("👤 Complete Your Profile")
    st.markdown("""
    **Help us personalize your financial insights!**
    
    This information is used only to tailor recommendations and benchmarks to your situation. 
    All fields are optional - you can skip this and complete it later in Settings.
    """)
    
    st.divider()
    
    profile = st.session_state.user_profile.copy()
    
    # Age Range
    age_ranges = ["", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Prefer not to say"]
    current_age_idx = age_ranges.index(profile.get("age_range", "")) if profile.get("age_range") in age_ranges else 0
    age_range = st.selectbox(
        "Age Range (optional):",
        options=age_ranges,
        index=current_age_idx,
        help="Used to adjust spending benchmarks and savings recommendations"
    )
    profile["age_range"] = age_range if age_range else None
    
    # Employment Status
    employment_options = ["", "Employed", "Self-employed", "Student", "Unemployed", "Retired", "Prefer not to say"]
    current_emp_idx = employment_options.index(profile.get("employment_status", "")) if profile.get("employment_status") in employment_options else 0
    employment_status = st.selectbox(
        "Employment Status (optional):",
        options=employment_options,
        index=current_emp_idx,
        help="Helps tailor financial advice to your situation"
    )
    profile["employment_status"] = employment_status if employment_status else None
    
    # Occupation/Industry
    occupation = st.text_input(
        "Occupation / Industry (optional):",
        value=profile.get("occupation", "") or "",
        placeholder="e.g., Software Engineer, Healthcare, Finance",
        help="Used to provide industry-specific insights"
    )
    profile["occupation"] = occupation.strip() if occupation.strip() else None
    
    # Location
    col1, col2, col3 = st.columns(3)
    with col1:
        location_country = st.text_input(
            "Country (optional):",
            value=profile.get("location_country", "") or "",
            placeholder="e.g., United States, Canada",
            help="Used to adjust for cost of living differences"
        )
        profile["location_country"] = location_country.strip() if location_country.strip() else None
    
    with col2:
        location_city = st.text_input(
            "City (optional):",
            value=profile.get("location_city", "") or "",
            placeholder="e.g., New York, Toronto",
            help="Provides more precise cost of living context"
        )
        profile["location_city"] = location_city.strip() if location_city.strip() else None
    
    with col3:
        location_zip = st.text_input(
            "ZIP Code (optional):",
            value=profile.get("location_zip", "") or "",
            placeholder="e.g., 10001, M5H 2N2",
            help="Used to refine store locator searches"
        )
        profile["location_zip"] = location_zip.strip() if location_zip.strip() else None
    
    st.divider()
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("💾 Save Profile", type="primary", use_container_width=True):
            profile["profile_completed"] = True
            st.session_state.user_profile = profile
            st.success("✅ Profile saved!")
            st.rerun()
    
    with col2:
        if st.button("⏭️ Skip for Now", use_container_width=True):
            st.session_state.user_profile = profile
            st.info("You can complete your profile later in Settings.")
            st.rerun()


def settings_page():
    """Settings page to view and edit user profile."""
    st.title("⚙️ Settings")
    
    st.markdown("### 👤 User Profile")
    st.markdown("Update your profile information to get personalized financial insights.")
    
    profile = st.session_state.user_profile.copy()
    
    # Show current profile status
    if profile.get("profile_completed"):
        st.success("✅ Profile completed")
    else:
        st.info("ℹ️ Profile not yet completed")
    
    st.divider()
    
    # Age Range
    age_ranges = ["", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Prefer not to say"]
    current_age_idx = age_ranges.index(profile.get("age_range", "")) if profile.get("age_range") in age_ranges else 0
    age_range = st.selectbox(
        "Age Range:",
        options=age_ranges,
        index=current_age_idx
    )
    profile["age_range"] = age_range if age_range else None
    
    # Employment Status
    employment_options = ["", "Employed", "Self-employed", "Student", "Unemployed", "Retired", "Prefer not to say"]
    current_emp_idx = employment_options.index(profile.get("employment_status", "")) if profile.get("employment_status") in employment_options else 0
    employment_status = st.selectbox(
        "Employment Status:",
        options=employment_options,
        index=current_emp_idx
    )
    profile["employment_status"] = employment_status if employment_status else None
    
    # Occupation/Industry
    occupation = st.text_input(
        "Occupation / Industry:",
        value=profile.get("occupation", "") or "",
        placeholder="e.g., Software Engineer, Healthcare, Finance"
    )
    profile["occupation"] = occupation.strip() if occupation.strip() else None
    
    # Location
    col1, col2, col3 = st.columns(3)
    with col1:
        location_country = st.text_input(
            "Country:",
            value=profile.get("location_country", "") or "",
            placeholder="e.g., United States, Canada"
        )
        profile["location_country"] = location_country.strip() if location_country.strip() else None
    
    with col2:
        location_city = st.text_input(
            "City:",
            value=profile.get("location_city", "") or "",
            placeholder="e.g., New York, Toronto"
        )
        profile["location_city"] = location_city.strip() if location_city.strip() else None
    
    with col3:
        location_zip = st.text_input(
            "ZIP Code:",
            value=profile.get("location_zip", "") or "",
            placeholder="e.g., 10001, M5H 2N2"
        )
        profile["location_zip"] = location_zip.strip() if location_zip.strip() else None
    
    st.divider()
    
    # Save button
    if st.button("💾 Save Profile", type="primary"):
        profile["profile_completed"] = True
        st.session_state.user_profile = profile
        st.success("✅ Profile updated successfully!")
        st.rerun()
    


def display_logo():
    """Display logo from assets folder if it exists."""
    import os
    logo_paths = [
        os.path.join("assets", "logo.svg"),
        os.path.join("assets", "logo.png"),
        os.path.join("assets", "Logo.svg"),
        os.path.join("assets", "Logo.png"),
    ]
    
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, use_container_width=True)
            return
    
    # If no logo found, show emoji as fallback
    st.sidebar.markdown("### 💰 Accountable AI v1")


def apply_custom_theme():
    """Apply custom color scheme: white main background, light blue sidebar."""
    st.markdown("""
        <style>
        /* Theme variables */
        :root {
            --bg-main: #FFFFFF;
            --bg-surface: #F5F5F5;
            --bg-surface-2: #E8E8E8;
            --border: #E0E0E0;
            --text-main: #111111;
            --primary: #4CAF50;
            --primary-hover: #45a049;
            --danger: #FF4B4B;
            --danger-hover: #FF3030;
        }
        
        /* Main app background - white */
        .stApp {
            background-color: #FFFFFF !important;
        }
        
        /* Main content area background - white */
        .main .block-container {
            background-color: #FFFFFF !important;
            padding-top: 2rem;
        }
        
        /* All text on white background - black */
        .main *,
        .stMarkdown,
        .stText,
        h1, h2, h3, h4, h5, h6,
        p, span, div,
        .element-container,
        .stDataFrame,
        .stMetric {
            color: #000000 !important;
        }
        
        /* Layer A: CSS guardrails - Fix dark BaseWeb buttons only, preserve gray alerts */
        /* BaseWeb / Streamlit button surfaces that show up dark */
        button[kind],
        .stButton button,
        [data-testid="baseButton-primary"],
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-tertiary"] {
            background-color: var(--bg-surface) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--border) !important;
        }
        
        .stButton button:hover {
            background-color: var(--bg-surface-2) !important;
        }
        
        /* Preserve semantic buttons */
        .stButton button.btn-primary,
        button.btn-primary,
        button[kind="primary"] {
            background-color: var(--primary) !important;
            color: #fff !important;
            border-color: var(--primary) !important;
        }
        
        .stButton button.btn-primary:hover,
        button.btn-primary:hover,
        button[kind="primary"]:hover {
            background-color: var(--primary-hover) !important;
            border-color: var(--primary-hover) !important;
        }
        
        .stButton button.btn-danger,
        button.btn-danger {
            background-color: var(--danger) !important;
            color: #fff !important;
            border-color: var(--danger) !important;
        }
        
        .stButton button.btn-danger:hover,
        button.btn-danger:hover {
            background-color: var(--danger-hover) !important;
            border-color: var(--danger-hover) !important;
        }
        
        /* Layer B: Dark-only override class */
        .no-dark-surface {
            background-color: var(--bg-surface) !important;
            color: var(--text-main) !important;
            border-color: var(--border) !important;
        }
        
        .no-dark-surface * {
            color: var(--text-main) !important;
            background: transparent !important;
        }
        
        /* Exception: preserve semantic button colors even with no-dark-surface */
        .no-dark-surface.btn-primary,
        .no-dark-surface.btn-danger {
            background-color: inherit !important;
            color: inherit !important;
        }
        
        /* Sidebar background - light blue */
        [data-testid="stSidebar"] {
            background-color: #E3F2FD !important;
        }
        
        /* Sidebar content sections */
        [data-testid="stSidebar"] > div {
            background-color: #E3F2FD !important;
        }
        
        /* Sidebar markdown and text elements */
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stText {
            background-color: #E3F2FD !important;
        }
        
        /* Ensure proper contrast for sidebar text */
        [data-testid="stSidebar"] * {
            color: #262730 !important;
        }
        
        /* Header and title areas */
        .stHeader {
            background-color: #FFFFFF !important;
        }
        
        /* Complete Profile button - light green (like success/ready color) */
        .complete-profile-button-container ~ div button[kind="secondary"],
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #4CAF50 !important;
            color: #FFFFFF !important;
            border-color: #4CAF50 !important;
        }
        
        .complete-profile-button-container ~ div button[kind="secondary"]:hover,
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: #45a049 !important;
            border-color: #45a049 !important;
        }
        
        /* Danger button styling - destructive actions (e.g., Clear Profile) */
        .btn-danger,
        .btn-danger[kind="secondary"],
        .btn-danger[data-baseweb="button"],
        button.btn-danger {
            background-color: #FF4B4B !important;
            border-color: #FF4B4B !important;
            color: #FFFFFF !important;
            background: #FF4B4B !important;
        }
        
        .btn-danger:hover,
        .btn-danger[kind="secondary"]:hover,
        .btn-danger[data-baseweb="button"]:hover,
        button.btn-danger:hover {
            background-color: #FF3030 !important;
            border-color: #FF3030 !important;
            background: #FF3030 !important;
        }
        
        /* Force white text on nested elements */
        .btn-danger *,
        .btn-danger div,
        .btn-danger span,
        .btn-danger svg,
        button.btn-danger *,
        button.btn-danger div,
        button.btn-danger span,
        button.btn-danger svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }
        
        /* Ensure nested button children don't keep dark backgrounds */
        .btn-danger > div,
        .btn-danger > div > div,
        .btn-danger[data-baseweb="button"] > div,
        button.btn-danger > div,
        button.btn-danger > div > div,
        button.btn-danger[data-baseweb="button"] > div {
            background-color: transparent !important;
            background: transparent !important;
        }
        
        /* Override BaseWeb button component styles */
        .btn-danger[data-baseweb="button"] > div > div,
        button.btn-danger[data-baseweb="button"] > div > div {
            background-color: transparent !important;
            background: transparent !important;
            color: #FFFFFF !important;
        }
        
        /* Transaction action buttons styling */
        .txn-actions button {
            height: 44px !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        
        /* Default secondary style for transaction action buttons */
        .txn-actions button[kind="secondary"],
        .txn-actions button:not(.btn-primary):not(.btn-danger) {
            background-color: #F5F5F5 !important;
            border: 1px solid #E0E0E0 !important;
            color: #111111 !important;
        }
        
        .txn-actions button[kind="secondary"]:hover,
        .txn-actions button:not(.btn-primary):not(.btn-danger):hover {
            background-color: #E8E8E8 !important;
            border-color: #D0D0D0 !important;
        }
        
        /* Primary button in transaction actions */
        .txn-actions .btn-primary,
        .txn-actions button.btn-primary {
            background-color: #4CAF50 !important;
            border-color: #4CAF50 !important;
            color: #FFFFFF !important;
        }
        
        .txn-actions .btn-primary:hover,
        .txn-actions button.btn-primary:hover {
            background-color: #45a049 !important;
            border-color: #45a049 !important;
        }
        
        /* Danger button in transaction actions */
        .txn-actions .btn-danger,
        .txn-actions button.btn-danger {
            background-color: #FF4B4B !important;
            border-color: #FF4B4B !important;
            color: #FFFFFF !important;
        }
        
        .txn-actions .btn-danger:hover,
        .txn-actions button.btn-danger:hover {
            background-color: #FF3030 !important;
            border-color: #FF3030 !important;
        }
        
        /* Ensure nested elements inherit correct colors */
        .txn-actions button *,
        .txn-actions button div,
        .txn-actions button span,
        .txn-actions button svg {
            color: inherit !important;
            fill: inherit !important;
        }
        
        .txn-actions .btn-primary *,
        .txn-actions .btn-primary div,
        .txn-actions .btn-primary span,
        .txn-actions .btn-primary svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }
        
        .txn-actions .btn-danger *,
        .txn-actions .btn-danger div,
        .txn-actions .btn-danger span,
        .txn-actions .btn-danger svg {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }
        
        /* Settings input boxes - lighter background for white theme */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stNumberInput > div > div > input,
        input[type="text"],
        input[type="number"],
        input[type="email"],
        input[type="password"],
        select {
            background-color: #F5F5F5 !important;
            border-color: #E0E0E0 !important;
            color: #000000 !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput > div > div > input:focus,
        input[type="text"]:focus,
        input[type="number"]:focus,
        input[type="email"]:focus,
        input[type="password"]:focus,
        select:focus {
            background-color: #FAFAFA !important;
            border-color: #4CAF50 !important;
            box-shadow: 0 0 0 1px #4CAF50 !important;
            outline: none !important;
        }
        
        /* Style the input containers (BaseWeb components) */
        [data-baseweb="input"],
        [data-baseweb="select"] {
            background-color: #F5F5F5 !important;
        }
        
        [data-baseweb="input"] input,
        [data-baseweb="select"] select {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }
        
        [data-baseweb="input"]:focus-within,
        [data-baseweb="select"]:focus-within {
            background-color: #FAFAFA !important;
        }
        
        /* Selectbox dropdowns - match text input styling */
        .stSelectbox > div > div,
        [data-baseweb="select"] > div {
            background-color: #F5F5F5 !important;
        }
        
        .stSelectbox > div > div > div,
        [data-baseweb="select"] > div > div {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }
        
        /* Selectbox dropdown menu options */
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [role="option"],
        div[data-baseweb="select"] + div [role="listbox"],
        div[data-baseweb="select"] + div [role="option"] {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }
        
        [data-baseweb="popover"] [role="option"]:hover,
        div[data-baseweb="select"] + div [role="option"]:hover {
            background-color: #E0E0E0 !important;
        }
        
        /* File uploader - lighter background like text inputs - entire container */
        .stFileUploader,
        [data-testid="stFileUploader"],
        .stFileUploader > div,
        [data-testid="stFileUploader"] > div,
        .stFileUploader > div > div,
        [data-testid="stFileUploader"] > div > div,
        .stFileUploader > div > div > div,
        [data-testid="stFileUploader"] > div > div > div,
        [data-baseweb="file-uploader"],
        [data-baseweb="file-uploader"] > div,
        [data-baseweb="file-uploader"] > div > div,
        [data-baseweb="file-uploader"] > div > div > div {
            background-color: #F5F5F5 !important;
            border-color: #E0E0E0 !important;
            color: #000000 !important;
        }
        
        /* File uploader drag and drop area - all nested elements */
        [data-baseweb="file-uploader"] *,
        .stFileUploader *,
        [data-testid="stFileUploader"] * {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }
        
        /* Override for specific elements that need different styling */
        [data-baseweb="file-uploader"] button,
        [data-baseweb="file-uploader"] [role="button"],
        .stFileUploader button {
            background-color: #F5F5F5 !important;
            border-color: #E0E0E0 !important;
            color: #000000 !important;
        }
        
        /* Text elements in file uploader */
        [data-baseweb="file-uploader"] p,
        [data-baseweb="file-uploader"] span,
        [data-baseweb="file-uploader"] div,
        .stFileUploader p,
        .stFileUploader span,
        .stFileUploader div {
            color: #000000 !important;
        }
        
        /* File uploader browse button - lighter background */
        [data-baseweb="file-uploader"] button,
        [data-baseweb="file-uploader"] [role="button"],
        [data-baseweb="file-uploader"] [data-baseweb="button"],
        .stFileUploader button,
        [data-testid="stFileUploader"] button,
        [data-baseweb="file-uploader"] > div > button,
        [data-baseweb="file-uploader"] > div > div > button,
        [data-baseweb="file-uploader"] > div > div > div > button {
            background-color: #F5F5F5 !important;
            border-color: #E0E0E0 !important;
            color: #000000 !important;
        }
        
        [data-baseweb="file-uploader"] button:hover,
        [data-baseweb="file-uploader"] [role="button"]:hover,
        [data-baseweb="file-uploader"] [data-baseweb="button"]:hover,
        .stFileUploader button:hover,
        [data-testid="stFileUploader"] button:hover {
            background-color: #E0E0E0 !important;
            border-color: #BDBDBD !important;
        }
        
        /* BaseWeb button component within file uploader */
        [data-baseweb="file-uploader"] [data-baseweb="button"] > div,
        [data-baseweb="file-uploader"] [data-baseweb="button"] > div > div {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }
        
        /* File input element styling */
        input[type="file"],
        input[type="file"]::-webkit-file-upload-button {
            background-color: #F5F5F5 !important;
            border-color: #E0E0E0 !important;
            color: #000000 !important;
        }
        
        /* Target all file uploader children and nested elements */
        [data-baseweb="file-uploader"] *,
        .stFileUploader *,
        [data-testid="stFileUploader"] * {
            color: #000000 !important;
        }
        
        /* Specifically target the browse button text/span */
        [data-baseweb="file-uploader"] button span,
        [data-baseweb="file-uploader"] [role="button"] span,
        .stFileUploader button span {
            color: #000000 !important;
        }
        
        /* Input wrapper backgrounds */
        .stTextInput > div,
        .stSelectbox > div,
        .stNumberInput > div {
            background-color: transparent !important;
        }
        
        /* Harmonize info/warning/error boxes - lighter backgrounds */
        .stAlert,
        [data-testid="stAlert"],
        div[data-baseweb="notification"],
        div[data-baseweb="toast"] {
            background-color: #F5F5F5 !important;
            border-color: #E0E0E0 !important;
            color: #000000 !important;
        }
        
        /* Info boxes - light blue tint */
        .stAlert[data-baseweb="notification-info"],
        div[data-baseweb="notification"][data-kind="info"],
        [data-testid="stAlert"]:has([data-icon="info"]),
        [data-testid="stAlert"]:has(div[data-icon="info"]) {
            background-color: #E3F2FD !important;
            border-color: #BBDEFB !important;
            color: #000000 !important;
        }
        
        /* Warning boxes - light yellow/amber tint */
        .stAlert[data-baseweb="notification-warning"],
        div[data-baseweb="notification"][data-kind="warning"],
        [data-testid="stAlert"]:has([data-icon="warning"]),
        [data-testid="stAlert"]:has(div[data-icon="warning"]) {
            background-color: #FFF3E0 !important;
            border-color: #FFE0B2 !important;
            color: #000000 !important;
        }
        
        /* Error boxes - light red/pink tint */
        .stAlert[data-baseweb="notification-error"],
        div[data-baseweb="notification"][data-kind="error"],
        [data-testid="stAlert"]:has([data-icon="error"]),
        [data-testid="stAlert"]:has(div[data-icon="error"]) {
            background-color: #FFEBEE !important;
            border-color: #FFCDD2 !important;
            color: #000000 !important;
        }
        
        /* Success boxes - light green tint (lighter than before) */
        .stAlert[data-baseweb="notification-success"],
        div[data-baseweb="notification"][data-kind="success"],
        [data-testid="stAlert"]:has([data-icon="success"]),
        [data-testid="stAlert"]:has(div[data-icon="success"]) {
            background-color: #E8F5E9 !important;
            border-color: #C8E6C9 !important;
            color: #000000 !important;
        }
        
        /* Target alert content directly */
        [data-baseweb="notification"] > div,
        [data-testid="stAlert"] > div {
            color: #000000 !important;
        }
        
        /* Expanders - lighter background */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] > div:first-child {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }
        
        .streamlit-expanderContent,
        [data-testid="stExpander"] > div:last-child {
            background-color: #FAFAFA !important;
        }
        
        /* Metrics - lighter background */
        [data-testid="stMetricContainer"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }
        
        /* DataFrames and tables - lighter backgrounds */
        .stDataFrame,
        [data-testid="stDataFrame"],
        table {
            background-color: #FFFFFF !important;
        }
        
        .stDataFrame thead,
        [data-testid="stDataFrame"] thead,
        table thead {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
        }
        
        .stDataFrame tbody tr:nth-child(even),
        [data-testid="stDataFrame"] tbody tr:nth-child(even),
        table tbody tr:nth-child(even) {
            background-color: #FAFAFA !important;
        }
        
        .stDataFrame tbody tr:nth-child(odd),
        [data-testid="stDataFrame"] tbody tr:nth-child(odd),
        table tbody tr:nth-child(odd) {
            background-color: #FFFFFF !important;
        }
        
        /* Cards and containers - lighter grey */
        .element-container,
        [data-testid="element-container"] {
            background-color: transparent !important;
        }
        
        /* Radio buttons and other form elements */
        .stRadio > div,
        .stCheckbox > div,
        .stSlider > div {
            background-color: transparent !important;
        }
        
        /* Progress bars and spinners - lighter */
        .stProgress > div > div {
            background-color: #E0E0E0 !important;
        }
        
        .stProgress > div > div > div {
            background-color: #4CAF50 !important;
        }
        
        /* Orange person icon in Complete Profile button */
        .profile-icon-orange {
            color: #FF9800 !important;
            font-size: 1.2em !important;
            display: inline-block !important;
            vertical-align: middle !important;
        }
        
        /* ==========================
           Chat input (st.chat_input) - SCOPED TO CHAT ASSISTANT ONLY
           ========================== */
        
        /* Only apply these styles when Chat Assistant page is active */
        /* This is handled via JavaScript detection */
        
        /* The actual textarea - general styling */
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] textarea:focus {
            background-color: #F5F5F5 !important;
            color: #000000 !important;
            border: 1px solid #E0E0E0 !important;
            box-shadow: none !important;
        }
        
        /* Placeholder text */
        [data-testid="stChatInput"] textarea::placeholder {
            color: #666666 !important;
            opacity: 1 !important;
        }
        
        /* Send button container */
        [data-testid="stChatInput"] button {
            background-color: #F5F5F5 !important;
            border: 1px solid #E0E0E0 !important;
            color: #000000 !important;
        }
        
        /* Send button hover */
        [data-testid="stChatInput"] button:hover {
            background-color: #E0E0E0 !important;
            border-color: #BDBDBD !important;
        }
        </style>
        <script>
        function styleCompleteProfileButton() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                const buttons = sidebar.querySelectorAll('button');
                buttons.forEach(button => {
                    const buttonText = button.textContent || button.innerText || '';
                    if (buttonText.includes('Complete Profile') && !button.classList.contains('profile-btn-styled')) {
                        // Mark as styled to avoid re-styling
                        button.classList.add('profile-btn-styled');
                        
                        // Force light green background with !important
                        button.style.setProperty('background-color', '#4CAF50', 'important');
                        button.style.setProperty('color', '#FFFFFF', 'important');
                        button.style.setProperty('border-color', '#4CAF50', 'important');
                        button.style.setProperty('border', '1px solid #4CAF50', 'important');
                        
                        // Style the person icon to orange
                        let buttonHTML = button.innerHTML || '';
                        if (buttonHTML.includes('👤') && !buttonHTML.includes('profile-icon-orange')) {
                            // Create a styled span for the orange icon
                            buttonHTML = buttonHTML.replace(/👤/g, '<span class="profile-icon-orange" style="color: #FF9800 !important; font-size: 1.2em; display: inline-block; vertical-align: middle;">👤</span>');
                            button.innerHTML = buttonHTML;
                            
                            // Re-apply button styles after innerHTML change
                            button.style.setProperty('background-color', '#4CAF50', 'important');
                            button.style.setProperty('color', '#FFFFFF', 'important');
                            button.style.setProperty('border-color', '#4CAF50', 'important');
                        }
                        
                        // Add hover effects (only once)
                        button.addEventListener('mouseenter', function() {
                            this.style.setProperty('background-color', '#45a049', 'important');
                            this.style.setProperty('border-color', '#45a049', 'important');
                        }, { once: false });
                        button.addEventListener('mouseleave', function() {
                            this.style.setProperty('background-color', '#4CAF50', 'important');
                            this.style.setProperty('border-color', '#4CAF50', 'important');
                        }, { once: false });
                    } else if (button.classList.contains('profile-btn-styled')) {
                        // Re-apply styles if already styled (in case Streamlit re-renders)
                        button.style.setProperty('background-color', '#4CAF50', 'important');
                        button.style.setProperty('color', '#FFFFFF', 'important');
                        button.style.setProperty('border-color', '#4CAF50', 'important');
                    }
                });
            }
        }
        // Run on page load and after Streamlit updates
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', styleCompleteProfileButton);
        } else {
            styleCompleteProfileButton();
        }
        // Also run after Streamlit reruns
        window.addEventListener('load', styleCompleteProfileButton);
        function isChatAssistantPage() {
            // Check if we're on the Chat Assistant page by looking for the page title
            const pageTitle = document.querySelector('h1, h2, [data-testid="stHeader"]');
            if (pageTitle) {
                const titleText = pageTitle.textContent || pageTitle.innerText || '';
                return titleText.includes('Chat Assistant') || titleText.includes('💬');
            }
            return false;
        }
        
        function fixChatAssistantBottomBar() {
            // Only run on Chat Assistant page
            if (!isChatAssistantPage()) {
                return;
            }
            
            // Target the sticky bottom container (Streamlit's stBottom)
            const bottomContainers = [
                document.querySelector('section[data-testid="stBottom"]'),
                document.querySelector('div[data-testid="stBottom"]'),
                document.querySelector('footer[data-testid="stFooter"]'),
                document.querySelector('[data-testid="stChatInput"]')?.closest('section'),
                document.querySelector('[data-testid="stChatInput"]')?.closest('div[style*="position"]'),
            ].filter(Boolean);
            
            bottomContainers.forEach(container => {
                if (container) {
                    container.style.setProperty('background-color', '#FFFFFF', 'important');
                    container.style.setProperty('background', '#FFFFFF', 'important');
                    container.style.setProperty('border-top', '1px solid #E0E0E0', 'important');
                    
                    // Force white on all immediate children
                    Array.from(container.children).forEach(child => {
                        child.style.setProperty('background-color', '#FFFFFF', 'important');
                        child.style.setProperty('background', '#FFFFFF', 'important');
                    });
                }
            });
            
            // Target chat input container specifically
            const chat = document.querySelector('[data-testid="stChatInput"]');
            if (chat) {
                chat.style.setProperty('background-color', '#FFFFFF', 'important');
                chat.style.setProperty('background', '#FFFFFF', 'important');
                chat.style.setProperty('border-top', '1px solid #E0E0E0', 'important');
                
                // Force white on all nested elements
                const allElements = chat.querySelectorAll('*');
                allElements.forEach(el => {
                    const computedBg = window.getComputedStyle(el).backgroundColor;
                    // Only change if it's dark/black
                    if (computedBg && (computedBg.includes('rgb(0, 0, 0') || computedBg.includes('rgba(0, 0, 0') || computedBg === 'black' || computedBg === '#000000' || computedBg.includes('rgb(18,') || computedBg.includes('rgb(19,'))) {
                        el.style.setProperty('background-color', '#FFFFFF', 'important');
                        el.style.setProperty('background', '#FFFFFF', 'important');
                    }
                });
            }
            
            // Target any fixed/sticky positioned elements near the bottom (likely the sticky footer)
            const allElements = document.querySelectorAll('*');
            allElements.forEach(el => {
                const rect = el.getBoundingClientRect();
                const computedStyle = window.getComputedStyle(el);
                const position = computedStyle.position;
                
                // Only check fixed/sticky elements near the bottom of the page
                if ((position === 'fixed' || position === 'sticky') && rect.bottom > window.innerHeight - 150) {
                    const computedBg = computedStyle.backgroundColor;
                    if (computedBg && (computedBg.includes('rgb(0, 0, 0') || computedBg.includes('rgba(0, 0, 0') || computedBg === 'black' || computedBg === '#000000' || computedBg.includes('rgb(18,') || computedBg.includes('rgb(19,'))) {
                        el.style.setProperty('background-color', '#FFFFFF', 'important');
                        el.style.setProperty('background', '#FFFFFF', 'important');
                        el.style.setProperty('border-top', '1px solid #E0E0E0', 'important');
                    }
                }
            });
        }
        
        function fixChatInputBar() {
            // General chat input fix (for all pages, but lighter)
            const chat = document.querySelector('[data-testid="stChatInput"]');
            if (chat) {
                chat.style.setProperty('border-top', '1px solid #E0E0E0', 'important');
            }
        }
        
        // Use MutationObserver to catch dynamic updates
        const observer = new MutationObserver(function() {
            styleCompleteProfileButton();
            fixChatInputBar();
            fixChatAssistantBottomBar(); // Scoped to Chat Assistant only
        });
        if (document.body) {
            observer.observe(document.body, { childList: true, subtree: true });
        }
        
        // Also call functions on load and after Streamlit updates
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                fixChatInputBar();
                fixChatAssistantBottomBar();
            });
        } else {
            fixChatInputBar();
            fixChatAssistantBottomBar();
        }
        window.addEventListener('load', function() {
            fixChatInputBar();
            fixChatAssistantBottomBar();
        });
        
        // Function to harmonize grey boxes and elements
        function harmonizeGreyBoxes() {
            // Style all alert/notification boxes
            const alerts = document.querySelectorAll('[data-testid="stAlert"], [data-baseweb="notification"]');
            alerts.forEach(alert => {
                const text = alert.textContent || '';
                if (text.includes('✅') || text.includes('success') || alert.getAttribute('data-kind') === 'success') {
                    alert.style.setProperty('background-color', '#E8F5E9', 'important');
                    alert.style.setProperty('border-color', '#C8E6C9', 'important');
                    alert.style.setProperty('color', '#000000', 'important');
                } else if (text.includes('⚠️') || text.includes('warning') || alert.getAttribute('data-kind') === 'warning') {
                    alert.style.setProperty('background-color', '#FFF3E0', 'important');
                    alert.style.setProperty('border-color', '#FFE0B2', 'important');
                    alert.style.setProperty('color', '#000000', 'important');
                } else if (text.includes('❌') || text.includes('error') || alert.getAttribute('data-kind') === 'error') {
                    alert.style.setProperty('background-color', '#FFEBEE', 'important');
                    alert.style.setProperty('border-color', '#FFCDD2', 'important');
                    alert.style.setProperty('color', '#000000', 'important');
                } else if (text.includes('💡') || text.includes('info') || alert.getAttribute('data-kind') === 'info') {
                    alert.style.setProperty('background-color', '#E3F2FD', 'important');
                    alert.style.setProperty('border-color', '#BBDEFB', 'important');
                    alert.style.setProperty('color', '#000000', 'important');
                } else {
                    // Default lighter grey
                    alert.style.setProperty('background-color', '#F5F5F5', 'important');
                    alert.style.setProperty('border-color', '#E0E0E0', 'important');
                    alert.style.setProperty('color', '#000000', 'important');
                }
            });
            
            // Style file uploader - lighter background - entire container
            const fileUploaders = document.querySelectorAll('[data-testid="stFileUploader"], [data-baseweb="file-uploader"]');
            fileUploaders.forEach(uploader => {
                uploader.style.setProperty('background-color', '#F5F5F5', 'important');
                uploader.style.setProperty('border-color', '#E0E0E0', 'important');
                uploader.style.setProperty('color', '#000000', 'important');
                // Style all child elements - divs, spans, paragraphs, etc.
                const allChildren = uploader.querySelectorAll('*');
                allChildren.forEach(element => {
                    // Skip buttons as they're handled separately
                    if (element.tagName !== 'BUTTON' && !element.hasAttribute('role') || element.getAttribute('role') !== 'button') {
                        element.style.setProperty('background-color', '#F5F5F5', 'important');
                        element.style.setProperty('color', '#000000', 'important');
                    }
                });
                // Style text elements specifically
                const textElements = uploader.querySelectorAll('p, span, div, label');
                textElements.forEach(element => {
                    element.style.setProperty('color', '#000000', 'important');
                });
                // Style browse button specifically - target all button types
                const buttons = uploader.querySelectorAll('button, [role="button"], [data-baseweb="button"]');
                buttons.forEach(button => {
                    button.style.setProperty('background-color', '#F5F5F5', 'important');
                    button.style.setProperty('border-color', '#E0E0E0', 'important');
                    button.style.setProperty('color', '#000000', 'important');
                    // Style nested divs in BaseWeb buttons
                    const buttonDivs = button.querySelectorAll('div');
                    buttonDivs.forEach(div => {
                        div.style.setProperty('background-color', '#F5F5F5', 'important');
                        div.style.setProperty('color', '#000000', 'important');
                    });
                    // Style spans/text in buttons
                    const buttonSpans = button.querySelectorAll('span');
                    buttonSpans.forEach(span => {
                        span.style.setProperty('color', '#000000', 'important');
                    });
                    // Add hover effect
                    button.addEventListener('mouseenter', function() {
                        this.style.setProperty('background-color', '#E0E0E0', 'important');
                        this.style.setProperty('border-color', '#BDBDBD', 'important');
                        const hoverDivs = this.querySelectorAll('div');
                        hoverDivs.forEach(div => {
                            div.style.setProperty('background-color', '#E0E0E0', 'important');
                        });
                    });
                    button.addEventListener('mouseleave', function() {
                        this.style.setProperty('background-color', '#F5F5F5', 'important');
                        this.style.setProperty('border-color', '#E0E0E0', 'important');
                        const leaveDivs = this.querySelectorAll('div');
                        leaveDivs.forEach(div => {
                            div.style.setProperty('background-color', '#F5F5F5', 'important');
                        });
                    });
                });
                // Style file input element
                const fileInputs = uploader.querySelectorAll('input[type="file"]');
                fileInputs.forEach(input => {
                    input.style.setProperty('background-color', '#F5F5F5', 'important');
                    input.style.setProperty('color', '#000000', 'important');
                });
            });
            
            // Style selectbox dropdowns - match text input styling
            const selectboxes = document.querySelectorAll('.stSelectbox, [data-baseweb="select"]');
            selectboxes.forEach(selectbox => {
                selectbox.style.setProperty('background-color', '#F5F5F5', 'important');
                const selectElement = selectbox.querySelector('select');
                if (selectElement) {
                    selectElement.style.setProperty('background-color', '#F5F5F5', 'important');
                    selectElement.style.setProperty('color', '#000000', 'important');
                    selectElement.style.setProperty('border-color', '#E0E0E0', 'important');
                }
                // Style the BaseWeb select component
                const baseWebSelect = selectbox.querySelector('[data-baseweb="select"]');
                if (baseWebSelect) {
                    baseWebSelect.style.setProperty('background-color', '#F5F5F5', 'important');
                    const innerSelect = baseWebSelect.querySelector('select');
                    if (innerSelect) {
                        innerSelect.style.setProperty('background-color', '#F5F5F5', 'important');
                        innerSelect.style.setProperty('color', '#000000', 'important');
                    }
                }
            });
            
            // Style expanders
            const expanders = document.querySelectorAll('[data-testid="stExpander"]');
            expanders.forEach(expander => {
                const header = expander.querySelector('.streamlit-expanderHeader') || expander.querySelector('div:first-child');
                const content = expander.querySelector('.streamlit-expanderContent') || expander.querySelector('div:last-child');
                if (header) {
                    header.style.setProperty('background-color', '#F5F5F5', 'important');
                    header.style.setProperty('color', '#000000', 'important');
                }
                if (content) {
                    content.style.setProperty('background-color', '#FAFAFA', 'important');
                }
            });
            
            // Style metrics
            const metrics = document.querySelectorAll('[data-testid="stMetricContainer"]');
            metrics.forEach(metric => {
                metric.style.setProperty('background-color', '#F5F5F5', 'important');
                const value = metric.querySelector('[data-testid="stMetricValue"]');
                const label = metric.querySelector('[data-testid="stMetricLabel"]');
                if (value) value.style.setProperty('color', '#000000', 'important');
                if (label) label.style.setProperty('color', '#000000', 'important');
            });
            
        }
        
        // Run harmonization on load and after updates
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', harmonizeGreyBoxes);
        } else {
            harmonizeGreyBoxes();
        }
        window.addEventListener('load', harmonizeGreyBoxes);
        
        // Use MutationObserver for dynamic content
        const harmonizeObserver = new MutationObserver(harmonizeGreyBoxes);
        if (document.body) {
            harmonizeObserver.observe(document.body, { childList: true, subtree: true });
        }
        
        // Add btn-danger class to Clear Profile button (no inline styles)
        function addDangerClass() {
            const allButtons = document.querySelectorAll('button');
            allButtons.forEach(button => {
                const buttonText = button.innerText || button.textContent || '';
                // Check if button contains "Clear Profile" or trash emoji
                if ((buttonText.includes('Clear Profile') || buttonText.includes('🗑️')) && !button.classList.contains('btn-danger')) {
                    button.classList.add('btn-danger');
                }
            });
        }
        
        // Run on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', addDangerClass);
        } else {
            addDangerClass();
        }
        window.addEventListener('load', addDangerClass);
        
        // Observe for dynamically added buttons
        const dangerObserver = new MutationObserver(addDangerClass);
        if (document.body) {
            dangerObserver.observe(document.body, { childList: true, subtree: true });
        }
        
        // Style transaction action buttons
        function styleTxnActionButtons() {
            const txnActions = document.querySelectorAll('.txn-actions');
            txnActions.forEach(container => {
                const buttons = container.querySelectorAll('button');
                buttons.forEach(button => {
                    const buttonText = button.innerText || button.textContent || '';
                    // Remove existing classes first
                    button.classList.remove('btn-primary', 'btn-danger');
                    // Apply classes based on text
                    if (buttonText.includes('Approve & Next')) {
                        button.classList.add('btn-primary');
                    } else if (buttonText.includes('Cancel')) {
                        button.classList.add('btn-danger');
                    }
                });
            });
        }
        
        // Run on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', styleTxnActionButtons);
        } else {
            styleTxnActionButtons();
        }
        window.addEventListener('load', styleTxnActionButtons);
        
        // Observe for dynamically added transaction action buttons
        const txnObserver = new MutationObserver(function(mutations) {
            let shouldRun = false;
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1 && (node.classList.contains('txn-actions') || node.querySelector('.txn-actions'))) {
                        shouldRun = true;
                    }
                });
            });
            if (shouldRun) {
                styleTxnActionButtons();
            }
        });
        if (document.body) {
            txnObserver.observe(document.body, { childList: true, subtree: true });
        }
        
        // Layer B: Targeted JS fixer - detect and fix dark backgrounds only
        function fixDarkSurfaces() {
            // Helper to calculate brightness from RGB
            function getBrightness(rgb) {
                if (!rgb || rgb === 'transparent' || rgb === 'rgba(0, 0, 0, 0)') {
                    return 255; // transparent = bright
                }
                const match = rgb.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
                if (!match) return 255;
                const r = parseInt(match[1]);
                const g = parseInt(match[2]);
                const b = parseInt(match[3]);
                const alpha = match[4] ? parseFloat(match[4]) : 1;
                // Skip if transparent
                if (alpha === 0) return 255;
                // Calculate relative luminance (brightness)
                const brightness = (r * 299 + g * 587 + b * 114) / 1000;
                return brightness;
            }
            
            // Target specific elements that might be dark
            const selectors = [
                'button',
                '[data-testid="stMetricContainer"]',
                '[data-baseweb="button"]',
                '[data-baseweb="card"]',
                '.stButton > button'
            ];
            
            selectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    // Skip if already processed or is an alert/notification
                    if (el.classList.contains('no-dark-surface') || 
                        el.closest('[data-testid="stAlert"]') ||
                        el.closest('[data-baseweb="notification"]') ||
                        el.closest('.stAlert')) {
                        return;
                    }
                    
                    // Skip semantic buttons
                    if (el.classList.contains('btn-primary') || 
                        el.classList.contains('btn-danger') ||
                        el.getAttribute('kind') === 'primary') {
                        return;
                    }
                    
                    const computedStyle = window.getComputedStyle(el);
                    const bgColor = computedStyle.backgroundColor;
                    const brightness = getBrightness(bgColor);
                    
                    // If brightness is below threshold (dark), apply fix
                    if (brightness < 60 && bgColor !== 'transparent' && bgColor !== 'rgba(0, 0, 0, 0)') {
                        el.classList.add('no-dark-surface');
                    }
                });
            });
        }
        
        // Run on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', fixDarkSurfaces);
        } else {
            fixDarkSurfaces();
        }
        window.addEventListener('load', fixDarkSurfaces);
        
        // Observe for dynamically added elements (scoped to avoid performance issues)
        const darkFixObserver = new MutationObserver(function(mutations) {
            let shouldRun = false;
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1 && (
                        node.tagName === 'BUTTON' ||
                        node.hasAttribute('data-testid') ||
                        node.hasAttribute('data-baseweb')
                    )) {
                        shouldRun = true;
                    }
                });
            });
            if (shouldRun) {
                setTimeout(fixDarkSurfaces, 100);
            }
        });
        if (document.body) {
            darkFixObserver.observe(document.body, { childList: true, subtree: true });
        }
        </script>
    """, unsafe_allow_html=True)


def main():
    # Apply custom color scheme
    apply_custom_theme()
    # Sidebar Navigation
    display_logo()
    st.sidebar.markdown("---")
    
    # Initialize page in session state if not set
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    
    # Get current page from session state (this is the source of truth)
    current_page = st.session_state.page
    
    # Determine index for radio button based on session state
    page_options = ["Home", "Upload Statement", "Dashboard", "Chat Assistant", "Settings"]
    try:
        current_index = page_options.index(current_page)
    except ValueError:
        current_index = 0
        st.session_state.page = "Home"
        current_page = "Home"
    
    # Always render the radio button to prevent it from disappearing
    # Check if we just navigated programmatically (before clearing the flag)
    was_navigated = st.session_state.get("page_navigated", False)
    if was_navigated:
        st.session_state.page_navigated = False
    
    # Always render the radio button
    page = st.sidebar.radio(
        "Navigation",
        page_options,
        index=current_index,
        label_visibility="collapsed"
    )
    
    # Only update session state if:
    # 1. Radio button value changed AND
    # 2. We didn't just navigate programmatically (to prevent overwriting)
    if page != current_page and not was_navigated:
        st.session_state.page = page
        st.rerun()
    elif was_navigated:
        # If we navigated programmatically, ensure page matches session state
        # This prevents the radio button from overriding our programmatic navigation
        page = current_page
    
    # Show profile questionnaire on first visit (optional, non-blocking)
    if not st.session_state.user_profile.get("profile_completed") and page != "Settings":
        with st.sidebar:
            st.markdown("---")
            st.markdown('<div class="complete-profile-button-container">', unsafe_allow_html=True)
            if st.button("👤 Complete Profile", use_container_width=True, key="complete_profile_btn"):
                st.session_state.show_questionnaire = True
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Show questionnaire if requested
    if st.session_state.get("show_questionnaire", False):
        show_profile_questionnaire()
        if st.session_state.user_profile.get("profile_completed"):
            st.session_state.show_questionnaire = False
        return
    
    # Route to appropriate page (use session_state.page to allow programmatic navigation)
    current_page = st.session_state.get("page", page)
    if current_page == "Home":
        home_page()
    elif current_page == "Upload Statement":
        upload_statement_page()
    elif current_page == "Dashboard":
        dashboard_page()
    elif current_page == "Chat Assistant":
        chat_assistant_page()
    elif current_page == "Settings":
        settings_page()


if __name__ == "__main__":
    main()
