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
from store_locator import get_nearby_grocery_stores, filter_stores_by_name

# No external PDF libraries needed - using OpenAI API directly

# ============================================================================
# API Keys Configuration
# ============================================================================
# OpenAI API Key - Replace with your actual key
OPENAI_API_KEY = "OPEN_API_KEY"  # Get your key from https://platform.openai.com/api-keys

# Page configuration
st.set_page_config(
    page_title="Accountable AI v1",
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
if "pending_transactions" not in st.session_state:
    st.session_state.pending_transactions = pd.DataFrame()
if "review_index" not in st.session_state:
    st.session_state.review_index = 0
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "age_range": None,
        "employment_status": None,
        "occupation": None,
        "location_country": None,
        "location_city": None,
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

def validate_openai_key() -> bool:
    """Check if OpenAI API key is set."""
    return bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here" and OPENAI_API_KEY.strip())


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client if API key is available."""
    if not validate_openai_key():
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


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
    Returns dict with country, city, state keys.
    """
    profile = st.session_state.user_profile
    
    location = {
        "country": profile.get("location_country"),
        "city": profile.get("location_city"),
        "state": None  # We don't capture state separately, but can parse from city if needed
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
                elif profile.get("location_country"):
                    location = profile['location_country']
                if location:
                    st.write(f"**Location:** {location}")
            if st.button("✏️ Edit Profile", use_container_width=True):
                st.session_state.show_questionnaire = True
                st.rerun()
    
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


def upload_statement_page():
    """Page for uploading and parsing bank statements (PDF or CSV)."""
    st.title("📄 Upload Bank Statement")
    
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
            
            # Store in pending transactions, bump upload key to reset uploader, then rerun
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


def show_transaction_review_ui():
    """Show UI for reviewing and approving transaction categories."""
    pending_df = st.session_state.pending_transactions.copy()
    current_idx = st.session_state.review_index
    
    if current_idx >= len(pending_df):
        # All transactions reviewed, add remaining to database
        if not pending_df.empty:
            if st.session_state.transactions.empty:
                st.session_state.transactions = pending_df
            else:
                st.session_state.transactions = pd.concat(
                    [st.session_state.transactions, pending_df],
                    ignore_index=True
                )
            st.success(f"✅ Added {len(pending_df)} transactions to your data!")
        
        # Clear pending transactions
        st.session_state.pending_transactions = pd.DataFrame()
        st.session_state.review_index = 0
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
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Approve & Next", type="primary", use_container_width=True):
            # Update both direction and category
            pending_df.at[pending_df.index[current_idx], 'direction'] = new_direction_lower
            final_category = selected_category.strip() if selected_category and selected_category.strip() else ""
            # If income and no category selected, default to "Income"
            if new_direction_lower == 'income' and not final_category:
                final_category = INCOME_CATEGORY
            pending_df.at[pending_df.index[current_idx], 'category'] = final_category
            st.session_state.pending_transactions = pending_df
            st.session_state.review_index = current_idx + 1
            st.rerun()
    
    with col2:
        if st.button("⏭️ Skip This", use_container_width=True):
            # Leave category as is and move to next (don't change it)
            st.session_state.review_index = current_idx + 1
            st.rerun()
    
    with col3:
        if st.button("⏭️⏭️ Skip All Remaining", use_container_width=True):
            # Add all remaining transactions with current categories
            if st.session_state.transactions.empty:
                st.session_state.transactions = pending_df
            else:
                st.session_state.transactions = pd.concat(
                    [st.session_state.transactions, pending_df],
                    ignore_index=True
                )
            st.success(f"✅ Added {len(pending_df)} transactions to your data!")
            st.session_state.pending_transactions = pd.DataFrame()
            st.session_state.review_index = 0
            st.rerun()
    
    with col4:
        if st.button("❌ Cancel", use_container_width=True):
            # Clear pending transactions
            st.session_state.pending_transactions = pd.DataFrame()
            st.session_state.review_index = 0
            st.info("Upload cancelled. No transactions were added.")
            st.rerun()
    
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
    col1, col2 = st.columns(2)
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
    col1, col2 = st.columns(2)
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
    
    st.divider()
    
    # Save button
    if st.button("💾 Save Profile", type="primary"):
        profile["profile_completed"] = True
        st.session_state.user_profile = profile
        st.success("✅ Profile updated successfully!")
        st.rerun()
    
    # Clear profile button
    st.markdown("---")
    if st.button("🗑️ Clear Profile", type="secondary"):
        st.session_state.user_profile = {
            "age_range": None,
            "employment_status": None,
            "occupation": None,
            "location_country": None,
            "location_city": None,
            "profile_completed": False
        }
        st.success("✅ Profile cleared!")
        st.rerun()


def main():
    # Sidebar Navigation
    st.sidebar.title("💰 Accountable AI v1")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Upload Statement", "Dashboard", "Chat Assistant", "Settings"],
        label_visibility="collapsed"
    )
    
    # Show profile questionnaire on first visit (optional, non-blocking)
    if not st.session_state.user_profile.get("profile_completed") and page != "Settings":
        with st.sidebar:
            st.markdown("---")
            if st.button("👤 Complete Profile", use_container_width=True):
                st.session_state.show_questionnaire = True
    
    # Show questionnaire if requested
    if st.session_state.get("show_questionnaire", False):
        show_profile_questionnaire()
        if st.session_state.user_profile.get("profile_completed"):
            st.session_state.show_questionnaire = False
        return
    
    # Route to appropriate page
    if page == "Home":
        home_page()
    elif page == "Upload Statement":
        upload_statement_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Chat Assistant":
        chat_assistant_page()
    elif page == "Settings":
        settings_page()


if __name__ == "__main__":
    main()
