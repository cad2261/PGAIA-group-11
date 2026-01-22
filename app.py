import streamlit as st
import pandas as pd
from datetime import datetime
from pdf_parser import parse_bank_statement_pdf

st.set_page_config(page_title="Finance Assistant MVP", layout="wide")

# ---- Sidebar Navigation ----
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "PDF Upload"],
    label_visibility="collapsed"
)

# ---- Fake users (for now) ----
USERS = [
    {"user_id": "u_001", "display_name": "Demo User"},
    {"user_id": "u_002", "display_name": "Test User"},
]

# ---- User selection (available on all pages) ----
st.sidebar.divider()
user_id = st.sidebar.selectbox(
    "Select user",
    options=[u["user_id"] for u in USERS],
    format_func=lambda x: next(u["display_name"] for u in USERS if u["user_id"] == x)
)
st.session_state["user_id"] = user_id

# ---- OpenAI API Key (optional, for enhanced PDF parsing) ----
st.sidebar.divider()
st.sidebar.subheader("OpenAI API (Optional)")
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key for enhanced PDF parsing. Leave empty to use standard parsing.",
    value=st.session_state.get("openai_api_key", "")
)
st.session_state["openai_api_key"] = openai_api_key
if openai_api_key:
    st.sidebar.success("✓ API key set")
else:
    st.sidebar.info("Using standard PDF parsing")

# ---- Page Routing ----
if page == "Home":
    st.title("💰 Finance Assistant – Streamlit Test v2")
    st.success(f"Active user: {user_id}")
    
    st.divider()
    
    # ---- File upload ----
    uploaded_file = st.file_uploader("Upload bank CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    
        st.subheader("Raw CSV Preview")
        st.dataframe(df.head())
    
        # ---- Minimal normalization (test only) ----
        normalized = []
    
        for _, row in df.iterrows():
            normalized.append({
                "user_id": user_id,
                "posted_date": row.iloc[0],
                "description_raw": row.iloc[1],
                "amount": abs(float(row.iloc[2])),
                "direction": "expense" if float(row.iloc[2]) < 0 else "income",
                "category": "Uncategorized",
                "status": "pending_review",
                "created_at": datetime.utcnow().isoformat()
            })
    
        norm_df = pd.DataFrame(normalized)
    
        st.subheader("Normalized Transactions (Preview)")
        st.dataframe(norm_df)
    
        st.info("This is a Streamlit test only. No data is saved yet.")

elif page == "PDF Upload":
    st.title("📄 PDF Bank Statement Parser")
    st.success(f"Active user: {user_id}")
    
    st.divider()
    
    # ---- PDF Upload Section ----
    st.subheader("Upload PDF Statement")
    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a bank statement PDF to extract transactions"
    )
    
    if uploaded_pdf is not None:
        # Parse button
        if st.button("Parse PDF", type="primary"):
            try:
                api_key = st.session_state.get("openai_api_key", "")
                parse_method = "OpenAI API" if api_key else "Standard"
                with st.spinner(f"Parsing PDF using {parse_method}... This may take a moment."):
                    file_bytes = uploaded_pdf.read()
                    transactions_df, summary = parse_bank_statement_pdf(file_bytes, openai_api_key=api_key if api_key else None)
                    
                    # Store in session state
                    st.session_state["transactions_df"] = transactions_df
                    st.session_state["summary"] = summary
                    st.session_state["pdf_parsed"] = True
                    
                st.success("PDF parsed successfully!")
            except Exception as e:
                st.error(f"Error parsing PDF: {str(e)}")
                st.info("Please ensure the PDF contains transaction data in a readable format.")
                st.session_state["pdf_parsed"] = False
    
    # ---- Display Results ----
    if st.session_state.get("pdf_parsed", False):
        transactions_df = st.session_state["transactions_df"]
        summary = st.session_state["summary"]
        
        st.divider()
        
        # ---- Summary Section ----
        st.subheader("📊 Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Transactions", summary["num_transactions"])
        
        with col2:
            st.metric("Total Income", f"${summary['total_income']:,.2f}")
        
        with col3:
            st.metric("Total Expenses", f"${summary['total_expenses']:,.2f}")
        
        with col4:
            net_color = "normal" if summary["net_flow"] >= 0 else "inverse"
            st.metric("Net Flow", f"${summary['net_flow']:,.2f}", delta=None)
        
        # Statement period
        if summary["statement_period"]:
            st.info(f"📅 Statement Period: {summary['statement_period']}")
        
        st.divider()
        
        # ---- Top Merchants ----
        st.subheader("🏪 Top 10 Merchants by Spend")
        if summary["top_merchants"]:
            merchants_df = pd.DataFrame({
                "Merchant": list(summary["top_merchants"].keys()),
                "Total Spend": [f"${v:,.2f}" for v in summary["top_merchants"].values()]
            })
            st.dataframe(merchants_df, use_container_width=True, hide_index=True)
        else:
            st.info("No merchant data available.")
        
        st.divider()
        
        # ---- Category Spend ----
        st.subheader("📂 Spend by Category")
        if summary["category_spend"]:
            category_df = pd.DataFrame({
                "Category": ["Uncategorized"] * len(summary["category_spend"]),
                "Direction": list(summary["category_spend"].keys()),
                "Amount": [f"${v:,.2f}" for v in summary["category_spend"].values()]
            })
            st.dataframe(category_df, use_container_width=True, hide_index=True)
        else:
            st.info("No category data available.")
        
        st.divider()
        
        # ---- Transactions Table ----
        st.subheader("💳 Extracted Transactions")
        st.dataframe(transactions_df, use_container_width=True, height=400)
        
        # Download button
        csv = transactions_df.to_csv(index=False)
        st.download_button(
            label="Download Transactions as CSV",
            data=csv,
            file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.info("💡 This is a preview only. No data is saved yet.")
