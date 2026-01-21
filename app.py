import streamlit as st
import pandas as pd
from datetime import datetime
st.set_page_config(page_title="Finance Assistant MVP", layout="wide")

st.title("💰 Finance Assistant – Streamlit Test" -- edit)

# ---- Fake users (for now) ----
USERS = [
    {"user_id": "u_001", "display_name": "Demo User"},
    {"user_id": "u_002", "display_name": "Test User"},
]

# ---- User selection ----
user_id = st.selectbox(
    "Select user",
    options=[u["user_id"] for u in USERS],
    format_func=lambda x: next(u["display_name"] for u in USERS if u["user_id"] == x)
)

st.session_state["user_id"] = user_id
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
###test
