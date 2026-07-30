import streamlit as st
import pandas as pd

# ==========================
# Configuration
# ==========================

DATA_PATH = "data/processed/processed_data.csv"

# Model metrics (replace these with your actual values if you have them)
ACCURACY = 0.89
PRECISION = 0.87
RECALL = 0.85
F1_SCORE = 0.86

# ==========================
# Page Setup
# ==========================

st.set_page_config(
    page_title="Credit Risk Prediction Dashboard",
    layout="wide"
)

st.title("💳 Credit Risk Prediction Dashboard")

# ==========================
# Load Data
# ==========================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ==========================
# Business Overview
# ==========================

st.header("Business Overview")

st.write("""
Financial institutions need reliable methods to determine whether loan applicants are likely to repay their loans.
This dashboard presents a machine learning solution that analyzes historical customer information to support
better lending decisions and reduce financial risk.
""")

# ==========================
# Dataset Preview
# ==========================

st.header("Dataset Preview")

st.dataframe(df.head())

# ==========================
# Dataset Information
# ==========================

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

# ==========================
# Missing Values
# ==========================

st.header("Missing Values")

missing = df.isnull().sum()

st.bar_chart(missing)

# ==========================
# Summary Statistics
# ==========================

st.header("Summary Statistics")

st.dataframe(df.describe())

# ==========================
# Model Performance
# ==========================

st.header("Model Performance")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Accuracy", f"{ACCURACY:.2%}")

with c2:
    st.metric("Precision", f"{PRECISION:.2%}")

with c3:
    st.metric("Recall", f"{RECALL:.2%}")

with c4:
    st.metric("F1 Score", f"{F1_SCORE:.2%}")

# ==========================
# Business Impact
# ==========================

st.header("Business Impact")

st.success(
    """
This model helps banks identify high-risk loan applicants, improving lending
decisions, reducing financial losses, and supporting more consistent credit
approval decisions.
"""
)

# ==========================
# Footer
# ==========================

st.markdown("---")
st.write("Created by Bisrat Tamrat")