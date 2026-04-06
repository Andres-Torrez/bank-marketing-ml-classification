from __future__ import annotations
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import streamlit as st
from utils import load_model, build_input_dataframe
from src.monitoring.feedback_logger import log_prediction


st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="📊",
    layout="wide",
)

model = load_model()

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("📌 Project Info")
    st.markdown("""
    **Problem:** Predict whether a client will subscribe to a term deposit.

    **Final Model:** GradientBoostingClassifier

    **Primary Metric:** ROC-AUC

    **Test ROC-AUC:** 0.8019

    **Overfitting Gap:** 0.0147 ✅
    """)
    
    st.markdown("---")
    st.markdown("""
    ### Notes
    - Target is imbalanced
    - `duration` was excluded due to leakage risk
    - Model optimized for stable generalization
    """)

# -------------------------
# Header
# -------------------------
st.title("📊 Bank Marketing Subscription Predictor")
st.markdown("""
This application estimates the probability that a client will subscribe to a term deposit based on demographic and campaign-related features.

Use the form below to simulate a client profile and obtain a prediction from the final production model.
""")

st.markdown("---")

# -------------------------
# Input Form
# -------------------------
st.subheader("Client Input Form")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Client Profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox(
        "Job",
        [
            "admin.", "unknown", "unemployed", "management", "housemaid",
            "entrepreneur", "student", "blue-collar", "self-employed",
            "retired", "technician", "services"
        ],
    )
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["unknown", "secondary", "primary", "tertiary"])
    default = st.selectbox("Credit in Default?", ["no", "yes"])
    balance = st.number_input("Average Yearly Balance", value=0)
    housing = st.selectbox("Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Personal Loan?", ["no", "yes"])

with col2:
    st.markdown("### Campaign Information")
    contact = st.selectbox("Contact Type", ["unknown", "telephone", "cellular"])
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
    month = st.selectbox(
        "Last Contact Month",
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    )
    campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
    pdays = st.number_input("Days Since Previous Contact", value=-1)
    previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)
    poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "other", "failure", "success"])

st.markdown("---")

user_input = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
}

# -------------------------
# Prediction Section
# -------------------------
if st.button("Run Prediction", use_container_width=True):
    input_df = build_input_dataframe(user_input)

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
    else:
        probability = None

    log_prediction(
        user_input=user_input,
        prediction=int(prediction),
        probability=float(probability) if probability is not None else None,
    )

    st.subheader("Prediction Result")

    if probability is not None:
        st.metric("Predicted Subscription Probability", f"{probability:.2%}")

        if probability >= 0.60:
            st.success("High likelihood of subscription")
        elif probability >= 0.35:
            st.warning("Medium likelihood of subscription")
        else:
            st.error("Low likelihood of subscription")

    if prediction == 1:
        st.success("Final Prediction: Client likely to subscribe.")
    else:
        st.info("Final Prediction: Client unlikely to subscribe.")

    with st.expander("See submitted input data"):
        st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.caption("Production note: `duration` is intentionally excluded because it introduces data leakage.")