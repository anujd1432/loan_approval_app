import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
)

# ── Load model & features ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("c:/Users/amand/Downloads/model (2).pkl")
    with open("c:/Users/amand/Downloads/features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

try:
    model, feature_names = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"⚠️ Could not load model files: {e}\n\n"
        "Make sure `model__2_.pkl` and `features.pkl` are in the **same folder** "
        "as this script, then restart the app."
    )
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏦 Loan Approval Predictor")
st.markdown(
    "Fill in the applicant details below and click **Predict** to see "
    "whether the loan is likely to be approved."
)
st.divider()

# ── Input form ───────────────────────────────────────────────────────────────
with st.form("loan_form"):
    st.subheader("👤 Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    with col2:
        dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=0, step=1
        )
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"]
        )
        credit_history = st.selectbox(
            "Credit History",
            [1, 0],
            format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)",
        )

    st.subheader("💰 Financial Details")
    col3, col4 = st.columns(2)

    with col3:
        applicant_income = st.number_input(
            "Applicant Income (₹)", min_value=0, value=5000, step=500
        )
        coapplicant_income = st.number_input(
            "Co-Applicant Income (₹)", min_value=0, value=0, step=500
        )

    with col4:
        loan_amount = st.number_input(
            "Loan Amount (₹ thousands)", min_value=1, value=150, step=10
        )
        loan_amount_term = st.selectbox(
            "Loan Term (months)",
            [360, 180, 120, 60, 36, 12],
            index=0,
        )

    submitted = st.form_submit_button("🔍 Predict Loan Approval", use_container_width=True)

# ── Prediction logic ─────────────────────────────────────────────────────────
if submitted:
    # Build a dict matching the one-hot encoded feature names
    data = {
        "Dependents": dependents,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        # Gender
        "Gender_Female": 1 if gender == "Female" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        # Married
        "Married_No": 1 if married == "No" else 0,
        "Married_Yes": 1 if married == "Yes" else 0,
        # Education
        "Education_Graduate": 1 if education == "Graduate" else 0,
        "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
        # Self Employed
        "Self_Employed_No": 1 if self_employed == "No" else 0,
        "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
        # Property Area
        "Property_Area_Rural": 1 if property_area == "Rural" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
    }

    # Align to model's expected feature order
    input_df = pd.DataFrame([data])[feature_names]

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]  # [prob_0, prob_1]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ **Loan Approved!**")
        st.metric("Approval Confidence", f"{proba[1]*100:.1f}%")
    else:
        st.error("❌ **Loan Not Approved**")
        st.metric("Rejection Confidence", f"{proba[0]*100:.1f}%")

    # Probability bar
    prob_df = pd.DataFrame(
        {"Outcome": ["Not Approved", "Approved"], "Probability": [proba[0], proba[1]]}
    )
    st.bar_chart(prob_df.set_index("Outcome"))

    # Show submitted input summary
    with st.expander("📋 Input Summary"):
        summary = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self Employed": self_employed,
            "Applicant Income": f"₹{applicant_income:,}",
            "Co-Applicant Income": f"₹{coapplicant_income:,}",
            "Loan Amount": f"₹{loan_amount}K",
            "Loan Term": f"{loan_amount_term} months",
            "Credit History": "Good" if credit_history == 1 else "Bad",
            "Property Area": property_area,
        }
        st.table(pd.DataFrame(summary.items(), columns=["Field", "Value"]))

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: Logistic Regression | Features: 17 | Built with Streamlit 🎈")