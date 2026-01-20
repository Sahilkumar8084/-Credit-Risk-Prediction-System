import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Credit Risk Prediction System")
st.caption(
    "Predict whether a loan applicant is **High Risk** or **Low Risk** using a trained ML model"
)

st.divider()

# =====================================================
# LOAD TRAINED PIPELINE
# =====================================================
@st.cache_resource
def load_pipeline():
    bundle = joblib.load(r"xgb_model.pkl")
    return (
        bundle["best_model"],
        bundle["scaler"],
        bundle["encoder"],        # home ownership encoder
        bundle["loanencoder"],    # loan intent encoder
        bundle["ohe"],            # loan grade ordinal encoder
        bundle["threshold"]
    )

model, scaler, home_encoder, intent_encoder, grade_encoder, threshold = load_pipeline()

# =====================================================
# INPUT SECTION
# =====================================================
st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income", min_value=0, value=50000)
    employment_length = st.number_input("Employment Length (Years)", min_value=0.0, value=5.0)
    credit_hist_length = st.number_input("Credit History Length (Years)", min_value=0, value=5)

with col2:
    home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

st.subheader("💰 Loan Details")

col3, col4 = st.columns(2)

with col3:
    loan_amount = st.number_input("Loan Amount", min_value=0, value=20000)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)

with col4:
    loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.3)

st.divider()

# =====================================================
# DATA PREPARATION
# =====================================================
input_df = pd.DataFrame({
    "age": [age],
    "income": [income],
    "home_ownership": [home_ownership],
    "employment_length": [employment_length],
    "loan_intent": [loan_intent],
    "loan_grade": [loan_grade],
    "loan_amount": [loan_amount],
    "interest_rate": [interest_rate],
    "loan_percent_income": [loan_percent_income],
    "credit_hist_length": [credit_hist_length]
})

# -------------------- ENCODING --------------------
# input_df["home_ownership"] = home_encoder.transform(input_df["home_ownership"])
# input_df["loan_intent"] = intent_encoder.transform(input_df["loan_intent"])
# input_df["loan_grade"] = grade_encoder.transform(input_df[["loan_grade"]])

home_ownership_map = {
    "MORTGAGE": 0,
    "OTHER": 1,
    "OWN": 2,
    "RENT": 3
}

input_df["home_ownership"] = input_df["home_ownership"].map(home_ownership_map)


input_df["loan_intent"] = intent_encoder.transform(
    input_df[["loan_intent"]]
)

grade_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
}

input_df["loan_grade"] = input_df["loan_grade"].map(grade_map)



# -------------------- SCALING --------------------
X_scaled = scaler.transform(input_df)

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔍 Predict Credit Risk", use_container_width=True):

    prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(prob >= threshold)

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(
            f"🚨 **High Credit Risk Applicant**\n\n"
            f"Probability of Default: **{prob:.2%}**"
        )
    else:
        st.success(
            f"✅ **Low Credit Risk Applicant**\n\n"
            f"Probability of Default: **{prob:.2%}**"
        )

    with st.expander("🔎 View Processed Input Data"):
        st.dataframe(input_df)

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption(
    "📌 This system uses an XGBoost-based Machine Learning model trained on real-world credit data. "
    "Predictions are for educational and decision-support purposes only."
)
