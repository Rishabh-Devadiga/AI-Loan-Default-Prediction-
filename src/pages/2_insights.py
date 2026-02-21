import streamlit as st

from src.models.inference import predict


def _build_minimal_profile():
    st.write("Enter borrower details to predict loan default risk.")
    with st.form(key="loan_form"):
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0, step=1000.0)
        interest_rate = st.number_input("Interest Rate", min_value=0.0, value=7.5, step=0.1, format="%.2f")
        term = st.number_input("Term (in months)", min_value=1, value=360, step=1)
        income = st.number_input("Annual Income", min_value=0.0, value=60000.0, step=1000.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=720, step=1)
        submitted = st.form_submit_button(label="Predict")

    profile = {
        "loan_amount": float(loan_amount),
        "rate_of_interest": float(interest_rate),
        "term": float(term),
        "income": float(income),
        "Credit_Score": float(credit_score),
    }
    return profile, submitted


st.title("Loan Default Prediction Insights")
input_profile, submit_button = _build_minimal_profile()

if submit_button:
    result = predict(input_profile)
    label = result["prediction"]
    prob = result["default_probability"]

    if label == 1:
        st.error("The borrower is likely to default on the loan.")
    else:
        st.success("The borrower is likely to repay the loan.")

    if prob is not None:
        st.metric("Predicted Default Probability", f"{prob * 100:.2f}%")

    st.write("### Recommendations")
    if label == 1:
        st.write("- Consider additional verification of the borrower's financial status.")
        st.write("- Offer a lower loan amount or stricter lending terms.")
    else:
        st.write("- Proceed with the loan approval process.")
