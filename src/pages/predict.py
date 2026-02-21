import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import RAW_DATA_PATH
from src.models.inference import predict


def _build_model_input(form_values: dict) -> dict:
    age = form_values["employment_years"]
    if age <= 2:
        age_bucket = "<25"
    elif age <= 5:
        age_bucket = "25-34"
    elif age <= 10:
        age_bucket = "35-44"
    else:
        age_bucket = "45-54"

    loan_purpose_map = {
        "Personal": "p1",
        "Home Improvement": "p2",
        "Business": "p3",
        "Education": "p4",
    }
    home_map = {"Own": "pr", "Rent": "ir", "Mortgage": "sr"}

    return {
        "Credit_Score": float(form_values["credit_score"]),
        "income": float(form_values["annual_income"]),
        "loan_amount": float(form_values["loan_amount"]),
        "dtir1": float(form_values["debt_to_income"]),
        "term": float(form_values["loan_term"]),
        "age": age_bucket,
        "loan_purpose": loan_purpose_map.get(form_values["loan_purpose"], "p1"),
        "open_credit": "opc" if form_values["open_accounts"] > 0 else "nopc",
        "Neg_ammortization": "neg_amm" if form_values["delinquencies_2y"] > 0 else "not_neg",
        "occupancy_type": home_map.get(form_values["home_ownership"], "pr"),
    }


def _render_gauge(probability: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Default Risk"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1d4ed8"},
                "steps": [
                    {"range": [0, 35], "color": "#ddf6e8"},
                    {"range": [35, 65], "color": "#fff0cf"},
                    {"range": [65, 100], "color": "#fbe1e3"},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _risk_factors(form_values: dict) -> pd.DataFrame:
    factors = {
        "Credit score quality": max(0.0, (700 - float(form_values["credit_score"])) / 700),
        "Debt-to-income pressure": min(1.0, float(form_values["debt_to_income"]) / 100),
        "Loan size exposure": min(1.0, float(form_values["loan_amount"]) / 300000),
        "Income strength": max(0.0, 1 - float(form_values["annual_income"]) / 150000),
        "Delinquency history": min(1.0, float(form_values["delinquencies_2y"]) / 5),
    }
    frame = pd.DataFrame({"factor": list(factors.keys()), "impact": list(factors.values())})
    return frame.sort_values("impact", ascending=False)


def app():
    st.title("Loan Risk Prediction")
    st.caption("Enter borrower details to predict default probability")

    left, right = st.columns([1, 1.35])

    with left:
        st.subheader("Borrower Details")
        with st.form("loan_risk_form"):
            c1, c2 = st.columns(2)
            with c1:
                credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=580)
                loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=35000, step=1000)
                employment_years = st.number_input("Employment Length (yrs)", min_value=0, value=2)
                loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home Improvement", "Business", "Education"])
                delinquencies_2y = st.number_input("Delinquencies (2yr)", min_value=0, value=1)
            with c2:
                annual_income = st.number_input("Annual Income ($)", min_value=0, value=42000, step=1000)
                debt_to_income = st.number_input("Debt-to-Income (%)", min_value=0, max_value=100, value=58)
                loan_term = st.selectbox("Loan Term", [120, 180, 240, 300, 360], index=4)
                open_accounts = st.number_input("Number of Open Accounts", min_value=0, value=12)
                home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])
            submitted = st.form_submit_button("Run Prediction")

    form_values = {
        "credit_score": credit_score,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "debt_to_income": debt_to_income,
        "employment_years": employment_years,
        "loan_term": loan_term,
        "loan_purpose": loan_purpose,
        "open_accounts": open_accounts,
        "delinquencies_2y": delinquencies_2y,
        "home_ownership": home_ownership,
    }

    if submitted:
        model_input = _build_model_input(form_values)
        result = predict(model_input)
        probability = float(result["default_probability"]) if result["default_probability"] is not None else 0.0
        st.session_state["last_prediction"] = {"label": int(result["prediction"]), "probability": probability, "form": form_values}

    with right:
        state = st.session_state.get("last_prediction")
        if not state:
            st.markdown("### Fill in borrower details and click **Run Prediction**")
            st.caption("Risk gauge, confidence bar, and recommendations will appear here.")
            return

        probability = state["probability"]
        label = state["label"]
        risk_text = "High Risk" if probability >= 0.65 else "Medium Risk" if probability >= 0.35 else "Low Risk"
        risk_color = "#d9363e" if probability >= 0.65 else "#e49b0f" if probability >= 0.35 else "#1f9d66"

        _render_gauge(probability)
        st.progress(min(max(probability, 0.0), 1.0))
        st.markdown(f"**Prediction:** <span style='color:{risk_color};'>{risk_text}</span>", unsafe_allow_html=True)

        rf = _risk_factors(state["form"])
        fig_rf = go.Figure(go.Bar(x=rf["impact"], y=rf["factor"], orientation="h", marker=dict(color="#3b82f6")))
        fig_rf.update_layout(height=250, title="Risk Factor Breakdown", margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_rf, use_container_width=True)

        st.subheader("Recommendations")
        if label == 1 or probability >= 0.5:
            st.error("Use stricter underwriting and ask for additional income verification.")
            st.write("- Consider reducing exposure or requiring stronger collateral.")
            st.write("- Offer shorter tenure and risk-adjusted pricing.")
        else:
            st.success("Profile appears acceptable under current risk threshold.")
            st.write("- Continue standard approval checks.")
            st.write("- Monitor account behavior in first 90 days.")
