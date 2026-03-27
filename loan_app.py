import streamlit as st
import pandas as pd
import joblib
import xgboost

st.set_page_config(
    page_title="Loan Default Risk Analyzer",
    page_icon="💳",
    layout="wide"
)

# --- UI Theme Styling ---
st.markdown(
    """
    <style>
    :root {
        --bg: #0d1117;
        --panel: #111827;
        --panel-2: #0f172a;
        --text: #e5e7eb;
        --muted: #9ca3af;
        --accent: #22c55e;
        --accent-2: #16a34a;
        --danger: #ef4444;
        --warning: #f59e0b;
    }

    .stApp {
        background: radial-gradient(1200px 800px at 15% 5%, #101827 0%, #0b101a 35%, #0a0f18 100%);
        color: var(--text);
    }

    div[data-testid="stVerticalBlock"] > div:has(div.stButton) {
        padding-top: 0.25rem;
    }

    .section-card {
        background: linear-gradient(145deg, rgba(17,24,39,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 1.25rem 1.25rem 0.75rem 1.25rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    .result-card {
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 12px 28px rgba(0,0,0,0.3);
        font-weight: 600;
        font-size: 1.05rem;
    }

    .result-low {
        background: rgba(34, 197, 94, 0.15);
        border-color: rgba(34, 197, 94, 0.5);
        color: #d1fae5;
    }

    .result-high {
        background: rgba(239, 68, 68, 0.15);
        border-color: rgba(239, 68, 68, 0.5);
        color: #fee2e2;
    }

    .caption {
        color: var(--muted);
        font-size: 0.9rem;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
        border: none !important;
        color: #051b10 !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.6rem !important;
    }

    button[kind="secondary"] {
        border-radius: 999px !important;
    }

    input, textarea, select, div[data-baseweb="select"] > div {
        border-radius: 12px !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 1. Load Trained Model and Supporting Artifacts ---
# Using joblib to load the model, feature list, and scaler
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("xgboost_model.pkl")
        features = joblib.load("model_features.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, features, scaler
    except Exception as e:
        st.error(f"Error loading model or supporting files: {e}")
        return None, None, None

model, model_features, scaler = load_artifacts()

# --- 2. Streamlit Page Layout Setup ---
st.markdown("## Loan Default Risk Analyzer")
st.markdown('<div class="caption">AI-powered credit risk prediction system</div>', unsafe_allow_html=True)

st.write("")

# --- Session State Defaults ---
if "Credit_Score" not in st.session_state:
    st.session_state.Credit_Score = 700
if "income" not in st.session_state:
    st.session_state.income = 50000.0
if "loan_amount" not in st.session_state:
    st.session_state.loan_amount = 200000.0
if "term" not in st.session_state:
    st.session_state.term = 360
if "dtir1" not in st.session_state:
    st.session_state.dtir1 = 35.0
if "loan_purpose_label" not in st.session_state:
    st.session_state.loan_purpose_label = "Home Loan"
if "loan_limit_label" not in st.session_state:
    st.session_state.loan_limit_label = "Conforming Loan"


def apply_example():
    st.session_state.Credit_Score = 745
    st.session_state.income = 82000.0
    st.session_state.loan_amount = 250000.0
    st.session_state.term = 360
    st.session_state.dtir1 = 28.0
    st.session_state.loan_purpose_label = "Home Loan"
    st.session_state.loan_limit_label = "Conforming Loan"


# --- 3. Borrower Information Input Section ---
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Financial Information")
    Credit_Score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=st.session_state.Credit_Score,
        help="Typical ranges: 300-579 (poor), 580-669 (fair), 670-739 (good), 740+ (excellent).",
        key="Credit_Score",
    )
    income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=st.session_state.income,
        step=1000.0,
        help="Enter the borrower's gross annual income.",
        key="income",
    )

    st.write("")
    st.markdown("### 🧾 Loan Details")
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=st.session_state.loan_amount,
        step=1000.0,
        help="Total principal requested.",
        key="loan_amount",
    )
    term = st.selectbox(
        "Term (months)",
        options=[120, 180, 240, 360],
        index=[120, 180, 240, 360].index(st.session_state.term),
        help="Choose the repayment duration in months.",
        key="term",
    )
    dtir1 = st.slider(
        "Debt-to-Income Ratio (DTI %)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.dtir1,
        help="Monthly debt payments divided by monthly income.",
        key="dtir1",
    )

    loan_purpose_map = {
        "Home Loan": "p1",
        "Education Loan": "p2",
        "Business Loan": "p3",
        "Personal Loan": "p4",
    }
    loan_purpose_label = st.selectbox(
        "Loan Purpose",
        options=list(loan_purpose_map.keys()),
        key="loan_purpose_label",
    )

    loan_limit_map = {
        "Conforming Loan": "cf",
        "Non-Conforming Loan": "ncf",
    }
    loan_limit_label = st.selectbox(
        "Loan Type",
        options=list(loan_limit_map.keys()),
        key="loan_limit_label",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    if st.button("Try Example", use_container_width=True):
        apply_example()

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Prediction Results")
    st.write("")
    predict_clicked = st.button("Predict Risk", type="primary", use_container_width=True)

    if predict_clicked:
        if model is None or scaler is None or model_features is None:
            st.error("Model artifacts are not loaded properly. Cannot proceed.")
        else:
            with st.spinner("Analyzing risk..."):
                loan_purpose = loan_purpose_map[st.session_state.loan_purpose_label]
                loan_limit = loan_limit_map[st.session_state.loan_limit_label]

                # Build dictionary of user inputs
                input_dict = {
                    "Credit_Score": st.session_state.Credit_Score,
                    "income": st.session_state.income,
                    "loan_amount": st.session_state.loan_amount,
                    "term": st.session_state.term,
                    "dtir1": st.session_state.dtir1,
                }

                # Convert categorical inputs to the same one-hot encoded format used during training
                if loan_purpose in ["p2", "p3", "p4"]:
                    input_dict[f"loan_purpose_{loan_purpose}"] = 1

                if loan_limit == "ncf":
                    input_dict["loan_limit_ncf"] = 1

                # Create pandas DataFrame from the gathered user inputs
                input_df = pd.DataFrame([input_dict])

                # Ensure the DataFrame columns are ordered exactly to match `model_features`
                input_df = input_df.reindex(columns=model_features, fill_value=0)

                # numerical columns used during training
                numeric_cols = ["Credit_Score", "income", "loan_amount", "term", "dtir1"]

                # scale only numeric columns
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0][1]

            risk_percentage = prediction_proba * 100
            st.metric(
                label="Default Probability",
                value=f"{risk_percentage:.2f}%"
            )

            if prediction == 1:
                st.markdown(
                    '<div class="result-card result-high">🚨 High Risk: Applicant is likely to default.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card result-low">✅ Low Risk: Applicant is unlikely to default.</div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown("#### 🔎 Risk Insight")
            st.progress(int(risk_percentage))

            if risk_percentage < 30:
                st.success("Low Risk Borrower")
            elif risk_percentage <= 60:
                st.warning("Moderate Risk Borrower")
            else:
                st.error("High Risk Borrower")
    else:
        st.info("Run a prediction to view risk insights and probability.")

    st.markdown("</div>", unsafe_allow_html=True)
