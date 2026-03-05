import streamlit as st
import pandas as pd
import joblib
import xgboost

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💳",
    layout="wide"
)

# --- 1. Load Trained Model and Supporting Artifacts ---
# Using joblib to load the model, feature list, and scaler
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('xgboost_model.pkl')
        features = joblib.load('model_features.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, features, scaler
    except Exception as e:
        st.error(f"Error loading model or supporting files: {e}")
        return None, None, None

model, model_features, scaler = load_artifacts()

# --- 2. Streamlit Page Layout Setup ---
st.title("Loan Default Prediction System")
st.write("This system predicts whether a loan applicant may default based on their profile and financial history.")

# --- 3. Borrower Information Input Section ---
st.header("Borrower Information Input")

# Use two columns for better layout
col1, col2 = st.columns(2)

with col1:
    Credit_Score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    income = st.number_input("Income", min_value=0.0, value=50000.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=200000.0)
    term = st.number_input("Term", min_value=12, max_value=360, value=360)
    dtir1 = st.number_input("dtir1", min_value=0.0, max_value=100.0, value=35.0)

with col2:
    # Mapping user-friendly labels to model categories
    loan_purpose_map = {
    "Home Loan": "p1",
    "Education Loan": "p2",
    "Business Loan": "p3",
    "Personal Loan": "p4"
}

    loan_purpose_label = st.selectbox(
        "Loan Purpose",
        options=list(loan_purpose_map.keys())
    )

    # Convert selected label to model value
    loan_purpose = loan_purpose_map[loan_purpose_label]
    # Mapping user-friendly labels to model categories
    loan_limit_map = {
        "Conforming Loan": "cf",
        "Non-Conforming Loan": "ncf"
    }

    loan_limit_label = st.selectbox(
        "Loan Limit Type",
        options=list(loan_limit_map.keys())
    )

    # Convert selected label to model value
loan_limit = loan_limit_map[loan_limit_label]

# --- 4. Prediction Result Section ---
st.header("Prediction Result")

if st.button("Predict Default Risk"):
    if model is None or scaler is None or model_features is None:
        st.error("Model artifacts are not loaded properly. Cannot proceed.")
    else:
        # Build dictionary of user inputs
        input_dict = {
            'Credit_Score': Credit_Score,
            'income': income,
            'loan_amount': loan_amount,
            'term': term,
            'dtir1': dtir1
        }
        
        # Convert categorical inputs to the same one-hot encoded format used during training
        if loan_purpose in ['p2', 'p3', 'p4']:
            input_dict[f'loan_purpose_{loan_purpose}'] = 1
            
        if loan_limit == 'ncf':
            input_dict['loan_limit_ncf'] = 1
            
        # Create pandas DataFrame from the gathered user inputs
        input_df = pd.DataFrame([input_dict])
        
        # Ensure the DataFrame columns are ordered exactly to match `model_features`
        input_df = input_df.reindex(columns=model_features, fill_value=0)
        
        # numerical columns used during training
        numeric_cols = ['Credit_Score', 'income', 'loan_amount', 'term', 'dtir1']

        # scale only numeric columns
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]
        
        # Display results clearly
        if prediction == 1:
            st.error("⚠️ High Risk: The applicant is likely to default.")
        else:
            st.success("✅ Low Risk: The applicant is unlikely to default.")
            
        st.write(f"**Default Probability:** {prediction_proba * 100:.2f}%")

        # --- 5. Risk Assessment Section ---
        st.header("Risk Assessment")
        risk_percentage = prediction_proba * 100
        st.progress(int(risk_percentage))

        if risk_percentage < 30:
            st.success("Low Risk Borrower")
        elif risk_percentage <= 60:
            st.warning("Moderate Risk Borrower")
        else:
            st.error("High Risk Borrower")
