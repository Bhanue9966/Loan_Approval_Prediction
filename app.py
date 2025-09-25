import streamlit as st
import numpy as np
import pandas as pd
import joblib # Used for saving/loading scikit-learn models and encoders

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide", # Use the full width of the browser
    initial_sidebar_state="collapsed"
)

# Configuration
MODEL_PATH = 'random_forest_model.joblib'
LE_EDUCATION_PATH = 'le_education.joblib'
LE_SELF_EMPLOYED_PATH = 'le_self_employed.joblib'

# --- Load Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        le_education = joblib.load(LE_EDUCATION_PATH)
        le_self_employed = joblib.load(LE_SELF_EMPLOYED_PATH)
        return model, le_education, le_self_employed
    except FileNotFoundError as e:
        st.error(f"Required model or encoder file not found: {e.filename}. Please run your training pipeline first.")
        return None, None, None

model, le_education, le_self_employed = load_assets()

# --- Streamlit App ---

st.title('🏦 Loan Approval Prediction Engine')
st.markdown("---") # Visual separator

if model and le_education and le_self_employed:
    
    # Use tabs for clean organization
    tab1, tab2 = st.tabs(["📊 Applicant Data", "🏠 Asset Valuation"])

    with tab1:
        st.subheader("Applicant and Loan Details")
        
        # --- Applicant Details ---
        col1, col2, col3 = st.columns(3)
        with col1:
            no_of_dependents = st.slider('Number of Dependents', 0, 5, 1)
            education = st.selectbox('Education Level', le_education.classes_)
        with col2:
            income_annum = st.number_input('Annual Income (₹)', min_value=100000, value=7500000, step=100000)
            self_employed = st.selectbox('Self Employed', le_self_employed.classes_)
        with col3:
            cibil_score = st.slider('CIBIL Score', 300, 900, 780, help="Higher score indicates lower risk.")
            loan_term = st.slider('Loan Term (Years)', 2, 20, 10)

        st.markdown('### Requested Amount')
        loan_amount = st.number_input('Requested Loan Amount (₹)', min_value=1000000, value=15000000, step=1000000)

    with tab2:
        st.subheader("Valuation of Assets")
        st.info('Please enter the current market value of all assets. (Values will be summed for Total Assets feature)')
        
        # --- Asset Details ---
        col_res, col_com, col_lux, col_bank = st.columns(4)
        with col_res:
            residential_assets_value = st.number_input('Residential Assets (₹)', min_value=0, value=5000000, help="Value of residential property.")
        with col_com:
            commercial_assets_value = st.number_input('Commercial Assets (₹)', min_value=0, value=2000000, help="Value of commercial property.")
        with col_lux:
            luxury_assets_value = st.number_input('Luxury Assets (₹)', min_value=0, value=8000000, help="Value of high-value non-essential items.")
        with col_bank:
            bank_asset_value = st.number_input('Bank Assets (₹)', min_value=0, value=4000000, help="Total cash and bank balance.")

    st.markdown("---")
    
    # Use a container for the prediction button and output
    prediction_container = st.container()
    
    with prediction_container:
        st.subheader("Run Prediction")
        
        # Center the button
        col_button = st.columns([1, 1, 1])
        if col_button[1].button('Analyze Loan Eligibility', use_container_width=True, type='primary'):
            
            # --- Prediction Logic ---
            
            # 1. Replicate Preprocessing
            total_assets_value = (
                residential_assets_value + 
                commercial_assets_value +
                luxury_assets_value +
                bank_asset_value
            )
            
            # Encode categorical inputs
            education_encoded = le_education.transform([education])[0]
            self_employed_encoded = le_self_employed.transform([self_employed])[0]
            
            # Prepare Input DataFrame
            feature_columns = [
                'no_of_dependents', 'education', 'self_employed', 'income_annum', 
                'loan_amount', 'loan_term', 'cibil_score', 'total_assets_value'
            ]
            input_data_list = [
                no_of_dependents, education_encoded, self_employed_encoded, 
                income_annum, loan_amount, loan_term, cibil_score, total_assets_value
            ]
            input_data = pd.DataFrame([input_data_list], columns=feature_columns)
            
            # 2. Predict Loan Status
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0][1] # Probability of Approval (class 1)

            st.markdown("## Prediction Outcome")
            
            # Display result with highly visual metrics
            col_metric_1, col_metric_2 = st.columns([1, 2])
            
            with col_metric_1:
                st.metric(
                    label="Probability of Approval", 
                    value=f"{prediction_proba * 100:.1f}%",
                    delta=f"{prediction_proba:.2f}" # Show raw probability as a delta for fun
                )
            
            with col_metric_2:
                if prediction == 1:
                    st.success('✅ **LOAN APPROVED!** - The model indicates a low risk of default.')
                    st.balloons()
                else:
                    st.error('❌ **LOAN REJECTED!** - The model indicates a significant risk.')
            
            # Optional: Display Feature Contributions (advanced explainability)
            st.markdown("### Model Confidence")
            st.progress(prediction_proba, text="Approval Confidence Score")