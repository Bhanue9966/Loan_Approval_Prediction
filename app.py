import streamlit as st
import numpy as np
import pandas as pd
import joblib # Used for saving/loading scikit-learn models and encoders

# Configuration
MODEL_PATH = 'random_forest_model.joblib'
LE_EDUCATION_PATH = 'le_education.joblib'
LE_SELF_EMPLOYED_PATH = 'le_self_employed.joblib'

# Load Assets
@st.cache_resource # Cache the model and encoders so they load once
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
st.title('💰 Customer Loan Approval Prediction')
st.markdown('### Input Applicant Financial and Credit Details')

if model and le_education and le_self_employed:
    
    # User Input Fields
    
    # --- Applicant Details ---
    col1, col2 = st.columns(2)
    with col1:
        no_of_dependents = st.slider('Number of Dependents', 0, 5, 2)
        income_annum = st.number_input('Annual Income (₹)', min_value=100000, value=5000000, step=100000)
        cibil_score = st.slider('CIBIL Score', 300, 900, 750)
    
    with col2:
        # Use classes from the fitted LabelEncoders
        education = st.selectbox('Education', le_education.classes_)
        self_employed = st.selectbox('Self Employed', le_self_employed.classes_)
        loan_term = st.slider('Loan Term (Years)', 2, 20, 10)

    # --- Loan & Asset Details ---
    loan_amount = st.number_input('Requested Loan Amount (₹)', min_value=1000000, value=15000000, step=1000000)
    st.markdown('---')
    st.subheader('Asset Valuation')
    st.markdown('*(These values will be summed to create the Total Assets feature)*')
    
    col3, col4 = st.columns(2)
    with col3:
        residential_assets_value = st.number_input('Residential Assets (₹)', min_value=0, value=5000000)
        luxury_assets_value = st.number_input('Luxury Assets (₹)', min_value=0, value=5000000)
    with col4:
        commercial_assets_value = st.number_input('Commercial Assets (₹)', min_value=0, value=5000000)
        bank_asset_value = st.number_input('Bank Assets (₹)', min_value=0, value=5000000)


    # --- Prediction Logic ---
    if st.button('Predict Loan Status'):
    
        # Calculate Total Assets
        total_assets_value = (
            residential_assets_value + 
            commercial_assets_value +
            luxury_assets_value +
            bank_asset_value
        )
        
        # Encode categorical inputs
        education_encoded = le_education.transform([education])[0]
        self_employed_encoded = le_self_employed.transform([self_employed])[0]
        
        # List of features in the EXACT order the model was trained on
        feature_columns = [
            'no_of_dependents', 
            'education', 
            'self_employed', 
            'income_annum', 
            'loan_amount', 
            'loan_term', 
            'cibil_score', 
            'total_assets_value'
        ]
        
        # Create a single row of data corresponding to the user input
        input_data_list = [
            no_of_dependents, 
            education_encoded, 
            self_employed_encoded, 
            income_annum, 
            loan_amount, 
            loan_term, 
            cibil_score, 
            total_assets_value
        ]
        
        # Create input DataFrame using the list and the explicit column names
        input_data = pd.DataFrame([input_data_list], columns=feature_columns)
        
        # 2. Predict Loan Status
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1] # Probability of Approval (class 1)

        st.subheader('Prediction Result')
        st.write(f'Probability of Approval: **{prediction_proba:.2f}**')
        
        if prediction == 1:
            st.success('✅ Loan is predicted to be **APPROVED**!')
            st.balloons()
        else:
            st.error('❌ Loan is predicted to be **REJECTED**.')