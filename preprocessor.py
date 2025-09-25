# preprocessor.py - FINAL CORRECTED CODE (Manual Target Mapping)

from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """Encodes categorical features, engineers new features, and cleans the DataFrame."""
    if df is None:
        return None

    # Drop the loan_id column
    df = df.drop('loan_id', axis=1)

    # *** STEP 1: Strip leading/trailing spaces from ALL categorical values ***
    df['education'] = df['education'].str.strip()
    df['self_employed'] = df['self_employed'].str.strip()
    df['loan_status'] = df['loan_status'].str.strip()
    
    # Initialize LabelEncoders for feature columns only
    le_education = LabelEncoder() 
    le_self_employed = LabelEncoder() 

    # Encode feature columns 
    df['education'] = le_education.fit_transform(df['education'])
    df['self_employed'] = le_self_employed.fit_transform(df['self_employed'])
    
    # *** STEP 2: MANUAL MAPPING FOR TARGET VARIABLE ***
    # This guarantees Approved = 1 and Rejected = 0
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

    # Feature Engineering: Create 'total_assets_value'
    df['total_assets_value'] = (
        df['residential_assets_value'] + 
        df['commercial_assets_value'] +
        df['luxury_assets_value'] +
        df['bank_asset_value']
    )

    # Drop individual asset columns
    df = df.drop([
        'residential_assets_value', 'commercial_assets_value', 
        'luxury_assets_value', 'bank_asset_value'
    ], axis=1)
    
    print("Data preprocessing and feature engineering complete (Approved=1, Rejected=0).")
    
    return df, le_education, le_self_employed