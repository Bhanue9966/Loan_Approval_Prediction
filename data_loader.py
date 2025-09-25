# data_loader.py

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

def load_data(file_path):
    """Loads and performs initial cleaning (column renaming) on the dataset."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

    # Fix column names with leading spaces
    new_column_names = {
        ' no_of_dependents': 'no_of_dependents',
        ' education': 'education',
        ' self_employed': 'self_employed',
        ' income_annum': 'income_annum',
        ' loan_amount': 'loan_amount',
        ' loan_term': 'loan_term',
        ' cibil_score': 'cibil_score',
        ' residential_assets_value': 'residential_assets_value',
        ' commercial_assets_value': 'commercial_assets_value',
        ' luxury_assets_value': 'luxury_assets_value',
        ' bank_asset_value': 'bank_asset_value',
        ' loan_status': 'loan_status'
    }
    df.rename(columns=new_column_names, inplace=True)
    
    print("Data loaded and columns cleaned successfully.")
    return df