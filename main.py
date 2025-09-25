import data_loader
import preprocessor
import model_trainer


def main():
    FILE_PATH = 'loan_approval_dataset.csv'
    TARGET_COLUMN = 'loan_status'
    
    print("--- Starting Loan Approval Prediction Pipeline ---")
    
    # 1. Load Data
    df = data_loader.load_data(FILE_PATH)
    if df is None: return

    # 2. Preprocess Data - Capture the returned encoders
    df_processed, le_education, le_self_employed = preprocessor.preprocess_data(df)
    if df_processed is None: return

    # 3. Prepare for Modeling
    X = df_processed.drop(TARGET_COLUMN, axis=1)
    y = df_processed[TARGET_COLUMN]
    
    # 4. Train Model
    model, X_test, y_test = model_trainer.train_model(X, y)
    
    # 5. Evaluate Model
    model_trainer.evaluate_model(model, X_test, y_test)
    
    # 6. SAVE ASSETS (NEW STEP)
    model_trainer.save_assets(model, le_education, le_self_employed)
    
    print("\n--- Pipeline execution complete. Assets saved. ---")


if __name__ == '__main__':
    main()