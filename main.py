import data_loader
import preprocessor
import model_trainer


def main():
    FILE_PATH = 'loan_approval_dataset.csv'
    TARGET_COLUMN = 'loan_status'
    
    print("--- Starting Loan Approval Prediction Pipeline ---")
    
    # Load Data
    df = data_loader.load_data(FILE_PATH)
    if df is None: return

    # Preprocess Data - Capture the returned encoders
    df_processed, le_education, le_self_employed = preprocessor.preprocess_data(df)
    if df_processed is None: return

    # Prepare for Modeling
    X = df_processed.drop(TARGET_COLUMN, axis=1)
    y = df_processed[TARGET_COLUMN]
    
    # Train Model
    model, X_test, y_test = model_trainer.train_model(X, y)
    
    # Evaluate Model
    model_trainer.evaluate_model(model, X_test, y_test)
    
    # Save assets
    model_trainer.save_assets(model, le_education, le_self_employed)
    
    print("\n--- Pipeline execution complete. Assets saved. ---")


if __name__ == '__main__':
    main()