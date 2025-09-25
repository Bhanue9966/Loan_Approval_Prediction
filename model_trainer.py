from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
from collections import Counter # For confirming balance

def train_model(X, y):

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model training complete.")
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)

    print("\n--- Model Performance Metrics ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred))


import pickle
import joblib 

def save_assets(model, le_education, le_self_employed):
    """Saves the trained model and LabelEncoders to disk."""
    
    # Save the trained Random Forest model
    joblib.dump(model, 'random_forest_model.joblib')

    # Save the LabelEncoders
    joblib.dump(le_education, 'le_education.joblib')
    joblib.dump(le_self_employed, 'le_self_employed.joblib')
    
    print("\nModel and encoders successfully saved to disk.")