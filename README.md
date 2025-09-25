-----

# 🏦 Loan Approval Prediction Engine: A Modular Machine Learning Solution

## 🎯 Project Goal and Business Context

This project delivers a complete, professional pipeline for predicting the **Loan Approval Status** (`Approved` or `Rejected`). The primary business objective is **risk mitigation**: to accurately identify high-risk applicants (`Rejected`) while maintaining a high rate of correctly identifying good applicants (`Approved`).

The entire solution is built on **Modular Python Programming** for scalability and culminates in a live, interactive **Streamlit web application** (`app.py`) for real-time demonstration.

### Key Technologies Used:

  * **Model:** Random Forest Classifier (Chosen for its robustness and ability to handle non-linear relationships).
  * **Imbalance Handling:** **SMOTE** (Synthetic Minority Oversampling Technique).
  * **Architecture:** Modular Python (Structured into distinct `.py` files).
  * **Deployment:** Streamlit (For the user-friendly web interface).

-----

## 📊 Data Preparation and Feature Engineering

Data quality was paramount for training a reliable risk model. The preprocessing phase included critical cleaning and feature derivation steps.

### Key Preprocessing Decisions:

1.  **Categorical String Cleanup:** A persistent issue with the source data was the presence of extraneous whitespace (leading/trailing spaces) in categorical values. This was fixed using **`df.column.str.strip()`** in the preprocessing stage to ensure all categorical inputs (e.g., `'Graduate'` instead of `' Graduate'`) were correctly interpreted by the model and the Streamlit app.
2.  **Target Encoding (Manual):** The target variable (`loan_status`) was explicitly mapped to prevent inverted model interpretation:
      * **`Approved` $\\rightarrow$ 1** (The positive class)
      * **`Rejected` $\\rightarrow$ 0** (The negative class)
3.  **Feature Encoding:** Standard `LabelEncoder` was applied to the `education` and `self_employed` features.

### Feature Engineering: Total Assets Value

A custom, high-impact feature, **`total_assets_value`**, was engineered. This feature aggregates four individual asset columns into one holistic metric, which proved highly effective in simplifying the assessment of the applicant's financial viability for the model.

$$\text{Total Assets} = \text{Residential} + \text{Commercial} + \text{Luxury} + \text{Bank}$$

-----

## 📈 Machine Learning Strategy and Evaluation

### Addressing Class Imbalance with SMOTE

The original dataset exhibited a significant class imbalance (**Approved: 2656**, **Rejected: 1613**), which biases standard models towards the majority class.

  * **Technique:** **SMOTE (Synthetic Minority Oversampling Technique)** from the `imblearn` library was used.
  * **Action:** SMOTE was applied *before* the final train/test split to generate synthetic samples for the minority class (`Rejected`).
  * **Result:** The training data was perfectly balanced at a $50:50$ ratio ($\\approx 2656:2656$), ensuring the model learns the true risk boundaries equally well for both outcomes.

### Model Performance Focus

The evaluation prioritized metrics relevant to financial risk:

  * **Precision (Approved Class):** Focused on minimizing **False Positives** (approving a bad loan). High precision indicates the bank is efficiently filtering out high-risk applicants.
  * **Recall (Rejected Class):** Focused on maximizing **True Positives** (correctly identifying risky clients). High recall indicates the model is successfully catching almost all genuinely high-risk applications.

-----

## 🧱 Modular Architecture and Code Structure

The entire solution is built on a clean, modular foundation for enhanced maintainability, testing, and professional practice.

### File Responsibilities:

  * **`main.py`**: **Orchestrator** – Controls the entire pipeline flow (Load $\\rightarrow$ Preprocess $\\rightarrow$ Train $\\rightarrow$ Evaluate $\\rightarrow$ Save Assets).
  * **`data_loader.py`**: **Data Ingestion** – Handles file reading and initial column renaming.
  * **`preprocessor.py`**: **Transformation** – Manages data cleaning, categorical string stripping, manual target mapping, and feature engineering.
  * **`model_trainer.py`**: **ML Core** – Implements **SMOTE resampling**, train-test splitting, `RandomForestClassifier` training, and saves model assets using `joblib`.
  * **`app.py`**: **Deployment** – The Streamlit application that loads the final model and enables real-time, interactive predictions.
  * **`requirements.txt`**: Lists all project dependencies (`pandas`, `scikit-learn`, `imblearn`, `streamlit`, etc.).

-----

## 🌐 Running the Application (End-to-End)

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/Loan_Approval_Prediction_Modular.git
cd Loan_Approval_Prediction_Modular

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Execute the Training Pipeline

This command executes the modular scripts, performs resampling and training, and saves the final assets (`random_forest_model.joblib`, etc.) required by the web app.

```bash
python main.py
```

### Step 3: Launch the Web Application

Access the real-time prediction engine by running the Streamlit app.

```bash
streamlit run app.py
```