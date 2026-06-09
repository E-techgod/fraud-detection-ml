# Fraud Detection App

Interactive Streamlit application for detecting potentially fraudulent financial transactions with a trained machine learning pipeline.

This project combines:

- Single-transaction prediction from a form
- Batch prediction from a CSV upload
- Fraud probability scoring with a configurable threshold
- Feature importance and SHAP-based explainability

## Overview

The application uses a scikit-learn pipeline to classify transactions as fraud or not fraud based on transaction type, account balances, engineered balance-difference features, and a high-risk-country flag.

The repository also includes a Jupyter notebook used for exploratory analysis, feature engineering, model comparison, and training experiments.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas joblib matplotlib shap scikit-learn
streamlit run fraud_detection.py
```

Before running the app, make sure `fraud_detection_model.pkl` exists in the project root.

## Features

- Predict fraud probability for one transaction at a time
- Upload a CSV for batch scoring
- Adjust the classification threshold in the UI
- Display warnings for high-risk transaction patterns
- Show model-wide feature importance
- Attempt SHAP explanations for single predictions

## Model Summary

The current app expects a saved model file named `fraud_detection_model.pkl` in the project root.

### Architecture

| Component | Value |
| --- | --- |
| Model file | `fraud_detection_model.pkl` |
| Pipeline type | `Pipeline` |
| Preprocessing | `ColumnTransformer` |
| Classifier | `RandomForestClassifier` |

### Classifier Settings

| Parameter | Value |
| --- | --- |
| `n_estimators` | `100` |
| `max_depth` | `15` |
| `class_weight` | `'balanced'` |
| `random_state` | `42` |
| `n_jobs` | `-1` |

### Input Features

| Feature | Type | Notes |
| --- | --- | --- |
| `type` | Categorical | Transaction type |
| `amount` | Numeric | Transaction amount |
| `oldbalanceOrg` | Numeric | Sender balance before transaction |
| `newbalanceOrig` | Numeric | Sender balance after transaction |
| `oldbalanceDest` | Numeric | Receiver balance before transaction |
| `newbalanceDest` | Numeric | Receiver balance after transaction |
| `balanceDiffOrig` | Engineered numeric | `oldbalanceOrg - newbalanceOrig` |
| `balanceDiffDest` | Engineered numeric | `newbalanceDest - oldbalanceDest` |
| `isHighRiskCountry` | Numeric / binary | Derived from country risk list |

## Training Notes

The project was developed from exploratory analysis on a fraud dataset with roughly:

- `6,362,620` rows
- `8,213` fraud cases
- About `0.13%` fraud prevalence

Model history:

1. Logistic regression baseline on the original imbalanced dataset
2. Logistic regression with engineered balance-difference features
3. Random forest trained on a rebalanced dataset

The notebook is intended for:

- Exploratory data analysis
- Feature engineering
- Model training experiments
- Evaluation and comparison across model versions

## Metrics

The notebook contains results for multiple model iterations.

### Model Comparison

| Model | Dataset Setup | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression Baseline | Original imbalanced data, 70/30 split, `class_weight='balanced'` | `94.63%` | `0.02` | `0.95` | `0.04` | Not reported |
| Logistic Regression With Engineered Features | Original imbalanced data with `balanceDiffOrig` and `balanceDiffDest` | `95%` | `0.02` | `0.94` | `0.04` | `0.990` |
| Final Random Forest | Rebalanced data using all fraud rows + `100,000` sampled non-fraud rows | Not reported directly | `0.96` | `0.98` | `0.97` | `0.999` |

### Final Random Forest Class Metrics

This is the model loaded by `fraud_detection.py`.

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| Not Fraud | `1.00` | `1.00` | `1.00` | `25,001` |
| Fraud | `0.96` | `0.98` | `0.97` | `2,053` |

### Baseline Confusion Matrix

| Metric | Value |
| --- | --- |
| True Negatives | `1,804,022` |
| False Positives | `102,300` |
| False Negatives | `132` |
| True Positives | `2,332` |

Note: the final random forest metrics were measured on a balanced dataset, so real-world precision may be lower on naturally imbalanced production data.

## Repository Structure

Core repository files:

```text
.
├── fraud_detection.py
├── analysis_method.ipynb
├── .gitignore
└── README.md
```

Runtime dependency:

- `fraud_detection_model.pkl` must exist in the project root when you run the app

If the model file is not committed to the repo, place it manually in the project root before starting the app.

## Requirements

Install these Python packages if you are setting the environment up from scratch:

```bash
pip install streamlit pandas joblib matplotlib shap scikit-learn
```

Recommended Python version:

- Python 3.11 or 3.12

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd fraud_detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install streamlit pandas joblib matplotlib shap scikit-learn
```

### 5. Add the trained model

Make sure `fraud_detection_model.pkl` exists in the project root.

## Run The App

Run the application with:

```bash
streamlit run fraud_detection.py
```

If `streamlit` is not available on your PATH:

```bash
python -m streamlit run fraud_detection.py
```

By default, Streamlit will open or print a local address similar to:

```text
http://localhost:8501
```

## How To Use

### Single Prediction

1. Launch the app.
2. Open the local Streamlit URL in your browser.
3. Choose a transaction type.
4. Enter the sender and receiver balances.
5. Choose the transaction origin country.
6. Set the fraud threshold.
7. Click `Predict Single Transaction`.

The app returns:

- Fraud probability
- Fraud / legitimate prediction
- High-risk rule warnings
- A probability bar visualization
- A SHAP explanation if available

### Batch Prediction

1. Scroll to the batch prediction section.
2. Upload a CSV file.
3. Review the prediction table and summary charts.

## CSV Format

For batch prediction, the app expects transaction data with these columns:

- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `isHighRiskCountry`

The current code also checks for `country_code` in some paths, so the safest CSV format is:

- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `country_code`
- `isHighRiskCountry`

Example:

```csv
type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,country_code,isHighRiskCountry
TRANSFER,950,0,0,0,950,US,0
CASH_OUT,980,980,0,0,0,FR,0
TRANSFER,200,1000,800,500,700,CN,1
```

High-risk countries currently used by the app:

- `NG`
- `RU`
- `CN`

## App Logic

In addition to the model prediction, the UI applies a warning rule for this pattern:

- `TRANSFER` or `CASH_OUT`
- `oldbalanceOrg == 0`
- `newbalanceOrig == 0`
- `oldbalanceDest == 0`
- `newbalanceDest >= amount`

This rule is surfaced even if the model score is below the selected threshold.

## Known Limitations

- The app hardcodes the model filename as `fraud_detection_model.pkl`.
- Batch CSV validation is inconsistent because the code expects both `isHighRiskCountry` and sometimes `country_code`.
- The batch section prepares a downloadable CSV in memory but does not expose a download button.
- SHAP visualizations may fail depending on the local environment even when prediction works.
- There is no pinned `requirements.txt` in the current repo layout.

## Next Improvements

- Add `requirements.txt`
- Add a downloadable results button
- Clean up batch CSV validation
- Allow selecting between model versions
- Add automated tests for app startup and prediction flow

## Quick Start

If the environment is already set up and the model file is present:

```bash
streamlit run fraud_detection.py
```
