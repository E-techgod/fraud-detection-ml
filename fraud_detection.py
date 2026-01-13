import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")  

st.title("💳 Fraud Detection App")
st.markdown("Enter transaction details to predict if it is fraudulent.")

st.divider()

# --- User inputs
transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT"])
amount = st.number_input("Amount", min_value=0.0, max_value=100000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)

country_code = st.selectbox("Transaction Origin Country", ["US", "GB", "FR", "DE", "IN", "CN", "NG", "RU", "BR", "ZA"])
high_risk_countries = ["NG", "RU", "CN"]
is_high_risk = 1 if country_code in high_risk_countries else 0


threshold = st.slider("Set Fraud Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if st.button("Predict Single Transaction"):
    # --- Construct DataFrame
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isHighRiskCountry": is_high_risk
    }])
    
    # --- Add engineered features
    input_data["balanceDiffOrig"] = input_data["oldbalanceOrg"] - input_data["newbalanceOrig"]
    input_data["balanceDiffDest"] = input_data["newbalanceDest"] - input_data["oldbalanceDest"]

    # --- Predict
    fraud_prob = model.predict_proba(input_data)[0][1]
    is_fraud = fraud_prob >= threshold

    st.subheader(f"Probability of Fraud: {fraud_prob:.2%}")

    # --- Fraud logic
    if is_fraud:
        st.error(f"🚨 Fraudulent Transaction Detected (>{threshold:.0%})")
    else:
        st.success(f"✅ Legitimate Transaction (<{threshold:.0%})")

    # --- High-risk override warning
    if (
        transaction_type in ["TRANSFER", "CASH_OUT"]
        and oldbalanceOrg == 0
        and newbalanceOrig == 0
        and oldbalanceDest == 0
        and newbalanceDest >= amount
    ):
        st.warning("⚠️ High-risk pattern detected — likely fraudulent, even if prediction is low.")

    if is_high_risk:
        st.warning("🌍 High-risk country — increased likelihood of fraud.")

    # Plot gauge-style visualization
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh([0], [fraud_prob], color="red" if is_fraud else "green")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Fraud Probability")
    ax.set_title("🔎 Probability Gauge")
    st.pyplot(fig)

    # === SHAP EXPLANATION ===
    try:
        import shap
        import matplotlib.pyplot as plt

        X_transformed = model.named_steps["prep"].transform(input_data)
        feature_names = model.named_steps["prep"].get_feature_names_out()

        explainer = shap.TreeExplainer(model.named_steps["clf"])
        shap_vals = explainer.shap_values(X_transformed)

        explanation = shap.Explanation(
            values=shap_vals[0][0],
            base_values=[explainer.expected_value],
            data=X_transformed[0],
            feature_names=feature_names
        )

        st.subheader("🔍 SHAP Feature Contribution")
        fig, ax = plt.subplots()
        shap.plots.bar(explanation, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")

    # === CSV UPLOAD AND BATCH PREDICTION ===
st.divider()
st.subheader("📁 Batch Prediction via CSV")

uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])

if uploaded_file is not None:
    df_csv = pd.read_csv(uploaded_file)

    required_columns = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isHighRiskCountry"]
    if all(col in df_csv.columns for col in required_columns):
        
        if "country_code" in df_csv.columns:
            high_risk_countries = ["NG", "RU", "CN"]
            df_csv["isHighRiskCountry"] = df_csv["country_code"].isin(high_risk_countries).astype(int)
        else:
            st.error("Missing 'country_code' column in CSV.")

        # Add engineered features if missing
        df_csv["balanceDiffOrig"] = df_csv["oldbalanceOrg"] - df_csv["newbalanceOrig"]
        df_csv["balanceDiffDest"] = df_csv["newbalanceDest"] - df_csv["oldbalanceDest"]

        # Predict fraud probability
        probabilities = model.predict_proba(df_csv)[:, 1]
        df_csv["Fraud_Probability"] = probabilities

        # Classify based on threshold
        df_csv["Fraud_Prediction"] = (df_csv["Fraud_Probability"] >= threshold).astype(int)

        # Show the updated dataframe
        st.success("✅ Predictions complete.")
        df_csv.index= df_csv.index+1
        st.dataframe(df_csv)

        # 📊 Add dynamic chart
        st.subheader("📊 Live Fraud Prediction Count by Threshold")
        fraud_count = df_csv["Fraud_Prediction"].value_counts().sort_index()
        fraud_labels = fraud_count.rename(index={0: "Not Fraud", 1: "Fraud"})
        st.bar_chart(fraud_labels)

        # 📊 Breakdown of frauds by transaction type
        st.subheader("🔎 Fraud Count by Transaction Type")
        fraud_by_type = df_csv[df_csv["Fraud_Prediction"] == 1]["type"].value_counts()
        st.bar_chart(fraud_by_type)


        # 📥 Downloadable CSV
        csv_output = df_csv.to_csv(index=False).encode("utf-8")
        
        # === Overall Feature Importance (No SHAP) ===
        st.subheader("📊 Overall Feature Importance (Model-Wide)")

        importances = model.named_steps["clf"].feature_importances_
        features = model.named_steps["prep"].get_feature_names_out()

        df_feat = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.bar_chart(df_feat.set_index("Feature"))

