
import streamlit as st
import pandas as pd
import pickle

# Saved files load karna
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("features.pkl", "rb") as file:
    features = pickle.load(file)


st.title("Customer Churn Risk Analyzer")
st.write("Upload customer data to identify customers who may churn.")

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # TotalCharges ko numeric banana
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

    data = data.dropna()

    # Churn column agar uploaded file mein ho
    if "Churn" in data.columns:
        data = data.drop("Churn", axis=1)

    # Categorical columns ko numbers mein convert karna
    data = pd.get_dummies(data, drop_first=True)

    # Model ke features ke according columns set karna
    data = data.reindex(columns=features, fill_value=0)

    # Scaling
    data_scaled = scaler.transform(data)

    # Prediction
    predictions = model.predict(data_scaled)
    probabilities = model.predict_proba(data_scaled)[:, 1] * 100

    # Results
    results = pd.DataFrame({
        "Churn Prediction": [
            "YES" if p else "NO" for p in predictions
        ],
        "Churn Risk (%)": probabilities.round(2)
    })

    # Risk level
    results["Risk Level"] = results["Churn Risk (%)"].apply(
        lambda x:
        "HIGH RISK" if x >= 70
        else "MEDIUM RISK" if x >= 40
        else "LOW RISK"
    )

    st.subheader("Churn Analysis")
    st.dataframe(results)

    st.metric(
        "High Risk Customers",
        (results["Risk Level"] == "HIGH RISK").sum()
    )
