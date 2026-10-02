
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Risk Analyzer",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_files():
    with open("churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, features = load_files()

st.title("📊 Customer Churn Risk Analyzer")
st.caption("AI-powered customer retention and risk insights")

uploaded_file = st.file_uploader(
    "Upload customer data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)

        st.subheader("Customer Data Preview")
        st.dataframe(raw_data.head(), use_container_width=True)

        data = raw_data.copy()

        if "Churn" in data.columns:
            data = data.drop(columns=["Churn"])

        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(
                data["TotalCharges"], errors="coerce"
            )

        data = data.dropna().copy()

        if data.empty:
            st.error("No valid customer rows found.")
            st.stop()

        # Match the training data format
        data = pd.get_dummies(data, drop_first=True)
        data = data.reindex(columns=features, fill_value=0)

        scaled_data = scaler.transform(data)
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)[:, 1] * 100

        results = pd.DataFrame({
            "Churn Prediction": [
                "YES" if p == 1 else "NO" for p in predictions
            ],
            "Churn Risk (%)": probabilities.round(2)
        })

        results["Risk Level"] = results["Churn Risk (%)"].apply(
            lambda x: "HIGH RISK" if x >= 70
            else "MEDIUM RISK" if x >= 40
            else "LOW RISK"
        )

        if "customerID" in raw_data.columns:
            results.insert(
                0, "Customer ID",
                raw_data.loc[data.index, "customerID"].values
            )

        # Summary cards
        total = len(results)
        high = (results["Risk Level"] == "HIGH RISK").sum()
        medium = (results["Risk Level"] == "MEDIUM RISK").sum()
        predicted_churn = (
            results["Churn Prediction"] == "YES"
        ).sum()

        st.subheader("Business Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers Analyzed", f"{total:,}")
        c2.metric("Predicted Churn", f"{predicted_churn:,}")
        c3.metric("High-Risk Customers", f"{high:,}")
        c4.metric("Medium-Risk Customers", f"{medium:,}")

        st.caption(
            f"Predicted churn rate: {predicted_churn / total * 100:.1f}%"
        )

        # Charts
        st.subheader("Risk Insights")
        left, right = st.columns(2)

        with left:
            st.write("**Customers by Risk Level**")
            risk_counts = (
                results["Risk Level"]
                .value_counts()
                .reindex(
                    ["HIGH RISK", "MEDIUM RISK", "LOW RISK"],
                    fill_value=0
                )
            )
            st.bar_chart(risk_counts)

        with right:
            st.write("**Churn Risk Distribution**")
            st.histogram if False else None
            histogram_data = pd.DataFrame({
                "Churn Risk (%)": results["Churn Risk (%)"]
            })
            st.area_chart(
                histogram_data["Churn Risk (%)"]
                .value_counts(bins=10)
                .sort_index()
            )

        # Results table
        st.subheader("Customer-Level Predictions")

        risk_filter = st.selectbox(
            "Filter customers",
            ["All", "HIGH RISK", "MEDIUM RISK", "LOW RISK"]
        )

        shown = results
        if risk_filter != "All":
            shown = results[results["Risk Level"] == risk_filter]

        st.dataframe(shown, use_container_width=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv,
            file_name="customer_churn_predictions.csv",
            mime="text/csv"
        )

        st.info(
            "Risk levels are model estimates, not guarantees. "
            "Review predictions before making business decisions."
        )

    except Exception as e:
        st.error(f"Could not analyze this CSV: {e}")
        st.write(
            "Please upload a CSV with columns matching the "
            "dataset used to train this model."
        )
else:
    st.info("Upload a customer CSV to start the analysis.")
