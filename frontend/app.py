import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def load_report(report_path: Path):
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def predict_local(model, invoice_amount, gst_rate, total_amount, vendor_frequency):
    features = np.array([[invoice_amount, gst_rate, total_amount, vendor_frequency]])
    prediction_idx = model.predict(features)[0]
    prediction = "Valid" if prediction_idx == 0 else "Suspicious"
    confidence = float(model.predict_proba(features).max())
    return prediction, confidence


def predict_api(endpoint, payload):
    response = requests.post(endpoint, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def render_confusion_matrix(report):
    cm = np.array(report["confusion_matrix"])
    labels = report.get("labels", ["Valid", "Suspicious"])

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    return fig


def main():
    st.set_page_config(
        page_title="GST Invoice Validator",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Session state initialization
    if "show_performance" not in st.session_state:
        st.session_state.show_performance = False

    # Custom CSS
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 20px;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #A23B72;
            text-align: center;
            margin-bottom: 30px;
        }
        .prediction-card {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2E86AB;
            margin: 20px 0;
        }
        .valid {
            color: #28a745;
            font-weight: bold;
        }
        .suspicious {
            color: #dc3545;
            font-weight: bold;
        }
        .metric-box {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            margin: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("🔍 About")

        st.write(
            """
            This application uses a Machine Learning model to classify
            GST invoices as Valid or Suspicious based on invoice details.
            """
        )

        if st.button("📊 View Model Performance"):
            st.session_state.show_performance = (
                not st.session_state.show_performance
            )

        st.header("📝 Sample Data")

        if st.button("Load Valid Invoice Example"):
            st.session_state.invoice_amount = 1200.0
            st.session_state.gst_rate = 18.0
            st.session_state.total_amount = 1416.0
            st.session_state.vendor_frequency = 12

        if st.button("Load Suspicious Invoice Example"):
            st.session_state.invoice_amount = 5000.0
            st.session_state.gst_rate = 0.0
            st.session_state.total_amount = 5200.0
            st.session_state.vendor_frequency = 2

    st.markdown(
        '<h1 class="main-header">📋 GST Invoice Validator</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sub-header">AI-Powered Invoice Classification for GST Compliance</p>',
        unsafe_allow_html=True,
    )

    repo_root = Path(__file__).resolve().parents[1]

    model_path = repo_root / "models" / "gst_invoice_validator.joblib"
    report_path = repo_root / "models" / "model_report.json"

    model = None
    report = load_report(report_path)

    if model_path.exists():
        model = load_model(model_path)
    else:
        st.warning("Model file not found. Local prediction unavailable.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Invoice Details")

        with st.form(key="invoice_form"):

            invoice_amount = st.number_input(
                "Invoice Amount (₹)",
                min_value=0.0,
                value=st.session_state.get("invoice_amount", 1000.0),
                step=50.0,
            )

            gst_rate = st.selectbox(
                "GST Rate (%)",
                [0.0, 5.0, 12.0, 18.0, 28.0],
                index=[0.0, 5.0, 12.0, 18.0, 28.0].index(
                    st.session_state.get("gst_rate", 18.0)
                ),
            )

            total_amount = st.number_input(
                "Total Amount (₹)",
                min_value=0.0,
                value=st.session_state.get("total_amount", 1180.0),
                step=10.0,
            )

            vendor_frequency = st.slider(
                "Vendor Frequency",
                min_value=1,
                max_value=50,
                value=st.session_state.get("vendor_frequency", 10),
            )

            use_backend_api = st.checkbox(
                "Use Backend API",
                value=True,
            )

            api_endpoint = st.text_input(
                "API Endpoint",
                value="https://gstinvoicevalidator1-pww5.onrender.com/predict",
            )

            submitted = st.form_submit_button(
                "🔍 Validate Invoice",
                use_container_width=True,
            )

    with col2:
        st.subheader("🎯 Prediction Result")

        if submitted:

            payload = {
                "invoice_amount": float(invoice_amount),
                "gst_rate": float(gst_rate),
                "total_amount": float(total_amount),
                "vendor_frequency": int(vendor_frequency),
            }

            with st.spinner("Analyzing invoice..."):

                try:

                    if use_backend_api:

                        result = predict_api(
                            api_endpoint,
                            payload,
                        )

                        prediction = result["prediction"]
                        confidence = result["confidence"]

                    else:

                        if model is None:
                            raise ValueError(
                                "Model file not found. Enable API mode."
                            )

                        prediction, confidence = predict_local(
                            model,
                            invoice_amount,
                            gst_rate,
                            total_amount,
                            vendor_frequency,
                        )

                    result_class = (
                        "valid"
                        if prediction == "Valid"
                        else "suspicious"
                    )

                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <h3>
                                Prediction:
                                <span class="{result_class}">
                                    {prediction}
                                </span>
                            </h3>

                            <p>
                                <strong>Confidence:</strong>
                                {confidence:.1%}
                            </p>

                            <p>
                                <strong>Invoice Amount:</strong>
                                ₹{invoice_amount:,.0f}
                            </p>

                            <p>
                                <strong>GST Rate:</strong>
                                {gst_rate}%
                            </p>

                            <p>
                                <strong>Total Amount:</strong>
                                ₹{total_amount:,.0f}
                            </p>

                            <p>
                                <strong>Vendor Frequency:</strong>
                                {vendor_frequency}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as exc:
                    st.error(f"❌ Error: {str(exc)}")

    if st.session_state.show_performance and report:

        st.markdown("---")
        st.subheader("📊 Model Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Accuracy",
                f"{report['accuracy']:.1%}",
            )

        with col2:
            precision = report["classification_report"]["weighted avg"]["precision"]
            st.metric(
                "Precision",
                f"{precision:.1%}",
            )

        with col3:
            recall = report["classification_report"]["weighted avg"]["recall"]
            st.metric(
                "Recall",
                f"{recall:.1%}",
            )

        st.pyplot(render_confusion_matrix(report))

        with st.expander("📋 Detailed Classification Report"):
            st.dataframe(
                pd.DataFrame(
                    report["classification_report"]
                ).transpose()
            )


if __name__ == "__main__":
    main()
