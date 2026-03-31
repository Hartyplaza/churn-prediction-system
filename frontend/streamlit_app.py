from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.core.config import PROJECT_ROOT, get_settings
from app.services.predictor import PredictorService


st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor() -> PredictorService:
    return PredictorService()


@st.cache_data
def load_training_data() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_csv(settings.raw_data_dir / "train.csv")


@st.cache_data
def load_metrics() -> dict:
    settings = get_settings()
    with settings.metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f1ea;
            --card: rgba(255,255,255,0.88);
            --ink: #102a43;
            --accent: #d97706;
            --accent-soft: #ffe8b5;
            --muted: #4f5d75;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(217,119,6,0.18), transparent 25%),
                linear-gradient(180deg, #f7f3eb 0%, #f3efe7 100%);
            color: var(--ink);
        }
        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(16,42,67,0.96), rgba(40,83,107,0.92));
            color: white;
            box-shadow: 0 20px 45px rgba(16,42,67,0.18);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: var(--card);
            border: 1px solid rgba(16,42,67,0.08);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            box-shadow: 0 8px 24px rgba(16,42,67,0.08);
        }
        .recommendation-card {
            background: rgba(255,255,255,0.9);
            border-left: 6px solid var(--accent);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_overview_page(predictor: PredictorService, metrics_payload: dict) -> None:
    validation_metrics = predictor.model_info()["validation_metrics"]
    st.markdown(
        """
        <div class="hero">
            <h1>Production-Ready Churn Prediction System</h1>
            <p>This app turns raw customer behavior into churn risk scores, confidence levels, explainability, and practical retention actions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Best Model", predictor.model_info()["best_model_name"])
    metric_columns[1].metric("Weighted F1", f"{validation_metrics.get('f1_weighted', 0):.3f}")
    metric_columns[2].metric("Accuracy", f"{validation_metrics.get('accuracy', 0):.3f}")
    metric_columns[3].metric(
        "QWK",
        f"{validation_metrics.get('quadratic_weighted_kappa', validation_metrics.get('r2', 0)):.3f}",
    )

    st.subheader("Why this system matters")
    st.write(
        "The model learns from behavior, complaints, loyalty signals, engagement depth, and lifecycle stage to predict who is most likely to slip away. The same artifact powers training, API inference, and this UI."
    )

    st.subheader("Modeling strategy")
    task_detection = predictor.model_info()["task_detection"]
    st.write(task_detection["strategy_details"]["notes"])
    st.json(metrics_payload)


def render_prediction_page(predictor: PredictorService) -> None:
    st.subheader("Manual Customer Prediction")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        payload = {
            "customer_id": col1.text_input("Customer ID", value="demo-customer-001"),
            "Name": col1.text_input("Name", value="Ava Johnson"),
            "age": col1.number_input("Age", min_value=10, max_value=90, value=35),
            "gender": col1.selectbox("Gender", ["F", "M", "Unknown"], index=0),
            "security_no": col1.text_input("Security No", value="SEC-DEMO-001"),
            "region_category": col2.selectbox("Region", ["Town", "City", "Village", "Missing"], index=1),
            "membership_category": col2.selectbox(
                "Membership",
                [
                    "No Membership",
                    "Basic Membership",
                    "Silver Membership",
                    "Gold Membership",
                    "Premium Membership",
                    "Platinum Membership",
                ],
                index=4,
            ),
            "joining_date": col2.date_input("Joining Date", value=pd.Timestamp("2017-05-10")).isoformat(),
            "joined_through_referral": col2.selectbox("Joined via referral", ["Yes", "No", "Unknown"], index=1),
            "referral_id": col2.text_input("Referral ID", value="CID-REF-001"),
            "preferred_offer_types": col2.selectbox(
                "Offer Preference",
                ["Gift Vouchers/Coupons", "Credit/Debit Card Offers", "Without Offers", "Missing"],
                index=0,
            ),
            "medium_of_operation": col3.selectbox("Primary Device", ["Desktop", "Smartphone", "Both", "Missing"], index=1),
            "internet_option": col3.selectbox("Internet Type", ["Wi-Fi", "Mobile_Data", "Fiber_Optic"], index=0),
            "last_visit_time": col3.text_input("Last Visit Time", value="21:15:00"),
            "days_since_last_login": col3.number_input("Days Since Last Login", min_value=0.0, max_value=90.0, value=19.0),
            "avg_time_spent": col3.number_input("Average Time Spent", min_value=0.0, max_value=5000.0, value=48.0),
            "avg_transaction_value": st.number_input(
                "Average Transaction Value", min_value=0.0, max_value=100000.0, value=12500.0
            ),
            "avg_frequency_login_days": st.text_input("Avg Frequency Login Days", value="12"),
            "points_in_wallet": st.number_input("Points In Wallet", min_value=0.0, max_value=5000.0, value=240.0),
            "used_special_discount": st.selectbox("Used Special Discount", ["Yes", "No"], index=0),
            "offer_application_preference": st.selectbox("Offer Application Preference", ["Yes", "No"], index=1),
            "past_complaint": st.selectbox("Past Complaint", ["Yes", "No"], index=0),
            "complaint_status": st.selectbox(
                "Complaint Status",
                ["Unsolved", "Solved", "Solved in Follow-up", "Not Applicable", "No Information Available"],
                index=0,
            ),
            "feedback": st.selectbox(
                "Feedback",
                [
                    "Poor Customer Service",
                    "Poor Product Quality",
                    "Poor Website",
                    "Too many ads",
                    "Reasonable Price",
                    "Products always in Stock",
                    "Quality Customer Care",
                    "User Friendly Website",
                    "No reason specified",
                ],
                index=0,
            ),
        }
        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        result = predictor.predict_record(payload)
        score_color = "#b91c1c" if result["risk_band"] == "High" else "#b45309" if result["risk_band"] == "Medium" else "#15803d"
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Predicted Class: {result['predicted_class']} ({result['predicted_label']})</h3>
                <p><strong>Risk Score:</strong> <span style="color:{score_color};">{result['risk_score']:.3f}</span></p>
                <p><strong>Risk Band:</strong> {result['risk_band']}</p>
                <p><strong>Confidence:</strong> {result['confidence'] if result['confidence'] is not None else 'N/A'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Top Risk Drivers")
        for driver in result["top_risk_drivers"]:
            st.write(f"- {driver}")

        st.subheader("Retention Recommendations")
        for recommendation in result["recommendations"]:
            st.markdown(f"<div class='recommendation-card'>{recommendation}</div>", unsafe_allow_html=True)

        st.subheader("Probability Breakdown")
        probability_frame = pd.DataFrame(
            {
                "class": list(result["probability_breakdown"].keys()),
                "probability": list(result["probability_breakdown"].values()),
            }
        )
        st.plotly_chart(
            px.bar(
                probability_frame,
                x="class",
                y="probability",
                color="probability",
                color_continuous_scale="YlOrBr",
                title="Per-Class Probability Distribution",
            ),
            use_container_width=True,
        )


def render_batch_page(predictor: PredictorService) -> None:
    st.subheader("Batch Scoring")
    upload = st.file_uploader("Upload a CSV file with customer records", type=["csv"])
    if upload is not None:
        frame = pd.read_csv(upload)
        predictions = predictor.predict_batch(frame)
        prediction_frame = pd.DataFrame(predictions)
        st.dataframe(prediction_frame, use_container_width=True)
        st.download_button(
            label="Download Scored Results",
            data=prediction_frame.to_csv(index=False).encode("utf-8"),
            file_name="batch_churn_predictions.csv",
            mime="text/csv",
        )


def render_insights_page(predictor: PredictorService) -> None:
    st.subheader("Model Insights")
    train_df = load_training_data()
    model_info = predictor.model_info()
    importance_frame = pd.DataFrame(model_info["global_feature_importance"])

    st.plotly_chart(
        px.histogram(
            train_df,
            x="churn_risk_score",
            color="churn_risk_score",
            title="Observed Churn Risk Distribution",
            color_discrete_sequence=px.colors.sequential.Sunsetdark,
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        px.bar(
            importance_frame.head(12),
            x="importance",
            y="base_feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Tealgrn",
            title="Top Global Feature Drivers",
        ),
        use_container_width=True,
    )

    plots_dir = PROJECT_ROOT / "artifacts" / "plots"
    for image_name in ["confusion_matrix.png", "high_risk_calibration.png", "target_distribution.png"]:
        image_path = plots_dir / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name.replace("_", " ").replace(".png", "").title(), use_container_width=True)


def render_about_page() -> None:
    st.subheader("About This Showcase")
    st.write(
        """
        This project demonstrates what an end-to-end productionized data science workflow looks like:
        robust feature engineering, leakage prevention, model comparison, artifact versioning, inference services,
        explainability, and business-oriented retention actions. It is designed to be both portfolio-ready and
        stakeholder-demo friendly.
        """
    )

    st.code(
        """
        python -m src.pipelines.training_pipeline
        uvicorn app.api.main:app --reload
        streamlit run frontend/streamlit_app.py
        """.strip(),
        language="bash",
    )


def main() -> None:
    inject_styles()
    predictor = load_predictor()
    metrics_payload = load_metrics()

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Predict", "Batch Scoring", "Model Insights", "About"],
    )

    if page == "Overview":
        render_overview_page(predictor, metrics_payload)
    elif page == "Predict":
        render_prediction_page(predictor)
    elif page == "Batch Scoring":
        render_batch_page(predictor)
    elif page == "Model Insights":
        render_insights_page(predictor)
    else:
        render_about_page()


if __name__ == "__main__":
    main()
