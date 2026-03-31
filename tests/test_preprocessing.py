from __future__ import annotations

import pandas as pd

from src.data.load_data import load_train_data
from src.features.build_features import ChurnFeatureBuilder


def test_feature_builder_removes_leakage_and_creates_engineered_columns() -> None:
    train_df = load_train_data().head(20)
    feature_builder = ChurnFeatureBuilder()
    features = feature_builder.fit_transform(train_df.drop(columns=["churn_risk_score"]))

    assert "customer_id" not in features.columns
    assert "Name" not in features.columns
    assert "security_no" not in features.columns
    assert "referral_id" not in features.columns
    assert "membership_days" in features.columns
    assert "visit_time_segment" in features.columns
    assert "low_activity_flag" in features.columns
    assert len(features) == len(train_df)


def test_feature_builder_handles_anomalies() -> None:
    sample = pd.DataFrame(
        [
            {
                "customer_id": "1",
                "Name": "Demo User",
                "age": 40,
                "gender": "F",
                "security_no": "ABC",
                "region_category": None,
                "membership_category": "Gold Membership",
                "joining_date": "2017-01-01",
                "joined_through_referral": "?",
                "referral_id": "xxxxxxxx",
                "preferred_offer_types": None,
                "medium_of_operation": "?",
                "internet_option": "Wi-Fi",
                "last_visit_time": "09:30:00",
                "days_since_last_login": -999,
                "avg_time_spent": -100.0,
                "avg_transaction_value": 5000.0,
                "avg_frequency_login_days": "Error",
                "points_in_wallet": -50.0,
                "used_special_discount": "Yes",
                "offer_application_preference": "No",
                "past_complaint": "Yes",
                "complaint_status": "Unsolved",
                "feedback": "Poor Website",
            }
        ]
    )
    features = ChurnFeatureBuilder().fit_transform(sample)
    assert pd.isna(features.loc[0, "days_since_last_login"])
    assert pd.isna(features.loc[0, "avg_time_spent"])
    assert pd.isna(features.loc[0, "points_in_wallet"])
    assert features.loc[0, "points_missing_flag"] == 1
