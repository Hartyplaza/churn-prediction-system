from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from app.core.config import get_settings


TIME_SEGMENT_BINS = [-1, 5, 11, 17, 21, 24]
TIME_SEGMENT_LABELS = ["Late Night", "Morning", "Afternoon", "Evening", "Night"]
TENURE_BINS = [-1, 90, 180, 365, 730, 10_000]
TENURE_LABELS = ["<3 Months", "3-6 Months", "6-12 Months", "1-2 Years", "2+ Years"]


@dataclass
class ChurnFeatureBuilder:
    reference_date: pd.Timestamp | None = None
    learned_thresholds: dict[str, float] = field(default_factory=dict)
    feature_columns_: list[str] = field(default_factory=list)
    numerical_columns_: list[str] = field(default_factory=list)
    categorical_columns_: list[str] = field(default_factory=list)
    binary_columns_: list[str] = field(default_factory=list)
    raw_feature_columns_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "ChurnFeatureBuilder":
        prepared = self._prepare_frame(df, is_training=True)
        self.feature_columns_ = list(prepared.columns)
        self.raw_feature_columns_ = list(df.columns)
        self.binary_columns_ = [
            "has_referral",
            "points_missing_flag",
            "complaint_flag",
            "unresolved_complaint_flag",
            "dissatisfied_feedback_flag",
            "low_activity_flag",
            "high_value_customer_flag",
        ]
        self.numerical_columns_ = [
            column
            for column in prepared.columns
            if pd.api.types.is_numeric_dtype(prepared[column]) and column not in self.binary_columns_
        ]
        self.categorical_columns_ = [
            column for column in prepared.columns if prepared[column].dtype == "object"
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare_frame(df, is_training=False)
        if self.feature_columns_:
            missing = [column for column in self.feature_columns_ if column not in prepared.columns]
            for column in missing:
                prepared[column] = np.nan
            prepared = prepared[self.feature_columns_]
        return prepared

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def _prepare_frame(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        settings = get_settings()
        frame = df.copy().astype(object)
        placeholder_values = {"?", "xxxxxxxx", "Error", "error", "Unknown"}
        for column in frame.columns:
            frame[column] = frame[column].apply(
                lambda value: np.nan if isinstance(value, str) and value.strip() in placeholder_values else value
            )

        leakage_columns = [column for column in settings.leakage_columns if column in frame.columns]
        frame = frame.drop(columns=leakage_columns, errors="ignore")

        frame["joined_through_referral"] = frame["joined_through_referral"].fillna("Unknown")
        frame["preferred_offer_types"] = frame["preferred_offer_types"].fillna("Missing")
        frame["medium_of_operation"] = frame["medium_of_operation"].fillna("Missing")
        frame["region_category"] = frame["region_category"].fillna("Missing")

        frame["avg_frequency_login_days"] = pd.to_numeric(
            frame["avg_frequency_login_days"], errors="coerce"
        )
        frame["days_since_last_login"] = pd.to_numeric(frame["days_since_last_login"], errors="coerce")
        frame["days_since_last_login"] = frame["days_since_last_login"].where(
            frame["days_since_last_login"] >= 0, np.nan
        )
        frame["avg_time_spent"] = pd.to_numeric(frame["avg_time_spent"], errors="coerce")
        frame["avg_time_spent"] = frame["avg_time_spent"].where(frame["avg_time_spent"] >= 0, np.nan)
        frame["avg_transaction_value"] = pd.to_numeric(frame["avg_transaction_value"], errors="coerce")
        frame["points_in_wallet"] = pd.to_numeric(frame["points_in_wallet"], errors="coerce")
        frame["points_in_wallet"] = frame["points_in_wallet"].where(frame["points_in_wallet"] >= 0, np.nan)
        frame["age"] = pd.to_numeric(frame["age"], errors="coerce")

        frame["joining_date"] = self._parse_datetime(frame["joining_date"])
        frame["last_visit_time"] = self._parse_time(frame["last_visit_time"])

        if is_training and self.reference_date is None:
            max_joining_date = frame["joining_date"].dropna().max()
            self.reference_date = (
                (max_joining_date + timedelta(days=1)).normalize()
                if pd.notna(max_joining_date)
                else pd.Timestamp.today().normalize()
            )

        if self.reference_date is None:
            self.reference_date = pd.Timestamp.today().normalize()

        frame["membership_days"] = (
            (self.reference_date - frame["joining_date"]).dt.days.clip(lower=0)
            if is_datetime64_any_dtype(frame["joining_date"])
            else np.nan
        )
        frame["joining_month"] = frame["joining_date"].dt.month
        frame["joining_year"] = frame["joining_date"].dt.year
        frame["joining_weekday"] = frame["joining_date"].dt.weekday
        frame["customer_tenure_bucket"] = pd.cut(
            frame["membership_days"], bins=TENURE_BINS, labels=TENURE_LABELS
        ).astype("object")

        frame["visit_hour"] = frame["last_visit_time"].dt.hour
        frame["visit_time_segment"] = pd.cut(
            frame["visit_hour"].fillna(-1),
            bins=TIME_SEGMENT_BINS,
            labels=TIME_SEGMENT_LABELS,
        ).astype("object")

        frame["has_referral"] = (
            (frame["joined_through_referral"].fillna("Unknown").str.lower() == "yes")
        ).astype(int)
        frame["points_missing_flag"] = frame["points_in_wallet"].isna().astype(int)
        frame["complaint_flag"] = (frame["past_complaint"].fillna("No").str.lower() == "yes").astype(int)
        frame["unresolved_complaint_flag"] = (
            frame["complaint_status"].fillna("Not Applicable").isin(["Unsolved", "No Information Available"])
        ).astype(int)
        frame["dissatisfied_feedback_flag"] = frame["feedback"].fillna("").isin(
            settings.negative_feedback_values
        ).astype(int)

        if is_training:
            self.learned_thresholds = {
                "activity_login_gap": float(frame["days_since_last_login"].quantile(0.75)),
                "avg_time_spent_low": float(frame["avg_time_spent"].quantile(0.25)),
                "avg_transaction_value_low": float(frame["avg_transaction_value"].quantile(0.25)),
                "high_value_spend": float(frame["avg_transaction_value"].quantile(0.75)),
                "wallet_low": float(frame["points_in_wallet"].dropna().quantile(0.25)),
            }

        frame["low_activity_flag"] = (
            frame["days_since_last_login"] >= self.learned_thresholds.get("activity_login_gap", 14.0)
        ).astype(int)
        frame["high_value_customer_flag"] = (
            frame["avg_transaction_value"] >= self.learned_thresholds.get("high_value_spend", 40_000.0)
        ).astype(int)

        frame["value_per_login"] = frame["avg_transaction_value"] / frame["avg_frequency_login_days"].clip(lower=1)
        frame["points_per_transaction"] = frame["points_in_wallet"] / frame["avg_transaction_value"].clip(lower=1)
        frame["engagement_intensity"] = frame["avg_time_spent"] / (frame["days_since_last_login"] + 1)
        frame["wallet_to_spend_ratio"] = frame["points_in_wallet"] / frame["avg_transaction_value"].clip(lower=1)

        frame["engagement_segment"] = np.select(
            [
                frame["avg_time_spent"].fillna(0) < self.learned_thresholds.get("avg_time_spent_low", 60.0),
                frame["days_since_last_login"].fillna(99) >= self.learned_thresholds.get("activity_login_gap", 14.0),
                frame["avg_frequency_login_days"].fillna(30) <= 7,
            ],
            ["Low Engagement", "Dormant", "Habitual"],
            default="Moderate Engagement",
        )
        frame["spend_segment"] = np.select(
            [
                frame["avg_transaction_value"].fillna(0)
                < self.learned_thresholds.get("avg_transaction_value_low", 15_000.0),
                frame["avg_transaction_value"].fillna(0)
                >= self.learned_thresholds.get("high_value_spend", 40_000.0),
            ],
            ["Low Spend", "High Spend"],
            default="Mid Spend",
        )

        frame = frame.drop(columns=["joining_date", "last_visit_time"], errors="ignore")

        ordered_columns = [
            "age",
            "gender",
            "region_category",
            "membership_category",
            "joined_through_referral",
            "preferred_offer_types",
            "medium_of_operation",
            "internet_option",
            "days_since_last_login",
            "avg_time_spent",
            "avg_transaction_value",
            "avg_frequency_login_days",
            "points_in_wallet",
            "used_special_discount",
            "offer_application_preference",
            "past_complaint",
            "complaint_status",
            "feedback",
            "membership_days",
            "joining_month",
            "joining_year",
            "joining_weekday",
            "customer_tenure_bucket",
            "visit_hour",
            "visit_time_segment",
            "has_referral",
            "points_missing_flag",
            "complaint_flag",
            "unresolved_complaint_flag",
            "dissatisfied_feedback_flag",
            "low_activity_flag",
            "high_value_customer_flag",
            "value_per_login",
            "points_per_transaction",
            "engagement_intensity",
            "wallet_to_spend_ratio",
            "engagement_segment",
            "spend_segment",
        ]
        return frame[ordered_columns]

    @staticmethod
    def _parse_datetime(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce")

    @staticmethod
    def _parse_time(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, format="%H:%M:%S", errors="coerce")

    def get_feature_summary(self) -> dict[str, Any]:
        return {
            "reference_date": None if self.reference_date is None else str(self.reference_date.date()),
            "feature_columns": self.feature_columns_,
            "numerical_columns": self.numerical_columns_,
            "categorical_columns": self.categorical_columns_,
            "binary_columns": self.binary_columns_,
            "thresholds": self.learned_thresholds,
        }
