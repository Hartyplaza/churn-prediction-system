from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ValidationReport:
    is_valid: bool
    missing_columns: list[str]
    duplicate_rows: int
    duplicate_customer_ids: int
    missing_target: int
    null_summary: dict[str, int]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "missing_columns": self.missing_columns,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_customer_ids": self.duplicate_customer_ids,
            "missing_target": self.missing_target,
            "null_summary": self.null_summary,
            "warnings": self.warnings,
        }


REQUIRED_FEATURE_COLUMNS = [
    "customer_id",
    "Name",
    "age",
    "gender",
    "security_no",
    "region_category",
    "membership_category",
    "joining_date",
    "joined_through_referral",
    "referral_id",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "last_visit_time",
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
]


def validate_training_frame(df: pd.DataFrame, target_column: str) -> ValidationReport:
    required_columns = REQUIRED_FEATURE_COLUMNS + [target_column]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    duplicate_rows = int(df.duplicated().sum())
    duplicate_customer_ids = int(df["customer_id"].duplicated().sum()) if "customer_id" in df.columns else 0
    missing_target = int(df[target_column].isna().sum()) if target_column in df.columns else len(df)
    null_summary = df[required_columns].isna().sum().astype(int).to_dict() if not missing_columns else {}

    warnings: list[str] = []
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} duplicate rows.")
    if duplicate_customer_ids:
        warnings.append(f"Found {duplicate_customer_ids} duplicate customer IDs.")
    if missing_target:
        warnings.append(f"Found {missing_target} records without a target value.")
    if "churn_risk_score" in df.columns and df["churn_risk_score"].nunique(dropna=True) <= 12:
        warnings.append("Target cardinality suggests a classification problem rather than regression.")

    is_valid = not missing_columns and missing_target == 0
    return ValidationReport(
        is_valid=is_valid,
        missing_columns=missing_columns,
        duplicate_rows=duplicate_rows,
        duplicate_customer_ids=duplicate_customer_ids,
        missing_target=missing_target,
        null_summary=null_summary,
        warnings=warnings,
    )


def validate_inference_frame(df: pd.DataFrame) -> ValidationReport:
    missing_columns = sorted(set(REQUIRED_FEATURE_COLUMNS) - set(df.columns))
    duplicate_rows = int(df.duplicated().sum())
    duplicate_customer_ids = int(df["customer_id"].duplicated().sum()) if "customer_id" in df.columns else 0
    null_summary = df.reindex(columns=REQUIRED_FEATURE_COLUMNS).isna().sum().astype(int).to_dict()
    warnings: list[str] = []
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} duplicate rows in inference payload.")

    return ValidationReport(
        is_valid=not missing_columns,
        missing_columns=missing_columns,
        duplicate_rows=duplicate_rows,
        duplicate_customer_ids=duplicate_customer_ids,
        missing_target=0,
        null_summary=null_summary,
        warnings=warnings,
    )
