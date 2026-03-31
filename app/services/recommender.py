from __future__ import annotations

from typing import Any

import pandas as pd


class RecommendationEngine:
    def __init__(self, threshold_config: dict[str, float] | None = None):
        self.threshold_config = threshold_config or {}

    def recommend(
        self,
        raw_profile: dict[str, Any],
        engineered_profile: dict[str, Any],
        risk_band: str,
        top_driver_messages: list[str] | None = None,
    ) -> list[str]:
        recommendations: list[str] = []

        days_since_login = engineered_profile.get("days_since_last_login")
        avg_time_spent = engineered_profile.get("avg_time_spent")
        avg_transaction_value = engineered_profile.get("avg_transaction_value")
        points_in_wallet = engineered_profile.get("points_in_wallet")
        joined_through_referral = raw_profile.get("joined_through_referral")
        complaint_status = raw_profile.get("complaint_status")
        past_complaint = raw_profile.get("past_complaint")
        feedback = raw_profile.get("feedback")

        if pd.notna(days_since_login) and days_since_login >= self.threshold_config.get("activity_login_gap", 14):
            recommendations.append(
                "Launch a personalized re-engagement campaign with timely reminders and a return incentive."
            )
        if pd.notna(avg_time_spent) and avg_time_spent <= self.threshold_config.get("avg_time_spent_low", 60):
            recommendations.append(
                "Offer tailored onboarding content or in-app guidance to increase engagement depth."
            )
        if pd.notna(avg_transaction_value) and avg_transaction_value <= self.threshold_config.get(
            "avg_transaction_value_low", 15_000
        ):
            recommendations.append(
                "Send a promo bundle or loyalty discount to increase near-term purchase activity."
            )
        if str(past_complaint).lower() == "yes" and complaint_status in {"Unsolved", "No Information Available"}:
            recommendations.append(
                "Escalate the account to customer success for proactive service recovery within 24 hours."
            )
        if feedback in {
            "Poor Product Quality",
            "Poor Website",
            "Poor Customer Service",
            "Too many ads",
            "No reason specified",
        }:
            recommendations.append(
                "Trigger a personalized outreach workflow to understand dissatisfaction and close the loop."
            )
        if pd.isna(points_in_wallet) or (
            pd.notna(points_in_wallet) and points_in_wallet <= self.threshold_config.get("wallet_low", 620)
        ):
            recommendations.append(
                "Promote rewards redemption and remind the customer about loyalty points they can use right away."
            )
        if str(joined_through_referral).lower() == "yes" and risk_band in {"Medium", "High"}:
            recommendations.append(
                "Offer a referral renewal incentive to re-activate both advocacy and product usage."
            )
        if risk_band == "High":
            recommendations.append(
                "Create a priority retention case with a named owner, next action, and follow-up deadline."
            )
        elif risk_band == "Medium":
            recommendations.append(
                "Add the customer to a monitored retention segment for targeted nudges over the next two weeks."
            )

        if top_driver_messages:
            for message in top_driver_messages[:2]:
                if "complaint" in message.lower():
                    recommendations.append(
                        "Pair the retention offer with an explicit service recovery message so the intervention feels relevant."
                    )
                    break

        deduplicated: list[str] = []
        for recommendation in recommendations:
            if recommendation not in deduplicated:
                deduplicated.append(recommendation)
        return deduplicated[:5]
