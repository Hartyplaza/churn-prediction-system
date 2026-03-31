from __future__ import annotations

from src.data.load_data import load_train_data
from app.services.predictor import PredictorService


def test_predictor_returns_expected_contract(ensure_trained_bundle) -> None:
    predictor = PredictorService()
    sample = load_train_data().drop(columns=["churn_risk_score"]).iloc[0].to_dict()
    result = predictor.predict_record(sample)

    assert "predicted_label" in result
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert result["recommendations"]
    assert result["driver_details"]


def test_recommendation_engine_is_deterministic(ensure_trained_bundle) -> None:
    predictor = PredictorService()
    sample = load_train_data().drop(columns=["churn_risk_score"]).iloc[1].to_dict()
    first = predictor.recommend_only(sample, risk_band="High")
    second = predictor.recommend_only(sample, risk_band="High")
    assert first == second
