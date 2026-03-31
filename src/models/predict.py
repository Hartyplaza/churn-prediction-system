from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import get_settings


def load_bundle(bundle_path: str | Path | None = None) -> dict[str, Any]:
    settings = get_settings()
    path = Path(bundle_path or settings.bundle_path)
    return joblib.load(path)


def score_dataframe(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    feature_builder = bundle["feature_builder"]
    model = bundle["model"]
    target_manager = bundle["target_manager"]
    original_ids = df.get("customer_id", pd.Series([None] * len(df)))
    features = feature_builder.transform(df)
    predictions = model.predict(features)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        confidence = probabilities.max(axis=1)
        risk_scores = target_manager.normalized_score_from_prediction(probabilities=probabilities)
    else:
        probabilities = None
        confidence = np.repeat(np.nan, len(predictions))
        risk_scores = target_manager.normalized_score_from_prediction(predictions=predictions)

    raw_predictions = target_manager.inverse_transform(predictions)

    output = pd.DataFrame(
        {
            "customer_id": original_ids,
            "predicted_class": raw_predictions,
            "predicted_label": [target_manager.label_name(value) for value in raw_predictions],
            "risk_score": np.round(risk_scores, 4),
            "risk_band": [target_manager.risk_band(score) for score in risk_scores],
            "confidence": np.round(confidence, 4),
        }
    )
    if probabilities is not None:
        for class_index, class_label in enumerate(target_manager.classes_):
            output[f"prob_class_{class_label}"] = np.round(probabilities[:, class_index], 4)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run churn model inference on a CSV file.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", default="artifacts/sample_outputs/cli_predictions.csv")
    parser.add_argument("--bundle-path", default=None)
    args = parser.parse_args()

    bundle = load_bundle(args.bundle_path)
    input_frame = pd.read_csv(args.input)
    predictions = score_dataframe(bundle, input_frame)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(json.dumps({"output_path": str(output_path), "rows_scored": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()
