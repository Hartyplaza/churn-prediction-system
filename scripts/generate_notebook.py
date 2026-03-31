from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            "# Productionized Churn Prediction System\n\n"
            "This notebook teaches the end-to-end workflow behind the churn prediction system: data validation, "
            "leakage prevention, feature engineering, ordinal-aware model selection, explainability, and "
            "stakeholder-friendly retention actions."
        ),
        markdown_cell(
            "## 1. Setup and Imports\n\n"
            "We load the project modules rather than rewriting logic in the notebook. That keeps the notebook "
            "fully aligned with the production training and inference code."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from app.core.config import get_settings\n"
            "from src.data.load_data import load_train_data, load_test_data\n"
            "from src.data.validate_data import validate_training_frame\n"
            "from src.data.preprocess import prepare_training_matrices\n"
            "from src.pipelines.training_pipeline import run_training_pipeline\n"
            "settings = get_settings()\n"
            "train_df = load_train_data()\n"
            "test_df = load_test_data()\n"
            "train_df.head()"
        ),
        markdown_cell(
            "## 2. Dataset Overview\n\n"
            "We inspect shape, schema, and target behavior first. This is also where we verify whether the target "
            "should be treated as binary, multiclass, ordinal, or continuous."
        ),
        code_cell(
            "print(train_df.shape)\n"
            "print(train_df.dtypes)\n"
            "train_df['churn_risk_score'].value_counts().sort_index()"
        ),
        markdown_cell(
            "## 3. Validation Report\n\n"
            "Production systems should fail early when data quality drifts. We validate required columns, duplicate "
            "IDs, and missing target values before training."
        ),
        code_cell(
            "report = validate_training_frame(train_df, settings.target_column)\n"
            "report.to_dict()"
        ),
        markdown_cell(
            "## 4. Missingness and Leakage Review\n\n"
            "This project intentionally removes customer identifiers and personally identifying fields from the "
            "training matrix so the model learns behavior rather than memorizing people."
        ),
        code_cell(
            "train_df.isna().sum().sort_values(ascending=False).head(10)"
        ),
        markdown_cell(
            "## 5. Feature Engineering Rationale\n\n"
            "We derive tenure, time-of-day, activity, complaint, dissatisfaction, value, and loyalty features. "
            "The same feature builder is serialized and reused at inference time."
        ),
        code_cell(
            "feature_builder, target_manager, X, y = prepare_training_matrices(train_df)\n"
            "print(target_manager.metadata())\n"
            "X.head()"
        ),
        markdown_cell(
            "## 6. Full Training Pipeline\n\n"
            "The training pipeline benchmarks multiple candidate models, scores them with weighted F1 plus "
            "ordinal-aware metrics, saves the winning model bundle, and produces plots and sample outputs."
        ),
        code_cell(
            "summary = run_training_pipeline()\n"
            "summary"
        ),
        markdown_cell(
            "## 7. Saved Metrics and Comparison Table\n\n"
            "These files are what the API and Streamlit interface use to present explainability and performance "
            "to technical and non-technical audiences."
        ),
        code_cell(
            "comparison = pd.read_csv(settings.comparison_path)\n"
            "metrics = json.loads(Path(settings.metrics_path).read_text())\n"
            "comparison"
        ),
        markdown_cell(
            "## 8. Explainability and Recommendations\n\n"
            "A production churn system should say more than 'this customer is risky'. It should also explain why "
            "and what the business should do next."
        ),
        code_cell(
            "from app.services.predictor import PredictorService\n"
            "predictor = PredictorService()\n"
            "example_result = predictor.predict_record(train_df.drop(columns=['churn_risk_score']).iloc[0].to_dict())\n"
            "example_result"
        ),
        markdown_cell(
            "## 9. Teaching Takeaways\n\n"
            "1. Start with validation, not modeling.\n"
            "2. Protect against leakage early.\n"
            "3. Reuse the exact same transformations in training and inference.\n"
            "4. Optimize for business-relevant metrics and readable outputs.\n"
            "5. Production value comes from usability, reliability, and actionability, not just a model score."
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    notebook_dir = PROJECT_ROOT / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / "exploratory_analysis.ipynb"
    root_notebook_path = PROJECT_ROOT / "Churn Prediction.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    root_notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print({"notebook_path": str(notebook_path), "root_notebook_path": str(root_notebook_path)})


if __name__ == "__main__":
    main()
