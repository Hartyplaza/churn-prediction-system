from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.core.config import get_settings


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_train_data() -> pd.DataFrame:
    settings = get_settings()
    return load_dataset(settings.raw_data_dir / "train.csv")


def load_test_data() -> pd.DataFrame:
    settings = get_settings()
    return load_dataset(settings.raw_data_dir / "test.csv")
