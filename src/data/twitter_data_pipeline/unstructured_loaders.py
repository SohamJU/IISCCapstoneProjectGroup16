from __future__ import annotations

"""Loading utilities for unstructured datasets."""

from pathlib import Path

import pandas as pd

from src.config.unstructured_data import (
    TWITTER_SUPPORT_DATA_DIR,
    TWITTER_SUPPORT_DATA_FILENAME,
)


def load_twitter_support_conversations(file_path: Path | None = None) -> pd.DataFrame:
    """Load the Kaggle Twitter customer support dataset."""

    if file_path is None:
        file_path = TWITTER_SUPPORT_DATA_DIR / TWITTER_SUPPORT_DATA_FILENAME

    return pd.read_csv(file_path)
