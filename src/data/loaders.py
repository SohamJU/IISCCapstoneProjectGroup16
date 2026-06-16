"""Dataset loading utilities."""

import pandas as pd

from src.config.data import (
    AMAZON_RAW_DATA_DIR,
    AMAZON_RAW_DATA_FILENAME,
    TWITTER_SUPPORT_DATA_DIR,
    TWITTER_SUPPORT_DATA_FILENAME,
)


def load_electronics_reviews():
    """
    Load Amazon Electronics ratings dataset.

    Returns
    -------
    pd.DataFrame
        Electronics ratings dataframe
    """
    file_path = AMAZON_RAW_DATA_DIR / AMAZON_RAW_DATA_FILENAME

    df = pd.read_csv(
        file_path,
        names=["user_id", "product_id", "rating", "timestamp"],
        header=None
    )

    return df


def load_twitter_support_conversations(file_path=None) -> pd.DataFrame:
    """Load the Kaggle Twitter customer support dataset."""

    if file_path is None:
        file_path = TWITTER_SUPPORT_DATA_DIR / TWITTER_SUPPORT_DATA_FILENAME

    return pd.read_csv(file_path)





if __name__ == "__main__":
    df = load_electronics_reviews()

    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
