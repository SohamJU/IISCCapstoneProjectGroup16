"""Dataset loading utilities."""

import pandas as pd
from src.config.data import AMAZON_RAW_DATA_DIR, AMAZON_RAW_DATA_FILENAME


def load_electronics_reviews() -> pd.DataFrame:
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





if __name__ == "__main__":
    df = load_electronics_reviews()

    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())