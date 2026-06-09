"""Data preprocessing utilities."""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def preprocess_ratings(
    df: pd.DataFrame,
    min_user_interactions: int = 5,
    min_product_interactions: int = 5
) -> pd.DataFrame:
    """
    Preprocess Amazon Electronics ratings dataset.

    Steps:
    1. Remove duplicates
    2. Remove missing values
    3. Convert timestamp to datetime
    4. Keep valid ratings only
    5. Remove inactive users
    6. Remove unpopular products
    """

    print(f"Initial shape: {df.shape}")

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna()

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s"
    )

    # Keep valid ratings
    df = df[
        (df["rating"] >= 1.0) &
        (df["rating"] <= 5.0)
    ]

    # Filter inactive users
    active_users = (
        df["user_id"]
        .value_counts()
    )

    active_users = active_users[
        active_users >= min_user_interactions
    ].index

    df = df[
        df["user_id"].isin(active_users)
    ]

    # Filter unpopular products
    popular_products = (
        df["product_id"]
        .value_counts()
    )

    popular_products = popular_products[
        popular_products >= min_product_interactions
    ].index

    df = df[
        df["product_id"].isin(popular_products)
    ]

    print(f"Final shape: {df.shape}")

    return df


def save_processed_data(
    df: pd.DataFrame,
    filename: str = "ratings_electronics_processed.csv"
):
    """
    Save processed dataframe.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = PROCESSED_DATA_DIR / filename

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved to: {output_path}")