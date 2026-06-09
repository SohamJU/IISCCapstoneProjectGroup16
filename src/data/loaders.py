"""Dataset loading utilities."""

from pathlib import Path
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_electronics_reviews():
    """
    Load Amazon Electronics ratings dataset.

    Returns
    -------
    pd.DataFrame
        Electronics ratings dataframe
    """
    file_path = RAW_DATA_DIR / "ratings_Electronics.csv"

    df = pd.read_csv(
        file_path,
        names=["user_id", "product_id", "rating", "timestamp"],
        header=None
    )

    return df

#How To Load Dataset in other modules
#from src.data.loaders import load_electronics_reviews
#df = load_electronics_reviews()
#print(df.head())



if __name__ == "__main__":
    df = load_electronics_reviews()

    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())