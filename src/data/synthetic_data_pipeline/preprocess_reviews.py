"""Preprocess raw Amazon Reviews 2023 reviews into a clean reviews file.

Joins reviews with the product catalog to keep only reviewed products that
exist in the catalog.  The resulting user_id pool becomes the source of truth
for all downstream synthetic customer data.

Usage:
    python -m src.data.pipeline.preprocess_reviews
"""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config.data import (
    AMAZON_2023_CATEGORIES,
    AMAZON_2023_RAW_DIR,
    PRODUCT_CATALOG_PATH,
    REVIEWS_PROCESSED_PATH,
)

# Simple HTML tag stripper
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(text: str | None) -> str:
    """Strip HTML tags, normalize whitespace."""
    if text is None or not isinstance(text, str):
        return ""
    text = _HTML_TAG_RE.sub("", text)
    text = " ".join(text.split())
    return text.strip()


def load_raw_reviews() -> pd.DataFrame:
    """Load and merge raw review Parquet files for all configured categories."""
    frames = []
    for category in AMAZON_2023_CATEGORIES:
        path = AMAZON_2023_RAW_DIR / f"reviews_{category}.parquet"
        if not path.exists():
            print(f"  [warn] Missing reviews file: {path.name} — skipping")
            continue
        df = pd.read_parquet(path)
        print(f"  Loaded {len(df):,} reviews from {path.name}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No review files found in {AMAZON_2023_RAW_DIR}. "
            "Run download_amazon_2023.py first."
        )
    return pd.concat(frames, ignore_index=True)


def preprocess_reviews(reviews_df: pd.DataFrame, catalog_product_ids: set[str]) -> pd.DataFrame:
    """Clean reviews and filter to products present in the catalog.

    Args:
        reviews_df: Raw reviews DataFrame.
        catalog_product_ids: Set of valid product_id (parent_asin) strings
            from the product catalog.

    Returns:
        Cleaned reviews DataFrame.
    """
    print(f"  Raw reviews shape: {reviews_df.shape}")

    # Ensure no images column
    for col in ["images", "videos"]:
        if col in reviews_df.columns:
            reviews_df = reviews_df.drop(columns=[col])
            print(f"    Dropped column: {col}")

    # Use parent_asin as the product_id join key
    asin_col = "parent_asin" if "parent_asin" in reviews_df.columns else "asin"
    reviews_df = reviews_df.dropna(subset=[asin_col, "user_id"])
    reviews_df["product_id"] = reviews_df[asin_col].astype(str).str.strip()

    # Filter to products in catalog
    before = len(reviews_df)
    reviews_df = reviews_df[reviews_df["product_id"].isin(catalog_product_ids)]
    print(f"  Filtered to catalog products: {before:,} → {len(reviews_df):,}")

    # Drop duplicates (same user reviewing same product)
    reviews_df = reviews_df.drop_duplicates(
        subset=["user_id", "product_id"], keep="first"
    )
    print(f"  After dedup (user, product): {len(reviews_df):,}")

    # Clean text fields
    reviews_df["title"] = reviews_df.get("title", pd.Series(dtype=str)).apply(_clean_text)
    reviews_df["text"] = reviews_df.get("text", pd.Series(dtype=str)).apply(_clean_text)

    # Generate review_id
    reviews_df["review_id"] = [str(uuid.uuid4()) for _ in range(len(reviews_df))]

    # Select and order final columns
    output_cols = [
        "review_id",
        "product_id",
        "user_id",
        "rating",
        "title",
        "text",
        "timestamp",
        "verified_purchase",
        "helpful_vote",
    ]
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in reviews_df.columns]
    result = reviews_df[output_cols].copy()

    # Convert timestamp if numeric (milliseconds)
    if "timestamp" in result.columns:
        ts = result["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            result["timestamp"] = pd.to_datetime(ts, unit="ms", errors="coerce")

    result = result.reset_index(drop=True)
    print(f"  Final reviews: {result.shape}")
    return result


def save_reviews(reviews: pd.DataFrame) -> Path:
    """Save cleaned reviews DataFrame to CSV."""
    REVIEWS_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(REVIEWS_PROCESSED_PATH, index=False)
    print(f"  ✅ Saved reviews → {REVIEWS_PROCESSED_PATH.name} ({len(reviews):,} rows)")
    return REVIEWS_PROCESSED_PATH


def run(force: bool = False) -> pd.DataFrame:
    """Full reviews preprocessing pipeline."""
    if REVIEWS_PROCESSED_PATH.exists() and not force:
        print(f"  [skip] Reviews already exist: {REVIEWS_PROCESSED_PATH.name}")
        return pd.read_csv(REVIEWS_PROCESSED_PATH)

    # Load product catalog to get valid product IDs
    if not PRODUCT_CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Product catalog not found at {PRODUCT_CATALOG_PATH}. "
            "Run preprocess_products.py first."
        )
    catalog = pd.read_csv(PRODUCT_CATALOG_PATH, usecols=["product_id"])
    catalog_ids = set(catalog["product_id"].astype(str))
    print(f"  Product catalog: {len(catalog_ids):,} products")

    raw = load_raw_reviews()
    cleaned = preprocess_reviews(raw, catalog_ids)
    save_reviews(cleaned)
    return cleaned


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess reviews.")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists.")
    args = parser.parse_args()
    run(force=args.force)
