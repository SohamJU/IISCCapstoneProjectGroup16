"""Preprocess raw Amazon Reviews 2023 metadata into a clean product catalog.

Merges metadata from all configured categories, cleans fields, drops
images/videos, and produces a single product_catalog.csv.

Usage:
    python -m src.data.pipeline.preprocess_products
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config.data import (
    AMAZON_2023_CATEGORIES,
    AMAZON_2023_RAW_DIR,
    PRODUCT_CATALOG_PATH,
)


def _safe_parse_price(val) -> float | None:
    """Parse price from various formats to float, returning None on failure."""
    if val is None or val == "" or val == "None":
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    # Remove currency symbols and commas
    for char in ["$", "€", "£", ","]:
        val_str = val_str.replace(char, "")
    # Handle ranges like "10.99 - 19.99" → take first price
    if " - " in val_str:
        val_str = val_str.split(" - ")[0].strip()
    if "-" in val_str and val_str[0] != "-":
        val_str = val_str.split("-")[0].strip()
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return None


def _flatten_list_field(val) -> str:
    """Convert a list field to a semicolon-separated string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return "; ".join(str(item).strip() for item in val if item)
    if isinstance(val, str):
        # Try to parse as Python list literal
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return "; ".join(str(item).strip() for item in parsed if item)
        except (ValueError, SyntaxError):
            pass
        return val.strip()
    return str(val).strip()


def _flatten_description(val) -> str:
    """Convert description field (often a list of paragraphs) to single text."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(p).strip() for p in val if p)
    return str(val).strip()


def _extract_bestseller(details) -> bool:
    """Try to extract best-seller flag from the details field."""
    if details is None:
        return False
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except (json.JSONDecodeError, TypeError):
            # Check for string-based hints
            return "best seller" in details.lower()
    if isinstance(details, dict):
        for key, value in details.items():
            key_lower = key.lower()
            if "best seller" in key_lower or "bestseller" in key_lower:
                return True
            if isinstance(value, str) and "best seller" in value.lower():
                return True
    return False


def _flatten_bought_together(val) -> str:
    """Convert bought_together field to comma-separated ASINs."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(item) for item in val if item)
    return str(val).strip()


def _flatten_categories(val) -> str:
    """Convert categories list to pipe-separated string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " | ".join(str(c).strip() for c in val if c)
    return str(val).strip()


def load_raw_metadata() -> pd.DataFrame:
    """Load and merge raw metadata Parquet files for all categories."""
    frames = []
    for category in AMAZON_2023_CATEGORIES:
        path = AMAZON_2023_RAW_DIR / f"meta_{category}.parquet"
        if not path.exists():
            print(f"  [warn] Missing metadata file: {path.name} — skipping")
            continue
        df = pd.read_parquet(path)
        print(f"  Loaded {len(df):,} products from {path.name}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No metadata files found in {AMAZON_2023_RAW_DIR}. "
            "Run download_amazon_2023.py first."
        )
    return pd.concat(frames, ignore_index=True)


def preprocess_product_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw metadata into a structured product catalog.

    Returns:
        DataFrame with cleaned product catalog columns.
    """
    print(f"  Raw metadata shape: {df.shape}")

    # Ensure no images/videos columns sneak through
    for col in ["images", "videos"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"    Dropped column: {col}")

    # Drop rows without essential fields
    df = df.dropna(subset=["parent_asin"])
    df = df[df["parent_asin"].astype(str).str.strip() != ""]
    df = df.dropna(subset=["title"])
    df = df[df["title"].astype(str).str.strip() != ""]
    print(f"  After dropping missing parent_asin/title: {df.shape}")

    # Deduplicate on parent_asin
    df = df.drop_duplicates(subset=["parent_asin"], keep="first")
    print(f"  After dedup on parent_asin: {df.shape}")

    # Build clean catalog
    catalog = pd.DataFrame()
    catalog["product_id"] = df["parent_asin"].astype(str).str.strip()
    catalog["title"] = df["title"].astype(str).str.strip()
    catalog["main_category"] = (
        df["main_category"].fillna("").astype(str).str.strip()
    )
    catalog["sub_categories"] = df.get("categories", pd.Series(dtype=object)).apply(
        _flatten_categories
    )
    catalog["price"] = df["price"].apply(_safe_parse_price)
    catalog["average_rating"] = pd.to_numeric(
        df.get("average_rating", pd.Series(dtype=float)), errors="coerce"
    )
    catalog["rating_count"] = pd.to_numeric(
        df.get("rating_number", pd.Series(dtype=int)), errors="coerce"
    ).fillna(0).astype(int)
    catalog["features"] = df.get("features", pd.Series(dtype=object)).apply(
        _flatten_list_field
    )
    catalog["description"] = df.get("description", pd.Series(dtype=object)).apply(
        _flatten_description
    )
    catalog["bought_together"] = df.get(
        "bought_together", pd.Series(dtype=object)
    ).apply(_flatten_bought_together)
    catalog["store"] = df.get("store", pd.Series(dtype=object)).fillna("").astype(str).str.strip()
    catalog["is_bestseller"] = df.get("details", pd.Series(dtype=object)).apply(
        _extract_bestseller
    )

    catalog = catalog.reset_index(drop=True)
    print(f"  Final product catalog: {catalog.shape}")
    return catalog


def save_product_catalog(catalog: pd.DataFrame) -> Path:
    """Save the product catalog DataFrame to CSV."""
    PRODUCT_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(PRODUCT_CATALOG_PATH, index=False)
    print(f"  ✅ Saved product catalog → {PRODUCT_CATALOG_PATH.name} ({len(catalog):,} rows)")
    return PRODUCT_CATALOG_PATH


def run(force: bool = False) -> pd.DataFrame:
    """Full preprocessing pipeline: load → clean → save."""
    if PRODUCT_CATALOG_PATH.exists() and not force:
        print(f"  [skip] Product catalog already exists: {PRODUCT_CATALOG_PATH.name}")
        return pd.read_csv(PRODUCT_CATALOG_PATH)

    raw = load_raw_metadata()
    catalog = preprocess_product_catalog(raw)
    save_product_catalog(catalog)
    return catalog


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess product catalog.")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists.")
    args = parser.parse_args()
    run(force=args.force)
