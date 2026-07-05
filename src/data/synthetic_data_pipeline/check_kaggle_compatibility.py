"""Check compatibility between the legacy Kaggle ratings dataset and Amazon Reviews 2023.

Compares product_id (ASIN) and user_id overlap between the existing Kaggle
`saurav9786/amazon-product-reviews` dataset and the new Amazon Reviews 2023
product catalog / reviews.

Usage:
    python -m src.data.pipeline.check_kaggle_compatibility
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config.data import (
    AMAZON_RAW_DATA_DIR,
    AMAZON_RAW_DATA_FILENAME,
    PRODUCT_CATALOG_PATH,
    REPORTS_DIR,
    REVIEWS_PROCESSED_PATH,
)

REPORT_PATH = REPORTS_DIR / "kaggle_compatibility.txt"


def run(force: bool = False) -> str:
    """Run the compatibility check and return the report text."""
    if REPORT_PATH.exists() and not force:
        print(f"  [skip] Report already exists: {REPORT_PATH.name}")
        return REPORT_PATH.read_text(encoding="utf-8")

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Kaggle ↔ Amazon Reviews 2023 Compatibility Report")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 70)
    lines.append("")

    # ── Load Kaggle data ──────────────────────────────────────────────────
    kaggle_path = AMAZON_RAW_DATA_DIR / AMAZON_RAW_DATA_FILENAME
    if not kaggle_path.exists():
        msg = f"⚠  Kaggle dataset not found at {kaggle_path}. Skipping check."
        lines.append(msg)
        print(f"  {msg}")
        report = "\n".join(lines)
        _save_report(report)
        return report

    kaggle_df = pd.read_csv(
        kaggle_path,
        names=["user_id", "product_id", "rating", "timestamp"],
        header=None,
    )
    kaggle_user_ids = set(kaggle_df["user_id"].astype(str).unique())
    kaggle_product_ids = set(kaggle_df["product_id"].astype(str).unique())
    lines.append(f"Kaggle dataset: {len(kaggle_df):,} ratings")
    lines.append(f"  Unique user_ids:    {len(kaggle_user_ids):,}")
    lines.append(f"  Unique product_ids: {len(kaggle_product_ids):,}")
    lines.append("")

    # ── Load Amazon 2023 product catalog ──────────────────────────────────
    if not PRODUCT_CATALOG_PATH.exists():
        msg = f"⚠  Product catalog not found at {PRODUCT_CATALOG_PATH}."
        lines.append(msg)
        print(f"  {msg}")
    else:
        catalog = pd.read_csv(PRODUCT_CATALOG_PATH, usecols=["product_id"])
        catalog_ids = set(catalog["product_id"].astype(str))
        overlap_products = kaggle_product_ids & catalog_ids
        pct = (len(overlap_products) / len(kaggle_product_ids) * 100) if kaggle_product_ids else 0

        lines.append(f"Amazon Reviews 2023 product catalog: {len(catalog_ids):,} products")
        lines.append(f"  Product ID overlap: {len(overlap_products):,} / {len(kaggle_product_ids):,} ({pct:.1f}%)")
        if pct < 50:
            lines.append("  ❌ LOW OVERLAP — Most Kaggle product IDs do NOT exist in Amazon 2023.")
        elif pct < 80:
            lines.append("  ⚠  PARTIAL OVERLAP — Some Kaggle products exist in Amazon 2023.")
        else:
            lines.append("  ✅ HIGH OVERLAP — Most Kaggle products found in Amazon 2023.")
        lines.append("")

    # ── Load Amazon 2023 reviews ──────────────────────────────────────────
    if not REVIEWS_PROCESSED_PATH.exists():
        msg = f"⚠  Reviews file not found at {REVIEWS_PROCESSED_PATH}."
        lines.append(msg)
        print(f"  {msg}")
    else:
        reviews = pd.read_csv(REVIEWS_PROCESSED_PATH, usecols=["user_id"])
        review_user_ids = set(reviews["user_id"].astype(str))
        overlap_users = kaggle_user_ids & review_user_ids
        pct = (len(overlap_users) / len(kaggle_user_ids) * 100) if kaggle_user_ids else 0

        lines.append(f"Amazon Reviews 2023 reviews: {len(review_user_ids):,} unique user_ids")
        lines.append(f"  User ID overlap: {len(overlap_users):,} / {len(kaggle_user_ids):,} ({pct:.1f}%)")
        if pct < 5:
            lines.append("  ❌ NEAR-ZERO OVERLAP — User IDs are from different anonymization eras.")
        else:
            lines.append(f"  ⚠  Some overlap detected ({pct:.1f}%).")
        lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────
    lines.append("-" * 70)
    lines.append("RECOMMENDATION")
    lines.append("-" * 70)
    lines.append(
        "The Kaggle dataset (saurav9786/amazon-product-reviews) uses older SNAP-era"
    )
    lines.append(
        "anonymized IDs. User IDs are almost certainly disjoint from Amazon Reviews"
    )
    lines.append(
        "2023. Product IDs (ASINs) may partially overlap since ASINs are persistent"
    )
    lines.append(
        "Amazon identifiers, but coverage depends on whether old products remain in"
    )
    lines.append(
        "the 2023 catalog."
    )
    lines.append("")
    lines.append(
        "→ Treat the Kaggle dataset as a SEPARATE, LEGACY dataset."
    )
    lines.append(
        "→ Do NOT use its IDs for the new synthetic pipeline."
    )
    lines.append(
        "→ All new synthetic data should be grounded in Amazon Reviews 2023 IDs."
    )

    report = "\n".join(lines)
    _save_report(report)
    print(report)
    return report


def _save_report(text: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"\n  ✅ Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check Kaggle ↔ Amazon 2023 compatibility.")
    parser.add_argument("--force", action="store_true", help="Re-run even if report exists.")
    args = parser.parse_args()
    run(force=args.force)
