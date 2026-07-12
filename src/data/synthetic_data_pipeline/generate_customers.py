"""Generate synthetic customer profiles from real Amazon Reviews 2023 user IDs.

Customer IDs are sampled directly from the reviews.csv user_id column —
every customer_id is a REAL reviewer ID. Faker is used only for profile
attributes (name, email, address, etc.).

Constraint: set(customers.customer_id) ⊆ set(reviews.user_id)

Usage:
    python -m src.data.pipeline.generate_customers
    python -m src.data.pipeline.generate_customers --num-customers 500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from typing import Any
from faker import Faker

from src.config.data import (
    CUSTOMERS_PATH,
    NUM_CUSTOMERS,
    PRODUCT_CATALOG_PATH,
    REVIEWS_PROCESSED_PATH,
)

fake = Faker()
Faker.seed(42)
random.seed(42)

# ── Weighted distributions ────────────────────────────────────────────────
LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
LOYALTY_WEIGHTS = [0.50, 0.30, 0.15, 0.05]

LANGUAGES = ["en", "es", "fr"]
LANGUAGE_WEIGHTS = [0.95, 0.03, 0.02]


def _pick_weighted(options: list[str], weights: list[float]) -> str:
    return random.choices(options, weights=weights, k=1)[0]


def _infer_preferred_categories(
    user_id: str,
    reviews_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
) -> str:
    """Infer preferred categories from the products this user actually reviewed."""
    user_products = reviews_df.loc[
        reviews_df["user_id"] == user_id, "product_id"
    ].unique()
    if len(user_products) == 0:
        return "Electronics"

    cats = catalog_df.loc[
        catalog_df["product_id"].isin(user_products), "main_category"
    ].dropna().unique()
    if len(cats) == 0:
        return "Electronics"
    return " | ".join(sorted(set(cats)))


def generate_customers(
    num_customers: int = NUM_CUSTOMERS,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate customer profiles using real user IDs from reviews.

    Args:
        num_customers: Target number of customer profiles.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of customer profiles.
    """
    random.seed(seed)
    Faker.seed(seed)

    # Load reviews to get real user IDs
    if not REVIEWS_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Reviews not found at {REVIEWS_PROCESSED_PATH}. "
            "Run preprocess_reviews.py first."
        )
    reviews = pd.read_csv(REVIEWS_PROCESSED_PATH, usecols=["user_id", "product_id"])

    # Load product catalog for category inference
    catalog = pd.DataFrame()
    if PRODUCT_CATALOG_PATH.exists():
        catalog = pd.read_csv(
            PRODUCT_CATALOG_PATH, usecols=["product_id", "main_category"]
        )

    # Select top N most active reviewers
    user_counts = reviews["user_id"].value_counts()
    top_users = user_counts.head(num_customers).index.tolist()

    if len(top_users) < num_customers:
        print(
            f"  [warn] Only {len(top_users):,} unique reviewers available "
            f"(requested {num_customers:,}). Using all."
        )

    print(f"  Generating {len(top_users):,} customer profiles …")
    rows: list[dict[str, Any]] = []
    for user_id in top_users:
        pref_cats = _infer_preferred_categories(user_id, reviews, catalog)
        rows.append(
            {
                "customer_id": user_id,
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "email": fake.email(),
                "loyalty_tier": _pick_weighted(LOYALTY_TIERS, LOYALTY_WEIGHTS),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "country": "US",
                "signup_date": fake.date_between(
                    start_date="-5y", end_date="-30d"
                ).isoformat(),
                "preferred_categories": pref_cats,
                "language": _pick_weighted(LANGUAGES, LANGUAGE_WEIGHTS),
            }
        )

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df):,} customer profiles")
    return df


def save_customers(df: pd.DataFrame) -> Path:
    """Save customer profiles to CSV."""
    CUSTOMERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CUSTOMERS_PATH, index=False)
    print(f"  ✅ Saved customers → {CUSTOMERS_PATH.name} ({len(df):,} rows)")
    return CUSTOMERS_PATH


def run(num_customers: int = NUM_CUSTOMERS, force: bool = False) -> pd.DataFrame:
    """Generate and save customer profiles."""
    if CUSTOMERS_PATH.exists() and not force:
        print(f"  [skip] Customers already exist: {CUSTOMERS_PATH.name}")
        return pd.read_csv(CUSTOMERS_PATH)

    df = generate_customers(num_customers=num_customers)
    save_customers(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic customer profiles.")
    parser.add_argument(
        "--num-customers",
        type=int,
        default=NUM_CUSTOMERS,
        help=f"Number of customers (default: {NUM_CUSTOMERS:,}).",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists.")
    args = parser.parse_args()
    run(num_customers=args.num_customers, force=args.force)
