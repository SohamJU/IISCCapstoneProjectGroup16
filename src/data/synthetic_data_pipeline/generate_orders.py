"""Generate synthetic orders and order items.

Two-phase approach:
  Phase A — Review-backed orders: for every (user_id, product_id) pair in
            reviews.csv (within the customer subset), create a DELIVERED order.
  Phase B — Synthetic fill: generate additional orders to reach the target
            count, using any valid customer × product combination.

Constraints enforced:
  - customer_id ∈ customers.csv
  - product_id ∈ product_catalog.csv
  - Every (user_id, product_id) in reviews has a delivered order
  - total_amount = Σ(quantity × unit_price)
  - Business-rule-consistent status / payment / delivery fields

Usage:
    python -m src.data.pipeline.generate_orders
    python -m src.data.pipeline.generate_orders --num-orders 5000
"""

from __future__ import annotations

import argparse
import random
import string
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from typing import Any
from faker import Faker

from src.config.data import (
    CUSTOMERS_PATH,
    NUM_ORDERS,
    ORDERS_PATH,
    ORDER_ITEMS_PATH,
    PRODUCT_CATALOG_PATH,
    REVIEWS_PROCESSED_PATH,
)

fake = Faker()
Faker.seed(42)
random.seed(42)

# ── Weighted distributions ────────────────────────────────────────────────
# Phase B only — Phase A is always "delivered"
STATUS_OPTIONS = ["delivered", "shipped", "processing", "cancelled", "returned"]
STATUS_WEIGHTS = [0.60, 0.15, 0.10, 0.10, 0.05]

QUANTITY_OPTIONS = [1, 2, 3]
QUANTITY_WEIGHTS = [0.70, 0.20, 0.10]

ITEMS_PER_ORDER_OPTIONS = [1, 2, 3, 4]
ITEMS_PER_ORDER_WEIGHTS = [0.50, 0.30, 0.15, 0.05]

# Status → payment_status mapping
PAYMENT_STATUS_MAP = {
    "delivered": "paid",
    "shipped": "paid",
    "processing": "pending",
    "cancelled": "refunded",
    "returned": "refunded",
}


def _random_tracking() -> str:
    return "TRK-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=12))


def _pick_weighted(options: list[Any], weights: list[float]) -> Any:
    return random.choices(options, weights=weights, k=1)[0]


def _build_order_record(
    order_id: str,
    customer_id: str,
    status: str,
    order_date: pd.Timestamp,
 ) -> dict[str, Any]:
    """Build a single order row with consistent business rules."""
    est_delivery = order_date + timedelta(days=random.randint(3, 10))
    has_delivered = status in ("delivered", "returned")

    if has_delivered:
        actual_delivery = est_delivery + timedelta(days=random.randint(-2, 5))
    else:
        actual_delivery = None

    tracking = _random_tracking() if status != "processing" else None
    payment_status = PAYMENT_STATUS_MAP.get(status, "paid")

    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "status": status,
        "tracking_number": tracking,
        "est_delivery_date": est_delivery.strftime("%Y-%m-%d"),
        "actual_delivery_date": (
            actual_delivery.strftime("%Y-%m-%d") if actual_delivery else None
        ),
        "payment_status": payment_status,
        "shipping_address": fake.address().replace("\n", ", "),
        "total_amount": 0.0,  # filled after items are generated
    }


def _build_order_item(
    order_item_id: str,
    order_id: str,
    customer_id: str,
    product_id: str,
    unit_price: float,
    status: str,
 ) -> dict[str, Any]:
    """Build a single order item row."""
    quantity = _pick_weighted(QUANTITY_OPTIONS, QUANTITY_WEIGHTS)
    return {
        "order_item_id": order_item_id,
        "order_id": order_id,
        "product_id": product_id,
        "customer_id": customer_id,
        "quantity": quantity,
        "unit_price": round(unit_price, 2),
        "item_status": status,
    }


def generate_orders(
    num_orders: int = NUM_ORDERS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate orders and order_items DataFrames.

    Returns:
        (orders_df, order_items_df)
    """
    random.seed(seed)
    Faker.seed(seed)

    # ── Load dependencies ──────────────────────────────────────────────────
    if not CUSTOMERS_PATH.exists():
        raise FileNotFoundError(f"Customers not found: {CUSTOMERS_PATH}")
    if not PRODUCT_CATALOG_PATH.exists():
        raise FileNotFoundError(f"Product catalog not found: {PRODUCT_CATALOG_PATH}")
    if not REVIEWS_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Reviews not found: {REVIEWS_PROCESSED_PATH}")

    customers = pd.read_csv(CUSTOMERS_PATH, usecols=["customer_id"])
    customer_ids = set(customers["customer_id"].astype(str))

    catalog = pd.read_csv(PRODUCT_CATALOG_PATH, usecols=["product_id", "price"])
    # Build product → price lookup; use median price as fallback for nulls
    median_price = catalog["price"].dropna().median()
    if pd.isna(median_price):
        median_price = 29.99
    product_prices = dict(
        zip(
            catalog["product_id"].astype(str),
            catalog["price"].fillna(median_price).astype(float),
        )
    )
    product_ids = list(product_prices.keys())

    reviews = pd.read_csv(REVIEWS_PROCESSED_PATH, usecols=["user_id", "product_id", "timestamp"])

    # ── Phase A: Review-backed orders ──────────────────────────────────────
    # For every (user_id, product_id) in reviews WHERE user_id ∈ customers
    review_pairs = reviews[reviews["user_id"].astype(str).isin(customer_ids)].copy()
    review_pairs = review_pairs.drop_duplicates(subset=["user_id", "product_id"])
    print(f"  Phase A: {len(review_pairs):,} review-backed (user, product) pairs")

    orders_rows: list[dict[str, Any]] = []
    items_rows: list[dict[str, Any]] = []
    order_counter = 1
    item_counter = 1

    for _, row in review_pairs.iterrows():
        uid = str(row["user_id"])
        pid = str(row["product_id"])
        oid = f"ORD-{order_counter:06d}"
        order_counter += 1

        # Order date = review timestamp – random days (they bought before reviewing)
        try:
            review_ts = pd.Timestamp(row["timestamp"])
            if pd.isna(review_ts):
                raise ValueError
            order_date = review_ts - timedelta(days=random.randint(1, 30))
        except (ValueError, TypeError):
            order_date = fake.date_time_between(start_date="-2y", end_date="-30d") # type: ignore[assignment]
            order_date = pd.Timestamp(order_date)

        order_rec = _build_order_record(oid, uid, "delivered", order_date)

        # Build the line item for the reviewed product
        price = product_prices.get(pid, median_price)
        price_variation = round(price * random.uniform(0.95, 1.05), 2)
        item_rec = _build_order_item(
            f"OI-{item_counter:07d}", oid, uid, pid, price_variation, "delivered"
        )
        item_counter += 1

        # Optionally add 1-2 extra items to the same order (30% chance)
        extra_items = [item_rec]
        if random.random() < 0.30:
            n_extra = random.randint(1, 2)
            for _ in range(n_extra):
                extra_pid = random.choice(product_ids)
                extra_price = product_prices.get(extra_pid, median_price)
                extra_price = round(extra_price * random.uniform(0.95, 1.05), 2)
                extra_items.append(
                    _build_order_item(
                        f"OI-{item_counter:07d}",
                        oid,
                        uid,
                        extra_pid,
                        extra_price,
                        "delivered",
                    )
                )
                item_counter += 1

        # Compute total
        order_rec["total_amount"] = round(
            sum(it["quantity"] * it["unit_price"] for it in extra_items), 2
        )
        orders_rows.append(order_rec)
        items_rows.extend(extra_items)

    print(f"  Phase A complete: {len(orders_rows):,} orders, {len(items_rows):,} items")

    # ── Phase B: Synthetic fill to reach target ────────────────────────────
    remaining = max(0, num_orders - len(orders_rows))
    if remaining > 0:
        print(f"  Phase B: generating {remaining:,} additional synthetic orders …")
        customer_list = list(customer_ids)
        # Power-law: some customers order more
        customer_weights = [1.0 / (i + 1) ** 0.5 for i in range(len(customer_list))]

        for _ in range(remaining):
            oid = f"ORD-{order_counter:06d}"
            order_counter += 1
            cid = random.choices(customer_list, weights=customer_weights, k=1)[0]
            status = _pick_weighted(STATUS_OPTIONS, STATUS_WEIGHTS)
            order_date = pd.Timestamp(
                fake.date_time_between(start_date="-2y", end_date="-1d")
            )

            order_rec = _build_order_record(oid, cid, status, order_date)

            # Generate 1-4 line items
            n_items = _pick_weighted(ITEMS_PER_ORDER_OPTIONS, ITEMS_PER_ORDER_WEIGHTS)
            order_items = []
            for _ in range(n_items):
                pid = random.choice(product_ids)
                price = product_prices.get(pid, median_price)
                price = round(price * random.uniform(0.95, 1.05), 2)
                item_rec = _build_order_item(
                    f"OI-{item_counter:07d}", oid, cid, pid, price, status
                )
                item_counter += 1
                order_items.append(item_rec)

            order_rec["total_amount"] = round(
                sum(it["quantity"] * it["unit_price"] for it in order_items), 2
            )
            orders_rows.append(order_rec)
            items_rows.extend(order_items)

    orders_df = pd.DataFrame(orders_rows)
    items_df = pd.DataFrame(items_rows)
    print(f"  Total: {len(orders_df):,} orders, {len(items_df):,} order items")
    return orders_df, items_df


def save_orders(orders_df: pd.DataFrame, items_df: pd.DataFrame) -> tuple[Path, Path]:
    """Save orders and order items to CSV."""
    ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    orders_df.to_csv(ORDERS_PATH, index=False)
    items_df.to_csv(ORDER_ITEMS_PATH, index=False)
    print(f"  ✅ Saved orders → {ORDERS_PATH.name} ({len(orders_df):,} rows)")
    print(f"  ✅ Saved order items → {ORDER_ITEMS_PATH.name} ({len(items_df):,} rows)")
    return ORDERS_PATH, ORDER_ITEMS_PATH


def run(num_orders: int = NUM_ORDERS, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and save orders + items."""
    if ORDERS_PATH.exists() and ORDER_ITEMS_PATH.exists() and not force:
        print(f"  [skip] Orders already exist: {ORDERS_PATH.name}")
        return pd.read_csv(ORDERS_PATH), pd.read_csv(ORDER_ITEMS_PATH)

    orders_df, items_df = generate_orders(num_orders=num_orders)
    save_orders(orders_df, items_df)
    return orders_df, items_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic orders.")
    parser.add_argument(
        "--num-orders", type=int, default=NUM_ORDERS,
        help=f"Target order count (default: {NUM_ORDERS:,}).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(num_orders=args.num_orders, force=args.force)
