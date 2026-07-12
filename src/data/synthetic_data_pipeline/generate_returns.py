"""Generate synthetic returns — ONLY for delivered orders.

Constraints enforced:
  - Only orders with status="delivered" are eligible
  - return.order_id → orders (delivered)
  - return.product_id → order_items for that order
  - return.customer_id → orders.customer_id for that order
  - request_date > actual_delivery_date

Usage:
    python -m src.data.pipeline.generate_returns
    python -m src.data.pipeline.generate_returns --num-returns 500
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config.data import (
    NUM_RETURNS,
    ORDERS_PATH,
    ORDER_ITEMS_PATH,
    RETURNS_PATH,
)

random.seed(42)

# ── Return reasons with weights ───────────────────────────────────────────
RETURN_REASONS = [
    "Defective/damaged",
    "Item not as described",
    "Wrong item received",
    "Changed my mind",
    "Better price available",
    "Arrived too late",
    "Missing parts/accessories",
]
REASON_WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]

# ── Return status weights ─────────────────────────────────────────────────
RETURN_STATUSES = ["approved", "pending", "rejected", "refunded"]
RETURN_STATUS_WEIGHTS = [0.60, 0.20, 0.10, 0.10]


def _pick_weighted(options: list[Any], weights: list[float]) -> Any:
    return random.choices(options, weights=weights, k=1)[0]


def generate_returns(
    num_returns: int = NUM_RETURNS,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate return records for delivered orders only.

    Args:
        num_returns: Target number of returns to generate.
        seed: Random seed.

    Returns:
        DataFrame of returns.
    """
    random.seed(seed)

    # ── Load orders & items ────────────────────────────────────────────────
    if not ORDERS_PATH.exists():
        raise FileNotFoundError(f"Orders not found: {ORDERS_PATH}")
    if not ORDER_ITEMS_PATH.exists():
        raise FileNotFoundError(f"Order items not found: {ORDER_ITEMS_PATH}")

    orders = pd.read_csv(ORDERS_PATH)
    items = pd.read_csv(ORDER_ITEMS_PATH)

    # Only delivered orders are eligible for returns
    delivered = orders[orders["status"] == "delivered"].copy()
    delivered = delivered.dropna(subset=["actual_delivery_date"])
    print(f"  Eligible delivered orders: {len(delivered):,}")

    if len(delivered) == 0:
        print("  ⚠ No delivered orders found. Cannot generate returns.")
        return pd.DataFrame()

    # Cap num_returns to available delivered orders
    actual_count = min(num_returns, len(delivered))
    if actual_count < num_returns:
        print(
            f"  [warn] Only {actual_count:,} delivered orders available "
            f"(requested {num_returns:,} returns)"
        )

    # Sample delivered orders for returns
    sampled_orders = delivered.sample(n=actual_count, random_state=seed)

    rows: list[dict[str, Any]] = []
    for idx, (_, order) in enumerate(sampled_orders.iterrows(), start=1):
        oid = order["order_id"]
        cid = order["customer_id"]
        delivery_date = pd.Timestamp(order["actual_delivery_date"])

        # Pick one item from this order
        order_items = items[items["order_id"] == oid]
        if len(order_items) == 0:
            continue  # shouldn't happen, but safety check

        chosen_item = order_items.sample(n=1, random_state=seed + idx).iloc[0]
        pid = chosen_item["product_id"]
        oiid = chosen_item["order_item_id"]
        unit_price = float(chosen_item["unit_price"])
        quantity = int(chosen_item["quantity"])

        # Return status
        ret_status = _pick_weighted(RETURN_STATUSES, RETURN_STATUS_WEIGHTS)

        # Refund amount
        full_refund = round(unit_price * quantity, 2)
        if ret_status == "rejected":
            refund_amount = round(full_refund * random.uniform(0.0, 0.3), 2)
        else:
            refund_amount = full_refund

        # Request date: 1-30 days after delivery
        request_date = delivery_date + timedelta(days=random.randint(1, 30))

        rows.append(
            {
                "return_id": f"RET-{idx:06d}",
                "order_id": oid,
                "order_item_id": oiid,
                "product_id": pid,
                "customer_id": cid,
                "reason": _pick_weighted(RETURN_REASONS, REASON_WEIGHTS),
                "status": ret_status,
                "refund_amount": refund_amount,
                "request_date": request_date.strftime("%Y-%m-%d"),
            }
        )

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df):,} returns")
    return df


def save_returns(df: pd.DataFrame) -> Path:
    """Save returns to CSV."""
    RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RETURNS_PATH, index=False)
    print(f"  ✅ Saved returns → {RETURNS_PATH.name} ({len(df):,} rows)")
    return RETURNS_PATH


def run(num_returns: int = NUM_RETURNS, force: bool = False) -> pd.DataFrame:
    """Generate and save returns."""
    if RETURNS_PATH.exists() and not force:
        print(f"  [skip] Returns already exist: {RETURNS_PATH.name}")
        return pd.read_csv(RETURNS_PATH)

    df = generate_returns(num_returns=num_returns)
    save_returns(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic returns.")
    parser.add_argument(
        "--num-returns", type=int, default=NUM_RETURNS,
        help=f"Target return count (default: {NUM_RETURNS:,}).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(num_returns=args.num_returns, force=args.force)
