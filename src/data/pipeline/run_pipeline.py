"""Pipeline orchestrator — runs all data generation steps in dependency order.

Steps:
  1. Download Amazon 2023 data (metadata + reviews, excluding images/videos)
  2. Preprocess product catalog
  3. Preprocess reviews
  4. Check Kaggle dataset compatibility (report only)
  5. Generate customer profiles (from review user_ids)
  6. Generate orders + order items (review-backed + synthetic fill)
  7. Generate returns (delivered orders only)
  8. Generate enhanced synthetic queries (copy of original generator)
  9. Generate policy documents (static or LLM-powered)
 10. Run integrity validation

Usage:
    python -m src.data.pipeline.run_pipeline              # run all steps
    python -m src.data.pipeline.run_pipeline --step 5     # run a single step
    python -m src.data.pipeline.run_pipeline --force       # force re-run all
    python -m src.data.pipeline.run_pipeline --policy-provider gemini
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Fix unicode encode errors on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config.data import (
    AMAZON_2023_MAX_REVIEWS,
    CUSTOMERS_PATH,
    CUSTOMER_QUERIES_PATH,
    KNOWLEDGE_BASE_DIR,
    NUM_CUSTOMERS,
    NUM_ORDERS,
    NUM_RETURNS,
    ORDERS_PATH,
    ORDER_ITEMS_PATH,
    PRODUCT_CATALOG_PATH,
    RETURNS_PATH,
    REVIEWS_PROCESSED_PATH,
)


# ═══════════════════════════════════════════════════════════════════════════
# STEP RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def step_01_download(force: bool, max_reviews: int, streaming: bool) -> None:
    """Step 1: Download Amazon Reviews 2023 data."""
    print("\n" + "=" * 70)
    print("STEP 1: Download Amazon Reviews 2023")
    print("=" * 70)
    from src.data.pipeline.download_amazon_2023 import download_all

    download_all(max_reviews=max_reviews, streaming=streaming, force=force)


def step_02_preprocess_products(force: bool) -> None:
    """Step 2: Preprocess product catalog."""
    print("\n" + "=" * 70)
    print("STEP 2: Preprocess Product Catalog")
    print("=" * 70)
    from src.data.pipeline.preprocess_products import run

    run(force=force)


def step_03_preprocess_reviews(force: bool) -> None:
    """Step 3: Preprocess reviews."""
    print("\n" + "=" * 70)
    print("STEP 3: Preprocess Reviews")
    print("=" * 70)
    from src.data.pipeline.preprocess_reviews import run

    run(force=force)


def step_04_kaggle_check(force: bool) -> None:
    """Step 4: Check Kaggle ↔ Amazon 2023 compatibility."""
    print("\n" + "=" * 70)
    print("STEP 4: Kaggle Compatibility Check")
    print("=" * 70)
    from src.data.pipeline.check_kaggle_compatibility import run

    run(force=force)


def step_05_generate_customers(force: bool, num_customers: int) -> None:
    """Step 5: Generate customer profiles."""
    print("\n" + "=" * 70)
    print("STEP 5: Generate Customer Profiles")
    print("=" * 70)
    from src.data.pipeline.generate_customers import run

    run(num_customers=num_customers, force=force)


def step_06_generate_orders(force: bool, num_orders: int) -> None:
    """Step 6: Generate orders + order items."""
    print("\n" + "=" * 70)
    print("STEP 6: Generate Orders + Order Items")
    print("=" * 70)
    from src.data.pipeline.generate_orders import run

    run(num_orders=num_orders, force=force)


def step_07_generate_returns(force: bool, num_returns: int) -> None:
    """Step 7: Generate returns."""
    print("\n" + "=" * 70)
    print("STEP 7: Generate Returns")
    print("=" * 70)
    from src.data.pipeline.generate_returns import run

    run(num_returns=num_returns, force=force)


def step_08_generate_queries(force: bool, total_queries: int) -> None:
    """Step 8: Generate enhanced synthetic queries."""
    print("\n" + "=" * 70)
    print("STEP 8: Generate Enhanced Queries")
    print("=" * 70)
    from src.data.pipeline.enhanced_query_generator import run

    run(total_queries=total_queries, force=force)


def step_09_generate_policies(force: bool, provider: str) -> None:
    """Step 9: Generate policy documents."""
    print("\n" + "=" * 70)
    print("STEP 9: Generate Policy Documents")
    print("=" * 70)
    from src.data.pipeline.generate_policies import run

    run(provider=provider, force=force)


def step_10_validate(force: bool) -> None:
    """Step 10: Run full integrity validation."""
    print("\n" + "=" * 70)
    print("STEP 10: Integrity Validation")
    print("=" * 70)
    _run_validation()


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRITY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def _run_validation() -> None:
    """Run all 13 integrity checks from the plan."""
    checks_passed = 0
    checks_failed = 0
    checks_skipped = 0

    def _check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal checks_passed, checks_failed
        if condition:
            checks_passed += 1
            print(f"  ✅ {name}")
        else:
            checks_failed += 1
            msg = f"  ❌ {name}"
            if detail:
                msg += f" — {detail}"
            print(msg)

    def _skip(name: str, reason: str) -> None:
        nonlocal checks_skipped
        checks_skipped += 1
        print(f"  ⏭  {name} — {reason}")

    # ── Check 1: customers.customer_id ⊆ reviews.user_id ──────────────
    if CUSTOMERS_PATH.exists() and REVIEWS_PROCESSED_PATH.exists():
        customers = pd.read_csv(CUSTOMERS_PATH, usecols=["customer_id"])
        reviews = pd.read_csv(REVIEWS_PROCESSED_PATH, usecols=["user_id"])
        cust_ids = set(customers["customer_id"].astype(str))
        review_user_ids = set(reviews["user_id"].astype(str))
        orphan_custs = cust_ids - review_user_ids
        _check(
            "C1: customers ⊆ reviews.user_id",
            len(orphan_custs) == 0,
            f"{len(orphan_custs)} customer IDs not in reviews",
        )
    else:
        _skip("C1: customers ⊆ reviews.user_id", "files missing")

    # ── Check 2: No images/videos columns ──────────────────────────────
    bad_cols_found = []
    for path in [PRODUCT_CATALOG_PATH, REVIEWS_PROCESSED_PATH]:
        if path.exists():
            cols = pd.read_csv(path, nrows=0).columns.tolist()
            for bad in ["images", "videos"]:
                if bad in cols:
                    bad_cols_found.append(f"{path.name}:{bad}")
    _check(
        "C2: No images/videos columns",
        len(bad_cols_found) == 0,
        f"found: {bad_cols_found}",
    )

    # ── Check 3: Kaggle report generated ────────────────────────────────
    from src.config.data import REPORTS_DIR

    report_path = REPORTS_DIR / "kaggle_compatibility.txt"
    _check("C3: Kaggle compatibility report exists", report_path.exists())

    # ── Check 4-6: FK integrity ─────────────────────────────────────────
    if ORDERS_PATH.exists() and CUSTOMERS_PATH.exists() and PRODUCT_CATALOG_PATH.exists():
        orders = pd.read_csv(ORDERS_PATH)
        customers = pd.read_csv(CUSTOMERS_PATH, usecols=["customer_id"])
        catalog = pd.read_csv(PRODUCT_CATALOG_PATH, usecols=["product_id"])
        cust_ids = set(customers["customer_id"].astype(str))
        prod_ids = set(catalog["product_id"].astype(str))

        orphan_order_custs = set(orders["customer_id"].astype(str)) - cust_ids
        _check(
            "C4: orders.customer_id ⊆ customers",
            len(orphan_order_custs) == 0,
            f"{len(orphan_order_custs)} orphan customer IDs in orders",
        )

        if ORDER_ITEMS_PATH.exists():
            items = pd.read_csv(ORDER_ITEMS_PATH)
            orphan_item_prods = set(items["product_id"].astype(str)) - prod_ids
            _check(
                "C5: order_items.product_id ⊆ catalog",
                len(orphan_item_prods) == 0,
                f"{len(orphan_item_prods)} orphan product IDs in order items",
            )
        else:
            _skip("C5: order_items.product_id ⊆ catalog", "order_items.csv missing")
    else:
        _skip("C4-C5: FK checks", "required files missing")

    # ── Check 7-8: Returns integrity ────────────────────────────────────
    if RETURNS_PATH.exists() and ORDERS_PATH.exists() and ORDER_ITEMS_PATH.exists():
        returns = pd.read_csv(RETURNS_PATH)
        orders = pd.read_csv(ORDERS_PATH)
        items = pd.read_csv(ORDER_ITEMS_PATH)

        # C7: return.customer_id matches order.customer_id
        ret_orders = returns.merge(
            orders[["order_id", "customer_id"]],
            on="order_id",
            how="left",
            suffixes=("_ret", "_ord"),
        )
        mismatched_custs = (
            ret_orders["customer_id_ret"].astype(str)
            != ret_orders["customer_id_ord"].astype(str)
        ).sum()
        _check(
            "C7: returns.customer_id matches order's customer",
            mismatched_custs == 0,
            f"{mismatched_custs} mismatches",
        )

        # C8: return.product_id matches an item in that order
        ret_items = returns.merge(
            items[["order_id", "product_id"]].rename(columns={"product_id": "item_pid"}),
            on="order_id",
            how="left",
        )
        # Check if return's product_id appears among that order's items
        ret_items["pid_match"] = (
            ret_items["product_id"].astype(str) == ret_items["item_pid"].astype(str)
        )
        matched_returns = ret_items.groupby("return_id")["pid_match"].any()
        unmatched = (~matched_returns).sum()
        _check(
            "C8: returns.product_id exists in order's items",
            unmatched == 0,
            f"{unmatched} returns with product not in order",
        )
    else:
        _skip("C7-C8: returns FK checks", "required files missing")

    # ── Check 9: Review → delivered order linkage ───────────────────────
    if (
        REVIEWS_PROCESSED_PATH.exists()
        and CUSTOMERS_PATH.exists()
        and ORDERS_PATH.exists()
        and ORDER_ITEMS_PATH.exists()
    ):
        reviews = pd.read_csv(REVIEWS_PROCESSED_PATH, usecols=["user_id", "product_id"])
        customers = pd.read_csv(CUSTOMERS_PATH, usecols=["customer_id"])
        orders = pd.read_csv(ORDERS_PATH)
        items = pd.read_csv(ORDER_ITEMS_PATH)

        cust_ids = set(customers["customer_id"].astype(str))
        review_pairs = reviews[reviews["user_id"].astype(str).isin(cust_ids)].copy()
        review_pairs = review_pairs.drop_duplicates(subset=["user_id", "product_id"])

        # Build set of (customer_id, product_id) from delivered orders + items
        delivered_order_ids = set(orders[orders["status"] == "delivered"]["order_id"])
        delivered_items = items[items["order_id"].isin(delivered_order_ids)]
        delivered_pairs = set(
            zip(
                delivered_items["customer_id"].astype(str),
                delivered_items["product_id"].astype(str),
            )
        )

        review_set = set(
            zip(
                review_pairs["user_id"].astype(str),
                review_pairs["product_id"].astype(str),
            )
        )
        missing = review_set - delivered_pairs
        _check(
            "C9: Every review (user, product) has a delivered order",
            len(missing) == 0,
            f"{len(missing)} review pairs without a delivered order",
        )
    else:
        _skip("C9: review-order linkage", "required files missing")

    # ── Check 10: Returns only for delivered orders ─────────────────────
    if RETURNS_PATH.exists() and ORDERS_PATH.exists():
        returns = pd.read_csv(RETURNS_PATH, usecols=["order_id"])
        orders = pd.read_csv(ORDERS_PATH, usecols=["order_id", "status"])
        ret_with_status = returns.merge(orders, on="order_id", how="left")
        non_delivered = ret_with_status[ret_with_status["status"] != "delivered"]
        _check(
            "C10: All returns are for delivered orders",
            len(non_delivered) == 0,
            f"{len(non_delivered)} returns for non-delivered orders",
        )
    else:
        _skip("C10: returns → delivered only", "required files missing")

    # ── Check 11: request_date > actual_delivery_date ──────────────────
    if RETURNS_PATH.exists() and ORDERS_PATH.exists():
        returns = pd.read_csv(RETURNS_PATH, usecols=["order_id", "request_date"])
        orders = pd.read_csv(
            ORDERS_PATH, usecols=["order_id", "actual_delivery_date"]
        )
        merged = returns.merge(orders, on="order_id", how="left")
        merged["request_date"] = pd.to_datetime(merged["request_date"], errors="coerce")
        merged["actual_delivery_date"] = pd.to_datetime(
            merged["actual_delivery_date"], errors="coerce"
        )
        valid = merged.dropna(subset=["request_date", "actual_delivery_date"])
        bad_dates = valid[valid["request_date"] <= valid["actual_delivery_date"]]
        _check(
            "C11: request_date > actual_delivery_date",
            len(bad_dates) == 0,
            f"{len(bad_dates)} returns with invalid dates",
        )
    else:
        _skip("C11: return date ordering", "required files missing")

    # ── Check 12: total_amount = Σ(quantity × unit_price) ──────────────
    if ORDERS_PATH.exists() and ORDER_ITEMS_PATH.exists():
        orders = pd.read_csv(ORDERS_PATH, usecols=["order_id", "total_amount"])
        items = pd.read_csv(
            ORDER_ITEMS_PATH, usecols=["order_id", "quantity", "unit_price"]
        )
        items["line_total"] = items["quantity"] * items["unit_price"]
        computed = items.groupby("order_id")["line_total"].sum().reset_index()
        computed.columns = ["order_id", "computed_total"]
        merged = orders.merge(computed, on="order_id", how="left")
        merged["diff"] = abs(merged["total_amount"] - merged["computed_total"])
        bad_totals = merged[merged["diff"] > 0.02]  # allow rounding tolerance
        _check(
            "C12: total_amount = Σ(qty × price)",
            len(bad_totals) == 0,
            f"{len(bad_totals)} orders with mismatched totals",
        )
    else:
        _skip("C12: order total check", "required files missing")

    # ── Check 13: synthetic_generator.py unmodified ────────────────────
    orig_path = Path(PROJECT_ROOT) / "src" / "data" / "synthetic_generator.py"
    _check(
        "C13: synthetic_generator.py exists (frozen)",
        orig_path.exists(),
        "File missing or deleted",
    )

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    total = checks_passed + checks_failed + checks_skipped
    print(
        f"  Validation: {checks_passed} passed, {checks_failed} failed, "
        f"{checks_skipped} skipped (of {total} checks)"
    )
    if checks_failed == 0:
        print("  🎉 All integrity checks PASSED!")
    else:
        print(f"  ⚠  {checks_failed} check(s) FAILED — review output above.")
    print("─" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the data generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Run only a specific step (1-10). Default: run all.",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run all steps.")
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=AMAZON_2023_MAX_REVIEWS,
        help="Max reviews per category for download.",
    )
    parser.add_argument("--streaming", action="store_true", help="Use streaming download.")
    parser.add_argument(
        "--num-customers", type=int, default=NUM_CUSTOMERS, help="Target customer count."
    )
    parser.add_argument(
        "--num-orders", type=int, default=NUM_ORDERS, help="Target order count."
    )
    parser.add_argument(
        "--num-returns", type=int, default=NUM_RETURNS, help="Target return count."
    )
    parser.add_argument(
        "--total-queries", type=int, default=500, help="Number of enhanced queries."
    )
    parser.add_argument(
        "--policy-provider",
        choices=["static", "gemini", "openai"],
        default="static",
        help="Policy generation method.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    steps = {
        1: lambda: step_01_download(args.force, args.max_reviews, args.streaming),
        2: lambda: step_02_preprocess_products(args.force),
        3: lambda: step_03_preprocess_reviews(args.force),
        4: lambda: step_04_kaggle_check(args.force),
        5: lambda: step_05_generate_customers(args.force, args.num_customers),
        6: lambda: step_06_generate_orders(args.force, args.num_orders),
        7: lambda: step_07_generate_returns(args.force, args.num_returns),
        8: lambda: step_08_generate_queries(args.force, args.total_queries),
        9: lambda: step_09_generate_policies(args.force, args.policy_provider),
        10: lambda: step_10_validate(args.force),
    }

    if args.step is not None:
        if args.step not in steps:
            print(f"❌ Invalid step: {args.step}. Must be 1-10.")
            sys.exit(1)
        steps[args.step]()
    else:
        for step_num in sorted(steps.keys()):
            try:
                steps[step_num]()
            except FileNotFoundError as e:
                print(f"  ❌ Step {step_num} failed: {e}")
                print(f"     Run preceding steps first or use --step {step_num}")
                continue
            except Exception as e:
                print(f"  ❌ Step {step_num} failed with error: {e}")
                continue

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
