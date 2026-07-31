"""Make a handful of orders recent so the happy-path return demo works.

Why this is needed
------------------
The synthetic orders are dated 2004-2026 but almost all are years old, and the
return window is 30 days. So *every* return request correctly comes back
"outside the return window" — the agent is behaving properly, but you can never
demonstrate a successful return.

Only 5 delivered orders currently fall inside the window, and each one is
consumed the moment you demo a return against it (the item flips to
``returned``).

What this does
--------------
Rewrites ``order_date`` / ``est_delivery_date`` / ``actual_delivery_date`` for
a small set of delivered orders so they land a few days ago. It picks a mix of
single-item and multi-item orders so both return flows can be demonstrated:

* single-item  -> agent resolves the item itself and proceeds
* multi-item   -> agent asks which product by name

Safety
------
This writes to a **shared** team database, so:

* dry-run by default — nothing is written without ``--apply``
* original values are copied to ``demo_date_backup`` before any update
* ``--revert`` restores them exactly
* only three date columns are ever touched; no table is dropped or recreated

Usage::

    python scripts/prepare_demo_data.py              # preview only
    python scripts/prepare_demo_data.py --apply      # make the changes
    python scripts/prepare_demo_data.py --revert     # undo them
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv()

from src.data.postgresql import (
    execute_sql_query,
    execute_sql_query_params,
    execute_sql_write,
)

BACKUP_TABLE = "demo_date_backup"

# Spread the demo orders across the window so you can also show an order that
# is close to expiring. Days-ago values, cycled over the chosen orders.
_DAYS_AGO_CYCLE = (2, 5, 9, 14, 21, 27)


def _rows(result) -> list[dict]:
    """Normalise the query helpers' ``list | str`` return type."""
    if isinstance(result, str):
        raise RuntimeError(f"Query failed: {result}")
    return result or []


def ensure_backup_table() -> None:
    """Create the backup table if it does not exist."""
    execute_sql_write(
        f"""
        CREATE TABLE IF NOT EXISTS {BACKUP_TABLE} (
            order_id TEXT PRIMARY KEY,
            order_date TEXT,
            est_delivery_date TEXT,
            actual_delivery_date TEXT,
            backed_up_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def pick_candidates(count: int) -> list[dict]:
    """Choose delivered orders that still have returnable items.

    Prefers a mix of single-item and multi-item orders, and orders belonging to
    customers with a real name so they are easy to find in the Gradio dropdown.
    """
    half = max(1, count // 2)

    query = """
        SELECT o.order_id,
               o.customer_id,
               c.first_name || ' ' || c.last_name AS customer_name,
               o.order_date,
               COUNT(oi.order_item_id) AS item_count
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.order_id
        JOIN customers c ON c.customer_id = o.customer_id
        WHERE o.status = 'delivered'
          AND oi.item_status NOT IN ('cancelled', 'returned')
          AND o.order_id NOT IN (SELECT order_id FROM returns)
          AND o.order_date::date < CURRENT_DATE - 30
        GROUP BY o.order_id, o.customer_id, customer_name, o.order_date
        HAVING COUNT(oi.order_item_id) %s
        ORDER BY o.order_id
        LIMIT %s
    """

    singles = _rows(execute_sql_query_params(query % ("= 1", "%s"), (half,)))
    multis = _rows(execute_sql_query_params(query % (">= 2", "%s"), (count - half,)))
    return singles + multis


def apply_changes(candidates: list[dict], dry_run: bool) -> None:
    """Back up and rewrite the date columns for the chosen orders."""
    today = date.today()

    for index, row in enumerate(candidates):
        order_id = str(row["order_id"])
        days_ago = _DAYS_AGO_CYCLE[index % len(_DAYS_AGO_CYCLE)]
        new_order_date = today - timedelta(days=days_ago)
        # Delivered a couple of days after ordering, but never in the future.
        new_delivery = min(new_order_date + timedelta(days=2), today)

        label = (
            f"  {order_id}  {str(row.get('customer_name'))[:24]:<24} "
            f"{row['item_count']} item(s)  {row['order_date']} -> {new_order_date} "
            f"({days_ago}d ago)"
        )

        if dry_run:
            print(label)
            continue

        # Snapshot the originals exactly once — re-running must not overwrite
        # a real historical date with an already-shifted demo date.
        execute_sql_write(
            f"""
            INSERT INTO {BACKUP_TABLE}
                (order_id, order_date, est_delivery_date, actual_delivery_date)
            SELECT order_id, order_date, est_delivery_date, actual_delivery_date
            FROM orders WHERE order_id = %s
            ON CONFLICT (order_id) DO NOTHING
            """,
            (order_id,),
        )

        execute_sql_write(
            """
            UPDATE orders
            SET order_date = %s,
                est_delivery_date = %s,
                actual_delivery_date = %s
            WHERE order_id = %s
            """,
            (
                new_order_date.isoformat(),
                new_delivery.isoformat(),
                new_delivery.isoformat(),
                order_id,
            ),
        )
        print(label + "  [applied]")


def revert() -> int:
    """Restore every order recorded in the backup table."""
    backed_up = _rows(execute_sql_query(f"SELECT order_id FROM {BACKUP_TABLE}"))
    if not backed_up:
        print("Nothing to revert — backup table is empty or absent.")
        return 0

    execute_sql_write(
        f"""
        UPDATE orders o
        SET order_date = b.order_date,
            est_delivery_date = b.est_delivery_date,
            actual_delivery_date = b.actual_delivery_date
        FROM {BACKUP_TABLE} b
        WHERE o.order_id = b.order_id
        """
    )
    execute_sql_write(f"DELETE FROM {BACKUP_TABLE}")
    print(f"Reverted {len(backed_up)} order(s) to their original dates.")
    return len(backed_up)


def show_demo_ready() -> None:
    """Print every order currently inside the return window."""
    rows = _rows(
        execute_sql_query(
            """
            SELECT o.order_id,
                   c.first_name || ' ' || c.last_name AS customer_name,
                   o.customer_id,
                   o.order_date,
                   COUNT(oi.order_item_id) AS items
            FROM orders o
            JOIN order_items oi ON oi.order_id = o.order_id
            JOIN customers c ON c.customer_id = o.customer_id
            WHERE o.status = 'delivered'
              AND oi.item_status NOT IN ('cancelled', 'returned')
              AND o.order_date::date >= CURRENT_DATE - 30
            GROUP BY o.order_id, customer_name, o.customer_id, o.order_date
            ORDER BY o.order_date DESC
            """
        )
    )

    print(f"\n{'=' * 78}\nRETURNABLE ORDERS (inside the 30-day window)\n{'=' * 78}")
    if not rows:
        print("  none — run with --apply")
        return
    for r in rows:
        kind = "single-item" if r["items"] == 1 else f"{r['items']}-item"
        print(
            f"  {r['order_id']}  {str(r['customer_name'])[:26]:<26} {kind:<12} {r['order_date']}"
        )
        print(f"      customer_id: {r['customer_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shift a few orders into the return window for demos."
    )
    parser.add_argument("--apply", action="store_true", help="Write the changes.")
    parser.add_argument("--revert", action="store_true", help="Restore original dates.")
    parser.add_argument(
        "--count", type=int, default=6, help="Orders to shift (default 6)."
    )
    args = parser.parse_args()

    if args.revert:
        ensure_backup_table()
        revert()
        show_demo_ready()
        return 0

    ensure_backup_table()
    candidates = pick_candidates(args.count)

    if not candidates:
        print("No suitable candidate orders found.")
        show_demo_ready()
        return 1

    mode = "APPLYING" if args.apply else "DRY RUN (pass --apply to write)"
    print(f"{mode} — shifting {len(candidates)} order(s) into the return window:\n")
    apply_changes(candidates, dry_run=not args.apply)

    if args.apply:
        show_demo_ready()
    else:
        print("\nNothing written. Re-run with --apply.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
