"""Test PostgreSQL queries against uploaded datasets.

Usage:
    python -m src.data.pipeline.test_postgres
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.data import POSTGRESQL_TABLES, POSTGRESQL_AIVEN_PASSWORD
from src.data.postgresql import execute_sql_query


def run() -> None:
    """Run diagnostic queries against the PostgreSQL database."""
    if not POSTGRESQL_AIVEN_PASSWORD:
        print("  ❌ [skip] PostgreSQL password not found in .env. Skipping database tests.")
        return

    print("  Running diagnostic queries on Aiven PostgreSQL...")

    # Test 1: Row Counts
    print("\n  [Test 1] Row Counts for all tables:")
    for table_name in POSTGRESQL_TABLES.values():
        query = f"SELECT COUNT(*) as count FROM {table_name};"
        result = execute_sql_query(query)
        if isinstance(result, str) and result.startswith("Error"):
            print(f"    ❌ {table_name}: {result}")
        elif isinstance(result, list) and len(result) > 0:
            count = result[0].get('count', 0)
            print(f"    ✅ {table_name}: {count} rows")
        else:
            print(f"    ⚠️ {table_name}: No data returned")

    # Test 2: Sample Data Fetch
    print("\n  [Test 2] Sample Data Fetch (customers):")
    sample_query = "SELECT customer_id, first_name, last_name, loyalty_tier FROM customers LIMIT 3;"
    sample_res = execute_sql_query(sample_query)
    if isinstance(sample_res, str) and sample_res.startswith("Error"):
         print(f"    ❌ Error: {sample_res}")
    else:
        for i, row in enumerate(sample_res, 1):
             print(f"    ✅ Row {i}: {row}")

    # Test 3: Relational Join
    print("\n  [Test 3] Relational Join Test (orders + order_items):")
    join_query = """
        SELECT o.order_id, o.customer_id, o.total_amount, i.product_id, i.quantity 
        FROM orders o 
        JOIN order_items i ON o.order_id = i.order_id 
        LIMIT 3;
    """
    join_res = execute_sql_query(join_query)
    if isinstance(join_res, str) and join_res.startswith("Error"):
        print(f"    ❌ Join Error: {join_res}")
    else:
        for i, row in enumerate(join_res, 1):
             print(f"    ✅ Joined Row {i}: {row}")

    print("\n  🎉 PostgreSQL Diagnostics Complete.")


if __name__ == "__main__":
    run()
