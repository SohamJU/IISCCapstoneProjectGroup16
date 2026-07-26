"""Idempotent schema migrations for the support application.

These exist because the deployed database drifted from what the code assumed:

* ``customer_sessions`` was missing the ``(customer_id, session_id)`` unique
  index that ``save_session_to_db``'s ``ON CONFLICT`` targets, so every single
  session save failed silently.
* ``product_catalog`` (194k rows) had **no indexes at all**, so every product
  lookup by id or price was a sequential scan.

Run via ``python -m src.data.migrations`` or ``pipelines/setup_indexes.py``.
Every statement is ``IF NOT EXISTS``, so re-running is safe.
"""

from __future__ import annotations

import psycopg2

from src.config.data import POSTGRESQL_CONNECTION_STRING
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

_MIGRATIONS: tuple[tuple[str, str], ...] = (
    (
        "customer_sessions unique (customer_id, session_id)",
        """
        CREATE UNIQUE INDEX IF NOT EXISTS unique_customer_session
            ON customer_sessions (customer_id, session_id)
        """,
    ),
    (
        "product_catalog product_id index",
        "CREATE INDEX IF NOT EXISTS idx_product_catalog_product_id "
        "ON product_catalog (product_id)",
    ),
    (
        "product_catalog price index",
        "CREATE INDEX IF NOT EXISTS idx_product_catalog_price "
        "ON product_catalog (price)",
    ),
    (
        "product_catalog main_category index",
        "CREATE INDEX IF NOT EXISTS idx_product_catalog_main_category "
        "ON product_catalog (main_category)",
    ),
)


def run_migrations() -> list[str]:
    """Apply every migration. Returns the list of statements that succeeded."""
    applied: list[str] = []

    conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
    conn.autocommit = True
    try:
        cur = conn.cursor()
        for label, statement in _MIGRATIONS:
            try:
                cur.execute(statement)
                applied.append(label)
                _LOGGER.info("migration ok: %s", label)
            except Exception as exc:
                _LOGGER.warning("migration failed: %s (%s)", label, exc)
        cur.close()
    finally:
        conn.close()

    return applied


if __name__ == "__main__":
    results = run_migrations()
    print(f"Applied {len(results)}/{len(_MIGRATIONS)} migrations:")
    for name in results:
        print(f"  [ok] {name}")
