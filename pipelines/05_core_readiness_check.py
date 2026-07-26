"""Core POC readiness checks.

Checks:
1. Required environment variables
2. Required local artifacts
3. PostgreSQL connectivity and table presence
4. Optional Pinecone config presence (warning only)

Usage:
    python pipelines/05_core_readiness_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import psycopg2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _print_check(ok: bool, label: str, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f" - {detail}"
    print(line)


def _require_env(name: str) -> tuple[bool, str]:
    value = os.getenv(name, "").strip()
    return (bool(value), "set" if value else "missing")


def _check_paths(paths: Iterable[Path]) -> tuple[bool, list[str]]:
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in paths if not p.exists()]
    return (len(missing) == 0, missing)


def _build_pg_conn_string() -> str | None:
    host = os.getenv("POSTGRESQL_HOST", "").strip()
    port = os.getenv("POSTGRESQL_PORT", "").strip()
    user = os.getenv("POSTGRESQL_USER", "").strip()
    db = os.getenv("POSTGRESQL_DB", "").strip()
    pwd = os.getenv("POSTGRESQL_AIVEN_PASSWORD", "").strip()

    if not all([host, port, user, db, pwd]):
        return None
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}?sslmode=require"


def _check_db_tables(conn_string: str, tables: list[str]) -> tuple[bool, list[str], str]:
    try:
        conn = psycopg2.connect(conn_string)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        _ = cur.fetchone()

        missing: list[str] = []
        for table in tables:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """,
                (table,),
            )
            row = cur.fetchone()
            exists = bool(row[0]) if row is not None else False
            if not exists:
                missing.append(table)

        cur.close()
        conn.close()
        return (len(missing) == 0, missing, "connected")
    except Exception as exc:
        return (False, tables, str(exc))


def main() -> None:
    print("Core POC Readiness Check")
    print("=" * 72)

    failures = 0

    required_env = ["POSTGRESQL_AIVEN_PASSWORD"]
    for env_name in required_env:
        ok, detail = _require_env(env_name)
        _print_check(ok, f"Env {env_name}", detail)
        if not ok:
            failures += 1

    groq_ok, groq_detail = _require_env("GROQ_API_KEY")
    if groq_ok:
        _print_check(True, "Env GROQ_API_KEY", groq_detail)
    else:
        _print_check(
            True,
            "Env GROQ_API_KEY",
            "missing (deterministic fallback mode can still run core POC)",
        )

    # Pinecone is required for full RAG, but we only warn here to keep POC flexible.
    for env_name in ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]:
        ok, detail = _require_env(env_name)
        _print_check(ok, f"Env {env_name}", detail)

    required_paths = [
        PROJECT_ROOT / "data" / "knowledge_base" / "return_policy.md",
        PROJECT_ROOT / "data" / "knowledge_base" / "shipping_policy.md",
        PROJECT_ROOT / "data" / "knowledge_base" / "payment_policy.md",
        PROJECT_ROOT / "data" / "processed" / "product_catalog.schema.json",
    ]
    paths_ok, missing_paths = _check_paths(required_paths)
    _print_check(paths_ok, "Required local artifacts", ", ".join(missing_paths) if missing_paths else "all present")
    if not paths_ok:
        failures += 1

    conn_string = _build_pg_conn_string()
    if conn_string is None:
        _print_check(False, "PostgreSQL connection config", "missing one or more POSTGRESQL_* variables")
        failures += 1
    else:
        required_tables = [
            "customers",
            "orders",
            "order_items",
            "returns",
            "product_catalog",
            "reviews",
        ]
        db_ok, missing_tables, detail = _check_db_tables(conn_string, required_tables)
        _print_check(db_ok, "PostgreSQL connectivity + required tables", detail if db_ok else f"{detail}; missing: {missing_tables}")
        if not db_ok:
            failures += 1

    print("-" * 72)
    if failures == 0:
        print("Readiness check passed. Core POC can be executed.")
        sys.exit(0)

    print(f"Readiness check failed with {failures} blocking issue(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()
