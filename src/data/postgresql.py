from typing import Any, Sequence

import psycopg2
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from src.config.data import POSTGRESQL_CONNECTION_STRING


def get_db_engine() -> Engine:
    """
    Get a SQLAlchemy engine connected to the PostgreSQL database.
    Useful for Pandas or Agent connections.
    """
    return create_engine(POSTGRESQL_CONNECTION_STRING)


#: Values pandas' ``to_sql`` actually accepts. Anything else (notably the
#: config's ``"skip"`` sentinel) must be resolved by the caller before it
#: reaches this layer.
VALID_IF_EXISTS = frozenset({"fail", "append", "replace"})


class DestructiveWriteNotConfirmed(RuntimeError):
    """Raised when a table-dropping write is attempted without explicit opt-in."""


def upload_dataframe_to_postgresql_db(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "fail",
    *,
    confirm_destructive: bool = False,
) -> None:
    """
    Upload a pandas DataFrame to a PostgreSQL database.

    Args:
        df: The pandas DataFrame to upload.
        table_name: The name of the target SQL table.
        if_exists: How to handle an existing table. One of 'fail', 'append',
            'replace'. Defaults to 'fail' — see the safety note below.
        confirm_destructive: Must be True to allow ``if_exists="replace"``,
            which DROPS the existing table.

    Raises:
        ValueError: If ``if_exists`` is not a value pandas understands.
        DestructiveWriteNotConfirmed: If a 'replace' was requested without
            ``confirm_destructive=True``.

    Safety note:
        This default used to be ``"replace"``, which drops and recreates the
        target table. This project points at a **shared** Aiven database used
        by the whole team, so any accidental call — a stray import, a notebook
        cell, a pipeline re-run — silently destroyed everyone's data. The
        default is now the non-destructive, non-duplicating 'fail', and
        dropping a table requires opting in twice: ``if_exists="replace"``
        *and* ``confirm_destructive=True``.

        'append' is deliberately not the default either: re-running a pipeline
        would silently duplicate every row, which is harder to notice than an
        outright error.
    """
    if if_exists not in VALID_IF_EXISTS:
        raise ValueError(
            f"if_exists must be one of {sorted(VALID_IF_EXISTS)}, got {if_exists!r}. "
            "(The 'skip' sentinel is a pipeline-level concept — resolve it before "
            "calling this function.)"
        )

    if if_exists == "replace" and not confirm_destructive:
        raise DestructiveWriteNotConfirmed(
            f"Refusing to replace table '{table_name}': this DROPS the existing "
            f"table on a database shared with the whole team. If that is genuinely "
            f"what you want, pass confirm_destructive=True."
        )

    # Create a SQLAlchemy engine using the connection string
    sql_engine = get_db_engine()

    # Upload DataFrame to PostgreSQL
    df.to_sql(
        name=table_name,            # Name of the target SQL table
        con=sql_engine,             # Database connection engine
        if_exists=if_exists,        # type: ignore[arg-type]
        index=False                 # Prevents pandas index from becoming a column
    )


def execute_sql_query(
    query: str
) -> list[dict[str, object]] | str:
    """
    Execute a SQL query against the PostgreSQL database and return the results.

    Args:
        query: SQL query string to execute.

    Returns:
        A list of dictionaries representing rows for SELECT queries,
        or an error message string if execution fails.
    """
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        # Create a cursor to perform database operations
        cur = conn.cursor()
        # Execute the query
        cur.execute(query)
        # Fetch all results
        columns = [desc[0] for desc in cur.description]  # type: ignore[union-attr]
        results = cur.fetchall()
        # Close communication with the database
        cur.close()
        conn.close()
        # Return as a list of dictionaries for easier handling by the LLM
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        return f"Error executing query: {e}"


def execute_sql_query_params(
    query: str,
    params: Sequence[object] | None = None,
) -> list[dict[str, object]] | str:
    """Execute a parameterized SQL query and return rows.

    Args:
        query: SQL query string to execute.
        params: Query parameter values.

    Returns:
        A list of dictionaries for SELECT-like queries,
        or an error message string if execution fails.
    """
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cur = conn.cursor()
        cur.execute(query, params)

        if cur.description is None:
            cur.close()
            conn.close()
            return []

        columns = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        return f"Error executing query: {e}"


def execute_sql_write(
    query: str,
    params: Sequence[object] | None = None,
    *,
    fetch_one: bool = False,
) -> dict[str, object] | str:
    """Execute a parameterized write query safely.

    Args:
        query: SQL write query (INSERT/UPDATE/DELETE) to execute.
        params: Query parameter values.
        fetch_one: Whether to return one row (useful with RETURNING).

    Returns:
        A dictionary with rowcount and optional returned row,
        or an error message string if execution fails.
    """
    try:
        conn = psycopg2.connect(POSTGRESQL_CONNECTION_STRING)
        cur = conn.cursor()
        cur.execute(query, params)

        payload: dict[str, object] = {"rowcount": cur.rowcount}
        if fetch_one and cur.description is not None:
            row = cur.fetchone()
            if row is not None:
                columns = [desc[0] for desc in cur.description]
                payload["row"] = dict(zip(columns, row))

        conn.commit()
        cur.close()
        conn.close()
        return payload
    except Exception as e:
        return f"Error executing write query: {e}"