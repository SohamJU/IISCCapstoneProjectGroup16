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


def upload_dataframe_to_postgresql_db(df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
    """
    Upload a pandas DataFrame to a PostgreSQL database.

    Args:
        df: The pandas DataFrame to upload.
        table_name: The name of the target SQL table.
        if_exists: How to handle existing data in the table. Options are 'fail', 'replace', or 'append'.
    """
    # Create a SQLAlchemy engine using the connection string
    sql_engine = get_db_engine()

    # Upload DataFrame to PostgreSQL
    df.to_sql(
        name=table_name,            # Name of the target SQL table
        con=sql_engine,             # Database connection engine
        if_exists=if_exists, # type: ignore[arg-type] # Drops and recreates the table if it exists
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