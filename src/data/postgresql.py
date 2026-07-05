from typing import Any

import psycopg2  # type: ignore[import-untyped]
from sqlalchemy import create_engine
from src.config.data import POSTGRESQL_CONNECTION_STRING


def get_db_engine():
    """
    Get a SQLAlchemy engine connected to the PostgreSQL database.
    Useful for Pandas or Agent connections.
    """
    return create_engine(POSTGRESQL_CONNECTION_STRING)


def upload_dataframe_to_postgresql_db(df, table_name: str, if_exists: str = "replace"):
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
        if_exists=if_exists,        # Drops and recreates the table if it exists
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
        columns = [desc[0] for desc in cur.description]
        results = cur.fetchall()
        # Close communication with the database
        cur.close()
        conn.close()
        # Return as a list of dictionaries for easier handling by the LLM
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        return f"Error executing query: {e}"