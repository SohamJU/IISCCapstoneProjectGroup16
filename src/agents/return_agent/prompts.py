"""Prompt templates for the Return Agent."""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.return_agent.config import RETURN_SCHEMA_PATHS

_ROLE_BLOCK = """\
You are an expert Return Support Agent for an e-commerce system. Your job is to answer user questions about returns, refunds, replacements, cancellations, and return status.

You can query the PostgreSQL database using the query_returns tool. Only use SQL when the user asks for factual return or refund details.

Always answer in clear natural language. The SQL query should be used only to retrieve data and must not be exposed directly in the final answer.

Use the schema information below to construct valid SQL queries for the returns, orders, and order_items tables.
"""

_SCHEMA_TEMPLATE = """\
## Available Tables
{schema_text}

Use these columns when building queries. Return records are recorded in the `returns` table while order metadata is available in `orders` and `order_items`.
"""


def _format_schema_paths(schema_paths: list[Path]) -> str:
    sections: list[str] = []
    for schema_path in schema_paths:
        if not schema_path.exists():
            continue

        with open(schema_path, "r", encoding="utf-8") as handle:
            schema = json.load(handle)

        table_name = schema_path.name.removesuffix(".schema.json")
        column_descriptions = []
        for column in schema.get("columns", []):
            name = column.get("name", "?")
            dtype = column.get("type", "?")
            description = column.get("description", "")
            if description:
                column_descriptions.append(f"{name} ({dtype}): {description}")
            else:
                column_descriptions.append(f"{name} ({dtype})")

        sections.append(f"- **{table_name}**: {', '.join(column_descriptions)}")

    return "\n".join(sections)


def build_system_prompt() -> str:
    schema_text = _format_schema_paths(RETURN_SCHEMA_PATHS)
    return "\n\n".join([_ROLE_BLOCK, _SCHEMA_TEMPLATE.format(schema_text=schema_text)])
