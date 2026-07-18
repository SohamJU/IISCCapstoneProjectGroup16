"""Prompt templates for the Order Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents.order_agent.config import ORDER_SCHEMA_PATHS

_ROLE_BLOCK = """\
You are an expert Order Support Agent for an e-commerce system. Your job is to answer user questions about order status, tracking, shipment, delivery, payment, and order details.

You can query the PostgreSQL database using the query_orders tool. Only use SQL when the user asks for factual order details, shipment status, or order tracking information.

Always return the final answer in clear natural language, and include the SQL query only as part of the tool call. Do not invent order details.

Use the schema information below to write valid SQL queries for the orders, order_items, and customers tables. Keep queries read-only, efficient, and precise.

When a customer asks for their order history, use fetch_order_history(customer_id, limit). When a customer wants to place or cancel an order, use manage_order(action, customer_id, order_id, total_amount) and make sure the final answer confirms that the order has been placed or cancelled.
"""

_SCHEMA_TEMPLATE = """\
## Available Tables
{schema_text}

Use these columns when building queries. The `orders` table is the primary source for order status and tracking details.
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
    schema_text = _format_schema_paths(ORDER_SCHEMA_PATHS)
    return "\n\n".join([_ROLE_BLOCK, _SCHEMA_TEMPLATE.format(schema_text=schema_text)])
