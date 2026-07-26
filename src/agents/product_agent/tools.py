"""Tool functions available to the Product Recommendation Agent.

Each tool is decorated with ``@tool`` so LangGraph can bind them
to the LLM's tool-calling interface.

Design change
-------------
The agent previously had exactly one catalog tool — raw text-to-SQL — plus a
~700-word prompt of hand-written ``NOT ILIKE '%case%'`` heuristics whose
stated premise was "there is no category column". That premise is false:
``product_catalog`` has a ``main_category`` column (All Electronics,
Computers, Camera & Photo, Home Audio & Theater, ...) and a
``sub_categories`` column.

So accessory filtering, price bounds, rating floors and result shaping now
live in :func:`search_products` as ordinary parameters resolved in Python.
The model picks arguments instead of authoring defensive SQL, which removes
the single largest source of malformed queries and empty result sets.
``query_products`` is retained for genuinely ad-hoc analytical questions.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import tool

from src.agents.common import limit_rows, reject_write_sql
from src.data.postgresql import execute_sql_query, execute_sql_query_params
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

# Raw SQL escape hatch keeps a tight row cap for token safety.
_MAX_RESULT_ROWS = 8

# search_products projects a narrow column set, so it can afford more rows.
# Five rows is not enough to recommend from.
_MAX_SEARCH_ROWS = 10
_ABS_MAX_SEARCH_ROWS = 20

# Words that almost always indicate a peripheral rather than the main device.
_ACCESSORY_TERMS: tuple[str, ...] = (
    "case", "cases", "cover", "sleeve", "pouch", "bag", "backpack",
    "charger", "charging", "adapter", "adaptor", "cable", "cord", "plug",
    "stand", "mount", "holder", "dock", "docking", "tripod", "clip",
    "screen protector", "tempered glass", "skin", "decal", "sticker",
    "replacement", "spare", "refill", "cleaning kit", "cleaner",
    "strap", "lanyard", "stylus",
    "extension", "splitter", "converter", "hub",
)

# Phrases that grammatically subordinate the item to another device,
# e.g. "Hard Case for MacBook Pro" or "Charger Compatible With Galaxy S21".
_SUBORDINATE_PATTERNS: tuple[str, ...] = (
    " for ", "compatible with", "fits ", "replacement for", "designed for",
)

# Sensible price floors for core hardware, applied only when the caller asks
# for accessory exclusion and supplies no explicit minimum.
_CATEGORY_PRICE_FLOOR: dict[str, float] = {
    "laptop": 200.0,
    "notebook": 200.0,
    "macbook": 400.0,
    "desktop": 200.0,
    "computer": 200.0,
    "smartphone": 100.0,
    "phone": 100.0,
    "iphone": 150.0,
    "tablet": 80.0,
    "ipad": 150.0,
    "tv": 150.0,
    "television": 150.0,
    "monitor": 70.0,
    "camera": 100.0,
    "dslr": 250.0,
    "printer": 50.0,
    "console": 100.0,
    "refrigerator": 300.0,
    "washing machine": 250.0,
    "headphone": 20.0,
    "headphones": 20.0,
    "earbuds": 15.0,
    "speaker": 20.0,
    "smartwatch": 40.0,
}


def _looks_like_accessory(title: str) -> bool:
    """Heuristically decide whether a catalog title is a peripheral."""
    lowered = f" {title.lower()} "

    for term in _ACCESSORY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return True

    for pattern in _SUBORDINATE_PATTERNS:
        if pattern in lowered:
            return True

    return False


def _infer_price_floor(query: str) -> float:
    """Return an implicit minimum price for core-hardware queries."""
    lowered = query.lower()
    floors = [
        floor
        for keyword, floor in _CATEGORY_PRICE_FLOOR.items()
        if re.search(rf"\b{re.escape(keyword)}\b", lowered)
    ]
    return max(floors) if floors else 0.0


@tool
def search_products(
    query: str,
    min_price: float = 0.0,
    max_price: float = 0.0,
    min_rating: float = 0.0,
    category: str = "",
    exclude_accessories: bool = True,
    bestsellers_only: bool = False,
    sort_by: str = "relevance",
    limit: int = _MAX_SEARCH_ROWS,
) -> str:
    """Search the product catalog by keywords with structured filters.

    This is the PREFERRED tool for finding products. Prefer it over
    query_products for any normal "find me a ..." request — it already
    handles accessory exclusion, price bounds and rating floors for you.

    Args:
        query: Product keywords, e.g. "wireless noise cancelling headphones".
        min_price: Minimum price. 0 means no minimum.
        max_price: Maximum price. 0 means no maximum.
        min_rating: Minimum average rating out of 5. 0 means no minimum.
        category: Optional main_category filter, e.g. "Computers",
            "All Electronics", "Camera & Photo", "Home Audio & Theater".
        exclude_accessories: When True (default), filters out cases, chargers,
            cables, mounts and similar peripherals so that searching for a
            "laptop" returns laptops rather than laptop sleeves.
        bestsellers_only: Restrict to items flagged as bestsellers.
        sort_by: One of "relevance", "price_asc", "price_desc", "rating".
        limit: Maximum products to return (1-20).

    Returns:
        JSON list of matching products with id, title, price, rating,
        rating_count, category and store.
    """
    cleaned = query.strip()
    if not cleaned:
        return "Please provide product keywords to search for."

    bounded_limit = max(1, min(int(limit), _ABS_MAX_SEARCH_ROWS))

    terms = [t for t in re.findall(r"[A-Za-z0-9\-]+", cleaned) if len(t) > 2][:6]
    if not terms:
        terms = [cleaned]

    where: list[str] = ["price IS NOT NULL", "price > 0"]
    params: list[Any] = []

    # Every keyword must appear somewhere in title/description (AND semantics
    # narrows far better than the OR the model tended to write).
    for term in terms:
        where.append("(title ILIKE %s OR description ILIKE %s)")
        params.extend([f"%{term}%", f"%{term}%"])

    effective_min = float(min_price)
    if exclude_accessories and effective_min <= 0:
        effective_min = _infer_price_floor(cleaned)

    if effective_min > 0:
        where.append("price >= %s")
        params.append(effective_min)
    if float(max_price) > 0:
        where.append("price <= %s")
        params.append(float(max_price))
    if float(min_rating) > 0:
        where.append("average_rating >= %s")
        params.append(float(min_rating))
    if category.strip():
        where.append("main_category ILIKE %s")
        params.append(f"%{category.strip()}%")
    if bestsellers_only:
        where.append("is_bestseller IS TRUE")

    order_clause = {
        "price_asc": "price ASC",
        "price_desc": "price DESC",
        "rating": "average_rating DESC NULLS LAST, rating_count DESC",
    }.get(sort_by, "(average_rating * LN(GREATEST(rating_count, 1) + 1)) DESC NULLS LAST")

    # Over-fetch so Python-side accessory filtering still leaves enough rows.
    fetch_limit = bounded_limit * 4 if exclude_accessories else bounded_limit
    params.append(fetch_limit)

    # Include a short feature/description snippet. Without it the model had to
    # follow every search with N get_product_details calls just to say *why* a
    # product is a good fit — which tripled the ReAct steps (and the latency)
    # of a routine "show me headphones under $250" turn.
    sql = f"""
        SELECT product_id, title, price, average_rating, rating_count,
               main_category, store, is_bestseller,
               LEFT(COALESCE(NULLIF(features, '[]'), description), 350) AS snippet
        FROM product_catalog
        WHERE {' AND '.join(where)}
        ORDER BY {order_clause}
        LIMIT %s
    """

    rows = execute_sql_query_params(sql, tuple(params))
    if isinstance(rows, str):
        _LOGGER.warning("search_products failed: %s", rows)
        return rows
    if not rows:
        return (
            f"No products matched '{cleaned}'"
            + (f" under {max_price}" if max_price else "")
            + ". Try broader keywords, a wider price range, or set "
            "exclude_accessories=false."
        )

    if exclude_accessories:
        kept = [row for row in rows if not _looks_like_accessory(str(row.get("title", "")))]
        # If filtering removed everything, the query was probably *for* an
        # accessory — return the unfiltered results with a note rather than
        # a dead end.
        if not kept:
            payload = json.dumps(rows[:bounded_limit], indent=2, default=str)
            return (
                "Only accessory-type products matched this search. "
                "The customer may have been asking about an accessory:\n" + payload
            )
        rows = kept

    rows = rows[:bounded_limit]
    return json.dumps(rows, indent=2, default=str)


@tool
def get_product_details(product_id: str) -> str:
    """Fetch full details for one product by its catalog ID (ASIN).

    Use after search_products when the customer asks for specifics about a
    particular item — features, full description, store, rating breakdown.

    Args:
        product_id: The product_id / ASIN, e.g. "B00MCW7G9M".

    Returns:
        JSON object with the product's full record, or a not-found message.
    """
    cleaned = product_id.strip()
    if not cleaned:
        return "Please provide a product_id."

    rows = execute_sql_query_params(
        """
        SELECT product_id, title, main_category, sub_categories, price,
               average_rating, rating_count, features, description, store,
               is_bestseller
        FROM product_catalog
        WHERE product_id = %s
        """,
        (cleaned,),
    )
    if isinstance(rows, str):
        return rows
    if not rows:
        return f"No product found with product_id={cleaned}."

    record = dict(rows[0])
    # These columns hold stringified Python lists that can run to thousands of
    # characters. Trim so one lookup cannot blow the context window.
    for field in ("description", "features"):
        value = str(record.get(field) or "")
        if len(value) > 1200:
            record[field] = value[:1200] + " ...(truncated)"

    return json.dumps(record, indent=2, default=str)


@tool
def list_product_categories() -> str:
    """List the available main_category values in the product catalog.

    Use this when you are unsure which category filter to pass to
    search_products.

    Returns:
        JSON list of category names with product counts.
    """
    rows = execute_sql_query(
        """
        SELECT main_category, COUNT(*) AS product_count
        FROM product_catalog
        WHERE main_category IS NOT NULL
        GROUP BY main_category
        ORDER BY product_count DESC
        LIMIT 40
        """
    )
    if isinstance(rows, str):
        return rows
    if not rows:
        return "No categories found."
    return json.dumps(rows, indent=2, default=str)


@tool
def query_products(sql_query: str) -> str:
    """Execute a read-only SQL SELECT query against the product_catalog table.

    ESCAPE HATCH — prefer search_products for normal product lookups. Use this
    only for analytical questions that structured filters cannot express, such
    as aggregations, grouping, or cross-category comparisons.

    Args:
        sql_query: A valid SQL SELECT statement targeting product_catalog.

    Returns:
        JSON-formatted results, or an error/safety message.
    """
    allowed, error = reject_write_sql(sql_query)
    if not allowed:
        return error

    result: list[dict[str, Any]] | str = execute_sql_query(sql_query)

    if isinstance(result, str):
        return result

    if not result:
        return "No results found for the given query."

    result, footer = limit_rows(result, max_rows=_MAX_RESULT_ROWS)

    return json.dumps(result, indent=2, default=str) + footer


@tool
def search_product_reviews(query: str, product_id: str = "", top_k: int = 5) -> str:
    """Search indexed product review snippets using vector retrieval.

    NOTE: this requires a configured Pinecone index. When retrieval is not
    available the tool says so plainly instead of raising, so do not retry it
    and do not treat the message as review content.

    Args:
        query: Review-related natural language query.
        product_id: Optional product ID (ASIN) to narrow search.
        top_k: Number of matches to return.

    Returns:
        Matching review snippets, or an unavailability notice.
    """
    try:
        from src.rag.retriever import format_matches, get_retriever

        retriever = get_retriever()
        pid = product_id.strip() or None
        matches = retriever.search_reviews(query=query, product_id=pid, top_k=top_k)
        return format_matches(matches)
    except Exception as exc:
        _LOGGER.debug("Review retrieval unavailable: %s", exc)
        # Previously returned "Review retrieval error: PINECONE_API_KEY is not
        # set." — the model then tried to interpret that string as review data
        # or retried it in a loop. Be explicit about what to do instead.
        return (
            "Review search is not configured in this environment. "
            "Do not retry this tool. Use the average_rating and rating_count "
            "fields from search_products as the rating evidence instead, and "
            "tell the customer you do not have individual review text."
        )
