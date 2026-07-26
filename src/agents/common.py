"""Shared guardrail and SQL safety helpers for agent packages."""

from __future__ import annotations

import re
from typing import Any

from src.agents.guardrails.ml_input_scanner import scan_input as _ml_scan

_WRITE_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# Fast-path regex: catches common injection phrases without any ML cost.
# Deliberately broad so that obvious variants are blocked before the ML
# scanner runs.  The ML scanner (opt-in via ENABLE_ML_GUARDRAIL=true) then
# catches obfuscated / paraphrased variants that regex misses.
#
# Design choices:
#   - (ignore|disregard|forget) + (\w+\s+){0,4} + <target noun>
#       The filler group handles "your", "all", "all previous", "the", etc.
#       Up to four filler words covers realistic variants without runaway
#       backtracking on short inputs (length cap is 4 000 chars).
#   - "bypass" requires a safety-context noun so "bypass the slow shipping
#       option" does NOT trigger.
#   - "act as (a|an)" requires a specific adversarial role so "act as a
#       product guide" does NOT trigger.
_INJECTION_PATTERN = re.compile(
    r"(?i)("
    # Core: ignore / disregard / forget + 0-4 filler words + target noun
    r"(ignore|disregard|forget)\s+(\w+\s+){0,4}"
    r"(instructions?|directives?|rules?|prompts?|guidelines?|constraints?|policies)"
    # pretend the/your system/instructions say/are ...
    r"|pretend\s+(your|the)\s+(system|instructions?|rules?)\s+(say|state|are|is)"
    # you are now <something>
    r"|you\s+are\s+now\b"
    # act as if you are [anything]
    r"|act\s+as\s+if\s+you\s+are"
    # act as a/an [specific adversarial role] — NOT generic "act as a guide"
    r"|act\s+as\s+(a|an)\s+(unrestricted|uncensored|unlimited|unfiltered"
    r"|jailbroken|evil|rogue|hacker"
    r"|different\s+(AI|model|assistant)|GPT|DAN)\b"
    # system prompt as a phrase
    r"|system\s+prompt"
    # developer message as a phrase
    r"|developer\s+message"
    # jailbreak keyword
    r"|jailbreak"
    # bypass only when followed by a safety-context noun (avoids false
    # positives on "bypass the slow shipping option")
    r"|bypass\s+(\w+\s+){0,3}"
    r"(safety|filters?|guardrails?|restrictions?|blocks?|limits?|policies|policy|controls?)"
    r")"
)


def standard_out_of_scope_message(user_message: str | None = None) -> str:
    """Return a warmer fallback response for unsupported or non-support requests."""
    message = (user_message or "").strip()
    lowered = message.lower()

    if lowered in {"hi", "hello", "hey", "hey there", "hi there"}:
        return (
            "Hi! I’m here to help with e-commerce support questions about products, "
            "orders, returns, recommendations, and escalation. What would you like help with?"
        )

    if any(token in lowered for token in ["thanks", "thank you", "good morning", "good afternoon", "good evening"]):
        return (
            "You’re welcome! I can help with product info, orders, returns, recommendations, "
            "or escalation support. What do you need today?"
        )

    return (
        "I can help with e-commerce support topics such as products, orders, returns, "
        "recommendations, and escalation. If you have a specific question, feel free to share it."
    )


def validate_user_input(user_message: str) -> tuple[bool, str]:
    """Apply input guardrails before sending prompts to the model.

    Checks run in order of cost — cheapest first:

    1. **Empty check** — always instant.
    2. **Length cap** (4 000 chars) — always instant.
    3. **Regex injection fast-path** — microsecond cost, catches obvious
       variants (ignore/disregard/forget instructions, jailbreak, etc.).
    4. **ML scanner** (opt-in, ``ENABLE_ML_GUARDRAIL=true``) — BERT-based
       classifier from ``llm-guard``.  Catches paraphrased / obfuscated
       injection attempts that the regex misses.  Adds ~80–200 ms on CPU.
       **Disabled by default** — the bundled model produces false positives
       on legitimate e-commerce phrases ("Return this and reorder it"
       scores 1.0).  Enable only after tuning ``ML_GUARDRAIL_THRESHOLD``
       against a sample of real customer messages.
    """
    trimmed = user_message.strip()
    if not trimmed:
        return False, "Please share your request so I can help."

    if len(trimmed) > 4000:
        return False, "Your message is too long. Please shorten it and try again."

    if _INJECTION_PATTERN.search(trimmed):
        return False, (
            "I can't follow instructions that try to override system rules. "
            "Please ask a normal product or support question."
        )

    # ML-based scanner — runs only when ENABLE_ML_GUARDRAIL=true.
    # Positioned after the regex so the common cases pay zero ML cost.
    ok, error = _ml_scan(trimmed)
    if not ok:
        return False, error

    return True, ""


def validate_agent_output(output_text: str) -> tuple[bool, str]:
    """Apply lightweight output guardrails before returning a response."""
    trimmed = output_text.strip()
    if not trimmed:
        return False, "I couldn't generate a response. Please try again."
    return True, ""


def reject_write_sql(sql_query: str) -> tuple[bool, str]:
    """Block write operations for read-only query tools."""
    if _WRITE_PATTERN.search(sql_query):
        return (
            False,
            "ERROR: Only SELECT queries are allowed. "
            "Write operations are blocked in this tool.",
        )
    return True, ""


#: The only tables the product agent's raw-SQL escape hatch may touch. These
#: hold public catalog content; everything else in this database is
#: customer-owned.
_CATALOG_TABLES = frozenset({"product_catalog", "reviews"})

#: Customer-owned tables, named explicitly. This is a backstop, not the primary
#: gate: the allowlist below already refuses anything it does not recognise.
#: Listing these by name means a query that mentions one is rejected on the
#: token alone, no matter how it is smuggled in — comma joins, correlated
#: subqueries, CTEs, aliases. Keep in sync with the schema; a table missing
#: from here is still caught by the allowlist unless it is also aliased past
#: the clause parser.
_CUSTOMER_TABLES = frozenset(
    {
        "customers",
        "orders",
        "order_items",
        "returns",
        "customer_sessions",
        "customer_queries",
        "demo_date_backup",
    }
)

_SQL_COMMENT_PATTERN = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
#: Single-quoted literals, including '' escapes. Stripped before scanning so a
#: product whose title contains the word "orders" is not mistaken for a table.
_SQL_STRING_PATTERN = re.compile(r"'(?:[^']|'')*'")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][\w$]*")

#: Keywords that terminate a FROM clause, so the comma-separated table list can
#: be isolated from what follows it.
_CLAUSE_END = (
    r"WHERE|GROUP|ORDER|LIMIT|HAVING|WINDOW|UNION|INTERSECT|EXCEPT|"
    r"INNER|LEFT|RIGHT|FULL|CROSS|JOIN|ON|OFFSET|FETCH"
)
_FROM_CLAUSE_PATTERN = re.compile(
    rf"\b(?:FROM|JOIN)\s+(.*?)(?=\b(?:{_CLAUSE_END})\b|\)|;|$)",
    re.IGNORECASE | re.DOTALL,
)


def _referenced_tables(sql_query: str) -> list[str]:
    """Return every table named after FROM or JOIN, comma lists included.

    ``FROM product_catalog p, customers c`` is one FROM clause naming two
    tables. Capturing only the first identifier after FROM — as the original
    version of this check did — silently allowed the second.
    """
    tables: list[str] = []
    for clause in _FROM_CLAUSE_PATTERN.findall(sql_query):
        for part in clause.split(","):
            identifiers = _IDENTIFIER_PATTERN.findall(part)
            if not identifiers:
                continue
            # First identifier is the (possibly schema-qualified) table; any
            # further ones are the alias or the AS keyword.
            tables.append(identifiers[0].lower())
    return tables


def restrict_to_catalog_tables(sql_query: str) -> tuple[bool, str]:
    """Confine an ad-hoc SELECT to the public product catalog.

    ``reject_write_sql`` blocks writes but says nothing about *which* tables
    are read. That left the product agent's raw-SQL escape hatch able to run
    ``SELECT customer_id, email FROM customers`` — a full dump of every
    customer's contact details reachable by asking a product question.

    Two layers, because regex-based SQL analysis is easy to get subtly wrong:

    1. Every table named after FROM/JOIN must be in :data:`_CATALOG_TABLES`.
       Allowlist, so an unrecognised table is refused rather than permitted.
    2. The query must not mention a known customer-owned table *anywhere*,
       whatever the syntax. This catches constructions the clause parser in
       layer 1 does not model.

    This is still not a SQL parser. It is a deliberately narrow gate on a tool
    whose legitimate use is ``SELECT ... FROM product_catalog`` — the correct
    long-term fix is a read-only database role scoped to the catalog tables,
    which would enforce this in the engine rather than in a regex.
    """
    # Strip comments and string literals first: both are places to hide a table
    # name from layer 2, and literals are a false-positive source for it.
    scrubbed = _SQL_STRING_PATTERN.sub(" ", _SQL_COMMENT_PATTERN.sub(" ", sql_query))

    tokens = {token.lower() for token in _IDENTIFIER_PATTERN.findall(scrubbed)}
    forbidden = sorted(tokens & _CUSTOMER_TABLES)
    if forbidden:
        return False, (
            f"ERROR: This tool reads the product catalog only; "
            f"'{forbidden[0]}' is not accessible. Customer, order and return "
            f"data cannot be queried here."
        )

    referenced = _referenced_tables(scrubbed)
    if not referenced:
        return False, (
            "ERROR: Could not identify the table being queried. "
            "This tool only reads the product catalog."
        )

    for reference in referenced:
        table = reference.split(".")[-1].lower()
        if table not in _CATALOG_TABLES:
            return False, (
                f"ERROR: This tool can only read the product catalog "
                f"({', '.join(sorted(_CATALOG_TABLES))}); '{table}' is not "
                f"accessible. Customer, order and return data cannot be "
                f"queried here."
            )

    return True, ""


def limit_rows(rows: list[dict[str, Any]], max_rows: int = 5) -> tuple[list[dict[str, Any]], str]:
    """Trim large result sets to keep token usage bounded."""
    total = len(rows)
    if total <= max_rows:
        return rows, ""

    footer = f"\n... (showing {max_rows} of {total} results)"
    return rows[:max_rows], footer