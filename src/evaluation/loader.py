"""Load and resolve ground-truth datasets."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.fixtures import Fixtures, required_placeholders
from src.evaluation.schema import Checks, EvalCase

DATASET_DIR = Path(__file__).parent / "datasets"

#: Datasets keyed by the agent they exercise. ``routing`` is handled separately
#: by :mod:`src.evaluation.routing` because it scores the router, not an agent.
AGENT_DATASETS: dict[str, str] = {
    "order": "order.jsonl",
    "return": "return.jsonl",
    "product": "product.jsonl",
    "recommendation": "recommendation.jsonl",
    "escalation": "escalation.jsonl",
    "fallback": "fallback.jsonl",
}


def load_raw_cases(path: Path) -> list[dict]:
    """Read a JSONL dataset, ignoring blank lines and ``//`` comments."""
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    cases: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            cases.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {number}: invalid JSON — {exc}") from exc
    return cases


def _resolve_case(raw: dict, fixtures: Fixtures) -> EvalCase:
    """Substitute placeholders throughout a case, then validate it."""
    checks_raw = raw.get("checks") or {}

    resolved = dict(raw)
    resolved["query"] = fixtures.resolve(str(raw.get("query", "")))
    if raw.get("customer_id") is not None:
        resolved["customer_id"] = fixtures.resolve(str(raw["customer_id"]))

    checks = Checks.from_dict(checks_raw)
    resolved["checks"] = {
        "expect_tools": list(checks.expect_tools),
        "expect_any_tools": list(checks.expect_any_tools),
        "forbid_tools": list(checks.forbid_tools),
        "must_contain": list(fixtures.resolve_all(checks.must_contain)),
        "must_not_contain": list(fixtures.resolve_all(checks.must_not_contain)),
        "must_match": list(fixtures.resolve_all(checks.must_match)),
        "expect_refusal": checks.expect_refusal,
        "allowed_ids": list(fixtures.resolve_all(checks.allowed_ids)),
    }
    return EvalCase.from_dict(resolved)


def _case_placeholders(raw: dict) -> set[str]:
    """Every placeholder a case depends on, across all of its fields."""
    checks = raw.get("checks") or {}
    blob = " ".join(
        [
            str(raw.get("query", "")),
            str(raw.get("customer_id") or ""),
            " ".join(str(v) for v in (checks.get("must_contain") or [])),
            " ".join(str(v) for v in (checks.get("must_not_contain") or [])),
            " ".join(str(v) for v in (checks.get("must_match") or [])),
            " ".join(str(v) for v in (checks.get("allowed_ids") or [])),
        ]
    )
    return required_placeholders(blob)


def load_cases(
    agent: str,
    fixtures: Fixtures,
) -> tuple[list[EvalCase], list[tuple[str, str]]]:
    """Load one agent's dataset.

    Returns ``(cases, unavailable)`` where ``unavailable`` lists
    ``(case_id, reason)`` for cases the current database cannot support — a
    returns case when the database holds no returns, for example. Those are
    reported as skipped rather than silently dropped, so a shrinking dataset
    is visible in the report instead of quietly inflating the pass rate.
    """
    filename = AGENT_DATASETS.get(agent)
    if filename is None:
        raise KeyError(f"no dataset registered for agent '{agent}'")

    cases: list[EvalCase] = []
    unavailable: list[tuple[str, str]] = []

    for raw in load_raw_cases(DATASET_DIR / filename):
        missing = _case_placeholders(raw) - set(fixtures.values)
        if missing:
            unavailable.append(
                (str(raw.get("id", "<unnamed>")), f"no data for {', '.join(sorted(missing))}")
            )
            continue
        cases.append(_resolve_case(raw, fixtures))

    return cases, unavailable


def load_all_cases(
    fixtures: Fixtures,
    agents: list[str] | None = None,
) -> tuple[dict[str, list[EvalCase]], dict[str, list[tuple[str, str]]]]:
    """Load datasets for the requested agents (all registered ones by default)."""
    selected = agents or list(AGENT_DATASETS)
    loaded: dict[str, list[EvalCase]] = {}
    unavailable: dict[str, list[tuple[str, str]]] = {}

    for agent in selected:
        cases, missing = load_cases(agent, fixtures)
        loaded[agent] = cases
        if missing:
            unavailable[agent] = missing

    return loaded, unavailable
