"""Deterministic evaluation harness for the support agents.

Run it with::

    uv run python -m src.evaluation                 # every agent + routing
    uv run python -m src.evaluation --agents order  # one agent
    uv run python -m src.evaluation --no-routing    # skip the router pass

See the "Evaluation" section of the repository ``README.md`` for the case
format and scoring rules.
"""

from __future__ import annotations

from src.evaluation.schema import AgentReport, CaseResult, CheckResult, Checks, EvalCase

__all__ = ["AgentReport", "CaseResult", "CheckResult", "Checks", "EvalCase"]
