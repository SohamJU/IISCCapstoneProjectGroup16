"""Typed structures for the agent evaluation harness.

An evaluation case is a question plus a set of *checkable* expectations. The
harness deliberately scores only things that can be verified mechanically —
which tools ran, which facts appear, whether a refusal happened — rather than
asking a model to grade prose. That keeps a run free, fast and reproducible:
the same dataset against the same code gives the same score, so a regression is
a real regression rather than judge variance.

The cost of that choice is that phrasing quality is not measured. A terse but
correct answer and a warm, well-structured one score identically here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Provider/transport failures that say nothing about agent quality. A run that
#: exhausts the daily token quota must not be reported as "the escalation agent
#: scored 0%" — that is a confident, wrong number, which is the specific failure
#: mode this harness exists to avoid.
_INFRASTRUCTURE_ERROR_PATTERN = re.compile(
    r"ratelimit|rate_limit|rate limit|429|quota|tokens per (?:day|minute)"
    r"|apiconnection|connection error|timeout|timed out|503|502|504"
    r"|service unavailable",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Checks:
    """The verifiable expectations attached to one case.

    Every field is optional. An empty ``Checks`` asserts only that the agent
    produced a non-empty answer without erroring.
    """

    #: Tools that MUST appear in the agent's tool calls — all of them.
    expect_tools: tuple[str, ...] = ()
    #: At least one of these must be called. For cases where more than one tool
    #: is a legitimate way to answer (list the categories, or search and infer
    #: them), so the check tests capability rather than dictating a strategy.
    expect_any_tools: tuple[str, ...] = ()
    #: Tools that must NOT be called (e.g. a write tool on a read-only query).
    forbid_tools: tuple[str, ...] = ()
    #: Case-insensitive substrings the answer must contain.
    must_contain: tuple[str, ...] = ()
    #: Case-insensitive substrings the answer must not contain — the main
    #: vehicle for asserting that another customer's data never leaks.
    must_not_contain: tuple[str, ...] = ()
    #: Regexes the answer must match (case-insensitive).
    must_match: tuple[str, ...] = ()
    #: When true, the answer must read as an inability/refusal rather than an
    #: attempt to satisfy the request.
    expect_refusal: bool = False
    #: When set, any ORD-/RET- identifier in the answer must be in this list.
    #: This is the hallucination guard: an agent inventing a plausible order
    #: number is a failure even if the rest of the answer looks right.
    allowed_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Checks:
        def _tuple(key: str) -> tuple[str, ...]:
            value = raw.get(key) or []
            if isinstance(value, str):
                value = [value]
            return tuple(str(item) for item in value)

        return cls(
            expect_tools=_tuple("expect_tools"),
            expect_any_tools=_tuple("expect_any_tools"),
            forbid_tools=_tuple("forbid_tools"),
            must_contain=_tuple("must_contain"),
            must_not_contain=_tuple("must_not_contain"),
            must_match=_tuple("must_match"),
            expect_refusal=bool(raw.get("expect_refusal", False)),
            allowed_ids=_tuple("allowed_ids"),
        )


@dataclass(frozen=True)
class EvalCase:
    """One ground-truth item: a query plus what a correct handling looks like."""

    id: str
    agent: str
    query: str
    checks: Checks
    #: Customer the query is asked as. ``None`` means a signed-out session,
    #: which is itself a meaningful case for the authorization checks.
    customer_id: str | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    #: True when running the case writes to the database (cancelling an order,
    #: creating a return). Skipped unless the runner is given --allow-writes,
    #: so a routine evaluation cannot mutate the demo data.
    mutates: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalCase:
        missing = [key for key in ("id", "agent", "query") if not raw.get(key)]
        if missing:
            raise ValueError(f"case is missing required field(s): {', '.join(missing)}")

        tags = raw.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]

        return cls(
            id=str(raw["id"]),
            agent=str(raw["agent"]),
            query=str(raw["query"]),
            checks=Checks.from_dict(raw.get("checks") or {}),
            customer_id=(
                str(raw["customer_id"]) if raw.get("customer_id") is not None else None
            ),
            description=str(raw.get("description", "")),
            tags=tuple(str(tag) for tag in tags),
            mutates=bool(raw.get("mutates", False)),
        )


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single assertion within a case."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    """Outcome of running one case end to end."""

    case: EvalCase
    checks: list[CheckResult] = field(default_factory=list)
    answer: str = ""
    tools_used: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    error: str = ""
    skipped: str = ""
    #: True when the answer came from the cache rather than a fresh model call.
    #: The score is still computed fresh; only the answer was reused.
    reused: bool = False

    @property
    def infrastructure_error(self) -> bool:
        """True when the case failed for provider/transport reasons.

        Such a case is *unscored*, not failed: no evidence about the agent was
        obtained. It is surfaced separately so an exhausted quota reads as
        "23 cases could not run", never as a low score.
        """
        return bool(self.error) and bool(
            _INFRASTRUCTURE_ERROR_PATTERN.search(self.error)
        )

    @property
    def passed(self) -> bool:
        """A case passes only when every one of its checks passes.

        Strict on purpose: a response that gets the order right but leaks
        another customer's ID has not "mostly passed".
        """
        if self.error or self.skipped:
            return False
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def checks_passed(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]


@dataclass
class AgentReport:
    """Aggregated results for one agent."""

    agent: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def executed(self) -> list[CaseResult]:
        """Cases that actually produced evidence about the agent.

        Excludes both skipped cases and those lost to provider failures — a
        case that never reached the model cannot inform the score either way.
        """
        return [r for r in self.results if not r.skipped and not r.infrastructure_error]

    @property
    def skipped(self) -> list[CaseResult]:
        return [r for r in self.results if r.skipped]

    @property
    def unscored(self) -> list[CaseResult]:
        """Cases lost to rate limits, quota exhaustion or connection failures."""
        return [r for r in self.results if r.infrastructure_error]

    @property
    def total(self) -> int:
        return len(self.executed)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.executed if r.passed)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases where every check passed."""
        return self.passed / self.total if self.total else 0.0

    @property
    def check_score(self) -> float:
        """Fraction of individual checks passed across all cases.

        Reported alongside ``pass_rate`` because the two answer different
        questions: pass_rate is "how many cases were fully correct", check_score
        is "how close were the failures". A run can have a poor pass rate and a
        high check score, which means many near-misses rather than deep breakage.
        """
        total = sum(len(r.checks) for r in self.executed)
        if not total:
            return 0.0
        return sum(r.checks_passed for r in self.executed) / total

    @property
    def reused(self) -> list[CaseResult]:
        """Cases scored from a stored answer instead of a fresh model call."""
        return [r for r in self.executed if r.reused]

    @property
    def mean_latency(self) -> float:
        """Mean latency over freshly executed cases only.

        Reused cases carry the latency of whenever they were first recorded, so
        averaging them in would describe a run that never happened.
        """
        fresh = [r for r in self.executed if not r.reused and r.latency_seconds > 0]
        if not fresh:
            return 0.0
        return sum(r.latency_seconds for r in fresh) / len(fresh)

    @property
    def errored(self) -> list[CaseResult]:
        return [r for r in self.executed if r.error]
