"""Intent-routing evaluation against a hand-authored dataset.

Previously this scored the router against the generated ``customer_queries``
table — 878 rows labelled with 17 intents, mapped onto this system's 6 routes.
That was abandoned for two reasons:

1. **The labels were unreliable.** Queries labelled ``product_search`` read
   "it is missing parts and I need to...", which is plainly a return. Scoring
   against them measured label noise as much as router quality, and no amount
   of strict/lenient splitting fixed that.
2. **The taxonomies did not line up.** Seventeen intents against six routes
   meant a third of the dataset described work the system has no agent for, and
   had to be excluded or scored against a set of "defensible" answers.

``datasets/routing.jsonl`` replaces it: fewer cases, every one written against
the routes that actually exist, and each stating exactly what a correct routing
decision is. Small enough to run on a constrained API budget, and every case
earns its place.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets" / "routing.jsonl"


@dataclass(frozen=True)
class RoutingCase:
    """One labelled query and the routing decision it expects."""

    id: str
    query: str
    #: Every one of these routes must be predicted.
    expect_all: tuple[str, ...] = ()
    #: At least one of these must be predicted. Used where several answers are
    #: genuinely defensible, so there is no single gold label.
    expect_any: tuple[str, ...] = ()
    #: None of these may be predicted.
    forbid: tuple[str, ...] = ()
    description: str = ""
    tags: tuple[str, ...] = ()
    #: Present so the answer cache can hash this like any other case.
    customer_id: str | None = None

    @property
    def is_multi_intent(self) -> bool:
        return len(self.expect_all) > 1

    @property
    def has_single_gold(self) -> bool:
        """True when the case has one unambiguous expected route set.

        Cases with ``expect_any`` do not, and are excluded from per-route
        precision and recall — crediting or penalising a route for a choice
        among equally acceptable answers would make those figures meaningless.
        """
        return bool(self.expect_all) and not self.expect_any

    @classmethod
    def from_dict(cls, raw: dict) -> RoutingCase:
        if not raw.get("id") or not raw.get("query"):
            raise ValueError(f"routing case needs an id and a query: {raw!r}")
        if (
            not raw.get("expect_all")
            and not raw.get("expect_any")
            and not raw.get("forbid")
        ):
            raise ValueError(
                f"routing case {raw['id']} asserts nothing — "
                "give it expect_all, expect_any or forbid"
            )

        def _tuple(key: str) -> tuple[str, ...]:
            value = raw.get(key) or []
            if isinstance(value, str):
                value = [value]
            return tuple(str(v) for v in value)

        return cls(
            id=str(raw["id"]),
            query=str(raw["query"]),
            expect_all=_tuple("expect_all"),
            expect_any=_tuple("expect_any"),
            forbid=_tuple("forbid"),
            description=str(raw.get("description", "")),
            tags=_tuple("tags"),
        )


@dataclass
class RoutingCaseResult:
    """Outcome of routing one case."""

    case: RoutingCase
    predicted: list[str] = field(default_factory=list)
    error: str = ""
    reused: bool = False

    @property
    def missing(self) -> list[str]:
        """Required routes the router failed to predict."""
        return [r for r in self.case.expect_all if r not in self.predicted]

    @property
    def forbidden_hit(self) -> list[str]:
        return [r for r in self.case.forbid if r in self.predicted]

    @property
    def any_satisfied(self) -> bool:
        if not self.case.expect_any:
            return True
        return any(r in self.predicted for r in self.case.expect_any)

    @property
    def extra(self) -> list[str]:
        """Predicted routes the case did not ask for.

        Not a failure on its own — a second route can be a reasonable reading —
        but tracked, because a router that always returns two routes would
        otherwise score well by covering every answer.
        """
        expected = set(self.case.expect_all) | set(self.case.expect_any)
        return [r for r in self.predicted if r not in expected]

    @property
    def correct(self) -> bool:
        if self.error:
            return False
        return not self.missing and not self.forbidden_hit and self.any_satisfied


@dataclass
class RoutingReport:
    """Aggregated routing results."""

    results: list[RoutingCaseResult] = field(default_factory=list)

    # ── Headline ──────────────────────────────────────────────────────────

    @property
    def scored(self) -> list[RoutingCaseResult]:
        return [r for r in self.results if not r.error]

    @property
    def errors(self) -> list[RoutingCaseResult]:
        return [r for r in self.results if r.error]

    @property
    def accuracy(self) -> float:
        if not self.scored:
            return 0.0
        return sum(1 for r in self.scored if r.correct) / len(self.scored)

    def _subset_accuracy(self, predicate) -> tuple[float, int]:
        subset = [r for r in self.scored if predicate(r.case)]
        if not subset:
            return 0.0, 0
        return sum(1 for r in subset if r.correct) / len(subset), len(subset)

    @property
    def single_intent(self) -> tuple[float, int]:
        return self._subset_accuracy(lambda c: not c.is_multi_intent)

    @property
    def multi_intent(self) -> tuple[float, int]:
        return self._subset_accuracy(lambda c: c.is_multi_intent)

    @property
    def over_routing_rate(self) -> float:
        """Share of cases where the router added a route nobody asked for.

        A router that always answers with two routes would pass many cases by
        brute force; this is the counterweight that makes that visible.
        """
        if not self.scored:
            return 0.0
        return sum(1 for r in self.scored if r.extra) / len(self.scored)

    # ── Per route ─────────────────────────────────────────────────────────

    def per_route_metrics(self) -> dict[str, dict[str, float]]:
        """Multi-label precision, recall and F1 per route.

        Computed only over cases with a single unambiguous gold set, and by
        hand rather than via sklearn so the harness needs no extra dependency.
        """
        scored = [r for r in self.scored if r.case.has_single_gold]
        routes = sorted(
            {route for r in scored for route in r.case.expect_all}
            | {route for r in scored for route in r.predicted}
        )

        metrics: dict[str, dict[str, float]] = {}
        for route in routes:
            gold = [route in r.case.expect_all for r in scored]
            predicted = [route in r.predicted for r in scored]

            true_positive = sum(1 for g, p in zip(gold, predicted) if g and p)
            false_positive = sum(1 for g, p in zip(gold, predicted) if not g and p)
            false_negative = sum(1 for g, p in zip(gold, predicted) if g and not p)

            precision = (
                true_positive / (true_positive + false_positive)
                if (true_positive + false_positive)
                else 0.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative)
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            metrics[route] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(sum(gold)),
            }
        return metrics

    def confusion(self) -> dict[str, Counter]:
        """Expected route -> what was predicted instead, for missed routes only."""
        matrix: dict[str, Counter] = {}
        for result in self.scored:
            for missed in result.missing:
                counter = matrix.setdefault(missed, Counter())
                for route in result.predicted or ["<none>"]:
                    counter[route] += 1
        return matrix


# ══════════════════════════════════════════════════════════════════════════
# Loading and execution
# ══════════════════════════════════════════════════════════════════════════


def load_routing_cases(path: Path = DATASET_PATH) -> list[RoutingCase]:
    """Read the hand-authored routing dataset."""
    if not path.exists():
        raise FileNotFoundError(f"routing dataset not found: {path}")

    cases: list[RoutingCase] = []
    seen: set[str] = set()
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name} line {number}: invalid JSON — {exc}"
            ) from exc

        case = RoutingCase.from_dict(raw)
        if case.id in seen:
            raise ValueError(f"{path.name}: duplicate routing case id {case.id}")
        seen.add(case.id)
        cases.append(case)

    _LOGGER.info("loaded %d routing cases from %s", len(cases), path.name)
    return cases


def evaluate_routing(
    router: object,
    cases: list[RoutingCase],
    progress: bool = True,
    cache: object | None = None,
) -> RoutingReport:
    """Run the router over the dataset and score its decisions.

    When ``cache`` is supplied, a case whose prediction is already recorded is
    reused rather than re-asked — routing is one model call per case, and a
    constrained API budget is better spent on cases that have changed.
    """
    report = RoutingReport()

    for index, case in enumerate(cases, start=1):
        if progress:
            print(f"  [{index}/{len(cases)}] {case.id:32} ", end="", flush=True)

        predicted: list[str] = []
        error = ""
        reused = False

        cached = cache.get("__routing__", case) if cache is not None else None
        if cached is not None:
            try:
                predicted = list(json.loads(cached[0]))
                reused = True
            except (json.JSONDecodeError, TypeError):
                cached = None  # unusable record; fall through to a live call

        if not reused:
            try:
                decision = router.classify_multi(case.query)  # type: ignore[attr-defined]
                predicted = [str(route) for route in (decision.get("routes") or [])]
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            if cache is not None and not error:
                cache.put_raw("__routing__", case, json.dumps(predicted))

        result = RoutingCaseResult(
            case=case, predicted=predicted, error=error, reused=reused
        )
        report.results.append(result)

        if progress:
            if error:
                print(f"ERROR {error[:60]}")
            elif result.correct:
                print(f"OK   {predicted}{' (reused)' if reused else ''}")
            else:
                detail = []
                if result.missing:
                    detail.append(f"missing {result.missing}")
                if result.forbidden_hit:
                    detail.append(f"forbidden {result.forbidden_hit}")
                if not result.any_satisfied:
                    detail.append(f"none of {list(case.expect_any)}")
                print(f"MISS {predicted} — {', '.join(detail)}")

    return report
