"""Intent-routing accuracy against the labelled ``customer_queries`` dataset.

The dataset carries 878 rows labelled with 17 intents. This system has six
routes. The mapping between them is **not** clean, and pretending otherwise
would manufacture a flattering accuracy number:

* Ten intents map unambiguously onto a route (``order_tracking`` -> order,
  ``refunds`` -> return, and so on). These form the **core** set and produce the
  headline accuracy figure.
* Seven intents describe support work this system has no agent for — billing
  disputes, account changes, loyalty schemes, general complaints. There is no
  single correct route for them, only a set of defensible ones. They are scored
  separately against that accepted set and excluded from headline accuracy.

Reporting one number over all 878 rows would either punish the router for
lacking a billing agent, or credit it for guessing inside a set of equally
acceptable answers. Splitting them keeps both figures meaningful.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from src.data.postgresql import execute_sql_query_params
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

#: Intents with exactly one defensible route. Headline accuracy uses only these.
CORE_INTENT_ROUTES: dict[str, str] = {
    "order_tracking": "order",
    "order_modification": "order",
    "order_cancellation": "order",
    "shipping_estimate": "order",
    "delivery_issues": "order",
    "returns": "return",
    "refunds": "return",
    "warranty_replacement": "return",
    "product_search": "product",
    "product_comparison": "product",
    "product_recommendation": "recommendation",
}

#: Intents outside this system's agent taxonomy. Scored against a set of
#: acceptable routes rather than a single gold label.
AMBIGUOUS_INTENT_ROUTES: dict[str, frozenset[str]] = {
    # No billing agent exists; a human handoff or a graceful decline both work.
    "payment_issues": frozenset({"escalation", "fallback"}),
    "account_issues": frozenset({"escalation", "fallback"}),
    "account_update": frozenset({"fallback", "escalation"}),
    # A complaint may be domain work or a handoff depending on its content.
    "complaints": frozenset({"escalation", "fallback", "return", "order"}),
    "discounts_offers": frozenset({"product", "fallback", "recommendation"}),
    "loyalty_inquiry": frozenset({"fallback", "escalation"}),
}


@dataclass
class RoutingSample:
    """One labelled query and what the router did with it."""

    query: str
    intent: str
    expected: str | frozenset[str]
    predicted: list[str] = field(default_factory=list)
    correct: bool = False
    error: str = ""
    #: Routes derived from the row's ``all_intents`` column. Many rows in this
    #: dataset are genuinely multi-intent ("it is defective and I want a better
    #: option"), and the ``intent`` column records only the primary one.
    accepted_routes: frozenset[str] = frozenset()
    #: True when the prediction matches ANY labelled intent on the row.
    lenient_correct: bool = False


@dataclass
class RoutingReport:
    """Aggregated routing results, core and ambiguous kept apart."""

    core: list[RoutingSample] = field(default_factory=list)
    ambiguous: list[RoutingSample] = field(default_factory=list)

    @property
    def core_accuracy(self) -> float:
        """Strict: the primary intent's route must be among the predictions."""
        scored = [s for s in self.core if not s.error]
        if not scored:
            return 0.0
        return sum(1 for s in scored if s.correct) / len(scored)

    @property
    def core_accuracy_lenient(self) -> float:
        """Lenient: any intent labelled on the row is accepted.

        Reported next to the strict figure because most rows in this dataset
        carry several intents and the ``intent`` column names only the primary
        one. A message reading "it is defective and I want a better option" is
        labelled ``product_comparison`` but routing it to ``return`` is a
        defensible read of the same text — the strict metric scores that as a
        miss, which measures label choice rather than router quality.
        """
        scored = [s for s in self.core if not s.error]
        if not scored:
            return 0.0
        return sum(1 for s in scored if s.lenient_correct) / len(scored)

    @property
    def errors(self) -> list[RoutingSample]:
        """Samples where the router itself failed (quota, malformed JSON)."""
        return [s for s in self.core + self.ambiguous if s.error]

    @property
    def ambiguous_acceptance(self) -> float:
        scored = [s for s in self.ambiguous if not s.error]
        if not scored:
            return 0.0
        return sum(1 for s in scored if s.correct) / len(scored)

    def per_route_metrics(self) -> dict[str, dict[str, float]]:
        """Precision, recall and F1 per route over the core set.

        Computed by hand rather than via sklearn so the harness keeps working
        with no extra dependency, and so multi-route predictions can be scored
        as "the gold route is among the predictions" — which is what the
        supervisor actually needs to be right about.
        """
        scored = [s for s in self.core if not s.error]
        routes = sorted({str(s.expected) for s in scored} | {r for s in scored for r in s.predicted})

        metrics: dict[str, dict[str, float]] = {}
        for route in routes:
            true_positive = sum(
                1 for s in scored if s.expected == route and route in s.predicted
            )
            false_positive = sum(
                1 for s in scored if s.expected != route and route in s.predicted
            )
            false_negative = sum(
                1 for s in scored if s.expected == route and route not in s.predicted
            )

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
                "support": float(sum(1 for s in scored if s.expected == route)),
            }
        return metrics

    def confusion(self) -> dict[str, Counter]:
        """Gold route -> counter of first predicted route, over the core set."""
        matrix: dict[str, Counter] = {}
        for sample in self.core:
            if sample.error:
                continue
            gold = str(sample.expected)
            predicted = sample.predicted[0] if sample.predicted else "<none>"
            matrix.setdefault(gold, Counter())[predicted] += 1
        return matrix


def routes_for_intents(all_intents: str) -> frozenset[str]:
    """Map a row's full ``all_intents`` string onto the routes it licenses."""
    routes: set[str] = set()
    for raw in (all_intents or "").split(","):
        intent = raw.strip()
        if intent in CORE_INTENT_ROUTES:
            routes.add(CORE_INTENT_ROUTES[intent])
        elif intent in AMBIGUOUS_INTENT_ROUTES:
            routes |= AMBIGUOUS_INTENT_ROUTES[intent]
    return frozenset(routes)


def load_labelled_queries(
    limit_per_intent: int = 5,
    seed: int = 20260726,
    include_ambiguous: bool = True,
) -> list[tuple[str, str, str]]:
    """Sample ``(query, intent)`` pairs, stratified across intents.

    Stratified rather than random because the raw distribution is skewed and a
    plain sample would under-represent the rarer intents. Every routing call
    costs an LLM round trip, so the default sample is small; raise
    ``limit_per_intent`` for a fuller run.
    """
    rows = execute_sql_query_params(
        "SELECT query, intent, all_intents FROM customer_queries WHERE query IS NOT NULL"
    )
    if isinstance(rows, str):
        raise RuntimeError(f"could not load customer_queries: {rows}")

    known = set(CORE_INTENT_ROUTES)
    if include_ambiguous:
        known |= set(AMBIGUOUS_INTENT_ROUTES)

    grouped: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        intent = str(row.get("intent", "")).strip()
        if intent not in known:
            continue
        grouped.setdefault(intent, []).append(
            (str(row["query"]), str(row.get("all_intents") or intent))
        )

    rng = random.Random(seed)
    sampled: list[tuple[str, str, str]] = []
    for intent in sorted(grouped):
        queries = grouped[intent]
        rng.shuffle(queries)
        for query, all_intents in queries[:limit_per_intent]:
            sampled.append((query, intent, all_intents))

    _LOGGER.info(
        "routing sample: %d queries across %d intents", len(sampled), len(grouped)
    )
    return sampled


def evaluate_routing(
    router: object,
    samples: list[tuple[str, str, str]],
    progress: bool = True,
) -> RoutingReport:
    """Run the router over labelled queries and score the predictions."""
    report = RoutingReport()

    for index, sample in enumerate(samples, start=1):
        query, intent = sample[0], sample[1]
        all_intents = sample[2] if len(sample) > 2 else intent

        if progress:
            print(f"  [{index}/{len(samples)}] {intent} ... ", end="", flush=True)

        try:
            decision = router.classify_multi(query)  # type: ignore[attr-defined]
            predicted = [str(route) for route in (decision.get("routes") or [])]
            error = ""
        except Exception as exc:
            predicted, error = [], f"{type(exc).__name__}: {exc}"

        accepted_routes = routes_for_intents(all_intents)
        lenient = bool(set(predicted) & accepted_routes)

        if intent in CORE_INTENT_ROUTES:
            expected: str | frozenset[str] = CORE_INTENT_ROUTES[intent]
            # Correct when the gold route is among the (at most two) predicted
            # routes: a compound query legitimately fans out, and penalising
            # that would measure verbosity rather than correctness.
            correct = str(expected) in predicted
            report.core.append(
                RoutingSample(
                    query, intent, expected, predicted, correct, error,
                    accepted_routes=accepted_routes, lenient_correct=lenient,
                )
            )
        else:
            accepted = AMBIGUOUS_INTENT_ROUTES[intent]
            correct = any(route in accepted for route in predicted)
            report.ambiguous.append(
                RoutingSample(
                    query, intent, accepted, predicted, correct, error,
                    accepted_routes=accepted_routes, lenient_correct=correct,
                )
            )

        if progress:
            marker = "OK  " if correct and not error else ("~   " if lenient else "MISS")
            print(f"{marker} -> {predicted or error}")

    return report
