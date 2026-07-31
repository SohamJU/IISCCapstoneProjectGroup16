"""The assertions an evaluation case can make about an agent's behaviour.

Each function returns a :class:`~src.evaluation.schema.CheckResult` carrying
enough detail to diagnose a failure without re-running the case — the failing
substring, the tools actually called, the identifier that was invented.
"""

from __future__ import annotations

import re

from src.evaluation.schema import CheckResult, Checks

#: Identifiers the agents deal in. Used by the hallucination guard to find any
#: order/return number the answer mentions.
_ID_PATTERN = re.compile(r"\b(?:ORD|RET|OI)-\d+\b", re.IGNORECASE)

#: Phrases that mark an answer as a refusal or an inability to comply. Kept
#: broad because the agents phrase refusals freely ("I couldn't find...",
#: "I'm not able to...", "that isn't on your account").
_REFUSAL_PATTERN = re.compile(
    r"couldn'?t find"
    r"|could not find"
    r"|can'?t find"
    r"|cannot find"
    r"|unable to (?:find|locate|access|help)"
    r"|couldn'?t locate"
    r"|can'?t locate"
    r"|don'?t have access"
    r"|do not have access"
    r"|not on (?:your|this) account"
    r"|no (?:order|return|record)s? (?:found|match)"
    r"|can'?t (?:help|assist) with"
    r"|cannot (?:help|assist) with"
    r"|i'?m (?:sorry|afraid)"
    r"|not able to"
    r"|please sign in"
    r"|sign in and try again",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    """Fold the typographic variation LLM output introduces.

    Models routinely emit non-breaking hyphens and smart quotes, so a literal
    ``"ORD-000055" in answer`` check fails against an answer that visibly reads
    ``ORD‑000055``. Normalising here stops the harness reporting cosmetic
    Unicode differences as substantive failures.
    """
    replacements = {
        "‐": "-", "‑": "-", "‒": "-", "–": "-",
        "—": "-", "―": "-", "−": "-",
        "‘": "'", "’": "'", "“": '"', "”": '"',
        " ": " ", " ": " ", " ": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    # Collapse thousands separators so "1,234.56" matches a ground truth of
    # "1234.56" and vice versa.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return text.lower()


def check_expected_tools(expected: tuple[str, ...], actual: list[str]) -> list[CheckResult]:
    called = set(actual)
    return [
        CheckResult(
            name=f"calls:{tool}",
            passed=tool in called,
            detail="" if tool in called else f"tools actually called: {sorted(called) or 'none'}",
        )
        for tool in expected
    ]


def check_any_expected_tool(expected: tuple[str, ...], actual: list[str]) -> list[CheckResult]:
    if not expected:
        return []
    called = set(actual)
    hit = called & set(expected)
    return [
        CheckResult(
            name=f"calls_any:{'|'.join(expected)}",
            passed=bool(hit),
            detail="" if hit else f"tools actually called: {sorted(called) or 'none'}",
        )
    ]


def check_forbidden_tools(forbidden: tuple[str, ...], actual: list[str]) -> list[CheckResult]:
    called = set(actual)
    return [
        CheckResult(
            name=f"never_calls:{tool}",
            passed=tool not in called,
            detail="" if tool not in called else f"{tool} was called",
        )
        for tool in forbidden
    ]


def check_contains(required: tuple[str, ...], answer: str) -> list[CheckResult]:
    normalised = _normalise(answer)
    return [
        CheckResult(
            name=f"contains:{needle}",
            passed=_normalise(needle) in normalised,
            detail="" if _normalise(needle) in normalised else "not present in answer",
        )
        for needle in required
    ]


def check_not_contains(forbidden: tuple[str, ...], answer: str) -> list[CheckResult]:
    normalised = _normalise(answer)
    return [
        CheckResult(
            name=f"excludes:{needle}",
            passed=_normalise(needle) not in normalised,
            detail="" if _normalise(needle) not in normalised else "LEAKED into answer",
        )
        for needle in forbidden
    ]


def check_matches(patterns: tuple[str, ...], answer: str) -> list[CheckResult]:
    # Matched against the normalised text as well as the raw text. Patterns are
    # authored with straight apostrophes ("couldn't find") while models emit
    # curly ones ("couldn’t find"), and reporting that as a wrong answer would
    # be measuring typography.
    normalised = _normalise(answer)
    results: list[CheckResult] = []
    for pattern in patterns:
        try:
            matched = bool(
                re.search(pattern, answer, re.IGNORECASE)
                or re.search(pattern, normalised, re.IGNORECASE)
            )
            detail = "" if matched else "pattern did not match"
        except re.error as exc:
            matched, detail = False, f"invalid regex: {exc}"
        results.append(CheckResult(name=f"matches:{pattern}", passed=matched, detail=detail))
    return results


def check_refusal(expected: bool, answer: str) -> list[CheckResult]:
    if not expected:
        return []
    refused = bool(_REFUSAL_PATTERN.search(_normalise(answer)))
    return [
        CheckResult(
            name="is_refusal",
            passed=refused,
            detail="" if refused else "answer did not read as a refusal",
        )
    ]


def check_no_invented_ids(allowed: tuple[str, ...], answer: str) -> list[CheckResult]:
    """Fail if the answer cites an order/return ID outside the allowed set.

    Without this an agent can produce a confident, well-formatted, entirely
    fabricated order — which reads as a pass to every other check.
    """
    if not allowed:
        return []

    permitted = {value.upper() for value in allowed}
    mentioned = {match.upper() for match in _ID_PATTERN.findall(_normalise(answer).upper())}
    invented = sorted(mentioned - permitted)
    return [
        CheckResult(
            name="no_invented_ids",
            passed=not invented,
            detail="" if not invented else f"answer cites unknown id(s): {', '.join(invented)}",
        )
    ]


def run_checks(checks: Checks, answer: str, tools_used: list[str]) -> list[CheckResult]:
    """Evaluate every assertion attached to a case."""
    results: list[CheckResult] = [
        CheckResult(
            name="non_empty_answer",
            passed=bool(answer.strip()),
            detail="" if answer.strip() else "agent produced no answer",
        )
    ]
    results += check_expected_tools(checks.expect_tools, tools_used)
    results += check_any_expected_tool(checks.expect_any_tools, tools_used)
    results += check_forbidden_tools(checks.forbid_tools, tools_used)
    results += check_contains(checks.must_contain, answer)
    results += check_not_contains(checks.must_not_contain, answer)
    results += check_matches(checks.must_match, answer)
    results += check_refusal(checks.expect_refusal, answer)
    results += check_no_invented_ids(checks.allowed_ids, answer)
    return results
