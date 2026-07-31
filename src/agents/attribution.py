"""Guard against attributing account data to the wrong person.

The access-control layer in :mod:`src.agents.authz` guarantees that a tool only
ever returns the signed-in customer's rows. It cannot guarantee that the agent
*describes* them correctly.

Asked "show me the orders placed by Mason Smith" while signed in as Danielle
Johnson, the agent did the right thing at the data layer — ``list_customer_orders``
returned Danielle's orders, because it takes no customer parameter — and then
wrote:

    Here are the most recent orders on the account for **Mason Smith**

Nothing leaked. But the customer is shown their own data under a stranger's
name, which is indistinguishable from a breach, implies that third-party lookup
is a supported feature, and is wrong on its face.

This module is the deterministic backstop. It scans the final reply for
*attribution phrases* — "the account for X", "X's orders", "orders placed by X"
— and fails the reply when X is not the signed-in customer.

Precision matters more than recall here. A guard that replaces a correct answer
with a refusal is itself a bug, and the phrase patterns alone are not precise
enough: "here are the orders for Maytag Refrigerator Water Filter replacements"
matches "orders for <Name>" perfectly well. Two constraints keep it tight:

1. Only the attribution *framings* above are considered, never a bare name.
2. A candidate is flagged only when it is confirmed to be a **real customer**
   other than the account holder, via an injected predicate.

Constraint 2 is what makes the guard usable. It also aims it at the case that
actually matters — the customer naming a real person — rather than at every
capitalised noun in the product catalog.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

__all__ = [
    "detect_misattribution",
    "detect_third_party_request",
    "find_attributed_names",
    "strip_markdown",
]

#: Markdown emphasis and links, stripped before scanning: the offending reply
#: wrote "the account for **Mason Smith**", and the asterisks would otherwise
#: sit between the phrase and the name.
_MARKDOWN_NOISE = re.compile(r"[*_`~]+")

#: A person-like name: two or three capitalised words. Requiring at least two
#: words keeps single capitalised nouns ("Delivered", "Shipped", "Refrigerator")
#: out of the match.
_NAME = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}"

#: Constructions that attribute account data to a named person. Each captures
#: the name in group "name".
#:
#: The phrase halves use scoped inline case-insensitivity ``(?i:...)`` while the
#: name half stays case-sensitive. A blanket ``re.IGNORECASE`` would make
#: ``[A-Z][a-z]+`` match lowercase words, so "orders for the account" would
#: report "the account" as a person. Without any case-insensitivity a reply
#: that *begins* "Orders placed by Mason Smith" is missed entirely.
_ATTRIBUTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "the account for Mason Smith", "orders for Mason Smith"
    re.compile(
        r"(?i:\b(?:account|orders?|purchases?|returns?|profile|history)\s+"
        r"(?:details\s+)?(?:for|of|belonging to)\s+)"
        rf"(?P<name>{_NAME})"
    ),
    # "orders placed by Mason Smith", "items bought by Mason Smith"
    re.compile(
        r"(?i:\b(?:placed|made|bought|purchased|ordered|submitted)\s+by\s+)"
        rf"(?P<name>{_NAME})"
    ),
    # "Mason Smith's orders", "Mason Smith's account"
    re.compile(
        rf"\b(?P<name>{_NAME})"
        r"(?i:['’]s\s+"
        r"(?:account|orders?|purchases?|returns?|profile|order history|history))"
    ),
    # "on behalf of Mason Smith"
    re.compile(rf"(?i:\bon behalf of\s+)(?P<name>{_NAME})"),
)

#: Phrases that are about the *absence* of access. A refusal legitimately names
#: the third party ("I can't look up Mason Smith's orders"), and must not be
#: flagged — otherwise the guard would reject the very message it wants.
_REFUSAL_CONTEXT = re.compile(
    r"can'?t|cannot|can not|unable to|not able to|don'?t have access"
    r"|do not have access|only access|only the account|no access"
    r"|not permitted|not authorised|not authorized|won'?t be able",
    re.IGNORECASE,
)


def strip_markdown(text: str) -> str:
    """Remove emphasis characters so phrases and names sit adjacent."""
    return _MARKDOWN_NOISE.sub("", text)


#: Words that appear in product titles and mark a candidate as merchandise
#: rather than a person. Only consulted in the fallback path used when no
#: customer predicate is supplied.
_PRODUCT_WORDS = frozenset(
    {
        "filter",
        "cable",
        "knob",
        "burner",
        "cooktop",
        "refrigerator",
        "washer",
        "dryer",
        "laptop",
        "computer",
        "monitor",
        "display",
        "case",
        "cover",
        "charger",
        "adapter",
        "battery",
        "screen",
        "drive",
        "card",
        "kit",
        "replacement",
        "replacements",
        "water",
        "door",
        "bin",
        "part",
        "parts",
        "headphones",
        "speaker",
        "camera",
        "lens",
        "printer",
        "keyboard",
        "mouse",
        "tablet",
        "phone",
        "watch",
        "band",
        "stand",
        "mount",
    }
)


def _name_variants(candidate: str) -> list[str]:
    """The candidate plus its trailing windows, longest first.

    The name pattern is greedy and a sentence-initial verb is capitalised, so
    "List Mason Smith's orders" captures "List Mason Smith". Testing the
    trailing two- and three-word windows recovers the actual name without
    loosening the pattern itself.
    """
    words = candidate.split()
    variants = [candidate]
    for size in range(len(words) - 1, 1, -1):
        variants.append(" ".join(words[-size:]))
    return variants


def _looks_like_person_name(candidate: str) -> bool:
    """Conservative structural test used when no customer lookup is available.

    Requires exactly two capitalised words, neither of which is a product noun.
    Deliberately strict: this path exists only as a degraded fallback, and a
    false positive here silently replaces a correct answer.
    """
    words = candidate.split()
    if len(words) != 2:
        return False
    return not any(word.lower() in _PRODUCT_WORDS for word in words)


def _names_match(candidate: str, holder: str) -> bool:
    """True when ``candidate`` refers to the account holder.

    Compares on the set of name words so "Danielle Johnson", "Johnson Danielle"
    and a middle name in either string all still count as the same person.
    """
    candidate_words = {w.lower() for w in re.findall(r"[A-Za-z]+", candidate)}
    holder_words = {w.lower() for w in re.findall(r"[A-Za-z]+", holder)}
    if not candidate_words or not holder_words:
        return False
    # Either name being a subset of the other is a match: the agent may use the
    # first name alone, or add a title.
    return candidate_words <= holder_words or holder_words <= candidate_words


def find_attributed_names(response: str) -> list[str]:
    """Return every person named as the owner of account data in ``response``."""
    cleaned = strip_markdown(response)
    found: list[str] = []
    for pattern in _ATTRIBUTION_PATTERNS:
        for match in pattern.finditer(cleaned):
            name = match.group("name").strip()

            # Skip when the sentence around the match is a refusal — naming the
            # person you are declining to look up is correct behaviour.
            start = max(0, match.start() - 120)
            if _REFUSAL_CONTEXT.search(cleaned[start : match.start()]):
                continue

            if name not in found:
                found.append(name)
    return found


#: Interrogative forms of a third-party data request, which the attribution
#: patterns above do not cover because they are phrased as questions rather
#: than as captions: "what has Mason Smith ordered recently?"
_THIRD_PARTY_QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "what has X ordered", and with an intervening noun: "what returns has X
    # filed", "how many orders did X place".
    re.compile(
        r"(?i:\b(?:what|which|how much|how many)\s+(?:\w+\s+){0,2}"
        r"(?:has|have|had|did|does|do)\s+)"
        rf"(?P<name>{_NAME})"
        r"(?i:\s+(?:order|buy|bought|purchas|return|spend|spent|receiv|file|"
        r"filed|place|placed|made|make))"
    ),
    re.compile(
        r"(?i:\b(?:look|pull|bring)\s+up\s+(?:the\s+)?(?:account|profile|orders?|"
        r"details?|history)?\s*(?:for|of)?\s*)"
        rf"(?P<name>{_NAME})"
    ),
    re.compile(
        r"(?i:\bswitch\s+to\s+)"
        rf"(?P<name>{_NAME})"
        r"(?i:['’]s\s+account|\s+account)"
    ),
)


def detect_third_party_request(
    message: str,
    account_holder: str,
    is_known_customer: Callable[[str], bool] | None = None,
) -> str:
    """Return the named third party when ``message`` asks about someone else.

    Checked on the customer's *input*, before any agent runs, so the refusal is
    deterministic rather than dependent on the model choosing to decline. Prior
    to this the behaviour was inconsistent: the same question sometimes drew a
    clear refusal and sometimes silently returned the signed-in customer's own
    orders, which invites the reader to assume the data belongs to the person
    they named.

    Only requests *for account data* are matched. "Order a gift for Jane Smith"
    is a shipping instruction, not a lookup, and must go through untouched.
    """
    if not message.strip() or not account_holder.strip():
        return ""

    cleaned = strip_markdown(message)
    candidates: list[str] = list(find_attributed_names(cleaned))
    for pattern in _THIRD_PARTY_QUESTION_PATTERNS:
        for match in pattern.finditer(cleaned):
            name = match.group("name").strip()
            if name not in candidates:
                candidates.append(name)

    for name in candidates:
        if _names_match(name, account_holder):
            continue
        resolved = _resolve_person(name, is_known_customer)
        if not resolved or _names_match(resolved, account_holder):
            continue
        _LOGGER.warning(
            "third-party request: %r asked about %r", account_holder, resolved
        )
        return resolved
    return ""


def _resolve_person(
    candidate: str,
    is_known_customer: Callable[[str], bool] | None,
) -> str:
    """Return the person ``candidate`` names, or "" if it names no one.

    Tries the candidate and its trailing windows so a captured sentence-initial
    verb ("List Mason Smith") still resolves to the real name.
    """
    for variant in _name_variants(candidate):
        if is_known_customer is not None:
            if is_known_customer(variant):
                return variant
        elif _looks_like_person_name(variant):
            return variant
    return ""


def detect_misattribution(
    response: str,
    account_holder: str,
    is_known_customer: Callable[[str], bool] | None = None,
) -> str:
    """Return the offending name when ``response`` credits data to someone else.

    Args:
        response: The customer-facing reply.
        account_holder: The signed-in customer's display name. When empty the
            check is skipped — without a known holder there is nothing to
            compare against, and guessing would produce false positives.
        is_known_customer: Predicate deciding whether a candidate name belongs
            to a real customer. Injected rather than queried directly so this
            module stays free of database access and testable without one.
            When omitted, only names that are *structurally* unambiguous
            person names are considered — see :func:`_looks_like_person_name`.

    Returns:
        The mis-attributed name, or "" when the reply is clean.
    """
    if not response.strip() or not account_holder.strip():
        return ""

    for name in find_attributed_names(response):
        if _names_match(name, account_holder):
            continue

        # Confirm the candidate is a person before rejecting the reply. Product
        # titles routinely satisfy the phrase patterns ("orders for Maytag
        # Refrigerator Water Filter"), and replacing a correct answer with a
        # refusal would be a worse bug than the one being guarded against.
        resolved = _resolve_person(name, is_known_customer)
        if not resolved or _names_match(resolved, account_holder):
            continue

        _LOGGER.error(
            "attribution guard: reply credited account data to %r while "
            "signed in as %r",
            resolved,
            account_holder,
        )
        return resolved
    return ""
