"""Typography normalisation for text that is parsed for identifiers.

Language models do not emit plain ASCII. Asked to mention order ``ORD-006762``
they routinely write ``ORD‑006762`` with U+2011 NON-BREAKING HYPHEN, along with
smart quotes and narrow no-break spaces. The characters are visually identical,
so the output looks correct while every regex written with an ASCII hyphen
fails against it.

That is not cosmetic. It broke the hand-off between specialists: the order
agent reported "your most recent order is ORD‑006762", the fact extractor
matched ``\\bORD-\\d{6}\\b`` against it, found nothing, and the return agent —
which had been handed no order id — asked the customer to supply one they had
just been told.

Anywhere model or user text is parsed for an identifier, run it through
:func:`normalise_typography` first.
"""

from __future__ import annotations

import re

__all__ = ["normalise_typography"]

#: Unicode characters that render like ASCII punctuation but are not.
_SUBSTITUTIONS = {
    # Hyphens and dashes -> "-"
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen — the one models actually emit
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "−": "-",  # minus sign
    "－": "-",  # fullwidth hyphen-minus
    # Quotes -> ' and "
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "′": "'",
    # Spaces -> " "
    " ": " ",  # no-break space
    " ": " ",  # figure space
    " ": " ",  # thin space
    " ": " ",  # narrow no-break space
    "​": "",   # zero-width space
    "﻿": "",   # zero-width no-break space
}

_TRANSLATION = str.maketrans(_SUBSTITUTIONS)

#: Thousands separators inside numbers, so "1,234.56" and "1234.56" compare equal.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalise_typography(text: str, *, strip_thousands: bool = False) -> str:
    """Replace look-alike Unicode punctuation with its ASCII equivalent.

    Args:
        text: The raw model or user text.
        strip_thousands: Also remove thousands separators from numbers. Off by
            default because it changes numeric content; useful when comparing
            an amount against a stored value.
    """
    if not text:
        return text
    normalised = text.translate(_TRANSLATION)
    if strip_thousands:
        normalised = _THOUSANDS.sub("", normalised)
    return normalised
