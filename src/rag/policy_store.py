"""Section-aware local retrieval over the policy knowledge base.

Why this exists
---------------
``PINECONE_API_KEY`` is not set in this deployment, so every call into
:mod:`src.rag.retriever` raises and the return agent's policy tool silently
fell through to a naive line scan: it OR-matched any query token longer than
two characters against every line and returned up to eight orphan bullet
lines. The agent received fragments with no headings and no context, so it
either hedged or invented policy terms.

The whole knowledge base is only ~390 lines across four markdown files, so a
local index is both practical and better than the fallback it replaces:

* Documents are split on ``##`` headings, so a match returns a **coherent
  section** ("## Return Window" with all its bullets) rather than one line.
* Scoring is TF-IDF-ish with heading boosts, so "how many days to return"
  ranks the Return Window section above an incidental mention elsewhere.
* Pinecone is still preferred when it is configured — this is the fallback,
  not a replacement.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tokens that carry no retrieval signal for policy questions.
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being do does did doing have has had i me my
    we our you your it its this that these those of to in on for with at by from
    as and or if then than so but not no can could should would will shall may
    might must about into over under again further once here there when where why
    how all any both each few more most other some such only own same too very
    s t just don now what which who whom whose am
    """.split()
)

# Query words that hint at a policy area, used to boost the right document.
_AREA_HINTS: dict[str, tuple[str, ...]] = {
    "return_policy": ("return", "returns", "refund", "refunds", "exchange", "restock", "rma"),
    "shipping_policy": ("ship", "shipping", "delivery", "deliver", "courier", "tracking", "dispatch"),
    "warranty_policy": ("warranty", "guarantee", "repair", "defect", "defective", "faulty", "coverage"),
    "payment_policy": ("payment", "pay", "card", "invoice", "billing", "charge", "emi", "installment"),
}


@dataclass(frozen=True)
class PolicySection:
    """One retrievable chunk of a policy document."""

    doc: str
    heading: str
    text: str

    @property
    def source(self) -> str:
        """Human-readable citation, e.g. ``return_policy.md > Return Window``."""
        return f"{self.doc} > {self.heading}" if self.heading else self.doc


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _split_sections(doc_name: str, raw: str) -> list[PolicySection]:
    """Split a markdown policy doc into ``##``-delimited sections.

    The document preamble (text before the first ``##``) is kept as its own
    section so the top-level intro is still retrievable.
    """
    sections: list[PolicySection] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(PolicySection(doc=doc_name, heading=heading, text=body))

    for line in raw.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            buffer = []
        elif line.startswith("# "):
            # Title line — treat as the preamble heading.
            flush()
            heading = line[2:].strip()
            buffer = []
        else:
            buffer.append(line)

    flush()
    return sections


class PolicyStore:
    """In-memory TF-IDF index over the policy knowledge base."""

    def __init__(self, knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR) -> None:
        self.knowledge_base_dir = knowledge_base_dir
        self.sections: list[PolicySection] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._heading_tokens: list[set[str]] = []
        self._load()

    # ── Index construction ────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.knowledge_base_dir.exists():
            _LOGGER.warning(
                "Knowledge base directory not found: %s", self.knowledge_base_dir
            )
            return

        for path in sorted(self.knowledge_base_dir.glob("*.md")):
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError as exc:
                _LOGGER.warning("Could not read policy doc %s: %s", path.name, exc)
                continue
            self.sections.extend(_split_sections(path.name, raw))

        for section in self.sections:
            tokens = _tokenize(f"{section.heading} {section.text}")
            freqs = Counter(tokens)
            self._term_freqs.append(freqs)
            self._doc_freq.update(freqs.keys())
            self._heading_tokens.append(set(_tokenize(section.heading)))

        _LOGGER.info(
            "Policy store indexed %d sections from %s",
            len(self.sections),
            self.knowledge_base_dir,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[tuple[PolicySection, float]]:
        """Return the top-scoring policy sections for a query."""
        query_tokens = _tokenize(query)
        if not query_tokens or not self.sections:
            return []

        total = len(self.sections)
        area_boosts = self._area_boosts(query_tokens)
        scored: list[tuple[PolicySection, float]] = []

        for index, section in enumerate(self.sections):
            freqs = self._term_freqs[index]
            if not freqs:
                continue

            length_norm = math.sqrt(sum(freqs.values()))
            score = 0.0
            matched = 0

            for token in set(query_tokens):
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                matched += 1
                idf = math.log((1 + total) / (1 + self._doc_freq[token])) + 1.0
                score += (tf / length_norm) * idf
                # A hit in the heading is a much stronger signal than in the body.
                if token in self._heading_tokens[index]:
                    score += 0.6 * idf

            if not matched:
                continue

            # Reward sections matching more of the query, and the right document.
            score *= 1.0 + (matched / len(set(query_tokens)))
            score *= area_boosts.get(section.doc, 1.0)
            scored.append((section, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(1, top_k)]

    @staticmethod
    def _area_boosts(query_tokens: list[str]) -> dict[str, float]:
        """Boost documents whose topic matches the query's vocabulary."""
        token_set = set(query_tokens)
        boosts: dict[str, float] = {}
        for stem, hints in _AREA_HINTS.items():
            if token_set.intersection(hints):
                boosts[f"{stem}.md"] = 1.75
        return boosts


@lru_cache(maxsize=1)
def get_policy_store() -> PolicyStore:
    """Return the process-wide policy store singleton."""
    return PolicyStore()


def format_policy_sections(results: list[tuple[PolicySection, float]]) -> str:
    """Render retrieved sections as labelled, citable blocks for the LLM."""
    if not results:
        return (
            "No matching policy section found. Ask the customer to rephrase, or "
            "state that you need to check with a specialist — do NOT guess policy terms."
        )

    blocks: list[str] = []
    for section, score in results:
        blocks.append(
            f"[source: {section.source} | relevance: {score:.2f}]\n{section.text}"
        )
    return "\n\n---\n\n".join(blocks)
