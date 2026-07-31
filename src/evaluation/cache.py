"""Reuse of previously recorded agent answers, to avoid re-spending API quota.

Every evaluation case costs a live model call, and a full sweep is enough to
exhaust a free-tier daily allowance. This module lets a run reuse the answers
already recorded in ``output/evaluation/`` instead of asking the model the same
question again.

What is cached is the **answer and the tools called** — not the score. Checks
are re-evaluated against the stored answer on every run, so tightening a check
or adding one to an existing case costs nothing. Only a change to the *prompt*
(the query text or the identity it is asked under) forces a fresh call, because
only that changes what the model would say.

Staleness
---------
A cached answer reflects the agent code as it stood when it was recorded. If
the agent has since changed, the answer may no longer be what the system would
produce. That cannot be detected from the answer alone, so each entry records a
fingerprint of the agent source tree and the harness reports how many reused
answers predate the current code. It is a warning, not a refusal: the point of
the cache is to make a full sweep affordable, and the operator decides when a
refresh is warranted.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation.schema import CaseResult, EvalCase
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

CACHE_PATH = Path("output/evaluation/answer-cache.json")
REPORT_DIR = Path("output/evaluation")

#: Source that determines what an agent will answer. A change anywhere here
#: makes previously recorded answers potentially stale.
_FINGERPRINT_ROOTS = (Path("src/agents"),)


def prompt_hash(case: Any) -> str:
    """Hash the inputs that determine the model's answer.

    Deliberately excludes the checks: re-scoring a stored answer against a
    changed rubric is free, so only the prompt itself invalidates a cache entry.

    Accepts anything exposing ``query`` and ``customer_id`` — both
    :class:`~src.evaluation.schema.EvalCase` and
    :class:`~src.evaluation.routing.RoutingCase` qualify.
    """
    payload = json.dumps(
        {"query": case.query, "customer_id": getattr(case, "customer_id", None)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def code_fingerprint() -> str:
    """Fingerprint the agent source tree.

    Used only to tell the operator that reused answers predate the current
    code, never to silently discard them.
    """
    digest = hashlib.sha256()
    for root in _FINGERPRINT_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


class AnswerCache:
    """Stored answers keyed by ``(agent, case_id)``."""

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None) -> None:
        self._entries: dict[str, dict[str, Any]] = entries or {}
        self.hits: list[str] = []
        self.stale_code: list[str] = []
        self.unverified_prompt: list[str] = []

    # ── Construction ──────────────────────────────────────────────────────

    @staticmethod
    def _key(agent: str, case_id: str) -> str:
        return f"{agent}::{case_id}"

    @classmethod
    def load(
        cls,
        cache_path: Path = CACHE_PATH,
        report_dir: Path = REPORT_DIR,
    ) -> AnswerCache:
        """Load the cache, seeding it from any existing report JSONs.

        Seeding matters for the first run after this feature is added: answers
        already recorded in ``output/evaluation/`` were paid for and should not
        be paid for again. Those legacy records do not carry a prompt hash, so
        they are reused but reported as unverified.
        """
        entries: dict[str, dict[str, Any]] = {}

        # Oldest first, so newer records overwrite older ones.
        for report in sorted(report_dir.glob("evaluation-*.json")):
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                _LOGGER.warning("could not read %s: %s", report.name, exc)
                continue

            recorded_at = str(payload.get("generated_at", ""))
            for agent, block in (payload.get("agents") or {}).items():
                for result in block.get("results") or []:
                    # Skipped and provider-error records carry no usable answer.
                    if result.get("skipped") or not (result.get("answer") or "").strip():
                        continue
                    entries[cls._key(agent, str(result.get("id")))] = {
                        "agent": agent,
                        "case_id": result.get("id"),
                        "prompt_hash": None,  # legacy: query was not recorded
                        "code_fingerprint": None,
                        "recorded_at": recorded_at,
                        "source": report.name,
                        "answer": result.get("answer", ""),
                        "tools_used": result.get("tools_used") or [],
                        "latency_seconds": result.get("latency_seconds") or 0.0,
                        "error": result.get("error") or "",
                    }

        if cache_path.exists():
            try:
                stored = json.loads(cache_path.read_text(encoding="utf-8"))
                entries.update(stored.get("entries") or {})
            except (json.JSONDecodeError, OSError) as exc:
                _LOGGER.warning("could not read answer cache: %s", exc)

        _LOGGER.info("answer cache loaded with %d entries", len(entries))
        return cls(entries)

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, agent: str, case: Any) -> tuple[str, list[str], str] | None:
        """Return ``(answer, tools_used, recorded_at)`` for a reusable entry.

        Returns ``None`` when there is no entry, or when the prompt has changed
        since the entry was recorded.
        """
        entry = self._entries.get(self._key(agent, case.id))
        if not entry:
            return None
        if entry.get("error"):
            return None  # a failed call is not an answer worth reusing

        stored_hash = entry.get("prompt_hash")
        current_hash = prompt_hash(case)
        if stored_hash is None:
            # Legacy record: the query it was produced from is unknown.
            self.unverified_prompt.append(case.id)
        elif stored_hash != current_hash:
            return None  # the question changed; the old answer is meaningless

        if entry.get("code_fingerprint") not in (None, code_fingerprint()):
            self.stale_code.append(case.id)

        self.hits.append(case.id)
        return (
            str(entry.get("answer", "")),
            list(entry.get("tools_used") or []),
            str(entry.get("recorded_at", "")),
        )

    # ── Recording ─────────────────────────────────────────────────────────

    def put(self, agent: str, case: EvalCase, result: CaseResult) -> None:
        """Record a freshly obtained answer."""
        if result.skipped or result.error or not result.answer.strip():
            return
        self._entries[self._key(agent, case.id)] = {
            "agent": agent,
            "case_id": case.id,
            "prompt_hash": prompt_hash(case),
            "code_fingerprint": code_fingerprint(),
            "recorded_at": datetime.now(UTC).isoformat(),
            "source": "run",
            "answer": result.answer,
            "tools_used": result.tools_used,
            "latency_seconds": result.latency_seconds,
            "error": "",
        }

    def put_raw(self, agent: str, case: Any, answer: str, tools_used: list[str] | None = None) -> None:
        """Record an answer for anything case-shaped.

        Used by the routing evaluation, whose "answer" is the serialised list of
        predicted routes rather than prose. It only needs ``id``, ``query`` and
        ``customer_id`` from the case, which :class:`RoutingCase` also provides.
        """
        if not answer.strip():
            return
        self._entries[self._key(agent, case.id)] = {
            "agent": agent,
            "case_id": case.id,
            "prompt_hash": prompt_hash(case),
            "code_fingerprint": code_fingerprint(),
            "recorded_at": datetime.now(UTC).isoformat(),
            "source": "run",
            "answer": answer,
            "tools_used": tools_used or [],
            "latency_seconds": 0.0,
            "error": "",
        }

    def save(self, cache_path: Path = CACHE_PATH) -> Path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "updated_at": datetime.now(UTC).isoformat(),
                    "code_fingerprint": code_fingerprint(),
                    "entries": self._entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return cache_path

    def __len__(self) -> int:
        return len(self._entries)
