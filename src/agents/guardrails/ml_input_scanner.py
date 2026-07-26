"""ML-based input scanner using LLM Guard.

Exposes :func:`scan_input` which returns the same ``(bool, str)`` contract as
:func:`src.agents.common.validate_user_input` so it can be composed with it
transparently.

Configuration (environment variables)
--------------------------------------
ENABLE_ML_GUARDRAIL
    Set to ``true`` to activate ML scanning.  Defaults to ``true``.

    .. warning::
        The bundled PromptInjection model produces false positives on
        legitimate e-commerce support messages (e.g. "Return this and
        reorder it" scores 1.0).  Enable only after tuning
        ``ML_GUARDRAIL_THRESHOLD`` against a sample of real customer
        messages from this domain.

ML_GUARDRAIL_THRESHOLD
    Confidence threshold for the PromptInjection scanner (0.0–1.0).
    Defaults to ``0.8``.  Lower values catch more attacks but raise
    false-positive rate.  Tune against a sample of real customer messages
    before enabling in a live demo.

ENABLE_TOXICITY_GUARDRAIL
    Set to ``true`` to also scan for toxic/abusive language.  Requires
    ``ENABLE_ML_GUARDRAIL=true``.  Defaults to ``false``.

TOXICITY_GUARDRAIL_THRESHOLD
    Confidence threshold for the Toxicity scanner (0.0–1.0).  Defaults to
    ``0.7`` (deliberately higher than the injection threshold to avoid
    flagging casual frustrated language like "this is terrible service").

Design notes
------------
* The scanner pipeline is built **once per process** via
  :func:`functools.lru_cache` so the ``~250 MB`` BERT weights are loaded only
  on the first call and reused for every subsequent request.
* On import error (``llm-guard`` not installed) or runtime error the scanner
  **fails open** — it returns ``(True, "")`` and logs a warning.  The regex
  layer in :func:`~src.agents.common.validate_user_input` has already run at
  that point, so the application degrades gracefully rather than crashing.
* All inference runs on the local machine (CPU or GPU).  When the app is
  shared via ``gradio launch(share=True)`` the ``gradio.live`` tunnel only
  proxies HTTP; the scanner executes here, not on Gradio's servers.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

_LOGGER = logging.getLogger(__name__)

_ENABLE: bool = os.getenv("ENABLE_ML_GUARDRAIL", "false").lower() == "true"
_THRESHOLD: float = float(os.getenv("ML_GUARDRAIL_THRESHOLD", "0.8"))
_ENABLE_TOXICITY: bool = os.getenv("ENABLE_TOXICITY_GUARDRAIL", "false").lower() == "true"
_TOXICITY_THRESHOLD: float = float(os.getenv("TOXICITY_GUARDRAIL_THRESHOLD", "0.7"))


@lru_cache(maxsize=1)
def _build_scanner_pipeline():
    """Lazily build and cache the LLM Guard scanner pipeline.

    Downloads model weights from the HuggingFace Hub on first call
    (~250 MB, cached in ``~/.cache/huggingface``).  Subsequent calls return
    the cached ``(scan_prompt, scanners)`` tuple instantly.

    Returns
    -------
    tuple[callable, list]
        ``(scan_prompt, scanners)`` ready to call as
        ``scan_prompt(scanners, message)``.

    Raises
    ------
    ImportError
        If ``llm-guard`` is not installed.  The caller catches this.
    """
    from llm_guard import scan_prompt  # type: ignore[import]
    from llm_guard.input_scanners import PromptInjection  # type: ignore[import]

    scanners = [PromptInjection(threshold=_THRESHOLD)]

    if _ENABLE_TOXICITY:
        from llm_guard.input_scanners import Toxicity  # type: ignore[import]
        scanners.append(Toxicity(threshold=_TOXICITY_THRESHOLD))
        _LOGGER.info(
            "ML guardrail: PromptInjection(threshold=%.2f) + Toxicity(threshold=%.2f)",
            _THRESHOLD,
            _TOXICITY_THRESHOLD,
        )
    else:
        _LOGGER.info(
            "ML guardrail: PromptInjection(threshold=%.2f)", _THRESHOLD
        )

    return scan_prompt, scanners


def scan_input(message: str) -> tuple[bool, str]:
    """Run ML scanners against a user message.

    Parameters
    ----------
    message:
        The raw user text to scan (already stripped of leading/trailing
        whitespace by the caller).

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when the message is safe, or ``(False, rejection)``
        when a scanner fires.  Falls back silently to ``(True, "")`` if the
        scanner cannot be loaded or fails at runtime, so the application is
        never blocked entirely by a guardrail error.
    """
    if not _ENABLE:
        return True, ""

    try:
        scan_prompt, scanners = _build_scanner_pipeline()
        _sanitized, results, _scores = scan_prompt(scanners, message)
    except Exception as exc:
        # Fail open: the regex layer in validate_user_input has already run.
        # A guardrail crash should not take down the whole support system.
        _LOGGER.warning(
            "ML guardrail scanner failed (failing open, regex layer still active): %s",
            exc,
        )
        return True, ""

    for scanner_name, is_valid in results.items():
        if not is_valid:
            _LOGGER.info(
                "ML guardrail blocked message: scanner=%s message_preview=%.80r",
                scanner_name,
                message,
            )
            name_lower = scanner_name.lower()
            if "injection" in name_lower:
                return False, (
                    "I can't follow instructions that try to override system rules. "
                    "Please ask a normal product or support question."
                )
            if "toxicity" in name_lower:
                return False, (
                    "I'm unable to process that message. "
                    "Please keep your request respectful."
                )
            # Catch-all for any future scanners added to the pipeline.
            return False, (
                "I can't process that request. Please try rephrasing."
            )

    return True, ""
