"""Logging utilities.

A single configured logger factory so failures surface instead of being
swallowed. Several ``except Exception: pass`` blocks in the agent stack used
to hide real misconfiguration (missing API key, unreachable database) behind
canned "deterministic mode" replies, which made bad answers impossible to
diagnose from the UI.

Set ``SUPPORT_LOG_LEVEL=DEBUG`` for verbose agent tracing.
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_CONFIGURED = False


def _configure_root() -> None:
    """Attach a single stderr handler to the package root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("SUPPORT_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("src")
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)

    # Don't let langchain/httpx debug noise drown the agent logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    _configure_root()
    return logging.getLogger(name)
