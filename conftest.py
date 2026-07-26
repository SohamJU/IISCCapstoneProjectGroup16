import os
import sys

import pytest

# Ensure the project root is on sys.path so `src` is importable in tests
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _disable_ml_guardrail_in_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the ML guardrail scanner for all tests by default.

    The bundled PromptInjection model produces false positives on legitimate
    e-commerce support messages (e.g. "Return my order" scores 1.0), which
    would break tests that aren't specifically testing guardrail behaviour.

    Tests that *do* want the ML scanner (i.e. tests/test_ml_guardrail.py)
    use their own autouse fixture to re-enable it by patching ``_ENABLE``
    directly, which takes precedence over this module-level patch.
    """
    try:
        import src.agents.guardrails.ml_input_scanner as scanner_mod
        monkeypatch.setattr(scanner_mod, "_ENABLE", False)
    except ImportError:
        pass  # llm-guard not installed — scanner is already a no-op
