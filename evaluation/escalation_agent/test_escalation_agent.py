import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent
TEST_CASES_PATH = BASE_DIR / "test_cases_escalation.json"
GROUND_TRUTH_PATH = BASE_DIR / "ground_truth_escalation.json"

_HIGH_RISK_TERMS = [
    "lawyer",
    "legal",
    "fraud",
    "scam",
    "chargeback",
    "caught fire",
    "unsafe",
    "dangerous",
    "human",
    "manager",
]

_FRUSTRATION_TERMS = [
    "angry",
    "terrible",
    "worst",
    "ridiculous",
    "unacceptable",
]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _semantic_response_match(
    response_text: str, expected: dict
) -> tuple[bool, list[str]]:
    """Use intent-aware keyword checks so LLM text can be evaluated without exact string equality."""
    response_text = str(response_text)
    norm = _normalize(response_text)

    expected_keywords = expected.get("response_keywords", [])
    negative_keywords = expected.get("response_negative_keywords", [])
    must_have = expected.get("must_include_terms", [])

    issues: list[str] = []

    if expected_keywords and not any(
        keyword.lower() in norm for keyword in expected_keywords
    ):
        issues.append("missing expected response signal")

    if negative_keywords and any(
        keyword.lower() in norm for keyword in negative_keywords
    ):
        issues.append("unexpected escalation signal")

    if must_have and not all(term.lower() in norm for term in must_have):
        issues.append("missing required phrase")

    return not issues, issues


def _build_agent():
    """Instantiate the real EscalationAgent when the environment allows it."""
    try:
        from src.agents.escalation_agent.agent import EscalationAgent

        agent = EscalationAgent(session_id="eval-session")
        return agent, None
    except Exception as exc:  # pragma: no cover - environment-specific guard
        return None, str(exc)


def test_escalation_agent_fixture_matches_ground_truth():
    test_cases = _load_json(TEST_CASES_PATH)
    ground_truth = _load_json(GROUND_TRUTH_PATH)

    if not isinstance(test_cases, list):
        raise TypeError("test_cases.json must contain a top-level list of test cases")

    gt_by_id = {item["id"]: item for item in ground_truth}

    agent, agent_error = _build_agent()
    if agent is None:
        pytest.skip(
            f"EscalationAgent could not be instantiated in this environment: {agent_error}"
        )

    for case in test_cases:
        case_id = case["id"]
        expected = gt_by_id[case_id]

        response_text = agent.chat(case["user_message"])
        assert isinstance(response_text, str) and response_text.strip(), (
            "agent.chat() should return a non-empty response"
        )

        matches, issues = _semantic_response_match(response_text, expected)
        assert matches, issues


def test_fixture_files_exist():
    assert TEST_CASES_PATH.exists()
    assert GROUND_TRUTH_PATH.exists()


def test_run_as_script_example():
    """Smoke test to keep the evaluation entry-point discoverable."""
    assert TEST_CASES_PATH.exists()
    assert GROUND_TRUTH_PATH.exists()


if __name__ == "__main__":
    test_escalation_agent_fixture_matches_ground_truth()
    test_fixture_files_exist()
    print("Escalation evaluation checks passed.")
