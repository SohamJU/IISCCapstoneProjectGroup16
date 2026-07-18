import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.agents.escalation_agent.agent import EscalationAgent

agent = EscalationAgent()
result = print(agent.chat('I need help with an unusual billing issue and a human specialist'))