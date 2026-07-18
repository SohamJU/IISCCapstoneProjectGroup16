import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.agents.router_agent.agent import RouterAgent

agent = RouterAgent()   
print(agent.chat("Can you recommend a good laptop for programming under $1000?"))
print(agent.chat("This has been the worst experience ever! "))
print(agent.chat("Can you help me track my order #ORD-000001?"))
print(agent.chat("I want to return my order #ORD-000005"))