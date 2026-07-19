from src.agents.orchestrator.agent import SupportOrchestrator

agent = SupportOrchestrator()
print(agent.handle(user_message="What is the capital of France?"))