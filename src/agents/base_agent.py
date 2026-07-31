"""Base agent abstraction shared by every specialist agent.

Why this exists
---------------
The five specialist agents were near-identical ~95-line copies of each other,
each carrying the same two defects:

1. **Its own ``MemorySaver`` keyed on ``session_id``.** Combined with the
   orchestrator prepending a ``CONVERSATION_HISTORY:`` block to every message,
   each agent saw its own private history *plus* a flattened transcript of a
   conversation it had only partly taken part in — duplicated and often
   contradictory context.

2. **Subtask scoping concatenated into the user turn.** The instruction
   "Subtask for Product Agent: Focus only on product facts..." was stored in
   the agent's history as if the *customer* had said it, so by turn three the
   model saw the user repeatedly issuing meta-instructions to itself.

Both are fixed here. A :class:`SpecialistAgent` is **stateless**: it holds no
checkpointer and no history. The supervisor graph owns the single canonical
message list and passes the relevant slice in on each call. Per-turn scoping
arrives as a ``SystemMessage``, which never pollutes the conversation record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from src.agents.common import validate_agent_output
from src.agents.llm import get_llm
from src.utils.logger import get_logger

_LOGGER = get_logger(__name__)

#: How many prior turns a specialist sees. Enough to resolve references
#: without paying for the whole conversation on every tool-calling step.
DEFAULT_HISTORY_WINDOW = 8


@dataclass
class AgentResult:
    """Outcome of one specialist invocation."""

    name: str
    text: str
    tool_calls: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        """True when the agent produced a usable answer."""
        return not self.error and bool(self.text.strip())


class SpecialistAgent:
    """A stateless LangGraph ReAct agent for one support domain.

    Parameters
    ----------
    name
        Route label, e.g. ``"product"``.
    system_prompt
        The agent's standing role prompt.
    tools
        LangChain tools bound to the model.
    max_iterations
        Safety cap on ReAct cycles, converted to a recursion limit.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list,
        max_iterations: int = 8,
        session_id: str = "default",
        debug: bool = False,
    ) -> None:
        self.name = name
        self.session_id = session_id
        self.max_iterations = max_iterations
        self.debug = debug
        self._system_prompt = system_prompt
        self._tools = tools

        # NOTE: deliberately no checkpointer. Conversation state lives in the
        # supervisor graph, which is the single source of truth.
        self._agent = create_react_agent(
            model=get_llm(),
            tools=tools,
            prompt=system_prompt,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        messages: list[AnyMessage],
        scope_instruction: str = "",
        history_window: int = DEFAULT_HISTORY_WINDOW,
        customer_id: str | None = None,
    ) -> AgentResult:
        """Invoke the agent over a slice of the canonical conversation.

        Parameters
        ----------
        messages
            The canonical conversation from graph state. The last entry is
            expected to be the current user message.
        scope_instruction
            Per-turn guidance injected as a ``SystemMessage`` — never appended
            to the user's own text.
        history_window
            How many trailing messages to pass through.
        customer_id
            The authenticated customer. Placed in ``configurable`` so that
            customer-scoped tools can enforce ownership against a value the
            model cannot see or set — see :mod:`src.agents.authz`. Passing it
            in the scope prose alone was the bug that let one customer read
            another's orders.

        Returns
        -------
        AgentResult
            The agent's answer plus the tools it invoked.
        """
        window = messages[-history_window:] if history_window > 0 else list(messages)

        payload: list[AnyMessage] = []
        if scope_instruction.strip():
            payload.append(SystemMessage(content=scope_instruction.strip()))
        payload.extend(window)

        config: RunnableConfig = {
            "recursion_limit": max(4, self.max_iterations * 2),
            "configurable": {"customer_id": customer_id},
        }

        try:
            result = self._agent.invoke({"messages": payload}, config=config)
        except Exception as exc:
            _LOGGER.error("[%s] agent invocation failed: %s", self.name, exc)
            return AgentResult(
                name=self.name,
                text="",
                error=f"{type(exc).__name__}: {exc}",
            )

        out_messages = result.get("messages", [])
        tool_calls = self._collect_tool_calls(out_messages)

        final_text = ""
        for message in reversed(out_messages):
            if isinstance(message, AIMessage) and str(message.content).strip():
                final_text = str(message.content).strip()
                break

        if self.debug:
            _LOGGER.debug(
                "[%s] tools=%s output=%s", self.name, tool_calls, final_text[:300]
            )

        valid, guardrail_message = validate_agent_output(final_text)
        if not valid:
            return AgentResult(
                name=self.name,
                text="",
                tool_calls=tool_calls,
                error=guardrail_message,
            )

        return AgentResult(name=self.name, text=final_text, tool_calls=tool_calls)

    def chat(self, user_message: str, customer_id: str | None = None) -> str:
        """Single-shot convenience wrapper.

        Retained so the agents remain usable standalone (tests, notebooks,
        the individual-agent demos). Production traffic goes through
        :meth:`run` via the supervisor graph.

        ``customer_id`` defaults to ``None``, which means customer-scoped tools
        fail closed. A standalone caller that wants order data must say who it
        is acting as, exactly like the graph does.
        """
        from langchain_core.messages import HumanMessage

        from src.agents.common import validate_user_input

        ok, error = validate_user_input(user_message)
        if not ok:
            return error

        result = self.run([HumanMessage(content=user_message)], customer_id=customer_id)
        if not result.ok:
            return (
                result.error
                or "I wasn't able to generate a response. Please try again."
            )
        return result.text

    def reset_memory(self) -> None:
        """No-op — specialists are stateless by design.

        Kept for API compatibility with the previous implementation. Clearing
        conversation state is now the supervisor graph's responsibility.
        """
        return None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _collect_tool_calls(messages: list[AnyMessage]) -> list[str]:
        """Return the names of tools invoked during this run, in order."""
        names: list[str] = []
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                name = (
                    call.get("name")
                    if isinstance(call, dict)
                    else getattr(call, "name", None)
                )
                if name:
                    names.append(str(name))
        return names
