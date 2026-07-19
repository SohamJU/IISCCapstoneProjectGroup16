"""
EXAMPLE: How to Integrate Session Persistence into Your Orchestrator

This file shows the exact changes needed to src/agents/orchestrator/agent.py
"""

# ============================================================================
# ORIGINAL CODE (CURRENT - IN-MEMORY ONLY)
# ============================================================================

"""
from src.memory.session_manager import SessionManager

class SupportOrchestrator:
    def __init__(self, ...):
        self.router = RouterAgent(...)
        ...
        self._session_manager = SessionManager()  # ← IN-MEMORY ONLY
        
    def handle(self, user_message: str, session_id: str = "default"):
        session = self._session_manager.get_or_create_session(session_id)
        # ... rest of implementation
"""


# ============================================================================
# MODIFIED CODE (WITH PERSISTENCE)
# ============================================================================

"""
from src.memory.session_manager import SessionManager
from src.memory.persistent_session_manager import PersistentSessionManager  # ← NEW
from src.utils.customer_history import enrich_message_with_history  # ← NEW

class SupportOrchestrator:
    def __init__(self, 
                 use_llm_router_fallback: bool = True,
                 deterministic_mode: bool | None = None,
                 auto_fallback_on_agent_init_error: bool = True,
                 enable_session_persistence: bool = True):  # ← NEW PARAMETER
        self.router = RouterAgent(use_llm_fallback=use_llm_router_fallback)
        ...
        # Use persistent session manager if enabled
        if enable_session_persistence:
            self._session_manager = PersistentSessionManager(persist_to_db=True)
        else:
            self._session_manager = SessionManager()
        
    def handle(self, 
               user_message: str, 
               session_id: str = "default",
               customer_id: str | None = None,  # ← NEW PARAMETER
               use_history: bool = True):  # ← NEW PARAMETER
        
        # Get or create session with customer context
        session = self._session_manager.get_or_create_session(
            session_id=session_id,
            customer_id=customer_id  # ← PASS CUSTOMER ID
        )
        
        # Enrich message with customer history (NEW)
        if use_history and customer_id:
            enriched_message = enrich_message_with_history(
                user_message=user_message,
                customer_id=customer_id,
                include_history=True,
                num_sessions=2,
                max_turns_per_session=3
            )
        else:
            enriched_message = user_message
        
        # Get history context
        history_context = session.build_context_window(limit=6)
        if history_context:
            full_context = enriched_message if enriched_message == user_message else enriched_message
        else:
            full_context = enriched_message
        
        # ... rest of routing logic ...
        
        # Save turns to database (NEW)
        if customer_id:
            self._session_manager.append_turn(
                session_id=session_id,
                role="user",
                text=user_message,
                customer_id=customer_id  # ← PASS CUSTOMER ID
            )
            
            self._session_manager.append_turn(
                session_id=session_id,
                role="assistant",
                text=result.response,
                customer_id=customer_id  # ← PASS CUSTOMER ID
            )
        
        return OrchestratorResponse(
            route=route,
            confidence=confidence,
            response=response,
            routes=routes,
        )
"""


# ============================================================================
# MINIMAL CHANGES APPROACH
# ============================================================================

"""
If you want to make minimal changes, here's the simplest integration:

1. Change this line:
   self._session_manager = SessionManager()
   
   To this:
   self._session_manager = PersistentSessionManager(persist_to_db=True)

2. Update the handle() method signature:
   def handle(self, user_message: str, session_id: str = "default", customer_id: str | None = None)

3. Pass customer_id to get_or_create_session:
   session = self._session_manager.get_or_create_session(session_id, customer_id)

4. Pass customer_id when appending turns:
   self._session_manager.append_turn(session_id, role="user", text=..., customer_id=customer_id)
   self._session_manager.append_turn(session_id, role="assistant", text=..., customer_id=customer_id)

That's it! Everything else stays the same.
"""


# ============================================================================
# GRADIO APP CHANGES
# ============================================================================

"""
Original Gradio App:

def _respond(message: str, history: list[dict[str, str]] | None, session_id: str) -> str:
    result = ORCHESTRATOR.handle(message, session_id=session_id or "gradio-default")
    ...

def build_app() -> gr.Blocks:
    with gr.Blocks(...) as demo:
        session_id = gr.Textbox(value="gradio-user-1", label="Session ID", ...)
        gr.ChatInterface(fn=_respond, additional_inputs=[session_id], ...)


Updated Gradio App with Persistence:

def _respond(message: str, 
             history: list[dict[str, str]] | None, 
             session_id: str,
             customer_id: str) -> str:  # ← ADD customer_id parameter
    
    # Use persistent session manager
    result = ORCHESTRATOR.handle(
        message, 
        session_id=session_id or "gradio-default",
        customer_id=customer_id or "anonymous",  # ← PASS customer_id
        use_history=True  # ← ENABLE history enrichment
    )
    ...

def build_app() -> gr.Blocks:
    with gr.Blocks(...) as demo:
        # Add customer_id input
        customer_id = gr.Textbox(
            value="gradio-user-1",
            label="Customer ID",
            info="Your customer identifier for tracking conversation history"
        )
        
        session_id = gr.Textbox(
            value="gradio-user-1-session",
            label="Session ID",
            info="Unique session identifier"
        )
        
        gr.ChatInterface(
            fn=_respond,
            additional_inputs=[session_id, customer_id],  # ← ADD customer_id
            ...
        )
"""


# ============================================================================
# COMPLETE WORKING EXAMPLE
# ============================================================================

"""
Below is a complete, working example showing integration:
"""

from src.agents.orchestrator.agent import SupportOrchestrator
from src.memory.persistent_session_manager import PersistentSessionManager
from src.utils.customer_history import enrich_message_with_history
import gradio as gr


# Initialize with persistence
ORCHESTRATOR = SupportOrchestrator()
ORCHESTRATOR._session_manager = PersistentSessionManager(persist_to_db=True)


def _respond(message: str, 
             history: list[dict[str, str]] | None, 
             session_id: str,
             customer_id: str) -> str:
    """Handle one chat turn with full persistence."""
    
    # Use provided IDs or defaults
    session_id = session_id or f"session-{customer_id}-default"
    customer_id = customer_id or "anonymous"
    
    # Get or create session
    session = ORCHESTRATOR._session_manager.get_or_create_session(
        session_id=session_id,
        customer_id=customer_id
    )
    
    # Enrich with history
    enriched_message = enrich_message_with_history(
        user_message=message,
        customer_id=customer_id,
        include_history=True,
        num_sessions=2,
        max_turns_per_session=3
    )
    
    # Handle request
    result = ORCHESTRATOR.handle(
        enriched_message,
        session_id=session_id,
        customer_id=customer_id,
        use_history=False  # Already enriched above
    )
    
    # Save to database
    ORCHESTRATOR._session_manager.append_turn(
        session_id=session_id,
        role="user",
        text=message,
        customer_id=customer_id
    )
    
    ORCHESTRATOR._session_manager.append_turn(
        session_id=session_id,
        role="assistant",
        text=result.response,
        customer_id=customer_id
    )
    
    # Return formatted response
    routes_text = ",".join(result.routes) if result.routes else result.route
    return f"[routes={routes_text} confidence={result.confidence:.2f}]\n{result.response}"


def build_app() -> gr.Blocks:
    """Build Gradio app with persistence support."""
    with gr.Blocks(title="Agentic Customer Support - With Session History") as demo:
        gr.Markdown("# Agentic Customer Support")
        gr.Markdown(
            "Product, Order, Return, Recommendation, and Escalation flows routed through a central router. "
            "**Your conversation history is automatically saved and retrieved across sessions.**"
        )

        # Customer ID
        customer_id = gr.Textbox(
            value="customer-001",
            label="Customer ID",
            info="Your unique customer identifier. Same ID = same conversation history.",
        )

        # Session ID
        session_id = gr.Textbox(
            value="session-001",
            label="Session ID",
            info="Unique identifier for this chat session.",
        )

        # Chat interface
        gr.ChatInterface(
            fn=_respond,
            additional_inputs=[session_id, customer_id],
            title="Support Assistant",
            description="Ask about products, orders, returns, recommendations, or escalation. "
                       "Your conversation will be saved in our database.",
        )

        # Information box
        gr.Markdown(
            """
            ---
            ## How This Works
            
            - **Customer ID**: Identifies you. Same ID = access to your full history
            - **Session ID**: Unique to each conversation
            - **History**: Previous conversations are automatically included for context
            - **Storage**: Everything is saved to PostgreSQL
            - **Limit**: We keep your 5 most recent sessions
            
            Try:
            1. Start a conversation with customer_id="alice"
            2. Send a message about a product
            3. Refresh the page (keep same customer_id)
            4. Start a new session - the chatbot will remember!
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()


# ============================================================================
# STEP-BY-STEP CHECKLIST
# ============================================================================

"""
To integrate session persistence into your project:

Step 1: Import New Classes
    ☐ Add: from src.memory.persistent_session_manager import PersistentSessionManager
    ☐ Add: from src.utils.customer_history import enrich_message_with_history

Step 2: Update Orchestrator __init__
    ☐ Replace: self._session_manager = SessionManager()
    ☐ With: self._session_manager = PersistentSessionManager(persist_to_db=True)

Step 3: Update Orchestrator handle() method
    ☐ Add customer_id parameter: session_id: str = "default", customer_id: str | None = None
    ☐ Pass customer_id to get_or_create_session()
    ☐ Pass customer_id when appending user turn
    ☐ Pass customer_id when appending assistant turn

Step 4: Update Gradio App
    ☐ Import PersistentSessionManager and enrich_message_with_history
    ☐ Add customer_id parameter to _respond function
    ☐ Add customer_id input field to UI
    ☐ Pass customer_id to orchestrator.handle()
    ☐ Use enrich_message_with_history() if desired

Step 5: Initialize Database
    ☐ Run: python -m pipelines.setup_sessions_table
    ☐ Verify: Table created in PostgreSQL

Step 6: Test
    ☐ Run: pytest tests/test_session_persistence.py
    ☐ Manual testing: python tests/test_session_persistence.py
    ☐ Try Gradio app with persistence

Step 7: Deploy
    ☐ Update your production database
    ☐ Deploy new code
    ☐ Monitor customer session retrieval
"""


# ============================================================================
# OPTIONAL ENHANCEMENTS
# ============================================================================

"""
After basic integration, you can add:

1. Customer History Dashboard
   - Show all sessions for a customer
   - Display conversation statistics
   - Analyze topics discussed

2. Sentiment Analysis
   - Track customer sentiment over time
   - Identify frustrated customers
   - Trigger proactive support

3. Smart Context Selection
   - Automatically select most relevant past sessions
   - Use embeddings to find similar conversations
   - Provide targeted context to agents

4. Session Tagging
   - Tag sessions by issue type
   - Create session templates
   - Build knowledge base from sessions

5. Performance Monitoring
   - Track average session length
   - Monitor customer satisfaction
   - Measure agent effectiveness

See docs/customer_session_persistence.md for details on each enhancement.
"""
