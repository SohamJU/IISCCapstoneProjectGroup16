# Memory Guide

This document explains how to use the runtime conversation memory layer in:
- `src/memory/conversation_memory.py`
- `src/memory/session_manager.py`

## What It Is

The memory layer stores short-term chat context while a user is actively interacting with the system.

- `ConversationMemory` stores the turns for one session
- `SessionManager` manages multiple `ConversationMemory` objects by `session_id`

This is runtime memory only.
It is not persistent storage and it is not a database-backed history system.

## Main Classes

### `ConversationMemory`

Purpose:
- store turns for a single conversation
- return recent turns
- build a prompt-ready context window

Main methods:
- `add_turn(...)`
- `get_recent_turns(...)`
- `build_context_window(...)`
- `clear()`

### `SessionManager`

Purpose:
- create or fetch session memory using a `session_id`
- append turns without manually managing `ConversationMemory`
- clear a session when needed

Main methods:
- `get_or_create_session(...)`
- `append_turn(...)`
- `clear_session(...)`

## Example Usage

```python
from src.memory.session_manager import SessionManager

session_manager = SessionManager()
session_id = "user_123"

session_manager.append_turn(
    session_id,
    role="customer",
    text="My order hasn't arrived yet.",
)

session_manager.append_turn(
    session_id,
    role="support",
    text="I can help with that. Can you share your order number?",
)

session_manager.append_turn(
    session_id,
    role="customer",
    text="It is ORD-9087.",
)

memory = session_manager.get_or_create_session(session_id)
print(memory.build_context_window())
```

Expected output:

```text
customer: My order hasn't arrived yet.
support: I can help with that. Can you share your order number?
customer: It is ORD-9087.
```

## Example With Recent Turns Limit

```python
from src.memory.session_manager import SessionManager

session_manager = SessionManager()
session_id = "user_456"

session_manager.append_turn(session_id, role="customer", text="My app keeps crashing.")
session_manager.append_turn(session_id, role="support", text="Which device are you using?")
session_manager.append_turn(session_id, role="customer", text="Samsung Galaxy Tab A.")
session_manager.append_turn(session_id, role="support", text="Thanks. Which app version are you on?")

memory = session_manager.get_or_create_session(session_id)
print(memory.build_context_window(limit=2))
```

Expected output:

```text
customer: Samsung Galaxy Tab A.
support: Thanks. Which app version are you on?
```

## How It Fits Into A Chat Flow

Typical flow:

1. User sends a message
2. Save the user turn with `append_turn(...)`
3. Build context using `build_context_window(...)`
4. Send that context to the LLM or agent prompt
5. Save the model reply with `append_turn(...)`
6. Repeat for the next user message

## Example Handler Pattern

```python
from src.memory.session_manager import SessionManager

session_manager = SessionManager()


def handle_message(session_id: str, user_message: str) -> str:
    session_manager.append_turn(
        session_id,
        role="customer",
        text=user_message,
    )

    memory = session_manager.get_or_create_session(session_id)
    recent_context = memory.build_context_window(limit=6)

    prompt = f"""
Recent conversation:
{recent_context}

Answer the customer's latest question helpfully.
"""

    llm_response = "Thanks for sharing that. Let me help you with the next step."

    session_manager.append_turn(
        session_id,
        role="support",
        text=llm_response,
    )

    return llm_response
```

## When To Use It

Use this layer when you want the assistant to remember things said earlier in the current session, for example:
- order ids
- device names
- issue history
- steps already tried
- follow-up questions like `I already did that`

## What It Does Not Do Yet

- it does not persist across app restarts
- it does not write to a database
- it is not connected to retrieval automatically
- it is not long-term user memory

## Relationship To Preprocessing And RAG

- preprocessing creates reusable historical knowledge from datasets
- memory stores the live current-session context
- RAG can later combine both:
  - retrieved historical knowledge from processed files
  - short-term live context from session memory
