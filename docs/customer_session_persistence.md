"""Documentation for Customer Session Persistence Feature"""

# Customer Session Persistence for PostgreSQL

## Overview

This feature enables persistent storage of customer conversation sessions in PostgreSQL (Aiven). When a customer returns to the chatbot, the system can retrieve their previous session history (limited to 5 most recent sessions) to provide context-aware support.

## Architecture

### Components

1. **Database Layer** (`src/data/session_persistence.py`)
   - Low-level functions for database operations
   - Table creation, CRUD operations on sessions
   - Session history retrieval and cleanup

2. **Session Manager** (`src/memory/persistent_session_manager.py`)
   - Extends in-memory session manager with database persistence
   - Handles automatic saving of conversations
   - Retrieves previous sessions for context

3. **History Utilities** (`src/utils/customer_history.py`)
   - Formats customer history for LLM context
   - Analyzes customer topics and sentiment
   - Enriches messages with historical context

4. **Migration Script** (`pipelines/setup_sessions_table.py`)
   - Creates the customer_sessions table
   - Sets up indexes for performance

## Database Schema

```sql
CREATE TABLE customer_sessions (
    id SERIAL PRIMARY KEY,
    customer_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL UNIQUE,
    conversation_turns JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    INDEX idx_customer_id (customer_id),
    INDEX idx_created_at (created_at DESC),
    INDEX idx_is_active (is_active)
);
```

### Fields

- **id**: Auto-incrementing primary key
- **customer_id**: Customer identifier (ties session to a customer)
- **session_id**: Unique session identifier
- **conversation_turns**: JSONB array storing conversation turns
  ```json
  [
    {"role": "user", "text": "...", "metadata": {...}},
    {"role": "assistant", "text": "...", "metadata": {...}}
  ]
  ```
- **created_at**: Session creation timestamp
- **updated_at**: Last modification timestamp
- **is_active**: Boolean flag for active sessions

### Indexes

- `idx_customer_id`: Fast lookup of sessions by customer
- `idx_created_at`: Retrieves most recent sessions efficiently
- `idx_is_active`: Filters active vs. closed sessions

## Setup

### 1. Initialize Database Table

```bash
# Run migration script
python -m pipelines.setup_sessions_table

# Or directly
python pipelines/setup_sessions_table.py
```

### 2. Environment Configuration

Ensure your `.env` file has PostgreSQL credentials:

```env
POSTGRESQL_HOST=your-aiven-host.aivencloud.com
POSTGRESQL_PORT=12548
POSTGRESQL_USER=avnadmin
POSTGRESQL_DB=defaultdb
POSTGRESQL_AIVEN_PASSWORD=your-password
```

## Usage

### Basic Usage in Orchestrator

```python
from src.memory.persistent_session_manager import PersistentSessionManager

# Initialize manager
session_mgr = PersistentSessionManager(persist_to_db=True)

# Get or create session with customer ID
session = session_mgr.get_or_create_session(
    session_id="customer-user-1-session-001",
    customer_id="customer-user-1"
)

# Append turns (automatically saved to DB)
session_mgr.append_turn(
    session_id="customer-user-1-session-001",
    role="user",
    text="I have a question about my order",
    customer_id="customer-user-1"
)

session_mgr.append_turn(
    session_id="customer-user-1-session-001",
    role="assistant",
    text="I'd be happy to help with your order!",
    customer_id="customer-user-1"
)

# Retrieve customer's last 5 sessions
history = session_mgr.get_customer_history(
    customer_id="customer-user-1",
    limit=5
)
```

### Enriching Messages with History

```python
from src.utils.customer_history import enrich_message_with_history, format_customer_history_context

# Retrieve history context
context = format_customer_history_context(
    customer_id="customer-user-1",
    num_sessions=2,
    max_turns_per_session=3
)
# Output:
# === Customer customer-user-1 - Recent Conversation History ===
# 
# --- Previous Session 1 ---
# [Session: session-001]
# Date: 2024-01-15T10:30:00
# ---
# USER: Can I return my order?
# ASSISTANT: Yes, we accept returns within 30 days...

# Enrich message with historical context
enriched_msg = enrich_message_with_history(
    user_message="What about other colors?",
    customer_id="customer-user-1",
    include_history=True,
    num_sessions=2
)

# Pass enriched_msg to orchestrator for better responses
```

### Analyzing Customer History

```python
from src.utils.customer_history import get_customer_sentiment_summary, get_customer_topics_from_history

# Get topics customer has discussed
topics = get_customer_topics_from_history(customer_id="customer-user-1")
# Returns: ["product", "order", "shipping"]

# Get sentiment summary
summary = get_customer_sentiment_summary(customer_id="customer-user-1")
# Returns: {
#     "customer_id": "customer-user-1",
#     "total_sessions": 3,
#     "total_interactions": 12,
#     "issue_count": 2,
#     "has_issues": True,
#     "topics": ["product", "order"]
# }
```

### Closing Sessions

```python
# Close session and clean up old sessions (keep 5 most recent)
session_mgr.close_session(
    session_id="customer-user-1-session-001",
    customer_id="customer-user-1"
)
```

## Integration with Orchestrator

To integrate with `SupportOrchestrator`:

```python
from src.agents.orchestrator.agent import SupportOrchestrator
from src.memory.persistent_session_manager import PersistentSessionManager

orchestrator = SupportOrchestrator()

# Replace in-memory session manager with persistent one
orchestrator._session_manager = PersistentSessionManager(persist_to_db=True)

# When handling requests, include customer_id
response = orchestrator.handle(
    user_message="I need help with my order",
    session_id="customer-user-1-session-001",
    customer_id="customer-user-1"  # Pass customer ID for persistence
)
```

## API Reference

### session_persistence.py

- `initialize_sessions_table()`: Create table if not exists
- `save_session_to_db(customer_id, session_id, conversation_memory)`: Save/update session
- `get_customer_session_history(customer_id, limit=5)`: Get recent sessions
- `load_session_from_db(session_id)`: Load a specific session
- `close_session(session_id)`: Mark session as inactive
- `get_session_count_for_customer(customer_id)`: Count sessions for customer
- `delete_old_sessions_for_customer(customer_id, keep_count=5)`: Clean up old sessions

### persistent_session_manager.py

- `get_or_create_session(session_id, customer_id)`: Get/create session with persistence
- `append_turn(session_id, role, text, customer_id)`: Add turn and save to DB
- `close_session(session_id, customer_id)`: Close session and cleanup
- `get_customer_history(customer_id, limit=5)`: Retrieve customer's recent sessions
- `get_last_session_for_customer(customer_id)`: Get most recent session

### customer_history.py

- `format_session_summary(session_data, max_turns=3)`: Format single session for display
- `format_customer_history_context(customer_id, num_sessions=3)`: Format full history
- `enrich_message_with_history(user_message, customer_id)`: Add history to message
- `get_customer_topics_from_history(customer_id)`: Extract discussed topics
- `get_customer_sentiment_summary(customer_id)`: Analyze customer sentiment

## Limitations & Best Practices

### 5-Session Limit

- Automatically maintained per customer
- When `close_session()` is called with customer_id, old sessions are cleaned up
- Most recent sessions are always preserved

### JSONB Storage

- Conversation turns stored as JSONB for efficient querying
- Can filter and search within turns if needed
- Full text search possible with PostgreSQL

### Performance Considerations

- Indexes on `customer_id`, `created_at`, and `is_active`
- For large conversation histories, use pagination
- Consider archiving very old sessions to a separate table

### Data Privacy

- Ensure customer_id is validated/authenticated before retrieval
- Consider encryption for sensitive conversation data
- Implement access controls on session endpoints

## Example: Complete Integration

```python
from src.agents.orchestrator.agent import SupportOrchestrator
from src.memory.persistent_session_manager import PersistentSessionManager
from src.utils.customer_history import enrich_message_with_history, format_customer_history_context

class EnhancedSupportOrchestrator(SupportOrchestrator):
    """Orchestrator with persistent customer history."""
    
    def __init__(self):
        super().__init__()
        # Use persistent session manager
        self._session_manager = PersistentSessionManager(persist_to_db=True)
    
    def handle_with_history(
        self,
        user_message: str,
        session_id: str,
        customer_id: str,
    ) -> dict:
        """Handle request with customer history context."""
        
        # Enrich message with customer history
        enriched_message = enrich_message_with_history(
            user_message=user_message,
            customer_id=customer_id,
            include_history=True,
            num_sessions=2
        )
        
        # Handle with enriched context
        response = self.handle(enriched_message, session_id=session_id)
        
        # Save to persistent storage
        session = self._session_manager.get_or_create_session(
            session_id=session_id,
            customer_id=customer_id
        )
        self._session_manager.append_turn(
            session_id=session_id,
            role="assistant",
            text=response.response,
            customer_id=customer_id
        )
        
        return response
```

## Troubleshooting

### Connection Errors

```
Error executing query: could not connect to server
```

- Check PostgreSQL credentials in `.env`
- Verify Aiven instance is running
- Ensure SSL mode is enabled (`sslmode=require`)

### Table Already Exists

Safe to run migration multiple times - uses `CREATE TABLE IF NOT EXISTS`

### Sessions Not Persisting

- Verify `persist_to_db=True` in PersistentSessionManager
- Check that `customer_id` is provided when appending turns
- Verify database connection is active

## Future Enhancements

- [ ] Implement session encryption
- [ ] Add full-text search across conversations
- [ ] Create dashboard for customer support teams
- [ ] Add session tagging/labeling
- [ ] Implement session archival to cold storage
- [ ] Add conversation analytics

## Related Files

- [Orchestrator](../agents/orchestrator/agent.py)
- [Session Manager](./session_manager.py)
- [Conversation Memory](./conversation_memory.py)
- [PostgreSQL Utilities](postgresql.py)
