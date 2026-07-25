"""
Complete Integration Architecture for Customer Session Persistence

This document visualizes how all components work together.
"""

# ARCHITECTURE DIAGRAM
# ====================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CUSTOMER SESSION PERSISTENCE                       │
│                              Architecture                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌──────────────────────┐
                            │   Gradio UI / Chat   │
                            │      Interface       │
                            └──────────┬───────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
           (customer_id, session_id)                      │
                    │                                     │
                    ▼                                     ▼
        ┌─────────────────────────────┐    ┌──────────────────────────┐
        │  Customer History Utils     │    │ Orchestrator             │
        │ (customer_history.py)       │    │ + PersistentSession Mgr  │
        │                             │    │                          │
        │ • format_session_summary    │    │ • Routes requests        │
        │ • enrich_message_with_      │    │ • Appends turns          │
        │   history                   │    │ • Tracks session state   │
        │ • get_customer_topics       │    └──────────┬───────────────┘
        │ • get_sentiment_summary     │              │
        └──────────┬────────────────┬─┘              │
                   │                │                │
                   │          ┌─────▼────────────────▼───────────────┐
                   │          │  Persistent Session Manager          │
                   │          │  (persistent_session_manager.py)      │
                   │          │                                       │
                   │          │ • get_or_create_session()            │
                   │          │ • append_turn() with customer_id     │
                   │          │ • close_session()                    │
                   │          │ • get_customer_history()             │
                   └──────────┼──────────┬────────────────────────────┤
                              │          │                            │
                    save/load │          │ retrieve/update            │
                              │          │                            │
                    ┌─────────▼──────────▼────────┐
                    │  Session Persistence        │
                    │  (session_persistence.py)   │
                    │                             │
                    │ • initialize_sessions_table │
                    │ • save_session_to_db()      │
                    │ • load_session_from_db()    │
                    │ • get_customer_history()    │
                    │ • delete_old_sessions()     │
                    │ • close_session()           │
                    └─────────┬──────────────────┘
                              │
                              │ (Connection String)
                              ▼
                    ┌─────────────────────────────┐
                    │  PostgreSQL (Aiven Cloud)   │
                    │                             │
                    │  customer_sessions table    │
                    │                             │
                    │ • id (PK)                   │
                    │ • customer_id               │
                    │ • session_id (UNIQUE)       │
                    │ • conversation_turns (JSONB)│
                    │ • created_at                │
                    │ • updated_at                │
                    │ • is_active                 │
                    │                             │
                    │ Indexes:                    │
                    │ • idx_customer_id           │
                    │ • idx_created_at            │
                    │ • idx_is_active             │
                    └─────────────────────────────┘
"""


# DATA FLOW EXAMPLE
# =================

"""
1. USER SENDS MESSAGE
   ┌─ User enters: "Can I return my order?"
   ├─ Gradio App captures: customer_id="customer-001", session_id="sess-001"
   └─ Calls orchestrator.handle(message, session_id, customer_id)

2. HISTORY ENRICHMENT (Optional)
   ┌─ enrich_message_with_history() retrieves past sessions
   ├─ Formats as: "Previous conversations about: returns, shipping"
   └─ Prepends to user message for LLM context

3. SESSION RETRIEVAL
   ┌─ get_or_create_session("sess-001", "customer-001")
   ├─ If first time: Creates new empty ConversationMemory
   └─ If returning: Loads from database

4. MESSAGE PROCESSING
   ┌─ Routes through specialized agents (return, product, etc)
   ├─ Agents generate response
   └─ Returns: OrchestratorResponse(route, confidence, response)

5. SESSION PERSISTENCE
   ┌─ Append user turn: role="user", text="Can I return..."
   ├─ Append assistant turn: role="assistant", text="Yes, 30 days..."
   └─ Save to database via save_session_to_db()

6. DATABASE UPSERT
   ┌─ INSERT if session_id doesn't exist
   ├─ UPDATE if session_id exists
   ├─ Stores conversation_turns as JSONB array
   └─ Updates updated_at timestamp

7. CLEANUP (On Session Close)
   ┌─ close_session("sess-001")
   ├─ Mark is_active=FALSE
   └─ delete_old_sessions_for_customer("customer-001", keep_count=5)
      └─ Removes sessions older than 5 most recent

8. NEXT VISIT BY SAME CUSTOMER
   ┌─ Customer returns with session_id="sess-002", customer_id="customer-001"
   ├─ enrich_message_with_history() retrieves past 2-3 sessions
   ├─ Shows previous conversations as context
   └─ Agents provide continuity-aware responses
"""


# INTEGRATION CHECKLIST
# =====================

"""
□ Phase 1: Setup
  □ Run: python -m pipelines.setup_sessions_table
  □ Verify: Table created in PostgreSQL
  □ Test: python -c "from src.data.session_persistence import *"

□ Phase 2: Basic Integration
  □ Replace SessionManager with PersistentSessionManager in orchestrator
  □ Update gradio app to pass customer_id
  □ Test: Manual session save/load

□ Phase 3: History Enrichment
  □ Integrate enrich_message_with_history() in message handling
  □ Pass enriched message to orchestrator
  □ Test: Verify history appears in LLM responses

□ Phase 4: Cleanup & Limits
  □ Call close_session() with customer_id on session end
  □ Verify old sessions are deleted (keep 5)
  □ Test: Create 7 sessions, verify only 5 remain

□ Phase 5: Analytics
  □ Add customer sentiment dashboard
  □ Track topics discussed
  □ Monitor session counts

□ Phase 6: Testing
  □ Run: pytest tests/test_session_persistence.py
  □ Manual testing: Show customer history
  □ Load testing: Create 1000 sessions
"""


# BEFORE & AFTER COMPARISON
# ==========================

"""
BEFORE (In-Memory Only):
├─ Conversation stored only in RAM
├─ Lost on server restart
├─ No customer history available
├─ Each session starts fresh
└─ No persistence

AFTER (With Database Persistence):
├─ ✓ Conversation saved to PostgreSQL
├─ ✓ Survives server restarts
├─ ✓ Past sessions available
├─ ✓ Context-aware responses
├─ ✓ Customer analytics possible
├─ ✓ Session limits enforced (5 per customer)
└─ ✓ Automatic cleanup
"""


# CONFIGURATION GUIDE
# ===================

"""
Environment Variables (.env):
  POSTGRESQL_HOST=your-instance.aivencloud.com
  POSTGRESQL_PORT=12548
  POSTGRESQL_USER=avnadmin
  POSTGRESQL_DB=defaultdb
  POSTGRESQL_AIVEN_PASSWORD=your-password

Orchestrator Setup:
  orchestrator = SupportOrchestrator()
  orchestrator._session_manager = PersistentSessionManager(persist_to_db=True)

Gradio App Updates:
  • Add customer_id input field
  • Pass customer_id to _respond function
  • Include in append_turn() calls
  • Use enrich_message_with_history() before handle()

Optional Features:
  • History enrichment: use_history=True/False
  • Session limit: keep_count=5 (or custom)
  • Context depth: num_sessions=2, max_turns_per_session=3
  • Sentiment analysis: get_customer_sentiment_summary()
"""


# PERFORMANCE CONSIDERATIONS
# ==========================

"""
Database Indexes:
  • idx_customer_id: Fast lookup by customer → O(log n)
  • idx_created_at: Recent sessions retrieval → O(log n)
  • idx_is_active: Active/closed filter → O(log n)

Query Performance:
  • Get 5 recent sessions: ~5-10ms (with index)
  • Save session: ~10-20ms (upsert)
  • Format history: ~50-100ms (Python processing)
  • Enrich message: ~100-200ms (with all utilities)

Optimization Tips:
  1. Use pagination for large customer histories
  2. Cache history context for same customer
  3. Lazy-load history only when needed
  4. Archive old sessions to separate table
  5. Consider read replicas for analytics

Scaling:
  • 1M customers × 5 sessions = 5M rows
  • ~500MB storage (conservative estimate)
  • Query patterns all indexed
  • Ready for production
"""


# EXAMPLE QUERIES (Direct PostgreSQL)
# ====================================

"""
-- Get customer's recent sessions
SELECT session_id, created_at, jsonb_array_length(conversation_turns) as turn_count
FROM customer_sessions
WHERE customer_id = 'customer-001'
ORDER BY created_at DESC
LIMIT 5;

-- Find customers with most sessions
SELECT customer_id, COUNT(*) as session_count
FROM customer_sessions
GROUP BY customer_id
ORDER BY session_count DESC
LIMIT 10;

-- Get average session length
SELECT 
    AVG(jsonb_array_length(conversation_turns)) as avg_turns,
    MAX(jsonb_array_length(conversation_turns)) as max_turns,
    MIN(jsonb_array_length(conversation_turns)) as min_turns
FROM customer_sessions;

-- Find sessions mentioning 'return' or 'refund'
SELECT session_id, customer_id, created_at
FROM customer_sessions
WHERE conversation_turns::text ILIKE '%return%'
   OR conversation_turns::text ILIKE '%refund%'
ORDER BY created_at DESC;

-- Get most active customers
SELECT customer_id, COUNT(*) as sessions, 
       MAX(updated_at) as last_active
FROM customer_sessions
WHERE is_active = TRUE
GROUP BY customer_id
ORDER BY sessions DESC
LIMIT 20;

-- Sessions lasting more than 10 turns
SELECT session_id, customer_id, jsonb_array_length(conversation_turns) as turns
FROM customer_sessions
WHERE jsonb_array_length(conversation_turns) > 10
ORDER BY turns DESC;
"""


# MONITORING & DEBUGGING
# ======================

"""
Monitor Database:
  • Size: SELECT pg_size_pretty(pg_total_relation_size('customer_sessions'));
  • Row count: SELECT COUNT(*) FROM customer_sessions;
  • Active sessions: SELECT COUNT(*) FROM customer_sessions WHERE is_active=TRUE;
  • Index usage: SELECT * FROM pg_stat_user_indexes WHERE relname='customer_sessions';

Debug Issues:
  1. Check connection: python -c "from src.data.session_persistence import *"
  2. Verify table exists: \dt customer_sessions (in psql)
  3. Check indexes: \d customer_sessions (in psql)
  4. Test save: save_session_to_db("test", "test", memory)
  5. Test load: load_session_from_db("test")

Common Issues:
  • "No such file or directory": Use remote connection (not local socket)
  • "Table already exists": Safe to ignore, uses IF NOT EXISTS
  • "Sessions not saving": Ensure persist_to_db=True and customer_id provided
  • "History empty": Verify customer_id matches between save/retrieve
"""


# FILE STRUCTURE
# ==============

"""
Project/
├── pipelines/
│   └── setup_sessions_table.py          ← Migration script
│
├── src/
│   ├── data/
│   │   └── session_persistence.py       ← Database layer (new)
│   │
│   ├── memory/
│   │   ├── conversation_memory.py       (existing)
│   │   ├── session_manager.py           (existing)
│   │   └── persistent_session_manager.py ← Extended manager (new)
│   │
│   ├── utils/
│   │   └── customer_history.py          ← Utilities (new)
│   │
│   └── agents/
│       └── orchestrator/
│           └── agent.py                 ← Update with new session mgr
│
├── app/
│   └── gradio_app.py                    ← Update with customer_id
│
├── docs/
│   ├── customer_session_persistence.md  ← Full documentation (new)
│   └── session_persistence_quick_start.md ← Quick start guide (new)
│
└── tests/
    └── test_session_persistence.py      ← Test suite (new)
"""
