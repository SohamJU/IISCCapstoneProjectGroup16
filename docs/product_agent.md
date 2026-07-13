# Product Agent — Design & Operations Guide

## Overview

The **Product Recommendation Agent** is a conversational AI agent that helps
customers of an electronics and appliances e-commerce store discover products.
It is one of several specialised agents in the multi-agent system (alongside
Order, Return, and Recommendation agents), all coordinated by the Router Agent.

The agent is built on **LangGraph's ReAct loop** and uses a **hybrid retrieval
pipeline** (BM25 keyword search + dense vector search) to find relevant
products, replacing the original SQL-only approach that caused infinite loops.

---

## Architecture

```
User Message
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  ProductRecommendationAgent  (LangGraph ReAct)              │
│                                                             │
│  System Prompt                                              │
│  ├─ Role description                                        │
│  ├─ Tool hierarchy (search_products first, SQL second)      │
│  ├─ Loop-prevention rules                                   │
│  └─ Product catalog schema reference                        │
│                                                             │
│  Tools                                                      │
│  ├─ search_products ──────────► hybrid_search()             │
│  │   (PRIMARY)                   ├─ BM25 index             │
│  │                               └─ Pinecone vector store  │
│  │                                   (+ RRF fusion)         │
│  ├─ query_products ──────────► PostgreSQL SQL               │
│  │   (SECONDARY, exact filters)                            │
│  └─ get_twitter_samples ─────► stub (demo)                  │
│                                                             │
│  Memory: LangGraph MemorySaver (per-session)                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Agent Response (formatted product recommendations)
```

## Tools Reference

### `search_products` — Primary Tool

Use this for **all natural-language product queries**. It runs a hybrid
retrieval combining BM25 and Pinecone vector search, then fuses results
using Reciprocal Rank Fusion (RRF).

| Argument | Type | Description |
|---|---|---|
| `query` | `str` | Natural-language description of what to find |
| `price_min` | `float \| None` | Minimum price in USD (optional) |
| `price_max` | `float \| None` | Maximum price in USD (optional) |
| `category` | `str \| None` | Category substring filter (optional) |
| `top_k` | `int` | Results to return (default 5, max 10) |

**Examples:**
```
search_products(query="gaming headset with surround sound")
search_products(query="laptop for students", price_max=500.0)
search_products(query="quiet washing machine", category="Appliances")
```

**Rules for the agent:**
- Call this tool **once per user intent**.
- If results are returned, present them immediately — do not re-search.
- If no results are found, tell the user and stop.

---

### `query_products` — Secondary Tool

Use this for **exact structured queries** that cannot be expressed as natural
language — e.g., counting products, looking up by exact `product_id`, or
aggregating with `GROUP BY`.

| Argument | Type | Description |
|---|---|---|
| `sql_query` | `str` | A valid SQL SELECT against `product_catalog` |

Write operations (INSERT, UPDATE, DELETE, DROP, etc.) are blocked by a regex
guard and will return an error message.

**When to use:**
- "How many products are in the Electronics category?" → `SELECT COUNT(*)`
- "Look up product B00MCW7G9M" → `WHERE product_id = 'B00MCW7G9M'`

**Do NOT use for:** keyword or semantic product searches — use `search_products` instead.

---

### `get_twitter_samples` — Demo Stub

Placeholder for future Pinecone-backed retrieval of historical customer
support conversations. Currently returns an empty result.

Kept in the agent for demo purposes only. Will be replaced with real
vector retrieval in a future iteration.

---

## Search Strategy: Hybrid BM25 + Dense Vector + RRF

### Why Hybrid?

| Strategy | Strength | Weakness |
|---|---|---|
| SQL only | Exact matches | Cannot handle semantic queries; causes loops |
| BM25 | Keyword matching, fast, no GPU | Misses synonyms and paraphrasing |
| Dense vector | Semantic understanding | May miss exact model numbers / brands |
| **Hybrid (BM25 + vector)** | Best of both worlds | Requires indexing pipeline |

### Embedding Model: `BAAI/bge-small-en-v1.5`

- **Size:** ~33M parameters, 384-dim embeddings
- **Device:** CPU (no GPU required)
- **Why chosen:** Top MTEB Retrieval score for small models; beats
  `all-MiniLM-L6-v2` on dense retrieval benchmarks
- **BGE Query Convention:** Query texts are prefixed with
  `"Represent this sentence for searching relevant passages: "` —
  handled automatically by `ProductEmbedder.embed_query()`

### Reciprocal Rank Fusion (RRF)

RRF combines rankings from BM25 and vector search:

```
score_rrf(document) = Σ  1 / (60 + rank_i)
                      i
```

A document appearing at rank 1 in both lists scores higher than one
appearing at rank 1 in only one list. The constant `k=60` smooths scores
for documents at high rank positions.

---

## Index Persistence (Two Layers)

| Layer | Location | Purpose |
|---|---|---|
| **Local cache** | `data/processed/embeddings_cache.parquet` | Resume pipeline without re-embedding |
| **Pinecone** | Cloud index `product-catalog` | Primary fast ANN search (194k scale) |

The local cache ensures embeddings are only computed once. On subsequent runs,
only new products (not yet in the cache) are embedded and uploaded to Pinecone.

---

## Setting Up: Running the Indexing Pipeline

> **Run this once before using the agent with vector search.**
> The agent works with BM25-only before indexing is complete.

### Prerequisites

1. **Pinecone account:** Sign up at https://www.pinecone.io/ (free Starter tier)
2. **Create a Pinecone index:**
   - Name: `product-catalog`
   - Dimensions: `384`
   - Metric: `cosine`
3. **Set your API key in `.env`:**
   ```ini
   PINECONE_API_KEY=your-actual-api-key-here
   PINECONE_INDEX_NAME=product-catalog
   ```
4. **Enable pgvector on PostgreSQL** (run once as admin):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

### Run the Pipeline

```bash
# From the project root:

# Full indexing (all 194k products):
python pipelines/03_index_product_catalog.py

# Test run (first 1000 products only):
python pipelines/03_index_product_catalog.py --max-rows 1000

# Skip Pinecone upload (embed + cache locally only):
python pipelines/03_index_product_catalog.py --no-pinecone
```

### Expected Runtime (Standard Laptop, CPU-only)

| Step | Time |
|---|---|
| Load catalog from PostgreSQL | ~1 min |
| Embed 194k products (bge-small-en-v1.5) | ~25–40 min |
| Pinecone upsert | ~5–10 min |
| **Total** | **~30–50 min** |

The pipeline is **resumable** — if interrupted, it re-reads the local
Parquet cache and only embeds products that haven't been processed yet.

### Outputs

```
data/processed/embeddings_cache.parquet   <- raw embedding vectors
Pinecone: product-catalog index           <- 194k cosine vectors
```

---

## Configuration Reference

File: [`src/agents/product_agent/config.py`](../src/agents/product_agent/config.py)

| Constant | Default | Description |
|---|---|---|
| `MAX_REACT_ITERATIONS` | `8` | Safety cap on ReAct reasoning loops |
| `CONVERSATION_MEMORY_TURNS` | `10` | Recent turns stored in memory |
| `USE_TWITTER_SAMPLES` | `False` | Enable/disable Twitter stub tool |
| `BM25_TOP_K` | `15` | Candidates from BM25 index |
| `VECTOR_TOP_K` | `15` | Candidates from Pinecone |
| `HYBRID_FINAL_TOP_K` | `5` | Final results after RRF fusion |

Environment variables (`.env`):

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | LLM provider (Groq) |
| `POSTGRESQL_AIVEN_PASSWORD` | Yes | Database password |
| `PINECONE_API_KEY` | For vector search | Pinecone API key |
| `PINECONE_INDEX_NAME` | For vector search | Index name (default: `product-catalog`) |

---

## Key Source Files

| File | Purpose |
|---|---|
| [`src/agents/product_agent/agent.py`](../src/agents/product_agent/agent.py) | Main agent class (LangGraph ReAct) |
| [`src/agents/product_agent/tools.py`](../src/agents/product_agent/tools.py) | Tool definitions (`search_products`, `query_products`) |
| [`src/agents/product_agent/prompts.py`](../src/agents/product_agent/prompts.py) | System prompt with tool hierarchy + loop guards |
| [`src/agents/product_agent/config.py`](../src/agents/product_agent/config.py) | Configuration constants |
| [`src/search/bm25_index.py`](../src/search/bm25_index.py) | BM25 in-memory keyword index |
| [`src/search/hybrid_search.py`](../src/search/hybrid_search.py) | BM25 + Pinecone + RRF fusion |
| [`src/embeddings/embedder.py`](../src/embeddings/embedder.py) | `BAAI/bge-small-en-v1.5` wrapper |
| [`src/embeddings/vector_store.py`](../src/embeddings/vector_store.py) | Pinecone store + local Parquet cache |
| [`src/rag/retriever.py`](../src/rag/retriever.py) | Clean retrieval interface |
| [`src/rag/pipeline.py`](../src/rag/pipeline.py) | Full RAG pipeline (retrieve + generate) |
| [`pipelines/03_index_product_catalog.py`](../pipelines/03_index_product_catalog.py) | Offline indexing pipeline |

---

## Demo

Run the interactive demo notebook:

```bash
jupyter notebook notebooks/03_product_agent_demo.ipynb
```

Example queries to try:

| Query | Expected behaviour |
|---|---|
| "Show me gaming headsets" | BM25 + vector finds headsets |
| "Something to watch movies in bed" | Vector search finds tablets, projectors |
| "Quiet washing machine under $800" | Hybrid + price filter |
| "Budget laptop for students" | Semantic match on "budget" + "student" |
| "How many products are in Electronics?" | Agent uses SQL `COUNT(*)` |

---

## Known Limitations

1. **BM25 startup time:** The BM25 index is rebuilt in memory on each process
   start (~10–30 sec for 194k products). Consider persisting the index to disk
   if startup time is a concern.
2. **Pinecone metadata filter:** Pinecone's metadata filter requires exact
   category matches. The hybrid search applies a post-fusion soft filter as
   a fallback for partial matches.
3. **Twitter samples:** `get_twitter_samples` is a non-functional stub and
   always returns an empty result. Kept for demo purposes only.
4. **No re-ranking:** The pipeline does not include a cross-encoder re-ranker.
   Adding one (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) would improve
   precision at the cost of latency.
