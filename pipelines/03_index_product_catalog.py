"""Pipeline 03: Index product catalog into BM25 + Pinecone.

Run this pipeline **once** (offline) to populate the Pinecone vector store
before starting the product agent with full hybrid retrieval.

The pipeline is resumable — it checks the local embedding cache and skips
products that have already been embedded.

Runtime estimate on a standard laptop (CPU only):
- Embedding 194k products: ~25–40 minutes (256-doc batches, bge-small-en-v1.5)
- Pinecone upsert: ~5–10 minutes

Usage::

    # From the project root:
    python pipelines/03_index_product_catalog.py

    # Index only the first 1000 rows (for testing):
    python pipelines/03_index_product_catalog.py --max-rows 1000

    # Skip Pinecone upload (embed + cache locally only):
    python pipelines/03_index_product_catalog.py --no-pinecone

Outputs
-------
data/processed/embeddings_cache.parquet
    Raw (product_id, embedding_bytes) pairs — used to resume the pipeline
    without re-running the embedding model.

Pinecone: product-catalog index
    194k vectors with title, price, rating, category as metadata.
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from pathlib import Path

# ── Ensure project root is on the path ───────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import src.windows_hotfixes  # Fix Windows DLL/UV crashes

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("index_pipeline")

# ── Constants ─────────────────────────────────────────────────────────────────
_CATALOG_CSV = _PROJECT_ROOT / "data" / "processed" / "product_catalog.csv"
_EMBED_BATCH_SIZE = 256
_CHECKPOINT_EVERY = 10  # save local cache every N batches (~2560 products)


# ── Text helpers ──────────────────────────────────────────────────────────────

def _safe_list_str(raw) -> str:
    """Convert a Python-list-as-string into a plain string."""
    if not raw or (isinstance(raw, float)):
        return ""
    s = str(raw).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return " ".join(str(x) for x in parsed)
    except Exception:
        pass
    return re.sub(r"[\[\]'\"]", " ", s)


def _build_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "")
    description = _safe_list_str(row.get("description", ""))
    features = _safe_list_str(row.get("features", ""))
    category = str(row.get("main_category", "") or "")
    store = str(row.get("store", "") or "")
    return f"{title} {category} {store} {description} {features}"


def _build_metadata(row: pd.Series) -> dict:
    """Build a Pinecone-compatible metadata dict (40KB per-vector limit)."""
    return {
        "title": str(row.get("title", "") or "")[:500],
        "main_category": str(row.get("main_category", "") or ""),
        "price": float(row["price"]) if pd.notna(row.get("price")) else 0.0,
        "average_rating": float(row["average_rating"]) if pd.notna(row.get("average_rating")) else 0.0,
        "rating_count": int(row["rating_count"]) if pd.notna(row.get("rating_count")) else 0,
        "store": str(row.get("store", "") or ""),
        "is_bestseller": bool(row.get("is_bestseller", False)),
    }


# ── Catalog loading ───────────────────────────────────────────────────────────

def _load_catalog(max_rows: int | None) -> pd.DataFrame:
    """Try PostgreSQL first, fall back to CSV."""
    try:
        from src.data.postgresql import get_db_engine

        engine = get_db_engine()
        query = "SELECT * FROM product_catalog"
        if max_rows:
            query += f" LIMIT {max_rows}"
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        logger.info("Loaded %d rows from PostgreSQL.", len(df))
        return df
    except Exception as exc:
        logger.warning("PostgreSQL load failed (%s) - falling back to CSV.", exc)

    df = pd.read_csv(_CATALOG_CSV, low_memory=False)
    if max_rows:
        df = df.head(max_rows)
    logger.info("Loaded %d rows from CSV.", len(df))
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    max_rows: int | None = None,
    skip_pinecone: bool = False,
    force_embed: bool = False,
) -> None:
    """Run the full indexing pipeline."""
    from src.embeddings.embedder import ProductEmbedder
    from src.embeddings.vector_store import LocalEmbeddingCache, PineconeVectorStore

    logger.info("=" * 60)
    logger.info("Product Catalog Indexing Pipeline")
    logger.info("  Embedding model : BAAI/bge-small-en-v1.5")
    logger.info("  Vector store    : Pinecone")
    logger.info("=" * 60)

    # ── Step 1: Load catalog ────────────────────────────────────────────────
    logger.info("Step 1/4 - Loading product catalog...")
    df = _load_catalog(max_rows)
    logger.info("Total products: %d", len(df))

    # ── Step 2: Check local cache — find unembedded products ────────────────
    logger.info("Step 2/4 - Checking local embedding cache...")
    cache = LocalEmbeddingCache()
    cached_ids = cache.get_cached_ids()
    logger.info("Already cached: %d products.", len(cached_ids))

    if force_embed:
        df_todo = df.reset_index(drop=True)
    else:
        todo_mask = ~df["product_id"].astype(str).isin(cached_ids)
        df_todo = df[todo_mask].reset_index(drop=True)
    logger.info("To embed: %d products.", len(df_todo))

    # ── Step 3: Embed remaining products ────────────────────────────────────
    if not df_todo.empty:
        logger.info("Step 3/4 - Embedding with BAAI/bge-small-en-v1.5 (CPU)...")
        # Initialize embedder to strictly use CPU
        embedder = ProductEmbedder(batch_size=_EMBED_BATCH_SIZE, show_progress=False, use_cpu=True)

        logger.info("Extracting %d texts from catalog...", len(df_todo))
        texts = [_build_text(row) for _, row in df_todo.iterrows()]
        pids = df_todo["product_id"].astype(str).tolist()

        logger.info("Starting single-process embedding (with batch checkpoints)...")
        
        chunk_size = _EMBED_BATCH_SIZE * _CHECKPOINT_EVERY
        for i in tqdm(range(0, len(texts), chunk_size), desc="Embedding chunks"):
            batch_texts = texts[i : i + chunk_size]
            batch_pids = pids[i : i + chunk_size]
            
            vecs = embedder.embed_texts(batch_texts, show_progress=False)
            cache.append(batch_pids, vecs)
            
        logger.info("Embedding complete. Local cache updated.")
    else:
        logger.info("Step 3/4 - All products already cached. Skipping embedding.")

    # ── Step 4: Load full cache and upsert to Pinecone ──────────────────────
    logger.info("Step 4/4 - Loading full embedding cache...")
    cached_product_ids, all_embeddings = cache.load()
    logger.info("Cache loaded: %d embeddings.", len(cached_product_ids))

    if skip_pinecone:
        logger.info("Step 4/4 - Skipping Pinecone (--no-pinecone flag set).")
    else:
        logger.info("Step 4/4 - Upserting to Pinecone...")
        pinecone_store = PineconeVectorStore()

        if not pinecone_store.is_available():
            logger.warning(
                "PINECONE_API_KEY is not set or is still a placeholder.\n"
                "  Set it in .env:  PINECONE_API_KEY=your-actual-key-here\n"
                "  Embeddings are safely cached locally at: %s",
                cache.path,
            )
        else:
            # Build product_id -> row lookup for metadata
            id_to_row = df.set_index("product_id")

            meta_list = []
            for pid in cached_product_ids:
                try:
                    row = id_to_row.loc[pid]
                    meta_list.append(_build_metadata(row))
                except KeyError:
                    meta_list.append({})

            upserted = pinecone_store.upsert(
                product_ids=cached_product_ids,
                embeddings=all_embeddings,
                metadata_list=meta_list,
            )
            stats = pinecone_store.index_stats()
            logger.info("Pinecone index stats: %s", stats)
            logger.info("Pinecone upsert complete: %d vectors.", upserted)

    logger.info("=" * 60)
    logger.info("Indexing pipeline complete!")
    logger.info("  Local cache : %s", cache.path)
    logger.info("  Pinecone    : %s", "updated" if not skip_pinecone else "skipped")
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index product catalog into BM25 + Pinecone.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Limit the number of products indexed (for testing). Default: all.",
    )
    parser.add_argument(
        "--no-pinecone",
        action="store_true",
        help="Skip Pinecone upsert — embed and cache locally only.",
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Bypass local cache and embed all loaded products (for debugging).",
    )
    args = parser.parse_args()

    run(
        max_rows=args.max_rows,
        skip_pinecone=args.no_pinecone,
        force_embed=args.force_embed,
    )
