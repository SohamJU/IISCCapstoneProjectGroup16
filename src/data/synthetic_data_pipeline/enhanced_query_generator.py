"""Enhanced Synthetic Customer Query Generator.

This is a COPY of src/data/synthetic_generator.py with enhancements:
  - Product names sampled from the real product catalog
  - Real order IDs injected into order-related queries
  - Additional intents: loyalty_inquiry, account_update, shipping_estimate
  - Default volume increased to 500 queries

The original src/data/synthetic_generator.py is FROZEN and not modified.

Usage:
    python -m src.data.pipeline.enhanced_query_generator
    python -m src.data.pipeline.enhanced_query_generator --total 1000
"""

import random
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config.data import (
    CUSTOMER_QUERIES_PATH,
    ORDERS_PATH,
    PRODUCT_CATALOG_PATH,
)


# -----------------------------------------------------------------------------
# INTENTS & INTENT-SPECIFIC VOCABULARY
# -----------------------------------------------------------------------------
INTENT_VOCAB = {
    "product_search": ["looking for", "find a", "searching for", "do you carry"],
    "product_recommendation": ["suggest a good", "recommendation for", "best option for"],
    "product_comparison": ["vs", "difference between", "better choice compared to"],
    "order_tracking": ["where is my", "track status of", "has it shipped yet"],
    "order_modification": ["change shipping address", "update items in", "modify my order"],
    "order_cancellation": ["cancel my order", "stop shipment", "don't want this anymore"],
    "returns": ["return this", "send back", "return policy for"],
    "refunds": ["get money back", "request a refund", "reimburse me for"],
    "warranty_replacement": ["warranty claim", "replace broken item", "swap under warranty"],
    "payment_issues": ["card declined", "charged twice", "payment failed at checkout"],
    "account_issues": ["cannot login", "reset password", "account locked"],
    "discounts_offers": ["promo code not working", "apply coupon", "any active discounts"],
    "complaints": ["terrible service", "very disappointed", "unacceptable quality"],
    "delivery_issues": ["package stolen", "delivered to wrong house", "delayed shipment"],
    # ── New intents ──
    "loyalty_inquiry": ["check loyalty points", "what tier am I", "loyalty rewards balance"],
    "account_update": ["update my email", "change phone number", "update shipping address"],
    "shipping_estimate": ["how long will shipping take", "estimated delivery time", "when will it arrive"],
}

ALL_INTENTS = list(INTENT_VOCAB.keys())

# -----------------------------------------------------------------------------
# FALLBACK PRODUCTS (used if catalog is not available)
# -----------------------------------------------------------------------------
FALLBACK_PRODUCTS = [
    "Bluetooth speaker", "wireless headphones", "laptop charger",
    "smartphone case", "USB cable", "monitor", "keyboard", "mouse",
    "webcam", "microphone",
]

PROBLEMS = [
    "broken", "not working", "defective", "missing parts",
    "wrong item", "poor quality", "scratched", "damaged",
]

TIME_MARKERS = [
    "yesterday", "2 days ago", "last week", "just now", "today",
]

# -----------------------------------------------------------------------------
# PERSONAS & NOISE
# -----------------------------------------------------------------------------
PERSONAS = ["angry", "frustrated", "neutral", "polite", "confused", "impatient"]

TYPOS = {
    "order": ["oder", "ordr", "order"],
    "refund": ["refnd", "refund", "reund"],
    "issue": ["isue", "issue", "issu"],
    "delivery": ["delivry", "delivery", "dilvery"],
}

FILLERS = ["pls help", "need support", "urgent", "help asap", "??", "!!!", ""]

TRANSITIONS = [
    ". Also, ", ". Another thing, I also need to ", " and ", " as well as ", ". Can you also help me ",
]


def _load_real_products() -> list[str]:
    """Load product titles from the real catalog, fallback to hardcoded list."""
    if PRODUCT_CATALOG_PATH.exists():
        try:
            catalog = pd.read_csv(PRODUCT_CATALOG_PATH, usecols=["title"])
            titles = catalog["title"].dropna().tolist()
            if len(titles) > 0:
                # Sample up to 200 to keep variety manageable
                return random.sample(titles, min(200, len(titles)))
        except Exception:
            pass
    return FALLBACK_PRODUCTS


def _load_real_order_ids() -> list[str]:
    """Load order IDs from the generated orders, return empty list if unavailable."""
    if ORDERS_PATH.exists():
        try:
            orders = pd.read_csv(ORDERS_PATH, usecols=["order_id"])
            return orders["order_id"].tolist()
        except Exception:
            pass
    return []


# -----------------------------------------------------------------------------
# CORE STYLE FUNCTIONS
# -----------------------------------------------------------------------------
def build_query(product, problem, persona, time_marker, intent_set, order_ids):
    """Builds single or multi-sentence queries organically matching intents and personas."""
    sentences = []

    # 1. Greetings & Context Setup
    if persona == "polite":
        sentences.append(random.choice(["Hi there!", "Hello, hope you are doing well.", "Good day."]))
    elif persona == "confused":
        sentences.append(random.choice(["I am completely lost.", "Not entirely sure how this works.", "Hey, I need some clarity."]))
    elif persona == "angry":
        sentences.append(random.choice(["This is unacceptable.", "I am highly annoyed.", "Unbelievable service."]))

    # 2. Primary Intent Execution
    intent_1 = intent_set[0]
    action_phrase_1 = random.choice(INTENT_VOCAB[intent_1])

    # Inject real order IDs for order-related intents
    order_ref = ""
    if intent_1 in ("order_tracking", "order_modification", "order_cancellation") and order_ids:
        order_ref = f" (order {random.choice(order_ids)})"

    core_templates = [
        f"I am {action_phrase_1} the {product}{order_ref}.",
        f"Regarding the {product} I got {time_marker}, it is {problem} and I need to {action_phrase_1} it{order_ref}.",
        f"Can you help with my {product}? It's {problem} and I am looking to {action_phrase_1}{order_ref}.",
        f"My {product} is {problem}. How do I handle {action_phrase_1}?{order_ref}",
    ]
    primary_clause = random.choice(core_templates)

    # 3. Secondary Intent Handling (Multi-Intent Mixing)
    if len(intent_set) > 1:
        intent_2 = intent_set[1]
        action_phrase_2 = random.choice(INTENT_VOCAB[intent_2])
        transition = random.choice(TRANSITIONS)

        if transition.startswith("."):
            sentences.append(primary_clause)
            sentences.append(f"{transition.strip('. ')} {action_phrase_2} for a different issue.")
        else:
            primary_clause = primary_clause.rstrip(".") + f"{transition}{action_phrase_2}."
            sentences.append(primary_clause)
    else:
        sentences.append(primary_clause)

    # 4. Closings & Sign-offs
    if persona in ["frustrated", "impatient", "angry"]:
        sentences.append(random.choice(["Please resolve this immediately.", "Let me know ASAP!", "Waiting for your prompt response."]))
    elif persona == "polite":
        sentences.append(random.choice(["Thank you for your assistance.", "Appreciate your time!", "Have a great day."]))
    elif persona == "confused":
        sentences.append(random.choice(["Can you walk me through this step by step?", "What should my next step be?"]))

    # 5. Persona Structural Shifts
    if persona == "impatient":
        query = " ".join(sentences[1:2]) if len(sentences) > 1 else sentences[0]
    else:
        joiner = random.choice(["\n", " "])
        query = joiner.join(sentences)

    if persona == "angry" and random.random() < 0.4:
        query = query.upper()

    # 6. Noise & Typo Injections
    if random.random() < 0.25:
        query += " " + random.choice(FILLERS)

    if random.random() < 0.20:
        for word, variations in TYPOS.items():
            if word in query.lower():
                query = query.replace(word, random.choice(variations))

    return query.strip()


# -----------------------------------------------------------------------------
# MULTI-INTENT MIXER
# -----------------------------------------------------------------------------
def sample_intents():
    """65% single intent, 35% multi-intent."""
    if random.random() < 0.65:
        return [random.choice(ALL_INTENTS)]
    else:
        return random.sample(ALL_INTENTS, k=2)


# -----------------------------------------------------------------------------
# MAIN GENERATOR
# -----------------------------------------------------------------------------
def generate_enhanced_queries(total_queries: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate enhanced synthetic customer queries.

    Links to real product names and order IDs when available.
    """
    random.seed(seed)

    products = _load_real_products()
    order_ids = _load_real_order_ids()

    print(f"  Using {len(products)} product names, {len(order_ids)} order IDs")

    rows = []
    for _ in range(total_queries):
        intents = sample_intents()
        product = random.choice(products)
        problem = random.choice(PROBLEMS)
        persona = random.choice(PERSONAS)
        time_marker = random.choice(TIME_MARKERS)

        query = build_query(product, problem, persona, time_marker, intents, order_ids)

        rows.append({
            "query": query,
            "intent": intents[0],
            "all_intents": ", ".join(intents),
            "persona": persona,
            "batch_id": 1,
        })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["query"]).reset_index(drop=True)
    print(f"  Generated {len(df):,} unique queries")
    return df


# -----------------------------------------------------------------------------
# SAVE FUNCTION
# -----------------------------------------------------------------------------
def save_enhanced_queries(df: pd.DataFrame) -> Path:
    """Save enhanced queries to the configured path."""
    CUSTOMER_QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CUSTOMER_QUERIES_PATH, index=False)
    print(f"  ✅ Saved queries → {CUSTOMER_QUERIES_PATH.name} ({len(df):,} rows)")
    return CUSTOMER_QUERIES_PATH


def run(total_queries: int = 500, force: bool = False) -> pd.DataFrame:
    """Generate and save enhanced queries."""
    if CUSTOMER_QUERIES_PATH.exists() and not force:
        print(f"  [skip] Queries already exist: {CUSTOMER_QUERIES_PATH.name}")
        return pd.read_csv(CUSTOMER_QUERIES_PATH)

    df = generate_enhanced_queries(total_queries=total_queries)
    save_enhanced_queries(df)
    return df


# -----------------------------------------------------------------------------
# RUNNER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate enhanced synthetic queries.")
    parser.add_argument("--total", type=int, default=500, help="Number of queries.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(total_queries=args.total, force=args.force)
