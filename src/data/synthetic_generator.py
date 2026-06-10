import random
import pandas as pd
from pathlib import Path
from datetime import datetime

"""
Synthetic Customer Query Generator

Generates realistic e-commerce customer support queries using:
- Intent-driven sampling (single + multi-intent)
- Persona-based behavior simulation (angry, polite, confused, etc.)
- Dynamic language construction (no templates)
- Noise injection (typos, fillers, urgency signals)
- Real-world query styles (structured, conversational, fragment-based)

Returns a pandas DataFrame with synthetic queries and primary intent labels.
"""

# -----------------------------------------------------------------------------
# INTENTS
# -----------------------------------------------------------------------------

INTENT_GROUPS = [
    ["product_search", "product_recommendation", "product_comparison"],
    ["order_tracking", "order_modification", "order_cancellation"],
    ["returns", "refunds", "warranty_replacement"],
    ["payment_issues", "account_issues", "discounts_offers"],
    ["complaints", "delivery_issues"],
]

ALL_INTENTS = [i for group in INTENT_GROUPS for i in group]


# -----------------------------------------------------------------------------
# DOMAIN VOCAB
# -----------------------------------------------------------------------------

PRODUCTS = [
    "Bluetooth speaker", "wireless headphones", "laptop charger",
    "smartphone case", "USB cable", "monitor", "keyboard", "mouse",
    "webcam", "microphone"
]

PROBLEMS = [
    "broken", "not working", "defective", "missing parts",
    "wrong item", "poor quality", "scratched", "damaged"
]

TIME_MARKERS = [
    "yesterday", "2 days ago", "last week", "just now", "today"
]


# -----------------------------------------------------------------------------
# PERSONAS
# -----------------------------------------------------------------------------

PERSONAS = [
    "angry",
    "frustrated",
    "neutral",
    "polite",
    "confused",
    "impatient"
]


# -----------------------------------------------------------------------------
# NOISE GENERATORS
# -----------------------------------------------------------------------------

TYPOS = {
    "order": ["oder", "ordr", "order"],
    "refund": ["refnd", "refund", "reund"],
    "issue": ["isue", "issue", "issu"],
    "delivery": ["delivry", "delivery", "dilvery"],
}

FILLERS = ["pls help", "need support", "urgent", "help asap", "??", "!!!", ""]


# -----------------------------------------------------------------------------
# CORE STYLE FUNCTIONS
# -----------------------------------------------------------------------------

def build_query(product, problem, persona, time_marker, intent_set):
    """
    Generates a realistic query based on persona + intent context.
    """

    base_mode = random.random()

    # -----------------------------------------------------------------
    # 1. Structured natural language (most realistic)
    # -----------------------------------------------------------------
    if base_mode < 0.5:
        templates = [
            f"My {product} is {problem} {time_marker}, can you help?",
            f"I received a {product} and it is {problem}",
            f"Why is my {product} {problem}?",
            f"I want refund for {product} because it is {problem}",
            f"Need help with my {product} - it is {problem}",
        ]
        query = random.choice(templates)

    # -----------------------------------------------------------------
    # 2. Conversational / chat-like input
    # -----------------------------------------------------------------
    elif base_mode < 0.8:
        fragments = [
            f"{product} issue",
            f"problem with order",
            f"return {product}",
            f"refund request",
            f"delivery problem",
            f"not working {product}",
        ]
        query = random.choice(fragments)

    # -----------------------------------------------------------------
    # 3. Minimal / search-style queries
    # -----------------------------------------------------------------
    else:
        query = random.choice([
            product,
            f"{product} {problem}",
            "order issue",
            "refund help",
            "delivery issue",
        ])

    # -----------------------------------------------------------------
    # Persona modulation
    # -----------------------------------------------------------------
    if persona == "angry":
        query = query.upper()
        query += random.choice(["!!!", "??", " WTF", ""])
    elif persona == "frustrated":
        query += random.choice([" pls help", " urgent", " need support"])
    elif persona == "confused":
        query = "not sure but " + query
    elif persona == "polite":
        query = "hello, " + query

    # -----------------------------------------------------------------
    # Noise injection (real-world imperfections)
    # -----------------------------------------------------------------
    if random.random() < 0.25:
        query += " " + random.choice(FILLERS)

    # -----------------------------------------------------------------
    # Typo injection (light realism)
    # -----------------------------------------------------------------
    if random.random() < 0.15:
        word = random.choice(list(TYPOS.keys()))
        typo = random.choice(TYPOS[word])
        query = query.replace(word, typo)

    return query.strip()


# -----------------------------------------------------------------------------
# MULTI-INTENT MIXER
# -----------------------------------------------------------------------------

def sample_intents():
    """
    80% single intent, 20% multi-intent (real support tickets)
    """
    if random.random() < 0.8:
        return [random.choice(ALL_INTENTS)]
    else:
        return random.sample(ALL_INTENTS, k=2)


# -----------------------------------------------------------------------------
# MAIN GENERATOR
# -----------------------------------------------------------------------------

def generate_synthetic_queries(total_queries: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Fully synthetic customer query generator with:
    - persona simulation
    - multi-intent support
    - time sensitivity
    - noise + typo injection
    - no templates
    """
    random.seed(seed)

    rows = []

    for _ in range(total_queries):

        intents = sample_intents()

        product = random.choice(PRODUCTS)
        problem = random.choice(PROBLEMS)
        persona = random.choice(PERSONAS)
        time_marker = random.choice(TIME_MARKERS)

        query = build_query(product, problem, persona, time_marker, intents)

        rows.append({
            "query": query,
            "intent": intents[0],   # primary intent (can extend later)
            "batch_id": 1
        })

    df = pd.DataFrame(rows)

    # light cleanup
    df = df.drop_duplicates(subset=["query"]).reset_index(drop=True)

    return df




def save_synthetic_queries(df: pd.DataFrame, filename: str = None) -> Path:
    """
    Save synthetic queries DataFrame to:
    <project_root>/data/synthetic/

    Project root is resolved as 2 levels above current file directory.

    Args:
        df: pandas DataFrame to save
        filename: optional custom filename

    Returns:
        Path to saved CSV file
    """

    # 2 levels above current file
    project_root = Path(__file__).resolve().parents[2]

    # target directory: /data/synthetic
    save_dir = project_root / "data" / "synthetic"
    save_dir.mkdir(parents=True, exist_ok=True)

    # filename handling
    if filename is None:
        filename = f"synthetic_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    output_path = save_dir / filename

    # save file
    df.to_csv(output_path, index=False)

    return output_path

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_synthetic_queries(20)
    print(df.to_string(index=False))