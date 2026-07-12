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
- Real-world query styles (structured, conversational, multi-line, fragment-based)

Returns a pandas DataFrame with synthetic queries and primary intent labels.
"""

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
    "delivery_issues": ["package stolen", "delivered to wrong house", "delayed shipment"]
}

ALL_INTENTS = list(INTENT_VOCAB.keys())

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
    ". Also, ", ". Another thing, I also need to ", " and ", " as well as ", ". Can you also help me "
]

# -----------------------------------------------------------------------------
# CORE STYLE FUNCTIONS
# -----------------------------------------------------------------------------
def build_query(
    product: str,
    problem: str,
    persona: str,
    time_marker: str,
    intent_set: list[str],
) -> str:
    """
    Builds single or multi-sentence queries organically matching intents and personas.
    """
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
    
    core_templates = [
        f"I am {action_phrase_1} the {product}.",
        f"Regarding the {product} I got {time_marker}, it is {problem} and I need to {action_phrase_1} it.",
        f"Can you help with my {product}? It's {problem} and I am looking to {action_phrase_1}.",
        f"My {product} is {problem}. How do I handle {action_phrase_1}?"
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
        # Impatient profiles bypass lines 1 and 3 completely for a snapshot 1-liner
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
def sample_intents() -> list[str]:
    """
    65% single intent, 35% multi-intent (to generate longer multi-line tickets)
    """
    if random.random() < 0.65:
        return [random.choice(ALL_INTENTS)]
    else:
        return random.sample(ALL_INTENTS, k=2)

# -----------------------------------------------------------------------------
# MAIN GENERATOR
# -----------------------------------------------------------------------------
def generate_synthetic_queries(total_queries: int = 50, seed: int = 42) -> pd.DataFrame:
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
            "intent": intents[0],  # primary intent
            "all_intents": ", ".join(intents),
            "persona": persona,
            "batch_id": 1
        })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["query"]).reset_index(drop=True)
    return df

# -----------------------------------------------------------------------------
# SAVE FUNCTION
# -----------------------------------------------------------------------------
def save_synthetic_queries(df: pd.DataFrame, filename: str | None = None) -> Path:
    """
    Save synthetic queries DataFrame to <project_root>/data/synthetic/
    Safely handles both scripted executions and interactive environments.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
    except NameError:
        # Fallback if executing inside an interactive environment like a notebook
        project_root = Path.cwd()

    save_dir = project_root / "data" / "synthetic"
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"synthetic_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    output_path = save_dir / filename
    df.to_csv(output_path, index=False)
    return output_path

# -----------------------------------------------------------------------------
# RUNNER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate DataFrame
    generated_df = generate_synthetic_queries(20)
    print("--- Sample Generated Data ---")
    print(generated_df[["query", "intent", "persona"]].head(5).to_string(index=False))
    print("\n-----------------------------\n")
    
    # 2. Save Output
    try:
        saved_file = save_synthetic_queries(generated_df)
        print(f"Success! File saved cleanly to: {saved_file}")
    except Exception as e:
        print(f"Could not save file automatically: {e}")