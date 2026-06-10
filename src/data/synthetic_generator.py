"""
Synthetic E-Commerce Query Generation

Generates diverse customer queries across key e-commerce intents using
an Instruction Fine-Tuned LLM. Supports batched generation, intent labeling, deduplication,
reproducibility, and timestamped dataset export.

Output:
    - query
    - intent

Saved to:
    data/synthetic/
"""

from pathlib import Path
from datetime import datetime
import json
import random

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

INTENT_GROUPS = [
    [
        "product_search",
        "product_recommendation",
        "product_comparison",
        "stock_availability",
    ],
    [
        "order_tracking",
        "order_modification",
        "order_cancellation",
        "shipping_questions",
    ],
    [
        "returns",
        "refunds",
        "warranty_replacement",
    ],
    [
        "payment_issues",
        "account_issues",
        "discounts_offers",
    ],
    [
        "complaints",
        "delivery_issues",
    ],
]


def load_model(model_name: str = MODEL_NAME):
    """Load tokenizer and LLM."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    return tokenizer, model


def build_prompt(
    batch_size: int,
    intents: list[str],
) -> str:
    """Build intent-focused generation prompt."""

    intent_text = "\n".join(
        f"- {intent}" for intent in intents
    )

    return f"""
Generate exactly {batch_size} unique customer queries for an e-commerce chatbot.

Target intents:
{intent_text}

Requirements:
- Natural customer language
- Short, medium, and long queries
- Questions, requests, complaints, and follow-ups
- Polite, neutral, and frustrated tones
- Occasional typos and informal wording
- Vague and incomplete requests
- High diversity with minimal repetition
- Realistic shopping and support scenarios

Return ONLY valid JSON:

[
  {{
    "query": "Where is my order?",
    "intent": "order_tracking"
  }}
]
"""


def extract_json(response: str):
    """Extract JSON payload from model output."""

    start = response.find("[")
    end = response.rfind("]") + 1

    if start == -1 or end <= 0:
        raise ValueError(
            "Failed to extract JSON from model output."
        )

    return json.loads(response[start:end])


def generate_batch(
    tokenizer,
    model,
    batch_size: int,
    intents: list[str],
) -> pd.DataFrame:
    """Generate one batch of synthetic queries."""

    prompt = build_prompt(
        batch_size=batch_size,
        intents=intents,
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    temperature = random.uniform(0.9, 1.3)

    outputs = model.generate(
        **inputs,
        max_new_tokens=3000,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    data = extract_json(response)

    return pd.DataFrame(data)


def generate_queries(
    tokenizer,
    model,
    total_queries: int = 500,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Generate and combine all batches."""

    dfs = []

    remaining = total_queries
    batch_idx = 0

    while remaining > 0:

        current_batch_size = min(
            batch_size,
            remaining,
        )

        intents = INTENT_GROUPS[
            batch_idx % len(INTENT_GROUPS)
        ]

        print(
            f"Generating batch "
            f"{batch_idx + 1} "
            f"({current_batch_size} queries)"
        )

        batch_df = generate_batch(
            tokenizer=tokenizer,
            model=model,
            batch_size=current_batch_size,
            intents=intents,
        )

        batch_df["batch_id"] = batch_idx + 1

        dfs.append(batch_df)

        remaining -= current_batch_size
        batch_idx += 1

    df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["query"])
        .reset_index(drop=True)
    )

    required_cols = {
        "query",
        "intent",
    }

    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_cols}, "
            f"found {set(df.columns)}"
        )

    return df


def save_queries(
    df: pd.DataFrame,
) -> Path:
    """Save dataset to synthetic data directory."""

    SYNTHETIC_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        SYNTHETIC_DATA_DIR
        / f"synthetic_ecommerce_queries_{timestamp}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return output_path


def main():

    tokenizer, model = load_model()

    df = generate_queries(
        tokenizer=tokenizer,
        model=model,
        total_queries=500,
        batch_size=100,
    )

    output_path = save_queries(df)

    print(
        f"\nGenerated {len(df)} unique queries"
    )
    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()