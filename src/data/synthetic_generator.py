"""
Synthetic E-Commerce Query Generator (Hugging Face API)

- IFT LLM prompting for realistic query generation
- Uses HF Inference API
- Batched synthetic query generation
- Intent-aware prompting
- Deduplication + CSV export
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import os
import requests


# ----------------------------
# CONFIG
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic"

HF_TOKEN = os.getenv("HF_TOKEN")  # safest approach

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


INTENT_GROUPS = [
    ["product_search", "product_recommendation", "product_comparison"],
    ["order_tracking", "order_modification", "order_cancellation"],
    ["returns", "refunds", "warranty_replacement"],
    ["payment_issues", "account_issues", "discounts_offers"],
    ["complaints", "delivery_issues"],
]


# ----------------------------
# PROMPT
# ----------------------------

def build_prompt(batch_size: int, intents: list[str]) -> str:
    intent_text = "\n".join([f"- {i}" for i in intents])

    return f"""
Generate {batch_size} realistic e-commerce customer support queries.

Cover these intents:
{intent_text}

Rules:
- natural human language
- mix short/long queries
- include typos, frustration, polite tone
- ambiguous and incomplete queries allowed
- ONLY return valid JSON array

Format:
[
  {{"query": "...", "intent": "..."}}
]
"""


# ----------------------------
# HF CALL
# ----------------------------

def call_hf(prompt: str):
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 1.0,
            "max_new_tokens": 800,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    response.raise_for_status()

    output = response.json()

    # HF sometimes returns list of dicts
    if isinstance(output, list):
        output = output[0]["generated_text"]

    return output


# ----------------------------
# JSON PARSE
# ----------------------------

def extract_json(text: str):
    start = text.find("[")
    end = text.rfind("]") + 1

    if start == -1 or end == -1:
        raise ValueError("Invalid JSON output from model")

    return json.loads(text[start:end])


# ----------------------------
# BATCH GENERATION
# ----------------------------

def generate_batch(batch_size: int, intents: list[str]) -> pd.DataFrame:

    prompt = build_prompt(batch_size, intents)
    raw = call_hf(prompt)
    data = extract_json(raw)

    return pd.DataFrame(data)


# ----------------------------
# MAIN GENERATOR
# ----------------------------

def generate_queries(total_queries: int = 500, batch_size: int = 100):

    dfs = []
    remaining = total_queries
    batch_id = 0

    while remaining > 0:

        current = min(batch_size, remaining)

        intents = INTENT_GROUPS[batch_id % len(INTENT_GROUPS)]

        print(f"Generating batch {batch_id + 1} ({current} queries)")

        df = generate_batch(current, intents)
        df["batch_id"] = batch_id + 1

        dfs.append(df)

        remaining -= current
        batch_id += 1

    final_df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["query"])
        .reset_index(drop=True)
    )

    return final_df


# ----------------------------
# SAVE
# ----------------------------

def save_queries(df: pd.DataFrame) -> Path:

    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SYNTHETIC_DATA_DIR / (
        f"synthetic_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    df.to_csv(output_path, index=False)

    return output_path