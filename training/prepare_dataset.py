"""
Fine-tuning dataset preparation pipeline.
Converts raw clinical Q&A pairs into OpenAI fine-tuning JSONL format.

Steps:
1. Load raw clinical case data (CSV/JSON)
2. Clean and validate entries
3. Format as system/user/assistant JSONL
4. Split train/validation (90/10)
5. Upload to OpenAI Files API

Usage:
  python training/prepare_dataset.py --input data/clinical_cases.csv --output data/finetune/
"""
import json
import csv
import argparse
import random
from pathlib import Path


def format_as_chat(case: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are HealthAI, a clinical decision-support AI."},
            {"role": "user", "content": case["prompt"]},
            {"role": "assistant", "content": case["response"]}
        ]
    }


def prepare_dataset(input_path: str, output_dir: str, val_split: float = 0.1):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cases = []

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("prompt") and row.get("response"):
                cases.append(format_as_chat(row))

    random.shuffle(cases)
    split = int(len(cases) * (1 - val_split))
    train, val = cases[:split], cases[split:]

    for name, data in [("train.jsonl", train), ("val.jsonl", val)]:
        with open(Path(output_dir) / name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    print(f"Train: {len(train)} | Validation: {len(val)}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/finetune")
    args = parser.parse_args()
    prepare_dataset(args.input, args.output)
