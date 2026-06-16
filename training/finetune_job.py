"""
Launches an OpenAI fine-tuning job on prepared JSONL data.
Monitors job status and saves the fine-tuned model ID.

Usage:
  python training/finetune_job.py --train data/finetune/train.jsonl --val data/finetune/val.jsonl
"""
import time
import argparse
from openai import OpenAI
import os

def launch_finetune(train_path: str, val_path: str, base_model: str = "gpt-3.5-turbo"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("Uploading training file...")
    with open(train_path, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")

    with open(val_path, "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")

    print(f"Creating fine-tune job on {base_model}...")
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model=base_model,
        hyperparameters={"n_epochs": 3, "learning_rate_multiplier": 0.1}
    )

    print(f"Job ID: {job.id} | Status: {job.status}")

    while job.status not in ("succeeded", "failed", "cancelled"):
        time.sleep(30)
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(f"Status: {job.status}")

    if job.status == "succeeded":
        print(f"SUCCESS! Fine-tuned model: {job.fine_tuned_model}")
        with open(".env", "a") as f:
            f.write(f"\nFINE_TUNED_MODEL={job.fine_tuned_model}")
    else:
        print(f"Job failed: {job.error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--model", default="gpt-3.5-turbo")
    args = parser.parse_args()
    launch_finetune(args.train, args.val, args.model)
