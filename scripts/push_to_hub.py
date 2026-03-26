"""Push fine-tuned DeBERTa checkpoint to HuggingFace Hub."""

import os
import sys

CHECKPOINT_PATH = "./checkpoints/deberta-v3-base-run1/checkpoint-6645"

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint not found at: {CHECKPOINT_PATH}")
    print("Make sure you run this script from the project root on DataHub.")
    sys.exit(1)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

token = os.environ["HF_TOKEN"]
repo_id = "jurinho17-sv/news-article-bias-classifier"

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
model.config.id2label = {0: "Left", 1: "Center", 2: "Right"}
model.config.label2id = {"Left": 0, "Center": 1, "Right": 2}

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

print(f"Pushing model to {repo_id}...")
model.push_to_hub(repo_id, token=token)

print(f"Pushing tokenizer to {repo_id}...")
tokenizer.push_to_hub(repo_id, token=token)

print("✅ Push complete")
