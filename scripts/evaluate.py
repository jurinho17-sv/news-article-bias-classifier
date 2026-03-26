"""
evaluate.py
===========
Loads a fine-tuned checkpoint and runs inference on a test or validation
split, printing a full evaluation report and saving results to JSON.

Does NOT retrain — loads weights only.

Usage:
    python scripts/evaluate.py \
        --config configs/config.yaml \
        --checkpoint checkpoints/smoke_test \
        --split test
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """
    Wraps a nested YAML dict so values are accessible as cfg.section.key.
    Integer YAML keys (e.g. id2label: {0: Left}) are stored as string
    attributes ('0', '1', '2') and can be recovered with int(k).
    """
    def __init__(self, d: dict):
        for k, v in d.items():
            safe_key = str(k) if not isinstance(k, str) else k
            setattr(self, safe_key, Config(v) if isinstance(v, dict) else v)

    def to_dict(self) -> dict:
        result = {}
        for k, v in self.__dict__.items():
            result[k] = v.to_dict() if isinstance(v, Config) else v
        return result


def load_config(path: Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


# ─────────────────────────────────────────────────────────────────────────────
#  Seeds
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_split(path: Path, label2id: dict, text_col: str,
               label_col: str) -> Dataset:
    """
    Load a CSV split and return a HuggingFace Dataset.
    Input text = title + " [SEP] " + content.
    """
    if not path.exists():
        print(f"ERROR: Split file not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[label_col].isin(label2id)]

    df["input_text"] = (
        df["title"].fillna("").str.strip()
        + " [SEP] "
        + df[text_col].fillna("").str.strip()
    )
    df["labels"] = df[label_col].map(label2id).astype(int)

    return Dataset.from_dict({
        "input_text": df["input_text"].tolist(),
        "labels":     df["labels"].tolist(),
    })


def make_tokenize_fn(tokenizer, max_length: int):
    def tokenize(batch):
        return tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    return tokenize


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_confusion_matrix(cm: np.ndarray, label_names: list[str]) -> None:
    """Print a confusion matrix as a formatted table."""
    col_width = max(len(n) for n in label_names + ["Actual \\ Pred"]) + 2
    # Header
    header = "Actual \\ Pred".ljust(col_width)
    for name in label_names:
        header += name.rjust(col_width)
    print(header)
    print("-" * len(header))
    # Rows
    for i, name in enumerate(label_names):
        row = name.ljust(col_width)
        for j in range(len(label_names)):
            row += str(cm[i, j]).rjust(col_width)
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a fine-tuned checkpoint on test or val split."
    )
    p.add_argument("--config", type=Path, default=Path("configs/config.yaml"),
                   help="Path to YAML config file")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to checkpoint directory")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"],
                   help="Which split to evaluate on (default: test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)

    # ── Seeds ────────────────────────────────────────────────────────────────
    set_seeds(cfg.training.seed)

    # ── Label mappings ───────────────────────────────────────────────────────
    label2id = cfg.model.label2id.__dict__
    id2label = {int(k): v for k, v in cfg.model.id2label.__dict__.items()}
    label_names = [id2label[i] for i in sorted(id2label)]

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer from: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading model from   : {args.checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=cfg.model.num_labels,
        label2id=label2id,
        id2label=id2label,
        use_safetensors=True,
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    split_file = cfg.data.test_file if args.split == "test" else cfg.data.val_file
    print(f"Loading {args.split} split : {split_file}")
    dataset = load_split(
        Path(split_file), label2id,
        cfg.data.text_column, cfg.data.label_column,
    )
    print(f"  Samples: {len(dataset):,}")

    tokenize_fn = make_tokenize_fn(tokenizer, cfg.model.max_seq_len)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["input_text"])
    dataset.set_format("torch")

    # ── Trainer (inference only) ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir="logs/eval_tmp",
        per_device_eval_batch_size=cfg.training.batch_size * 2,
        fp16=cfg.training.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {args.split} split ...")
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=-1)

    # ── Metrics ──────────────────────────────────────────────────────────────
    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    per_class_f1 = f1_score(
        labels, preds, average=None,
        labels=sorted(id2label.keys()), zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=sorted(id2label.keys()))
    cls_report = classification_report(
        labels, preds,
        labels=sorted(id2label.keys()),
        target_names=label_names,
        zero_division=0,
    )

    # ── Print report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — {args.split.upper()} SPLIT")
    print("=" * 60)

    print(f"\n  Accuracy : {accuracy:.4f}")
    print(f"  Macro-F1 : {macro_f1:.4f}")

    print("\n  Per-class F1:")
    for name, score in zip(label_names, per_class_f1):
        print(f"    {name:<10} {score:.4f}")

    print(f"\n  Confusion Matrix:")
    print_confusion_matrix(cm, label_names)

    print(f"\n  Classification Report:")
    print(cls_report)

    print("=" * 60)

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "num_samples": len(dataset),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": {
            name: float(score) for name, score in zip(label_names, per_class_f1)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
    }

    out_path = Path("logs/test_eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}\n")


if __name__ == "__main__":
    main()
