"""
train.py
========
Fine-tunes microsoft/deberta-v3-base (or any AutoModel) for 3-class
political bias classification (Left / Center / Right).

Key design choices
------------------
- Title + " [SEP] " + content is used as the input sequence so the model
  sees both the headline signal and the article body.
- Weighted CrossEntropyLoss (class weights from sklearn 'balanced') is
  applied via a Trainer subclass to handle the Center class being
  underrepresented in train (24.8%) vs val/test (37-38%).
- Temporal splits: train ≤ 2018, val = 2019, test = 2020 (Baly only;
  all Qbias articles are appended to train). See scripts/make_splits.py.

Usage:
    # Full training run (DataHub / GPU machine):
    python scripts/train.py --config configs/config.yaml

    # Local smoke test (CPU, 200 samples, 2 epochs, W&B disabled):
    python scripts/train.py --config configs/config.yaml --smoke_test true
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed — experiment logging disabled.")


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
               label_col: str, n_samples: int = None) -> Dataset:
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

    if n_samples is not None:
        df = df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)

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
            padding=False,   # DataCollatorWithPadding handles dynamic padding
        )
    return tokenize


# ─────────────────────────────────────────────────────────────────────────────
#  Custom Trainer — weighted CrossEntropyLoss
# ─────────────────────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Overrides compute_loss to apply per-class weights, which compensates
    for the Center class being underrepresented in the training split
    (~24.8%) relative to val/test (~37-38%).
    """
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(device=logits.device, dtype=logits.dtype)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def make_compute_metrics(id2label: dict):
    """Returns a compute_metrics fn that reports accuracy, macro_f1, per-class f1."""
    label_ids   = sorted(id2label.keys())
    label_names = [id2label[i] for i in label_ids]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds       = np.argmax(logits, axis=-1)
        per_class   = f1_score(labels, preds, average=None,
                               labels=label_ids, zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro",
                                       zero_division=0)),
        }
        for name, score in zip(label_names, per_class):
            metrics[f"f1_{name}"] = float(score)
        return metrics

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  W&B confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def log_confusion_matrix(trainer: Trainer, dataset: Dataset,
                         id2label: dict, split_name: str = "val") -> None:
    label_names  = [id2label[i] for i in sorted(id2label)]
    preds_output = trainer.predict(dataset)
    preds        = np.argmax(preds_output.predictions, axis=-1)
    labels       = preds_output.label_ids

    wandb.log({
        f"confusion_matrix_{split_name}": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels.tolist(),
            preds=preds.tolist(),
            class_names=label_names,
        )
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a transformer for news political bias classification."
    )
    p.add_argument("--config",     type=Path, default=Path("configs/config.yaml"),
                   help="Path to YAML config file")
    p.add_argument("--smoke_test", type=lambda x: x.lower() == "true", default=False,
                   help="Run with 200/50 samples, 2 epochs, W&B disabled (true/false)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)

    # ── Smoke-test overrides ─────────────────────────────────────────────────
    if args.smoke_test:
        print("=" * 60)
        print("  SMOKE TEST MODE")
        print("  200 train / 50 val samples | 2 epochs | W&B disabled")
        print("=" * 60)
        cfg.training.num_epochs  = 2
        cfg.training.batch_size  = 4
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.output_dir  = "checkpoints/smoke_test"
        n_train, n_val = 200, 50
    else:
        n_train, n_val = None, None

    # ── Seeds ────────────────────────────────────────────────────────────────
    set_seeds(cfg.training.seed)

    # ── W&B init ─────────────────────────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE and not args.smoke_test
    if WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            tags=cfg.wandb.tags if isinstance(cfg.wandb.tags, list) else [],
            config=cfg.training.to_dict(),
            mode="disabled" if args.smoke_test else "online",
        )

    # ── Label mappings ───────────────────────────────────────────────────────
    # label2id: {"Left": 0, "Center": 1, "Right": 2}
    label2id = cfg.model.label2id.__dict__
    # id2label: {0: "Left", 1: "Center", 2: "Right"}  (keys are int)
    id2label = {int(k): v for k, v in cfg.model.id2label.__dict__.items()}

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer : {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=False)

    # ── Datasets ─────────────────────────────────────────────────────────────
    print(f"Loading train     : {cfg.data.train_file}")
    train_ds = load_split(Path(cfg.data.train_file), label2id,
                          cfg.data.text_column, cfg.data.label_column, n_train)
    print(f"Loading val       : {cfg.data.val_file}")
    val_ds   = load_split(Path(cfg.data.val_file), label2id,
                          cfg.data.text_column, cfg.data.label_column, n_val)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    tokenize_fn = make_tokenize_fn(tokenizer, cfg.model.max_seq_len)
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["input_text"])
    val_ds   = val_ds.map(tokenize_fn,   batched=True, remove_columns=["input_text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # ── Class weights ────────────────────────────────────────────────────────
    train_labels    = np.array(train_ds["labels"])
    unique_classes  = np.unique(train_labels)
    class_weights   = compute_class_weight("balanced",
                                           classes=unique_classes,
                                           y=train_labels)
    weights_tensor  = torch.tensor(class_weights, dtype=torch.float)
    weight_display  = {id2label[i]: f"{w:.4f}"
                       for i, w in zip(unique_classes.tolist(), class_weights)}
    print(f"  Class weights: {weight_display}")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\nLoading model     : {cfg.model.name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        use_safetensors=True,
    )

    # ── TrainingArguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size * 2,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        fp16=cfg.training.fp16 and torch.cuda.is_available(),
        eval_strategy=cfg.training.eval_strategy,
        save_strategy=cfg.training.save_strategy,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=True,
        seed=cfg.training.seed,
        report_to="wandb" if use_wandb else "none",
        logging_steps=10,
        dataloader_num_workers=0,
    )

    compute_metrics  = make_compute_metrics(id2label)
    data_collator    = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        class_weights=weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,  # 'tokenizer' removed in transformers 5.x
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nStarting training ({cfg.training.num_epochs} epochs) ...\n")
    trainer.train()

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\nRunning final val evaluation ...")
    metrics = trainer.evaluate()

    print("\n" + "=" * 55)
    print("  VALIDATION METRICS")
    print("=" * 55)
    for k, v in sorted(metrics.items()):
        if "runtime" not in k and "samples_per_second" not in k and "steps_per_second" not in k:
            print(f"  {k:<35} {v:.4f}" if isinstance(v, float) else f"  {k:<35} {v}")
    print("=" * 55)

    # ── Confusion matrix → W&B ───────────────────────────────────────────────
    if use_wandb:
        log_confusion_matrix(trainer, val_ds, id2label, split_name="val")
        wandb.finish()

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(cfg.training.output_dir)
    # save_pretrained works for both tokenizers and processors
    tokenizer.save_pretrained(cfg.training.output_dir)
    print(f"\nModel saved → {cfg.training.output_dir}\n")


if __name__ == "__main__":
    main()
