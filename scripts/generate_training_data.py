"""
generate_training_data.py
=========================
Produces the canonical training dataset for the news bias classifier.

Usage:
    python scripts/generate_training_data.py [--input PATH]

By default loads data/processed/preprocess3.csv (output of
04_data_transformation.ipynb), applies a single consistent text-length
filter (>= 50 chars), and writes ALL articles — regardless of topic —
to data/processed/full_training_data.csv.

Why this script exists
----------------------
07_final_model.ipynb previously loaded 'cor.csv', a Coronavirus-only
subset created ad-hoc on Colab and never committed to the repo. That
meant reported accuracy was measured on a single-topic slice (~1,200
articles) rather than the full 24K+ article corpus, and the pipeline
was not reproducible by anyone else.

This script replaces that ad-hoc step with an explicit, versioned stage.

Date column note
----------------
preprocess3.csv does NOT contain a Date column — it was dropped in
04_data_transformation.ipynb Cell 1. Date-based stats and temporal
splits require pointing --input at preprocess1.csv (output of
03_data_cleaning.ipynb), which retains the Date field. The script
handles both cases and reports which stats are available.

Text-length filter consolidation
---------------------------------
Four inconsistent thresholds appeared across the notebooks:
  NB 01 / NB 03 : text_len >  25 chars
  NB 04         : text_len >= 10 chars
  NB 05 / NB 06 : text_len > 150 chars

This script standardises on >= 50 chars, which removes truly trivial
entries while keeping the majority of valid short articles.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Reproducibility seeds ────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "preprocess3.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "full_training_data.csv"
TEXT_LEN_MIN = 50
REQUIRED_COLS = {"text", "lean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the canonical training CSV from cleaned pipeline output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the cleaned input CSV (default: data/processed/preprocess3.csv). "
             "Use data/processed/preprocess1.csv to retain the Date column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the output CSV (default: data/processed/full_training_data.csv).",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: Input file not found: {path}", file=sys.stderr)
        print(
            "\nThis file is a Colab artifact not committed to the repo. "
            "Run notebooks 01–04 in sequence (on Colab or locally with the "
            "full Kaggle dataset) to produce it, then copy the output CSV "
            "to data/processed/ before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(path, engine="python")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"ERROR: Input CSV is missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)

    # Drop rows with null text or label
    df = df.dropna(subset=["text", "lean"])
    after_null = len(df)

    # Unified text-length filter (>= 50 chars)
    df["text_len"] = df["text"].str.len()
    df = df[df["text_len"] >= TEXT_LEN_MIN].copy()
    after_len = len(df)

    print(f"  Rows in input            : {initial:>8,}")
    print(f"  After dropping nulls     : {after_null:>8,}  (-{initial - after_null:,})")
    print(f"  After text_len >= {TEXT_LEN_MIN} chars : {after_len:>8,}  (-{after_null - after_len:,})")

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("  TRAINING DATA SUMMARY")
    print("=" * 55)

    print(f"\nTotal rows : {len(df):,}")

    # Class distribution
    print("\nClass distribution:")
    counts = df["lean"].value_counts()
    total = len(df)
    for label in sorted(counts.index):
        n = counts[label]
        print(f"  {label:<8} {n:>7,}  ({100 * n / total:5.1f}%)")

    # Date range and articles per year (only if Date column present)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        valid_dates = df["Date"].dropna()
        if len(valid_dates):
            print(f"\nDate range : {valid_dates.min().date()}  →  {valid_dates.max().date()}")
            print("\nArticles per year:")
            year_counts = df["Date"].dt.year.value_counts().sort_index()
            for year, n in year_counts.items():
                bar = "█" * (n // 100)
                print(f"  {int(year)}  {n:>5,}  {bar}")
        else:
            print("\nDate range : all Date values are null after parsing")
    else:
        print(
            "\nDate range : N/A — Date column absent from this input file.\n"
            "             Re-run with --input pointing to preprocess1.csv\n"
            "             (output of 03_data_cleaning.ipynb) for date stats\n"
            "             and temporal train/test splits."
        )

    print("\n" + "=" * 55)


def main() -> None:
    args = parse_args()

    print(f"Input  : {args.input}")
    print(f"Output : {args.output}\n")

    print("Loading and validating...")
    df = load_and_validate(args.input)

    print("Applying filters...")
    df = apply_filters(df)

    # Keep only the columns we need for training
    keep_cols = ["text", "lean"]
    if "Topics" in df.columns:
        keep_cols.append("Topics")
    if "Date" in df.columns:
        keep_cols.append("Date")
    if "source" in df.columns:
        keep_cols.append("source")
    df = df[keep_cols].reset_index(drop=True)

    print_summary(df)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df):,} rows → {args.output}")


if __name__ == "__main__":
    main()
