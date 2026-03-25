"""
make_splits.py
==============
Produces temporal train / val / test splits from combined_dataset.csv.

Split strategy
--------------
Temporal splits are applied to Baly articles only (Qbias has no dates).

  train : Baly year <= 2018  +  all Qbias articles
  val   : Baly year == 2019
  test  : Baly year == 2020

Baly rows with unparseable dates (empty strings in source — 4,443 rows,
~11.8%) are assigned to train with date_parse_ok=False. They cannot be
placed in val/test without a date so train is the conservative choice.

The single Baly article dated 2050 (data entry error) is dropped entirely.
Baly articles dated before 2012 (4 articles: 2001, 2007, 2010×2) are
assigned to train — valid data, just pre-main-corpus coverage.

Usage:
    python scripts/make_splits.py \\
        --input     data/processed/combined_dataset.csv \\
        --output_dir data/processed/
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

TRAIN_MAX_YEAR = 2018
VAL_YEAR       = 2019
TEST_YEAR      = 2020
DROP_YEAR      = 2050          # known bad data-entry date


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create temporal train/val/test splits.")
    p.add_argument("--input",      type=Path,
                   default=Path("data/processed/combined_dataset.csv"),
                   help="Path to combined_dataset.csv")
    p.add_argument("--output_dir", type=Path,
                   default=Path("data/processed/"),
                   help="Directory to write train/val/test CSVs")
    return p.parse_args()


def assign_split(row: pd.Series) -> str:
    """Return 'train', 'val', or 'test' for a single Baly row."""
    if pd.isna(row["date"]):
        return "train"
    year = row["date"].year
    if year == DROP_YEAR:
        return "drop"
    if year == VAL_YEAR:
        return "val"
    if year == TEST_YEAR:
        return "test"
    return "train"   # covers <= 2018, pre-2012 outliers, and any year > 2020


def print_summary(train: pd.DataFrame, val: pd.DataFrame,
                  test: pd.DataFrame, dropped: int) -> None:
    print("\n" + "=" * 75)
    print("  SPLIT SUMMARY")
    print("=" * 75)
    print(f"  (Dropped {dropped} article(s) with year == {DROP_YEAR})\n")

    header = f"  {'Split':<8} {'Total':>7}  {'Left':>7}  {'Center':>7}  {'Right':>7}  Date range"
    print(header)
    print("  " + "-" * 70)

    for name, df in [("train", train), ("val", val), ("test", test)]:
        lc    = df["label"].value_counts()
        left  = lc.get("Left",   0)
        ctr   = lc.get("Center", 0)
        right = lc.get("Right",  0)
        total = len(df)

        # Date range — only meaningful for Baly rows with valid dates
        valid_dates = df["date"].dropna()
        if len(valid_dates):
            dr = f"{valid_dates.min().date()} → {valid_dates.max().date()}"
        else:
            dr = "N/A (no dates)"

        print(f"  {name:<8} {total:>7,}  {left:>7,}  {ctr:>7,}  {right:>7,}  {dr}")

    print("  " + "-" * 70)
    grand_total = len(train) + len(val) + len(test)
    print(f"  {'TOTAL':<8} {grand_total:>7,}")

    # Qbias contribution to train
    qbias_in_train = (train["dataset_source"] == "qbias").sum()
    baly_in_train  = (train["dataset_source"] == "baly").sum()
    print(f"\n  Train breakdown:")
    print(f"    Baly (year <= {TRAIN_MAX_YEAR} + undated) : {baly_in_train:>7,}")
    print(f"    Qbias (all)                     : {qbias_in_train:>7,}")

    # Date parse ok flag for Baly
    baly_train = train[train["dataset_source"] == "baly"]
    dated_ok   = baly_train["date_parse_ok"].sum()
    dated_fail = (~baly_train["date_parse_ok"]).sum()
    print(f"\n  Baly train date_parse_ok=True  : {int(dated_ok):>7,}")
    print(f"  Baly train date_parse_ok=False : {int(dated_fail):>7,}  (assigned to train; date was empty string in source)")

    # Val/test class balance check
    for name, df in [("val", val), ("test", test)]:
        lc    = df["label"].value_counts()
        total = len(df)
        left_pct  = 100 * lc.get("Left",   0) / total if total else 0
        ctr_pct   = 100 * lc.get("Center", 0) / total if total else 0
        right_pct = 100 * lc.get("Right",  0) / total if total else 0
        print(f"\n  {name.capitalize()} class balance: "
              f"Left {left_pct:.1f}% / Center {ctr_pct:.1f}% / Right {right_pct:.1f}%")

    print("=" * 75)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"  Loaded {len(df):,} rows")

    # ── Separate sources ──────────────────────────────────────────────────────
    baly  = df[df["dataset_source"] == "baly"].copy()
    qbias = df[df["dataset_source"] == "qbias"].copy()
    print(f"  Baly: {len(baly):,}  |  Qbias: {len(qbias):,}")

    # ── Tag date_parse_ok ─────────────────────────────────────────────────────
    baly["date_parse_ok"] = baly["date"].notna()
    qbias["date_parse_ok"] = True   # N/A for Qbias but keep schema consistent

    # ── Assign splits to Baly ─────────────────────────────────────────────────
    baly["split"] = baly.apply(assign_split, axis=1)

    dropped_count = (baly["split"] == "drop").sum()
    baly = baly[baly["split"] != "drop"]

    # ── Assign all Qbias to train ─────────────────────────────────────────────
    qbias["split"] = "train"

    # ── Combine and split ─────────────────────────────────────────────────────
    combined = pd.concat([baly, qbias], ignore_index=True)

    train = combined[combined["split"] == "train"].drop(columns=["split"])
    val   = combined[combined["split"] == "val"].drop(columns=["split"])
    test  = combined[combined["split"] == "test"].drop(columns=["split"])

    print_summary(train, val, test, dropped_count)

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train.csv", index=False)
    val.to_csv(args.output_dir   / "val.csv",   index=False)
    test.to_csv(args.output_dir  / "test.csv",  index=False)
    print(f"\nSaved → {args.output_dir}/  [train.csv  val.csv  test.csv]\n")


if __name__ == "__main__":
    main()
