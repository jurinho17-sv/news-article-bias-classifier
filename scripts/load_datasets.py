"""
load_datasets.py
================
Unified loader for the two primary training data sources:

  1. Baly et al. (EMNLP 2020) — Article-Bias-Prediction
     https://github.com/ramybaly/Article-Bias-Prediction
     37,554 JSON files; AllSides labels; dates present (YYYY-MM-DD).

  2. Qbias (Haak & Schaer, WebSci 2023)
     https://github.com/irgroup/Qbias
     21,747 rows; AllSides labels; NO date or URL columns.

Combined output is deduplicated, filtered (content >= 50 chars), and
written to data/processed/combined_dataset.csv.

Label convention (output):  "Left" | "Center" | "Right"

Usage:
    python scripts/load_datasets.py \\
        --baly_dir  /tmp/baly/data/jsons \\
        --qbias_csv /tmp/qbias/allsides_balanced_news_headlines-texts.csv \\
        --output    data/processed/combined_dataset.csv
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Label normalisation map ──────────────────────────────────────────────────
LABEL_MAP = {
    "left":   "Left",
    "center": "Center",
    "centre": "Center",
    "right":  "Right",
    "0":      "Left",
    "1":      "Center",
    "2":      "Right",
    0:        "Left",
    1:        "Center",
    2:        "Right",
}

TEXT_LEN_MIN = 50


# ─────────────────────────────────────────────────────────────────────────────
#  Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_baly(baly_dir: Path) -> pd.DataFrame:
    """
    Load all JSON files from the Article-Bias-Prediction jsons/ directory.

    Extracts: ID, title, content (content_original preferred), source,
    url, date, bias_text → normalised label.
    Date format discovered in corpus: YYYY-MM-DD (ISO 8601).
    """
    if not baly_dir.exists():
        print(f"ERROR: Baly dir not found: {baly_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = list(baly_dir.glob("*.json"))
    # Also handle files with no extension (repo stores files without .json)
    if not json_files:
        json_files = [p for p in baly_dir.iterdir() if p.is_file()]

    if not json_files:
        print(f"ERROR: No files found in {baly_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading {len(json_files):,} Baly JSON files...")

    records = []
    content_fallback_count = 0
    for path in json_files:
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Prefer content_original (retains original punctuation/spacing)
        content = (d.get("content_original") or "").strip()
        if not content:
            content = (d.get("content") or "").strip()
            if content:
                content_fallback_count += 1

        records.append({
            "id":             d.get("ID", path.stem),
            "title":          (d.get("title") or "").strip(),
            "content":        content,
            "source":         (d.get("source") or "").strip(),
            "url":            (d.get("url") or "").strip(),
            "date":           d.get("date"),          # raw string, parsed below
            "label_raw":      d.get("bias_text", ""),
            "dataset_source": "baly",
        })

    df = pd.DataFrame(records)

    # Parse dates — format is YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    print(f"    Date format detected   : YYYY-MM-DD")
    print(f"    content_original used  : {len(df) - content_fallback_count:,}")
    print(f"    content fallback used  : {content_fallback_count:,}")
    print(f"    Date parse failures    : {df['date'].isna().sum():,}")

    return df


def load_qbias(qbias_csv: Path) -> pd.DataFrame:
    """
    Load the Qbias CSV (allsides_balanced_news_headlines-texts.csv).

    Column mapping:
        heading      → title   (individual article headline)
        text         → content (full article body — present)
        source       → source
        bias_rating  → label
        title        → topic_group (dropped; it's a shared topic label, not article title)

    Date: NOT present in this dataset — set to NaT and flagged.
    URL:  NOT present in this dataset.
    """
    if not qbias_csv.exists():
        print(f"ERROR: Qbias CSV not found: {qbias_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading Qbias CSV: {qbias_csv.name} ...")
    raw = pd.read_csv(qbias_csv)
    print(f"    Raw shape              : {raw.shape}")

    # Validate expected columns
    required = {"heading", "text", "bias_rating"}
    missing = required - set(raw.columns)
    if missing:
        # Try alternate column names
        alt_map = {"heading": "title", "bias_rating": "label"}
        missing_after_alt = {c for c in missing if alt_map.get(c) not in raw.columns}
        if missing_after_alt:
            print(f"ERROR: Qbias CSV missing columns: {missing_after_alt}", file=sys.stderr)
            print(f"  Found columns: {raw.columns.tolist()}", file=sys.stderr)
            sys.exit(1)

    title_col   = "heading" if "heading" in raw.columns else "title"
    content_col = "text"
    label_col   = "bias_rating"
    source_col  = "source" if "source" in raw.columns else None

    df = pd.DataFrame({
        "id":             raw.index.astype(str).map(lambda x: f"qbias_{x}"),
        "title":          raw[title_col].fillna("").str.strip(),
        "content":        raw[content_col].fillna("").str.strip(),
        "source":         raw[source_col].fillna("") if source_col else "",
        "url":            "",                                 # not available
        "date":           pd.NaT,                             # not available
        "label_raw":      raw[label_col].fillna(""),
        "dataset_source": "qbias",
    })

    print(f"    ⚠  No date column in Qbias — date set to NaT for all rows.")
    print(f"    ⚠  No URL column in Qbias — url set to empty string.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["label_raw"].map(LABEL_MAP)
    unrecognised = df["label"].isna().sum()
    if unrecognised:
        print(f"  WARNING: {unrecognised:,} rows have unrecognised labels — dropping.")
        df = df.dropna(subset=["label"])
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # 1. Deduplicate on URL (non-empty only)
    url_mask = df["url"].str.strip().ne("")
    url_dups = df[url_mask].duplicated(subset=["url"], keep="first")
    df = df[~(url_mask & url_dups)]
    after_url = len(df)

    # 2. Deduplicate on exact title match (case-sensitive, across datasets)
    title_dups = df.duplicated(subset=["title"], keep="first")
    df = df[~title_dups]
    after_title = len(df)

    print(f"  Dedup on URL   : removed {before - after_url:,} rows")
    print(f"  Dedup on title : removed {after_url - after_title:,} rows")
    return df


def filter_content(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["content"].str.len() >= TEXT_LEN_MIN].copy()
    print(f"  Content filter (>= {TEXT_LEN_MIN} chars) : removed {before - len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(raw_baly: pd.DataFrame, raw_qbias: pd.DataFrame,
                  combined: pd.DataFrame) -> None:
    total_raw = len(raw_baly) + len(raw_qbias)
    print("\n" + "=" * 60)
    print("  COMBINED DATASET SUMMARY")
    print("=" * 60)

    print(f"\nBefore dedup/filter:")
    print(f"  Baly     : {len(raw_baly):>7,} articles")
    print(f"  Qbias    : {len(raw_qbias):>7,} articles")
    print(f"  Total    : {total_raw:>7,} articles")

    print(f"\nAfter dedup + content filter:")
    print(f"  Total    : {len(combined):>7,} articles  "
          f"(-{total_raw - len(combined):,} removed)")

    # Per-dataset breakdown
    print("\nPer-dataset breakdown:")
    for src in ["baly", "qbias"]:
        sub = combined[combined["dataset_source"] == src]
        print(f"\n  [{src.upper()}]  {len(sub):,} articles")
        lc = sub["label"].value_counts()
        for lbl in ["Left", "Center", "Right"]:
            n = lc.get(lbl, 0)
            print(f"    {lbl:<8} {n:>6,}  ({100 * n / len(sub):5.1f}%)")

    # Combined label distribution
    print("\nCombined label distribution:")
    lc = combined["label"].value_counts()
    total = len(combined)
    for lbl in ["Left", "Center", "Right"]:
        n = lc.get(lbl, 0)
        bar = "█" * (n // 500)
        print(f"  {lbl:<8} {n:>6,}  ({100 * n / total:5.1f}%)  {bar}")

    # Date range (Baly only — Qbias has no dates)
    baly_combined = combined[combined["dataset_source"] == "baly"]
    valid_dates = baly_combined["date"].dropna()
    if len(valid_dates):
        print(f"\nDate range (Baly only — Qbias has no dates):")
        print(f"  Min : {valid_dates.min().date()}")
        print(f"  Max : {valid_dates.max().date()}")

        print("\nArticles per year (Baly):")
        year_counts = baly_combined["date"].dt.year.value_counts().sort_index()
        for year, n in year_counts.items():
            bar = "█" * (n // 100)
            print(f"  {int(year)}  {n:>5,}  {bar}")
    else:
        print("\nDate range : no valid dates found")

    # Sources
    unique_sources = combined["source"].replace("", pd.NA).dropna().nunique()
    print(f"\nUnique sources : {unique_sources:,}")
    print("\nTop 10 sources:")
    top_sources = combined["source"].replace("", pd.NA).dropna().value_counts().head(10)
    for src, n in top_sources.items():
        print(f"  {src:<40} {n:>5,}")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build combined training dataset.")
    p.add_argument("--baly_dir",  type=Path, required=True,
                   help="Path to Article-Bias-Prediction jsons/ directory")
    p.add_argument("--qbias_csv", type=Path, required=True,
                   help="Path to Qbias CSV file")
    p.add_argument("--output",    type=Path,
                   default=Path("data/processed/combined_dataset.csv"),
                   help="Output CSV path (default: data/processed/combined_dataset.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("─" * 60)
    print("Loading Baly et al.")
    print("─" * 60)
    baly_raw = load_baly(args.baly_dir)

    print("\n" + "─" * 60)
    print("Loading Qbias")
    print("─" * 60)
    qbias_raw = load_qbias(args.qbias_csv)

    print("\n" + "─" * 60)
    print("Combining and cleaning")
    print("─" * 60)
    combined = pd.concat([baly_raw, qbias_raw], ignore_index=True)
    combined = normalise_labels(combined)
    combined = deduplicate(combined)
    combined = filter_content(combined)

    # Final column order
    combined = combined[[
        "id", "title", "content", "source", "url",
        "date", "label", "dataset_source",
    ]].reset_index(drop=True)

    print_summary(baly_raw, qbias_raw, combined)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nSaved {len(combined):,} rows → {args.output}\n")


if __name__ == "__main__":
    main()
