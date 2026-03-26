# News Article Political Bias Classifier

Fine-tuned DeBERTa-v3 transformer for three-class political bias detection in news articles.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-Logging-orange.svg)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Results

| Split | Articles | Accuracy | Macro-F1 |
|-------|----------|----------|----------|
| **Val (2019)** | 4,621 | **91.1%** | **91.1%** |
| **Test (2020)** | 3,946 | **82.9%** | **83.0%** |

**Per-class F1 on test set:**

| Class | F1 |
|-------|----|
| Left | 82.3% |
| Center | 81.8% |
| Right | 85.0% |

---

## Dataset

Two corpora, unified under a single label taxonomy (Left / Center / Right) derived from [AllSides](https://www.allsides.com/) media bias ratings:

| Source | Articles | Notes |
|--------|----------|-------|
| Baly et al. (EMNLP 2020) | 37,554 | JSON files with dates, used for temporal splits |
| Qbias (WebSci 2023) | 13,512 | CSV, no dates — assigned entirely to training |
| **Combined (after dedup)** | **51,066** | **719 unique news sources** |

**Label distribution (full dataset):**

| Label | Count | Share |
|-------|-------|-------|
| Left | 19,308 | 37.8% |
| Center | 13,798 | 27.0% |
| Right | 17,960 | 35.2% |

See [Temporal Split Design](#temporal-split-design) for how train/val/test were constructed.

**References:**

- Baly, R., Da San Martino, G., Glass, J., & Nakov, P. (2020). [We Can Detect Your Bias: Predicting the Political Ideology of News Articles](https://aclanthology.org/2020.emnlp-main.404/). *Proceedings of EMNLP 2020*.
- Haak, F. & Schaer, P. (2023). [Qbias — A Dataset on Media Bias in Search Queries and Query Suggestions](https://dl.acm.org/doi/10.1145/3578503.3583605). *Proceedings of WebSci 2023*.

---

## Model & Training

**Architecture:** [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) with a 3-class sequence classification head.

**Why DeBERTa-v3 over RoBERTa:** DeBERTa-v3 uses disentangled attention (separate content and position embeddings) and an enhanced mask decoder, consistently outperforming RoBERTa-base on GLUE/SuperGLUE benchmarks at the same parameter count. The v3 variant further improves efficiency with ELECTRA-style replaced-token detection pretraining.

**Input format:** `title [SEP] content`, tokenized to a maximum of 512 tokens with dynamic padding per batch.

**Hyperparameters** (from `configs/config.yaml`):

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 16 (x2 gradient accumulation = 32 effective) |
| Epochs | 5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| FP16 | Enabled (CUDA) |
| Loss | Cross-entropy with balanced class weights |
| Best-model metric | Macro F1 |
| Seed | 42 |

Class weights are computed via `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` to compensate for Center-class underrepresentation in training data.

**Infrastructure:** NVIDIA L40 (48 GB), DataHub. Experiment tracking with [Weights & Biases](https://wandb.ai/).

---

## Temporal Split Design

Random train/test splits on news data cause **temporal leakage**: articles from the same news cycle share vocabulary, framing, and topics. A model trained on 2020 articles and tested on other 2020 articles exploits this overlap, inflating accuracy without learning generalizable bias signals.

We solve this with a strict year-based split on the Baly corpus (the only source with publication dates):

```
           TRAIN              VAL       TEST
  ┌─────────────────────┐ ┌────────┐ ┌────────┐
  │  Baly articles      │ │  Baly  │ │  Baly  │
  │  year ≤ 2018        │ │  2019  │ │  2020  │
  │  + ALL Qbias        │ │        │ │        │
  │  + undated Baly     │ │        │ │        │
  └─────────────────────┘ └────────┘ └────────┘
  ◄──────────────────────────────────────────────►
  2012              2018   2019       2020    time
```

| Split | Articles |
|-------|----------|
| Train | 42,499 |
| Val | 4,621 |
| Test | 3,946 |

Qbias articles lack dates and are conservatively assigned to training. Baly articles with unparseable dates (~4,400) are also routed to training. Articles dated 2050 (data entry errors) are dropped.

This design means the model is evaluated on news it has never seen from a future time period — a realistic proxy for production deployment.

---

## Repo Structure

```
news-article-bias-classifier/
├── configs/
│   └── config.yaml                    # All hyperparameters and paths
├── scripts/
│   ├── load_datasets.py               # Merge Baly + Qbias into unified CSV
│   ├── make_splits.py                 # Temporal train/val/test split
│   ├── generate_training_data.py      # Canonical training data with filters
│   ├── train.py                       # DeBERTa fine-tuning (W&B + weighted loss)
│   └── evaluate.py                    # Test-set evaluation and metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA
│   ├── 02_web_scraping.ipynb          # NewsAPI + Trafilatura scraping
│   ├── 03_data_cleaning.ipynb         # Text normalization and dedup
│   ├── 04_data_transformation.ipynb   # Wide-to-long ETL
│   ├── 05_feature_extraction.ipynb    # Embedding experiments
│   ├── 06_model_experiments.ipynb     # Baseline model comparisons
│   └── 07_final_model.ipynb           # Final model notebook
├── data/
│   ├── raw/                           # Source data (not committed)
│   └── processed/                     # Generated CSVs
├── checkpoints/                       # Saved model weights
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/jurinho17-sv/news-article-bias-classifier.git
cd news-article-bias-classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data

Download the raw datasets and place them under `data/raw/`:
- **Baly:** JSON files in `data/raw/baly/jsons/`
- **Qbias:** CSV at `data/raw/qbias/allsides_news_complete.csv`

### 3. Run the pipeline

```bash
# Step 1: Merge datasets into a single CSV
python scripts/load_datasets.py \
    --baly_dir data/raw/baly/jsons \
    --qbias_csv data/raw/qbias/allsides_news_complete.csv \
    --output data/processed/combined_dataset.csv

# Step 2: Create temporal train/val/test splits
python scripts/make_splits.py \
    --input data/processed/combined_dataset.csv \
    --output_dir data/processed/

# Step 3: Generate canonical training data
python scripts/generate_training_data.py \
    --input data/processed/preprocess3.csv \
    --output data/processed/full_training_data.csv

# Step 4: Fine-tune DeBERTa
python scripts/train.py --config configs/config.yaml

# Step 5: Evaluate on held-out test set
python scripts/evaluate.py --config configs/config.yaml
```

For a quick sanity check, run training in smoke-test mode (200 samples, 2 epochs, CPU):

```bash
python scripts/train.py --config configs/config.yaml --smoke_test true
```

---

## Demo

🚀 [**Live Demo**](https://huggingface.co/spaces/jurinho17-sv/news-article-bias-classifier) — try it on HuggingFace Spaces | [Model weights](https://huggingface.co/jurinho17-sv/news-article-bias-classifier)

---

## Limitations

- **Three-class granularity only.** The Left/Center/Right taxonomy collapses a continuous spectrum into three bins. Finer-grained bias detection (e.g., lean-left vs. left) is not captured.
- **English only.** All training data is English-language news from predominantly US sources. The model will not generalize to other languages or political systems.
- **Source-level labels, not article-level.** Bias labels originate from AllSides ratings of *news outlets*, not individual articles. An article from a Left-rated source may itself be centrist. This is a known ceiling on label quality shared by both source corpora.

---

## Citation

If you use this work, please cite the underlying datasets:

```bibtex
@inproceedings{baly-etal-2020-detect,
    title     = "We Can Detect Your Bias: Predicting the Political Ideology
                 of News Articles",
    author    = "Baly, Ramy and Da San Martino, Giovanni and Glass, James
                 and Nakov, Preslav",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods
                 in Natural Language Processing (EMNLP)",
    year      = "2020",
    publisher = "Association for Computational Linguistics",
    pages     = "4982--4991",
}

@inproceedings{haak-schaer-2023-qbias,
    title     = "Qbias -- A Dataset on Media Bias in Search Queries and
                 Query Suggestions",
    author    = "Haak, Fabian and Schaer, Philipp",
    booktitle = "Proceedings of the 15th ACM Web Science Conference (WebSci)",
    year      = "2023",
    publisher = "Association for Computing Machinery",
}
```
