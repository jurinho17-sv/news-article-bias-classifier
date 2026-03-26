---
language: en
license: mit
tags:
  - text-classification
  - political-bias
  - deberta
  - news
datasets:
  - baly-et-al-2020
  - qbias-2023
metrics:
  - accuracy
  - f1
model-index:
  - name: news-article-bias-classifier
    results:
      - task:
          type: text-classification
          name: Political Bias Classification
        dataset:
          name: Baly + Qbias (temporal test split, 2020)
          type: custom
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.829
          - name: Macro F1
            type: f1
            value: 0.830
---

# News Article Political Bias Classifier

A fine-tuned [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) model for classifying the political bias of English-language news articles into three classes: **Left**, **Center**, and **Right**.

## Model Description

This model takes a news article's title and body as input and predicts its political leaning on a three-point scale. It builds on `microsoft/deberta-v3-base` (86M parameters) by adding a sequence classification head trained with weighted cross-entropy loss to handle class imbalance.

The input is formatted as `title [SEP] content` and tokenized to a maximum of 512 tokens. The model was trained on 42,499 articles and evaluated using a strict temporal hold-out: training data covers articles published through 2018, the validation set contains 2019 articles, and the test set contains 2020 articles.

## Intended Use

**Intended uses:**
- Academic research on media bias and political framing
- Media literacy tools that help readers understand the political leaning of news sources
- Exploratory analysis of news corpora for bias patterns

**Out-of-scope uses:**
- Automated political censorship or content moderation decisions
- Assigning political labels to individuals based on their reading habits
- High-stakes decision-making without human review
- Non-English text or non-US political contexts

## Training Data

The training corpus combines two publicly available datasets, both labeled using [AllSides](https://www.allsides.com/) media bias ratings:

| Source | Articles | Citation |
|--------|----------|----------|
| Baly et al. (EMNLP 2020) | 37,554 | Article-Bias-Prediction dataset |
| Qbias (WebSci 2023) | 13,512 | Haak & Schaer query-bias dataset |
| **Combined (after dedup)** | **51,066** | **719 unique news sources** |

**Label distribution (full dataset):**

| Label | Count | Share |
|-------|-------|-------|
| Left | 19,308 | 37.8% |
| Center | 13,798 | 27.0% |
| Right | 17,960 | 35.2% |

**Temporal split design:** To prevent temporal leakage (where articles from the same news cycle appear in both train and test), splits are defined by publication year on the Baly corpus. Qbias articles lack dates and are assigned entirely to training.

| Split | Source | Articles |
|-------|--------|----------|
| Train | Baly year ≤ 2018 + all Qbias + undated Baly | 42,499 |
| Val | Baly year = 2019 | 4,621 |
| Test | Baly year = 2020 | 3,946 |

## Evaluation

All metrics are computed on temporally held-out data the model never saw during training.

| Split | Accuracy | Macro-F1 | F1-Left | F1-Center | F1-Right |
|-------|----------|----------|---------|-----------|----------|
| Val (2019) | 91.1% | 91.1% | — | — | — |
| Test (2020) | 82.9% | 83.0% | 82.3% | 81.8% | 85.0% |

The 8.2-point accuracy gap between validation and test reflects genuine temporal drift: the 2020 news landscape (dominated by COVID-19 and the US presidential election) introduced topics and framing not present in earlier training data.

## Limitations & Bias

- **Source-level labels, not article-level.** Bias labels come from AllSides ratings of news *outlets*, not individual articles. A specific article from a Left-rated source may itself be centrist or factual reporting. This is a known ceiling on label quality inherited from both source corpora.
- **Three-class granularity is a simplification.** The Left/Center/Right taxonomy collapses a continuous political spectrum into three bins. Finer distinctions (e.g., lean-left vs. far-left) are not captured.
- **English only, US-centric.** All training data is English-language news from predominantly US sources. The model will not generalize to other languages, countries, or political systems.
- **Annotator and platform bias.** AllSides ratings reflect the judgments of their editorial process. Systematic biases in those ratings propagate into the model.
- **Temporal generalization.** The model was trained on articles through 2018 and tested on 2020. Performance on articles from substantially later periods (e.g., 2024+) is unknown and may degrade as political discourse and media landscapes evolve.

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "jurinho17-sv/news-article-bias-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

title = "Senate passes bipartisan infrastructure bill"
content = "The US Senate voted 69-30 to pass a $1 trillion infrastructure package..."

inputs = tokenizer(
    title + " [SEP] " + content,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()

labels = {0: "Left", 1: "Center", 2: "Right"}
print(f"Prediction: {labels[pred_id]} ({probs[0][pred_id]:.1%})")
```

## Training Procedure

| Parameter | Value |
|-----------|-------|
| Base model | `microsoft/deberta-v3-base` |
| Max sequence length | 512 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 16 per device (x2 gradient accumulation = 32 effective) |
| Epochs | 5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| FP16 | Enabled |
| Loss function | CrossEntropyLoss with balanced class weights |
| Best-model selection | Macro F1 on validation set |
| Seed | 42 |
| Hardware | NVIDIA L40 (48 GB), DataHub (UC Berkeley) |
| Framework | PyTorch + HuggingFace Transformers |

Class weights were computed using `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` to compensate for Center-class underrepresentation in the training split (~27%) relative to the uniform prior.

## References

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
