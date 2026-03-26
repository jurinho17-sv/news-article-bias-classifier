"""Gradio demo for News Article Political Bias Classifier."""

import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "jurinho17-sv/news-article-bias-classifier"
LABELS = ["Left", "Center", "Right"]

print("Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model.eval()
print("Model loaded.")


def predict(headline: str, body: str) -> dict[str, float]:
    if not headline.strip() and not body.strip():
        print("Warning: empty input received")
        return {"Left": 0.33, "Center": 0.34, "Right": 0.33}

    text = f"{headline} [SEP] {body[:2000]}"
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    return {label: prob for label, prob in zip(LABELS, probs)}


examples = [
    [
        "Biden Signs Climate Bill Into Law, Calling It Historic Victory",
        "President Biden signed the sweeping climate legislation on Tuesday, surrounded by Democratic lawmakers who called it the most significant environmental action in decades.",
    ],
    [
        "Federal Reserve Raises Interest Rates Amid Inflation Concerns",
        "The Federal Reserve raised its benchmark interest rate by a quarter point Wednesday, citing persistent inflation while acknowledging risks to economic growth.",
    ],
    [
        "Border Crisis Worsens as Illegal Crossings Hit Record High",
        "The number of illegal border crossings reached a record high last month, according to federal data, as Republican lawmakers called on the administration to take immediate action.",
    ],
]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Headline"),
        gr.Textbox(label="Article Body", lines=6),
    ],
    outputs=gr.Label(label="Predicted Bias"),
    title="News Article Political Bias Classifier",
    description=(
        "Fine-tuned DeBERTa-v3-base on 51K news articles "
        "(Baly et al. EMNLP 2020 + Qbias WebSci 2023). "
        "Trained on articles \u22642018, evaluated on held-out 2020 articles "
        "\u2014 82.9% accuracy, 83.0% macro-F1."
    ),
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()
