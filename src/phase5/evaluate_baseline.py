import os
import json
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.models.saf_rag.sycophancy_detector import SycophancyDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "google/flan-t5-base"
DATA_PATH = "data/processed/saf_training_clean"

OUTPUT_FILE = "results/baseline_eval.json"

os.makedirs("results", exist_ok=True)


def load_model():
    print("[INFO] Loading baseline model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

    detector = SycophancyDetector(device=DEVICE)

    model.eval()
    detector.eval()

    return tokenizer, model, detector


def load_data():
    dataset = load_from_disk(DATA_PATH)
    return dataset["test"]


def generate(model, tokenizer, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_length=128,
        do_sample=True,
        top_p=0.9,
        temperature=1.0
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate():
    tokenizer, model, detector = load_model()
    dataset = load_data()

    predictions = []
    targets = []

    print("[INFO] Running baseline evaluation...")

    for sample in tqdm(dataset):
        inp = sample["input"]
        tgt = sample["target"]

        pred = generate(model, tokenizer, inp)

        predictions.append(pred)
        targets.append(tgt)

    # Exact match
    exact = sum(
        1 for p, t in zip(predictions, targets)
        if p.strip().lower() == t.strip().lower()
    ) / len(predictions)

    # Sycophancy
    risks = [
        detector.get_risk_score(p)
        for p in predictions
    ]

    avg_risk = sum(risks) / len(risks)

    results = {
        "exact_match": exact,
        "avg_sycophancy_risk": avg_risk,
        "num_samples": len(predictions)
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== BASELINE RESULTS =====")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    evaluate()