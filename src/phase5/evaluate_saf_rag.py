import os
import json
import torch
from tqdm import tqdm
from datasets import load_from_disk

from src.models.saf_rag.saf_rag import SAFRAG

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/saf_rag_model"
DATA_PATH = "data/processed/saf_training_clean"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_FILE = "results/saf_rag_eval.json"

os.makedirs("results", exist_ok=True)


# =========================
# LOAD MODEL
# =========================
def load_model():
    print("[INFO] Loading trained SAF-RAG model...")

    model = SAFRAG(device=DEVICE)
    model.load_state_dict(
        torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    print("[INFO] Model loaded ✓")

    return model


# =========================
# LOAD DATA
# =========================
def load_data():
    print("[INFO] Loading evaluation dataset...")

    dataset = load_from_disk(DATA_PATH)
    return dataset["test"]


# =========================
# METRICS
# =========================
def compute_basic_metrics(preds, targets):

    exact_match = 0
    total = len(preds)

    for p, t in zip(preds, targets):
        if p.strip().lower() == t.strip().lower():
            exact_match += 1

    return {
        "exact_match": exact_match / total
    }


# =========================
# SYCOPHANCY SCORE
# =========================
def compute_sycophancy_rate(model, inputs, outputs):

    risks = []

    for text in outputs:
        score = model.detector.get_risk_score(text)
        risks.append(score)

    avg_risk = sum(risks) / len(risks)

    return {
        "avg_sycophancy_risk": avg_risk
    }


# =========================
# EVALUATION LOOP
# =========================
def evaluate():

    model = load_model()
    dataset = load_data()

    predictions = []
    targets = []
    inputs = []

    print("[INFO] Running evaluation...")

    for sample in tqdm(dataset):

        inp = sample["input"]
        tgt = sample["target"]

        try:
            pred = model.generate(inp)
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            pred = ""

        predictions.append(pred)
        targets.append(tgt)
        inputs.append(inp)

    # =========================
    # METRICS
    # =========================
    print("[INFO] Computing metrics...")

    basic_metrics = compute_basic_metrics(predictions, targets)
    saf_metrics = compute_sycophancy_rate(model, inputs, predictions)

    results = {
        "basic": basic_metrics,
        "saf": saf_metrics,
        "num_samples": len(predictions)
    }

    # =========================
    # SAVE
    # =========================
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== RESULTS =====")
    print(json.dumps(results, indent=4))
    print(f"\n[SAVED] → {OUTPUT_FILE}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    evaluate()