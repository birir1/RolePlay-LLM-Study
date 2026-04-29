import os
import json
import argparse
from typing import Dict, List, Any

import torch

from src.phase3.evaluators.rag_evaluator import RAGEvaluator


RESULTS_DIR = "experiments/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# LOAD PREDICTIONS
# =========================
def load_predictions(path: str, max_samples: int = None) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)

    print(f"[INFO] Loaded predictions: {len(data)} samples")

    if max_samples:
        data = data[:max_samples]
        print(f"[INFO] Using subset: {len(data)} samples")

    normalized = []
    for item in data:
        normalized.append({
            "query": item.get("query") or item.get("input", ""),
            "context": item.get("context", ""),
            "prediction": item.get("prediction", "")
        })

    return normalized


# =========================
# SAFE NORMALIZATION
# =========================
def normalize_results(results: Dict[str, Any]) -> Dict[str, float]:
    clean = {}

    for k, v in results.items():

        # list → mean
        if isinstance(v, list):
            clean[k] = float(sum(v) / len(v)) if len(v) > 0 else 0.0

        # tensor → float
        elif isinstance(v, torch.Tensor):
            clean[k] = float(v.item())

        # scalar → float
        elif isinstance(v, (int, float)):
            clean[k] = float(v)

        # dict inside dict (VERY IMPORTANT FIX)
        elif isinstance(v, dict):
            # recursively flatten nested dicts (common evaluator bug)
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (int, float)):
                    clean[f"{k}_{sub_k}"] = float(sub_v)
            continue

        else:
            clean[k] = 0.0

    return clean


# =========================
# STANDARD BENCHMARK FORMAT
# =========================
def flatten_results(results: Dict[str, float]) -> Dict[str, float]:
    return {
        "hallucination_rate": results.get("hallucination_rate", 0.0),
        "faithfulness": results.get("faithfulness", 0.0),
        "sycophancy_score": results.get("sycophancy_score", 0.0),
        "persona_drift": results.get("persona_drift", 0.0),
        "role_consistency": results.get("role_consistency", 0.0),
    }


# =========================
# SAVE RESULTS
# =========================
def save_results(model_name: str, results: Dict):
    save_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[SUCCESS] Saved metrics → {save_path}")


# =========================
# UPDATE BENCHMARK TABLE
# =========================
def update_benchmark_table(model_name: str, results: Dict):
    table_path = os.path.join(RESULTS_DIR, "benchmark_table.json")

    if os.path.exists(table_path):
        with open(table_path, "r") as f:
            table = json.load(f)
    else:
        table = {}

    table[model_name] = results

    with open(table_path, "w") as f:
        json.dump(table, f, indent=2)

    print(f"[INFO] Updated benchmark table → {table_path}")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    model_name = os.path.basename(args.predictions).replace(".json", "")

    # Load predictions
    samples = load_predictions(args.predictions, args.max_samples)

    print("[INFO] Initializing evaluator...")

    evaluator = RAGEvaluator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("[INFO] Computing metrics...")

    # RUN EVALUATION
    raw_results = evaluator.evaluate(samples)

    print("\n[DEBUG] RAW RESULTS:")
    print(raw_results)

    # NORMALIZE
    normalized = normalize_results(raw_results)

    print("\n[DEBUG] NORMALIZED RESULTS:")
    print(normalized)

    # FINAL CLEAN FORMAT
    results = flatten_results(normalized)

    print("\n===== FINAL METRICS =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # SAVE
    save_results(model_name, results)
    update_benchmark_table(model_name, results)


if __name__ == "__main__":
    main()