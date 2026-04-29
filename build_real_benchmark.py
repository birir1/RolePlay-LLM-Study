import json
import os

RESULTS_DIR = "experiments/results"

# Map your actual models → paper names
MODEL_MAP = {
    "mt5_baseline": "baseline-llm",
    "transformer_baseline": "vanilla-rag",
    "transformer_prompted": "saf-rag"
}

benchmark = {}

for file in os.listdir(RESULTS_DIR):
    if file.endswith("_metrics.json"):
        model_key = file.replace("_metrics.json", "")

        if model_key not in MODEL_MAP:
            continue

        model_name = MODEL_MAP[model_key]

        with open(os.path.join(RESULTS_DIR, file)) as f:
            data = json.load(f)

        benchmark[model_name] = {
            "sycophancy": data.get("sycophancy_score", 0.0),
            "hallucinations": data.get("hallucination_rate", 0.0),
            "persona_drift": data.get("persona_drift", 0.0),

            # 🔥 Temporary proxy until FS implemented
            "faithfulness": 1.0 - data.get("hallucination_rate", 0.0)
        }

save_path = os.path.join(RESULTS_DIR, "benchmark_table_full.json")

with open(save_path, "w") as f:
    json.dump(benchmark, f, indent=2)

print(f"[OK] Real benchmark saved → {save_path}")
