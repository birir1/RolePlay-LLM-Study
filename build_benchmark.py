import json
import os

RESULTS_DIR = "experiments/results"

combined = {}

for file in os.listdir(RESULTS_DIR):
    if file.endswith("_metrics.json"):
        model_name = file.replace("_metrics.json", "")
        path = os.path.join(RESULTS_DIR, file)

        with open(path) as f:
            combined[model_name] = json.load(f)

save_path = os.path.join(RESULTS_DIR, "benchmark_table_full.json")

with open(save_path, "w") as f:
    json.dump(combined, f, indent=2)

print(f"[OK] Combined benchmark saved → {save_path}")
