import os
import json
import pandas as pd

RESULTS_DIR = "experiments/results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "benchmark_table.csv")

rows = []

for file in os.listdir(RESULTS_DIR):
    if file.endswith("_metrics.json"):
        path = os.path.join(RESULTS_DIR, file)

        with open(path, "r") as f:
            data = json.load(f)

        model_name = file.replace("_metrics.json", "")
        data["model"] = model_name

        rows.append(data)

df = pd.DataFrame(rows)

# Order columns
df = df[[
    "model",
    "role_consistency",
    "persona_drift",
    "hallucination_rate",
    "sycophancy_score"
]]

df.to_csv(OUTPUT_PATH, index=False)

print(df)
print(f"\n[SUCCESS] Table saved → {OUTPUT_PATH}")