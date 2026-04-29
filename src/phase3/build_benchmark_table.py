import os
import json
import pandas as pd

RESULTS_DIR = "experiments/results"

METRICS_FILES = [
    "mt5_baseline_metrics.json",
    "transformer_baseline_metrics.json",
    "transformer_prompted_metrics.json"
]

SAF_FILE = "saf_rag_eval.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():

    print("[INFO] Loading model metrics...")

    table = {}

    # ----------------------------
    # LOAD BASE MODELS
    # ----------------------------
    for file in METRICS_FILES:
        path = os.path.join(RESULTS_DIR, file)

        if not os.path.exists(path):
            print(f"[WARNING] Missing {path}")
            continue

        model_name = file.replace("_metrics.json", "")
        table[model_name] = load_json(path)

    # ----------------------------
    # LOAD SAF (if exists)
    # ----------------------------
    saf_path = os.path.join(RESULTS_DIR, SAF_FILE)

    if os.path.exists(saf_path):
        saf_data = load_json(saf_path)

        # SAF structure: {"SAF": {...}, "BASE": {...}}
        if "SAF" in saf_data:
            table["saf"] = saf_data["SAF"]

        if "BASE" in saf_data:
            table["base"] = saf_data["BASE"]

    # ----------------------------
    # PRINT CLEAN TABLE
    # ----------------------------
    print("\n[INFO] FINAL CLEAN METRICS:\n")

    df = pd.DataFrame(table).T
    print(df)

    # ----------------------------
    # SAVE CLEAN OUTPUT
    # ----------------------------
    json_path = os.path.join(RESULTS_DIR, "benchmark_table_clean.json")
    csv_path = os.path.join(RESULTS_DIR, "benchmark_table_clean.csv")

    df.to_json(json_path, indent=2)
    df.to_csv(csv_path)

    print(f"\n[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()