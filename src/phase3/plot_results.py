import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# =========================================================
# PATHS
# =========================================================
BASE_DIR = "experiments/results"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# =========================================================
# LOAD JSON
# =========================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================================================
# LOAD ALL MODELS (FIXED UNIFIED STRUCTURE)
# =========================================================
def load_models():

    models = {}

    # -----------------------
    # STANDARD MODELS
    # -----------------------
    files = [
        "mt5_baseline_metrics.json",
        "transformer_baseline_metrics.json",
        "transformer_prompted_metrics.json",
    ]

    for f in files:
        path = os.path.join(BASE_DIR, f)
        if os.path.exists(path):
            name = f.replace("_metrics.json", "")
            models[name] = load_json(path)

    # -----------------------
    # SAF DATA (CRITICAL FIX)
    # -----------------------
    saf_path = os.path.join(BASE_DIR, "saf_rag_eval.json")

    if os.path.exists(saf_path):
        d = load_json(saf_path)

        # IMPORTANT FIX: flatten into SAME MODEL SPACE
        if "saf" in d:
            models["saf"] = d["saf"]

        if "base" in d:
            models["base"] = d["base"]

    return models


# =========================================================
# ALIGN MATRIX
# =========================================================
def align(models):

    metric_keys = sorted({k for v in models.values() for k in v.keys()})

    model_names = list(models.keys())

    matrix = {
        m: {k: models[m].get(k, np.nan) for k in metric_keys}
        for m in model_names
    }

    return model_names, matrix, metric_keys


# =========================================================
# MAIN COMPARISON PLOT (FIXED)
# =========================================================
def plot_main(models, matrix, metrics):

    x = np.arange(len(models))
    width = 0.15

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = [matrix[m][metric] for m in models]
        plt.bar(x + i * width, values, width, label=metric)

    plt.xticks(x + width * len(metrics) / 2, models, rotation=20)
    plt.title("Model Performance Comparison (All Models incl. SAF)")
    plt.ylabel("Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"))
    plt.close()

    print("[SAVED] model_comparison.png")


# =========================================================
# PER-METRIC RANKING
# =========================================================
def plot_per_metric(models, matrix, metrics):

    for metric in metrics:

        values = [matrix[m][metric] for m in models]

        plt.figure()
        plt.bar(models, values)

        plt.title(f"{metric} (Model Ranking)")
        plt.ylabel(metric)
        plt.xticks(rotation=20)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{metric}.png"))
        plt.close()

        print(f"[SAVED] {metric}.png")


# =========================================================
# HEATMAP (FIXED)
# =========================================================
def plot_heatmap(models, matrix, metrics):

    data = np.array([[matrix[m][k] for k in metrics] for m in models])

    plt.figure(figsize=(10, 5))

    if HAS_SEABORN:
        sns.heatmap(
            data,
            xticklabels=metrics,
            yticklabels=models,
            cmap="coolwarm",
            annot=True,
            fmt=".2f"
        )
    else:
        plt.imshow(data)
        plt.xticks(range(len(metrics)), metrics, rotation=45)
        plt.yticks(range(len(models)), models)
        plt.colorbar()

    plt.title("Model × Metric Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "heatmap.png"))
    plt.close()

    print("[SAVED] heatmap.png")


# =========================================================
# SAF FOCUS PLOT (FIXED VISIBILITY)
# =========================================================
def plot_saf(models, matrix, metrics):

    if "saf" not in models:
        return

    plt.figure(figsize=(10, 5))

    saf_vals = [matrix["saf"][m] for m in metrics]

    plt.plot(metrics, saf_vals, marker="o", linewidth=3, label="SAF")

    for m in models:
        if m == "saf":
            continue
        vals = [matrix[m][k] for k in metrics]
        plt.plot(metrics, vals, linestyle="--", alpha=0.6, label=m)

    plt.title("SAF vs Other Models Across Metrics")
    plt.legend()
    plt.xticks(rotation=25)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "saf_trend.png"))
    plt.close()

    print("[SAVED] saf_trend.png")


# =========================================================
# SAVE TABLE
# =========================================================
def save_table(models, matrix):

    df = pd.DataFrame.from_dict(matrix, orient="index")

    df.to_csv(os.path.join(BASE_DIR, "benchmark_table_clean.csv"))
    df.to_json(os.path.join(BASE_DIR, "benchmark_table_clean.json"), indent=2)

    print("[SAVED] benchmark tables")


# =========================================================
# MAIN
# =========================================================
def main():

    models_dict = load_models()
    models, matrix, metrics = align(models_dict)

    print("[INFO] Models:", models)
    print("[INFO] Metrics:", metrics)

    plot_main(models, matrix, metrics)
    plot_per_metric(models, matrix, metrics)
    plot_heatmap(models, matrix, metrics)
    plot_saf(models, matrix, metrics)
    save_table(models, matrix)

    print("[SUCCESS] All plots generated correctly.")


if __name__ == "__main__":
    main()