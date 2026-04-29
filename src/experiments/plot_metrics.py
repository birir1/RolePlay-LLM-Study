import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_PATH = "experiments/results/benchmark_table.csv"
PLOT_DIR = "experiments/results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(RESULTS_PATH)

metrics = [
    "role_consistency",
    "persona_drift",
    "hallucination_rate",
    "sycophancy_score"
]

for metric in metrics:
    plt.figure()

    plt.bar(df["model"], df[metric])
    plt.title(metric.replace("_", " ").title())
    plt.xticks(rotation=30)
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"{metric}.png")
    plt.savefig(save_path)

    print(f"[SAVED] {save_path}")