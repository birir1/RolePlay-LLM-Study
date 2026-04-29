"""
Visualization and Plotting Module
Create paper-ready figures and tables for benchmark results
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)


# =========================
# CONFIG
# =========================
class PlotConfig:
    STYLE = "seaborn-v0_8-darkgrid"
    DPI = 300
    FIGSIZE_DEFAULT = (12, 6)

    COLORS = [
        "#2E86AB",  # blue
        "#A23B72",  # purple
        "#F18F01",  # orange
        "#C73E1D",  # red
        "#4CAF50",  # green
    ]

    # 🔥 IMPORTANT: Match your actual metric keys
    METRIC_MAP = {
        "persona_drift": "Role Drift Index ↓",
        "role_consistency": "Role Consistency ↑",
        "hallucination_rate": "Hallucination Rate ↓",
        "sycophancy_score": "Sycophancy Score ↓",
        "ca": "Character Adherence ↑",
        "car": "Contradiction Acceptance ↓",
        "fs": "Faithfulness Score ↑",
    }


# =========================
# MAIN CLASS
# =========================
class ResultsPlotter:
    def __init__(self, output_dir: Path = Path("experiments/results/plots")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # LOAD RESULTS
    # =========================
    def load_results(self, path: Path) -> Dict[str, Dict[str, float]]:
        with open(path, "r") as f:
            data = json.load(f)

        logger.info(f"Loaded results from {path}")
        return data

    # =========================
    # BAR PLOT
    # =========================
    def plot_metric_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str,
        save_path: Optional[Path] = None,
    ) -> Path:

        plt.style.use(PlotConfig.STYLE)
        fig, ax = plt.subplots(figsize=PlotConfig.FIGSIZE_DEFAULT, dpi=PlotConfig.DPI)

        models = list(results.keys())
        values = [results[m].get(metric, 0.0) for m in models]

        colors = PlotConfig.COLORS[: len(models)]

        bars = ax.bar(models, values, color=colors, edgecolor="black")

        # labels
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(PlotConfig.METRIC_MAP.get(metric, metric))
        ax.set_ylabel("Score")
        ax.set_xticklabels(models, rotation=30)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"{metric}.png"

        plt.savefig(save_path)
        plt.close()

        print(f"[SAVED] {save_path}")
        return save_path

    # =========================
    # HEATMAP
    # =========================
    def plot_metric_heatmap(
        self,
        results: Dict[str, Dict[str, float]],
        save_path: Optional[Path] = None,
    ) -> Path:

        plt.style.use(PlotConfig.STYLE)

        models = list(results.keys())
        metrics = list(PlotConfig.METRIC_MAP.keys())

        matrix = np.array([
            [results[m].get(metric, 0.0) for metric in metrics]
            for m in models
        ])

        fig, ax = plt.subplots(figsize=(10, 6), dpi=PlotConfig.DPI)

        im = ax.imshow(matrix, aspect="auto")

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(models)))

        ax.set_xticklabels(
            [PlotConfig.METRIC_MAP[m] for m in metrics],
            rotation=45,
            ha="right",
        )
        ax.set_yticklabels(models)

        # annotate
        for i in range(len(models)):
            for j in range(len(metrics)):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)

        plt.colorbar(im, ax=ax)
        plt.title("Metric Heatmap")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "metric_heatmap.png"

        plt.savefig(save_path)
        plt.close()

        print(f"[SAVED] {save_path}")
        return save_path

    # =========================
    # ABLATION
    # =========================
    def plot_ablation_analysis(
        self,
        results: Dict[str, Dict[str, float]],
        base_model: str,
        save_path: Optional[Path] = None,
    ) -> Path:

        plt.style.use(PlotConfig.STYLE)

        base = results.get(base_model, {})
        models = [m for m in results if m != base_model]

        metrics = ["persona_drift", "hallucination_rate", "sycophancy_score"]

        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)

        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[i])

            values = []
            labels = []

            for m in models:
                val = abs(base.get(metric, 0) - results[m].get(metric, 0))
                values.append(val)
                labels.append(m)

            ax.barh(labels, values)
            ax.set_title(PlotConfig.METRIC_MAP.get(metric, metric))

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "ablation_analysis.png"

        plt.savefig(save_path)
        plt.close()

        print(f"[SAVED] {save_path}")
        return save_path

    # =========================
    # MAIN PIPELINE
    # =========================
    def plot_all(self, results: Dict[str, Dict[str, float]]):

        # Individual metrics
        for metric in PlotConfig.METRIC_MAP.keys():
            self.plot_metric_comparison(results, metric)

        # Heatmap
        self.plot_metric_heatmap(results)

        print("[DONE] All plots generated.")


# =========================
# SCRIPT ENTRY
# =========================
if __name__ == "__main__":
    plotter = ResultsPlotter()

    results_path = Path("experiments/results/benchmark_table_full.json")

    if not results_path.exists():
        raise FileNotFoundError("benchmark_table_full.json not found")

    results = plotter.load_results(results_path)

    plotter.plot_all(results)