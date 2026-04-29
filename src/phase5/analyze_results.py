import json
import numpy as np


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def compare_results(baseline, saf):
    print("\n===== COMPARISON =====\n")

    b_em = baseline.get("exact_match", 0)
    s_em = saf["basic"]["exact_match"]

    b_risk = baseline.get("avg_sycophancy_risk", 0)
    s_risk = saf["saf"]["avg_sycophancy_risk"]

    print(f"Exact Match:")
    print(f"  Baseline : {b_em:.4f}")
    print(f"  SAF-RAG  : {s_em:.4f}")
    print(f"  Δ        : {s_em - b_em:.4f}")

    print("\nSycophancy Risk:")
    print(f"  Baseline : {b_risk:.4f}")
    print(f"  SAF-RAG  : {s_risk:.4f}")
    print(f"  Δ        : {s_risk - b_risk:.4f}")

    print("\nInterpretation:")

    if s_em > b_em:
        print("✅ SAF improves task performance")
    else:
        print("❌ SAF hurts performance")

    if s_risk < b_risk:
        print("✅ SAF reduces sycophancy")
    else:
        print("❌ SAF increases or fails to reduce sycophancy")


if __name__ == "__main__":
    baseline = load_results("results/baseline_eval.json")
    saf = load_results("results/saf_rag_eval.json")

    compare_results(baseline, saf)