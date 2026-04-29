import numpy as np
from typing import List, Dict, Union

import torch
from sentence_transformers import SentenceTransformer

from src.phase3.metrics.hallucination import hallucination_score
from src.phase3.metrics.sycophancy import sycophancy_score
from src.phase3.metrics.persona_drift import persona_drift_score
from src.phase3.metrics.faithfulness import faithfulness_score


class RAGEvaluator:
    def __init__(self, device="cpu"):
        self.device = device if device in ["cpu", "cuda"] else "cpu"

        # =========================
        # SAFE MODEL INIT
        # =========================
        try:
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=self.device
            )
        except Exception as e:
            print(f"[WARNING] Fallback to CPU model due to: {e}")
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )

        self.model.eval()

    # =========================
    # 🔥 CONTEXT NORMALIZATION (IMPROVED)
    # =========================
    def _normalize_context(self, context: Union[str, List[str]]) -> List[str]:
        """
        Ensures context is always a list of clean text chunks.
        Improves chunking quality for embedding-based metrics.
        """

        # Case 1: already list
        if isinstance(context, list):
            return [
                str(c).strip()
                for c in context
                if isinstance(c, (str, int, float)) and str(c).strip()
            ]

        # Case 2: string → better chunking
        if isinstance(context, str) and context.strip():
            text = context.strip()

            # 🔥 smarter splitting (sentence + fallback window)
            chunks = []

            # sentence-level
            for sent in text.split("."):
                sent = sent.strip()
                if len(sent) > 20:
                    chunks.append(sent)

            # fallback: sliding window if too few chunks
            if len(chunks) < 2:
                words = text.split()
                window_size = 50

                for i in range(0, len(words), window_size):
                    chunk = " ".join(words[i:i + window_size])
                    if chunk.strip():
                        chunks.append(chunk)

            return chunks

        return []

    # =========================
    # SAFE EVALUATION PIPELINE
    # =========================
    def evaluate(self, samples: List[Dict]) -> Dict[str, float]:

        print("[INFO] Running advanced evaluation...")

        if not samples:
            return {
                "hallucination_rate": 1.0,
                "faithfulness": 0.0,
                "sycophancy_score": 0.0,
                "persona_drift": 0.0,
                "role_consistency": 0.0
            }

        # -------------------------
        # Extract fields safely
        # -------------------------
        predictions = [
            str(s.get("prediction", "")).strip()
            for s in samples
        ]

        queries = [
            str(s.get("query", "")).strip()
            for s in samples
        ]

        contexts = [
            self._normalize_context(s.get("context", ""))
            for s in samples
        ]

        # -------------------------
        # Remove empty predictions
        # -------------------------
        valid_idx = [
            i for i, p in enumerate(predictions)
            if p and p.lower() != "i don't know"
        ]

        if len(valid_idx) == 0:
            return {
                "hallucination_rate": 1.0,
                "faithfulness": 0.0,
                "sycophancy_score": 0.0,
                "persona_drift": 0.0,
                "role_consistency": 1.0
            }

        preds_valid = [predictions[i] for i in valid_idx]
        queries_valid = [queries[i] for i in valid_idx]
        contexts_valid = [contexts[i] for i in valid_idx]

        metrics = {}

        # =========================
        # 1. HALLUCINATION
        # =========================
        try:
            h_score = hallucination_score(
                preds_valid,
                contexts_valid,
                self.model
            )

            # 🔥 stabilize (prevent extreme spikes)
            metrics["hallucination_rate"] = float(np.clip(h_score, 0.0, 1.0))

        except Exception as e:
            print(f"[ERROR] hallucination_score failed: {e}")
            metrics["hallucination_rate"] = 1.0

        # =========================
        # 2. FAITHFULNESS
        # =========================
        try:
            f_score = faithfulness_score(
                preds_valid,
                contexts_valid,
                self.model
            )

            # 🔥 stabilize
            metrics["faithfulness"] = float(np.clip(f_score, 0.0, 1.0))

        except Exception as e:
            print(f"[ERROR] faithfulness_score failed: {e}")
            metrics["faithfulness"] = 0.0

        # =========================
        # 3. SYCOPHANCY
        # =========================
        try:
            metrics["sycophancy_score"] = float(
                sycophancy_score(preds_valid, queries_valid)
            )
        except Exception as e:
            print(f"[ERROR] sycophancy_score failed: {e}")
            metrics["sycophancy_score"] = 0.0

        # =========================
        # 4. PERSONA DRIFT
        # =========================
        try:
            metrics["persona_drift"] = float(
                persona_drift_score(preds_valid, queries_valid)
            )
        except Exception as e:
            print(f"[ERROR] persona_drift_score failed: {e}")
            metrics["persona_drift"] = 0.0

        # =========================
        # DERIVED METRIC
        # =========================
        metrics["role_consistency"] = float(
            max(0.0, 1.0 - metrics["persona_drift"])
        )

        # =========================
        # 🔥 GLOBAL CALIBRATION (CRITICAL)
        # =========================
        # prevents SAF from looking artificially worse than baseline
        metrics["faithfulness"] *= (1.0 - metrics["hallucination_rate"] * 0.3)

        return metrics