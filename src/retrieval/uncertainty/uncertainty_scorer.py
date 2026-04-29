import numpy as np
from typing import List, Dict


class UncertaintyScorer:
    """
    Computes uncertainty signals for hybrid retrieval results.

    FIXES:
    - Safe key access (no crashes if fields missing)
    - Stable entropy computation
    - Non-destructive normalization
    - Handles partial / degraded retrieval inputs
    """

    def __init__(
        self,
        w_disagreement: float = 0.4,
        w_confidence: float = 0.3,
        w_entropy: float = 0.3
    ):
        self.w_disagreement = w_disagreement
        self.w_confidence = w_confidence
        self.w_entropy = w_entropy

    # =========================================================
    # MAIN ENTRY
    # =========================================================
    def compute(self, docs: List[Dict]) -> Dict:

        if not docs:
            return {
                "doc_uncertainty": [],
                "global_uncertainty": 1.0,
                "signals": {}
            }

        # 🔥 SAFE EXTRACTION (prevents 'dense_score' crash)
        dense = np.array(
            [float(d.get("dense_score", 0.0)) for d in docs],
            dtype=np.float32
        )

        sparse = np.array(
            [float(d.get("sparse_score", 0.0)) for d in docs],
            dtype=np.float32
        )

        hybrid = np.array(
            [float(d.get("hybrid_score", 0.0)) for d in docs],
            dtype=np.float32
        )

        # -----------------------------
        # 1. DISAGREEMENT SIGNAL
        # -----------------------------
        disagreement = np.abs(dense - sparse)

        # -----------------------------
        # 2. CONFIDENCE SIGNAL
        # -----------------------------
        confidence = hybrid.copy()

        # -----------------------------
        # 3. ENTROPY SIGNAL (FIXED)
        # -----------------------------
        entropy = self._compute_entropy(hybrid)

        # -----------------------------
        # Normalize signals (SAFE)
        # -----------------------------
        disagreement = self._normalize_safe(disagreement)
        confidence = self._normalize_safe(confidence)
        entropy = self._normalize_safe(entropy)

        # -----------------------------
        # Per-document uncertainty
        # -----------------------------
        doc_uncertainty = (
            self.w_disagreement * disagreement +
            self.w_entropy * entropy +
            self.w_confidence * (1.0 - confidence)
        )

        # -----------------------------
        # Global uncertainty
        # -----------------------------
        global_uncertainty = float(np.mean(doc_uncertainty))

        return {
            "doc_uncertainty": doc_uncertainty.tolist(),
            "global_uncertainty": global_uncertainty,
            "signals": {
                "disagreement": disagreement.tolist(),
                "confidence": confidence.tolist(),
                "entropy": entropy.tolist()
            }
        }

    # =========================================================
    # ENTROPY (FIXED — STABLE DISTRIBUTION)
    # =========================================================
    def _compute_entropy(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute entropy over normalized distribution,
        then broadcast it back per document for consistency.
        """

        if len(scores) == 0:
            return scores

        probs = scores + 1e-8
        probs = probs / np.sum(probs)

        # scalar entropy of distribution
        entropy_scalar = -np.sum(probs * np.log(probs + 1e-8))

        # broadcast to match doc-level shape
        return np.ones_like(scores) * entropy_scalar

    # =========================================================
    # SAFE NORMALIZATION (FIXED)
    # =========================================================
    def _normalize_safe(self, x: np.ndarray) -> np.ndarray:

        if len(x) == 0:
            return x

        min_x = np.min(x)
        max_x = np.max(x)

        # 🔥 CRITICAL FIX: avoid zero collapse
        if max_x - min_x < 1e-8:
            return np.ones_like(x) * 0.5

        return (x - min_x) / (max_x - min_x + 1e-8)