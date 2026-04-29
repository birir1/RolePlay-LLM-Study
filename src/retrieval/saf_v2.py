import numpy as np
from typing import List, Dict


class SAFV2Scorer:
    """
    SAF v2: calibrated uncertainty + agreement modeling
    """

    # =====================================================
    # CALIBRATED UNCERTAINTY (Z-SCORE BASED)
    # =====================================================
    @staticmethod
    def calibrated_uncertainty(scores: np.ndarray) -> np.ndarray:

        mean = np.mean(scores)
        std = np.std(scores) + 1e-8

        z = (scores - mean) / std

        # squash into [0,1]
        return 1 / (1 + np.exp(z))

    # =====================================================
    # AGREEMENT BETWEEN RETRIEVERS
    # =====================================================
    @staticmethod
    def agreement_score(a: np.ndarray, b: np.ndarray) -> np.ndarray:

        # normalized difference
        diff = np.abs(a - b)

        return 1.0 - diff

    # =====================================================
    # ENTROPY (distribution instability)
    # =====================================================
    @staticmethod
    def entropy(scores: np.ndarray) -> float:

        p = scores + 1e-8
        p = p / p.sum()

        return float(-np.sum(p * np.log(p + 1e-8)))

    # =====================================================
    # FINAL SAF v2 RISK SCORE
    # =====================================================
    @staticmethod
    def compute_risk(
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray
    ) -> Dict:

        if len(dense_scores) == 0 or len(sparse_scores) == 0:
            return {
                "risk": [],
                "confidence": [],
                "entropy": 1.0
            }

        # align lengths safely
        n = min(len(dense_scores), len(sparse_scores))

        d = dense_scores[:n]
        s = sparse_scores[:n]

        # normalize
        d = SAFV2Scorer.calibrated_uncertainty(d)
        s = SAFV2Scorer.calibrated_uncertainty(s)

        agreement = SAFV2Scorer.agreement_score(d, s)

        entropy = SAFV2Scorer.entropy((d + s) / 2)

        # FINAL RISK
        risk = (0.5 * d + 0.5 * s) * (1.0 - agreement)

        confidence = 1.0 - risk

        return {
            "risk": risk.tolist(),
            "confidence": confidence.tolist(),
            "entropy": entropy
        }