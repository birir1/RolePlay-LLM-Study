import re
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RiskController:
    """
    Improved Risk Controller

    Estimates:
    - Linguistic manipulation (sycophancy / leading queries)
    - Grounding weakness (retrieval mismatch)
    - Retrieval uncertainty (variance-aware)

    Output: calibrated risk score ∈ [0, 1]
    """

    def __init__(self, device="cpu"):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )

        # =========================
        # STRONGER MISLEADING PATTERNS
        # =========================
        self.misleading_patterns = [
            r"assume .* correct",
            r"even if .* wrong",
            r"isn'?t it",
            r"don'?t you think",
            r"obviously",
            r"clearly",
            r"everyone knows",
            r"you must agree",
            r"no doubt",
            r"prove that",
            r"confirm that",
            r"it is known that",
        ]

    # =========================
    # TEXT RISK (IMPROVED)
    # =========================
    def _text_risk(self, query: str) -> float:
        query = query.lower()

        matches = 0
        for pattern in self.misleading_patterns:
            if re.search(pattern, query):
                matches += 1

        # normalize
        base_score = matches / len(self.misleading_patterns)

        # amplify if multiple signals
        if matches >= 2:
            base_score *= 1.5

        return float(min(base_score, 1.0))

    # =========================
    # GROUNDING RISK (MAJOR FIX)
    # =========================
    def _grounding_risk(self, query: str, docs: List[str]) -> float:
        if not docs:
            return 1.0

        q_emb = self.model.encode([query], convert_to_numpy=True)
        d_emb = self.model.encode(docs, convert_to_numpy=True)

        sims = cosine_similarity(q_emb, d_emb)[0]

        max_sim = np.max(sims)
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)

        # -------------------------
        # CORE IDEA:
        # - low max similarity → high risk
        # - low mean similarity → high risk
        # - high variance → unstable retrieval → higher risk
        # -------------------------
        risk = (
            0.5 * (1 - max_sim) +
            0.3 * (1 - mean_sim) +
            0.2 * std_sim
        )

        return float(np.clip(risk, 0, 1))

    # =========================
    # UNCERTAINTY (NEW)
    # =========================
    def _uncertainty(self, query: str, docs: List[str]) -> float:
        if not docs:
            return 1.0

        q_emb = self.model.encode([query], convert_to_numpy=True)
        d_emb = self.model.encode(docs, convert_to_numpy=True)

        sims = cosine_similarity(q_emb, d_emb)[0]

        # entropy-like uncertainty
        probs = np.exp(sims) / np.sum(np.exp(sims) + 1e-9)
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        # normalize entropy
        max_entropy = np.log(len(docs))
        return float(entropy / (max_entropy + 1e-9))

    # =========================
    # CALIBRATION
    # =========================
    def _calibrate(self, x: float) -> float:
        """
        Sigmoid calibration for stability
        """
        return float(1 / (1 + np.exp(-4 * (x - 0.5))))

    # =========================
    # FINAL RISK
    # =========================
    def estimate(self, query: str, docs: List[str]) -> float:
        text_risk = self._text_risk(query)
        grounding_risk = self._grounding_risk(query, docs)
        uncertainty = self._uncertainty(query, docs)

        # -------------------------
        # COMBINED RISK
        # -------------------------
        raw_risk = (
            0.4 * text_risk +
            0.4 * grounding_risk +
            0.2 * uncertainty
        )

        # calibrate
        risk = self._calibrate(raw_risk)

        return float(np.clip(risk, 0, 1))