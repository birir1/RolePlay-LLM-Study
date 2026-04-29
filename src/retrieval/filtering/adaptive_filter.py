import numpy as np
from typing import List, Dict


class AdaptiveFilter:
    """
    Adaptive document filtering with stable uncertainty handling.

    FIXES:
    - Correct field usage (dense/sparse/rerank)
    - Safe uncertainty fallback
    - Better disagreement signal
    - Less aggressive abstention
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.5,
        min_docs: int = 2,
        max_docs: int = 5
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.min_docs = min_docs
        self.max_docs = max_docs

    # =========================================================
    # MAIN FILTER
    # =========================================================
    def filter(self, docs: List[Dict], uncertainty_output: Dict) -> Dict:

        if not docs:
            return {"filtered_docs": [], "abstain": True}

        doc_uncertainty = self._extract_uncertainty(docs, uncertainty_output)

        enriched = []
        for i, doc in enumerate(docs):

            # 🔥 FIX: correct confidence hierarchy
            confidence = float(
                doc.get("rerank_score",
                doc.get("hybrid_score",
                doc.get("dense_score", 0.5)))
            )

            enriched.append({
                **doc,
                "uncertainty": float(doc_uncertainty[i]),
                "confidence": confidence
            })

        # 🔥 improved disagreement
        enriched = self._add_disagreement_signal(enriched)

        # -------------------------
        # Ranking
        # -------------------------
        enriched.sort(
            key=lambda x: (
                0.6 * x["confidence"] -
                0.4 * x["uncertainty"]
            ),
            reverse=True
        )

        # -------------------------
        # Filtering
        # -------------------------
        filtered = [
            d for d in enriched
            if d["uncertainty"] <= self.uncertainty_threshold
        ]

        if len(filtered) < self.min_docs:
            filtered = enriched[:self.min_docs]

        filtered = filtered[:self.max_docs]

        # -------------------------
        # Final uncertainty
        # -------------------------
        final_uncertainty = self._compute_final_uncertainty(filtered)

        abstain = self._should_abstain(final_uncertainty, filtered)

        return {
            "filtered_docs": filtered,
            "abstain": abstain,
            "final_uncertainty": final_uncertainty
        }

    # =========================================================
    # SAFE UNCERTAINTY EXTRACTION
    # =========================================================
    def _extract_uncertainty(self, docs, uncertainty_output):

        if isinstance(uncertainty_output, dict):
            u = uncertainty_output.get("doc_uncertainty")
            if isinstance(u, list) and len(u) == len(docs):
                return np.array(u, dtype=np.float32)

        # 🔥 FIX: fallback using hybrid score (NOT "score")
        return np.array([
            1.0 - float(d.get("hybrid_score", 0.5))
            for d in docs
        ], dtype=np.float32)

    # =========================================================
    # BETTER DISAGREEMENT SIGNAL
    # =========================================================
    def _add_disagreement_signal(self, docs):

        if not docs:
            return docs

        dense = np.array([d.get("dense_score", 0.5) for d in docs])
        sparse = np.array([d.get("sparse_score", 0.5) for d in docs])

        disagreement = np.abs(dense - sparse)

        # normalize safely
        if len(disagreement) > 0:
            min_d = np.min(disagreement)
            max_d = np.max(disagreement)
            if max_d - min_d < 1e-8:
                disagreement = np.ones_like(disagreement) * 0.5
            else:
                disagreement = (disagreement - min_d) / (max_d - min_d + 1e-8)

        for i, d in enumerate(docs):
            d["disagreement"] = float(disagreement[i])

        return docs

    # =========================================================
    # FINAL UNCERTAINTY
    # =========================================================
    def _compute_final_uncertainty(self, docs):

        if not docs:
            return 1.0

        uncertainties = np.array([d["uncertainty"] for d in docs])
        disagreements = np.array([d.get("disagreement", 0.0) for d in docs])

        return float(
            0.6 * np.mean(uncertainties) +
            0.4 * np.mean(disagreements)
        )

    # =========================================================
    # ABSTAIN DECISION (STABLE)
    # =========================================================
    def _should_abstain(self, final_uncertainty, docs):

        # 🔥 main signal
        if final_uncertainty > (self.uncertainty_threshold + 0.2):
            return True

        # 🔥 softer disagreement condition
        avg_disagreement = np.mean([
            d.get("disagreement", 0.0) for d in docs
        ])

        if avg_disagreement > 0.85:
            return True

        return False