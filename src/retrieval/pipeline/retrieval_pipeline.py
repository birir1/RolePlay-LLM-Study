import logging
from typing import Dict, List, Any
import numpy as np

from ..retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RetrievalPipeline:

    def __init__(self, retriever: HybridRetriever, uncertainty_threshold=0.6, top_k=5):
        self.retriever = retriever
        self.uncertainty_threshold = uncertainty_threshold
        self.top_k = top_k

    # =========================
    # NORMALIZATION (FIXED)
    # =========================
    def _normalize_docs(self, docs: List[Dict]) -> List[Dict]:

        normalized = []

        for d in docs:
            if isinstance(d, str):
                d = {"text": d}

            # SAFE extraction with fallbacks across possible upstream schemas
            text = d.get("text", "")

            dense = (
                d.get("dense_score")
                or d.get("score_dense")
                or 0.0
            )

            sparse = (
                d.get("sparse_score")
                or d.get("score_sparse")
                or 0.0
            )

            hybrid = (
                d.get("hybrid_score")
                or d.get("score")
                or (float(dense) * 0.6 + float(sparse) * 0.4)
            )

            uncertainty = d.get("uncertainty", 0.5)

            normalized.append({
                "text": text,
                "dense_score": float(dense),
                "sparse_score": float(sparse),
                "hybrid_score": float(hybrid),
                "uncertainty": float(uncertainty)
            })

        return normalized

    # =========================
    # GLOBAL UNCERTAINTY
    # =========================
    def _compute_global_uncertainty(self, docs: List[Dict]) -> float:

        if not docs:
            return 1.0

        scores = np.array([
            d.get("hybrid_score", 0.0)
            for d in docs
        ])

        confidence = float(np.mean(scores))
        dispersion = float(np.std(scores))

        uncertainty = (1 - confidence) * 0.7 + dispersion * 0.3

        return float(np.clip(uncertainty, 0.0, 1.0))

    # =========================
    # RANKING (FIXED SAFETY)
    # =========================
    def _rank_documents(self, docs: List[Dict]) -> List[Dict]:

        docs = sorted(
            docs,
            key=lambda x: x.get("hybrid_score", 0.0),
            reverse=True
        )

        seen = set()
        selected = []

        for d in docs:
            key = d.get("text", "")[:120]

            if key in seen:
                continue

            selected.append(d)
            seen.add(key)

            if len(selected) >= self.top_k * 2:
                break

        return selected

    # =========================
    # MAIN PIPELINE
    # =========================
    def run(self, query: str) -> Dict[str, Any]:

        logger.info(f"[Pipeline] {query}")

        output = self.retriever.retrieve(query=query, top_k=self.top_k * 3)

        docs = self._normalize_docs(output.get("documents", []))

        if not docs:
            return {
                "query": query,
                "documents": [],
                "uncertainty": {"global_uncertainty": 1.0},
                "abstain": True
            }

        docs = self._rank_documents(docs)

        unc = self._compute_global_uncertainty(docs)

        filtered = [
            d for d in docs
            if d.get("uncertainty", 0.5) <= self.uncertainty_threshold
        ]

        if len(filtered) < 2:
            filtered = docs[:2]

        filtered = filtered[:self.top_k]

        abstain = unc > (self.uncertainty_threshold + 0.15)

        return {
            "query": query,
            "documents": filtered,
            "uncertainty": {"global_uncertainty": unc},
            "abstain": abstain
        }