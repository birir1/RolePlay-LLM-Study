# /src/retrieval/pipeline/saf_pipeline.py

import numpy as np
from typing import List, Dict, Any

from src.retrieval.dense.dense_retriever import DenseRetriever
from src.retrieval.sparse.sparse_retriever import SparseRetriever
from src.retrieval.hybrid.hybrid_merger import HybridMerger
from src.retrieval.filtering.adaptive_filter import AdaptiveFilter


class SAFRetrievalPipeline:

    def __init__(
        self,
        corpus: List[str],
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        fusion_alpha: float = 0.6
    ):

        self.corpus = [c.strip() for c in corpus if isinstance(c, str) and c.strip()]

        if not self.corpus:
            raise ValueError("Empty corpus provided")

        self.dense = DenseRetriever(
            corpus=self.corpus,
            model_name=dense_model,
            device=device
        )

        self.sparse = SparseRetriever(documents=self.corpus)

        self.merger = HybridMerger(alpha=fusion_alpha)

        self.filter = AdaptiveFilter(
            uncertainty_threshold=0.7,
            min_docs=2,
            max_docs=5
        )

        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse

    # =====================================================
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores

        min_s, max_s = np.min(scores), np.max(scores)

        if max_s - min_s < 1e-8:
            return np.ones_like(scores)

        return (scores - min_s) / (max_s - min_s + 1e-8)

    # =====================================================
    def _compute_uncertainty(self, docs: List[Dict]) -> Dict[str, Any]:

        if not docs:
            return {"doc_uncertainty": [], "global_uncertainty": 1.0}

        dense = np.array([d.get("dense_score", 0.0) for d in docs])
        sparse = np.array([d.get("sparse_score", 0.0) for d in docs])
        hybrid = np.array([d.get("hybrid_score", 0.0) for d in docs])

        dense = self._normalize_scores(dense)
        sparse = self._normalize_scores(sparse)
        hybrid = self._normalize_scores(hybrid)

        if len(hybrid) == 0:
            return {"doc_uncertainty": [], "global_uncertainty": 1.0}

        top_score = np.max(hybrid)
        mean_score = np.mean(hybrid)

        confidence = top_score - mean_score

        agreement = 1.0 - np.abs(dense - sparse)

        probs = hybrid / (np.sum(hybrid) + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        entropy_norm = entropy / np.log(len(probs) + 1e-8)

        doc_uncertainty = (
            0.5 * (1 - hybrid) +
            0.3 * (1 - agreement) +
            0.2 * entropy_norm
        )

        return {
            "doc_uncertainty": doc_uncertainty.tolist(),
            "global_uncertainty": float(np.mean(doc_uncertainty))
        }

    # =====================================================
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:

        dense_results = self.dense.retrieve(query, top_k=self.top_k_dense) or []
        sparse_results = self.sparse.retrieve(query, top_k=self.top_k_sparse) or []

        dense_pairs, sparse_pairs = [], []

        # SAFE dense parsing
        for r in dense_results:
            if isinstance(r, dict):
                dense_pairs.append((
                    r.get("text", ""),
                    float(r.get("dense_score", 0.0))
                ))

        # SAFE sparse parsing
        for r in sparse_results:
            if isinstance(r, dict):
                sparse_pairs.append((
                    r.get("text", ""),
                    float(r.get("bm25_score", 0.0))
                ))
            elif isinstance(r, (tuple, list)) and len(r) >= 2:
                sparse_pairs.append((r[0], float(r[1])))

        merged = self.merger.merge(
            dense_pairs,
            sparse_pairs,
            top_k=max(self.top_k_dense, self.top_k_sparse)
        ) or []

        normalized = []

        for d in merged:
            if isinstance(d, dict):
                normalized.append({
                    "text": d.get("text", ""),
                    "dense_score": float(d.get("dense_score", 0.0)),
                    "sparse_score": float(d.get("sparse_score", 0.0)),
                    "hybrid_score": float(d.get("hybrid_score", 0.0))
                })

        if not normalized:
            return {
                "query": query,
                "documents": [{"text": ""}],
                "uncertainty": {"global_uncertainty": 1.0},
                "abstain": True
            }

        uncertainty = self._compute_uncertainty(normalized)

        filtered = self.filter.filter(
            docs=normalized,
            uncertainty_output=uncertainty
        )

        docs = filtered.get("filtered_docs", []) or normalized[:1]

        return {
            "query": query,
            "documents": docs[:top_k],
            "uncertainty": uncertainty,
            "abstain": filtered.get("abstain", False)
        }