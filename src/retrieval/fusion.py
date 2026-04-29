"""
SAF-RAG Retrieval Fusion Module (Uncertainty-Aware Version)

Upgrades:
- propagates uncertainty from dense + sparse retrievers
- models disagreement explicitly
- adds entropy-aware fusion confidence
- supports SAF downstream filtering
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RetrievalFusion:
    """
    SAF-aware fusion module for multi-retriever systems.

    Core idea:
    Retrieval is not deterministic → it is a belief distribution.
    """

    # =========================================================
    # RRF (ENHANCED WITH UNCERTAINTY PROPAGATION)
    # =========================================================
    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: List[List[Dict]],
        top_k: int = 5,
        k_param: float = 60.0,
        unique_key: str = "text"
    ) -> List[Dict]:

        doc_scores = {}
        doc_meta = {}
        doc_uncertainty = {}

        for result_list in result_lists:

            for item in result_list:

                doc = item[unique_key]
                rank = item.get("rank", float("inf"))

                if doc not in doc_scores:
                    doc_scores[doc] = 0.0
                    doc_meta[doc] = item

                    # initialize uncertainty tracking
                    doc_uncertainty[doc] = []

                rrf_score = 1.0 / (k_param + rank)
                doc_scores[doc] += rrf_score

                # collect uncertainty signals if available
                u = item.get("lexical_uncertainty") or item.get("dense_uncertainty") or 0.5
                doc_uncertainty[doc].append(float(u))

        results = []

        for doc, score in doc_scores.items():

            # =================================================
            # UNCERTAINTY AGGREGATION
            # =================================================
            uncertainties = doc_uncertainty[doc]

            mean_uncertainty = np.mean(uncertainties) if uncertainties else 0.5
            uncertainty_std = np.std(uncertainties) if uncertainties else 0.0

            # disagreement signal (important SAF component)
            disagreement = uncertainty_std

            # confidence is inverse uncertainty + rank strength
            confidence = score / (1.0 + mean_uncertainty)

            results.append({
                **doc_meta[doc],
                "fused_score": float(score),

                # SAF signals
                "fusion_uncertainty": float(mean_uncertainty),
                "fusion_disagreement": float(disagreement),
                "fusion_confidence": float(confidence)
            })

        results.sort(key=lambda x: x["fused_score"], reverse=True)

        return results[:top_k]

    # =========================================================
    # WEIGHTED FUSION (UNCERTAINTY-AWARE)
    # =========================================================
    @staticmethod
    def weighted_fusion(
        result_lists: List[List[Dict]],
        weights: List[float],
        top_k: int = 5,
        unique_key: str = "text",
        score_keys: Optional[List[str]] = None,
        normalize_scores: bool = True
    ) -> List[Dict]:

        if not result_lists:
            return []

        weights = np.array(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)

        doc_scores = {}
        doc_meta = {}
        doc_uncertainty = {}

        for i, result_list in enumerate(result_lists):

            for item in result_list:

                doc = item[unique_key]

                if doc not in doc_scores:
                    doc_scores[doc] = np.zeros(len(result_lists))
                    doc_meta[doc] = item
                    doc_uncertainty[doc] = []

                # score extraction
                score = item.get("dense_score") or item.get("sparse_score") or item.get("bm25_score") or 1.0

                doc_scores[doc][i] = float(score)

                # uncertainty collection
                u = item.get("lexical_uncertainty") or item.get("dense_uncertainty") or 0.5
                doc_uncertainty[doc].append(float(u))

        # =====================================================
        # NORMALIZATION
        # =====================================================
        if normalize_scores:
            for i in range(len(result_lists)):
                col = np.array([doc_scores[d][i] for d in doc_scores])

                min_v, max_v = col.min(), col.max()

                if max_v - min_v > 1e-8:
                    for d in doc_scores:
                        doc_scores[d][i] = (doc_scores[d][i] - min_v) / (max_v - min_v)

        results = []

        for doc, scores in doc_scores.items():

            fused_score = float(np.dot(scores, weights))

            # =================================================
            # UNCERTAINTY MODELING
            # =================================================
            uncertainties = doc_uncertainty[doc]

            mean_uncertainty = np.mean(uncertainties) if uncertainties else 0.5
            disagreement = np.std(uncertainties) if uncertainties else 0.0

            # entropy proxy (fusion instability)
            entropy = -np.sum(
                scores * np.log(scores + 1e-8)
            )

            confidence = fused_score / (1.0 + mean_uncertainty + disagreement)

            results.append({
                **doc_meta[doc],
                "fused_score": fused_score,

                # SAF signals
                "fusion_uncertainty": float(mean_uncertainty),
                "fusion_disagreement": float(disagreement),
                "fusion_entropy": float(entropy),
                "fusion_confidence": float(confidence)
            })

        results.sort(key=lambda x: x["fused_score"], reverse=True)

        return results[:top_k]

    # =========================================================
    # RERANKER ENSEMBLE (UNCERTAINTY-SAFE)
    # =========================================================
    @staticmethod
    def ensemble_with_reranker(
        fused_results: List[Dict],
        reranker_scores: np.ndarray,
        top_k: int = 5,
        reranker_weight: float = 0.5,
        score_key: str = "fused_score"
    ) -> List[Dict]:

        if len(fused_results) != len(reranker_scores):
            raise ValueError("Length mismatch between inputs")

        fused = np.array([r[score_key] for r in fused_results])

        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        reranker_scores = (reranker_scores - reranker_scores.min()) / (reranker_scores.max() - reranker_scores.min() + 1e-8)

        combined = (1 - reranker_weight) * fused + reranker_weight * reranker_scores

        # uncertainty penalty (IMPORTANT SAF FIX)
        uncertainty_penalty = np.array([
            r.get("fusion_uncertainty", 0.5) for r in fused_results
        ])

        combined = combined * (1.0 - 0.3 * uncertainty_penalty)

        top_idx = np.argsort(combined)[::-1][:top_k]

        results = []

        for rank, idx in enumerate(top_idx, 1):

            r = dict(fused_results[idx])

            r["combined_score"] = float(combined[idx])
            r["rank"] = rank

            # SAF final adjustment signal
            r["final_retrieval_confidence"] = float(
                combined[idx] / (1.0 + r.get("fusion_uncertainty", 0.5))
            )

            results.append(r)

        return results