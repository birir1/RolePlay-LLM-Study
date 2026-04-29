import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .bm25 import BM25Retriever
from .dense_retriever import DenseRetriever
from .reranker import CrossEncoderReranker
from .fusion import RetrievalFusion

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================
@dataclass
class RetrieverConfig:
    use_bm25: bool = True
    use_dense: bool = True
    use_reranker: bool = True

    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_batch_size: int = 32

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32

    top_k: int = 10
    device: str = "cpu"


# =========================================================
# HYBRID RETRIEVER (FIXED)
# =========================================================
class HybridRetriever:

    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()

        logger.info("[HybridRetriever] Initializing...")

        self.bm25 = BM25Retriever(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b
        ) if self.config.use_bm25 else None

        self.dense = DenseRetriever(
            model_name=self.config.dense_model,
            device=self.config.device,
            batch_size=self.config.dense_batch_size
        ) if self.config.use_dense else None

        self.reranker = CrossEncoderReranker(
            model_name=self.config.reranker_model,
            device=self.config.device,
            batch_size=self.config.reranker_batch_size
        ) if self.config.use_reranker else None

        logger.info("[HybridRetriever] Ready ✓")

    # =========================================================
    # INDEXING
    # =========================================================
    def index_corpus(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        if self.bm25:
            self.bm25.index_corpus(documents, metadata)
        if self.dense:
            self.dense.index_corpus(documents, metadata)

    # =========================================================
    # 🔥 HYBRID SCORE (CRITICAL FIX)
    # =========================================================
    def _compute_hybrid_score(self, dense_score, sparse_score):
        return 0.5 * dense_score + 0.5 * sparse_score

    # =========================================================
    # 🔥 UNCERTAINTY
    # =========================================================
    def _compute_uncertainty(self, dense_score, sparse_score):
        disagreement = abs(dense_score - sparse_score)
        confidence = (dense_score + sparse_score) / 2.0
        return 0.5 * (1 - confidence) + 0.5 * disagreement

    # =========================================================
    # RETRIEVE
    # =========================================================
    def retrieve(self, query: str, top_k: int = 5) -> Dict:

        # -------------------------
        # Dense
        # -------------------------
        dense_results = (
            self.dense.retrieve(query, top_k=self.config.top_k)
            if self.dense else []
        )

        dense_map = {
            r.get("text", ""): float(r.get("dense_score", 0.0))
            for r in dense_results
        }

        # -------------------------
        # Sparse (BM25)
        # -------------------------
        sparse_results = (
            self.bm25.retrieve(query, top_k=self.config.top_k)
            if self.bm25 else []
        )

        sparse_map = {
            r.get("text", ""): float(r.get("bm25_score", 0.0))
            for r in sparse_results
        }

        # -------------------------
        # Fusion
        # -------------------------
        fused = RetrievalFusion.reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=self.config.top_k
        )

        if not fused:
            return {
                "query": query,
                "documents": [],
                "abstain": True
            }

        # -------------------------
        # 🔥 Attach scores (FIXED)
        # -------------------------
        for doc in fused:
            text = doc.get("text", "")

            d_score = dense_map.get(text, 0.0)
            s_score = sparse_map.get(text, 0.0)

            doc["dense_score"] = d_score
            doc["sparse_score"] = s_score
            doc["hybrid_score"] = self._compute_hybrid_score(d_score, s_score)

            doc["uncertainty"] = self._compute_uncertainty(d_score, s_score)
            doc["confidence"] = 1.0 - doc["uncertainty"]

        # -------------------------
        # Sort by hybrid score FIRST
        # -------------------------
        fused = sorted(fused, key=lambda x: x["hybrid_score"], reverse=True)

        # -------------------------
        # Rerank (if enabled)
        # -------------------------
        if self.reranker:
            fused = self.reranker.rerank(query, fused, top_k=self.config.top_k)

        # -------------------------
        # Global uncertainty
        # -------------------------
        global_uncertainty = float(np.mean([d["uncertainty"] for d in fused]))

        return {
            "query": query,
            "documents": fused[:top_k],
            "uncertainty": {
                "global_uncertainty": global_uncertainty,
                "global_confidence": 1.0 - global_uncertainty
            },
            "abstain": global_uncertainty > 0.7
        }