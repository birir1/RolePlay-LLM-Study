import numpy as np
from typing import List, Tuple, Dict

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class CrossEncoderReranker:
    """
    Cross-encoder reranker (FINAL ranking stage only).

    Responsibility:
    - Input: (query, document) pairs
    - Output: relevance scores
    - NO retrieval logic (kept clean for HybridRetriever)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.model = None

        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(model_name, device=device)
            except Exception:
                self.model = None

    # -----------------------------------------------------
    # Score query-document pairs
    # -----------------------------------------------------
    def score_query_document_pairs(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[float]:

        if not pairs:
            return []

        if self.model is None:
            return [0.5 for _ in pairs]

        try:
            scores = self.model.predict(pairs)
            return scores.tolist() if hasattr(scores, "tolist") else list(scores)
        except Exception:
            return [0.5 for _ in pairs]

    # -----------------------------------------------------
    # Rerank documents
    # -----------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        text_key: str = "text",
        top_k: int = 5
    ) -> List[Dict]:

        if not documents:
            return []

        pairs = [(query, d.get(text_key, "")) for d in documents]
        scores = self.score_query_document_pairs(pairs)

        reranked = []
        for doc, score in zip(documents, scores):
            d = dict(doc)
            d["reranker_score"] = float(score)
            reranked.append(d)

        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)

        return reranked[:top_k]