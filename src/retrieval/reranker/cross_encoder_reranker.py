from typing import List, Tuple, Dict
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


class CrossEncoderReranker:
    """
    SAF-RAG Cross-Encoder Reranker (CLEAN + STABLE)

    Responsibilities:
    - Score (query, document) pairs
    - Normalize scores
    - Return ranked documents
    - Provide SAFE fallback if model unavailable
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

    # =========================================================
    # SCORE FUNCTION
    # =========================================================
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Returns relevance scores for (query, doc) pairs
        """

        if not pairs:
            return []

        # ---------------------------
        # fallback mode
        # ---------------------------
        if self.model is None:
            return [0.5 for _ in pairs]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True
            )

            scores = np.array(scores, dtype=np.float32)

            # safety cleanup
            scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

            return scores.tolist()

        except Exception:
            return [0.5 for _ in pairs]

    # =========================================================
    # RERANK FUNCTION
    # =========================================================
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        text_key: str = "text",
        top_k: int = 5
    ) -> List[Dict]:

        if not documents:
            return []

        pairs = []
        safe_docs = []

        # ---------------------------
        # safe extraction
        # ---------------------------
        for doc in documents:
            text = doc.get(text_key, "")

            if not isinstance(text, str) or len(text.strip()) == 0:
                text = "empty document"

            safe_docs.append(doc)
            pairs.append((query, text))

        # ---------------------------
        # scoring
        # ---------------------------
        scores = self.score_pairs(pairs)

        # attach raw scores
        for doc, score in zip(safe_docs, scores):
            doc["reranker_score_raw"] = float(score)

        # ---------------------------
        # normalize scores (IMPORTANT for SAF)
        # ---------------------------
        score_array = np.array(
            [d["reranker_score_raw"] for d in safe_docs],
            dtype=np.float32
        )

        if len(score_array) > 0:
            min_v, max_v = float(np.min(score_array)), float(np.max(score_array))

            if max_v - min_v > 1e-8:
                norm = (score_array - min_v) / (max_v - min_v)
            else:
                norm = np.ones_like(score_array) * 0.5
        else:
            norm = np.array([])

        # attach normalized score
        for i, doc in enumerate(safe_docs):
            doc["reranker_score"] = float(norm[i])

        # ---------------------------
        # final ranking
        # ---------------------------
        safe_docs.sort(
            key=lambda x: x.get("reranker_score", 0.0),
            reverse=True
        )

        return safe_docs[:top_k]