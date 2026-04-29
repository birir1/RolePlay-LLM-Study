from typing import List, Dict, Tuple, Union
import numpy as np


class HybridMerger:
    """
    SAF-RAG Hybrid Retrieval Merger (ROBUST)

    Fixes:
    - handles tuple + dict inputs
    - guarantees schema consistency
    - prevents missing dense/sparse score crashes
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.beta = 1.0 - alpha

    # =========================================================
    # MAIN MERGE FUNCTION
    # =========================================================
    def merge(
        self,
        dense_results: List[Union[Tuple[str, float], Dict]],
        sparse_results: List[Union[Tuple[str, float], Dict]],
        top_k: int = 10
    ) -> List[Dict]:

        merged: Dict[str, Dict] = {}

        # -------------------------
        # 1. Collect dense (ROBUST)
        # -------------------------
        for item in dense_results:
            text, score = self._parse_item(item)

            if not text:
                continue

            merged.setdefault(text, {
                "text": text,
                "dense_score": 0.0,
                "sparse_score": 0.0
            })

            merged[text]["dense_score"] = float(score)

        # -------------------------
        # 2. Collect sparse (ROBUST)
        # -------------------------
        for item in sparse_results:
            text, score = self._parse_item(item)

            if not text:
                continue

            merged.setdefault(text, {
                "text": text,
                "dense_score": 0.0,
                "sparse_score": 0.0
            })

            merged[text]["sparse_score"] = float(score)

        results = list(merged.values())

        if not results:
            return []

        # -------------------------
        # 3. Normalize scores
        # -------------------------
        results = self._normalize_scores(results)

        enriched = []

        # -------------------------
        # 4. Compute signals
        # -------------------------
        for doc in results:

            d = doc.get("dense_score", 0.0)
            s = doc.get("sparse_score", 0.0)

            hybrid_score = self.alpha * d + self.beta * s
            agreement = 1.0 - abs(d - s)
            confidence = 0.5 * hybrid_score + 0.5 * agreement
            uncertainty = 1.0 - confidence

            enriched.append({
                "text": doc["text"],
                "dense_score": float(d),
                "sparse_score": float(s),
                "hybrid_score": float(hybrid_score),
                "confidence": float(confidence),
                "agreement": float(agreement),
                "uncertainty": float(uncertainty)
            })

        # -------------------------
        # 5. Ranking
        # -------------------------
        enriched.sort(
            key=lambda x: (
                0.7 * x["hybrid_score"] +
                0.3 * x["confidence"]
            ),
            reverse=True
        )

        return enriched[:top_k]

    # =========================================================
    # 🔥 NEW: SAFE INPUT PARSER
    # =========================================================
    def _parse_item(self, item: Union[Tuple[str, float], Dict]) -> Tuple[str, float]:

        if isinstance(item, tuple) and len(item) == 2:
            return item[0], float(item[1])

        if isinstance(item, dict):
            text = item.get("text", "")
            score = item.get("dense_score", item.get("sparse_score", item.get("score", 0.0)))
            return text, float(score)

        return "", 0.0

    # =========================================================
    # NORMALIZATION
    # =========================================================
    def _normalize_scores(self, docs: List[Dict]) -> List[Dict]:

        dense = np.array([d.get("dense_score", 0.0) for d in docs], dtype=np.float32)
        sparse = np.array([d.get("sparse_score", 0.0) for d in docs], dtype=np.float32)

        dense = self._minmax_safe(dense)
        sparse = self._minmax_safe(sparse)

        for i, d in enumerate(docs):
            d["dense_score"] = float(dense[i])
            d["sparse_score"] = float(sparse[i])

        return docs

    # =========================================================
    # SAFE MINMAX
    # =========================================================
    def _minmax_safe(self, x: np.ndarray) -> np.ndarray:

        if len(x) == 0:
            return x

        min_v = np.min(x)
        max_v = np.max(x)

        if max_v - min_v < 1e-8:
            return np.ones_like(x) * 0.5

        return (x - min_v) / (max_v - min_v + 1e-8)