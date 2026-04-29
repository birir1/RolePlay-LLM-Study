import numpy as np
import re
from typing import List, Optional


# =========================================================
# 🔥 TEXT QUALITY SIGNALS (important for drift)
# =========================================================
GENERIC_PATTERNS = [
    r"\bexample\b",
    r"\bplaceholder\b",
    r"\bgeneric\b",
    r"\bsample\b",
    r"\bthis is just\b",
    r"\byou can\b",
    r"\betc\b",
    r"\blorem ipsum\b",
]


def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _generic_penalty(text: str) -> float:
    if not text:
        return 1.0

    hits = sum(bool(re.search(p, text)) for p in GENERIC_PATTERNS)

    # soft saturation penalty
    return min(hits * 0.4, 1.0)


def _low_info_penalty(text: str) -> float:
    words = text.split()
    n = len(words)

    if n == 0:
        return 1.0
    if n < 5:
        return 0.8
    if n < 10:
        return 0.5
    if n < 20:
        return 0.2
    return 0.0


def _repetition_penalty(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0

    unique_ratio = len(set(words)) / len(words)

    if unique_ratio > 0.7:
        return 0.0

    return (0.7 - unique_ratio) * 1.2


# =========================================================
# 🧠 MAIN FUNCTION
# =========================================================
def persona_drift_score(
    predictions: List[str],
    queries: List[str],
    model: Optional[object] = None
) -> float:
    """
    Returns:
        drift score in [0, 1]
        0 = perfectly aligned persona
        1 = complete drift
    """

    if not predictions:
        return 1.0

    predictions = [_normalize(p) for p in predictions]
    queries = [_normalize(q) for q in queries]

    # =========================================================
    # 🧪 HEURISTIC MODE (NO MODEL)
    # =========================================================
    if model is None:
        total = 0.0

        for p in predictions:
            total += (
                _generic_penalty(p) * 0.4 +
                _low_info_penalty(p) * 0.4 +
                _repetition_penalty(p) * 0.2
            )

        return float(min(total / len(predictions), 1.0))

    # =========================================================
    # 🧠 SEMANTIC MODE (EMBEDDINGS)
    # =========================================================
    try:
        pred_emb = model.encode(
            predictions,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_emb = model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # cosine similarity matrix
        sims = np.matmul(pred_emb, query_emb.T)

        # ❗ FIX: use soft alignment instead of max-only
        best_match = np.max(sims, axis=1)
        avg_match = np.mean(sims, axis=1)

        semantic_drift = 1.0 - (0.7 * best_match + 0.3 * avg_match)

        # clamp
        semantic_drift = np.clip(semantic_drift, 0.0, 1.0)

        # combine with textual quality penalties
        quality_penalty = np.mean([
            _generic_penalty(p) * 0.4 +
            _low_info_penalty(p) * 0.4 +
            _repetition_penalty(p) * 0.2
            for p in predictions
        ])

        final = 0.7 * semantic_drift + 0.3 * quality_penalty

        return float(np.clip(final, 0.0, 1.0))

    except Exception:
        # safe fallback if embeddings fail
        return persona_drift_score(predictions, queries, None)