import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def faithfulness_score(predictions, contexts, model):
    """
    Robust faithfulness metric (RAG-aware).

    Supports:
    - Multiple retrieved documents per query
    - Variable-length context lists
    - Proper grounding aggregation

    Combines:
    - Best support (strongest alignment)
    - Mean support (overall grounding)
    - Variance penalty (consistency)
    """

    # =========================
    # 0. BUILD FLAT CONTEXT SET
    # =========================
    flat_contexts = []
    mapping = []  # maps each context → prediction index

    for i, ctx_list in enumerate(contexts):

        # ensure list format
        if not isinstance(ctx_list, list):
            continue

        valid_chunks = [
            c for c in ctx_list
            if isinstance(c, str) and c.strip()
        ]

        if len(valid_chunks) == 0:
            continue

        for c in valid_chunks:
            flat_contexts.append(c)
            mapping.append(i)

    # If no valid context → zero faithfulness
    if len(flat_contexts) == 0:
        return 0.0

    # =========================
    # 1. EMBEDDINGS
    # =========================
    pred_emb = model.encode(
        predictions,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    ctx_emb = model.encode(
        flat_contexts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sims = cosine_similarity(pred_emb, ctx_emb)

    # =========================
    # 2. GROUP SIMILARITIES PER PREDICTION
    # =========================
    grouped_sims = [[] for _ in range(len(predictions))]

    for ctx_idx, pred_idx in enumerate(mapping):
        grouped_sims[pred_idx].append(sims[pred_idx, ctx_idx])

    # =========================
    # 3. COMPUTE SIGNALS
    # =========================
    best_support = []
    mean_support = []
    variance_penalty = []

    for g in grouped_sims:
        if len(g) == 0:
            best_support.append(0.0)
            mean_support.append(0.0)
            variance_penalty.append(1.0)
        else:
            g = np.array(g)
            best_support.append(np.max(g))
            mean_support.append(np.mean(g))
            variance_penalty.append(np.std(g))

    best_support = np.array(best_support)
    mean_support = np.array(mean_support)
    variance_penalty = np.array(variance_penalty)

    # =========================
    # 4. FINAL FAITHFULNESS
    # =========================
    faithfulness = (
        0.6 * best_support +
        0.3 * mean_support -
        0.1 * variance_penalty
    )

    faithfulness = np.clip(faithfulness, 0.0, 1.0)

    return float(np.mean(faithfulness))