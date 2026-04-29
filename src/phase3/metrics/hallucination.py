import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def hallucination_score(predictions, contexts, model):
    """
    Robust hallucination detection (RAG-aware).

    Supports:
    - Multiple context chunks per sample
    - Variable-length retrieval
    - Proper grounding aggregation

    Combines:
    - Best support (strongest evidence)
    - Mean support (overall grounding)
    - Uncertainty penalty (spread)
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

        # if no valid context → skip (will penalize later)
        if len(valid_chunks) == 0:
            continue

        for c in valid_chunks:
            flat_contexts.append(c)
            mapping.append(i)

    # If nothing valid → everything hallucinated
    if len(flat_contexts) == 0:
        return 1.0

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
    uncertainty = []

    for g in grouped_sims:
        if len(g) == 0:
            # no grounding → worst case
            best_support.append(0.0)
            mean_support.append(0.0)
            uncertainty.append(1.0)
        else:
            g = np.array(g)
            best_support.append(np.max(g))
            mean_support.append(np.mean(g))
            uncertainty.append(np.std(g))

    best_support = np.array(best_support)
    mean_support = np.array(mean_support)
    uncertainty = np.array(uncertainty)

    # =========================
    # 4. GROUNDEDNESS
    # =========================
    groundedness = (
        0.65 * best_support +
        0.25 * mean_support -
        0.10 * uncertainty
    )

    groundedness = np.clip(groundedness, 0.0, 1.0)

    # =========================
    # 5. HALLUCINATION
    # =========================
    hallucination = 1.0 - groundedness

    return float(np.mean(hallucination))