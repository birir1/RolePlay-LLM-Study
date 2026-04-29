import numpy as np

def filter_passages(passages, scores, threshold=0.2):
    """
    Remove weak retrievals
    """
    filtered = []

    for p, s in zip(passages, scores):
        if s > threshold:
            filtered.append(p)

    return filtered if filtered else passages[:1]