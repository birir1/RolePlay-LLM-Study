def hallucination_rate(predictions, contexts):
    total = len(predictions)
    hallucinated = 0

    for pred, ctx in zip(predictions, contexts):
        if pred not in ctx:
            hallucinated += 1

    return hallucinated / total