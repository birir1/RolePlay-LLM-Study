import json
from build_hybrid_retriever import HybridRetriever


def load_data():
    with open("data/processed/test.json") as f:
        return json.load(f)


def evaluate():
    data = load_data()
    corpus = [d["context"] for d in data]

    retriever = HybridRetriever(corpus)

    hits = 0

    for d in data:
        query = d["query"]
        true_context = d["context"]

        retrieved = retriever.retrieve(query, top_k=3)

        if any(true_context[:50] in r for r in retrieved):
            hits += 1

    recall = hits / len(data)
    print(f"\nRetrieval Recall@3: {recall:.4f}")


if __name__ == "__main__":
    evaluate()