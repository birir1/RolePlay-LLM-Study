from typing import List, Dict, Any, Optional

from src.phase3.generation.generate_candidates import generate_candidates
from src.retrieval.reranker import CrossEncoderReranker
from src.phase3.saf_reranker import SAFReranker


# =========================================================
# SAF PIPELINE
# =========================================================
class SAFPipeline:

    def __init__(self):
        self.reranker = CrossEncoderReranker()
        self.saf = SAFReranker()

    # -----------------------------------------------------
    # MAIN PIPELINE
    # -----------------------------------------------------
    def run(
        self,
        query: str,
        context: str = "",
        models: Optional[List[str]] = None,
        n_candidates: int = 2
    ) -> Dict[str, Any]:

        # =================================================
        # STEP 1: GENERATE CANDIDATES
        # =================================================
        candidates = generate_candidates(
            query=query,
            context=context,
            models=models,
            n_per_model=n_candidates
        )

        if not candidates:
            return {
                "final_answer": None,
                "error": "No candidates generated",
                "candidates": []
            }

        # =================================================
        # STEP 2: SEMANTIC RERANKING
        # =================================================
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.score_query_document_pairs(pairs)

        for c, s in zip(candidates, scores):
            c["reranker_score"] = float(s)

        # =================================================
        # STEP 3: SAF SCORING
        # =================================================
        for c in candidates:
            c["hallucination_rate"] = c.get("hallucination_rate", 0.3)
            c["faithfulness"] = c.get("faithfulness", 0.5)
            c["persona_drift"] = c.get("persona_drift", 0.1)

        ranked = self.saf.rerank_candidates(query, candidates)

        # =================================================
        # STEP 4: SELECT BEST
        # =================================================
        best = ranked[0] if ranked else None

        return {
            "final_answer": best["text"] if best else None,
            "best_model": best.get("model") if best else None,
            "candidates": ranked
        }

    # -----------------------------------------------------
    # DEBUG VIEW (FOR RESEARCH)
    # -----------------------------------------------------
    def explain(self, result: Dict[str, Any]) -> None:

        print("\n================ SAF PIPELINE ================\n")

        for i, c in enumerate(result.get("candidates", [])):
            print(f"[{i}] MODEL: {c['model']}")
            print(f"    RERANK: {c.get('reranker_score'):.3f}")
            print(f"    SAF:    {c.get('saf_score', 0):.3f}")
            print(f"    TEXT:   {c['text'][:120]}...\n")

        print("FINAL ANSWER:\n")
        print(result.get("final_answer"))
        
        # =========================================================
# CLI TEST RUNNER
# =========================================================
if __name__ == "__main__":

    pipeline = SAFPipeline()

    print("\n==============================")
    print("      SAF PIPELINE TEST")
    print("==============================\n")

    while True:
        query = input("\nQuery (type 'exit' to quit): ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        context = input("Context (optional): ").strip()

        print("\n[RUNNING SAF PIPELINE...]\n")

        result = pipeline.run(
            query=query,
            context=context,
            models=["mt5", "transformer"],
            n_candidates=2
        )

        pipeline.explain(result)