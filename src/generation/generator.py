import numpy as np
from typing import List, Dict, Any


class SAFGenerator:
    """
    Improved SAF-RAG Generation Layer

    Fixes:
    - Strong grounding enforcement
    - Score-aware context filtering
    - Calibrated abstention
    - Better prompt control
    """

    def __init__(
        self,
        llm=None,
        max_context_docs: int = 5,
        uncertainty_threshold: float = 0.7,
        min_doc_score: float = 0.2
    ):
        """
        llm: callable → llm(prompt: str) -> str
        """
        self.llm = llm
        self.max_context_docs = max_context_docs
        self.uncertainty_threshold = uncertainty_threshold
        self.min_doc_score = min_doc_score

    # =========================================================
    # SCORE NORMALIZATION
    # =========================================================
    def _normalize_scores(self, documents: List[Dict]) -> List[Dict]:
        if not documents:
            return documents

        scores = [doc.get("score", doc.get("reranker_score", 0.0)) for doc in documents]
        min_s, max_s = min(scores), max(scores)

        for doc in documents:
            s = doc.get("score", doc.get("reranker_score", 0.0))
            if max_s - min_s > 1e-6:
                doc["norm_score"] = (s - min_s) / (max_s - min_s)
            else:
                doc["norm_score"] = 0.5

        return documents

    # =========================================================
    # CONTEXT BUILDER (FILTERED + RANKED)
    # =========================================================
    def _build_context(self, documents: List[Dict]) -> str:
        documents = self._normalize_scores(documents)

        # Filter weak docs
        documents = [d for d in documents if d.get("norm_score", 0) >= self.min_doc_score]

        # Sort
        documents = sorted(documents, key=lambda x: x.get("norm_score", 0), reverse=True)

        context_parts = []

        for i, doc in enumerate(documents[:self.max_context_docs]):
            text = doc.get("text") or doc.get("document") or ""
            score = doc.get("norm_score", 0.0)

            context_parts.append(
                f"[Doc {i+1} | relevance={score:.2f}]\n{text}"
            )

        return "\n\n".join(context_parts), documents

    # =========================================================
    # UNCERTAINTY ESTIMATION (ROBUST)
    # =========================================================
    def _estimate_uncertainty(self, documents: List[Dict], retrieval_output: Dict[str, Any]) -> float:
        # Prefer provided uncertainty
        uncertainty = retrieval_output.get("uncertainty", {})

        if isinstance(uncertainty, dict):
            u = uncertainty.get("global_uncertainty", None)
            if u is not None:
                return float(u)

        # Fallback: derive from doc scores
        if not documents:
            return 1.0

        avg_score = np.mean([d.get("norm_score", 0.0) for d in documents])
        return 1 - avg_score

    # =========================================================
    # ABSTAIN LOGIC (CRITICAL FIX)
    # =========================================================
    def should_abstain(self, uncertainty: float, documents: List[Dict]) -> bool:
        if len(documents) == 0:
            return True

        if uncertainty >= self.uncertainty_threshold:
            return True

        return False

    # =========================================================
    # PROMPT BUILDER (STRONG GROUNDING)
    # =========================================================
    def _build_prompt(self, query: str, context: str, abstain: bool) -> str:

        if abstain:
            return f"""
You are a safety-aware AI system.

The system has determined that the available information is insufficient or unreliable.

Question:
{query}

Respond EXACTLY with:
"I don't have enough reliable information to answer this."
""".strip()

        return f"""
You are a highly reliable AI assistant.

You MUST follow these rules strictly:

1. Use ONLY the provided context
2. If the answer is not explicitly in the context → say "I don't know"
3. Do NOT infer, assume, or hallucinate
4. Keep the answer factual and concise

Context:
{context}

Question:
{query}

Answer:
""".strip()

    # =========================================================
    # MAIN GENERATION PIPELINE
    # =========================================================
    def generate(
        self,
        query: str,
        retrieval_output: Dict[str, Any]
    ) -> Dict[str, Any]:

        raw_docs = retrieval_output.get("documents", [])

        # -------------------------
        # Build filtered context
        # -------------------------
        context, filtered_docs = self._build_context(raw_docs)

        # -------------------------
        # Estimate uncertainty
        # -------------------------
        uncertainty = self._estimate_uncertainty(filtered_docs, retrieval_output)

        # -------------------------
        # Abstain decision
        # -------------------------
        abstain = retrieval_output.get("abstain", False)

        if not abstain:
            abstain = self.should_abstain(uncertainty, filtered_docs)

        # -------------------------
        # Build prompt
        # -------------------------
        prompt = self._build_prompt(query, context, abstain)

        # -------------------------
        # LLM call
        # -------------------------
        if self.llm is None:
            return {
                "query": query,
                "response": "[NO LLM ATTACHED]",
                "abstain": abstain,
                "uncertainty": uncertainty,
                "context_used": context
            }

        try:
            response = self.llm(prompt)
        except Exception as e:
            response = f"[GENERATION ERROR]: {str(e)}"

        return {
            "query": query,
            "response": response,
            "abstain": abstain,
            "uncertainty": uncertainty,
            "context_used": context,
            "num_docs": len(filtered_docs)
        }