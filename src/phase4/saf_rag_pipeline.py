import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, List, Dict, Union

from src.models.saf_rag.sycophancy_detector import SycophancyDetector
from src.models.saf_rag.fusion_gate import FusionGate

from src.retrieval.hybrid.hybrid_merger import HybridMerger
from src.retrieval.uncertainty.uncertainty_scorer import UncertaintyScorer
from src.retrieval.filtering.adaptive_filter import AdaptiveFilter
from src.retrieval.reranker.cross_encoder_reranker import CrossEncoderReranker

from src.phase4.grounding_scorer import GroundingScorer
from src.phase4.response_filter import ResponseFilter


class SAFRAG(nn.Module):

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = None,
        use_fusion_gate: bool = True
    ):
        super().__init__()

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_fusion_gate = use_fusion_gate

        print("[SAF-RAG] Loading FLAN-T5...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # 🔥 stability fix
        self.model.config.tie_word_embeddings = False

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()  # 🔥 IMPORTANT

        print("[SAF-RAG] Loading SAF components...")

        # Core components
        self.detector = SycophancyDetector(device=self.device)
        self.detector.eval()

        self.response_filter = ResponseFilter()

        self.fusion_gate = FusionGate() if use_fusion_gate else None
        self.hybrid_merger = HybridMerger(alpha=0.6)
        self.uncertainty_scorer = UncertaintyScorer()

        self.adaptive_filter = AdaptiveFilter(
            uncertainty_threshold=0.7,
            min_docs=2,
            max_docs=5
        )

        self.reranker = CrossEncoderReranker()
        self.grounding_scorer = GroundingScorer(device=self.device)

        print("[SAF-RAG] Ready ✓")

    # =========================================================
    # TRAINING FORWARD
    # =========================================================
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        lm_loss = outputs.loss

        if lm_loss is None or torch.isnan(lm_loss):
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}

        return {"loss": lm_loss}

    # =========================================================
    # CONTEXT
    # =========================================================
    def _build_context(self, docs: List[Dict], max_chars: int = 1500) -> str:
        context_blocks = []
        total_len = 0

        for i, d in enumerate(docs):
            text = d.get("text", "").strip()
            if not text:
                continue

            block = f"[Doc {i+1}] {text}"

            if total_len + len(block) > max_chars:
                break

            context_blocks.append(block)
            total_len += len(block)

        return "\n".join(context_blocks)

    # =========================================================
    # PROMPT
    # =========================================================
    def _build_prompt(self, query: str, context: str) -> str:
        return f"""
You are a factual QA system.

STRICT RULES:
- Use ONLY the provided documents
- DO NOT generate example code, pseudo code, or dummy code
- DO NOT roleplay or simulate scenarios
- Give direct factual answers only
- If unsure, say "I don't know"

Documents:
{context}

Question:
{query}

Answer:
""".strip()

    # =========================================================
    # GENERATION
    # =========================================================
    @torch.no_grad()
    def generate(
        self,
        query: str,
        documents: Optional[List[Union[str, Dict]]] = None,
        dense_results: Optional[List[Dict]] = None,
        sparse_results: Optional[List[Dict]] = None,
        max_length: int = 128
    ) -> str:

        try:
            # -------------------------
            # Input validation
            # -------------------------
            if not isinstance(query, str) or not query.strip():
                return "I don't know"

            # -------------------------
            # 🔥 Sycophancy score (NO hard rejection)
            # -------------------------
            syco_score = self.detector.score(query)

            # -------------------------
            # Normalize docs
            # -------------------------
            docs: List[Dict] = []

            if documents:
                for d in documents:
                    if isinstance(d, str):
                        docs.append({"text": d})
                    elif isinstance(d, dict):
                        text = d.get("text", "").strip()
                        if text:
                            docs.append({"text": text})
            else:
                docs = self.hybrid_merger.merge(
                    dense_results or [],
                    sparse_results or [],
                    top_k=10
                )

            if not docs:
                return "I don't know"

            # -------------------------
            # RERANK
            # -------------------------
            docs = self.reranker.rerank(query, docs, top_k=5)

            # -------------------------
            # UNCERTAINTY FILTER
            # -------------------------
            uncertainty_output = self.uncertainty_scorer.compute(docs)

            filter_output = self.adaptive_filter.filter(
                docs,
                uncertainty_output
            )

            if filter_output.get("abstain", False):
                docs = docs[:2]  # fallback instead of full rejection
            else:
                docs = filter_output.get("filtered_docs", [])

            if not docs:
                return "I don't know"

            # -------------------------
            # CONTEXT + PROMPT
            # -------------------------
            context = self._build_context(docs)
            prompt = self._build_prompt(query, context)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            # -------------------------
            # GENERATE
            # -------------------------
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=2
            )

            result = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()

            if not result:
                return "I don't know"

            # -------------------------
            # 🔥 Response filtering (SOFT, not destructive)
            # -------------------------
            roleplay_score = self.response_filter.detect_roleplay(result)
            code_score = self.response_filter.detect_code(result)
            syco_out = self.response_filter.detect_sycophancy(result)

            if roleplay_score > 0.4 or code_score > 0.4:
                result = self.response_filter.clean(result)

            # 🔥 only lightly adjust sycophancy (don't overwrite answer)
            if syco_out > 0.5:
                result = "The claim should be evaluated critically. " + result

            # -------------------------
            # 🔥 Grounding check
            # -------------------------
            grounding_score = self.grounding_scorer.score(
                query,
                result,
                docs
            )

            if grounding_score < 0.25:
                return "I don't know"

            return result

        except Exception as e:
            print(f"[GENERATION ERROR]: {e}")
            return "I don't know"