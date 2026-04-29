"""
Retrieval Module

Hybrid SAF-RAG retrieval system:
- BM25 (lexical + uncertainty)
- Dense retrieval (semantic)
- Fusion (RRF + uncertainty modeling)
- Cross-encoder reranking
- SAF pipeline orchestration
"""

from .bm25 import BM25Retriever
from .dense_retriever import DenseRetriever
from .fusion import RetrievalFusion
from .retriever import HybridRetriever, RetrieverConfig
from .reranker import CrossEncoderReranker

# SAF pipeline (correct path)
from .pipeline.saf_pipeline import SAFRetrievalPipeline


__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "RetrievalFusion",
    "HybridRetriever",
    "RetrieverConfig",
    "CrossEncoderReranker",
    "SAFRetrievalPipeline"
]