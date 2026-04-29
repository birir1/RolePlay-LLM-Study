# Phase 2: Retrieval Engine

> Production-grade hybrid retrieval system combining lexical, semantic, and neural reranking.

## Architecture

```
Query
  ↓
┌─────────────────────────────────────┐
│ BM25 Lexical Search                 │ → Top-10 lexical matches
│ (rank_bm25, full-text indexing)     │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Dense Semantic Search               │ → Top-10 semantic matches
│ (BGE-base-en, FAISS if scaled)      │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Reciprocal Rank Fusion (RRF)        │ → Fused top-10
│ λ_fused = Σ(1 / (k + rank))         │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Cross-Encoder Reranking             │ → Top-5 reranked
│ (MARCO MiniLM / qnli-distilroberta)  │
└─────────────────────────────────────┘
  ↓
Output (Top-5 documents)
```

## Components

### 1. BM25Retriever

Lexical full-text search using Okapi BM25 algorithm.

```python
from src.retrieval import BM25Retriever

retriever = BM25Retriever(k1=1.5, b=0.75)
retriever.index_corpus(documents)

results = retriever.retrieve("What is Paris?", top_k=5)
# Returns: [{"document": "...", "bm25_score": 4.5, "rank": 1}, ...]

# Save/load index
retriever.save_index("indexes/bm25.pkl")
```

**Use when:**
- Exact keyword matching is important
- Query and documents share exact terms
- Lightweight, CPU-only retrieval needed

### 2. DenseRetriever

Neural dense retrieval using sentence transformers.

```python
from src.retrieval import DenseRetriever

retriever = DenseRetriever(
    model_name="BAAI/bge-base-en-v1.5",
    device="cuda",
    normalize=True
)
retriever.index_corpus(documents)

results = retriever.retrieve("What is Paris?", top_k=5)
# Returns: [{"document": "...", "similarity_score": 0.87, "rank": 1}, ...]

# Batch retrieval
all_results = retriever.retrieve_batch(queries, top_k=5)

# Save/load corpus
retriever.save_corpus("indexes/dense")
retriever.load_corpus("indexes/dense")
```

**Supported models:**
- `BAAI/bge-base-en-v1.5` (recommended, 768 dims)
- `intfloat/e5-base-v2` (768 dims)
- `sentence-transformers/all-mpnet-base-v2` (768 dims)
- `BAAI/bge-small-en-v1.5` (384 dims, faster)

**Use when:**
- Semantic similarity matters more than exact keywords
- Paraphrases should match
- GPU available for fast inference

### 3. CrossEncoderReranker

Neural reranking using cross-encoders.

```python
from src.retrieval import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda"
)

# Rerank raw list
results = reranker.rerank("What is Paris?", doc_list, top_k=5)
# Returns: [{"document": "...", "cross_encoder_score": 0.92, "rank": 1}, ...]

# Rerank with metadata preservation
results = reranker.rerank_with_metadata(
    query,
    doc_list_with_metadata,
    doc_key="document"
)
```

**Supported models:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (recommended, fast)
- `cross-encoder/qnli-distilroberta-base`
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual)

**Use when:**
- Final ranking stage after initial retrieval
- Query-document interaction matters
- Willing to trade speed for quality

### 4. RetrievalFusion

Combines multiple ranking signals.

```python
from src.retrieval import RetrievalFusion

# Reciprocal Rank Fusion
fused = RetrievalFusion.reciprocal_rank_fusion(
    [bm25_results, dense_results],
    top_k=5,
    k_param=60.0
)

# Weighted combination
fused = RetrievalFusion.weighted_fusion(
    [bm25_results, dense_results],
    weights=[0.3, 0.7],
    top_k=5,
    normalize_scores=True
)

# Ensemble with reranker
ensemble = RetrievalFusion.ensemble_with_reranker(
    fused_results,
    reranker_scores,
    top_k=5,
    reranker_weight=0.4  # 40% reranker, 60% fusion
)
```

### 5. HybridRetriever

Orchestrates all components into production system.

```python
from src.retrieval import HybridRetriever, RetrieverConfig

# Configuration
config = RetrieverConfig(
    use_bm25=True,
    use_dense=True,
    use_reranker=True,
    dense_model="BAAI/bge-base-en-v1.5",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=10,
    top_k_after_rerank=5,
    fusion_method="rrf",  # or "weighted"
    device="cuda"
)

# Initialize and index
retriever = HybridRetriever(config)
retriever.index_corpus(documents, metadata)

# Retrieve with intermediate results
result = retriever.retrieve(
    "What is Paris?",
    return_intermediate=True
)
docs = result["results"]
intermediate = result["intermediate"]
# intermediate has: bm25, dense, fused, reranked results

# Batch retrieve
all_results = retriever.retrieve_batch(queries, top_k=5)

# Get stats
stats = retriever.stats()
```

## Performance Characteristics

| Component | Speed | Quality | Memory |
|-----------|-------|---------|--------|
| BM25 | Fast | Medium | Low |
| Dense (BGE) | Medium | High | High |
| Cross-Encoder | Slow | Highest | High |
| Fusion (RRF) | Very Fast | High | Low |
| **Hybrid** | Medium | Highest | High |

## Real-World Usage Example

```python
from src.retrieval import HybridRetriever, RetrieverConfig

# Production config
config = RetrieverConfig(
    use_bm25=True,
    use_dense=True,
    use_reranker=True,
    dense_model="BAAI/bge-base-en-v1.5",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=10,
    top_k_after_rerank=5,
    fusion_method="rrf",
    device="cuda"
)

retriever = HybridRetriever(config)

# Load real corpus from files
with open("corpus.txt") as f:
    documents = [line.strip() for line in f]

retriever.index_corpus(documents)

# Retrieve
user_query = input("Query: ")
results = retriever.retrieve(user_query, top_k=5)

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc['document'][:100]}...")
    print(f"   Score: {doc['combined_score']:.3f}\n")
```

## Integration with Phase 1 (SAF-RAG)

```python
from src.retrieval import HybridRetriever, RetrieverConfig
from src.models.saf_rag import SAFRAG, SAFRAGConfig

# Initialize both
retriever = HybridRetriever(RetrieverConfig(...))
saf_rag = SAFRAG(SAFRAGConfig(...))

# Index corpus
retriever.index_corpus(documents)

# User query
user_query = "Isn't it obvious that vaccines are dangerous?"

# Retrieve
retrieved_docs_raw = retriever.retrieve(user_query, top_k=5)
retrieved_texts = [d["document"] for d in retrieved_docs_raw]

# Generate with sycophancy awareness
result = saf_rag.generate(
    query=user_query,
    documents=retrieved_texts,
    return_details=True
)

print(result["answer"])
print(f"Risk Score: {result['sycophancy_risk']['risk_score']:.3f}")
print(f"Lambda: {result['lambda']:.3f}")
```

## Requirements

```bash
pip install rank-bm25 sentence-transformers torch
```

## Files

- `bm25.py` - BM25 lexical retriever
- `dense_retriever.py` - Dense neural retriever (BGE/E5)
- `reranker.py` - Cross-encoder reranking
- `fusion.py` - RRF and weighted fusion
- `retriever.py` - HybridRetriever orchestrator
- `__init__.py` - Package exports

## Tests

```bash
# Phase 2 only
python scripts/test_phase2_retrieval.py

# Phase 1 + Phase 2 integration
python scripts/test_phase1_phase2_integration.py
```

## Configuration Examples

### Fast but Good (Recommended for Initial)
```python
RetrieverConfig(
    use_bm25=True,
    use_dense=True,
    use_reranker=False,  # Skip reranking
    top_k=5
)
```

### High Quality (Full Production)
```python
RetrieverConfig(
    use_bm25=True,
    use_dense=True,
    use_reranker=True,
    dense_model="BAAI/bge-base-en-v1.5",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=10,
    top_k_after_rerank=5,
    fusion_method="rrf"
)
```

### Semantic Only (For Very High-Quality Queries)
```python
RetrieverConfig(
    use_bm25=False,
    use_dense=True,
    use_reranker=True
)
```
