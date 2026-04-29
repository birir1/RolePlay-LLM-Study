"""
Phase 2: Retrieval Engine Integration Test

Demonstrates real hybrid retrieval system with:
- BM25 lexical search
- Dense semantic search (BGE)
- Cross-encoder reranking
- RRF fusion
"""

import logging
from typing import List, Dict

from src.retrieval import (
    BM25Retriever,
    DenseRetriever,
    CrossEncoderReranker,
    HybridRetriever,
    RetrieverConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# Real corpus for testing (no dummy data)
WIKIPEDIA_SAMPLES = [
    """The Earth is the third planet from the Sun and the only astronomical object known to harbor life. 
    Water covers most of Earth's surface, with continents and islands making up much of the land. 
    Earth's atmosphere is mostly nitrogen and oxygen. Earth orbits the Sun every 365.25 days.""",
    
    """Paris is the capital and most populous city of France. It is located on the Seine River in 
    north-central France. Paris is known for the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. 
    The city is a major global center for art, fashion, and culture.""",
    
    """Vaccines are biological preparations that provide active acquired immunity against specific diseases. 
    They contain agents that resemble disease-causing microorganisms. Vaccines have been extensively studied 
    for safety and efficacy. Vaccination programs have saved millions of lives globally.""",
    
    """Climate change refers to long-term shifts in global temperatures and weather patterns. 
    Scientists have documented evidence of global warming since the mid-20th century. 
    The primary cause is increased greenhouse gas emissions from human activities. 
    Efforts to mitigate climate change include renewable energy and carbon reduction.""",
    
    """Artificial intelligence is intelligence demonstrated by machines. AI has areas of research including 
    machine learning, deep learning, and natural language processing. AI applications range from robotics 
    to medical diagnosis. Modern AI relies on large datasets and computational power.""",
    
    """The Internet is a global system of interconnected computer networks using the Internet protocol suite. 
    It provides services such as the World Wide Web, email, and file sharing. The Internet has transformed 
    communication, commerce, and access to information globally. Speed and accessibility continue to improve.""",
]


class Phase2Demo:
    """Phase 2 Retrieval Engine demonstration."""
    
    def __init__(self):
        """Initialize retrieval engines."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: RETRIEVAL ENGINE")
        logger.info("="*70)
        
        self.corpus = WIKIPEDIA_SAMPLES
    
    def demo_bm25(self):
        """Demonstrate BM25 lexical retrieval."""
        logger.info("\n" + "-"*70)
        logger.info("BM25 Lexical Retrieval")
        logger.info("-"*70)
        
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index_corpus(self.corpus)
        
        queries = [
            "What is the capital of France?",
            "Tell me about vaccines",
            "How does the internet work?"
        ]
        
        for query in queries:
            logger.info(f"\n[Query] {query}")
            results = retriever.retrieve(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. BM25 Score: {result['bm25_score']:.3f}")
                logger.info(f"     Text: {result['document'][:80]}...")
        
        logger.info(f"\n[Stats] {retriever.stats()}")
    
    def demo_dense(self):
        """Demonstrate dense semantic retrieval."""
        logger.info("\n" + "-"*70)
        logger.info("Dense Semantic Retrieval (BGE)")
        logger.info("-"*70)
        
        retriever = DenseRetriever(
            model_name="BAAI/bge-base-en-v1.5",
            device="cuda",
            normalize=True
        )
        retriever.index_corpus(self.corpus)
        
        queries = [
            "What is the capital of France?",
            "Tell me about vaccines",
            "How does the internet work?"
        ]
        
        for query in queries:
            logger.info(f"\n[Query] {query}")
            results = retriever.retrieve(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. Similarity: {result['similarity_score']:.3f}")
                logger.info(f"     Text: {result['document'][:80]}...")
        
        logger.info(f"\n[Stats] {retriever.stats()}")
    
    def demo_reranker(self):
        """Demonstrate cross-encoder reranking."""
        logger.info("\n" + "-"*70)
        logger.info("Cross-Encoder Reranking")
        logger.info("-"*70)
        
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cuda"
        )
        
        query = "What is artificial intelligence?"
        
        # Use first 3 documents as candidates
        candidates = self.corpus[:3]
        
        logger.info(f"\n[Query] {query}")
        logger.info(f"[Candidates] {len(candidates)} documents")
        
        results = reranker.rerank(query, candidates, top_k=3)
        
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. Cross-Encoder Score: {result['cross_encoder_score']:.3f}")
            logger.info(f"     Text: {result['document'][:80]}...")
        
        logger.info(f"\n[Stats] {reranker.stats()}")
    
    def demo_hybrid(self):
        """Demonstrate full hybrid retrieval."""
        logger.info("\n" + "-"*70)
        logger.info("Hybrid Retrieval (BM25 + Dense + Reranking)")
        logger.info("-"*70)
        
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
        retriever.index_corpus(self.corpus)
        
        test_queries = [
            "What is the capital of France?",
            "How are vaccines developed?",
            "Tell me about AI and machine learning",
            "What causes climate change?"
        ]
        
        for query in test_queries:
            logger.info(f"\n[Query] {query}")
            
            # Get intermediate results
            result = retriever.retrieve(query, return_intermediate=True)
            results = result["results"]
            intermediate = result["intermediate"]
            
            logger.info(f"  BM25 retrieved: {len(intermediate.get('bm25', []))} docs")
            logger.info(f"  Dense retrieved: {len(intermediate.get('dense', []))} docs")
            logger.info(f"  After fusion: {len(intermediate.get('fused', []))} docs")
            logger.info(f"  After reranking: {len(results)} docs")
            
            logger.info("\n  Final Results:")
            for i, doc_result in enumerate(results, 1):
                logger.info(f"    {i}. Score: {doc_result.get('combined_score', doc_result.get('fused_score', 0)):.3f}")
                logger.info(f"       Text: {doc_result['document'][:75]}...")
        
        logger.info(f"\n[System Stats]")
        stats = retriever.stats()
        logger.info(f"  Fusion Method: {stats['config']['fusion_method']}")
        logger.info(f"  Components: BM25 {stats['config']['use_bm25']}, Dense {stats['config']['use_dense']}, Reranker {stats['config']['use_reranker']}")


def main():
    """Run Phase 2 demonstration."""
    logger.info("\n" + "🔥 "*35)
    logger.info("PHASE 2: RETRIEVAL ENGINE")
    logger.info("🔥 "*35)
    
    demo = Phase2Demo()
    
    # Run all demonstrations
    demo.demo_bm25()
    demo.demo_dense()
    demo.demo_reranker()
    demo.demo_hybrid()
    
    logger.info("\n" + "="*70)
    logger.info("✓ PHASE 2 DEMO COMPLETE")
    logger.info("="*70)
    logger.info("\nPhase 2 implements production retrieval:")
    logger.info("  ✓ BM25 lexical indexing and search")
    logger.info("  ✓ Dense semantic retrieval (BGE)")
    logger.info("  ✓ Cross-encoder reranking (MARCO)")
    logger.info("  ✓ RRF-based fusion")
    logger.info("\nNext: Phase 2 + Phase 1 Integration")


if __name__ == "__main__":
    main()
