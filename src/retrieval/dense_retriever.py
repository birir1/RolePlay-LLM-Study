"""
Dense Retriever Module

Neural dense retrieval using sentence transformers (BGE, E5, etc.).
Production-grade with batch processing and GPU support.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import logging

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense vector retrieval using sentence transformers.
    
    Supports models like:
    - BGE (BAAI/bge-base-en-v1.5)
    - E5 (intfloat/e5-base-v2)
    - MPNet (sentence-transformers/all-mpnet-base-v2)
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        normalize: bool = True
    ):
        """
        Initialize dense retriever.
        
        Args:
            model_name: HuggingFace model identifier
            device: torch device
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings to unit length
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        
        logger.info(f"Loading dense retriever: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Set max_seq_length if available
        if hasattr(self.model, 'max_seq_length'):
            logger.info(f"Max sequence length: {self.model.max_seq_length}")
        
        self.corpus_embeddings = None
        self.corpus = []
        self.corpus_metadata = []
    
    def index_corpus(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        use_instruction: bool = True
    ) -> None:
        """
        Encode and index corpus for retrieval.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            use_instruction: Whether to prepend instruction prefix (for BGE)
        """
        self.corpus = documents
        self.corpus_metadata = metadata or [{"id": i} for i in range(len(documents))]
        
        # Prepare texts for encoding
        if use_instruction and "bge" in self.model_name.lower():
            # BGE instruction prefix for corpus
            texts = [f"Represent this sentence for searching relevant passages: {doc}" 
                     for doc in documents]
        else:
            texts = documents
        
        # Encode in batches
        logger.info(f"Encoding {len(documents)} documents...")
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=False,
                normalize_embeddings=self.normalize
            )
            embeddings.append(batch_embeddings)
        
        self.corpus_embeddings = torch.cat(embeddings, dim=0)
        
        # Move to appropriate device for retrieval
        if self.corpus_embeddings.device != self.device:
            self.corpus_embeddings = self.corpus_embeddings.to(self.device)
        
        logger.info(f"Indexed {len(documents)} documents")
        logger.info(f"Embedding dimension: {self.corpus_embeddings.shape[1]}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_instruction: bool = True
    ) -> List[Dict]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            use_instruction: Whether to prepend instruction prefix
            
        Returns:
            List of documents with scores and metadata
        """
        if self.corpus_embeddings is None:
            raise RuntimeError("Corpus not indexed. Call index_corpus() first.")
        
        # Prepare query
        if use_instruction and "bge" in self.model_name.lower():
            query_text = f"Represent this sentence for searching relevant passages: {query}"
        else:
            query_text = query
        
        # Encode query
        query_embedding = self.model.encode(
            query_text,
            convert_to_numpy=False,
            normalize_embeddings=self.normalize
        )
        query_embedding = query_embedding.to(self.device)
        
        # Compute similarities using cosine similarity
        similarities = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Get top-k
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(self.corpus)))
        
        # Build results
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append({
                "document": self.corpus[idx.item()],
                "metadata": self.corpus_metadata[idx.item()],
                "similarity_score": float(score.item()),
                "rank": len(results) + 1
            })
        
        return results
    
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        use_instruction: bool = True,
        batch_size: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Retrieve for multiple queries efficiently.
        
        Args:
            queries: List of queries
            top_k: Number of documents per query
            use_instruction: Whether to prepend instruction prefix
            batch_size: Optional batch size (default: self.batch_size)
            
        Returns:
            List of result lists
        """
        batch_size = batch_size or self.batch_size
        
        # Prepare queries
        if use_instruction and "bge" in self.model_name.lower():
            query_texts = [f"Represent this sentence for searching relevant passages: {q}" 
                          for q in queries]
        else:
            query_texts = queries
        
        # Encode all queries at once
        logger.info(f"Encoding {len(queries)} queries...")
        query_embeddings = self.model.encode(
            query_texts,
            show_progress_bar=False,
            convert_to_numpy=False,
            normalize_embeddings=self.normalize
        )
        query_embeddings = query_embeddings.to(self.device)
        
        # Compute similarities
        similarities = util.cos_sim(query_embeddings, self.corpus_embeddings)
        
        # Get top-k for each query
        results_list = []
        for query_idx, similarities_row in enumerate(similarities):
            top_scores, top_indices = torch.topk(
                similarities_row, 
                k=min(top_k, len(self.corpus))
            )
            
            results = []
            for rank, (score, doc_idx) in enumerate(zip(top_scores, top_indices)):
                results.append({
                    "document": self.corpus[doc_idx.item()],
                    "metadata": self.corpus_metadata[doc_idx.item()],
                    "similarity_score": float(score.item()),
                    "rank": rank + 1
                })
            results_list.append(results)
        
        return results_list
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for arbitrary texts (not in corpus).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tensor of embeddings [len(texts), dim]
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=False,
            normalize_embeddings=self.normalize
        )
        return embeddings.to(self.device)
    
    def save_corpus(self, path: str) -> None:
        """
        Save corpus and embeddings to disk.
        
        Args:
            path: Path to save (will create .corpus.npy, .metadata.json)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save embeddings
        embeddings_path = f"{path}.embeddings.pt"
        torch.save(self.corpus_embeddings, embeddings_path)
        
        # Save corpus as JSON
        corpus_path = f"{path}.corpus.json"
        import json
        with open(corpus_path, "w") as f:
            json.dump({
                "corpus": self.corpus,
                "metadata": self.corpus_metadata,
                "model": self.model_name
            }, f)
        
        logger.info(f"Corpus saved to {corpus_path} and {embeddings_path}")
    
    def load_corpus(self, path: str) -> None:
        """
        Load corpus and embeddings from disk.
        
        Args:
            path: Path to load from
        """
        import json
        
        # Load corpus JSON
        corpus_path = f"{path}.corpus.json"
        with open(corpus_path, "r") as f:
            data = json.load(f)
        
        self.corpus = data["corpus"]
        self.corpus_metadata = data["metadata"]
        
        # Load embeddings
        embeddings_path = f"{path}.embeddings.pt"
        self.corpus_embeddings = torch.load(embeddings_path, map_location=self.device)
        
        logger.info(f"Corpus loaded: {len(self.corpus)} documents")
    
    def stats(self) -> Dict:
        """Get retriever statistics."""
        if self.corpus_embeddings is None:
            return {}
        
        return {
            "model": self.model_name,
            "num_documents": len(self.corpus),
            "embedding_dim": self.corpus_embeddings.shape[1],
            "device": str(self.device),
            "normalize": self.normalize
        }
