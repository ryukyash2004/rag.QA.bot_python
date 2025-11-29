from typing import List, Tuple
import numpy as np
from models.vector_store import Document

class RetrievalPipeline:
    """Handles semantic search and retrieval"""
    
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def retrieve(
        self, 
        query: str, 
        k: int = 3,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for query"""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search vector store
        results = self.vector_store.similarity_search(query_embedding, k=k)
        
        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in results 
            if score >= score_threshold
        ]
        
        return filtered_results
    
    def retrieve_with_reranking(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve and rerank results (placeholder for advanced retrieval)"""
        # Initial retrieval
        results = self.retrieve(query, k=k*2)
        
        # Reranking logic would go here
        # For now, just return top-k
        return [doc for doc, score in results[:k]]

