import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class Document:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict = field(default_factory=dict)
    embedding: np.ndarray = None


class FAISSVectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None
    
    def add_documents(self, docs: List[Document], embeddings: np.ndarray):
        """Add documents with their embeddings"""
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
        
        self.documents.extend(docs)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Find top-k most similar documents with scores"""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Compute cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        similarities = np.dot(doc_norms, query_norm)
        
        # Get top-k indices
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        results = [
            (self.documents[i], float(similarities[i])) 
            for i in top_k_idx
        ]
        
        return results
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            'total_documents': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'total_embeddings': len(self.embeddings) if self.embeddings is not None else 0
        }

