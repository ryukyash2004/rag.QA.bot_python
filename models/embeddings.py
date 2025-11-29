import numpy as np
from typing import List

class SimpleEmbedding:
    """Embedding model wrapper - replace with actual model in production"""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        np.random.seed(42)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        # Production: Use sentence-transformers, OpenAI, etc.
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # return model.encode(text)
        return np.random.randn(self.dim)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        return np.array([self.embed_text(t) for t in texts])