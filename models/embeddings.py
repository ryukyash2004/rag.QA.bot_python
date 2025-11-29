from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class RealEmbedding:
    """Real embedding model using sentence-transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded! Embedding dimension: {self.dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)