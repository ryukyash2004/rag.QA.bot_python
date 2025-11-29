from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunking_method: str = 'sentences'
    
    # Embeddings
    embedding_dim: int = 384
    embedding_model: str = 'sentence-transformers'
    
    # Retrieval
    top_k: int = 3
    score_threshold: float = 0.0
    
    # Generation
    llm_model: str = 'gpt-3.5-turbo'
    max_tokens: int = 500
    
    # API Keys
    openai_api_key: str = None