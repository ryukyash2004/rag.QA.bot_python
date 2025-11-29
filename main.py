from typing import Dict
from utils.config import RAGConfig
from models.embeddings import SimpleEmbedding
from models.chunker import DocumentChunker
from models.vector_store import FAISSVectorStore
from models.llm import LLMInterface
from pipeline.ingestion import IngestionPipeline
from pipeline.retrieval import RetrievalPipeline
from pipeline.generation import GenerationPipeline



class RAGPipeline:
    """Complete RAG pipeline orchestration"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # Initialize components
        self.embedding_model = SimpleEmbedding(dim=self.config.embedding_dim)
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )
        self.vector_store = FAISSVectorStore(self.config.embedding_dim)
        self.llm = LLMInterface(
            model_name=self.config.llm_model,
            api_key=self.config.openai_api_key
        )
        
        # Initialize pipelines
        self.ingestion = IngestionPipeline(
            self.chunker, 
            self.embedding_model, 
            self.vector_store
        )
        self.retrieval = RetrievalPipeline(
            self.embedding_model, 
            self.vector_store
        )
        self.generation = GenerationPipeline(self.llm)
    
    def add_document(self, text: str, metadata: Dict = None) -> int:
        """Add a document to the knowledge base"""
        return self.ingestion.ingest_document(
            text, 
            metadata, 
            self.config.chunking_method
        )
    
    def query(self, question: str) -> Dict:
        """Ask a question and get an answer"""
        # Retrieve relevant documents
        results = self.retrieval.retrieve(
            question, 
            k=self.config.top_k,
            score_threshold=self.config.score_threshold
        )
        
        context_docs = [doc for doc, score in results]
        
        # Generate answer
        answer = self.generation.generate_answer(
            question, 
            context_docs,
            max_tokens=self.config.max_tokens
        )
        
        return {
            'question': question,
            'answer': answer,
            'sources': results,
            'num_sources': len(results)
        }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return self.vector_store.get_stats()
