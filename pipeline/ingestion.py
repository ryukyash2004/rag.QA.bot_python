from typing import List, Dict
from models.vector_store import Document

class IngestionPipeline:
    """Handles document ingestion and processing"""
    
    def __init__(self, chunker, embedding_model, vector_store):
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def ingest_document(
        self, 
        text: str, 
        metadata: Dict = None,
        chunking_method: str = 'sentences'
    ) -> int:
        """Process and ingest a document"""
        # Step 1: Chunk document
        if chunking_method == 'sentences':
            chunks = self.chunker.chunk_by_sentences(text)
        elif chunking_method == 'paragraphs':
            chunks = self.chunker.chunk_by_paragraphs(text)
        else:
            chunks = self.chunker.chunk_by_fixed_size(text)
        
        print(f"📄 Created {len(chunks)} chunks")
        
        # Step 2: Create Document objects
        docs = [
            Document(content=chunk, metadata=metadata or {}) 
            for chunk in chunks
        ]
        
        # Step 3: Generate embeddings
        embeddings = self.embedding_model.embed_batch(chunks)
        
        # Step 4: Store in vector database
        self.vector_store.add_documents(docs, embeddings)
        print(f"✅ Ingested {len(chunks)} chunks into vector store")
        
        return len(chunks)
    
    def ingest_multiple(self, documents: List[Dict]) -> Dict:
        """Ingest multiple documents"""
        stats = {'total_chunks': 0, 'documents_processed': 0}
        
        for doc in documents:
            chunks = self.ingest_document(
                doc['text'],
                doc.get('metadata', {}),
                doc.get('chunking_method', 'sentences')
            )
            stats['total_chunks'] += chunks
            stats['documents_processed'] += 1
        
        return stats

