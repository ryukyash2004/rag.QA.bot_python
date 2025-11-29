print("🚀 Demo script started!")

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import RAGPipeline
from utils.config import RAGConfig


def run_basic_demo():
    """Basic RAG pipeline demonstration"""
    print("=" * 70)
    print("RAG Q&A Bot - Basic Demo")
    print("=" * 70)
    
    # Initialize pipeline
    config = RAGConfig(chunk_size=400, top_k=2)
    rag = RAGPipeline(config)
    
    # Sample documents
    documents = [
        {
            'text': """Machine learning is a subset of artificial intelligence that 
            enables systems to learn and improve from experience without being explicitly 
            programmed. It focuses on developing computer programs that can access data 
            and use it to learn for themselves.""",
            'metadata': {'source': 'ml_intro.txt', 'topic': 'machine_learning'}
        },
        {
            'text': """Deep learning is part of machine learning based on artificial 
            neural networks. Deep learning architectures such as convolutional neural 
            networks have been applied to computer vision, speech recognition, and 
            natural language processing.""",
            'metadata': {'source': 'deep_learning.txt', 'topic': 'deep_learning'}
        }
    ]
    
    # Ingest documents
    print("\n📚 Ingesting documents...")
    for doc in documents:
        rag.add_document(doc['text'], doc['metadata'])
    
    # Query
    print("\n" + "=" * 70)
    print("Q&A Session")
    print("=" * 70)
    
    queries = [
        "What is machine learning?",
        "What are deep learning architectures used for?"
    ]
    
    for query in queries:
        print(f"\n❓ {query}")
        result = rag.query(query)
        print(f"💡 {result['answer']}")
        print(f"📖 Used {result['num_sources']} sources")
    
    # Stats
    print("\n" + "=" * 70)
    print("📊 Pipeline Statistics")
    print("=" * 70)
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    print("\n✅ Demo Complete!")


def run_advanced_demo():
    """Advanced demo with multiple chunking strategies"""
    print("=" * 70)
    print("RAG Q&A Bot - Advanced Demo (Multiple Strategies)")
    print("=" * 70)
    
    strategies = ['sentences', 'paragraphs', 'fixed']
    
    for strategy in strategies:
        print(f"\n🔧 Testing chunking strategy: {strategy}")
        config = RAGConfig(chunking_method=strategy, chunk_size=300)
        rag = RAGPipeline(config)
        
        doc_text = """Natural Language Processing (NLP) is a branch of AI.
        
        It helps computers understand human language. NLP is used in many applications.
        
        Common tasks include translation, sentiment analysis, and chatbots."""
        
        chunks = rag.add_document(doc_text, {'source': 'nlp.txt'})
        print(f"   Created {chunks} chunks")

if __name__ == "__main__":
    # Run the basic demo by default
    run_basic_demo()