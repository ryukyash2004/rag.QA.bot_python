# 🤖 RAG Q&A Bot - Document Question Answering System

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline built with Python, featuring semantic search, document chunking, and modular architecture.

## 🎯 What is RAG?

RAG (Retrieval-Augmented Generation) combines:
- **Retrieval**: Finding relevant information from documents using semantic search
- **Generation**: Using LLMs to generate accurate answers based on retrieved context

This prevents hallucinations and grounds AI responses in your actual documents.

## ✨ Features

- 📄 **Document Chunking**: Multiple strategies (sentences, paragraphs, fixed-size)
- 🧠 **Semantic Search**: FAISS vector database with cosine similarity
- 🔍 **Context Retrieval**: Find most relevant document chunks for queries
- 🎯 **Modular Architecture**: Easy to extend and customize
- ⚡ **Production Ready**: Clean separation of concerns, typed interfaces

## 🏗️ Architecture

```
Document → Chunking → Embeddings → Vector Store
                                        ↓
Query → Embed Query → Semantic Search → Top-K Chunks → LLM → Answer
```

### Project Structure

```
rag-qa-bot/
├── models/              # Core components
│   ├── embeddings.py    # Embedding models (mock & real)
│   ├── chunker.py       # Document chunking strategies
│   ├── vector_store.py  # FAISS vector database
│   └── llm.py          # LLM interface
├── pipeline/            # RAG workflow
│   ├── ingestion.py     # Document processing
│   ├── retrieval.py     # Semantic search
│   └── generation.py    # Answer generation
├── utils/
│   └── config.py        # Configuration management
├── examples/
│   └── demo.py          # Demo scripts
├── main.py              # RAG pipeline orchestration
├── app.py               # Web API (optional)
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-qa-bot.git
cd rag-qa-bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
python examples/demo.py
```

## 📖 Usage

### Basic Usage

```python
from main import RAGPipeline
from utils.config import RAGConfig

# Initialize pipeline
config = RAGConfig(chunk_size=500, top_k=3)
rag = RAGPipeline(config)

# Add documents
rag.add_document(
    text="Your document content here...",
    metadata={'source': 'doc1.txt'}
)

# Ask questions
result = rag.query("What is machine learning?")
print(result['answer'])
print(f"Sources: {result['num_sources']}")
```

### Advanced Configuration

```python
from utils.config import RAGConfig

config = RAGConfig(
    chunk_size=500,           # Characters per chunk
    chunk_overlap=50,         # Overlap between chunks
    chunking_method='sentences',  # 'sentences', 'paragraphs', or 'fixed'
    top_k=3,                  # Number of chunks to retrieve
    embedding_dim=384,        # Embedding dimension
    max_tokens=500           # Max tokens for LLM response
)

rag = RAGPipeline(config)
```

### Different Chunking Strategies

```python
# Sentence-based chunking (default)
config = RAGConfig(chunking_method='sentences', chunk_size=500)

# Paragraph-based chunking
config = RAGConfig(chunking_method='paragraphs', chunk_size=800)

# Fixed-size chunking
config = RAGConfig(chunking_method='fixed', chunk_size=400)
```

## 🔧 Upgrading to Production

### Use Real Embeddings (Sentence Transformers)

Update `models/embeddings.py`:

```python
from sentence_transformers import SentenceTransformer

class RealEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str):
        return self.model.encode(text)
    
    def embed_batch(self, texts: list):
        return self.model.encode(texts)
```

Then in `main.py`:
```python
from models.embeddings import RealEmbedding
# ...
self.embedding_model = RealEmbedding()
```

### Use Real LLM (OpenAI)

```bash
pip install openai
```

Update `models/llm.py`:

```python
from openai import OpenAI

class OpenAILLM:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int = 500):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
```

Set your API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key"

# Linux/Mac
export OPENAI_API_KEY="your-api-key"
```

## 📊 Key Concepts Demonstrated

- ✅ **Document Chunking**: Breaking documents into manageable pieces
- ✅ **Embeddings**: Converting text to vector representations
- ✅ **Semantic Search**: Finding similar content using cosine similarity
- ✅ **Vector Database**: FAISS for efficient similarity search
- ✅ **Context Window Management**: Retrieving relevant chunks for LLM
- ✅ **Prompt Engineering**: Constructing effective prompts with context

## 🧪 Testing

Run basic demo:
```bash
python examples/demo.py
```

Run advanced demo (tests multiple chunking strategies):
```python
# In examples/demo.py, uncomment:
# run_advanced_demo()
```

## 📝 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 500 | Characters per chunk |
| `chunk_overlap` | 50 | Overlap between chunks |
| `chunking_method` | 'sentences' | Chunking strategy |
| `top_k` | 3 | Number of chunks to retrieve |
| `embedding_dim` | 384 | Embedding vector dimension |
| `score_threshold` | 0.0 | Minimum similarity score |
| `max_tokens` | 500 | Max LLM response tokens |

## 🛠️ Tech Stack

- **Python 3.8+**
- **NumPy**: Vector operations
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings (optional)
- **OpenAI/Anthropic**: LLM APIs (optional)

## 📚 Use Cases

- 📄 **Document Q&A**: Answer questions from company documents
- 📖 **Knowledge Base**: Search through documentation
- 🎓 **Study Assistant**: Query textbooks and notes
- 💼 **Legal/Medical**: Search through specialized documents
- 📰 **News Analysis**: Query large article collections

## 🔮 Future Enhancements

- [ ] Add support for PDF/Word documents
- [ ] Implement query expansion
- [ ] Add re-ranking of results
- [ ] Cache embeddings for faster retrieval
- [ ] Add web interface (Streamlit/Gradio)
- [ ] Support for multiple document formats
- [ ] Hybrid search (keyword + semantic)
- [ ] Add conversation memory

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

Built to demonstrate RAG fundamentals for production LLM applications.

Inspired by:
- LangChain
- LlamaIndex
- OpenAI RAG best practices

## 📞 Contact

Questions or feedback? Open an issue or reach out!

---

⭐ **Star this repo if you found it helpful!**