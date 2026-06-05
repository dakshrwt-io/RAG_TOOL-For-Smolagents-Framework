🤖 Modular RAG Tool with Chroma & LLM Reranking

A production-ready, modular Retrieval-Augmented Generation (RAG) system with:
- **Local chunking** (no LLM cost for splitting)
- **OpenAI embeddings** for semantic search
- **Chroma** for persistent vector storage
- **LLM-based reranking** for improved relevance

Built with clean separation of concerns — each module handles one responsibility.

________________________________________
⚙️ Requirements

Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

Or install core dependencies manually:

```bash
pip install chromadb openai litellm pydantic python-dotenv
```

Core dependencies:
- `chromadb>=0.4.0` — Vector database
- `openai>=1.0.0` — OpenAI embeddings & LLM API
- `litellm>=1.0.0` — LLM wrapper for reranking
- `pydantic>=2.0.0` — Data validation
- `python-dotenv>=1.0.0` — Environment variable management

________________________________________
🔑 Environment Variables

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your-openai-api-key
```

________________________________________
▶️ Quick Start

**1. Upload & Embed Documents**

```python
from rag_tool import upload_doc

# Load documents from directory or file paths
result = upload_doc(
    input_paths=["path/to/docs/folder"],
    average_chunk_size=500,      # Characters per chunk
    chunk_overlap=None,           # Auto-calculated as 25% of chunk_size
    embedding_model="text-embedding-3-large",
    reset_collection=True         # Delete old data
)

print(result)
# {
#   "documents": 5,
#   "chunks": 45,
#   "collection_name": "docs",
#   "db_path": "chroma_db",
#   "stored": 45
# }
```

**2. Retrieve & Rerank**

```python
from rag_tool import retrieve

# Query the vector store and rerank results
results = retrieve(
    query="What is the main topic?",
    retrieval_k=8,              # Number of initial results
    rerank=True                 # Enable LLM reranking
)

for chunk in results:
    print(f"Source: {chunk.metadata['source']}")
    print(f"Content: {chunk.page_content}\n")
```

________________________________________
📁 Project Structure

```
rag_tool/
├── __init__.py          # Public API (upload_doc, retrieve)
├── config.py            # Default constants
├── models.py            # Pydantic models (Result, RankOrder)
├── documents.py         # File loading utilities
├── chunking.py          # Local text splitting with overlap
├── embeddings.py        # OpenAI embedding operations
├── storage.py           # Chroma DB add/query
└── reranking.py         # LLM-based reranking
```

Each module has a single responsibility for easy testing and maintenance.

________________________________________
🔧 Advanced Usage

**Custom chunk sizes:**
```python
upload_doc(["docs/"], average_chunk_size=1000, chunk_overlap=200)
```

**Disable reranking (faster retrieval):**
```python
retrieve("query", rerank=False)  # Skip LLM reranking
```

**Custom OpenAI client:**
```python
from openai import OpenAI
client = OpenAI(api_key="your-key")
upload_doc(["docs/"], openai_client=client)
retrieve("query", openai_client=client)
```

**Import internal modules:**
```python
from rag_tool.documents import load_documents
from rag_tool.chunking import create_chunks_local
from rag_tool.storage import query_collection
```

________________________________________
🛡️ Notes

- **No LLM cost for chunking** — Uses local character-based splitting with overlap
- **Flexible retrieval** — Enable/disable reranking per query
- **Persistent storage** — Chroma DB stored locally in `chroma_db/` folder
- **Modular design** — Import and use individual modules
- **Type hints** — Full Pydantic validation for Results

________________________________________
📝 API Reference

### `upload_doc(input_paths, **kwargs) → dict`
Load documents, chunk, embed, and store in Chroma.

### `retrieve(query, **kwargs) → List[Result]`
Retrieve chunks from Chroma and optionally rerank.



________________________________________
📜 License

MIT License — free to use, modify, and share.
