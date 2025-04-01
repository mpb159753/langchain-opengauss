# OpenGauss Vector Store for LangChain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OpenGauss integration for LangChain providing scalable vector storage , powered by openGauss.

## Features

- 🚀 **High-performance vector search** using  [HNSW](https://docs.opengauss.org/zh/docs/7.0.0-RC1/docs/SQLReference/%E5%90%91%E9%87%8F%E7%B4%A2%E5%BC%95.html##IVFFlat) indexing
- 🔧 **Auto-schema management** with table creation/initialization
- 🛡️ **ACID-compliant** storage with openGauss
- 📦 **Batched operations** for efficient document handling
- 🔍 **Hybrid search** combining vector similarity and metadata filtering
- 🧩 **LangChain compatible** API design

## Installation

```bash
pip install langchain-opengauss
```

**Prerequisites**:
- Running openGauss instance (Docker recommended)
- Python 3.8+

## Quick Start

### 1. Start openGauss Docker Container
```bash
docker run --name opengauss \
  --privileged=true \
  -d \
  -e GS_PASSWORD=MyStrongPass@123 \
  -p 5432:5432 \
  opengauss/opengauss-server:latest
```

### 2. Basic Usage
```python
from langchain_opengauss import OpenGauss, OpenGaussSettings
from langchain_openai import OpenAIEmbeddings

# Configuration
config = OpenGaussSettings(
    user="gaussdb",
    password="MyStrongPass@123",
    table_name="doc_store"
)

# Initialize vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = OpenGauss(embedding=embeddings, config=config)

# Add documents
docs = [
    Document(page_content="Quantum computing basics", metadata={"topic": "physics"}),
    Document(page_content="Advanced machine learning", metadata={"topic": "ai"})
]
vector_store.add_documents(docs)

# Semantic search
results = vector_store.similarity_search("computer science", k=2)
for doc in results:
    print(f"• {doc.page_content} [{doc.metadata}]")
```

## Configuration

### OpenGaussSettings Parameters
| Parameter           | Default       | Description                          |
|---------------------|---------------|--------------------------------------|
| `host`              | localhost     | Database host address                |
| `port`              | 5432          | Database port                        |
| `user`              | gaussdb       | Database username                    |
| `password`          | -             | Database password                    |
| `database`          | postgres      | Default database name                |
| `table_name`        | langchain_docs| Collection table name                |
| `embedding_dimension` | 1536        | Vector dimension (OpenAI default)    |
| `min_connections`   | 1             | Connection pool minimum              |
| `max_connections`   | 5             | Connection pool maximum              |

## Advanced Usage

### Hybrid Search with Filters
```python
results = vector_store.similarity_search(
    query="neural networks",
    k=5,
    filter={"category": "machine-learning"}
)
```

### ID Management Strategies
```python
# Custom IDs
vector_store.add_documents(docs, ids=["doc1", "doc2"])

# Auto-generated UUIDs
vector_store.add_documents(docs)  # Generates UUIDs automatically
```

### Performance Tuning
```python
# Batch insert with 100 documents per transaction
vector_store.add_documents(large_docs, batch_size=100)

# Adjust HNSW index parameters
vector_store.create_hnsw_index(m=24, ef_construction=128)
```

## API Reference

### Core Methods
- `add_documents()`: Insert documents with automatic embedding
- `similarity_search()`: Basic vector similarity search
- `similarity_search_with_score()`: Search with similarity scores
- `delete()`: Remove documents by ID
- `drop_table()`: Delete entire collection


---

**Note**: Requires openGauss 7.0+ with vector extension enabled.
