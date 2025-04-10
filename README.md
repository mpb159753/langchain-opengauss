# openGauss Vector Store for LangChain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[openGauss](https://opengauss.org/zh/) integration for LangChain providing scalable vector storage and search capabilities, powered by openGauss.

## Features

- 🚀 **Multi-Index Support** - HNSW and IVFFLAT vector indexing algorithms
- 📐 **Multiple Distance Metrics** - EUCLIDEAN/COSINE/MANHATTAN/NEGATIVE_INNER_PRODUCT
- 🔧 **Auto-Schema Management** - Automatic table creation and validation
- 🧮 **Dimension Validation** - Type-safe dimension constraints for different vector types
- 🛡️ **ACID Compliance** - Transaction-safe operations with connection pooling
- 🔀 **Hybrid Search** - Combine vector similarity with metadata filtering

## Installation

```bash
pip install langchain-opengauss
```

**Prerequisites**:

- openGauss >= 7.0.0
- Python 3.8+
- psycopg2-binary

## Quick Start

### 1. Start openGauss Container

```bash
docker run --name opengauss \
  --privileged=true \
  -d \
  -e GS_PASSWORD=MyStrongPass@123 \
  -p 8888:5432 \
  opengauss/opengauss-server:latest
```

### 2. Basic Usage

```python
from langchain_opengauss import OpenGauss, OpenGaussSettings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Configuration with validation
config = OpenGaussSettings(
    table_name="research_papers",
    embedding_dimension=1536,
    index_type="HNSW",
    distance_strategy="COSINE",
)

# Initialize with OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = OpenGauss(embedding=embeddings, config=config)

# Insert documents
docs = [
    Document(page_content="Quantum computing basics", metadata={"field": "physics"}),
    Document(page_content="Neural network architectures", metadata={"field": "ai"})
]
vector_store.add_documents(docs)

# Semantic search
results = vector_store.similarity_search("deep learning models", k=1)
print(f"Found {len(results)} relevant documents")
```

## Configuration Guide

### Connection Settings

| Parameter           | Default                 | Description                                            |
|---------------------|-------------------------|--------------------------------------------------------|
| `host`              | localhost               | Database server address                                |
| `port`              | 8888                    | Database connection port                               |
| `user`              | gaussdb                 | Database username                                      |
| `password`          | -                       | Complex password string                                |
| `database`          | postgres                | Default database name                                  |
| `min_connections`   | 1                       | Connection pool minimum size                           |
| `max_connections`   | 5                       | Connection pool maximum size                           |
| `table_name`        | langchain_docs          | Name of the table for storing vector data and metadata |
| `index_type`        | IndexType.HNSW          |Vector index algorithm type. Options: HNSW or IVFFLAT\nDefault is HNSW.|
| `vector_type`       | VectorType.vector       |Type of vector representation to use. Default is Vector.|
| `distance_strategy` | DistanceStrategy.COSINE |Vector similarity metric to use for retrieval. Options: euclidean (L2 distance), cosine (angular distance, ideal for text embeddings), manhattan (L1 distance for sparse data), negative_inner_product (dot product for normalized vectors).\n Default is cosine.|
|`embedding_dimension`| 1536                    |Dimensionality of the vector embeddings.|

### Vector Configuration

```python
class OpenGaussSettings(BaseModel):
    index_type: IndexType = IndexType.HNSW  # HNSW or IVFFLAT
    vector_type: VectorType = VectorType.vector  # Currently supports float vectors
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE
    embedding_dimension: int = 1536  # Max 2000 for vector type
```

#### Supported Combinations

| Vector Type | Dimensions | Index Types  | Supported Distance Strategies         |
|-------------|------------|--------------|---------------------------------------|
| vector      | ≤2000      | HNSW/IVFFLAT | COSINE/EUCLIDEAN/MANHATTAN/INNER_PROD |

## Advanced Usage

### Hybrid Search with Metadata

```python
# Filter by metadata with vector search
results = vector_store.similarity_search(
    query="machine learning",
    k=3,
    filter={"publish_year": 2023, "category": "research"},
)
```

### Index Management

```python
# Create optimized HNSW index
vector_store.create_hnsw_index(
    m=24,  # Number of bi-directional links
    ef_construction=128,  # Search scope during build
    ef=64,  # Search scope during queries
)


```

## API Reference

### Core Methods
| Method                         | Description                                   |
|--------------------------------|-----------------------------------------------|
| `add_documents`                | Insert documents with automatic embedding     |
| `similarity_search `           | Basic vector similarity search                |
| `similarity_search_with_score` | Return (document, similarity_score) tuples   |
| `delete`                       | Remove documents by ID list                  |
| `drop_table`                   | Delete entire collection                     |


## Performance Tips

### 1. **Index Tuning**

#### HNSW Index Optimization

- `m` (max connections per layer)
    - **Default**: 16
    - **Range**: 2~100
    - Tradeoff: Higher values improve recall but increase index build time and memory usage

- `ef_construction` (construction search scope)
    - **Default**: 64
    - **Range**: 4~1000 (must ≥ 2*m)

```python
# Example HNSW configuration
vector_store.create_hnsw_index(
    m=16,  # Balance between recall and performance
    ef_construction=64,  # Ensure >2*m (48) and >ef_search
)
```

#### IVFFLAT Index Optimization

- `lists`
    - **Calculation**:
      ```python
      # Recommended formula
      lists = min(int(math.sqrt(total_rows)) if total_rows > 1e6 else int(total_rows / 1000),
           2000,  # openGauss maximum
      )
      ```
    - **Adjustment Guide**:
        - Start with 1000 lists for 1M vectors
        - 2000 lists for 10M+ vectors
        - Monitor recall rate and adjust

### 2. **Connection Pooling**

   ```python
   OpenGaussSettings(
    min_connections=3,
    max_connections=20
)
   ```

## Limitations

- Vector type `bit` and `sparsevec` currently under development

