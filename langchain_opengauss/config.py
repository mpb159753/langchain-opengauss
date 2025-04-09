from __future__ import annotations

from enum import Enum
from typing import Dict, ClassVar

from pydantic import BaseModel, field_validator, Field, ConfigDict


class IndexType(str, Enum):
    """Supported index types for vector search"""
    IVFFLAT = "IVFFLAT"  # Inverted file with flat compression
    HNSW = "HNSW"  # Hierarchical Navigable Small World graphs


class DistanceStrategy(str, Enum):
    """Supported distance metrics for vector similarity search"""
    EUCLIDEAN = "euclidean"  # EUCLIDEAN L2 distance, ideal for geometric comparisons
    NEGATIVE_INNER_PRODUCT = "negative_inner_product"  # Dot product, used for normalized vectors
    COSINE = "cosine"  # Angular distance, ideal for text embeddings
    MANHATTAN = "manhattan"  # manhattan L1 distance, useful for sparse high-dimensional data
    # HAMMING = "hamming"  # Hamming distance, for bit vectors
    # JACCARD = "jaccard"  # Jaccard distance, for bit vectors


class VectorType(str, Enum):
    """Supported vector data types in openGauss"""
    vector = "vector"  # Floating-point vectors for general-purpose embeddings
    # bit = "bit"  # Binary/bit vectors
    # sparsevec = "sparsevec"  # Sparse vectors for efficient storage of mostly-zero embeddings


class OpenGaussSettings(BaseModel):
    """Configuration settings for openGauss database connection and vector search capabilities"""

    # Dimension limits constants
    DIMENSION_LIMITS: ClassVar[Dict[VectorType, int]] = {
        VectorType.vector: 2000,
        # VectorType.bit: 64000,
        # VectorType.sparsevec: 1000
    }

    # Operator mapping for SQL queries
    OPERATOR_MAP: ClassVar[Dict[DistanceStrategy, str]] = {
        DistanceStrategy.EUCLIDEAN: "<->",
        DistanceStrategy.NEGATIVE_INNER_PRODUCT: "<#>",
        DistanceStrategy.COSINE: "<=>",
        DistanceStrategy.MANHATTAN: "<+>",
        # DistanceStrategy.HAMMING: "<=>",
        # DistanceStrategy.JACCARD: "<=>",
    }

    INDEX_OPERATOR_MAP: ClassVar[Dict[VectorType, Dict[DistanceStrategy, str]]] = {
        VectorType.vector: {
            DistanceStrategy.EUCLIDEAN: "vector_l2_ops",
            DistanceStrategy.NEGATIVE_INNER_PRODUCT: "vector_ip_ops",
            DistanceStrategy.COSINE: "vector_cosine_ops",
            DistanceStrategy.MANHATTAN: "vector_l1_ops"
        },
        # VectorType.bit: {
        #     DistanceStrategy.HAMMING: "bit_hamming_ops",
        #     DistanceStrategy.JACCARD: "bit_jaccard_ops"
        # },
        # VectorType.sparsevec: {
        #     DistanceStrategy.EUCLIDEAN: "sparsevec_l2_ops",
        #     DistanceStrategy.NEGATIVE_INNER_PRODUCT: "sparsevec_ip_ops",
        #     DistanceStrategy.COSINE: "sparsevec_cosine_ops",
        #     DistanceStrategy.MANHATTAN: "sparsevec_l1_ops"
        # }
    }

    # Database connection settings
    host: str = Field(
        default="localhost",
        description="Database server hostname or IP address"
    )
    port: int = Field(
        default=5432,
        description="Database server port number (default: 5432)"
    )
    user: str = Field(
        default="gaussdb",
        description="Database username for authentication"
    )
    password: str = Field(
        default="MyStrongPass@123",
        description="Database password for authentication"
    )
    database: str = Field(
        default="postgres",
        description="Target database name to connect to"
    )
    min_connections: int = Field(
        default=1,
        description="Minimum number of connections to maintain in the connection pool"
    )
    max_connections: int = Field(
        default=5,
        description="Maximum number of connections allowed in the connection pool"
    )
    table_name: str = Field(
        default="langchain_docs",
        description="Name of the table for storing vector data and metadata"
    )

    # Vector index settings
    index_type: IndexType = Field(
        default=IndexType.HNSW,
        description="Vector index algorithm type. Options: HNSW or IVFFLAT\nDefault is HNSW."
    )

    vector_type: VectorType = Field(
        default=VectorType.vector,
        description="Type of vector representation to use. Vector (float-based, max 2000 dimensions), "
                    "bitvec (binary/bit-based, max 64000 dimensions), sparsevec (sparse representation, "
                    "max 1000 non-zero elements for efficiency in high-dimensional sparse spaces)."
                    "\nDefault is Vector."
    )

    distance_strategy: DistanceStrategy = Field(
        default=DistanceStrategy.COSINE,
        description="Vector similarity metric to use for retrieval. Options: euclidean (L2 distance), "
                    "cosine (angular distance, ideal for text embeddings), manhattan (L1 distance for sparse data), "
                    "negative_inner_product (dot product for normalized vectors).\n Default is cosine."
    )

    embedding_dimension: int = Field(
        default=1536,
        ge=1,
        description="Dimensionality of the vector embeddings. "
                    "The maximum allowed dimensions depend on the vector_type selection."
                    "\nDefault is 1536 for OpenAI."
    )

    model_config = ConfigDict(frozen=False, extra="forbid")

    @field_validator('index_type', mode='before')
    def validate_index_type(cls, v):
        if isinstance(v, str):
            v = v.upper()
            if v not in IndexType.__members__:
                raise ValueError(f"Unsupported index type. Valid options: {', '.join(IndexType.__members__.keys())}")
            return IndexType(v)
        return v

    @field_validator('vector_type', mode='before')
    def validate_vector_type(cls, v):
        if isinstance(v, str):
            v = v.lower()
            if v not in VectorType.__members__:
                raise ValueError(f"Unsupported vector type. Valid options: {', '.join(VectorType)}")
            return VectorType(v)
        return v

    @field_validator('distance_strategy', mode='after')
    def validate_distance_strategy(cls, v, info):
        if isinstance(v, str):
            v = v.lower()

            if v not in [item.value for item in DistanceStrategy]:
                raise ValueError(
                    f"Invalid distance strategy. Valid options: {', '.join(e.value for e in DistanceStrategy)}")

            # Check compatibility with vector type
            vector_type = info.data.get('vector_type', VectorType.vector)
            if vector_type in cls.INDEX_OPERATOR_MAP:
                valid_strategies = [s for s in DistanceStrategy if s in cls.INDEX_OPERATOR_MAP[vector_type]]
                if DistanceStrategy(v) not in valid_strategies:
                    raise ValueError(
                        f"Distance strategy '{v}' is not compatible with vector type '{vector_type.value}'. "
                        f"Valid options: {', '.join(s.value for s in valid_strategies)}")

            return DistanceStrategy(v)
        return v

    @field_validator('embedding_dimension')
    def validate_dimension(cls, v, info):
        vector_type = info.data.get('vector_type', VectorType.vector)
        max_dim = cls.DIMENSION_LIMITS[vector_type]

        if v > max_dim:
            type_descriptions = {
                VectorType.vector: "float vectors",
                # VectorType.bit: "bit vectors",
                # VectorType.sparsevec: "non-zero elements in sparse vectors"
            }
            raise ValueError(f"{type_descriptions[vector_type]} support maximum {max_dim} dimensions")
        return v

    @property
    def operator(self) -> str:
        """Returns the appropriate SQL operator for the selected distance strategy"""
        return self.OPERATOR_MAP[self.distance_strategy]

    @property
    def index_operator(self) -> str:
        """Returns the appropriate SQL index operator for the selected vector type and distance strategy"""
        try:
            return self.INDEX_OPERATOR_MAP[self.vector_type][self.distance_strategy]
        except KeyError:
            raise ValueError(
                f"Unsupported combination of vector type '{self.vector_type.value}' and "
                f"distance strategy '{self.distance_strategy.value}'")
