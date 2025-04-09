from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import psycopg2
import psycopg2.extras
import psycopg2.pool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_opengauss.config import OpenGaussSettings, IndexType


class OpenGauss(VectorStore):
    """openGauss vector store integration.

    Setup:
        Install ``opengauss`` and run the docker container.

        .. code-block:: bash

            pip install psycopg2-binary
            # The password must include all of the following: uppercase letters, lowercase letters, numbers, and special characters.
            docker run --name opengauss --privileged=true -d -e GS_PASSWORD=Test@123456 -p 5432:5432 opengauss/opengauss-server:latest

    Key init args — core params:
        embedding: Embeddings
            Embedding function to use
        config: OpenGaussSettings
            Configuration settings for openGauss connection and indexing

    Key init args — connection params:
        config contains:
            host: str = "localhost"
            port: int = 5432
            user: str = "gaussdb"
            password: str = "Test@123456"
            database: str = "postgres"
            table_name: str = "langchain_docs"
            embedding_dimension: int = 1536  # Default for OpenAI embeddings

    Instantiate:
        .. code-block:: python

            from langchain_opengauss import OpenGauss, OpenGaussSettings
            from langchain_openai import OpenAIEmbeddings

            config = OpenGaussSettings(
                host="localhost",
                port=5432,
                user="gaussdb",
                password="Test@123456",
                database="postgres",
                table_name="my_docs"
            )

            vector_store = OpenGauss(
                embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
                config=config
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            # When explicitly providing 'ids' parameter:
            #    - Uses provided IDs and ignores any existing 'id' in Document objects
            #    - IDs list length MUST match documents list length
            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(", id="this id will not working")

            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=[document_1, document_2, document_3], ids=ids)


            # When NOT providing 'ids' parameter:
            #    - Uses Document's 'id' field if present
            #    - Auto-generates UUIDs for documents without IDs
            document_4 = Document(page_content="xxxx", metadata={"xxx": "xxx"}, id="4")
            document_5 = Document(page_content="ssss", metadata={"xxx": "xxx"})
            vector_store.add_documents(documents=[document_4, document_5])

            # When documents have no IDs and no 'ids' parameter provided:
            #    - Generates UUIDs for all documents
            document_6 = Document(page_content="xxxx", metadata={"xxx": "xxx"})
            document_7 = Document(page_content="ssss", metadata={"xxx": "xxx"})
            vector_store.add_documents(documents=[document_6, document_7])

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'bar': 'baz'}]

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'bar': 'baz'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:.3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.499] foo [{'baz': 'bar'}]

    Advanced:
        Table management:
            .. code-block:: python

                # Create new collection/table
                vector_store._create_table()

                # Drop entire table
                vector_store.drop_table()
        """

    def __init__(self, embedding: Embeddings, config: OpenGaussSettings):
        """Initialize OpenGauss vector store.

        Args:
            embedding: Embedding function to use
            config: OpenGauss configuration settings
        """
        self.embedding_function = embedding
        self.config = config
        self._init_pool()
        self._create_table()

    def _init_pool(self) -> None:
        """Initialize connection pool"""
        self.pool = psycopg2.pool.SimpleConnectionPool(
            self.config.min_connections,
            self.config.max_connections,
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
        )

    @contextmanager
    def _get_cursor(self):
        """Get database cursor with context management"""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        finally:
            self.pool.putconn(conn)

    def _index_exists(self, index_name: str) -> bool:
        """Check if index exists in the database"""
        check_sql = """
            SELECT EXISTS (
                SELECT 1 
                FROM pg_indexes 
                WHERE tablename = %s AND indexname = %s
            );
        """
        with self._get_cursor() as cur:
            cur.execute(check_sql, (self.config.table_name, index_name))
            return cur.fetchone()[0]

    def _create_table(self) -> None:
        """Create table with index if not exists"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.config.table_name} (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            metadata JSONB,
            embedding {self.config.vector_type.name}({self.config.embedding_dimension}) NOT NULL
        );
        """
        with self._get_cursor() as cur:
            cur.execute(create_table_sql)

        index_name = f"idx_{self.config.table_name}_embedding"
        if not self._index_exists(index_name):
            if self.config.index_type == IndexType.HNSW:
                self.create_hnsw_index(index_name=index_name, drop_if_exists=False)
            else:
                self.create_ivfflat_index(index_name=index_name, drop_if_exists=False)

    def create_hnsw_index(self, index_name=None, m=16, ef_construction=64, drop_if_exists=True) -> None:
        if self.config.index_type != IndexType.HNSW:
            raise ValueError("HNSW index requires index_type to be set to HNSW")
        if index_name is None:
            index_name = f"idx_{self.config.table_name}_embedding"
        if not drop_if_exists:
            drop_sql = f"DROP INDEX IF EXISTS {index_name};"
            with self._get_cursor() as cur:
                cur.execute(drop_sql)

        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {self.config.table_name} 
            USING {self.config.index_type.name} (embedding {self.config.index_operator})
            WITH (m = {m}, ef_construction = {ef_construction});
        """
        with self._get_cursor() as cur:
            cur.execute(create_index_sql)

    def create_ivfflat_index(self, index_name=None, lists=200, drop_if_exists=True) -> None:
        if self.config.index_type != IndexType.IVFFLAT:
            raise ValueError("IVFFLAT index requires index_type to be set to IVFFLAT")
        if index_name is None:
            index_name = f"idx_{self.config.table_name}_embedding"
        if not drop_if_exists:
            drop_sql = f"DROP INDEX IF EXISTS {index_name};"
            with self._get_cursor() as cur:
                cur.execute(drop_sql)

        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {self.config.table_name} 
            USING {self.config.index_type.name} (embedding {self.config.index_operator})
            WITH (lists = {lists});
        """
        with self._get_cursor() as cur:
            cur.execute(create_index_sql)

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[list[dict]] = None,
            *,
            ids: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store

        Args:
            texts: Iterable of strings to add
            metadatas: Optional list of metadata dicts
            ids: Optional list of ids
            kwargs: Additional arguments

        Returns:
            List of ids for the added texts

        Raises:
            ValueError: If metadatas or ids length doesn't match texts
        """
        texts = list(texts)
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        elif len(metadatas) != len(texts):
            raise ValueError(f"Mismatched metadatas: {len(metadatas)} vs {len(texts)} texts")

        if ids is None:
            generated_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        elif len(ids) != len(texts):
            raise ValueError(f"Mismatched IDs: {len(ids)} vs {len(texts)} texts")
        else:
            # Create a copy to avoid issues with list object modifications
            generated_ids = list(ids)

        embeddings = self.embedding_function.embed_documents(texts)
        records = []

        for doc_id, text, metadata, embedding in zip(generated_ids, texts, metadatas, embeddings):
            records.append((
                doc_id,
                text,
                json.dumps(metadata),
                json.dumps(embedding)
            ))

        insert_sql = (f"""INSERT INTO {self.config.table_name} (id, text, metadata, embedding)
                            VALUES %s
                            ON DUPLICATE KEY UPDATE
                              text = VALUES(text),
                              metadata = VALUES(metadata),
                              embedding = VALUES(embedding);
                            """)
        with self._get_cursor() as cur:
            psycopg2.extras.execute_values(cur, insert_sql, records, template=None, page_size=100)

        return generated_ids

    @property
    def embeddings(self) -> Embeddings:
        """Get the embedding function"""
        return self.embedding_function

    def delete(
            self,
            ids: Optional[list[str]] = None,
            **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete. If None, delete all. Default is None.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        with self._get_cursor() as cur:
            if ids is None:
                # 删除全部记录
                query = f"DELETE FROM {self.config.table_name}"
                cur.execute(query)
            else:
                # 删除指定ID记录
                query = f"DELETE FROM {self.config.table_name} WHERE id = ANY(%s)"
                cur.execute(query, (ids,))
            return True

    def get_by_ids(
            self,
            ids: Sequence[str],
    ) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of ids to retrieve.

        Returns:
            List of Documents.
        """
        with self._get_cursor() as cur:
            cur.execute(f"SELECT id, metadata, text FROM {self.config.table_name} WHERE id IN %s", (tuple(ids),))
            docs = []
            for _id, metadata, text in cur:
                docs.append(Document(page_content=text, metadata=metadata, id=_id))
        return docs

    def add_documents(
            self,
            documents: list[Document],
            **kwargs: Any
    ) -> list[str]:
        """Add or update documents in the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            kwargs: Additional keyword arguments.
                if kwargs contains ids and documents contain ids,
                the ids in the kwargs will receive precedence.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: If the number of ids does not match the number of documents.
        """
        metadatas = []
        texts = []

        for doc in documents:
            metadatas.append(doc.metadata)
            texts.append(doc.page_content)
        if 'ids' not in kwargs:
            # Fill missing IDs with UUIDs
            ids = []
            for i, doc in enumerate(documents):
                if hasattr(doc, 'id') and doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))
            kwargs['ids'] = tuple(ids)
        else:
            if len(kwargs['ids']) != len(documents):
                raise ValueError(f"Mismatched IDs: {len(kwargs['ids'])} vs {len(documents)} documents")

        return self.add_texts(texts, metadatas, **kwargs)

    def search(
            self,
            query: str,
            search_type: str,
            **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to query using a specified search type.

        Args:
            query: Input text
            search_type: Type of search to perform. Can be "similarity",
                "mmr", or "similarity_score_threshold".
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of Documents most similar to the query.

        Raises:
            ValueError: If search_type is not supported.
        """
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        elif search_type == "mmr":
            raise ValueError(f"Search type {search_type} is not supported yet.")
        elif search_type == "similarity_score_threshold":
            return self.similarity_score_threshold(query, **kwargs)

    def similarity_score_threshold(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
            threshold=0
    ) -> list[Document]:
        result = self.similarity_search(query, k=k, filter=filter)
        need_docs = []
        for doc, score in result:
            if score > threshold:
                need_docs.append(doc)
        return need_docs

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional metadata filter dict.

        Returns:
            List of Documents most similar to the query.
        """
        return self.similarity_search_by_vector(
            self.embedding_function.embed_query(query), k=k, filter=filter
        )

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Run similarity search with distance.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional metadata filter dict.

        Returns:
            List of Tuples of (doc, similarity_score).
        """
        return self.similarity_search_with_score_by_vector(
            self.embedding_function.embed_query(query),
            k=k,
            filter=filter
        )

    def similarity_search_by_vector(
            self,
            embedding: list[float],
            k: int = 4,
            filter: Optional[dict] = None
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional metadata filter dict.

        Returns:
            List of Documents most similar to the query vector.
        """
        res = self.similarity_search_with_score_by_vector(embedding=embedding, k=k, filter=filter)
        return [doc for doc, _ in res]

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Search for documents similar to embedding vector with scores.

        Args:
            embedding: Embedding vector to search for
            k: Number of documents to return
            filter: Optional metadata filter dict

        Returns:
            List of (Document, score) tuples
        """
        with self._get_cursor() as cur:
            if filter is None:
                sql = f"""SELECT id, metadata, text, embedding {self.config.operator} %s AS distance FROM {self.config.table_name} 
                            ORDER BY distance LIMIT %s"""
                cur.execute(sql, (json.dumps(embedding), k),
                            )
            else:
                sql = f"""SELECT id, metadata, text, embedding {self.config.operator} %s AS distance FROM {self.config.table_name} 
                            WHERE metadata @> %s
                            ORDER BY distance LIMIT %s"""
                cur.execute(sql, (json.dumps(embedding), json.dumps(filter), k), )
            results = []
            for _id, metadata, text, distance in cur:
                results.append((Document(page_content=text, metadata=metadata, id=_id), 1 - distance))
        return results

    @classmethod
    def from_documents(
            cls: type[OpenGauss],
            documents: list[Document],
            embedding: Embeddings,
            config: Optional[OpenGaussSettings] = None,
            **kwargs: Any,
    ) -> OpenGauss:
        """Return VectorStore initialized from documents and embeddings.

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use.
            config: OpenGauss settings configuration.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from documents and embeddings.
        """
        if config is None:
            raise ValueError("OpenGaussSettings config is required")

        metadatas = []
        texts = []

        for doc in documents:
            metadatas.append(doc.metadata)
            texts.append(doc.page_content)
        if 'ids' not in kwargs:
            # Fill missing IDs with UUIDs
            ids = []
            for i, doc in enumerate(documents):
                if hasattr(doc, 'id') and doc.id is not None:
                    ids.append(doc.metadata["id"])
                else:
                    ids.append(str(uuid.uuid4()))
            kwargs['ids'] = tuple(ids)
        else:
            if len(kwargs['ids']) != len(documents):
                raise ValueError(f"Mismatched IDs: {len(kwargs['ids'])} vs {len(documents)} documents")

        return cls.from_texts(texts, embedding, metadatas=metadatas, config=config, **kwargs)

    @classmethod
    def from_texts(
            cls: type[OpenGauss],
            texts: list[str],
            embedding: Embeddings,
            metadatas: Optional[list[dict]] = None,
            *,
            ids: Optional[list[str]] = None,
            config: Optional[OpenGaussSettings] = None,
            **kwargs: Any,
    ) -> OpenGauss:
        """Return VectorStore initialized from texts and embeddings.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs associated with the texts.
            config: OpenGauss settings configuration.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from texts and embeddings.
        """
        if config is None:
            raise ValueError("OpenGaussSettings config is required")

        vector_db = cls(
            embedding=embedding,
            config=config
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_db

    def drop_table(self):
        with self._get_cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.config.table_name}")
