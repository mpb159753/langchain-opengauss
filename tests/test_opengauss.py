# coding:utf-8
from typing import Generator

import pytest
from langchain_opengauss import OpenGauss, OpenGaussSettings
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests


class TestOpenGauss(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        setting = OpenGaussSettings(host='90.91.42.222', port=8888, embedding_dimension=6)
        store = OpenGauss(self.get_embeddings(), setting)
        try:
            yield store
        finally:
            store.drop_table()
            pass
