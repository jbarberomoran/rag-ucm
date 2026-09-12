from dataclasses import dataclass

import pytest

from src.retrieval import RetrievalEngine


@dataclass
class FakeDocument:
    page_content: str


class FakeDatabase:
    def __init__(self, documents=None, metadatas=None):
        self.documents = documents or []
        self.metadatas = metadatas or []
        self.search_kwargs = None

    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return "dense-retriever"

    def get(self):
        return {"documents": self.documents, "metadatas": self.metadatas}


def test_dense_retriever_sets_requested_k():
    engine = RetrievalEngine()
    engine._db = FakeDatabase()

    assert engine.get_retriever("dense", k=3) == "dense-retriever"
    assert engine._db.search_kwargs == {"k": 3}


@pytest.mark.parametrize(
    ("method", "k", "message"),
    [("unknown", 4, "Unsupported"), ("dense", 0, "at least 1")],
)
def test_get_retriever_validates_arguments(method, k, message):
    engine = RetrievalEngine()
    engine._db = FakeDatabase()

    with pytest.raises(ValueError, match=message):
        engine.get_retriever(method, k)


def test_bm25_fails_clearly_for_empty_database():
    engine = RetrievalEngine()
    engine._db = FakeDatabase()

    with pytest.raises(RuntimeError, match="empty vector database"):
        engine.get_retriever("bm25")


def test_rerank_documents_returns_highest_scoring_documents():
    engine = RetrievalEngine()
    engine._reranker = type("Reranker", (), {"predict": lambda self, pairs: [0.1, 0.9]})()
    documents = [FakeDocument("low"), FakeDocument("high")]

    result = engine.rerank_documents("question", documents, top_k=1)

    assert [document.page_content for document in result] == ["high"]


def test_unload_db_clears_all_cached_resources():
    engine = RetrievalEngine()
    engine._db = object()
    engine._embeddings = object()
    engine._bm25_retriever = object()
    engine._reranker = object()

    engine.unload_db()

    assert engine._db is None
    assert engine._embeddings is None
    assert engine._bm25_retriever is None
    assert engine._reranker is None
