from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from src.rag_pipeline import query_rag


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.questions = []

    def invoke(self, question):
        self.questions.append(question)
        return self.documents


class FakeEngine:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def get_retriever(self, method, k):
        self.calls.append((method, k))
        return FakeRetriever(self.documents)

    def rerank_documents(self, question, documents, top_k):
        self.calls.append(("rerank", question, top_k))
        return list(reversed(documents))[:top_k]


class FakeLlm:
    def __init__(self, response="B"):
        self.response = response
        self.prompts = []

    def count_tokens(self, text):
        return len(text.split())

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response)


def options():
    return {"A": "one", "B": "two", "C": "three", "D": "four"}


def test_baseline_does_not_load_retrieval_engine():
    llm = FakeLlm()

    answer, documents = query_rag(
        "Question?",
        options(),
        "baseline",
        "test-key",
        engine=object(),
        llm_factory=lambda **_: llm,
    )

    assert answer == "B"
    assert documents == []
    assert "using your internal knowledge" in llm.prompts[0]
    assert "ONLY on the provided context" not in llm.prompts[0]


def test_dense_retrieval_adds_document_context():
    documents = [FakeDocument("retrieved evidence")]
    engine = FakeEngine(documents)
    llm = FakeLlm("A")

    answer, returned_documents = query_rag(
        "Question?",
        options(),
        "dense",
        "test-key",
        engine=engine,
        llm_factory=lambda **_: llm,
    )

    assert answer == "A"
    assert returned_documents == documents
    assert engine.calls == [("dense", 5)]
    assert "retrieved evidence" in llm.prompts[0]


def test_cross_encoder_uses_broad_hybrid_retrieval_then_reranks():
    documents = [FakeDocument("first"), FakeDocument("second")]
    engine = FakeEngine(documents)

    _, returned_documents = query_rag(
        "Question?",
        options(),
        "cross_encoder",
        "test-key",
        engine=engine,
        llm_factory=lambda **_: FakeLlm(),
    )

    assert [document.page_content for document in returned_documents] == ["second", "first"]
    assert engine.calls == [("hybrid", 20), ("rerank", "Question?", 5)]


@pytest.mark.parametrize(
    ("method", "api_key", "answer_options", "message"),
    [
        ("unknown", "key", options(), "Unsupported method"),
        ("baseline", "key", {"A": "only"}, "exactly A, B, C, and D"),
    ],
)
def test_query_rag_validates_inputs(method, api_key, answer_options, message):
    with pytest.raises(ValueError, match=message):
        query_rag("Question?", answer_options, method, api_key)
