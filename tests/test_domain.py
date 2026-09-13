from dataclasses import dataclass, field

import pytest

from src.domain import (
    classify_result,
    extract_answer,
    normalize_text,
    serialize_documents,
    verify_evidence,
)


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


def test_normalize_text_ignores_case_spacing_and_punctuation():
    assert normalize_text("ReFrag:\nTwo-stage RAG!") == "refragtwostagerag"


def test_extract_answer_accepts_a_standalone_letter():
    assert extract_answer("The answer is c.") == "C"


def test_extract_answer_rejects_letters_embedded_in_words():
    assert extract_answer("database") == "X"


def test_verify_evidence_detects_exact_normalized_passage():
    documents = [FakeDocument("The model uses two-stage retrieval.")]

    assert verify_evidence(documents, "two stage retrieval") == (True, 1.0)


def test_verify_evidence_returns_fuzzy_similarity():
    documents = [FakeDocument("abcdefghijXXXX")]

    found, score = verify_evidence(documents, "abcdefghijzzzz", threshold=0.7)

    assert found is True
    assert score == pytest.approx(10 / 14)


def test_verify_evidence_handles_missing_reference_or_documents():
    assert verify_evidence([], "reference") == (False, 0.0)
    assert verify_evidence([FakeDocument("context")], "") == (False, 0.0)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_verify_evidence_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        verify_evidence([], "reference", threshold)


@pytest.mark.parametrize(
    ("correct", "evidence", "expected"),
    [
        (True, True, "Correct / reference overlap detected"),
        (True, False, "Correct / reference overlap not detected"),
        (False, True, "Incorrect / reference overlap detected"),
        (False, False, "Incorrect / reference overlap not detected"),
    ],
)
def test_classify_result_covers_all_outcomes(correct, evidence, expected):
    assert classify_result(correct, evidence) == expected


def test_serialize_documents_produces_json_safe_evidence():
    documents = [FakeDocument("chunk", {"page": 7})]

    assert serialize_documents(documents) == [
        {"page_content": "chunk", "metadata": {"page": 7}}
    ]
