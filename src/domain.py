"""Dependency-free domain helpers for scoring and serializing experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any, Protocol


class DocumentLike(Protocol):
    page_content: str
    metadata: dict[str, Any]


ANSWER_PATTERN = re.compile(r"(?:The answer is\s+)?([A-D])\.?", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize extracted PDF text for deterministic evidence comparison."""
    return "".join(character for character in text.casefold() if character.isalnum())


def extract_answer(raw_answer: str) -> str:
    """Accept only one letter or the exact legacy wrapper; reject ambiguity."""
    match = ANSWER_PATTERN.fullmatch(raw_answer.strip())
    return match.group(1).upper() if match else "X"


def verify_evidence(
    retrieved_docs: Iterable[DocumentLike],
    ground_truth_ref: str,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Return whether retrieved text contains enough of the reference passage."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    reference = normalize_text(ground_truth_ref)
    if not reference:
        return False, 0.0

    context = normalize_text("".join(doc.page_content for doc in retrieved_docs))
    if not context:
        return False, 0.0
    if reference in context:
        return True, 1.0

    match = SequenceMatcher(None, reference, context).find_longest_match(
        0, len(reference), 0, len(context)
    )
    similarity = match.size / len(reference)
    return similarity >= threshold, similarity


def classify_result(is_correct: bool, found_evidence: bool) -> str:
    """Classify answer correctness independently from retrieval evidence."""
    statuses = {
        (True, True): "Correct / reference overlap detected",
        (True, False): "Correct / reference overlap not detected",
        (False, True): "Incorrect / reference overlap detected",
        (False, False): "Incorrect / reference overlap not detected",
    }
    return statuses[(is_correct, found_evidence)]


def score_retrieval(documents, relevant_chunk_ids):
    """Score ranked chunk IDs against optional human relevance annotations.

    Missing annotations yield no metrics, never an assumed negative label.
    """
    if not relevant_chunk_ids:
        return {"recall_at_k": None, "precision_at_k": None, "reciprocal_rank": None}
    relevant = set(relevant_chunk_ids)
    ranked = list(dict.fromkeys(doc.metadata.get("chunk_id") for doc in documents))
    hits = relevant.intersection(ranked)
    first = next((i for i, value in enumerate(ranked, 1) if value in relevant), None)
    return {
        "recall_at_k": len(hits) / len(relevant),
        "precision_at_k": len(hits) / len(ranked) if ranked else 0.0,
        "reciprocal_rank": 1 / first if first else 0.0,
    }


def serialize_documents(documents: Iterable[DocumentLike]) -> list[dict[str, Any]]:
    """Convert retrieved documents into JSON-safe experiment evidence."""
    return [
        {
            "page_content": document.page_content,
            "metadata": dict(getattr(document, "metadata", {}) or {}),
        }
        for document in documents
    ]
