"""Question dataset loading and validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {"question", "answers", "correct_answer"}
VALID_ANSWERS = {"A", "B", "C", "D"}


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate the question dataset."""
    try:
        questions = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Question dataset not found: {path}") from error

    if not isinstance(questions, list):
        raise ValueError("Question dataset must be a JSON list")

    for index, question in enumerate(questions):
        if not isinstance(question, dict) or not REQUIRED_FIELDS <= question.keys():
            raise ValueError(f"Question {index} is missing required fields")
        if set(question["answers"]) != VALID_ANSWERS:
            raise ValueError(f"Question {index} must define answers A, B, C, and D")
        if question["correct_answer"] not in VALID_ANSWERS:
            raise ValueError(f"Question {index} has an invalid correct answer")
    return questions


def select_questions(
    questions: Sequence[dict[str, Any]], indices: Sequence[int] | None
) -> list[tuple[int, dict[str, Any]]]:
    """Select questions while preserving their one-based dataset identifiers."""
    if indices is None:
        return list(enumerate(questions, start=1))

    if len(indices) != len(set(indices)):
        raise ValueError("Question indices must not contain duplicates")

    selected = []
    for index in indices:
        if index < 0 or index >= len(questions):
            raise IndexError(f"Question index {index} is outside 0..{len(questions) - 1}")
        selected.append((index + 1, questions[index]))
    return selected
