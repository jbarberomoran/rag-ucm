import json

import pytest

from src.config import QUESTIONS_PATH
from src.question_data import load_questions, select_questions


def question(correct_answer="A"):
    return {
        "question": "Example?",
        "answers": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "correct_answer": correct_answer,
    }


def test_repository_dataset_is_valid():
    questions = load_questions(QUESTIONS_PATH)

    assert len(questions) == 70


def test_load_questions_rejects_missing_fields(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"question": "Incomplete"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_questions(path)


def test_load_questions_rejects_invalid_answer_options(tmp_path):
    path = tmp_path / "questions.json"
    invalid = question()
    invalid["answers"].pop("D")
    path.write_text(json.dumps([invalid]), encoding="utf-8")

    with pytest.raises(ValueError, match="answers A, B, C, and D"):
        load_questions(path)


def test_select_questions_preserves_original_ids():
    questions = [question("A"), question("B"), question("C")]

    selected = select_questions(questions, [2, 0])

    assert [(identifier, item["correct_answer"]) for identifier, item in selected] == [
        (3, "C"),
        (1, "A"),
    ]


def test_select_questions_rejects_out_of_range_index():
    with pytest.raises(IndexError, match="outside"):
        select_questions([question()], [1])
