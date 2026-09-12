import json
from dataclasses import dataclass, field

import pytest

from src.queries import _warm_up, run_questions


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self._reranker = object()

    def get_retriever(self, method, k):
        self.calls.append((method, k))
        return object()

    @property
    def reranker(self):
        self.calls.append(("reranker",))
        return self._reranker


def write_questions(path):
    questions = [
        {
            "question": "What is supported?",
            "answers": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            "correct_answer": "B",
            "paper_reference": "supporting evidence",
        }
    ]
    path.write_text(json.dumps(questions), encoding="utf-8")


def test_run_questions_builds_and_persists_result_row(tmp_path):
    questions_path = tmp_path / "questions.json"
    partial_path = tmp_path / "results" / "partial.csv"
    write_questions(questions_path)

    frame = run_questions(
        methods=["baseline"],
        api_key="test-key",
        partial_file=partial_path,
        sleep_time=0,
        questions_path=questions_path,
        query_fn=lambda *_: ("The answer is B", [FakeDocument("supporting evidence", {"page": 1})]),
    )

    assert frame.loc[0, "question_id"] == 1
    assert bool(frame.loc[0, "correct"]) is True
    assert frame.loc[0, "retrieval_score"] == 1.0
    assert json.loads(frame.loc[0, "retrieved_docs"])[0]["metadata"] == {"page": 1}
    assert partial_path.exists()


def test_run_questions_rejects_unknown_method(tmp_path):
    questions_path = tmp_path / "questions.json"
    write_questions(questions_path)

    with pytest.raises(ValueError, match="Unsupported retrieval methods"):
        run_questions(methods=["magic"], questions_path=questions_path)


@pytest.mark.parametrize(
    ("methods", "expected"),
    [
        (["baseline"], []),
        (["dense"], [("dense", 1)]),
        (["bm25"], [("hybrid", 1)]),
        (["cross_encoder"], [("hybrid", 1), ("reranker",)]),
    ],
)
def test_warm_up_loads_only_required_models(methods, expected):
    engine = FakeEngine()

    _warm_up(engine, methods)

    assert engine.calls == expected
