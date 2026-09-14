import copy
import json
from collections import Counter
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import DATA_DIR
from src.context import pack_context
from src.dataset_balance import rebalance
from src.domain import extract_answer
from src.provenance import prepare_manifest
from src.queries import run_questions
from src.question_data import select_questions
from src.run_summary import summarize_run


@pytest.mark.parametrize("raw", ["A is wrong; the correct answer is C", "B or C", "", "A B"])
def test_ambiguous_answers_are_rejected(raw):
    assert extract_answer(raw) == "X"


def test_dataset_balance_preserves_every_field_and_correct_text():
    balanced = json.loads((DATA_DIR / "questions.json").read_text())
    audit = json.loads((DATA_DIR / "questions-permutation.json").read_text())
    original = copy.deepcopy(balanced)
    for old, current, mapping in zip(original, balanced, audit["permutations"], strict=True):
        old["answers"] = {letter: current["answers"][new]
                          for new, letter in mapping["new_to_old"].items()}
        old["correct_answer"] = mapping["old_correct"]
        assert (old["answers"][old["correct_answer"]]
                == current["answers"][current["correct_answer"]])
    regenerated, regenerated_audit = rebalance(original, audit["seed"])
    assert regenerated == balanced
    assert regenerated_audit["permutations"] == audit["permutations"]
    assert regenerated_audit["original_content_sha256"] == audit["original_content_sha256"]
    assert Counter(q["correct_answer"] for q in balanced) == {"A": 18, "B": 18, "C": 17, "D": 17}
    assert len(balanced) == 70


def test_rebalance_refuses_position_dependent_options():
    with pytest.raises(ValueError, match="manual"):
        rebalance([{"answers": {"A": "All of the above"}, "correct_answer": "A"}])


def test_common_context_budget_includes_separators_and_whole_chunks():
    docs = [SimpleNamespace(page_content=t) for t in ["oversized", "one", "two", "end"]]
    assert pack_context(docs, len, 8) == docs[1:3]
    assert pack_context(docs * 4, lambda _: 1, 8) == (docs * 4)[:5]
    assert pack_context(docs, len, 0) == []


def test_summary_accounts_for_failures_and_missing_rows():
    frame = pd.DataFrame([
        dict(run_id=1, question_id=1, method="baseline", correct=1, error="", predicted="A"),
        dict(run_id=1, question_id=2, method="baseline", correct=None,
             error="TimeoutError", predicted="X"),
    ])
    report = summarize_run(frame, [1, 2, 3], ["baseline"], 1)
    assert not report["complete"]
    method = report["methods"]["baseline"]
    assert method["end_to_end_accuracy"] == 1 / 3
    assert method["valid_response_accuracy"] == 1
    assert method["failed"] == method["missing"] == 1


def test_resume_rejects_changed_environment(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    prepare_manifest(manifest, {})
    monkeypatch.setattr("src.provenance.platform.python_version", lambda: "different")
    with pytest.raises(ValueError, match="environment"):
        prepare_manifest(manifest, {}, True)


def test_duplicate_question_selection_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        select_questions([{}], [0, 0])


def test_invalid_output_is_retried(tmp_path):
    path = tmp_path / "results.csv"
    settings = dict(methods=["baseline"], questions_slice=[0], partial_file=path, sleep_time=0)
    first = run_questions(**settings, query_fn=lambda *a: ("B or C", []))
    assert first.iloc[0].error == "InvalidAnswer"
    second = run_questions(**settings, resume=True, query_fn=lambda *a: ("D", []))
    assert len(second) == 1
