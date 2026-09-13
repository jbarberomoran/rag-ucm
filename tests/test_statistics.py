import pandas as pd
import pytest

from src.statistics import paired_report


def test_paired_report_clusters_repeated_questions_and_counts_exclusions():
    rows = [
        {"run_id": r, "question_id": q, "method": m, "correct": m == "a", "error": ""}
        for r in (1, 2) for q in (1, 2) for m in ("a", "b")
    ]
    rows.append({"run_id": 1, "question_id": 3, "method": "a", "correct": True, "error": ""})
    report = paired_report(pd.DataFrame(rows))
    pair = report["comparisons"][0]
    assert pair["unique_questions"] == 2
    assert pair["paired_observations"] == 4
    assert pair["excluded_a"] == 1
    assert pair["excluded_b"] == 0
    assert pair["ci95_low_pp"] == pair["ci95_high_pp"] == 100
    assert report == paired_report(pd.DataFrame(rows))


def test_paired_report_rejects_duplicates_and_aggregates():
    row = {"run_id": 1, "question_id": 1, "method": "a", "correct": True}
    with pytest.raises(ValueError, match="Duplicate"):
        paired_report(pd.DataFrame([row, row]))
    with pytest.raises(ValueError, match="requires"):
        paired_report(pd.DataFrame({"accuracy": [0.9]}))


def test_failures_and_no_pairs_do_not_create_false_zero_estimate():
    frame = pd.DataFrame([
        {"run_id": 1, "question_id": 1, "method": "a", "correct": True, "error": ""},
        {"run_id": 1, "question_id": 1, "method": "b", "correct": None, "error": "Timeout"},
    ])
    result = paired_report(frame)["comparisons"][0]
    assert result["unique_questions"] == 0
    assert result["accuracy_difference_pp"] is None

