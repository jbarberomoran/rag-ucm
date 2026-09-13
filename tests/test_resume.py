import json
from types import SimpleNamespace

import pandas as pd
import pytest

import main
from src.domain import score_retrieval
from src.queries import run_questions


def dataset(path):
    path.write_text(json.dumps([{
        "question": "question", "answers": dict(zip("ABCD", ["a", "b", "c", "d"], strict=True)),
        "correct_answer": "B", "paper_reference": "evidence",
    }]))


def test_resume_retries_errors_and_skips_successes(tmp_path):
    questions = tmp_path / "questions.json"
    dataset(questions)
    partial = tmp_path / "partial.csv"
    def failure(*args):
        raise TimeoutError("sensitive message must not be persisted")
    run_questions(methods=["baseline"], partial_file=partial, questions_path=questions,
                  query_fn=failure, sleep_time=0, run_id=3)
    assert "sensitive" not in partial.read_text()
    calls = []
    def success(*args):
        calls.append(args)
        return "B", []
    for _ in range(2):
        run_questions(methods=["baseline"], partial_file=partial, questions_path=questions,
                      query_fn=success, sleep_time=0, run_id=3, resume=True)
    assert len(calls) == 1
    rows = pd.read_csv(partial)
    assert len(rows) == 2
    assert set(rows.run_id) == {3}


def test_annotated_retrieval_metrics_and_missing_labels():
    docs = [SimpleNamespace(metadata={"chunk_id": value}) for value in ["x", "a", "a"]]
    result = score_retrieval(docs, ["a", "b"])
    assert result == {"recall_at_k": 0.5, "precision_at_k": 0.5, "reciprocal_rank": 0.5}
    assert score_retrieval(docs, [])["recall_at_k"] is None


def test_cli_baseline_resume_preserves_rows_without_building_index(tmp_path, monkeypatch):
    questions = tmp_path / "questions.json"
    dataset(questions)
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(main, "QUESTIONS_PATH", questions)
    monkeypatch.setattr(main, "PAPER_PATH", questions)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(main, "create_generator", lambda settings: SimpleNamespace(
        identity=lambda: {"provider": "ollama", "digest": "test-digest"}
    ))
    monkeypatch.setattr(main, "db_setup", lambda *a: pytest.fail("baseline must not load index"))
    def offline(*args, **kwargs):
        kwargs.update(questions_path=questions, query_fn=lambda *a: ("B", []))
        return run_questions(*args, **kwargs)
    monkeypatch.setattr(main, "run_questions", offline)
    args = SimpleNamespace(
        runs=2, sleep=0, name="test", questions=None, methods=["baseline"],
        resume=False, keep_existing=False, rebuild_db=False, chunking="semantic",
        annotations=None, skip_plots=True,
    )
    result = main.run_experiment(args)
    assert len(pd.read_csv(result)) == 2
    with pytest.raises(ValueError, match="not empty"):
        main.run_experiment(args)
    args.resume = True
    main.run_experiment(args)
    assert len(pd.read_csv(result)) == 2
    assert (result.parent / "paired_statistics.json").exists()


def test_incomplete_cli_saves_summary_and_exits_nonzero(tmp_path, monkeypatch):
    questions = tmp_path / "questions.json"
    dataset(questions)
    monkeypatch.setattr(main, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(main, "QUESTIONS_PATH", questions)
    monkeypatch.setattr(main, "PAPER_PATH", questions)
    monkeypatch.setattr(main, "create_generator", lambda settings: SimpleNamespace(
        identity=lambda: {"digest": "test"}
    ))
    def offline(*args, **kwargs):
        kwargs.update(questions_path=questions, query_fn=lambda *a: ("B or C", []))
        return run_questions(*args, **kwargs)
    monkeypatch.setattr(main, "run_questions", offline)
    args = SimpleNamespace(
        runs=1, sleep=0, name="failed", questions=None, methods=["baseline"],
        resume=False, keep_existing=False, rebuild_db=False, chunking="semantic",
        annotations=None, skip_plots=True,
    )
    monkeypatch.setattr(main, "parse_args", lambda: args)
    with pytest.raises(SystemExit) as caught:
        main.main()
    assert caught.value.code != 0
    result = tmp_path / "results" / "persistent_results" / "failed"
    report = json.loads((result / "summary.json").read_text())
    assert report["methods"]["baseline"]["failed"] == 1
    assert report["methods"]["baseline"]["end_to_end_accuracy"] == 0
    assert (result / "resultados_finales.csv").exists()
    assert (result / "paired_statistics.json").exists()
