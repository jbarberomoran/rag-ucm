import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src.evaluation import (  # noqa: E402
    clean_emojis,
    evaluate_results,
    generate_dashboard,
    load_data,
)


def sample_results():
    return pd.DataFrame(
        [
            {
                "method": "bm25",
                "correct": True,
                "status": "✅ ACIERTO PERFECTO (RAG)",
                "response_time": 0.2,
                "retrieval_score": 1.0,
            },
            {
                "method": "dense",
                "correct": False,
                "status": "❌ FALLO TOTAL",
                "response_time": 0.4,
                "retrieval_score": 0.1,
            },
        ]
    )


def test_clean_emojis_removes_status_icons():
    assert clean_emojis("✅ ACIERTO") == "ACIERTO"
    assert clean_emojis(None) is None


def test_load_data_returns_none_for_missing_file(tmp_path):
    assert load_data(str(tmp_path / "missing.csv")) is None


def test_load_data_normalizes_correct_column(tmp_path):
    path = tmp_path / "results.csv"
    sample_results().to_csv(path, index=False)

    loaded = load_data(str(path))

    assert loaded["correct"].dtype.kind in {"i", "u"}


def test_evaluate_results_prints_accuracy(capsys):
    evaluate_results(sample_results(), "final.csv")

    output = capsys.readouterr().out
    assert "ACCURACY ENTRE RESPUESTAS VÁLIDAS" in output
    assert "summary.json" in output
    assert "final.csv" in output


def test_generate_dashboard_creates_all_plots(tmp_path):
    input_path = tmp_path / "results.csv"
    output_dir = tmp_path / "plots"
    sample_results().to_csv(input_path, index=False)

    generate_dashboard(str(input_path), str(output_dir))

    assert {path.name for path in output_dir.iterdir()} == {
        "1_accuracy.png",
        "2_rag_quality_pct.png",
        "3_latency.png",
        "4_retrieval_fidelity.png",
    }
