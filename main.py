"""Command-line entry point for reproducible RAG experiments."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import CHROMA_PATH, PAPER_PATH, QUESTIONS_PATH, RESULTS_DIR, SUPPORTED_METHODS
from src.evaluation import evaluate_results, generate_dashboard
from src.generation import create_generator, resolve_settings
from src.ingestion import db_setup
from src.provenance import experiment_config, load_annotations, prepare_manifest, save_json
from src.queries import run_questions
from src.question_data import load_questions, select_questions
from src.run_summary import summarize_run
from src.statistics import paired_report

RESULT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Store results under results/persistent_results/NAME")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs")
    parser.add_argument("--questions", type=int, nargs="*", help="Zero-based question indices")
    parser.add_argument(
        "--methods", nargs="+", choices=SUPPORTED_METHODS, default=list(SUPPORTED_METHODS)
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls")
    parser.add_argument("--rebuild-db", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Retry incomplete observations")
    parser.add_argument("--chunking", choices=["recursive", "semantic"], default="semantic")
    parser.add_argument("--annotations", type=Path, help="Optional question-to-chunk-ID JSON")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--provider", choices=["ollama", "gemini"])
    parser.add_argument("--model", help="Generation model name (provider-specific)")
    parser.add_argument("--base-url", help="Ollama server URL")
    parser.add_argument("--timeout", type=float, help="Generation request timeout in seconds")
    parser.add_argument("--context-tokens", type=int, help="Ollama context window")
    parser.add_argument("--max-tokens", type=int, help="Maximum generated tokens")
    parser.add_argument("--tokenizer", help="Hugging Face tokenizer for a custom Ollama model")
    parser.add_argument("--tokenizer-revision", help="Immutable tokenizer commit SHA")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> Path:
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.sleep < 0:
        raise ValueError("--sleep must be non-negative")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must not contain duplicates")
    selected = select_questions(load_questions(QUESTIONS_PATH), args.questions)
    if not selected:
        raise ValueError("No questions selected")

    load_dotenv()
    settings = resolve_settings(args)
    generator = create_generator(settings)

    if args.name and not RESULT_NAME_PATTERN.fullmatch(args.name):
        raise ValueError("--name may contain only letters, numbers, dots, dashes, and underscores")

    results_dir = (
        RESULTS_DIR / "persistent_results" / args.name
        if args.name
        else RESULTS_DIR / "local_results"
    )
    final_file = results_dir / "resultados_finales.csv"
    partial_file = results_dir / "resultados_parciales.csv"
    resume = args.resume or args.keep_existing
    if results_dir.exists() and any(results_dir.iterdir()) and not resume:
        raise ValueError("Experiment directory is not empty; choose --name or use --resume")
    results_dir.mkdir(parents=True, exist_ok=True)
    config = experiment_config(args, QUESTIONS_PATH, PAPER_PATH)
    config["generation_identity"] = generator.identity()
    prepare_manifest(results_dir / "manifest.json", config, resume)
    if any(method != "baseline" for method in args.methods):
        db_setup(args.rebuild_db, args.chunking)
    annotations = load_annotations(args.annotations, CHROMA_PATH / "chunks.json", QUESTIONS_PATH)

    for run_id in range(1, args.runs + 1):
        run_questions(
            args.questions,
            args.methods,
            settings.api_key,
            partial_file,
            args.sleep,
            run_id=run_id,
            resume=resume,
            annotations=annotations,
            generator=generator,
            generation_settings=settings,
        )

    if not partial_file.exists():
        raise ValueError("No questions selected; no observations produced")
    combined = pd.read_csv(partial_file).drop_duplicates(
        ["run_id", "question_id", "method"], keep="last"
    )
    combined["correct"] = combined["correct"].map(
        lambda value: {"True": 1.0, "False": 0.0}.get(str(value), value)
    )
    combined["correct"] = pd.to_numeric(combined["correct"], errors="raise")
    results_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(final_file, index=False)
    summary = summarize_run(combined, [q for q, _ in selected], args.methods, args.runs)
    answer_counts = Counter(q["correct_answer"] for _, q in selected)
    summary["dataset_controls"] = {
        "answer_distribution": dict(answer_counts),
        "majority_letter_accuracy": max(answer_counts.values()) / len(selected),
        "uniform_random_expected_accuracy": 0.25,
        "note": "Analytical controls, not additional model requests.",
    }
    save_json(results_dir / "summary.json", summary)
    print(f"Experiment complete: {summary['complete']}; "
          f"full accounting: {results_dir / 'summary.json'}", flush=True)
    save_json(results_dir / "paired_statistics.json", paired_report(combined))
    successful = combined[combined["error"].fillna("") == ""]
    evaluate_results(successful, str(final_file))
    if not args.skip_plots and not successful.empty:
        generate_dashboard(str(final_file), str(results_dir / "plots"))
    if not summary["complete"]:
        raise IncompleteExperimentError(
            f"Incomplete experiment; see {results_dir / 'summary.json'} and retry with --resume"
        )
    return final_file


class IncompleteExperimentError(RuntimeError):
    """Artifacts were saved, but one or more observations are invalid or missing."""


def main() -> None:
    try:
        run_experiment(parse_args())
    except IncompleteExperimentError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
