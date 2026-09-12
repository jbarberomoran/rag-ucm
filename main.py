"""Command-line entry point for reproducible RAG experiments."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import RESULTS_DIR, SUPPORTED_METHODS
from src.evaluation import evaluate_results, generate_dashboard
from src.launcher import setup_environment
from src.queries import run_questions

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
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> Path:
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing; copy .env.example to .env")

    if args.name and not RESULT_NAME_PATTERN.fullmatch(args.name):
        raise ValueError("--name may contain only letters, numbers, dots, dashes, and underscores")

    results_dir = (
        RESULTS_DIR / "persistent_results" / args.name
        if args.name
        else RESULTS_DIR / "local_results"
    )
    final_file = results_dir / "resultados_finales.csv"
    partial_file = RESULTS_DIR / "resultados_parciales.csv"
    setup_environment(args.rebuild_db, not args.keep_existing, results_dir)

    runs = []
    for run_id in range(1, args.runs + 1):
        frame = run_questions(
            args.questions,
            args.methods,
            api_key,
            partial_file,
            args.sleep,
        )
        frame.insert(0, "run_id", run_id)
        runs.append(frame)

    combined = pd.concat(runs, ignore_index=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(final_file, index=False)
    evaluate_results(combined, str(final_file))
    if not args.skip_plots:
        generate_dashboard(str(final_file), str(results_dir / "plots"))
    return final_file


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
