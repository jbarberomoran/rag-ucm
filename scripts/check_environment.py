"""Validate the local environment without making paid API calls by default."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from src.config import QUESTIONS_PATH
from src.generation import create_generator, resolve_settings
from src.question_data import load_questions


def check_generation(settings) -> None:
    generator = create_generator(settings)
    print(f"Model identity: {generator.identity()}")
    response = generator.invoke("Reply with exactly: OK")
    print(f"{settings.provider}/{settings.model} response: {response.content}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-generation", "--check-api", dest="check_generation", action="store_true",
        help="Make one request to the selected generation provider",
    )
    parser.add_argument("--provider", choices=["ollama", "gemini"])
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    args = parser.parse_args()

    load_dotenv()
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Python 3.11 is required")

    questions = load_questions(QUESTIONS_PATH)
    print(f"Question dataset: {len(questions)} valid questions")

    settings = resolve_settings(args)
    print(f"Generation provider: {settings.provider}; model: {settings.model}")
    if args.check_generation:
        check_generation(settings)


if __name__ == "__main__":
    main()
