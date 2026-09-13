"""Validate the local environment without making paid API calls by default."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from src.config import QUESTIONS_PATH
from src.question_data import load_questions


def check_api(api_key: str) -> None:
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
    )
    response = model.invoke("Reply with exactly: OK")
    print(f"Gemini response: {response.content}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-api", action="store_true", help="Make one Gemini API call")
    args = parser.parse_args()

    load_dotenv()
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Python 3.11 is required")

    questions = load_questions(QUESTIONS_PATH)
    print(f"Question dataset: {len(questions)} valid questions")

    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"GOOGLE_API_KEY: {'configured' if api_key else 'missing'}")
    if args.check_api:
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY is required for --check-api")
        check_api(api_key)


if __name__ == "__main__":
    main()
