# Repository guidelines

## Purpose

This repository compares several retrieval strategies for a university RAG
experiment. Preserve reproducibility, source attribution, and the separation
between retrieval quality and answer quality.

## Working agreements

- Use Python 3.11.
- Never commit API keys, local vector stores, virtual environments, or generated
  experiment outputs.
- Keep ingestion, retrieval, generation, orchestration, and evaluation concerns
  separated.
- Keep calls to Gemini and model downloads outside unit tests; use test doubles.
- Do not delete research inputs or final academic artifacts without explaining
  why they are redundant and confirming that Git history preserves them.

## Verification

- Run `python -m pytest` after changing Python code.
- Run `python -m ruff check .` before committing.
- Unit tests must not require network access, an API key, a vector database, or
  downloaded ML models.

## Code review rules

- Flag any path that can leak secrets or private source documents.
- Flag evaluation changes that mix retrieval evidence with answer correctness.
- Flag tests that call paid or nondeterministic external services.
