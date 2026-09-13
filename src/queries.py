"""Experiment orchestration for the question dataset."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from src.config import QUESTIONS_PATH, SUPPORTED_METHODS
from src.domain import classify_result, extract_answer, serialize_documents, verify_evidence
from src.question_data import load_questions, select_questions
from src.rag_pipeline import query_rag
from src.retrieval import RetrievalEngine

SLEEP_TIME = 0.2


def _validate_methods(methods: Sequence[str]) -> list[str]:
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unsupported retrieval methods: {', '.join(unknown)}")
    return list(methods)


def _warm_up(engine, methods: Sequence[str]) -> None:
    if any(method in methods for method in ("bm25", "hybrid", "cross_encoder")):
        engine.get_retriever("hybrid", k=1)
    elif "dense" in methods:
        engine.get_retriever("dense", k=1)
    if "cross_encoder" in methods:
        _ = engine.reranker


def run_questions(
    questions_slice: Sequence[int] | None = None,
    methods: Sequence[str] | None = None,
    api_key: str | None = None,
    partial_file: str | Path = "./results/resultados_parciales.csv",
    sleep_time: float = SLEEP_TIME,
    *,
    questions_path: str | Path = QUESTIONS_PATH,
    query_fn: Callable = query_rag,
    engine=None,
) -> pd.DataFrame:
    """Run selected questions and persist a transparent row per experiment."""
    selected_methods = _validate_methods(methods or SUPPORTED_METHODS)
    question_records = select_questions(load_questions(Path(questions_path)), questions_slice)
    if any(method != "baseline" for method in selected_methods):
        engine = engine or RetrievalEngine.get_instance()
        _warm_up(engine, selected_methods)

    output_path = Path(partial_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for question_id, question in question_records:
        for method in selected_methods:
            start = time.perf_counter()
            raw_answer, retrieved_docs = query_fn(
                question["question"], question["answers"], method, api_key
            )
            latency = time.perf_counter() - start
            if sleep_time:
                time.sleep(sleep_time)

            predicted = extract_answer(raw_answer)
            is_correct = predicted == question["correct_answer"]
            found_evidence, evidence_score = verify_evidence(
                retrieved_docs, question.get("paper_reference", "")
            )
            row = {
                "question_id": question_id,
                "method": method,
                "correct": is_correct,
                "predicted": predicted,
                "ground_truth": question["correct_answer"],
                "response_time": round(latency, 4),
                "raw_output": raw_answer,
                "status": classify_result(is_correct, found_evidence),
                "retrieval_score": evidence_score,
                "retrieved_docs": json.dumps(
                    serialize_documents(retrieved_docs), ensure_ascii=False
                ),
            }
            results.append(row)
            pd.DataFrame([row]).to_csv(
                output_path,
                mode="a",
                header=not output_path.exists(),
                index=False,
            )

    return pd.DataFrame(results)
