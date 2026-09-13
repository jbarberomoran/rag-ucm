"""Human relevance-review packets with dataset/index binding."""
import json
from pathlib import Path

from src.provenance import file_hash


def prepare_review(questions_path, catalog_path):
    questions = json.loads(Path(questions_path).read_text())
    return {
        "schema": 1, "questions_sha256": file_hash(questions_path),
        "catalog_sha256": file_hash(catalog_path),
        "instructions": "Read the paper and all chunks. Mark relevant IDs, reviewer and reviewed. "
                        "Do not infer labels solely from string overlap or model predictions.",
        "chunks": json.loads(Path(catalog_path).read_text()),
        "questions": [{"question_id": index, "question": question["question"],
                       "reference": question.get("paper_reference", ""),
                       "relevant_chunk_ids": [], "reviewer": "", "reviewed": False,
                       "notes": ""} for index, question in enumerate(questions, 1)],
    }


def export_review(packet, questions_path, catalog_path):
    if (packet.get("questions_sha256") != file_hash(questions_path)
            or packet.get("catalog_sha256") != file_hash(catalog_path)):
        raise ValueError("Review packet does not match the current dataset/index")
    valid_ids = {chunk["chunk_id"] for chunk in json.loads(Path(catalog_path).read_text())}
    expected = set(range(1, len(json.loads(Path(questions_path).read_text())) + 1))
    rows = packet["questions"]
    if len(rows) != len(expected) or {r["question_id"] for r in rows} != expected:
        raise ValueError("Review packet must contain every question exactly once")
    annotations = {}
    for row in rows:
        if row.get("reviewed") is not True or not str(row.get("reviewer", "")).strip():
            raise ValueError("All questions require human review and reviewer attribution")
        labels = row.get("relevant_chunk_ids")
        if not isinstance(labels, list) or any(label not in valid_ids for label in labels):
            raise ValueError("Unknown chunk ID in review")
        if labels:
            annotations[str(row["question_id"])] = list(dict.fromkeys(labels))
    return annotations
