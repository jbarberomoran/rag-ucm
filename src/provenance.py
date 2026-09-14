"""Non-secret experiment and index provenance."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.config import (
    CONTEXT_TOKEN_BUDGET,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_REVISION,
    RERANKER_MODEL_NAME,
    RERANKER_REVISION,
)
from src.generation import resolve_settings


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def index_config(paper, method, size, overlap):
    return {
        "schema": 1, "paper_sha256": file_hash(paper),
        "embedding_model": EMBEDDING_MODEL_NAME, "chunking": method,
        "embedding_revision": EMBEDDING_REVISION,
        "chunk_size": size, "chunk_overlap": overlap, "semantic_percentile": 95,
    }


def experiment_config(args, questions, paper):
    return {
        "source_sha256": hashlib.sha256(b"".join(
            path.read_bytes() for path in (
                sorted(Path(__file__).parent.glob("*.py"))
                + [Path(__file__).parent.parent / "main.py"]
            )
        )).hexdigest(),
        "schema": 2, "questions_sha256": file_hash(questions), "paper_sha256": file_hash(paper),
        "methods": list(args.methods), "questions": args.questions, "runs": args.runs,
        "generation": resolve_settings(args).public_config(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_revision": EMBEDDING_REVISION,
        "reranker_revision": RERANKER_REVISION,
        "context_token_budget": CONTEXT_TOKEN_BUDGET,
        "reranker_model": RERANKER_MODEL_NAME, "chunking": args.chunking,
        "chunk_size": 1200, "chunk_overlap": 350, "semantic_percentile": 95,
        "retrieval_k": 5, "candidate_k_per_retriever": 20, "hybrid_weights": [0.5, 0.5],
        "overlap_threshold": 0.5, "annotation_sha256": (
            file_hash(args.annotations) if args.annotations else None
        ),
    }


def prepare_manifest(path, config, resume=False):
    path = Path(path)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if resume:
        if not path.exists():
            raise ValueError("Cannot resume without manifest.json")
        manifest = json.loads(path.read_text())
        if manifest["config"] != config:
            raise ValueError("Resume configuration differs from the saved experiment")
        if (manifest.get("python") != platform.python_version()
                or manifest.get("packages") != packages):
            raise ValueError("Resume environment differs from the saved experiment")
        return manifest
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {
        "created_at": datetime.now(UTC).isoformat(), "git_revision": revision,
        "python": platform.python_version(), "packages": packages, "config": config,
        "limitations": "Single-paper dataset; optional cloud aliases may change server-side.",
    }
    save_json(path, manifest)
    return manifest


def load_annotations(path, catalog_path, questions_path):
    if path is None:
        return {}
    annotations = json.loads(Path(path).read_text())
    if not isinstance(annotations, dict):
        raise ValueError("Annotations must map question IDs to lists of chunk IDs")
    catalog = json.loads(Path(catalog_path).read_text())
    valid_chunks = {chunk["chunk_id"] for chunk in catalog}
    count = len(json.loads(Path(questions_path).read_text()))
    for question, chunks in annotations.items():
        if question not in {str(i) for i in range(1, count + 1)}:
            raise ValueError("Unknown annotation question ID")
        if not isinstance(chunks, list) or any(
            not isinstance(chunk, str) or chunk not in valid_chunks for chunk in chunks
        ):
            raise ValueError("Unknown chunk IDs: annotations must match the current index")
    return annotations
