"""Project-wide paths and experiment defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
QUESTIONS_PATH = DATA_DIR / "questions.json"
PAPER_PATH = DATA_DIR / "paper_refrag.pdf"
CHROMA_PATH = DATA_DIR / "chroma_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
CONTEXT_TOKEN_BUDGET = 2048

SUPPORTED_METHODS = ("baseline", "bm25", "dense", "hybrid", "cross_encoder")
