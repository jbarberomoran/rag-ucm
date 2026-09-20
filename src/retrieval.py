import gc
import warnings

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import (
    CHROMA_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_REVISION,
    RERANKER_MODEL_NAME,
    RERANKER_REVISION,
)


class RetrievalEngine:
    """Lazily load and cache retrieval models and indexes."""

    _instance = None

    def __init__(self):
        self._db = None
        self._embeddings = None
        self._bm25_retriever = None
        self._reranker = None

    @classmethod
    def get_instance(cls):
        """Return the process-wide retrieval engine."""
        if cls._instance is None:
            cls._instance = RetrievalEngine()
        return cls._instance

    @property
    def db(self):
        """Load the embedding model and persistent Chroma index on first use."""
        if self._db is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME, model_kwargs={"revision": EMBEDDING_REVISION}
            )
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self._db = Chroma(
                persist_directory=str(CHROMA_PATH), embedding_function=self._embeddings
            )
        return self._db

    def unload_db(self):
        """Release cached retrieval resources."""
        had_loaded_resources = any(
            resource is not None
            for resource in (
                self._db,
                self._embeddings,
                self._bm25_retriever,
                self._reranker,
            )
        )
        self._db = None
        self._embeddings = None
        self._bm25_retriever = None
        self._reranker = None
        if had_loaded_resources:
            gc.collect()

    def _get_bm25_retriever(self):
        """Build or return the cached BM25 index."""
        if self._bm25_retriever is not None:
            return self._bm25_retriever

        try:
            raw_data = self.db.get()
            texts = raw_data["documents"]
            metadatas = raw_data["metadatas"]

            if not texts:
                print("Warning: the vector database is empty.")
                return None

            docs_obj = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas, strict=True)
            ]

            self._bm25_retriever = BM25Retriever.from_documents(docs_obj)
            return self._bm25_retriever

        except Exception as error:
            print(f"Unable to build the BM25 index: {error}")
            return None

    def get_retriever(self, method, k=4):
        """Return a dense, BM25, or equally weighted hybrid retriever."""
        if method not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Unsupported retrieval method: {method}")
        if k < 1:
            raise ValueError("k must be at least 1")

        dense_retriever = self.db.as_retriever(search_kwargs={"k": k})

        if method == "dense":
            return dense_retriever

        bm25_retriever = self._get_bm25_retriever()
        if bm25_retriever is None:
            raise RuntimeError("Cannot build BM25 retriever from an empty vector database")

        bm25_retriever.k = k

        if method == "bm25":
            return bm25_retriever

        if method == "hybrid":
            return EnsembleRetriever(
                retrievers=[bm25_retriever, dense_retriever],
                weights=[0.5, 0.5],
            )

        raise AssertionError("unreachable")

    @property
    def reranker(self):
        """Load the cross-encoder only when reranking is requested."""
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL_NAME, revision=RERANKER_REVISION)
        return self._reranker

    def rerank_documents(self, query, docs, top_k=5):
        """Score candidate documents against the query and return the best matches."""
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        docs_with_scores = sorted(
            zip(docs, scores, strict=True), key=lambda item: item[1], reverse=True
        )
        return [doc for doc, _score in docs_with_scores[:top_k]]
