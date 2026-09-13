"""Real storage integration without model downloads or API calls."""
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.domain import score_retrieval


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float("alpha" in text), float("beta" in text), 0.1]


def test_persisted_chroma_returns_annotated_document(tmp_path):
    store = Chroma.from_documents(
        [Document(page_content="alpha", metadata={"chunk_id": "a"}),
         Document(page_content="beta", metadata={"chunk_id": "b"})],
        LocalEmbeddings(), persist_directory=str(tmp_path / "index"),
        collection_name="integration",
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    reopened = Chroma(
        persist_directory=str(tmp_path / "index"), collection_name="integration",
        embedding_function=LocalEmbeddings(),
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    retrieved = reopened.similarity_search("alpha", k=1)
    assert retrieved[0].metadata["chunk_id"] == "a"
    assert score_retrieval(retrieved, ["a"])["recall_at_k"] == 1
    store.delete_collection()
