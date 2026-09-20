import gc
import hashlib
import json
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_REVISION, PAPER_PATH
from src.provenance import index_config, save_json
from src.retrieval import RetrievalEngine

# Default chunking strategy used for vector-index creation.
CHUNKING_METHOD = "semantic"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 350


def get_text_splitter(method, embedding_model=None):
    """Create the configured semantic or recursive text splitter."""
    if method == "semantic":
        return SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
    if method == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    raise ValueError("chunking method must be 'recursive' or 'semantic'")


def ingest_data(chunking_method=CHUNKING_METHOD):
    """Load the source PDF and split it with the selected strategy."""
    if not PAPER_PATH.exists():
        print(f"\nSource PDF not found: {PAPER_PATH}")
        return []

    print("Loading source PDF...")
    loader = PyPDFLoader(str(PAPER_PATH))
    docs = loader.load()
    print(f"Loaded {len(docs)} PDF pages.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs={"revision": EMBEDDING_REVISION}
    )

    splitter = get_text_splitter(chunking_method, embeddings)

    print("Splitting document into chunks...")
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = hashlib.sha256(
            f"{index}:{chunk.metadata.get('page')}:{chunk.page_content}".encode()
        ).hexdigest()

    print(f"Generated {len(chunks)} chunks.")

    return chunks


def create_vector_db(chunks):
    """Persist document chunks in Chroma."""
    if not chunks:
        return

    print("Persisting vectors...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs={"revision": EMBEDDING_REVISION}
    )
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    save_json(CHROMA_PATH / "chunks.json", [
        {"chunk_id": chunk.metadata["chunk_id"], "page": chunk.metadata.get("page"),
         "text": chunk.page_content} for chunk in chunks
    ])
    print("Vector database saved.")


def clear_existing_db():
    """Release retrieval resources and remove the generated vector index."""
    print("\nReleasing retrieval resources...")
    try:
        engine = RetrievalEngine.get_instance()
        engine.unload_db()
        gc.collect()
        if CHROMA_PATH.exists():
            shutil.rmtree(CHROMA_PATH)
    except Exception as error:
        print(f"\nUnable to remove the vector database: {error}")
        return False

    return True


# --- ENTRY POINT ---
def db_setup(rebuild_db: bool = False, chunking_method=CHUNKING_METHOD):
    expected = index_config(PAPER_PATH, chunking_method, CHUNK_SIZE, CHUNK_OVERLAP)
    manifest_path = CHROMA_PATH / "index_manifest.json"
    db_exists = CHROMA_PATH.is_dir() and any(CHROMA_PATH.iterdir())

    if not rebuild_db and db_exists:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != expected:
            raise ValueError("Index configuration changed or is unknown; use --rebuild-db")
        print("\nCompatible vector database found; skipping ingestion.")
        return

    if not db_exists:
        print("\nVector database not found; creating it now.")

    if not clear_existing_db():
        raise RuntimeError("Unable to clear the previous vector database")

    chunks = ingest_data(chunking_method)
    if not chunks:
        raise ValueError("No chunks extracted; index was not created")
    create_vector_db(chunks)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    save_json(manifest_path, expected)
    print("Vector database setup complete.")


if __name__ == "__main__":
    db_setup(rebuild_db=True)
