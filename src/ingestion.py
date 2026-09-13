import gc
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_PATH, EMBEDDING_MODEL_NAME, PAPER_PATH
from src.retrieval import RetrievalEngine

# --- CONFIGURACIÓN ---
# Chunking y almacenamiento vectorial
# SELECTOR DE ESTRATEGIA: "recursive" o "semantic"
# - "recursive": Rápido, corta por tamaño fijo (Recomendado para empezar).
# - "semantic": Lento, usa IA para cortar por temas (Mejor calidad, requiere rebuild).
CHUNKING_METHOD = "semantic"

# Estrategia "recursive": Tamaño mediano con overlap del ~30% para mantener contexto
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 350


def get_text_splitter(method, embedding_model=None):
    """
    Fábrica de Splitters: Devuelve la herramienta de corte según la configuración.
    """
    if method == "semantic":
        # Corta cuando la diferencia semántica entre frases es muy alta
        return SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
    if method == "recursive":
        # Corte recursivo clásico por tamaño fijo
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    raise ValueError("chunking method must be 'recursive' or 'semantic'")


def ingest_data(chunking_method=CHUNKING_METHOD):
    """Carga el PDF y lo trocea en chunks usando la estrategia seleccionada"""
    if not PAPER_PATH.exists():
        print(f"\n❌ ERROR: No encuentro el archivo '{PAPER_PATH}'")
        return []

    print("📄 Cargando PDF...")
    loader = PyPDFLoader(str(PAPER_PATH))
    docs = loader.load()
    print(f"   -> PDF cargado: {len(docs)} páginas.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    splitter = get_text_splitter(chunking_method, embeddings)

    print("✂️ Procesando fragmentos")
    chunks = splitter.split_documents(docs)

    print(f"   -> Generados {len(chunks)} fragmentos.")

    return chunks


def create_vector_db(chunks):
    """Guarda los chunks en ChromaDB."""
    if not chunks:
        return

    print("🧠 Guardando vectores en disco...")
    # Volvemos a instanciar embeddings (ligero) para Chroma
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    print("💾 Base de datos guardada exitosamente.")


def clear_existing_db():
    """Borrado seguro de la base de datos."""
    print("\n🔌 Desconectando motor de búsqueda...")
    try:
        engine = RetrievalEngine.get_instance()
        engine.unload_db()
        gc.collect()
        if CHROMA_PATH.exists():
            shutil.rmtree(CHROMA_PATH)
    except Exception as error:
        print(f"\n❌ Error inesperado borrando DB: {error}")
        return False

    return True


# --- ENTRY POINT ---
def db_setup(rebuild_db: bool = False, chunking_method=CHUNKING_METHOD):
    db_exists = CHROMA_PATH.is_dir() and any(CHROMA_PATH.iterdir())

    if not rebuild_db and db_exists:
        print("\n⏩ Base de datos encontrada. Saltando ingesta.")
        return

    if not db_exists:
        print("\n⚠️  Base de datos no encontrada. Creando nueva...")

    if not clear_existing_db():
        raise RuntimeError("\nNo se pudo limpiar la base de datos antigua.")

    chunks = ingest_data(chunking_method)
    create_vector_db(chunks)
    print("✅ Setup completado.")


if __name__ == "__main__":
    # Si ejecutas este archivo directamente, fuerza reconstrucción
    db_setup(rebuild_db=True)
