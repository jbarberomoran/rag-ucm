import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURACIÓN ---
# Chunking y almacenamiento vectorial
FILE_PATH = "./data/paper_refrag.pdf"
CHROMA_PATH = "./data/chroma_db"

# Estrategia: Tamaño mediano con overlap del 20% para mantener contexto
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200 

# force_build - flag para reiniciar la bd
def db_setup(force_rebuild=False):
    """Crea o reutiliza la base vectorial según el estado actual."""

    # Caso 1 — BD ya existe
    if os.path.exists(CHROMA_PATH):
        if force_rebuild:
            print("🔄 Reconstruyendo la base vectorial (force_rebuild=True)...")
            shutil.rmtree(CHROMA_PATH)
        else:
            print("📦 Base vectorial ya existe → Reutilizando.")
            return True 
        
    # Caso 2 — BD no existe o la estamos regenerando
    ingest_data()

def ingest_data():

    
    # Caso 2 — BD no existe o la estamos regenerando
    print("🚀 INICIANDO PROCESO DE INGESTA DE DATOS...")

    # 1. Verificar que el PDF existe
    if not os.path.exists(FILE_PATH):
        print(f"❌ ERROR: No encuentro el archivo '{FILE_PATH}'")
        print("   -> Asegúrate de que el PDF del paper está en la carpeta 'data' y se llama 'paper_refrag.pdf'")
        return False

    # 2. Limpiar base de datos anterior (para empezar de cero siempre)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 Base de datos anterior eliminada (limpieza).")

    # 3. Cargar el PDF
    print("📄 Cargando documento PDF...")
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    print(f"   -> Documento cargado: {len(docs)} páginas.")

    # 4. Chunking (La parte creativa)
    print(f"✂️  Troceando texto (Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""], # Intenta no romper párrafos
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   -> ¡Éxito! Se han generado {len(chunks)} fragmentos (chunks).")

    # 5. Crear Embeddings y Guardar en ChromaDB
    print("💾 Generando vectores (esto puede tardar un poco)...")
    
    # Usamos "all-MiniLM-L6-v2" que es el estándar de oro gratuito y rápido (Source 13)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Creamos la BD y la guardamos en disco inmediatamente
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    # Forzar guardado (aunque en versiones nuevas es automático, es buena práctica)
    db.persist() 
    
    print(f"✅ Base de datos vectorial lista en: {CHROMA_PATH}")
    print(f"   -> Ejemplo de chunk: '{chunks[0].page_content[:100]}...'")
    return True