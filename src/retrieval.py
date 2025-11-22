import warnings
# Silenciador local
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Chroma.*")

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# Configuración
CHROMA_PATH = "./data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modelo ligero y gratuito [cite: 13]

def get_retriever(method="hybrid", k=4):
    """
    Configura el sistema de recuperación según la estrategia elegida.
    Args:
        method: 'dense', 'bm25', o 'hybrid'
        k: Número de fragmentos de texto a recuperar (contexto)
    """
    # 1. Cargar Embeddings (El traductor texto -> números)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Filtramos por el texto exacto del error justo antes de llamar a Chroma
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # 2. Conectar con la Base de Datos Vectorial (Chroma)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # -- ESTRATEGIA 1: DENSE RETRIEVAL [cite: 35] --
    dense_retriever = db.as_retriever(search_kwargs={"k": k})
    
    if method == "dense":
        return dense_retriever

    # -- ESTRATEGIA 2: BM25 (KEYWORD SEARCH) [cite: 34] --
    # Truco: Sacamos los textos de Chroma para no volver a leer el PDF
    try:
        # Recuperamos todos los documentos guardados
        docs_data = db.get() 
        texts = docs_data['documents']
        metadatas = docs_data['metadatas']
        
        # Reconstruimos los objetos Document para que BM25 los entienda
        docs_obj = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        
        bm25_retriever = BM25Retriever.from_documents(docs_obj)
        bm25_retriever.k = k
    except Exception as e:
        print(f"❌ Error iniciando BM25 (¿Base de datos vacía?): {e}")
        return None

    if method == "bm25":
        return bm25_retriever

    # -- ESTRATEGIA 3: HYBRID (BONUS POINTS) [cite: 36] --
    if method == "hybrid":
        # Combinamos los resultados: 50% peso a vectores, 50% a palabras clave
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever

    else:
        raise ValueError(f"Método '{method}' no reconocido. Usa: dense, bm25, hybrid")

# Bloque de prueba rápida
if __name__ == "__main__":
    print("Probando buscador Híbrido...")
    try:
        r = get_retriever("hybrid")
        res = r.invoke("What is REFRAG?")
        print(f"✅ Éxito. Encontrados {len(res)} fragmentos.")
        print(f"   Fragmento 1: {res[0].page_content[:100]}...")
    except Exception as e:
        print(f"❌ Falló: {e}")