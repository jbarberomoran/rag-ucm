import gc
import warnings

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import CHROMA_PATH, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME


# --- CONFIGURACIÓN ---
class RetrievalEngine:
    _instance = None

    def __init__(self):
        """
        Constructor privado (simulado). 
        En Python no se puede hacer privado real, pero si alguien llama a 
        RetrievalEngine() directamente, creará una instancia nueva desconectada 
        del Singleton. Por eso usaremos get_instance().
        """
        # Inicializamos atributos en None (Lazy)
        self._db = None
        self._embeddings = None
        self._bm25_retriever = None
        self._reranker = None

    @classmethod
    def get_instance(cls):
        """
        Equivalente a: public static RetrievalEngine getInstance()
        """
        # 1. Si no existe la instancia, la creamos (Lazy Creation)
        if cls._instance is None:
            cls._instance = RetrievalEngine()
        
        # 2. Devolvemos la instancia almacenada
        return cls._instance

    @property
    def db(self):
        """
        Getter inteligente. Aquí está el truco para que no explote en Windows.
        Se conecta solo cuando le pides la DB.
        """
        if self._db is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self._db = Chroma(
                persist_directory=str(CHROMA_PATH), embedding_function=self._embeddings
            )
        return self._db

    def unload_db(self):
        """Método para desconectar manualmente"""
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
        """Construye o devuelve el índice BM25 cacheado."""
        if self._bm25_retriever is not None:
            return self._bm25_retriever
        
        try:
            # Sacamos todos los documentos de Chroma para crear el índice inverso
            raw_data = self.db.get()
            texts = raw_data["documents"]
            metadatas = raw_data["metadatas"]
            
            if not texts:
                print("⚠️  ADVERTENCIA: La base de datos está vacía.")
                return None
                
            docs_obj = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas, strict=True)
            ]
            
            self._bm25_retriever = BM25Retriever.from_documents(docs_obj)
            return self._bm25_retriever
            
        except Exception as e:
            print(f"❌ Error construyendo BM25: {e}")
            return None

    def get_retriever(self, method, k=4):
        """
        Función principal para obtener el retriever configurado.
        
        Args:
            method (str): "dense", "bm25", o "hybrid"
            k (int): Número de documentos a recuperar
        """
        
        if method not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Unsupported retrieval method: {method}")
        if k < 1:
            raise ValueError("k must be at least 1")

        # 1. Retriever Denso (Vectorial) - Siempre disponible desde self.db
        dense_retriever = self.db.as_retriever(search_kwargs={"k": k})
        
        if method == "dense":
            return dense_retriever
            
        # 2. Retriever BM25
        bm25_retriever = self._get_bm25_retriever()
        if bm25_retriever is None:
            raise RuntimeError("Cannot build BM25 retriever from an empty vector database")
            
        # Actualizamos K dinámicamente en el objeto cacheado
        bm25_retriever.k = k
        
        if method == "bm25":
            return bm25_retriever
            
        # 3. Híbrido (Ensemble)
        if method == "hybrid":
            # 50% de peso a cada uno
            return EnsembleRetriever(
                retrievers=[bm25_retriever, dense_retriever],
                weights=[0.5, 0.5]
            )
            
        raise AssertionError("unreachable")
    
    @property
    def reranker(self):
        """Carga el modelo Cross-Encoder solo si se necesita."""
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL_NAME)
        return self._reranker

    # RE-RANKING
    def rerank_documents(self, query, docs, top_k=5):   
        """
        Recibe una lista de documentos candidatos, los puntúa contra la query
        y devuelve los top_k mejores.
        """
        if not docs:
            return []
            
        # 1. Preparamos los pares
        pairs = [[query, doc.page_content] for doc in docs]
        
        # 2. Obtenemos las puntuaciones
        scores = self.reranker.predict(pairs)
        
        # 3. Ordenamos de mayor a menor puntuación
        docs_with_scores = sorted(
            zip(docs, scores, strict=True), key=lambda item: item[1], reverse=True
        )
        
        # 4. Devolvemos solo los objetos Document del top_k
        return [doc for doc, _score in docs_with_scores[:top_k]]
