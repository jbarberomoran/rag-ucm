import gc
import warnings

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# --- CONFIGURACIÓN ---
CHROMA_PATH = "./data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self._db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self._embeddings)
        return self._db

    def unload_db(self):
        """Método para desconectar manualmente"""
        if self._db is not None:
            self._db = None
            self._embeddings = None
            gc.collect()

    def _get_bm25_retriever(self):
        """Construye o devuelve el índice BM25 cacheado."""
        if self._bm25_retriever is not None:
            return self._bm25_retriever
        
        try:
            # Sacamos todos los documentos de Chroma para crear el índice inverso
            # Nota: Esto puede ser lento si hay gigas de datos, para tu paper está bien.
            raw_data = self.db.get()
            texts = raw_data['documents']
            metadatas = raw_data['metadatas']
            
            if not texts:
                print("⚠️  ADVERTENCIA: La base de datos está vacía.")
                return None
                
            docs_obj = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
            
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
        
        # 1. Retriever Denso (Vectorial) - Siempre disponible desde self.db
        dense_retriever = self.db.as_retriever(search_kwargs={"k": k})
        
        if method == "dense":
            return dense_retriever
            
        # 2. Retriever BM25 (Palabras Clave)
        bm25_retriever = self._get_bm25_retriever()
            
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
            
        # Default fallback
        return dense_retriever
    
    @property
    def reranker(self):
        """Carga el modelo Cross-Encoder solo si se necesita."""
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    # RE-RANKING
    def rerank_documents(self, query, docs, top_k=5):   
        """
        Recibe una lista de documentos candidatos, los puntúa contra la query
        y devuelve los top_k mejores.
        """
        if not docs: return []
            
        # 1. Preparamos los pares [Query, Documento]
        pairs = [[query, doc.page_content] for doc in docs]
        
        # 2. Obtenemos las puntuaciones (scores)
        scores = self.reranker.predict(pairs)
        
        # 3. Ordenamos de mayor a menor puntuación
        docs_with_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # 4. Devolvemos solo los objetos Document del top_k
        final_docs = [doc for doc, score in docs_with_scores[:top_k]]
        return final_docs