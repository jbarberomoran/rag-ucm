from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retrieval import RetrievalEngine 
from difflib import SequenceMatcher
import time

# Modelo usado
MODEL_NAME = "models/gemini-2.0-flash" # O "gemini-2.5-flash" si quieres asegurar

# PLANTILLA DEL PROMPT [cite: 38]
# Instruimos al modelo para que actúe como experto y cite fuentes.
rag_template = """
You are a strict exam grading machine.
Answer the multiple-choice question based ONLY on the provided context.

CONTEXT FROM PAPER:
{context}

QUESTION: 
{question}

OPTIONS:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

INSTRUCTIONS:
1. Analyze the text deepy to find evidence for each option.
2. Discard options that are not supported by the text.
3. Select the single correct option (A, B, C, or D).
4. CRITICAL: Output ONLY the single letter.
5. Do NOT write "The answer is...". Do NOT explain your reasoning. Do NOT use punctuation.

EXAMPLE OUTPUT:
A
"""

prompt = PromptTemplate(
    template=rag_template,
    input_variables=["context", "question", "option_a", "option_b", "option_c", "option_d"]
)

def query_rag(question, options, method, api_key):
    """
    Ejecuta el ciclo RAG completo para una pregunta.
    """
    engine = RetrievalEngine.get_instance()
    relevant_docs = []
    
    # 1. Obtener Contexto (Si no es Baseline)
    if method == "baseline":
        # [cite: 33] Baseline: El modelo responde "de memoria" (alucinará o acertará por suerte)
        context_text = "NO CONTEXT AVAILABLE. Use your internal knowledge."

    else:
        if method == "cross_encoder":
            # PASO 1: Broad Retrieval (Traemos MUCHOS candidatos)
            # Usamos Hybrid porque es el mejor "cazador" inicial
            # Pedimos k=10 para asegurar que la respuesta esté ahí dentro
            initial_retriever = engine.get_retriever(method="hybrid", k=10)
            candidate_docs = initial_retriever.invoke(question)
        
            # PASO 2: Fine-Grained Reranking (Filtramos a los mejores)
            # Nos quedamos con los 5 mejores para Gemini (menos ruido = más acierto)
            relevant_docs = engine.rerank_documents(question, candidate_docs, top_k=5)
        
        else:
            # Buscamos los 8 fragmentos más relevantes usando el método elegido
            retriever = engine.get_retriever(method=method, k=8)
            relevant_docs = retriever.invoke(question)

        # Unimos el texto de los chunks recuperados
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
    # 2. Configurar el LLM (Gemini)
    # Usamos temperature=0 para resultados reproducibles [cite: 47]
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, 
        google_api_key=api_key,
        temperature=0
    )

    # 3. Rellenar la plantilla con los datos reales
    formatted_prompt = prompt.format(
        context=context_text,
        question=question,
        option_a=options["A"],
        option_b=options["B"],
        option_c=options["C"],
        option_d=options["D"]
    )

    # 4. Enviar a Google y obtener respuesta
    response = llm.invoke(formatted_prompt)
    
    # Devolvemos la respuesta Y TAMBIÉN los documentos usados (si los hubo)
    # Si es baseline, relevant_docs no existe, devolvemos lista vacía
    docs_used = relevant_docs if method != "baseline" else []
    
    return response.content.strip(), docs_used

# --- EL JUEZ (LLM-as-a-Judge) ---
judge_template = """
You are an impartial grader evaluating a RAG system.
Your job is to determine if the provided CONTEXT contains sufficient information to answer the QUESTION correctly.

QUESTION: {question}
CORRECT ANSWER: {correct_answer}
RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
1. Read the context and the question.
2. Determine if the context justifies the correct answer.
3. If the context mentions the answer explicitly or implicitly, return "YES".
4. If the context is irrelevant or misses the key fact, return "NO".
5. Response format: Just one word ("YES" or "NO").
"""

judge_prompt = PromptTemplate(
    template=judge_template,
    input_variables=["question", "correct_answer", "context"]
)

def verify_context_with_llm(question, correct_answer, docs, api_key):
    """
    Consulta al LLM si los documentos recuperados realmente justifican la respuesta.
    Devuelve True si el contexto es válido, False si fue suerte.
    """
    # Si no hay documentos (caso Baseline), obviamente no está justificado en el texto
    if not docs:
        return False

    # Unimos el texto de los chunks
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Configuramos un modelo 'Flash' barato para juzgar rápido
    llm_judge = ChatGoogleGenerativeAI(
        model=MODEL_NAME, # Usa el mismo modelo que tengas disponible
        google_api_key=api_key,
        temperature=0
    )

    # Preguntamos al juez
    formatted_prompt = judge_prompt.format(
        question=question,
        correct_answer=correct_answer,
        context=context_text
    )
    
    try:
        verdict = llm_judge.invoke(formatted_prompt).content.strip().upper()
        # Limpiamos por si responde "YES." o "Answer: YES"
        return "YES" in verdict
    except Exception as e:
        print(f"   [Juez] Error: {e}")
        return False
    
def verify_ground_truth(retrieved_docs, ground_truth_ref, threshold=0.5):
    """
    Comprueba si el párrafo de referencia ('paper_reference') está contenido
    dentro de los documentos recuperados, permitiendo ligeras variaciones.
    
    Args:
        retrieved_docs: Lista de documentos recuperados.
        ground_truth_ref: El texto original del JSON (La verdad absoluta).
        threshold: 0.5 significa que al menos el 50% del texto debe coincidir.
    
    Returns:
        tuple: (bool: Encontrado/No, float: Puntuación de similitud)
    """
    if not ground_truth_ref:
        return False, 0.0

    # --- EL SUPER LIMPIADOR ---
    def super_clean(text):
        # 1. Pasar a minúsculas
        text = text.lower()
        # 2. Eliminar saltos de línea y tabulaciones explícitamente
        text = text.replace("\n", "").replace("\r", "").replace("\t", "")
        # 3. Eliminar CUALQUIER carácter que no sea letra o número
        # Esto convierte "accu- racy" en "accuracy"
        return "".join([c for c in text if c.isalnum()])

    # Limpiamos la referencia
    ref_clean = super_clean(ground_truth_ref)
    
    # Unimos todo el contexto recuperado y lo limpiamos igual
    full_context = "".join([doc.page_content for doc in retrieved_docs])
    context_clean = super_clean(full_context)
    
    # DEBUG: Descomenta esto para ver por qué fallaba antes
    # print(f"\n[JUEZ] Ref: {ref_clean[:30]}... | Ctx: {context_clean[:30]}...")

    # 1. Búsqueda Exacta en la sopa de letras (Infalible si el texto está completo)
    if ref_clean in context_clean:
        return True, 1.0
        
    # 2. Fuzzy Match (Por si faltan letras o hay errores de OCR)
    matcher = SequenceMatcher(None, ref_clean, context_clean)
    match = matcher.find_longest_match(0, len(ref_clean), 0, len(context_clean))
    
    # Calculamos porcentaje
    similarity = match.size / len(ref_clean)
    
    return similarity >= threshold, similarity