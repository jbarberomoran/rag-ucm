from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retrieval import get_retriever

# Modelo usado
MODEL_NAME = "models/gemma-3-27b-it" # O "gemini-2.5-flash" si quieres asegurar

# PLANTILLA DEL PROMPT [cite: 38]
# Instruimos al modelo para que actúe como experto y cite fuentes.
rag_template = """
You are a technical expert taking an exam based on a research paper.
Use ONLY the provided context to answer the question.

CONTEXT FROM PAPER:
{context}

---
QUESTION: {question}

OPTIONS:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

INSTRUCTIONS:
1. Analyze the context carefully.
2. You MUST select one option (A, B, C, or D).
3. If the answer is not in the context, you MUST GUESS logically. Do not refuse to answer.
4. OUTPUT FORMAT: Start your response with the letter (A, B, C, or D) and nothing else.
"""

prompt = PromptTemplate(
    template=rag_template,
    input_variables=["context", "question", "option_a", "option_b", "option_c", "option_d"]
)

def query_rag(question, options, method, api_key):
    """
    Ejecuta el ciclo RAG completo para una pregunta.
    """
    # 1. Obtener Contexto (Si no es Baseline)
    if method == "baseline":
        # [cite: 33] Baseline: El modelo responde "de memoria" (alucinará o acertará por suerte)
        context_text = "NO CONTEXT AVAILABLE. Use your internal knowledge."
    else:
        # Buscamos los 4 fragmentos más relevantes usando el método elegido
        retriever = get_retriever(method=method, k=4)
        relevant_docs = retriever.invoke(question)
        # Unimos los fragmentos en un solo texto
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