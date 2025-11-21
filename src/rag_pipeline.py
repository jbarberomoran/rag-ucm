from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retrieval import get_retriever

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
        model="models/gemma-3-27b-it", 
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
    
    # Limpiamos la respuesta (quitamos espacios extra)
    return response.content.strip()