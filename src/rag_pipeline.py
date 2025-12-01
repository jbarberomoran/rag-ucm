from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retrieval import RetrievalEngine 
from difflib import SequenceMatcher
import time

# Modelo usado
MODEL_NAME = "models/gemini-2.5-flash-lite" # O "gemini-2.5-flash" si quieres asegurar


    # --- EL SUPER LIMPIADOR ---
def super_clean(text):
    # 1. Pasar a minúsculas
    text = text.lower()
    # 2. Eliminar saltos de línea y tabulaciones explícitamente
    text = text.replace("\n", "").replace("\r", "").replace("\t", "")
    # 3. Eliminar CUALQUIER carácter que no sea letra o número
    # Esto convierte "accu- racy" en "accuracy"
    return "".join([c for c in text if c.isalnum()])



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
            # Pedimos k=20 para asegurar que la respuesta esté ahí dentro
            initial_retriever = engine.get_retriever(method="hybrid", k=20)
            candidate_docs = initial_retriever.invoke(question)
        
            # PASO 2: Fine-Grained Reranking (Filtramos a los mejores)
            # Nos quedamos con los 5 mejores para Gemini (menos ruido = más acierto)
            relevant_docs = engine.rerank_documents(question, candidate_docs, top_k=5)
        
        else:
            # Buscamos los 8 fragmentos más relevantes usando el método elegido
            retriever = engine.get_retriever(method=method, k=5)
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
    
    return response.content.strip(), relevant_docs

# JUEZ V1: COINCIDENCIA DE TEXTO SIMPLE
def verify_ground_truth_v1(retrieved_docs, ground_truth_ref, threshold=0.5):
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

# PROMPT DEL JUEZ (Solo valida coincidencia de significado)
JUDGE_PROMPT = """
You are an objective evaluator text comparator.
Check if the RETRIEVED CONTEXT contains the semantic information present in the GROUND TRUTH.

GROUND TRUTH:
{reference}

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
1. Ignore OCR errors, line breaks, or minor spelling mistakes.
2. Ignore extra information in the context.
3. If the core fact of the Ground Truth is present in the Context, answer YES.
4. Otherwise, answer NO.

ANSWER (YES/NO):
"""

JUDGE_PROMPT2 ="""
You are a strict text comparison engine, not an interpreter.
Your task is to verify if the GROUND TRUTH text passage is physically present within the RETRIEVED CONTEXT.

GROUND TRUTH (Target Text):
{reference}

RETRIEVED CONTEXT (Search Space):
{context}

INSTRUCTIONS:
1. **LITERAL MATCH ONLY:** Do not evaluate meaning. Do not accept synonyms or paraphrasing. The words must be the same.
2. **FORMATTING TOLERANCE:** You MUST ignore ONLY the following artifacts caused by PDF extraction:
   - Line breaks (`\\n`) or carriage returns.
   - Multiple spaces or tabs.
   - Hyphenation at the end of lines (e.g., "algo- rithm" matches "algorithm").
   - OCR artifacts (e.g., "fi" read as "f i").
3. **DECISION:**
   - If the *sequence of words* in Ground Truth appears in the Context (ignoring the formatting issues above), answer **YES**.
   - If the wording is different, even if the meaning is identical, answer **NO**.

ANSWER (YES/NO):
"""

# JUEZ 
def verify_ground_truth_v2(paper_ref, retrieved_docs, api_key=None):
    """
    Verifica si la referencia está en los docs.
    Usa una estrategia en cascada: String Match -> LLM Match.
    """
    # Si no hay documentos (caso Baseline), obviamente no está justificado en el texto
    if not retrieved_docs:
        return False, 0.0

    # Limpiamos la referencia
    ref_clean = super_clean(paper_ref)
    
    # Unimos todo el contexto recuperado y lo limpiamos igual
    full_context = "".join([doc.page_content for doc in retrieved_docs])
    context_clean = super_clean(full_context)

    # Configuramos un modelo 'Flash' barato para juzgar rápido
    llm_judge = ChatGoogleGenerativeAI(
        model=MODEL_NAME, # Usa el mismo modelo que tengas disponible
        google_api_key=api_key,
        temperature=0.0
    )
    
    # Preguntamos al juez
    #formatted_prompt = JUDGE_PROMPT.format(
    #    reference=ref_clean,
    #    context=context_clean
    #)

    formatted_prompt = JUDGE_PROMPT2.format(
        reference=ref_clean,
        context=context_clean
    )
    
    try:
        verdict = llm_judge.invoke(formatted_prompt).content.strip().upper()
        # Limpiamos por si responde si en varias formas
        return "YES" in verdict.upper()
    except Exception as e:
        print(f"   [Juez] Error: {e}")
        return False
    

def verify_ground_truth_v3(retrieved_docs, ground_truth_ref, api_key=None, threshold=0.5):
    """
    Verifica si la referencia está en los docs usando:
    1. Búsqueda exacta
    2. Fuzzy matching
    3. Juez LLM (solo si las anteriores fallan)

    Se llama al LLM Judge solo si la respuesta es correcta pero se acierta por suerte
    """
    if not ground_truth_ref:
        return False, 0.0  # Nada que comparar

    # PREPARACIÓN DE TEXTOS
    ref_clean = super_clean(ground_truth_ref)
    full_context = "".join([doc.page_content for doc in retrieved_docs])
    context_clean = super_clean(full_context)

    # 1. MATCH EXACTO
    if ref_clean in context_clean:
        return True, 1.0  # match perfecto

    # 2. FUZZY MATCH 
    matcher = SequenceMatcher(None, ref_clean, context_clean)
    match = matcher.find_longest_match(0, len(ref_clean), 0, len(context_clean))
    similarity = match.size / len(ref_clean)

    if similarity >= threshold:
        return True, similarity  # match aproximado aceptable

    # 3. LLM JUDGE
    llm_judge = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0
    )

    formatted_prompt = JUDGE_PROMPT.format(
        reference=ref_clean,
        context=context_clean
    )

    try:
        verdict = llm_judge.invoke(formatted_prompt).content.strip().upper()
        llm_result = "YES" in verdict
        if llm_result: similarity = 0.95 # Si el LLM dice que sí, asumimos match perfecto

        # devolvemos el veredicto del LLM y la similitud del fuzzy
        return llm_result, similarity

    except Exception as e:
        print(f"[Juez] Error LLM: {e}")
        return False, similarity
    

    #Esto es lo mismo que lo de arriba, el v1. No se si quieres hacer algo
"""# JUEZ V1: COINCIDENCIA DE TEXTO SIMPLE
def verify_ground_truth_v1(retrieved_docs, ground_truth_ref, threshold=0.5):
    
    Comprueba si el párrafo de referencia ('paper_reference') está contenido
    dentro de los documentos recuperados, permitiendo ligeras variaciones.
    
    Args:
        retrieved_docs: Lista de documentos recuperados.
        ground_truth_ref: El texto original del JSON (La verdad absoluta).
        threshold: 0.5 significa que al menos el 50% del texto debe coincidir.
    
    Returns:
        tuple: (bool: Encontrado/No, float: Puntuación de similitud)
    
    if not ground_truth_ref:
        return False, 0.0

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
    
    return similarity >= threshold, similarity"""