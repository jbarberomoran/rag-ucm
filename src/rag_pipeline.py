"""RAG answer generation and deterministic evidence verification."""

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import MODEL_NAME, SUPPORTED_METHODS
from src.domain import normalize_text, verify_evidence
from src.retrieval import RetrievalEngine

RAG_TEMPLATE = """
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
1. Analyze the text deeply to find evidence for each option.
2. Discard options that are not supported by the text.
3. Select the single correct option (A, B, C, or D).
4. Output ONLY the single letter.
5. Do not explain your reasoning or use punctuation.
"""

PROMPT = PromptTemplate(
    template=RAG_TEMPLATE,
    input_variables=["context", "question", "option_a", "option_b", "option_c", "option_d"],
)

BASELINE_PROMPT = PromptTemplate.from_template("""
Answer this multiple-choice question using your internal knowledge.
QUESTION: {question}
OPTIONS:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Output ONLY the single correct letter (A, B, C, or D), without explanation.
""")


def query_rag(question, options, method, api_key, *, engine=None, llm_factory=None):
    """Answer one multiple-choice question with the selected retrieval method."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required")
    if set(options) != {"A", "B", "C", "D"}:
        raise ValueError("options must contain exactly A, B, C, and D")

    relevant_docs = []

    if method == "baseline":
        context_text = "NO CONTEXT AVAILABLE. Use your internal knowledge."
    elif method == "cross_encoder":
        engine = engine or RetrievalEngine.get_instance()
        initial_retriever = engine.get_retriever(method="hybrid", k=20)
        candidate_docs = initial_retriever.invoke(question)
        relevant_docs = engine.rerank_documents(question, candidate_docs, top_k=5)
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
    else:
        engine = engine or RetrievalEngine.get_instance()
        retriever = engine.get_retriever(method=method, k=5)
        relevant_docs = retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

    factory = llm_factory or ChatGoogleGenerativeAI
    llm = factory(model=MODEL_NAME, google_api_key=api_key, temperature=0)
    prompt = BASELINE_PROMPT if method == "baseline" else PROMPT
    formatted_prompt = prompt.format(
        context=context_text,
        question=question,
        option_a=options["A"],
        option_b=options["B"],
        option_c=options["C"],
        option_d=options["D"],
    )
    response = llm.invoke(formatted_prompt)
    return response.content.strip(), relevant_docs


# Backwards-compatible names used by the accompanying notebook.
super_clean = normalize_text
verify_ground_truth_v1 = verify_evidence
