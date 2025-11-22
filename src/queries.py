import os
import json
import time
import pandas as pd
import re
from src.rag_pipeline import query_rag, verify_context_with_llm


    # --- CONFIGURACIÓN DE TIEMPOS (CONSTANTES) ---
SLEEP_TIME = 5

def run_questions(questions_slice=None, methods=None, api_key=None, partial_file="./results/resultados_parciales.csv", sleep_time=SLEEP_TIME):
    """
    Ejecuta preguntas del dataset y devuelve resultados.

    Args:
        questions_slice (list/dict): Lista de preguntas a evaluar. Si None, se carga todo el dataset.
        methods (list): Lista de métodos a usar. Default ["baseline", "bm25", "dense", "hybrid"]
        api_key (str): API Key para el LLM
        partial_file (str): Ruta donde guardar resultados parciales
        sleep_time (int): Segundos de pausa entre consultas

    Returns:
        pd.DataFrame: DataFrame con resultados de todas las preguntas y métodos
    """
    if methods is None:
        methods = ["baseline", "bm25", "dense", "hybrid"]

    # 1. Cargar dataset si no se pasa
    path_json = "./data/questions.json"
    if not os.path.exists(path_json):
        print("❌ ERROR: No encuentro 'data/questions.json'")
        return
    with open(path_json, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if questions_slice is None:
        # Por defecto, solo un subconjunto para pruebas
        questions_slice = questions
    else:
        questions_slice = questions[0:5]

    # Calentamiento
    print("\nCALENTANDO MOTORES (Cargando índices en memoria)...")
    for method in methods:
        try:
                # Pregunta dummy para activar @lru_cache
            _ = query_rag("Warm up", {"A":".","B":".","C":".","D":"."}, method, api_key)
        except:
            pass
    print("✅ Calentamiento completado. Empezando...\n")

    results = []

    # Bucle principal
    print(f"🚀 Evaluando {len(questions_slice)} preguntas con modelos: {methods}")

    for i, q in enumerate(questions_slice):
        print(f"\n--- Q{i+1}: {q['question'][:50]}... ---")
        for method in methods:
            try:
                print(f"   [{method.upper()}] Procesando...", end=" ")

                start_ts = time.time()

                # LLAMADA RESPUESTA
                raw_answer, retrieved_docs = query_rag(q['question'], q['answers'], method, api_key)

                # Latencia de usuario y pausa anit-429 para Gemma/Gemini Free
                latency = time.time() - start_ts
                time.sleep(sleep_time)

                # Limpieza de respuesta
                match = re.search(r'(?i)\b([A-D])\b', raw_answer)
                predicted_letter = match.group(1).upper() if match else "X"

                # Evaluación básica
                correct_letter = q['correct_answer']
                is_correct = (predicted_letter == correct_letter)

                # Juez LLM
                context_hit = False
                if is_correct and method != "baseline":
                    correct_text = q['answers'][correct_letter]
                    context_hit = verify_context_with_llm(q['question'], correct_text, retrieved_docs, api_key)
                    time.sleep(sleep_time)

                # Clasificación
                status = "❌ FALLO"
                if is_correct:
                    if method == "baseline": status = "🧠 MEMORIA"
                    elif context_hit: status = "✅ RAG VERIFICADO"
                    else: status = "⚠️ SUERTE"

                print(f"{status} (Pred: {predicted_letter} | T: {latency:.2f}s)")

                # Guardar resultado
                row = {
                    "question_id": i+1,
                    "method": method,
                    "correct": is_correct,
                    "verified_rag": context_hit,
                    "status_label": status,
                    "predicted": predicted_letter,
                    "ground_truth": correct_letter,
                    "response_time": round(latency, 2),
                    "raw_output": raw_answer
                }
                results.append(row)

                # Guardado parcial
                df_temp = pd.DataFrame([row])
                os.makedirs(os.path.dirname(partial_file), exist_ok=True)
                file_exists = os.path.isfile(partial_file)
                df_temp.to_csv(partial_file, mode='a', header=not file_exists, index=False)

            except Exception as e:
                print(f"❌ Error crítico en {method}: {e}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Ejemplo de uso directo
    from dotenv import load_dotenv
    load_dotenv()
    import os
    API_KEY = os.getenv("GOOGLE_API_KEY")
    df_results = run_questions(api_key=API_KEY)
    print(df_results.groupby("method")["correct"].mean() * 100)
