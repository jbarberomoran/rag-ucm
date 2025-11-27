import os
import json
import time
import pandas as pd
import re
from src.rag_pipeline import query_rag, verify_context_with_llm
from src.rag_pipeline import verify_ground_truth
from src.retrieval import RetrievalEngine


    # --- CONFIGURACIÓN DE TIEMPOS (CONSTANTES) ---
SLEEP_TIME = 2  # Segundos de espera entre consultas para evitar 429

def run_questions(questions_slice=None, methods=None, api_key=None, partial_file="./results/resultados_parciales.csv", sleep_time=SLEEP_TIME):
    """
    Ejecuta preguntas del dataset y devuelve resultados.

    Args:
        questions_slice (list[int]): Lista de preguntas a evaluar. Si None, se carga todo el dataset.
        methods (list): Lista de métodos a usar. Default ["baseline", "bm25", "dense", "hybrid", "cross_encoder"]
        api_key (str): API Key para el LLM
        partial_file (str): Ruta donde guardar resultados parciales
        sleep_time (int): Segundos de pausa entre consultas

    Returns:
        pd.DataFrame: DataFrame con resultados de todas las preguntas y métodos
    """
    if methods is None:
        methods = ["baseline", "bm25", "dense", "hybrid", "cross_encoder"]

    # 1. Cargar dataset si no se pasa
    path_json = "./data/questions.json"
    if not os.path.exists(path_json):
        print("❌ ERROR: No encuentro 'data/questions.json'")
        return
    with open(path_json, "r", encoding="utf-8") as f:
        questions = json.load(f)

    questions_to_run = []
    
    if questions_slice is None:
        # Por defecto, solo un subconjunto para pruebas
        questions_to_run = questions
    else:
        for idx in questions_slice:
            # Protección: Verificamos que el índice existe
            if 0 <= idx < len(questions):
                questions_to_run.append(questions[idx])
            else:
                print(f"⚠️ AVISO: El índice {idx} no existe (Dataset tiene {len(questions)} preguntas). Se omite.")

    results = []

    print("\n⚙️  Inicializando y calentando motores...")
    try:
        engine = RetrievalEngine.get_instance()
        
        # 1. Calentamiento de BUSCADORES (Dense, BM25, Hybrid)
        # El Cross-Encoder también usa Hybrid por debajo, así que necesita esto también.
        needs_bm25 = any(m in methods for m in ["bm25", "hybrid", "cross_encoder"])
        needs_dense = "dense" in methods
        
        if needs_bm25:
            print("   -> Construyendo índices Híbridos (BM25 + Vectores)...")
            # Al pedir 'hybrid', forzamos la carga de Chroma Y la construcción del índice BM25
            engine.get_retriever("hybrid", k=1)
            
        elif needs_dense:
            print("   -> Conectando a Base de Datos Vectorial...")
            # Si solo usamos dense, no perdemos tiempo construyendo BM25
            engine.get_retriever("dense", k=1)

        # 2. Calentamiento de MODELO DE IA (Cross-Encoder)
        if "cross_encoder" in methods:
            print("   -> Cargando modelo Cross-Encoder en RAM...")
            # Accedemos a la propiedad para disparar la carga
            _ = engine.reranker 
            
        print("\n✅ Todo listo")

    except Exception as e:
        print(f"⚠️ Error no crítico en calentamiento: {e}")
        print("   (El programa continuará, pero la primera pregunta podría ir lenta)")

    # Bucle principal
    print(f"\n🚀 Evaluando {len(questions_to_run)} preguntas con modelos: {methods}")

    for i, q in enumerate(questions_to_run):
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

                # 3. --- NUEVO: JUEZ DE GROUND TRUTH + LLM Judge ---
                paper_ref = q.get('paper_reference', "")
                found_evidence, evidence_score = False, 0.0

                if paper_ref:
                    # Llamamos a la función que creamos en el Paso 1
                    # from src.rag_pipeline import verify_ground_truth y 
                    found_evidence, evidence_score = verify_ground_truth(retrieved_docs, paper_ref)
                    if is_correct and not found_evidence:
                        found_evidence = verify_context_with_llm(q['question'], paper_ref, retrieved_docs, api_key)

                # 4. Clasificación del Resultado (Para tu Excel)
                status_tag = ""
                if is_correct and found_evidence:
                    status_tag = "✅ ACIERTO PERFECTO (RAG)"
                elif is_correct and not found_evidence:
                    status_tag = "⚠️ ACIERTO SUERTE (Sin Evidencia)"
                elif not is_correct and found_evidence:
                    status_tag = "📉 FALLO RAZONAMIENTO (Contexto OK)"
                else:
                    status_tag = "❌ FALLO TOTAL"

                print(f"{status_tag} (Pred: {predicted_letter} | T: {latency:.2f}s)")

                # Guardar resultado
                row = {
                    "question_id": i+1,
                    "method": method,
                    "correct": is_correct,
                    "predicted": predicted_letter,
                    "ground_truth": correct_letter,
                    "response_time": round(latency, 2),
                    "raw_output": raw_answer,
                    "status": status_tag,
                    "retrieval_score": evidence_score,
                    "retrieved_docs": retrieved_docs
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
