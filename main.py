import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from src.ingestion import db_setup
from src.rag_pipeline import query_rag
from src.rag_pipeline import query_rag, verify_context_with_llm

# Cargar clave API
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

def main():
    print("🧪 INICIANDO EXPERIMENTO RAG UCM...")

    # Archivos de salida
    FINAL_FILE = "./results/resultados_finales.csv"
    PARTIAL_FILE = "./results/resultados_parciales.csv"

    # --- CONFIGURACIÓN DE TIEMPOS (CONSTANTES) ---
    SLEEP_TIME = 5
    
    # --- 1. LIMPIEZA INICIAL ---
    # Borramos el final anterior
    if os.path.exists(FINAL_FILE):
        os.remove(FINAL_FILE)
        
    # Borramos el parcial anterior para empezar el log de cero
    if os.path.exists(PARTIAL_FILE):
        os.remove(PARTIAL_FILE)

    # 2. Cargar Dataset de Preguntas
    # --- MANEJO BASE DE DATOS: Comprobar si existe una o no, y generarla si se pide o necesario
        db_setup()   #Poner true para automatizar y sacar muestreo

    # 1. Cargar Dataset de Preguntas [cite: 20]
    path_json = "./data/questions.json"
    if not os.path.exists(path_json):
        print("❌ ERROR: No encuentro 'data/questions.json'")
        return

    with open(path_json, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    # --- CONFIGURACIÓN ---
    questions_to_run = questions[10:22]  # Descomentar para pruebas rápidas
    # questions_to_run = questions      # Descomentar para EXAMEN COMPLETO (70 preguntas)
    
    methods = ["baseline", "bm25", "dense", "hybrid"]
    results = []

   # --- CALENTAMIENTO (WARM-UP) ---
    print("\n CALENTANDO MOTORES (Cargando índices en memoria)...")
    for method in methods:
        try:
            # Pregunta dummy para activar @lru_cache
            _ = query_rag("Warm up", {"A":".","B":".","C":".","D":"."}, method, API_KEY)
        except:
            pass
    print("✅ Calentamiento completado. Empezando...\n")

    # 3. Bucle Principal
    print(f"🚀 Evaluando {len(questions_to_run)} preguntas con modelos: {methods}")
    
    for i, q in enumerate(questions_to_run):
        print(f"\n--- Q{i+1}: {q['question'][:50]}... ---")
        
        for method in methods:
            try:
                print(f"   [{method.upper()}] Procesando...", end=" ")
                
                # CRONÓMETRO USUARIO
                start_ts = time.time()
                
                # A) LLAMADA RESPUESTA
                raw_answer, retrieved_docs = query_rag(q['question'], q['answers'], method, API_KEY)
                
                # Paramos el reloj (Latencia de usuario)
                user_latency = time.time() - start_ts
                
                # PAUSA ANTI-429 (Vital para Gemma/Gemini Free)
                time.sleep(SLEEP_TIME) 

                # B) LIMPIEZA (REGEX)
                import re
                match = re.search(r'(?i)\b([A-D])\b', raw_answer)
                predicted_letter = match.group(1).upper() if match else "X"
                
                # C) EVALUACIÓN BÁSICA
                correct_letter = q['correct_answer']
                is_correct = (predicted_letter == correct_letter)
                
                # D) EL JUEZ (LLM-as-a-Judge) - Solo si acertó
                context_hit = False 
                if is_correct and method != "baseline":
                    correct_text = q['answers'][correct_letter]
                    context_hit = verify_context_with_llm(q['question'], correct_text, retrieved_docs, API_KEY)
                    # PAUSA ANTI-429 EXTRA (Porque hemos llamado al juez)
                    time.sleep(SLEEP_TIME)
                
                # Clasificación para la consola
                status = "❌ FALLO"
                if is_correct:
                    if method == "baseline": status = "🧠 MEMORIA"
                    elif context_hit: status = "✅ RAG VERIFICADO"
                    else: status = "⚠️ SUERTE"
                
                print(f"{status} (Pred: {predicted_letter} | T: {user_latency:.2f}s)")
                
                # E) GUARDAR DATO EN LISTA (Memoria)
                row = {
                    "question_id": i+1,
                    "method": method,
                    "correct": is_correct,
                    "verified_rag": context_hit,
                    "status_label": status,
                    "predicted": predicted_letter,
                    "ground_truth": correct_letter,
                    "response_time": round(user_latency, 2),
                    "raw_output": raw_answer
                }
                results.append(row)

                # --- GUARDADO PARCIAL (APPEND) ---
                # Esto es lo que se actualiza pregunta a pregunta
                df_temp = pd.DataFrame([row])
                file_exists = os.path.isfile(PARTIAL_FILE)
                df_temp.to_csv(PARTIAL_FILE, mode='a', header=not file_exists, index=False)
                # ---------------------------------
                
            except Exception as e:
                print(f"❌ Error crítico en {method}: {e}")


    # 4. Exportar Resultados y Resumen
    if not os.path.exists("./results"):
        os.makedirs("./results")
        
    df = pd.DataFrame(results)
    df.to_csv(FINAL_FILE, index=False)
    
    print("\n" + "="*30)
    print("📊 RESUMEN DE PRECISIÓN (ACCURACY)")
    print("="*30)
    # Calcula el porcentaje de aciertos por método
    print(df.groupby("method")["correct"].mean() * 100)
    print(f"📁 Resultados parciales (log): {PARTIAL_FILE}")
    print(f"📁 Resultados finales (clean): {FINAL_FILE}")

if __name__ == "__main__":
    main()