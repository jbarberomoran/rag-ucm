import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from src.rag_pipeline import query_rag
from src.rag_pipeline import query_rag, verify_context_with_llm

# Cargar clave API
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

def run_experiment():
    print("🧪 INICIANDO EXPERIMENTO RAG UCM...")
    
    # --- LIMPIEZA INICIAL: Borrar backup anterior ---
    backup_path = "./results/resultados_parciales.csv"
    if os.path.exists(backup_path):
        os.remove(backup_path)

    # 1. Cargar Dataset de Preguntas [cite: 20]
    path_json = "./data/questions.json"
    if not os.path.exists(path_json):
        print("❌ ERROR: No encuentro 'data/questions.json'")
        return

    with open(path_json, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    # --- CONFIGURACIÓN DE LA PRUEBA ---
    # Para probar rápido, usa solo las 2 primeras. 
    # Para el final, comenta esta línea:
    questions_to_run = questions[:4]  
    # questions_to_run = questions # <-- Descomenta esta para correr las 50 preguntas
    
    # Métodos a comparar [cite: 32]
    # Puedes añadir "dense" y "bm25" a la lista si quieres comparar los 4
    methods = ["baseline", "bm25", "dense", "hybrid"] 
    
    results = []

    # 2. Bucle Principal
    print(f"🚀 Evaluando {len(questions_to_run)} preguntas con modelos: {methods}")
    
    for i, q in enumerate(questions_to_run):
        print(f"\n--- Q{i+1}: {q['question'][:50]}... ---")
        
        for method in methods:
            try:
                print(f"   [{method.upper()}] Procesando...", end=" ")
                start_ts = time.time()
                
                # 1. LLAMADA AL CEREBRO (Responder)
                raw_answer, retrieved_docs = query_rag(q['question'], q['answers'], method, API_KEY)
                
                # 2. LIMPIEZA DE RESPUESTA
                import re
                match = re.search(r'(?i)\b([A-D])\b', raw_answer)
                predicted_letter = match.group(1).upper() if match else "X"
                
                # 3. EVALUACIÓN (¿Acertó?)
                correct_letter = q['correct_answer']
                is_correct = (predicted_letter == correct_letter)
                
                # 4. EL JUICIO (Solo si acertó y no es baseline)
                context_hit = False # Por defecto asumimos que no
                
                if is_correct and method != "baseline":
                    # ¡Llamamos al segundo LLM para verificar!
                    # Le pasamos la pregunta, la respuesta correcta (texto) y los docs
                    correct_text = q['answers'][correct_letter]
                    context_hit = verify_context_with_llm(q['question'], correct_text, retrieved_docs, API_KEY)
                
                elapsed = time.time() - start_ts

                # CLASIFICACIÓN FINAL
                status = "❌ FALLO"
                if is_correct:
                    if method == "baseline":
                        status = "🧠 MEMORIA" # Acertó sin documentos
                    elif context_hit:
                        status = "✅ RAG VERIFICADO" # Acertó y el texto lo respaldaba
                    else:
                        status = "⚠️ SUERTE" # Acertó pero el texto no tenía la info (Alucinación positiva)
                
                print(f"{status} (Pred: {predicted_letter})")
                
                # 5. GUARDAR DATOS
                row = {
                    "question_id": i+1,
                    "method": method,
                    "correct": is_correct,
                    "verified_rag": context_hit, # <--- Métrica Premium
                    "status_label": status,
                    "predicted": predicted_letter,
                    "ground_truth": correct_letter,
                    "response_time": round(elapsed, 2)
                }
                results.append(row)

                # Guardado de seguridad
                df_temp = pd.DataFrame([row])
                file_exists = os.path.isfile("./results/resultados_finales.csv")
                df_temp.to_csv("./results/resultados_finales.csv", mode='a', header=not file_exists, index=False)
                
                time.sleep(3) 
                
            except Exception as e:
                print(f"❌ Error crítico en {method}: {e}")


    # 3. Exportar Resultados y Resumen
    if not os.path.exists("./results"):
        os.makedirs("./results")
        
    df = pd.DataFrame(results)
    df.to_csv("./results/resultados_finales.csv", index=False)
    
    print("\n" + "="*30)
    print("📊 RESUMEN DE PRECISIÓN (ACCURACY)")
    print("="*30)
    # Calcula el porcentaje de aciertos por método
    print(df.groupby("method")["correct"].mean() * 100)
    print(f"\n📁 Resultados detallados guardados en: ./results/resultados_finales.csv")

if __name__ == "__main__":
    run_experiment()