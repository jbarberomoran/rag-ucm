import sys
import warnings

# Esto debe ir ANTES de importar cualquier cosa de LangChain o Src
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Chroma.*")

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from src.rag_pipeline import query_rag

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
    # questions_to_run = questions[:3]  
    questions_to_run = questions # <-- Descomenta esta para correr las 50 preguntas
    
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
                
                # Medimos el tiempo (Métrica extra para bonus) [cite: 43]
                start_ts = time.time()
                
                # LLAMADA PRINCIPAL AL CEREBRO
                raw_answer = query_rag(q['question'], q['answers'], method, API_KEY)
                
                elapsed = time.time() - start_ts
                
                # Limpieza: Si el modelo responde "A)", nos quedamos solo con "A"
                predicted_letter = raw_answer[0].upper() 
                correct_letter = q['correct_answer']
                
                # Evaluación: ¿Acertó? [cite: 42]
                is_correct = (predicted_letter == correct_letter)
                
                print(f"{'✅' if is_correct else '❌'} (Pred: {predicted_letter} | Real: {correct_letter})")
                
                # Guardar datos
                results.append({
                    "question_id": i+1,
                    "method": method,
                    "correct": is_correct,
                    "predicted": predicted_letter,
                    "ground_truth": correct_letter,
                    "response_time": round(elapsed, 2),
                    "raw_output": raw_answer
                })

                # Guardamos cada vez que terminamos un método. 
                # mode='a' (append) añade al final sin borrar lo anterior.
                # header=False evita repetir los títulos si el archivo ya existe.
                temp_df = pd.DataFrame([results[-1]])
                file_exists = os.path.isfile("./results/resultados_parciales.csv")
                temp_df.to_csv("./results/resultados_parciales.csv", mode='a', header=not file_exists, index=False)
                
                
                # Pausa ética para no saturar la API gratuita
                time.sleep(2)
                
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