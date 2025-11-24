import os
from dotenv import load_dotenv
import pandas as pd
from src.evaluation import generate_dashboard, evaluate_results
from src.launcher import setup_enviroment
from src.queries import run_questions

# Cargar clave API
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


# Archivos de salida
FINAL_FILE = "./results/resultados importantes/24-11-4.csv"
PARTIAL_FILE = "./results/resultados_parciales.csv"

def multiple_runs(n = 10):
    print("\n🧪 INICIANDO MUESTREO RAG UCM...")
    
    # --- Cargado de datos
    setup_enviroment(False, True)

    # --- Preguntas - cambiar el primero a None para ejecutarlo entero y lista no vacia para pruebas
    all_results = []


    for i in range(n):
        df = run_questions(None, None, API_KEY, PARTIAL_FILE)
        all_results.append(df)

    df_all = pd.concat(all_results, ignore_index=True)

    # Guardar resultados finales acumulados
    os.makedirs(os.path.dirname(FINAL_FILE), exist_ok=True)
    df_all.to_csv(FINAL_FILE, index=False)

    # --- Evaluación y dashboard
    evaluate_results(df_all, FINAL_FILE, PARTIAL_FILE)
    generate_dashboard()

def main():
    print("\n🧪 INICIANDO QUERY UCM...")

    # --- Cargado de datos - no se vuelve a crear la bd y borra resultados anteriores
    setup_enviroment(False, True)

    # --- Preguntas - cambiar el primero a None para ejecutarlo entero y lista no vacia para pruebas
    df = run_questions(None, None, API_KEY, PARTIAL_FILE)

    # --- Exportar Resultados y Resumen
    df.to_csv(FINAL_FILE, index=False)

    evaluate_results(df, FINAL_FILE) # np funciona
    generate_dashboard(dir_input= FINAL_FILE, dir_output="./results/plots")

if __name__ == "__main__":
    main()
    #multiple_runs(5)