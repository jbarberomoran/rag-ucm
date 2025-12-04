import os
import sys
from dotenv import load_dotenv
import pandas as pd
from src.evaluation import generate_dashboard, evaluate_results
from src.launcher import setup_enviroment
from src.queries import run_questions

#Ejecutar python main.py
#En caso de querer guardar los resultados de la ejecucion de antemano: pyhton main.py nombre_carpeta o bien "Nombre carpeta"

# Cargar clave API
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

def build_paths(base_dir):
    """Genera las rutas de salida según carpeta seleccionada."""
    final_file = os.path.join(base_dir, "resultados_finales.csv")
    return final_file, "./results/resultados_parciales.csv"

def multiple_runs(n = 10):
    print("\n🧪 INICIANDO MUESTREO RAG UCM...")
    
    #miramos si queremos resultados persistentes o no
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        results_dir = f"./results/persistent_results/{test_name}"
        clear_results = False
        print(f"📁 Modo PERSISTENTE: {results_dir}")
    else:
        results_dir = "./results/local_results"
        clear_results = True
        print(f"📁 Modo LOCAL: {results_dir}")

    # Crear carpeta si no existe
    os.makedirs(results_dir, exist_ok=True)

    FINAL_FILE, PARTIAL_FILE = build_paths(results_dir)

    # --- Cargado de datos
    setup_enviroment(False, clear_results, results_dir)

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
    evaluate_results(df_all, FINAL_FILE)
    generate_dashboard(dir_input= FINAL_FILE, dir_output=os.path.join(results_dir, "plots"))

def main():
    print("\n🧪 INICIANDO QUERY UCM...")

    #miramos si queremos resultados persistentes o no
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        results_dir = f"./results/persistent_results/{test_name}"
        clear_results = False
        print(f"📁 Modo PERSISTENTE: {results_dir}")
    else:
        results_dir = "./results/local_results"
        clear_results = True
        print(f"📁 Modo LOCAL: {results_dir}")

    # Crear carpeta si no existe
    os.makedirs(results_dir, exist_ok=True)

    FINAL_FILE, PARTIAL_FILE = build_paths(results_dir)

    # --- Cargado de datos - no se vuelve a crear la bd y borra resultados anteriores
    setup_enviroment(None, clear_results, results_dir)

    # --- Preguntas - cambiar el primero a None para ejecutarlo entero y lista no vacia para pruebas
    #df = run_questions(range(0,3), None, API_KEY, PARTIAL_FILE)
    df = run_questions(None, None, API_KEY, PARTIAL_FILE)

    # --- Exportar Resultados y Resumen
    df.to_csv(FINAL_FILE, index=False)

    evaluate_results(df, FINAL_FILE)
    generate_dashboard(dir_input= FINAL_FILE, dir_output=os.path.join(results_dir, "plots"))

if __name__ == "__main__":
    multiple_runs()

#Ejecutar python main.py
#En caso de querer guardar los resultados de la ejecucion de antemano: pyhton main.py nombre_carpeta o bien "Nombre carpeta"