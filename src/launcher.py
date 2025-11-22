import os
from src.ingestion import db_setup

    
def setup_enviroment(rebuild_db=False, clear_results=True, results_dir="./results"):
    """
    Configura el entorno de experimentos: BD vectorial y resultados.
    
    Args:
        rebuild_db (bool): Si True fuerza reconstrucción de la base vectorial.
        clear_results (bool): Si True borra CSVs y carpetas de resultados previos.
        results_dir (str): Carpeta donde se estás los resultados de otras queries.
    """
    print("🧪 INICIANDO EXPERIMENTO RAG UCM...")
    # --- 1. LIMPIEZA INICIAL ---
    # Borramos el final anterior
    if clear_results and os.path.exists(results_dir+ "/resultados_finales.csv"):
        os.remove(results_dir+"/resultados_finales.csv")
        
    # Borramos el parcial anterior para empezar el log de cero
    if clear_results and os.path.exists(results_dir + "/resultados_parciales.csv"):
        os.remove(results_dir + "/resultados_parciales.csv")

    # 2. Cargar Dataset de Preguntas
    # --- MANEJO BASE DE DATOS: Comprobar si existe una o no, y generarla si se pide o necesario
    db_setup(rebuild_db)

    # Creamos la carpeta de resultados si no existe
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    # Crear carpeta para gráficos si no existe
    if not os.path.exists(results_dir + "/plots"):
        os.makedirs(results_dir + "/plots")