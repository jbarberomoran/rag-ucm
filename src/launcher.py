import os
import shutil
from src.ingestion import db_setup
 
def setup_enviroment(rebuild_db=False, clear_results=True, results_dirt="./results"):
    """
    Configura el entorno de experimentos: BD vectorial y resultados.
    
    Args:
        rebuild_db (bool): Si True fuerza reconstrucción de la base vectorial.
        clear_results (bool): Si True borra CSVs y carpetas de resultados previos.
        results_dir (str): Carpeta donde se estás los resultados de otras queries.
    """
    print("\n🧪 PREPARANDO ENTORNO PARA RAG UCM...")

    # Creamos la carpeta de resultados si no existe
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Crear carpeta para gráficos si no existe
    if not os.path.exists(results_dirt + "/plots"):
        os.makedirs(results_dirt + "/plots")

    # --- 1. LIMPIEZA INICIAL ---
    # Borramos el final anterior
    if clear_results and os.path.exists(results_dirt+ "/resultados_finales.csv"):
        os.remove(results_dirt+"/resultados_finales.csv")
        
    # Borramos el parcial anterior para empezar el log de cero
    if clear_results and os.path.exists("./results/resultados_parciales.csv"):
        os.remove("./results/resultados_parciales.csv")

    #Borramos las tablas anteriores
    plots_dir = os.path.join(results_dirt, "plots")
    if clear_results and os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)

    # 2. Cargar Dataset de Preguntas
    # --- MANEJO BASE DE DATOS: Comprobar si existe una o no, y generarla si se pide o necesario
    db_setup(rebuild_db)