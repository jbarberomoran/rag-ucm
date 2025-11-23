import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración
RESULTS_PATH = "./results/resultados_finales.csv"
OUTPUT_DIR = "./results/plots"

def evaluate_results(df : pd.DataFrame, ff  = "./results/resultados_finales.csv", pf = "./results/resultados_parciales.csv"):
    df.to_csv(ff, index=False)

    print("\n" + "="*30)
    print("📊 RESUMEN DE PRECISIÓN (ACCURACY)")
    print("="*30)
    # Calcula el porcentaje de aciertos por método
    print(df.groupby("method")["correct"].mean() * 100)
    print(f"📁 Resultados parciales (log): {pf}")
    print(f"📁 Resultados finales (clean): {ff}")

def load_data():
    if not os.path.exists(RESULTS_PATH):
        print(f"❌ No se encuentra el archivo de resultados: {RESULTS_PATH}")
        return None
    return pd.read_csv(RESULTS_PATH)

def plot_accuracy(df):
    """Gráfico 1: Precisión General (Acierto vs Fallo)"""
    plt.figure(figsize=(10, 6))
    
    # Calculamos porcentaje de aciertos por método
    accuracy_df = df.groupby("method")["correct"].mean() * 100
    accuracy_df = accuracy_df.reset_index()
    
    sns.barplot(x="method", y="correct", data=accuracy_df, palette="viridis")
    plt.title("Precisión General (Accuracy) por Método")
    plt.ylabel("% de Acierto")
    plt.ylim(0, 100)
    
    # Guardar
    plt.savefig(OUTPUT_DIR + "/grafico_accuracy.png")
    print("📊 Gráfico de Precisión guardado.")
    plt.close()

def plot_rag_quality(df):
    """
    Gráfico 2: Calidad del RAG (Conocimiento vs Suerte)
    Este es el gráfico ESTRELLA para el concurso.
    """
    if "status" not in df.columns:
        print("⚠️ No se encontró la columna 'status'. Ejecuta main.py con el nuevo código primero.")
        return

    plt.figure(figsize=(12, 7))
    
    # Definimos una paleta de colores semántica
    # Verde = Perfecto
    # Amarillo = Suerte
    # Naranja = Fallo del Modelo (Pero el buscador funcionó)
    # Rojo = Fallo Total
    
    # Mapeo de colores basado en el texto que pusimos en main.py
    # Nota: Ajusta las claves si cambiaste el texto en main.py
    custom_palette = {
        "✅ ACIERTO PERFECTO (RAG)": "#2ecc71",           # Verde Esmeralda
        "⚠️ ACIERTO SUERTE (Sin Evidencia)": "#f1c40f",  # Amarillo
        "📉 FALLO RAZONAMIENTO (Contexto OK)": "#e67e22",# Naranja
        "❌ FALLO TOTAL": "#e74c3c"                       # Rojo
    }
    
    # Filtramos colores para que coincidan con lo que hay en el DF (para evitar errores)
    unique_statuses = df["status"].unique()
    palette_to_use = {k: v for k, v in custom_palette.items() if k in unique_statuses}
    
    # Si hay algún estado nuevo que no esté en mi lista, le damos gris
    for s in unique_statuses:
        if s not in palette_to_use:
            palette_to_use[s] = "#95a5a6"

    # Crear el gráfico (Countplot agrupado)
    ax = sns.countplot(
        data=df, 
        x="method", 
        hue="status", 
        palette=palette_to_use,
        edgecolor="black"
    )
    
    plt.title("Calidad Real del RAG: ¿Ingeniería o Suerte?", fontsize=14, fontweight='bold')
    plt.ylabel("Cantidad de Preguntas")
    plt.legend(title="Diagnóstico", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Guardar
    plt.savefig(OUTPUT_DIR + "/grafico_calidad_rag.png")
    print("📊 Gráfico de Calidad RAG guardado (EL IMPORTANTE).")
    plt.show() # Mostrar si usas Jupyter
    plt.close()

def plot_time_comparison(df):
    """Gráfico 3: Tiempo de respuesta"""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="method", y="response_time", data=df, palette="pastel")
    plt.title("Distribución de Latencia por Método")
    plt.ylabel("Segundos")
    
    plt.savefig(OUTPUT_DIR + "/grafico_tiempo.png")
    print("📊 Gráfico de Tiempos guardado.")
    plt.close()

def plot_latency_boxplot(df):
    """
    Gráfico 3: Distribución de Tiempos (BOXPLOT)
    Muestra medianas, cuartiles y outliers.
    """
    plt.figure(figsize=(10, 6))
    
    # Boxplot con Seaborn
    # Showfliers=True muestra los puntos atípicos (outliers)
    sns.boxplot(
        data=df, 
        x="method", 
        y="response_time", 
        hue="method",
        palette="Set2", 
        showfliers=True
    )
    
    # Opcional: Añadir puntos reales encima (Swarmplot) para ver la distribución real
    # sns.stripplot(data=df, x="method", y="response_time", color="black", alpha=0.3, jitter=True)

    plt.title("Comparativa de Latencia (Tiempo de Respuesta)", fontsize=14)
    plt.ylabel("Tiempo (Segundos)")
    plt.xlabel("Método de Recuperación")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(OUTPUT_DIR + "grafico_latencia_boxplot.png")
    print("📊 Gráfico de Latencia (Boxplot) guardado.")
    plt.close()

def generate_dashboard():
    print("\n📈 Generando Dashboard de Evaluación...")
    
    # Asegurar que existe la carpeta results
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = load_data()
    if df is not None:
        # Estilo visual general
        sns.set_theme(style="whitegrid")
        
        # 1. Precisión
        plot_accuracy(df)
        
        # 2. Calidad (Verdad Terreno)
        plot_rag_quality(df)
        
        # 3. Tiempos (NUEVO)
        plot_latency_boxplot(df)
        
        print(f"✅ ¡Todo listo! Revisa la carpeta {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dashboard()