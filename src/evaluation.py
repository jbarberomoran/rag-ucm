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

def generate_dashboard():
    print("📊 GENERANDO DASHBOARD DE EVALUACIÓN...")

    # 1. Cargar Datos
    if not os.path.exists(RESULTS_PATH):    #creo que ahora esto no puede saltar porque siempre se ha ejecutado en el launcher
        print(f"❌ Error: No encuentro el archivo '{RESULTS_PATH}'. Ejecuta main.py primero.")
        return

    df = pd.read_csv(RESULTS_PATH)

    # Configurar estilo visual (Estilo 'Clean Code')
    sns.set_theme(style="whitegrid")

    # --- GRÁFICO 1: PRECISIÓN (ACCURACY) ---
    # Este es el gráfico MÁS IMPORTANTE para el concurso
    plt.figure(figsize=(10, 6))
    
    # Calculamos porcentaje de aciertos
    accuracy_df = df.groupby("method")["correct"].mean() * 100
    accuracy_df = accuracy_df.reset_index()
    
    # Dibujamos
    ax = sns.barplot(x="method", y="correct", data=accuracy_df, palette="viridis")
    
    # Añadir etiquetas con el % exacto encima de cada barra
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f%%', padding=3)

    plt.title("Comparativa de Precisión (Accuracy) por Modelo", fontsize=14, fontweight='bold')
    plt.ylabel("Precisión (%)")
    plt.xlabel("Método RAG")
    plt.ylim(0, 100) # Escala de 0 a 100 siempre
    
    # Guardar
    save_path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de Precisión guardado en: {save_path}")
    plt.close()

    # --- GRÁFICO 2: TIEMPO DE RESPUESTA (LATENCIA) ---
    plt.figure(figsize=(10, 6))
    
    # Dibujamos tiempos
    sns.boxplot(x="method", y="response_time", data=df, palette="pastel")
    
    plt.title("Distribución de Tiempos de Respuesta", fontsize=14, fontweight='bold')
    plt.ylabel("Segundos (s)")
    plt.xlabel("Método RAG")
    
    # Guardar
    save_path = os.path.join(OUTPUT_DIR, "latency_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de Latencia guardado en: {save_path}")
    plt.close()

    # --- TABLA RESUMEN (Para copiar al informe) ---
    print("\n📋 TABLA RESUMEN PARA EL INFORME:")
    summary = df.groupby("method").agg(
        Accuracy=('correct', lambda x: f"{x.mean()*100:.1f}%"),
        Avg_Time=('response_time', lambda x: f"{x.mean():.2f}s"),
        Total_Questions=('question_id', 'count')
    )
    print(summary)
    
    # Guardar tabla en CSV
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_table.csv"))

if __name__ == "__main__":
    generate_dashboard()