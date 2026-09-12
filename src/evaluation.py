import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def evaluate_results(df : pd.DataFrame, ff  : str):
    print("\n" + "="*30)
    print("📊 RESUMEN DE PRECISIÓN (ACCURACY)")
    print("="*30)
    # Calcula el porcentaje de aciertos por método
    print(df.groupby("method")["correct"].mean() * 100)
    print(f"📁 Resultados finales (clean): {ff}")

def load_data(dir_input : str):
    """Carga los datos y asegura que las columnas tengan el tipo correcto."""
    if not os.path.exists(dir_input):
        print(f"❌ Error: No se encuentra el archivo {dir_input}")
        print("   Ejecuta primero 'launcher.py' para generar datos.")
        return None
    
    df = pd.read_csv(dir_input)
    
    # Aseguramos que 'correct' sea numérico (1/0) para calcular porcentajes
    # En tu main lo guardas como booleano o int, esto lo estandariza
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(int)
        
    return df

def setup_plot_style():
    """Configura el estilo visual de las gráficas."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.autolayout': True})

def clean_emojis(text):
    """Elimina emojis para evitar warnings de fuentes en Windows."""
    if not isinstance(text, str):
        return text
    # Eliminamos caracteres no ASCII (emojis suelen serlo) o limpiamos chars específicos
    # Forma simple: Reemplazo directo de los que usas
    text = text.replace("✅", "").replace("⚠️", "").replace("📉", "").replace("❌", "")
    return text.strip()

def plot_accuracy(df, dir_output : str):
    """
    1. GRÁFICO DE PRECISIÓN (% de Aciertos)
    Usa la columna 'correct'.
    """
    if "correct" not in df.columns:
        print("⚠️ Columna 'correct' no encontrada. Saltando gráfico de precisión.")
        return

    plt.figure(figsize=(10, 6))
    
    # Agrupar por método y calcular la media de aciertos
    acc_df = df.groupby("method")["correct"].mean() * 100
    acc_df = acc_df.reset_index()
    
    # Crear gráfico de barras
    barplot = sns.barplot(
        x="method", 
        y="correct", 
        hue="method",
        data=acc_df, 
        palette="viridis",
        edgecolor="black",
        legend=False
    )
    
    # Añadir etiquetas de valor encima de las barras
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.1f}%', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 9), 
                         textcoords='offset points',
                         fontweight='bold')

    plt.title("Precisión de Respuesta (Accuracy) por Método", fontsize=14, fontweight='bold')
    plt.ylabel("% de Acierto")
    plt.xlabel("Método de Recuperación")
    plt.ylim(0, 115) # Margen superior para las etiquetas
    
    save_path = os.path.join(dir_output, "1_accuracy.png")
    plt.savefig(save_path, dpi=300)
    print("📊 Gráfico 1 guardado: Accuracy")
    plt.close()

def plot_rag_quality(df, dir_output : str):
    """
    Gráfico 2: Calidad del RAG en PORCENTAJE (%)
    """
    if "status" not in df.columns:
        return

    # 1. Limpiamos emojis de la columna status para evitar warnings de fuente
    df = df.copy()
    df["status_clean"] = df["status"].apply(clean_emojis)

    plt.figure(figsize=(12, 7))
    
    # 2. Calcular porcentajes
    # Agrupamos por método y status, contamos, y dividimos por el total de cada método
    counts = df.groupby(['method', 'status_clean']).size().reset_index(name='count')
    totals = df.groupby('method').size().reset_index(name='total')
    data_pct = pd.merge(counts, totals, on='method')
    data_pct['percentage'] = (data_pct['count'] / data_pct['total']) * 100

    # 3. Mapeo de colores (usando los nombres SIN emojis)
    status_palette = {
        "ACIERTO PERFECTO (RAG)": "#2ecc71",           # Verde
        "ACIERTO SUERTE (Sin Evidencia)": "#f1c40f",  # Amarillo
        "FALLO RAZONAMIENTO (Contexto OK)": "#e67e22",# Naranja
        "FALLO TOTAL": "#e74c3c"                       # Rojo
    }
    
    # Asegurar que la paleta cubra todo lo que hay en los datos
    unique = data_pct["status_clean"].unique()
    palette_final = {k: status_palette.get(k, "#95a5a6") for k in unique}

    # 4. Graficar con porcentajes
    barplot = sns.barplot(
        data=data_pct,
        x="method",
        y="percentage",
        hue="status_clean",
        palette=palette_final,
        edgecolor="black"
    )

    # Añadir etiquetas de % en las barras
    for p in barplot.patches:
        height = p.get_height()
        if height > 0: # Solo etiquetar si la barra existe
            barplot.annotate(f'{height:.1f}%',
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom',
                             fontsize=9, color='black', xytext=(0, 3),
                             textcoords='offset points')

    plt.title("Diagnóstico de Calidad RAG (Porcentajes Relativos)", fontsize=14, fontweight='bold')
    plt.ylabel("Porcentaje del Total (%)")
    plt.xlabel("Método")
    plt.legend(title="Diagnóstico", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.ylim(0, 110) # Margen para etiquetas

    plt.savefig(os.path.join(dir_output, "2_rag_quality_pct.png"), dpi=300, bbox_inches='tight')
    print("📊 Gráfico 2 guardado: RAG Quality (%)")
    plt.close()

def plot_latency(df, dir_output : str):
    """
    3. GRÁFICO DE LATENCIA (Boxplot)
    Usa la columna 'response_time'.
    """
    if "response_time" not in df.columns:
        print("⚠️ Columna 'response_time' no encontrada. Saltando gráfico de latencia.")
        return

    plt.figure(figsize=(10, 6))
    
    sns.boxplot(
        data=df, 
        x="method", 
        y="response_time", 
        hue="method",
        palette="pastel",
        showfliers=True, # Mostrar outliers
        legend=False
    )
    
    plt.title("Latencia del Sistema (Tiempo de Respuesta)", fontsize=14)
    plt.ylabel("Segundos")
    plt.xlabel("Método")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    save_path = os.path.join(dir_output, "3_latency.png")
    plt.savefig(save_path, dpi=300)
    print("📊 Gráfico 3 guardado: Latency")
    plt.close()

def plot_retrieval_score(df, dir_output : str):
    """
    4. GRÁFICO DE FIDELIDAD DE RECUPERACIÓN (Violin Plot)
    Usa la columna 'retrieval_score'.
    """
    if "retrieval_score" not in df.columns:
        print("⚠️ Columna 'retrieval_score' no encontrada. Saltando gráfico de fidelidad.")
        return

    plt.figure(figsize=(10, 6))
    
    # Usamos Violinplot porque muestra la densidad de distribución mejor que el boxplot
    sns.violinplot(
        data=df,
        x="method",
        y="retrieval_score",
        hue="method",
        palette="Set3",
        inner="quartile", # Muestra líneas de cuartiles dentro del violín
        legend=False
    )
    
    plt.title("Fidelidad de Recuperación (Similitud con Ground Truth)", fontsize=14)
    plt.ylabel("Puntuación de Similitud (0-1)")
    plt.xlabel("Método")
    plt.ylim(-0.1, 1.1) # Márgenes para ver bien los extremos
    
    save_path = os.path.join(dir_output, "4_retrieval_fidelity.png")
    plt.savefig(save_path, dpi=300)
    print("📊 Gráfico 4 guardado: Retrieval Fidelity")
    plt.close()

def generate_dashboard(dir_input, dir_output : str):
    print(f"\n📈 Iniciando generación de gráficos desde: {dir_input}")
    
    # 1. Crear carpeta de plots si no existe
    if not os.path.exists(dir_output):
        print(f"📁 Creando directorio de salida: {dir_output}")
        os.makedirs(dir_output)
        
    # 2. Cargar datos
    df = load_data(dir_input)
    
    if df is not None:
        setup_plot_style()
        
        # 3. Generar gráficas
        try:
            plot_accuracy(df, dir_output)
            plot_rag_quality(df, dir_output)
            plot_latency(df, dir_output)
            plot_retrieval_score(df, dir_output)
            print(f"\n✅ ¡Éxito! Gráficos generados en: {os.path.abspath(dir_output)}")
        except Exception as e:
            print(f"❌ Error generando gráficos: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    generate_dashboard(
        "results/persistent_results/resultados_definitivos/resultados_finales.csv",
        "results/persistent_results/resultados_definitivos/plots",
    )
