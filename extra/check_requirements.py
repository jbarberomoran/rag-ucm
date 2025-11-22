import os
import sys
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Cargar variables de entorno
load_dotenv()

def test_sistema():
    print("--- INICIANDO DIAGNÓSTICO ---")
    
    # 1. Verificar versión de Python (Debe ser 3.11.x, NO 3.13)
    print(f"1. Versión de Python: {sys.version.split()[0]}")
    
    # 2. Verificar API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        print("2. API Key detectada: ✅")
    else:
        print("2. API Key detectada: ❌ (Revisa tu archivo .env)")
        return

    # 3. Verificar Datos
    try:
        ruta_json = os.path.join("data", "questions.json")
        with open(ruta_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"3. Archivo questions.json: ✅ (Contiene {len(data)} preguntas)")
    except Exception as e:
        print(f"3. Archivo questions.json: ⚠️ No encontrado o error de formato ({e})")

    # 4. Prueba de Fuego: Llamar a Gemini
    print("4. Conectando con Gemini... (Espere)")
    try:
        # En main.py
        llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite", # Cambia esto por el modelo que quieras probar
                google_api_key=api_key,
                temperature=0  #Para que las respuestas sean consistentes y reproducibles
        )
        respuesta = llm.invoke("Responde con una sola palabra: ¿Estás listo?")
        print(f"   🤖 Respuesta del Modelo: {respuesta.content}")
        print("\n✨ ¡SISTEMA OPERATIVO! LISTO PARA GANAR EL CONCURSO ✨")
    except Exception as e:
        print(f"   ❌ Error de conexión: {e}")

if __name__ == "__main__":
    test_sistema()