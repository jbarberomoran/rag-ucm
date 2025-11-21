import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Cargar entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: No se encontró la API Key en .env")
else:
    # 2. Configurar la librería
    genai.configure(api_key=api_key)

    print("🔍 Buscando modelos disponibles para tu API Key...\n")
    print(f"{'NOMBRE DEL MODELO':<30} | {'DESCRIPCIÓN'}")
    print("-" * 60)

    try:
        # 3. Listar modelos
        for m in genai.list_models():
            # Filtramos solo los que sirven para generar contenido (texto/chat)
            if 'generateContent' in m.supported_generation_methods:
                print(f"{m.name:<30} | {m.description}")
        
        print("\n✅ Copia uno de estos nombres (ej: models/gemini-pro) en tu main.py")
        
    except Exception as e:
        print(f"❌ Error al conectar: {e}")