"""
Nombre: Edcarlin Angeuris Celeten Benitez
Matrícula: 24-EISN-2-017
Configuración de la aplicación
"""

class Config:
    # Configuración del modelo - CAMBIAR A False para NO usar Ollama
    USE_OLLAMA = False 
    OLLAMA_MODEL = 'llama3.2-vision' 
    
    # Configuración de TTS
    TTS_LANGUAGE = 'es'
    TTS_SLOW = False
    
    # Configuración de la app
    APP_TITLE = "Lector Inteligente de Recetas Médicas"
    APP_PORT = 7861
    APP_SHARE = True
    
    # Mensajes
    WARNING_MESSAGE = """
⚠️ **ADVERTENCIA IMPORTANTE**
Esta es una interpretación automática realizada por inteligencia artificial.
Siempre consulta con un profesional de la salud para confirmar la información.
"""