"""
Nombre: Edcarlin Angeuris Celeten Benitez
Matrícula: 24-EISN-2-017
Utilidades: TTS y formateo de texto
"""

import tempfile
from gtts import gTTS
import re
class TextToSpeech:
    def __init__(self, language='es'):
        self.language = language
    
    def _limpiar_texto(self, text):
        """Limpia caracteres especiales para que el TTS suene natural"""
        
        # Eliminar asteriscos, numerales, guiones, etc.
        text = re.sub(r'[\*#_\-=+]', ' ', text)
        
        # Eliminar múltiples espacios
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar emojis 
        text = re.sub(r'[^\w\sáéíóúñÁÉÍÓÚÑ,.;:¡!¿?()]', '', text)
        
        # Reemplazar abreviaturas comunes
        reemplazos = {
            'mg': 'miligramos',
            'ml': 'mililitros',
            'g': 'gramos',
            'cada': 'cada',
            'h': 'horas',
            'kg': 'kilos',
            'cm': 'centímetros',
            'Dr.': 'doctor',
            'Dra.': 'doctora',
        }
        
        for abrev, completo in reemplazos.items():
            text = text.replace(abrev, completo)
        
        return text.strip()
    
    def convert(self, text, slow=False):
        try:
            # Limpiar texto antes de convertir
            texto_limpio = self._limpiar_texto(text)
            
            if len(texto_limpio) > 1000:
                texto_limpio = texto_limpio[:1000] + "..."
            
            if not texto_limpio:
                return None
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                audio_path = tmp_file.name
            
            tts = gTTS(text=texto_limpio, lang=self.language, slow=slow)
            tts.save(audio_path)
            
            return audio_path
        except Exception as e:
            print(f"Error generando audio: {e}")
            return None
def format_output(text):
    header = """
🏥 **LECTOR DE RECETAS MÉDICAS**
═══════════════════════════════════

"""
    
    footer = """

═══════════════════════════════════
⚠️ **NOTA IMPORTANTE**
Esta información es generada automáticamente.
"""
    
    return header + text + footer