"""
Nombre: Edcarlin Angeuris Celeten Benitez
Matrícula: 24-EISN-2-017
Aplicación principal con interfaz Gradio
"""

import sys
import os

# Agregar la ruta padre para poder importar modelos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from modelos.prescription_reader import MedicalPrescriptionModel
from app.utils import TextToSpeech, format_output
from app.config import Config

class PrescriptionApp:
    def __init__(self):
        print("🚀 Inicializando aplicación...")
        
        self.model = MedicalPrescriptionModel(
            use_ollama=Config.USE_OLLAMA,
            model_name=Config.OLLAMA_MODEL
        )
        
        self.tts = TextToSpeech(language=Config.TTS_LANGUAGE)
    
    def process_prescription(self, image):
        if image is None:
            return "❌ No se ha cargado ninguna imagen.", None
        
        try:
            extracted_text = self.model.extract_text(image)
            formatted_text = format_output(extracted_text)
            formatted_text += Config.WARNING_MESSAGE
            
            audio_file = self.tts.convert(extracted_text, slow=Config.TTS_SLOW)
            
            return formatted_text, audio_file
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def create_interface(self):
        with gr.Blocks(title=Config.APP_TITLE, theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🏥 Lector Inteligente de Recetas Médicas
            ### Con reconocimiento de texto y conversión a voz
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📷 Sube tu receta médica",
                        type="numpy",
                        height=400
                    )
                    process_btn = gr.Button("🔍 Procesar Receta", variant="primary")
                
                with gr.Column(scale=1):
                    text_output = gr.Textbox(
                        label="📄 Información Extraída",
                        lines=20,
                        interactive=False
                    )
                    audio_output = gr.Audio(
                        label="🔊 Escucha la receta",
                        type="filepath"
                    )
            
            process_btn.click(
                fn=self.process_prescription,
                inputs=[image_input],
                outputs=[text_output, audio_output]
            )
        
        return interface

if __name__ == "__main__":
    app = PrescriptionApp()
    interface = app.create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=Config.APP_PORT,
        share=Config.APP_SHARE
    )