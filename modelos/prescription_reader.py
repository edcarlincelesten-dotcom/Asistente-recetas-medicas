"""
Nombre: Edcarlin Angeuris Celeten Benitez
Matrícula: 24-EISN-2-017
Modelo de Deep Learning para lectura de recetas médicas
"""

import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Importar el cargador de modelo
from modelos.cargar_modelo import CargadorModeloRecetas

class MedicalPrescriptionModel:
    def __init__(self, use_ollama=False, model_name=''):
        self.use_ollama = False
        self.is_ready = False
        self.modelo_ml = None
        self._init_easyocr()
        self._cargar_modelo_ml()
    
    def _init_easyocr(self):
        """Inicializa EasyOCR para OCR básico"""
        try:
            import easyocr
            print("🔄 Inicializando EasyOCR...")
            self.reader = easyocr.Reader(['es', 'en'], gpu=False)
            self.is_ready = True
            print("✅ EasyOCR listo")
        except ImportError:
            print("❌ EasyOCR no instalado. Ejecuta: pip install easyocr")
            self.is_ready = False
        except Exception as e:
            print(f"❌ Error: {e}")
            self.is_ready = False
    
    def _cargar_modelo_ml(self):
        """Carga el modelo de ML entrenado"""
        try:
            print("🔍 Cargando modelo de ML...")
            self.modelo_ml = CargadorModeloRecetas()
            
            if self.modelo_ml.esta_cargado():
                print("✅ Modelo ML listo")
            else:
                print("⚠️ Modelo ML no disponible, usando solo OCR")
                
        except Exception as e:
            print(f"⚠️ Error cargando modelo ML: {e}")
            self.modelo_ml = None
    
    def extract_text(self, image):
        """Extrae texto y detecta medicamentos"""
        if not self.is_ready:
            return "⚠️ EasyOCR no disponible"
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        try:
            # 1. EasyOCR para leer texto
            results = self.reader.readtext(image)
            
            if not results:
                return self._no_text_found()
            
            texto_completo = ' '.join([text for (_, text, _) in results])
            
            # 2. Modelo ML para detectar medicamentos
            medicamentos = []
            if self.modelo_ml and self.modelo_ml.esta_cargado():
                print("🔍 Ejecutando modelo ML...")
                medicamentos = self.modelo_ml.predecir(image)
                print(f"💊 Medicamentos detectados: {medicamentos}")
            
            # 3. Formatear salida
            return self._formato_interactivo(medicamentos, texto_completo)
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _formato_interactivo(self, medicamentos, texto_completo):
        """Formatea la salida sin opciones"""
        
        if medicamentos:
            salida = """
## 🏥 **¡Encontramos esto en tu receta!**

"""
            for i, med in enumerate(medicamentos, 1):
                confianza_texto = f" (confianza: {med['confianza']:.0%})"
                salida += f"""
**{i}. {med['nombre']}{confianza_texto}**

"""
            salida += """
---
**💡 Consejos:**
- ✅ Verifica que coincida con tu receta
- 📞 Confirma dosis con tu médico
- 💊 Nunca te automediques
"""
        else:
            salida = f"""
## 📋 **Lectura de receta médica**

**Texto detectado:**
{texto_completo[:400]}

**No se reconocieron medicamentos automáticamente**

### ¿Qué puedes hacer?
1. 📸 Asegúrate que la receta esté bien iluminada
2. 🔍 La letra debe ser clara y legible
3. 📄 Intenta tomar una foto más cerca
"""
        
        return salida
    
    def _get_opciones_medicamento(self, nombre):
        """Opciones de presentación para cada medicamento"""
        opciones = {
            'metoprolol': ['Metoprolol 50 mg', 'Metoprolol 100 mg', 'Metoprolol 200 mg'],
            'dorzolamida': ['Dorzolamida 2% Colirio', 'Dorzolamida 2% + Timolol Colirio'],
            'cimetidina': ['Cimetidina 200 mg', 'Cimetidina 400 mg', 'Cimetidina 300 mg/2 mL'],
            'oxprenolol': ['Oxprenolol 50 mg', 'Oxprenolol 100 mg', 'Oxprenolol 80 mg'],
            'amoxicilina': ['Amoxicilina 500 mg', 'Amoxicilina 250 mg/5 mL', 'Amoxicilina 875 mg'],
            'ibuprofeno': ['Ibuprofeno 400 mg', 'Ibuprofeno 600 mg', 'Ibuprofeno 200 mg/5 mL'],
            'paracetamol': ['Paracetamol 500 mg', 'Paracetamol 1 g', 'Paracetamol 750 mg'],
            'omeprazol': ['Omeprazol 20 mg', 'Omeprazol 40 mg', 'Omeprazol 10 mg'],
        }
        return opciones.get(nombre, [])
    
    def _no_text_found(self):
        return """
## ❌ **No se detectó texto en la imagen**

### Consejos para mejorar:
- 📸 Toma la foto con buena iluminación
- 🔍 Asegúrate que la receta esté enfocada
- 📄 La receta debe estar plana, sin dobleces
"""
    
    def get_model_info(self):
        info = "EasyOCR"
        if self.modelo_ml and self.modelo_ml.esta_cargado():
            info += f" + Modelo ML ({len(self.modelo_ml.clases)} clases)"
        return info