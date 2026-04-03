"""
Nombre: Edcarlin Angeuris Celeten Benitez
Matrícula: 24-EISN-2-017
Cargador de modelo entrenado para recetas médicas
"""

import torch
import torch.nn as nn
from torchvision import models
import json
import os

class CargadorModeloRecetas:
    def __init__(self, model_path='modelos_entrenados/modelo_recetas.pth', 
                 clases_path='modelos_entrenados/clases.json'):
        
        self.model_path = model_path
        self.clases_path = clases_path
        self.modelo = None
        self.clases = []
        self.transform = None
        self._cargar()
    
    def _cargar(self):
        """Carga el modelo y las clases"""
        try:
            # Cargar clases
            if os.path.exists(self.clases_path):
                with open(self.clases_path, 'r', encoding='utf-8') as f:
                    self.clases = json.load(f)
                print(f"📋 Clases cargadas: {len(self.clases)}")
            else:
                print(f"⚠️ No se encontró archivo de clases: {self.clases_path}")
                return False
            
            # Crear modelo
            backbone = models.resnet18(pretrained=False)
            num_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(self.clases)),
                nn.Sigmoid()
            )
            
            # Cargar pesos
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location='cpu')
                
                # Remover el prefijo 'backbone.' si existe
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('backbone.'):
                        new_key = key[9:]  # Remover 'backbone.'
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                
                # Cargar los pesos
                backbone.load_state_dict(new_state_dict, strict=False)
                print(f"✅ Modelo cargado desde: {self.model_path}")
                
                self.modelo = backbone
                self.modelo.eval()
                
                # Transformaciones
                from torchvision import transforms
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                return True
            else:
                print(f"⚠️ No se encontró modelo en: {self.model_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def predecir(self, image):
        """Predice medicamentos en una imagen"""
        if self.modelo is None:
            return []
        
        try:
            from PIL import Image
            import numpy as np
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Transformar
            img_tensor = self.transform(image).unsqueeze(0)
            
            # Predecir
            with torch.no_grad():
                outputs = self.modelo(img_tensor)
                predicciones = outputs.numpy()[0]
            
            # Obtener medicamentos con confianza > 0.5
            medicamentos = []
            for i, confianza in enumerate(predicciones):
                if confianza > 0.5 and i < len(self.clases):
                    medicamentos.append({
                        'nombre': self.clases[i].capitalize(),
                        'confianza': float(confianza)
                    })
            
            return medicamentos
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return []
    
    def esta_cargado(self):
        return self.modelo is not None


# Prueba rápida si se ejecuta directamente
if __name__ == "__main__":
    print("=== Probando cargador de modelo ===\n")
    cargador = CargadorModeloRecetas()
    
    if cargador.esta_cargado():
        print("\n✅ Modelo listo para usar!")
        print(f"📋 Clases: {cargador.clases[:5]}...")
    else:
        print("\n❌ No se pudo cargar el modelo")