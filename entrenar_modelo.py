"""
Nombre: Edcarlin Angeuris Celesten Benitez
Matrícula: 24-EISN-2-017
Entrenamiento de modelo para reconocimiento de recetas médicas
Versión con dataset.json simplificado
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN
# ============================================

# Diccionario de medicamentos (clases)
CLASES_MEDICAMENTOS = [
    'metoprolol', 'enalapril', 'losartan', 'amlodipino',
    'amoxicilina', 'azitromicina', 'ciprofloxacino',
    'paracetamol', 'ibuprofeno', 'diclofenaco',
    'omeprazol', 'cimetidina', 'salbutamol',
    'prednisona', 'metformina', 'dorzolamida', 
    'oxprenolol', 'ceftriaxona', 'dexametasona', 'otro'

]

NUM_CLASES = len(CLASES_MEDICAMENTOS)

# ============================================
# DATASET CON DATASET.JSON
# ============================================

class RecetaDataset(Dataset):
    """Dataset que lee desde un solo archivo dataset.json"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        
        # Cargar dataset.json
        dataset_path = os.path.join(data_dir, 'dataset.json')
        
        if not os.path.exists(dataset_path):
            print(f"❌ No se encontró {dataset_path}")
            print("📌 Crea el archivo dataset.json con el formato:")
            print('''
            {
              "recetas": [
                {
                  "imagen": "receta1.jpg",
                  "medicamentos": ["metoprolol", "paracetamol"]
                }
              ]
            }
            ''')
            return
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        recetas = dataset.get('recetas', [])
        
        for receta in recetas:
            imagen_nombre = receta.get('imagen')
            medicamentos = receta.get('medicamentos', [])
            
            if not imagen_nombre or not medicamentos:
                print(f"⚠️ Saltando receta sin imagen o medicamentos: {receta}")
                continue
            
            img_path = os.path.join(data_dir, 'imagenes', imagen_nombre)
            
            if os.path.exists(img_path):
                self.data.append({
                    'path': img_path,
                    'medicamentos': medicamentos
                })
            else:
                print(f"⚠️ Imagen no encontrada: {img_path}")
        
        print(f"📊 Cargadas {len(self.data)} imágenes de entrenamiento")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Cargar imagen
        image = Image.open(item['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convertir etiquetas a tensor
        label_tensor = torch.zeros(NUM_CLASES)
        
        for med in item['medicamentos']:
            med_lower = med.lower().strip()
            if med_lower in CLASES_MEDICAMENTOS:
                label_tensor[CLASES_MEDICAMENTOS.index(med_lower)] = 1
            else:
                # Si no está en la lista, usar "otro"
                label_tensor[CLASES_MEDICAMENTOS.index('otro')] = 1
        
        return image, label_tensor

# ============================================
# MODELO
# ============================================

class ModeloRecetas(nn.Module):
    def __init__(self, num_clases=NUM_CLASES):
        super(ModeloRecetas, self).__init__()
        
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_clases),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)

# ============================================
# ENTRENAMIENTO
# ============================================

def entrenar_modelo():
    print("\n" + "="*50)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Dispositivo: {device}")
    
    # Transformaciones
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Cargar dataset
    print("\n📂 Cargando datos de entrenamiento...")
    dataset = RecetaDataset('datos_entrenamiento', transform=transform_train)
    
    if len(dataset) == 0:
        print("\n❌ No hay datos de entrenamiento.")
        print("\n📌 DEBES CREAR EL ARCHIVO: datos_entrenamiento/dataset.json")
        print('''
        Formato:
        {
          "recetas": [
            {
              "imagen": "receta1.jpg",
              "medicamentos": ["metoprolol", "paracetamol"]
            }
          ]
        }
        ''')
        return None
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Crear modelo
    model = ModeloRecetas().to(device)
    
    # Configurar entrenamiento
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entrenar
    num_epochs = 30
    print(f"\n📚 Entrenando por {num_epochs} épocas...")
    print("-"*50)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Época [{epoch+1}/{num_epochs}] - Pérdida: {epoch_loss:.4f}")
    
    # Guardar modelo
    os.makedirs('modelos_entrenados', exist_ok=True)
    torch.save(model.state_dict(), 'modelos_entrenados/modelo_recetas.pth')
    
    # Guardar clases
    with open('modelos_entrenados/clases.json', 'w', encoding='utf-8') as f:
        json.dump(CLASES_MEDICAMENTOS, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*50)
    print(f"✅ MODELO GUARDADO EN: modelos_entrenados/modelo_recetas.pth")
    print(f"✅ CLASES GUARDADAS EN: modelos_entrenados/clases.json")
    print("="*50)
    
    return model

# ============================================
# MENÚ PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🎯 SISTEMA DE ENTRENAMIENTO - RECETAS MÉDICAS        ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print(f"CLASES DE MEDICAMENTOS: {len(CLASES_MEDICAMENTOS)}")
    print("="*50)
    print("OPCIONES:")
    print("   1. Entrenar modelo")
    print("   2. Salir")
    print("="*50)
    
    opcion = input("\n👉 Selecciona una opción : ")
    
    if opcion == '1':
       entrenar_modelo()
    else:
        print("👋 Hasta luego!")