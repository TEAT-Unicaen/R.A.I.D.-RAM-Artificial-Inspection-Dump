import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

import transformer.visionTransformer as vt

def predict_scratch(image_path, model_path="shrekTransformerResult.pth"):
    # 1. Vérification du fichier
    if not os.path.exists(image_path):
        print(f"Erreur : Impossible de trouver l'image '{image_path}'")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test du modèle 'From Scratch' sur : {device}")

    # 2. Architecture (Identique à l'entraînement)
    # On crée un ViT vide
    model = vt.VisionTransformer()

    # 3. Chargement des poids
    if not os.path.exists(model_path):
        print(f"Erreur : Le modèle '{model_path}' est introuvable. L'entraînement est-il fini ?")
        return
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except Exception as e:
        print(f"Erreur lors du chargement des poids : {e}")
        return
 
    model.eval()
    model.to(device)

    # 4. Prétraitement (DOIT MATCHER L'ENTRAÎNEMENT)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # Gestion des images PNG (transparence) ou N&B
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 5. Prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        score, predicted = torch.max(probs, 1)

    # Ordre alphabétique des dossiers
    classes = ['Pas Shrek', 'Shrek']
    
    print(f"\n--- RÉSULTAT ---")
    print(f"Image     : {image_path}")
    print(f"Verdict   : {classes[predicted.item()]}")
    print(f"Confiance : {score.item()*100:.2f}%")

if __name__ == "__main__":
    predict_scratch("dataset/val/notShrek/cetelem.png")
    predict_scratch("dataset/val/notShrek/cheval.jpg")
    predict_scratch("dataset/val/notShrek/disney.jpg")
    print('---')
    predict_scratch("dataset/val/shrek/10472692.jpg")
    predict_scratch("dataset/val/shrek/images.jpg")