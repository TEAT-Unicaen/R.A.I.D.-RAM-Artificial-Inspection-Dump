import configparser
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import transformer.visionTransformer as vt
from utils.config import TrainingConfig
from utils.modelSelector import list_trained_models, select_model_interactive, get_latest_model

def predict_scratch(image_path, model_path="shrekTransformerResult.pth", config=None):
    # 1. Vérification du fichier
    if not os.path.exists(image_path):
        print(f"Erreur : Impossible de trouver l'image '{image_path}'")
        return

    # Lecture des décimaux (Float)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.use_cpu else "cpu")
    if config.debug: print(f"Test du modèle 'From Scratch' sur : {device}")

    # 2. Architecture (Identique à l'entraînement)
    # On crée un ViT vide
    model = vt.VisionTransformer(embedDim=config.embed_dim, dropout=config.dropout, depth=config.depth, heads=config.heads)

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
    
    if config.debug:
        print(f"\n--- RÉSULTAT ---")
        print(f"Image     : {image_path}")
        print(f"Verdict   : {classes[predicted.item()]}")
        print(f"Confiance : {score.item()*100:.2f}%")

    return predicted.item(), score.item()*100

if __name__ == "__main__":
    
    # Sélection du modèle
    print("Recherche des modèles disponibles...\n")
    models = list_trained_models("output")
    
    if not models:
        print("Aucun modèle trouvé. Veuillez d'abord entraîner un modèle.")
        exit(1)
    
    # Sélection interactive
    model_path, config_path = select_model_interactive(models)
    
    if model_path is None:
        print("Aucun modèle sélectionné. Arrêt.")
        exit(0)
    
    # Charger la configuration du modèle sélectionné
    if config_path and os.path.exists(config_path):
        config = TrainingConfig(config_path)
        print(f"Configuration chargée : {config}\n")
    else:
        print("Pas de config trouvée, utilisation de la config par défaut.\n")
        config = TrainingConfig('config.cfg')
    
    base_dir = "dataset/val"
    classes = ["notShrek", "shrek"]

    total = 0
    correct = 0

    confidences_all = []
    confidences_correct = []

    for cls_idx, cls_name in enumerate(classes):
        folder = os.path.join(base_dir, cls_name)
        if not os.path.isdir(folder):
            print(f"Dossier introuvable : {folder}")
            continue

        print(f"\nTest du dossier : {cls_name}\n")

        for file in os.listdir(folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                img_path = os.path.join(folder, file)

                pred, score = predict_scratch(img_path, model_path=model_path, config=config)
                if pred is None:
                    continue

                score_percent = score

                total += 1
                confidences_all.append(score_percent)

                if pred == cls_idx:
                    correct += 1
                    confidences_correct.append(score_percent)
                    print(f"Correct pour {file} (confiance {score_percent:.2f}%)")
                else:
                    print(f"\033[91mFaux pour {file} (prédit {classes[pred]} avec {score_percent:.2f}%)\033[0m")

    if total > 0:
        accuracy = (correct / total) * 100
        avg_conf_global = sum(confidences_all) / len(confidences_all)
        avg_conf_correct = sum(confidences_correct) / len(confidences_correct) if confidences_correct else 0

        print("\n==============================")
        print("     RÉSUMÉ DU MODÈLE")
        print("==============================")
        print(f"Accuracy totale           : {accuracy:.2f}% ({correct}/{total})")
        print(f"Confiance moyenne globale : {avg_conf_global:.2f}%")
        print(f"Confiance moyenne correcte: {avg_conf_correct:.2f}%")
        print("==============================\n")
    else:
        print("\nAucune image trouvée.")
