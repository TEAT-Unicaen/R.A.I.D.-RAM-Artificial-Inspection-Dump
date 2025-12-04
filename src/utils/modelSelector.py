"""
Model selector utility for choosing trained models from output directory.
"""

import os
from datetime import datetime

def list_trained_models(output_dir="output"):
    """
    List all trained models in the output directory.
    
    Args:
        output_dir (str): Path to the output directory
        
    Returns:
        list: List of tuples (index, folder_name, timestamp, model_path, config_path)
    """
    if not os.path.exists(output_dir):
        print(f"Erreur : Le dossier '{output_dir}' n'existe pas.")
        return []
    
    models = []
    folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    folders.sort(reverse=True)  # Plus récent en premier
    
    for idx, folder in enumerate(folders, 1):
        folder_path = os.path.join(output_dir, folder)
        model_path = os.path.join(folder_path, "shrekTransformerResult.pth")
        config_path = os.path.join(folder_path, "config.cfg")
        
        if os.path.exists(model_path):
            # Parse timestamp from folder name (format: YYYY-MM-DD_HH-MM-SS)
            try:
                timestamp = datetime.strptime(folder, "%Y-%m-%d_%H-%M-%S")
                timestamp_str = timestamp.strftime("%d/%m/%Y à %H:%M:%S")
            except:
                timestamp_str = folder
            
            models.append((idx, folder, timestamp_str, model_path, config_path))
    
    return models

def display_models(models):
    """
    Display available models in a formatted way.
    
    Args:
        models (list): List of model tuples from list_trained_models
    """
    if not models:
        print("\nAucun modèle trouvé dans le dossier 'output'.\n")
        return
    
    print("\n" + "="*60)
    print("MODÈLES DISPONIBLES")
    print("="*60)
    
    for idx, folder, timestamp, model_path, config_path in models:
        has_config = "✓" if os.path.exists(config_path) else "✗"
        print(f"{idx}. {folder}")
        print(f"   Date: {timestamp}")
        print(f"   Config: {has_config}")
        print()
    
    print("="*60 + "\n")

def select_model_interactive(models):
    """
    Let user select a model interactively.
    
    Args:
        models (list): List of model tuples from list_trained_models
        
    Returns:
        tuple: (model_path, config_path) or (None, None) if cancelled
    """
    if not models:
        return None, None
    
    display_models(models)
    
    while True:
        try:
            choice = input(f"Choisissez un modèle (1-{len(models)}) [défaut: 1] ou 'q' pour quitter : ").strip()
            
            if choice.lower() == 'q':
                print("Sélection annulée.\n")
                return None, None
            
            # Si entrée vide, sélectionner le modèle 1 (le plus récent)
            if choice == '':
                choice_idx = 1
            else:
                choice_idx = int(choice)
            
            if 1 <= choice_idx <= len(models):
                selected = models[choice_idx - 1]
                print(f"\nModèle sélectionné : {selected[1]}\n")
                return selected[3], selected[4]  # model_path, config_path
            else:
                print(f"Veuillez entrer un nombre entre 1 et {len(models)}.\n")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.\n")
        except KeyboardInterrupt:
            print("\nSélection annulée.\n")
            return None, None

def get_latest_model(output_dir="output"):
    """
    Get the most recent trained model.
    
    Args:
        output_dir (str): Path to the output directory
        
    Returns:
        tuple: (model_path, config_path) or (None, None) if no model found
    """
    models = list_trained_models(output_dir)
    if models:
        return models[0][3], models[0][4]  # First is most recent
    return None, None
