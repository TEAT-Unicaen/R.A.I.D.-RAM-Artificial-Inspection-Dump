# R.A.I.D — Guide d'exécution

Petit guide rapide pour exécuter le projet (générer les données, entraîner et évaluer). Il se concentre uniquement sur les commandes et les chemins utiles.

**Prérequis**
- Python 3.10+
- pytorch
- tqdm
- numpy
- cryptography
- GPU Nvidia recommandé pour l'entraînement et l'inférence (fortement conseillé).

**Fichiers clés**
- Configuration générale : [src/config.py](src/config.py)
- Lancer l'entraînement / l'évaluation : [src/main.py](src/main.py)
- Script d'entraînement (options avancées) : [src/train.py](src/train.py)
- Checkpoints : [checkpoints](checkpoints)
- Modèle exporté : [bytes_transformer_classifier_all.pth](bytes_transformer_classifier_all.pth)

Préparez un environnement virtuel et installez les dépendances requises (`torch`, `numpy`, etc.) selon votre configuration.

Commandes principales
- Générer le dataset :

```bash
python src/dumpManager/dataSetGenerator.py
```

- Entraîner (exécution depuis le dossier racine) :

```bash
cd src
python main.py --train
```

- Reprendre un entraînement depuis un checkpoint :

```bash
cd src
python train.py --train --checkpoint checkpoint_epoch_10.pt
```

- Évaluer / tester un checkpoint :

```bash
cd src
python main.py --checkpoint checkpoint_epoch_10.pt
```

Options utiles
- `--no-export`  : ne génère pas les fichiers JSON pour le viewer
- `--no-tv`      : désactive la TV loss
- `--no-conv`    : désactive les couches de convolution

Sorties attendues
- Pendant l'entraînement : résumé par epoch avec `Loss`, `Acc` et `Time`, puis sauvegarde du modèle sous le chemin défini dans la configuration.
- Pendant l'évaluation : `Accuracy (raw)`, `Vote Accuracy` et un résumé des erreurs par type.

Remarques
- Vérifiez et adaptez les chemins définis dans [src/config.py](src/config.py) si nécessaire.
- Les checkpoints se trouvent dans le dossier [checkpoints](checkpoints) et le modèle final est `bytes_transformer_classifier_all.pth` à la racine.

Docker (recommandé)
- Pour tirer parti du `max-autotune` et obtenir les meilleures performances, il est recommandé d'exécuter le projet via l'image Docker fournie (qui permet d'utiliser Triton).
- Exemple rapide :

```bash
docker build -t raid:latest .
docker run --gpus all -v %CD%:/app -w /app raid:latest python src/main.py --train
```

Sur Windows vous pouvez aussi utiliser `docker_image.bat` pour simplifier la construction/exécution.

Un modèle entrainé sur 1GO de données est disponible à cet URL: https://huggingface.co/Hysteryx/Raid1Go 
Ses paramètres sont ceux en défaut dans `src/config.py`. Vous pouvez le télécharger et l'utiliser directement pour l'évaluation ou comme point de départ pour un entraînement ultérieur et / ou du fine-tuning.