# R.A.I.D - RAM Artificial Inspection Dump

## Objectif du Projet

**R.A.I.D** est un système d'apprentissage automatique conçu pour **classifier les bytes dans des dumps RAM** afin de déterminer s'ils sont **chiffrés ou clairs**. 

Ce projet utilise une **architecture Transformer** pour analyser les séquences de bytes et prédire leur nature (données chiffrées, compressées, texte, images, etc.).

### Cas d'usage
- Analyse de mémoire RAM pour la sécurité informatique
- Détection de données chiffrées dans les dumps mémoire
- Support pour les forensiques numériques
- Identification automatique de types de données

---

## Prérequis

- **Python 3.10+**
- **Training & Inference GPU Nvidia fortement recommandé (fonctionnel sur 5080 en train et 3070Ti pour inférence sans probleme)**

---

## Installation des Dépendances

### Dépendances principales

| Paquet | Version | Usage |
|--------|---------|-------|
| **torch** | >=1.10.0 | Framework deep learning principal |
| **numpy** | >=1.20.0 | Opérations numériques |
| **cryptography** | >=3.4 | Chiffrement AES pour génération données |
| **opencv-python** | >=4.5.0 | Traitement d'images |
| **pillow** | >=8.0.0 | Support supplémentaire images |

---

## Structure du Projet

```
raid/
├── src/
│   ├── main.py                          # Script d'évaluation du modèle
│   ├── train.py                         # Script d'entraînement
│   ├── config.py                        # Configuration centralisée
│   │
│   ├── dumpManager/
│   │   ├── dataSetGenerator.py          # Générateur de datasets synthétiques
│   │   └── RamDumpDataset.py            # Dataset custom PyTorch
│   │
│   ├── transformers/
│   │   └── bytesClassifier/
│   │       ├── BytesTransformerClassifier.py    # Architecture du modèle
│   │       └── PositionalEncoding.py            # Encodage positional
│   │
│   └── tools/
│       ├── visualizerExport.py          # Export et visualisation
│       └── downloader.py                # Outils de téléchargement
│
├── data/
│   ├── image/                           # Données d'images (sources)
│   ├── pdf/                             # Données PDF (sources)
│   └── text/                            # Données texte (sources)
│
├── output/
│   ├── ram_dump.bin                     # Dump RAM généré (binaire)
│   ├── metadata.json                    # Métadonnées d'annotation
│   └── SEEDS/                           # Résultats sérialisés
│
├── rapport/
│   ├── rapport.tex                      # Documentation LaTeX
│   └── references.bib                   # Bibliographie
│
├── bytes_transformer_classifier_all.pth # Modèle pré-entraîné
├── config.py                            # Configuration globale
├── LICENSE
└── README.md                            # Ce fichier
```

---

## Configuration

La configuration centralisée se trouve dans [src/config.py](src/config.py).

### Paramètres clés

```python
# Chemins des données
BIN_PATH = "../output/ram_dump.bin"           # Chemin du dump RAM
META_PATH = "../output/metadata.json"         # Chemin des métadonnées
MODEL_PATH = "../bytes_transformer_classifier_all.pth"  # Chemin du modèle

# Architecture du modèle Transformer
MODEL_CONFIG = {
    "dim_model": 128,        # Dimension cachée du modèle
    "num_heads": 4,          # Nombre de têtes d'attention
    "num_layers": 4,         # Nombre de couches Transformer
    "dim_ff": 512,           # Dimension feed-forward
    "dropout": 0.1,          # Dropout rate
    "max_len": 5000,         # Longueur maximale de séquence
    "local_conv_kernel_size": 3,
    "classifier_hidden_dim": 256,
}

# Configuration d'entraînement
TRAIN_CONFIG = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "num_epochs": 15,
    "batch_size": 32,
}

# Configuration du dataset
DATASET_CONFIG = {
    "chunk_size": 512,       # Taille des chunks de bytes
    "offset": 512,           # Offset entre les samples d'entraînement
}

# Configuration d'évaluation
EVAL_DATASET_CONFIG = {
    "chunk_size": 512,
    "offset": 128,           # Offset plus petit pour meilleure couverture
}
```

Modifiez ces valeurs dans `src/config.py` selon vos besoins.

---

##  Quick Start

### Générer un Dataset

```bash
cd src
python -c "from dumpManager.dataSetGenerator import DumpGenerator; \
gen = DumpGenerator(size_mb=100, seed=42); \
gen.generate_and_save('../output/ram_dump.bin', '../output/metadata.json')"
```

### Entraîner le Modèle

```bash
cd src
python train.py
```

**Sortie attendue:**
```
--- Entraînement sur cuda ---
Démarrage de l'entraînement...
Epoch 1 | Loss: 0.6521 | Acc: 62.34% | Time: 45.23s
Epoch 2 | Loss: 0.4891 | Acc: 75.12% | Time: 44.89s
...
Epoch 15 | Loss: 0.0912 | Acc: 98.45% | Time: 43.56s
Modèle sauvegardé sous : ../bytes_transformer_classifier_all.pth
Entraînement terminé.
```

```bash
cd src
python main.py
```

Tester un checkpoint spécifique:

```bash
cd src
python main.py --checkpoint checkpoint_epoch_10.pt
```

Vous pouvez aussi passer un chemin relatif ou absolu:

```bash
python main.py --checkpoint ../checkpoints/checkpoint_epoch_10.pt
```

**Sortie attendue:**
```
--- Évaluation sur cuda ---
Construction du modèle...
Poids chargés avec succès depuis : ../bytes_transformer_classifier_all.pth
Chargement du Dataset...
Démarrage de l'analyse...
Accuracy (raw): xx.xx%
Vote Accuracy: xx.xx%
Erreur Type: {...}
```

---

## Détails des Composants

### 1. Dataset Generator (`dumpManager/dataSetGenerator.py`)

Génère des dumps RAM synthétiques avec diverse contenu pour l'entraînement.

**Features:**
- Chiffrement AES des données
- Support images (JPG, PNG, BMP, etc.)
- Support PDF
- Support texte
- Compression (zlib, gzip)
- Encodage Base64
- Données de bruit contrôlées
- Métadonnées structurées

**Usage:**
```python
from dumpManager.dataSetGenerator import DumpGenerator

generator = DumpGenerator(
    size_mb=100,          # Taille du dump en MB
    seed=42,              # Seed aléatoire
    balance_mode="size"   # Mode de balance
)

#Il est possible de donner des poids par type (voir directement dans le fichier)

generator.generate_and_save(
    bin_output_path="../output/ram_dump.bin",
    meta_output_path="../output/metadata.json"
)
```

### 2. Custom Dataset (`dumpManager/RamDumpDataset.py`)

Dataset PyTorch pour charger les dumps RAM avec métadonnées.

**Labels:**
```
1 = ENCRYPTED      (Données chiffrées)
0 = COMPRESSED     (Données compressées)
0 = BINARY_TEXT    (Texte binaire)
0 = BINARY_IMAGE   (Images binaires)
0 = BINARY_PDF     (PDF binaires)
0 = BASE64         (Encodage Base64)
0 = NOISE          (Données de bruit)
```

### 3. Modèle Transformer (`transformers/bytesClassifier/BytesTransformerClassifier.py`)

Architecture Transformer pour classification séquence-à-séquence de bytes.

**Architecture:**
```
Entrée (bytes) → Embedding → Conv1D résiduelle → Positional Encoding → Transformer Encoder → MLP → Logits
                   256→128        motifs locaux       Sinusoïdal          4 couches       MLP élargi
```

Les anciens checkpoints ne sont pas tous directement chargeables. Si un checkpoint provient d'une ancienne tête de classification, certains tenseurs du classifier peuvent avoir les mêmes noms mais des dimensions différentes, ce qui empêche leur chargement automatique même avec `strict=False`. Dans ce cas, le checkpoint doit être converti pour ignorer/remplacer ces poids incompatibles, ou le modèle doit être réentraîné pour exploiter pleinement la Conv1D et la tête MLP plus profonde.

---

## Scripts Disponibles

### `train.py` - Entraînement

```bash
python src/train.py [--learning_rate LR] [--weight_decay WD] [--num_epochs NE] [--batch_size BS]
```

**Paramètres:**
- `--learning_rate`: Taux d'apprentissage (défaut: 1e-4)
- `--weight_decay`: Weight decay L2 (défaut: 1e-2)
- `--num_epochs`: Nombre d'epochs (défaut: 15)
- `--batch_size`: Batch size (défaut: 32)

**Outputs:**
- Modèle sauvegardé: `bytes_transformer_classifier_all.pth`
- Inclut la config du modèle et du training

### `main.py` - Évaluation

```bash
python src/main.py [--export]
```

**Paramètres:**
- `--export`: Exporter les résultats de visualisation (optionnel)

**Outputs:**
- Accuracy sur le dataset de test
- Matrice de confusion
- Rapport d'erreurs par type

### `dataSetGenerator.py` - Génération de Dataset

```bash
python -c "from dumpManager.dataSetGenerator import DumpGenerator; \
gen = DumpGenerator(size_mb=100, seed=42); \
gen.generate_and_save('path/to/output.bin', 'path/to/metadata.json')"
```

---

## Workflow Complet

### Étape 1: Préparer les données sources
```bash
# Utilisez l'utilitaire de telechargement ou placer vos fichiers dans data/
# - data/image/  (images JPG, PNG, etc.)
# - data/pdf/    (fichiers PDF)
# - data/text/   (fichiers TXT, JSON, etc.)
```

### Étape 2: Générer le dataset
```bash
cd src
python -c "
from dumpManager.dataSetGenerator import DumpGenerator
gen = DumpGenerator(size_mb=500, seed=123)
gen.generate_and_save('../output/ram_dump.bin', '../output/metadata.json')
"
```

### Étape 3: Entraîner le modèle
```bash
cd src
python train.py
```

### Étape 4: Évaluer les résultats
```bash
cd src
python main.py --export
```

### Étape 5: Analyser les résultats
```bash
# Les résultats sont dans output/SEEDS/
# Consultez les fichiers de visualisation générés
```

---

## Troubleshooting

### Erreur: "Aucun GPU détécté"
```python
# Le script utilise automatiquement CPU si GPU non disponible
# Pour forcer GPU ou CPU, modifiez dans config.py:
device = torch.device("cuda:0")  # Force GPU
device = torch.device("cpu")     # Force CPU, impossible pour train, utilisable en inférence sur très petit dataset
```

### Erreur: "Données introuvables"
```bash
# S'assurer que les fichiers existent:
ls output/ram_dump.bin
ls output/metadata.json

# Ou générer les données:
python -c "from dumpManager.dataSetGenerator import DumpGenerator; ..."
```

### CUDA out of memory
```python
# Réduire batch_size dans config.py:
TRAIN_CONFIG["batch_size"] = 16  # au lieu de 32
```

---

## Références

- **PyTorch**: https://pytorch.org/
- **Transformer Architecture**: "Attention is All You Need" (Vaswani et al., 2017)
- **RAID Project**: Voir [rapport/rapport.tex](rapport/rapport.tex)

---

## License

Ce projet est fourni sous la license [LICENSE](LICENSE).

---

---

## Support

Pour les questions ou problèmes, veuillez ouvrir une issue sur GitHub.

---

**Dernière mise à jour**: Mai 2026
