import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BIN_PATH = os.path.join(BASE_DIR, "../output/ram_dump.bin")
META_PATH = os.path.join(BASE_DIR, "../output/metadata.json")
MODEL_PATH = os.path.join(BASE_DIR, "../bytes_transformer_classifier_all.pth")
VISUAL_EXPORT_DIR = os.path.join(BASE_DIR, "../output")