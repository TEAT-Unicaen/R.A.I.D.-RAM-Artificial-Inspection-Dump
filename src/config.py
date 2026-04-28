import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BIN_PATH = os.path.join(BASE_DIR, "../output/ram_dump.bin")
META_PATH = os.path.join(BASE_DIR, "../output/metadata.json")
MODEL_PATH = os.path.join(BASE_DIR, "../bytes_transformer_classifier_all.pth")
VISUAL_EXPORT_DIR = os.path.join(BASE_DIR, "../output")

DEFAULT_CHUNK_SIZE = 512
DEFAULT_DATASET_OFFSET = 512
DEFAULT_EVAL_OFFSET = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_PIN_MEMORY = True

MODEL_CONFIG = {
	"dim_model": 128,
	"num_heads": 4,
	"num_layers": 4,
	"dim_ff": 512,
	"dropout": 0.1,
	"max_len": 5000,
}

TRAIN_CONFIG = {
	"learning_rate": 1e-3,
	"weight_decay": 1e-2,
	"num_epochs": 15,
	"batch_size": 32,
}

DATASET_CONFIG = {
	"chunk_size": DEFAULT_CHUNK_SIZE,
	"offset": DEFAULT_DATASET_OFFSET,
}

EVAL_DATASET_CONFIG = {
	"chunk_size": DEFAULT_CHUNK_SIZE,
	"offset": DEFAULT_EVAL_OFFSET,
}

EVAL_CONFIG = {
	"batch_size": DEFAULT_BATCH_SIZE,
}

TRAIN_LOADER_CONFIG = {
	"num_workers": DEFAULT_NUM_WORKERS,
	"pin_memory": DEFAULT_PIN_MEMORY,
}

GENERATOR_CONFIG = {
	"default_size_mb": 10,
	"default_seed": 42,
	"memory_alignment": 16,
	"image_fragment_threshold": 5000,
	"image_fragment_min": 5000,
	"image_fragment_max": 200000,
	"chunk_fragment_min": 1024,
	"chunk_fragment_max": 500000,
	"noise_gap_min": 32,
	"noise_gap_max": 256,
	"share_quantum": DEFAULT_CHUNK_SIZE,
	"default_noise_level": 0.2,
	"default_balance_mode": "size",
}

EVAL_LOADER_CONFIG = {
	"num_workers": DEFAULT_NUM_WORKERS,
	"pin_memory": DEFAULT_PIN_MEMORY,
}