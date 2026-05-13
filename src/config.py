import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BIN_PATH = os.path.join(BASE_DIR, "../output/ram_dump.bin")
META_PATH = os.path.join(BASE_DIR, "../output/metadata.json")
MODEL_PATH = os.path.join(BASE_DIR, "../bytes_transformer_classifier_all.pth")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "../checkpoints")
VISUAL_EXPORT_DIR = os.path.join(BASE_DIR, "../output")

DO_COMPILE_MODEL = True

# "aot_eager" pour compatibilité Windows (ne supporte pas mode)
# "inductor" stable et équilibré (need triton)
COMPILE_BACKEND = "aot_eager"
# "default"
# "reduce-overhead" pour accélérer les petits modèles (mais peut causer des problèmes de convergence)
# "max-autotune" pour laisser plus de liberté au backend (peut améliorer les performances mais aussi causer des problèmes de convergence)
COMPILE_MODE = "default"  # aot_eager ne supporte pas les modes d'optimisation


DEFAULT_CHUNK_SIZE = 512
DEFAULT_DATASET_OFFSET = 512
DEFAULT_EVAL_OFFSET = 128
DEFAULT_BATCH_SIZE = 128 #A passer à 128 dans test 2
DEFAULT_NUM_WORKERS = 8
DEFAULT_PIN_MEMORY = True

MODEL_CONFIG = {
	"padding_idx": 256,
	"vocab_size": 257,
	"dim_model": 128,
	"num_heads": 4,
	"num_layers": 4,
	"dim_ff": 512,
	"dropout": 0.05,
	"max_len": DEFAULT_CHUNK_SIZE,
	"local_conv_kernel_size": 3,
	"classifier_hidden_dim": 512, #256 normalement, à 512 pour test si donner + de place au classifieur améliore les résultats
}

TRAIN_CONFIG = {
	"learning_rate": 5e-4, #Unused if scheduler is enabled, as lr will be managed by the scheduler
	"weight_decay": 1e-2,
	"num_epochs": 20,
	"batch_size": DEFAULT_BATCH_SIZE,
}

SCHEDULER_CONFIG = {
	"enabled": True,
	"type": "cosine",  # "cosine" | "plateau"
	"cosine": {
		"T_max": TRAIN_CONFIG["num_epochs"],
		"eta_min": 1e-6,
	},
	"plateau": {
		"mode": "min",
		"factor": 0.5,
		"patience": 4,
		"min_lr": 1e-6,
	},
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
    "prefetch_factor": 2, #Nombre de batchs à précharger par worker
}

GENERATOR_CONFIG = {
	"default_size_mb": 50,
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
    "prefetch_factor": 2, #Nombre de batchs à précharger par worker
}