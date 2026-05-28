# Created on 5/18/2026 at 12:58:22 AM by Hysteryx

import os
import config as cfg

def resolve_checkpoint_path(checkpoint_name=None, is_model = False):
    """
    Resolve a training checkpoint path from an optional CLI argument.

    Accepted values for checkpoint_name:
    - None: start from scratch or load model from cfg.MODEL_PATH if is_model is True
    - simple file name (e.g. checkpoint_epoch_10.pt): searched in cfg.CHECKPOINT_DIR
    - relative path
    - absolute path
    """
    if not checkpoint_name:
        return None if not is_model else cfg.MODEL_PATH
    
    if os.path.isabs(checkpoint_name):
        return checkpoint_name

    if os.path.sep in checkpoint_name or "/" in checkpoint_name:
        return os.path.abspath(checkpoint_name)

    return os.path.join(cfg.CHECKPOINT_DIR, checkpoint_name)


def _unwrap_compiled_state_dict(state_dict):
    """
    Remove _orig_mod. prefix from compiled model state_dict keys.
    Useful when loading checkpoints saved with torch.compile.
    """
    unwrapped = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            unwrapped[key[10:]] = value
        else:
            unwrapped[key] = value
    return unwrapped