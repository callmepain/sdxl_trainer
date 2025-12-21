"""
Configuration utilities for distillation.

Provides separate DEFAULT configs for:
- Cache builder (build_teacher_cache.py)
- Distillation training (train_distill.py)
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CACHE_BUILDER_CONFIG = {
    "device": "cuda",
    "teacher": {
        "checkpoint_path": None,  # Required: path to diffusers checkpoint
        "teacher_id": None,       # Optional: derived from checkpoint_path if not set
        "use_bf16": True,
    },
    "data": {
        "image_dir": "./data/images",
        "size": 1024,
        "bucket": {
            "enabled": False,
            "resolutions": [],
            "divisible_by": 64,
        },
        "latent_cache": {
            "enabled": True,
            "cache_dir": "./cache/latents",
        },
        "num_workers": 0,  # For cache building, single-threaded is often better
    },
    "cache": {
        "output_dir": "./cache/teacher_predictions",
        "version": "1",
        "dtype": "fp16",  # fp16 or bf16
        "base_seed": 42,
        "min_timestep": 0,
        "max_timestep": 1000,
        "skip_existing": True,  # Skip if cache file already exists
    },
    "model": {
        # Base model for tokenizers (if different from teacher checkpoint)
        "id": None,  # If None, uses teacher checkpoint for tokenizers
    },
}


DEFAULT_DISTILL_CONFIG = {
    "device": "cuda",
    "run": {
        "name": "distill_run",
        "output_root": ".output/distill",
    },
    "student": {
        "checkpoint_path": None,  # Starting checkpoint for student (required)
        "use_bf16": True,
        "use_gradient_checkpointing": True,
        "use_ema": True,
        "ema_decay": 0.9999,
        "ema_update_every": 10,
    },
    "teachers": [
        # List of teacher configurations
        # Each entry: {"teacher_id": "...", "cache_dir": "...", "weight": 1.0}
    ],
    "training": {
        "batch_size": 4,
        "num_steps": None,
        "num_epochs": None,
        "lr": 5e-6,
        "grad_accum_steps": 1,
        "log_every": 50,
        "checkpoint_every": 1000,
        "max_grad_norm": 1.0,
        "loss_type": "mse",  # mse, huber, smooth_l1
        "lr_scheduler": {
            "type": "cosine_decay",
            "warmup_steps": 200,
            "min_factor": 0.1,
        },
        "tensorboard": {
            "enabled": True,
            "base_dir": "./logs/tensorboard/distill",
            "log_per_teacher_loss": True,
            "log_grad_norm": True,
        },
    },
    "data": {
        "image_dir": "./data/images",
        "size": 1024,
        "bucket": {
            "enabled": False,
            "resolutions": [],
            "drop_last": True,
            "per_resolution_batch_sizes": {},
        },
        "latent_cache": {
            "enabled": True,
            "cache_dir": "./cache/latents",
        },
        "num_workers": 4,
        "pin_memory": True,
    },
    "cache": {
        "version": "1",  # Must match cache builder version
        "require_all_teachers": True,  # Skip samples missing any teacher cache
    },
    "optimizer": {
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    },
    "export": {
        "save_single_file": True,
        "converter_script": "./converttosdxl.py",
        "half_precision": True,
    },
}


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively update base dict with updates."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_cache_builder_config(path: Path) -> Dict[str, Any]:
    """
    Load cache builder configuration with defaults.

    Args:
        path: Path to config JSON file

    Returns:
        Merged configuration dict
    """
    config = copy.deepcopy(DEFAULT_CACHE_BUILDER_CONFIG)
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
        config = _deep_update(config, user_config)
    else:
        raise FileNotFoundError(f"Config file not found: {path}")
    return config


def load_distill_config(path: Path) -> Dict[str, Any]:
    """
    Load distillation training configuration with defaults.

    Args:
        path: Path to config JSON file

    Returns:
        Merged configuration dict
    """
    config = copy.deepcopy(DEFAULT_DISTILL_CONFIG)
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
        config = _deep_update(config, user_config)
    else:
        raise FileNotFoundError(f"Config file not found: {path}")
    return config
