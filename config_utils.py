import copy
import json
import warnings
from pathlib import Path


DEFAULT_CONFIG = {
    "device": "cuda",
    "run": {
        "name": "new_run",
        "output_root": ".output",
        "checkpoint_root": ".output/safetensors",
    },
    "model": {
        "id": "./new_beginning",
        "use_ema": True,
        "ema_decay": 0.9999,
        "use_bf16": True,
        "use_gradient_checkpointing": True,
        "train_text_encoder_1": True,
        "train_text_encoder_2": False,
        "use_torch_compile": False,
        "torch_compile_kwargs": {},
    },
    "training": {
        "output_dir": None,
        "batch_size": 4,
        "num_steps": 10_000,
        "num_epochs": None,
        "lr_unet": 5e-6,
        "lr_text_encoder_1": None,
        "lr_text_encoder_2": None,
        "log_every": 50,
        "checkpoint_every": 1_000,
        "grad_accum_steps": 1,
        "noise_offset": 0.1,
        "min_sigma": 0.4,
        "min_sigma_warmup_steps": 20,
        "min_timestep": None,
        "max_timestep": None,
        "prediction_type": "v_prediction",
        "snr_gamma": None,
        "max_grad_norm": None,
        "detect_anomaly": True,
        "lr_warmup_steps": 0,
        "ema_update_every": 10,
        "te_freeze_fraction": 0.7,
        "resume_from": None,
        "resume_state_path": None,
        "state_path": None,
        "seed": None,
        "lr_scheduler": {
            "type": None,
            "warmup_steps": 0,
            "min_factor": 0.0,
            "cycle_steps": 0,
            "cycle_mult": 1.0,
        },
        "tensorboard": {
            "enabled": False,
            "log_dir": None,
            "base_dir": "./logs/tensorboard",
            "log_grad_norm": False,
            "log_scaler": True,
        },
    },
    "data": {
        "image_dir": "./data/images",
        "size": 1024,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "caption_dropout_prob": 0.0,
        "caption_shuffle_prob": 0.0,
        "caption_shuffle_separator": ",",
        "caption_shuffle_min_tokens": 2,
        "bucket": {
            "enabled": False,
            "resolutions": [],
            "divisible_by": 64,
            "batch_size": None,
            "drop_last": True,
            "per_resolution_batch_sizes": {},
            "log_switches": False,
        },
        "latent_cache": {
            "enabled": False,
            "cache_dir": "./cache/latents",
            "dtype": "auto",
            "build_batch_size": 1,
        },
    },
    "optimizer": {
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
    },
    "export": {
        "save_single_file": True,
        "checkpoint_path": None,
        "converter_script": "./converttosdxl.py",
        "half_precision": True,
        "use_safetensors": True,
        "extra_args": [],
    },
    "eval": {
        "backend": "diffusers",
        "sampler_name": "dpmpp_2m_sde_heun",
        "scheduler": "EulerAncestralDiscreteScheduler",
        "num_inference_steps": 35,
        "cfg_scale": 6.5,
        "prompts_path": None,
        "height": None,
        "width": None,
        "use_ema": True,
        "live": {
            "enabled": False,
            "every_n_steps": 200,
            "max_batches": None,
        },
        "final": {
            "enabled": False,
            "max_batches": None,
        },
    },
}


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config

    try:
        with path.open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        config = _deep_update(config, user_config)
    except json.JSONDecodeError as err:
        warnings.warn(
            f"Konfiguration konnte nicht gelesen werden ({err}). Fallback auf Defaults.",
            stacklevel=2,
        )
    return config


__all__ = ["DEFAULT_CONFIG", "load_config"]
