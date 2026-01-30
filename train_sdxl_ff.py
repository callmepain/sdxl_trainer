import warnings
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import bitsandbytes as bnb
warnings.filterwarnings(
    "ignore",
    message=".*cuda capability.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Skipping import of cpp extensions due to incompatible torch version.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="pkg_resources",
)
warnings.filterwarnings(
    "ignore",
    message="Should have ta>=t0 but got ta=.*",
    category=UserWarning,
    module="torchsde._brownian.brownian_interval",
)
from torch.utils.data import DataLoader
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DEISMultistepScheduler,
    EDMEulerScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.training_utils import EMAModel
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from dataset import BucketBatchSampler, SimpleCaptionDataset
from config_utils import load_config
from eval_export import EvalRunner
from optim_utils import LearningRateController
from training.trainer import (
    BucketState,
    ResumeState,
    TrainerIO,
    TrainerModules,
    TrainingLoopSettings,
    TrainingPaths,
    run_training_loop,
)
from training.bucket_utils import bucket_key_from_target_size, bucket_sort_key, normalize_bucket_key
from training.min_sigma import MinSigmaController


CONFIG_PATH = Path("config.json")


cfg = load_config(CONFIG_PATH)

flash_attn_func = None
flash_attn_backend = None
flash_attn_supports_dropout = False
flash_attn_supports_softmax_scale = False

try:
    from flash_attn_interface import flash_attn_func  # FlashAttention-3 wheels expose this module
    flash_attn_backend = "flash_attn_interface"
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func  # FlashAttention<=2.x
        flash_attn_backend = "flash_attn.flash_attn_interface"
    except ImportError:
        try:
            from flash_attn import flash_attn_func  # Legacy API
            flash_attn_backend = "flash_attn"
        except ImportError:
            flash_attn_func = None

if flash_attn_func is None:
    warnings.warn(
        "FlashAttention wurde nicht gefunden. Verwende Standard-Attention – aktiviere die richtige venv oder installiere flash-attn-3.",
        stacklevel=2,
    )
else:
    import inspect

    try:
        signature = inspect.signature(flash_attn_func)
        flash_attn_supports_dropout = "dropout_p" in signature.parameters
        flash_attn_supports_softmax_scale = "softmax_scale" in signature.parameters
    except (ValueError, TypeError):
        pass

    print(
        f"FlashAttention aktiviert (Backend: {flash_attn_backend}, "
        f"dropout={'ja' if flash_attn_supports_dropout else 'nein'}, "
        f"scale_kw={'ja' if flash_attn_supports_softmax_scale else 'nein'})"
    )


class FlashAttnProcessor:
    """Wrapper, der automatisch auf FlashAttention-3 fällt, falls verfügbar."""

    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(self):
        self._warned = False
        self._dropout_warned = False

    def _can_use_flash(self, hidden_states, attention_mask, head_dim):
        if flash_attn_func is None:
            return False
        if attention_mask is not None:
            return False
        if hidden_states.device.type != "cuda":
            return False
        if hidden_states.dtype not in self.SUPPORTED_DTYPES:
            return False
        if head_dim > 256 or head_dim % 8 != 0:
            return False
        return True

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "Das `scale`-Argument wird ignoriert. Bitte entferne die Übergabe."
            )
            warnings.warn(deprecation_message, stacklevel=2)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        height = width = channel = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        use_flash = self._can_use_flash(hidden_states, attention_mask, head_dim)

        if use_flash:
            try:
                flash_query = query.transpose(1, 2).contiguous()
                flash_key = key.transpose(1, 2).contiguous()
                flash_value = value.transpose(1, 2).contiguous()
                dropout_p = attn.dropout if attn.training else 0.0
                attn_scale = attn.scale if getattr(attn, "scale_qk", True) else None
                flash_kwargs = {"causal": False}
                if attn_scale is not None and flash_attn_supports_softmax_scale:
                    flash_kwargs["softmax_scale"] = attn_scale
                if dropout_p and dropout_p > 0.0:
                    if flash_attn_supports_dropout:
                        flash_kwargs["dropout_p"] = dropout_p
                    elif not self._dropout_warned:
                        warnings.warn(
                            "FlashAttention-Backend unterstützt kein Dropout. Dropout wird ignoriert.",
                            stacklevel=2,
                        )
                        self._dropout_warned = True
                flash_hidden_states = flash_attn_func(
                    flash_query,
                    flash_key,
                    flash_value,
                    **flash_kwargs,
                )
                hidden_states = flash_hidden_states.transpose(1, 2)
            except RuntimeError as err:
                if not self._warned:
                    warnings.warn(
                        f"FlashAttention deaktiviert, fallback auf SDP: {err}",
                        stacklevel=2,
                    )
                    self._warned = True
                hidden_states = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        else:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

device = cfg["device"]

run_cfg = cfg.get("run", {})
run_name = run_cfg.get("name")
output_root = Path(run_cfg.get("output_root", ".")).expanduser()
checkpoint_root_cfg = run_cfg.get("checkpoint_root")
if checkpoint_root_cfg is not None:
    checkpoint_root = Path(checkpoint_root_cfg).expanduser()
else:
    checkpoint_root = output_root / "safetensors"

model_cfg = cfg["model"]
training_cfg = cfg["training"]
data_cfg = cfg["data"]
optimizer_cfg = cfg["optimizer"]
export_cfg = cfg.get("export", {})
eval_cfg = cfg.get("eval", {}) or {}

model_id = model_cfg["id"]
resume_from_cfg = training_cfg.get("resume_from")
resume_from_path = Path(resume_from_cfg).expanduser() if resume_from_cfg else None
if resume_from_path is not None and not resume_from_path.exists():
    warnings.warn(f"Resume-Pfad {resume_from_path} nicht gefunden. Starte vom Basis-Modell.", stacklevel=2)
    resume_from_path = None
output_dir_cfg = training_cfg.get("output_dir")
if output_dir_cfg:
    output_dir = str(output_dir_cfg)
elif resume_from_path is not None:
    output_dir = str(resume_from_path)
else:
    if not run_name:
        raise ValueError("Setze entweder `training.output_dir` oder `run.name` in der config.")
    output_dir = str(output_root / run_name)
    training_cfg["output_dir"] = output_dir
model_load_path = str(resume_from_path) if resume_from_path is not None else model_id
state_path_cfg = training_cfg.get("state_path")
if state_path_cfg:
    state_save_path = Path(state_path_cfg).expanduser()
else:
    state_save_path = Path(output_dir).expanduser() / "trainer_state.pt"
resume_state_path_cfg = training_cfg.get("resume_state_path")
if resume_state_path_cfg:
    resume_state_path = Path(resume_state_path_cfg).expanduser()
elif resume_from_path is not None:
    resume_state_path = resume_from_path / "trainer_state.pt"
else:
    resume_state_path = state_save_path
resume_state_dict = None
resume_global_step = 0
resume_epoch = 0
resume_state_active = bool(resume_from_path is not None or resume_state_path_cfg is not None)
if resume_state_active and resume_state_path is not None and resume_state_path.exists():
    try:
        resume_state_dict = torch.load(resume_state_path, map_location="cpu")
        resume_global_step = int(resume_state_dict.get("global_step", 0) or 0)
        resume_epoch = int(resume_state_dict.get("epoch", 0) or 0)
        print(
            f"Trainer-State geladen aus {resume_state_path} "
            f"(step={resume_global_step}, epoch={resume_epoch})"
        )
    except Exception as exc:
        warnings.warn(f"Trainer-State konnte nicht geladen werden ({exc}). Starte frisch.", stacklevel=2)
        resume_state_dict = None
        resume_global_step = 0
        resume_epoch = 0
elif resume_state_active and resume_state_path is not None:
    warnings.warn(
        f"Kein Trainer-State unter {resume_state_path} gefunden. Gewichte werden geladen, Optimizer startet frisch.",
        stacklevel=2,
    )
batch_size = training_cfg["batch_size"]
num_steps = training_cfg.get("num_steps")
if num_steps is not None:
    num_steps = int(num_steps)
num_epochs = training_cfg.get("num_epochs")
if num_epochs is not None:
    num_epochs = int(num_epochs)
if num_steps is None and num_epochs is None:
    raise ValueError("Setze bitte entweder `training.num_steps` oder `training.num_epochs` in der config.")
lr_unet = training_cfg["lr_unet"]
use_ema = model_cfg["use_ema"]
ema_decay_cfg = model_cfg.get("ema_decay", 0.9999)
if ema_decay_cfg is None:
    ema_decay = 0.9999
else:
    try:
        ema_decay = float(ema_decay_cfg)
    except (TypeError, ValueError):
        warnings.warn(f"Ungültiger ema_decay-Wert ({ema_decay_cfg}), fallback zu 0.9999.", stacklevel=2)
        ema_decay = 0.9999
if not (0.0 < ema_decay < 1.0):
    warnings.warn(
        f"ema_decay muss zwischen 0 und 1 liegen (exklusive). Erhalte {ema_decay_cfg}, fallback zu 0.9999.",
        stacklevel=2,
    )
    ema_decay = 0.9999
use_bf16 = model_cfg["use_bf16"]
use_gradient_checkpointing = model_cfg["use_gradient_checkpointing"]
train_text_encoder_1 = model_cfg.get("train_text_encoder_1")
if train_text_encoder_1 is None:
    train_text_encoder_1 = True
train_text_encoder_1 = bool(train_text_encoder_1)
train_text_encoder_2 = bool(model_cfg.get("train_text_encoder_2", False))
lr_te1 = training_cfg.get("lr_text_encoder_1")
lr_te2 = training_cfg.get("lr_text_encoder_2")
if train_text_encoder_1 and lr_te1 is None:
    raise ValueError("Setze `training.lr_text_encoder_1`, um Text-Encoder 1 zu trainieren.")
if train_text_encoder_2 and lr_te2 is None:
    raise ValueError("Setze `training.lr_text_encoder_2`, um Text-Encoder 2 zu trainieren.")
use_torch_compile = bool(model_cfg.get("use_torch_compile", False))
torch_compile_kwargs = model_cfg.get("torch_compile_kwargs", {}) or {}
log_every = training_cfg.get("log_every", 50)
if log_every is not None:
    log_every = max(1, int(log_every))
checkpoint_every = training_cfg.get("checkpoint_every", 1_000)
if checkpoint_every is not None:
    checkpoint_every = max(1, int(checkpoint_every))
grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
noise_offset = float(training_cfg.get("noise_offset", 0.0) or 0.0)
min_sigma = training_cfg.get("min_sigma")
if min_sigma is not None:
    min_sigma = float(min_sigma)
if min_sigma is not None and min_sigma <= 0:
    min_sigma = None
min_sigma_warmup_steps = int(training_cfg.get("min_sigma_warmup_steps", 0) or 0)
if min_sigma is not None and min_sigma_warmup_steps <= 0:
    warnings.warn(
        "min_sigma ist gesetzt, aber min_sigma_warmup_steps <= 0. Feature wird deaktiviert.",
        stacklevel=2,
    )
    min_sigma = None
    min_sigma_warmup_steps = 0
else:
    min_sigma_warmup_steps = max(0, min_sigma_warmup_steps)

min_timestep = training_cfg.get("min_timestep")
max_timestep = training_cfg.get("max_timestep")
if min_timestep is not None:
    min_timestep = int(min_timestep)
    if min_timestep < 0:
        warnings.warn("min_timestep < 0, wird auf 0 gesetzt.", stacklevel=2)
        min_timestep = 0
if max_timestep is not None:
    max_timestep = int(max_timestep)
    if max_timestep < 0:
        warnings.warn("max_timestep < 0, wird ignoriert.", stacklevel=2)
        max_timestep = None
if min_timestep is not None and max_timestep is not None and min_timestep >= max_timestep:
    warnings.warn(
        f"min_timestep ({min_timestep}) >= max_timestep ({max_timestep}), beide werden ignoriert.",
        stacklevel=2,
    )
    min_timestep = None
    max_timestep = None

prediction_type_override = training_cfg.get("prediction_type")
snr_gamma = training_cfg.get("snr_gamma")
if snr_gamma is not None:
    snr_gamma = float(snr_gamma)
max_grad_norm = training_cfg.get("max_grad_norm")
if max_grad_norm is not None:
    max_grad_norm = float(max_grad_norm)
    if max_grad_norm < 0:
        warnings.warn(
            f"max_grad_norm muss >= 0 sein, erhalten: {max_grad_norm}. Feature wird deaktiviert.",
            stacklevel=2,
        )
        max_grad_norm = None
    elif max_grad_norm == 0:
        warnings.warn(
            "max_grad_norm=0 deaktiviert Gradient Clipping (alle Gradienten auf 0 gesetzt). Meintest du None?",
            stacklevel=2,
        )
detect_anomaly = bool(training_cfg.get("detect_anomaly", True))
ema_update_every = int(training_cfg.get("ema_update_every", 10) or 10)
if ema_update_every < 1:
    warnings.warn("ema_update_every < 1 ist ungültig – fallback auf 1.", stacklevel=2)
    ema_update_every = 1
tensorboard_cfg = training_cfg.get("tensorboard", {}) or {}
use_tensorboard = bool(tensorboard_cfg.get("enabled", False))
tb_log_grad_norm = bool(tensorboard_cfg.get("log_grad_norm", False))
tb_log_scaler = bool(tensorboard_cfg.get("log_scaler", True))
tb_writer = None
if use_tensorboard:
    log_dir_value = tensorboard_cfg.get("log_dir")
    if log_dir_value:
        tb_log_dir = Path(log_dir_value).expanduser()
    else:
        base_dir = Path(tensorboard_cfg.get("base_dir", "./logs/tensorboard")).expanduser()
        run_dir_name = run_name or "run"
        tb_log_dir = base_dir / run_dir_name
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
    try:
        tb_writer.add_text("run/config", json.dumps(cfg, indent=2))
    except Exception:
        pass

checkpoint_path_cfg = export_cfg.get("checkpoint_path")
if checkpoint_path_cfg:
    checkpoint_path = str(checkpoint_path_cfg)
else:
    if not run_name:
        raise ValueError("Setze entweder `export.checkpoint_path` oder `run.name` in der config.")
    checkpoint_path = str(checkpoint_root / f"{run_name}.safetensors")
    export_cfg["checkpoint_path"] = checkpoint_path

dtype = torch.bfloat16 if use_bf16 else torch.float16
vae_dtype = torch.float32


def _format_dtype(value):
    if value is None:
        return "unknown"
    if value == torch.bfloat16:
        return "bfloat16"
    if value == torch.float16:
        return "float16"
    if value == torch.float32:
        return "float32"
    return str(value)


def _module_dtype(module):
    if module is None:
        return None
    for param in module.parameters():
        return param.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return None


def _wrap_vae_decode_for_dtype(vae_module, target_dtype):
    if vae_module is None or target_dtype is None:
        return
    if getattr(vae_module, "_decode_wrapped_for_dtype", False):
        return
    original_decode = vae_module.decode

    def decode_with_cast(latents, *args, **kwargs):
        latents = latents.to(dtype=target_dtype)
        return original_decode(latents, *args, **kwargs)

    vae_module.decode = decode_with_cast
    vae_module._decode_wrapped_for_dtype = True

seed_value = training_cfg.get("seed")
if seed_value is not None:
    try:
        seed_value = int(seed_value)
    except (TypeError, ValueError):
        warnings.warn(f"Ungültiger seed-Wert ({seed_value}), verwende zufälligen Seed.", stacklevel=2)
        seed_value = None
if seed_value is not None and seed_value <= 0:
    seed_value = None
if seed_value is not None:
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    print(f"Seed gesetzt: {seed_value}")
else:
    print("Seed: random (kein fixer Seed gesetzt)")

# 1) Tokenizer laden (ohne Pipeline)
tokenizer_1 = AutoTokenizer.from_pretrained(
    model_load_path, subfolder="tokenizer", use_fast=False
)
tokenizer_2 = AutoTokenizer.from_pretrained(
    model_load_path, subfolder="tokenizer_2", use_fast=False
)

# 2) Dataset + Dataloader
caption_dropout_prob = data_cfg.get("caption_dropout_prob", 0.0)
caption_shuffle_prob = data_cfg.get("caption_shuffle_prob", 0.0)
caption_shuffle_separator = data_cfg.get("caption_shuffle_separator", ",")
caption_shuffle_min_tokens = data_cfg.get("caption_shuffle_min_tokens", 2)
bucket_cfg = data_cfg.get("bucket", {})
latent_cache_cfg = data_cfg.get("latent_cache", {})
bucket_log_switches = bool(bucket_cfg.get("log_switches", False))
per_bucket_cfg = bucket_cfg.get("per_resolution_batch_sizes") or {}
bucket_batch_size_map = {}
if isinstance(per_bucket_cfg, dict):
    for raw_key, raw_value in per_bucket_cfg.items():
        normalized = normalize_bucket_key(raw_key)
        if not normalized:
            continue
        try:
            bucket_batch_size_map[normalized] = max(1, int(raw_value))
        except (TypeError, ValueError):
            continue

bucket_enabled = bool(bucket_cfg.get("enabled", False))
bucket_default_batch_size = int(bucket_cfg.get("batch_size") or batch_size) if bucket_enabled else batch_size
effective_batch = bucket_default_batch_size * grad_accum_steps
bucket_effective_batches = {
    key: value * grad_accum_steps for key, value in bucket_batch_size_map.items()
}
run_summary = []
run_summary.append(("batch.effective_default", effective_batch))
if bucket_effective_batches:
    run_summary.append(("batch.effective_buckets", bucket_effective_batches))
    eff_values = [effective_batch, *bucket_effective_batches.values()]
    run_summary.append(("batch.effective_range", {"min": min(eff_values), "max": max(eff_values)}))
run_summary.append(("batch.grad_accum_steps", grad_accum_steps))
run_summary.append(("batch.training", batch_size))
run_summary.append(("bucket.enabled", bucket_enabled))
if bucket_enabled:
    run_summary.append(("bucket.batch_size_default", bucket_default_batch_size))
    if bucket_batch_size_map:
        run_summary.append(("bucket.per_resolution_batch_sizes", bucket_batch_size_map))
run_summary.append(("bucket.log_switches", bucket_log_switches))
min_sigma_summary = (
    f"{min_sigma} (warmup_steps={min_sigma_warmup_steps})" if min_sigma is not None else "deaktiviert"
)
timestep_range_parts = []
if min_timestep is not None:
    timestep_range_parts.append(f"min={min_timestep}")
if max_timestep is not None:
    timestep_range_parts.append(f"max={max_timestep}")
timestep_range_summary = ", ".join(timestep_range_parts) if timestep_range_parts else "unrestricted"
run_summary.append(("device", device))
run_summary.append(("dtype.train", _format_dtype(dtype)))
run_summary.append(("min_sigma", min_sigma_summary))
run_summary.append(("timestep_range", timestep_range_summary))
run_summary.append(("model.id", model_load_path))
run_summary.append(("run.name", run_name or "n/a"))
run_summary.append(
    (
        "text_encoder.1",
        "train" + (f" (lr={lr_te1})" if train_text_encoder_1 else ""),
    )
    if train_text_encoder_1
    else ("text_encoder.1", "frozen"),
)
run_summary.append(
    (
        "text_encoder.2",
        "train" + (f" (lr={lr_te2})" if train_text_encoder_2 else ""),
    )
    if train_text_encoder_2
    else ("text_encoder.2", "frozen"),
)
if tb_writer is not None:
    tb_writer.add_scalar("data/effective_batch", effective_batch, 0)
    tb_writer.add_scalar("data/grad_accum_steps", grad_accum_steps, 0)
    tb_writer.add_text("run/meta", f"model_id: {model_load_path}\nrun_name: {run_name or 'n/a'}\noutput_dir: {output_dir}")

train_dataset = SimpleCaptionDataset(
    img_dir=data_cfg["image_dir"],
    tokenizer_1=tokenizer_1,
    tokenizer_2=tokenizer_2,
    size=data_cfg["size"],
    caption_dropout_prob=caption_dropout_prob,
    caption_shuffle_prob=caption_shuffle_prob,
    caption_shuffle_separator=caption_shuffle_separator,
    caption_shuffle_min_tokens=caption_shuffle_min_tokens,
    bucket_config=bucket_cfg,
    latent_cache_config=latent_cache_cfg,
    pixel_dtype=vae_dtype,
)
if tb_writer is not None:
    tb_writer.add_scalar("data/num_samples", len(train_dataset), 0)

if bucket_enabled:
    bucket_counts = Counter(train_dataset.sample_buckets)
    if bucket_counts:
        print("Bucket-Verteilung:")
        for key in sorted(bucket_counts.keys(), key=bucket_sort_key):
            normalized_key = normalize_bucket_key(key) or key
            effective_size = bucket_batch_size_map.get(normalized_key, bucket_default_batch_size)
            print(f"  {key}: {bucket_counts[key]} samples (batch_size={effective_size})")
            if tb_writer is not None:
                tb_writer.add_scalar(f"data/buckets/{key}", bucket_counts[key], 0)

    bucket_sampler = BucketBatchSampler(
        train_dataset.sample_buckets,
        batch_size=bucket_default_batch_size,
        shuffle=data_cfg.get("shuffle", True),
        drop_last=bucket_cfg.get("drop_last", True),
        per_bucket_batch_sizes=bucket_batch_size_map,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=bucket_sampler,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
else:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=data_cfg.get("shuffle", True),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

if len(train_loader) == 0:
    raise ValueError("Trainings-Dataloader ist leer. Bitte überprüfe den Dataset-Pfad oder den Inhalt.")

steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
if num_steps is not None:
    total_progress_steps = num_steps
elif num_epochs is not None:
    total_progress_steps = steps_per_epoch * num_epochs
else:
    total_progress_steps = None

te_freeze_step = None
te_freeze_fraction = training_cfg.get("te_freeze_fraction", 0.7)
if te_freeze_fraction is not None:
    try:
        te_freeze_fraction = float(te_freeze_fraction)
    except (TypeError, ValueError):
        warnings.warn(f"Ungültiger te_freeze_fraction-Wert ({te_freeze_fraction}), Feature deaktiviert.", stacklevel=2)
        te_freeze_fraction = None
if te_freeze_fraction is not None:
    if te_freeze_fraction <= 0.0 or te_freeze_fraction >= 1.0:
        warnings.warn("te_freeze_fraction muss zwischen 0 und 1 liegen – Feature deaktiviert.", stacklevel=2)
        te_freeze_fraction = None

if total_progress_steps is not None and te_freeze_fraction is not None:
    te_freeze_step = max(1, int(total_progress_steps * te_freeze_fraction))
    run_summary.append(("text_encoder.freeze_step", te_freeze_step))
else:
    run_summary.append(("text_encoder.freeze_step", "disabled (num_steps/epochs missing or fraction unset)"))

def encode_text(captions_batch):
    # Wir haben schon input_ids aus dem Dataset, daher hier just forward:
    input_ids_1 = captions_batch["input_ids_1"].to(device)
    attn_1 = captions_batch["attention_mask_1"].to(device)
    input_ids_2 = captions_batch["input_ids_2"].to(device)
    attn_2 = captions_batch["attention_mask_2"].to(device)

    with torch.set_grad_enabled(train_text_encoder_1):
        enc_1 = te1(
            input_ids_1,
            attention_mask=attn_1,
            output_hidden_states=True,
            return_dict=True,
        )
    with torch.set_grad_enabled(train_text_encoder_2):
        enc_2 = te2(
            input_ids_2,
            attention_mask=attn_2,
            output_hidden_states=True,
            return_dict=True,
        )

    prompt_embeds_1 = enc_1.hidden_states[-2]
    prompt_embeds_2 = enc_2.hidden_states[-2]

    if prompt_embeds_2.shape[1] != prompt_embeds_1.shape[1]:
        prompt_embeds_2 = prompt_embeds_2.expand(
            -1, prompt_embeds_1.shape[1], -1
        )

    text_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    pooled_embeds = enc_2.text_embeds

    # Text-Embeddings im Training-dtype (BF16/FP16) für Kompatibilität mit UNet
    return text_embeds.to(dtype), pooled_embeds.to(dtype)


VAE_SCALING_FACTOR = 0.18215


def _count_parameters(modules):
    total = 0
    trainable = 0
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            numel = param.numel()
            total += numel
            if param.requires_grad:
                trainable += numel
    return total, trainable


def prepare_latents(pixel_values, generator=None):
    pixel_values = pixel_values.to(device=device, dtype=vae_dtype)
    # VAE encode
    with torch.no_grad():
        posterior = vae.encode(pixel_values).latent_dist
        latents = posterior.sample(generator=generator) * VAE_SCALING_FACTOR
    return latents.to(dtype=dtype)


def _resolve_cache_dtype(value, default_dtype):
    if value is None:
        return default_dtype
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("auto", "none", ""):
            return default_dtype
        if key in ("fp16", "float16", "half"):
            return torch.float16
        if key in ("bf16", "bfloat16"):
            return torch.bfloat16
        if key in ("fp32", "float32", "full"):
            return torch.float32
    if isinstance(value, torch.dtype):
        return value
    raise ValueError(f"Unbekannter Latent-Cache-Datentyp: {value}")


def ensure_latent_cache(dataset, cache_cfg):
    if not getattr(dataset, "latent_cache_enabled", False):
        return

    dataset.refresh_latent_cache_state()
    missing = dataset.get_missing_latent_indices()
    if not missing:
        dataset.activate_latent_cache()
        print("Latent-Cache: bestehende Dateien werden verwendet.")
        return

    build_batch_size = max(1, int(cache_cfg.get("build_batch_size", 1)))
    cache_dtype = _resolve_cache_dtype(cache_cfg.get("dtype"), dtype)
    total = len(missing)
    print(f"Latent-Cache: Generiere {total} Einträge in {dataset.latent_cache_dir} ...")

    vae_local = AutoencoderKL.from_pretrained(model_load_path, subfolder="vae", torch_dtype=vae_dtype)
    vae_local.to(device=device, dtype=vae_dtype)
    vae_local.eval()
    cache_scaling_factor = getattr(getattr(vae_local, "config", None), "scaling_factor", 0.18215)

    @torch.no_grad()
    def build_latents(pixel_batch):
        posterior = vae_local.encode(pixel_batch).latent_dist
        latents = posterior.sample() * cache_scaling_factor
        return latents

    bucket_groups = {}
    sample_buckets = getattr(dataset, "sample_buckets", None)
    for idx in missing:
        bucket_key = None
        if sample_buckets is not None and idx < len(sample_buckets):
            bucket_key = sample_buckets[idx]
        bucket_groups.setdefault(bucket_key, []).append(idx)

    progress_bar = tqdm(total=total, desc="Latent Cache", unit="img")
    try:
        for bucket_key, idx_list in bucket_groups.items():
            for start in range(0, len(idx_list), build_batch_size):
                batch_indices = idx_list[start : start + build_batch_size]
                pixel_tensors = [
                    dataset.load_image_tensor_for_cache(idx) for idx in batch_indices
                ]
                pixel_batch = torch.stack(pixel_tensors).to(device=device, dtype=vae_dtype)
                latents_batch = build_latents(pixel_batch)
                latents_batch = latents_batch.to(cache_dtype).cpu()
                for idx, latent_tensor in zip(batch_indices, latents_batch):
                    dataset.save_latent(idx, latent_tensor)
                progress_bar.update(len(batch_indices))
    finally:
        progress_bar.close()

    vae_local.cpu()
    del vae_local
    torch.cuda.empty_cache()

    dataset.refresh_latent_cache_state()
    dataset.activate_latent_cache()
    print("Latent-Cache aktiviert.")


if latent_cache_cfg.get("enabled", False):
    ensure_latent_cache(train_dataset, latent_cache_cfg)

cache_dtype_resolved = _resolve_cache_dtype(latent_cache_cfg.get("dtype"), dtype)
run_summary.append(("latent_cache.enabled", train_dataset.latent_cache_enabled))
if train_dataset.latent_cache_enabled:
    run_summary.append(("latent_cache.dtype", _format_dtype(cache_dtype_resolved)))
    if train_dataset.latent_cache_dir is not None:
        run_summary.append(("latent_cache.dir", str(train_dataset.latent_cache_dir)))


def _move_ema_to_device(ema_model, device):
    if ema_model is None:
        return
    tensor_lists = [
        getattr(ema_model, "shadow_params", None),
        getattr(ema_model, "ema_params", None),
        getattr(ema_model, "reference_params", None),
    ]
    for tensor_list in tensor_lists:
        if tensor_list is None:
            continue
        for tensor in tensor_list:
            if tensor is None:
                continue
            tensor.data = tensor.data.to(device)
    if hasattr(ema_model, "device"):
        ema_model.device = device

# 3) Pipeline laden
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_load_path,
    torch_dtype=dtype,
    use_safetensors=True,
)

unet = pipe.unet
vae = pipe.vae
te1 = pipe.text_encoder
te2 = pipe.text_encoder_2

pipe.to(device)
unet = unet.to(device=device, dtype=dtype)
te1 = te1.to(device=device, dtype=dtype)
te2 = te2.to(device=device, dtype=dtype)
vae = vae.to(device=device, dtype=vae_dtype)
pipe.vae = vae
_wrap_vae_decode_for_dtype(vae, vae_dtype)

run_summary.append(("dtype.te1", _format_dtype(_module_dtype(te1))))
run_summary.append(("dtype.te2", _format_dtype(_module_dtype(te2))))
run_summary.append(("dtype.unet", _format_dtype(_module_dtype(unet))))
run_summary.append(("dtype.vae", _format_dtype(_module_dtype(vae))))

vae_scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
if vae_scaling_factor is not None:
    VAE_SCALING_FACTOR = float(vae_scaling_factor)
print(f"VAE scaling_factor verwendet: {VAE_SCALING_FACTOR}")

if use_torch_compile:
    compile_kwargs = dict(torch_compile_kwargs)
    unet = torch.compile(unet, **compile_kwargs)
    if train_text_encoder_1:
        te1 = torch.compile(te1, **compile_kwargs)
    if train_text_encoder_2:
        te2 = torch.compile(te2, **compile_kwargs)

if use_gradient_checkpointing:
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    if train_text_encoder_1 and hasattr(te1, "gradient_checkpointing_enable"):
        te1.gradient_checkpointing_enable()
    if train_text_encoder_2 and hasattr(te2, "gradient_checkpointing_enable"):
        te2.gradient_checkpointing_enable()

for p in vae.parameters():
    p.requires_grad_(False)

unet.requires_grad_(True)
te1.requires_grad_(train_text_encoder_1)
te2.requires_grad_(train_text_encoder_2)

unet.set_attn_processor(FlashAttnProcessor())

param_total, param_trainable = _count_parameters([unet, vae, te1, te2])
run_summary.append(("parameters.total", f"{param_total:,}"))
run_summary.append(("parameters.trainable", f"{param_trainable:,}"))


# 4) Noise-Scheduler (DDPM oder dein Favorit)
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
if prediction_type_override:
    noise_scheduler.register_to_config(prediction_type=prediction_type_override)
prediction_type = noise_scheduler.config.prediction_type
prediction_source = "override" if prediction_type_override else "model_config"
run_summary.append(("prediction.type", f"{prediction_type} ({prediction_source})"))

print("==== Trainingskonfiguration ====")
for label, value in sorted(run_summary, key=lambda x: x[0]):
    print(f"{label}: {value}")
print("================================")

alphas_cumprod_tensor = noise_scheduler.alphas_cumprod.to(device=device)

sigma_lookup = None
if min_sigma is not None:
    alpha_vals = torch.clamp(alphas_cumprod_tensor.to(torch.float32), min=1e-9, max=1.0).clone()
    sigma_lookup = torch.sqrt(torch.clamp((1 - alpha_vals) / alpha_vals, min=0.0))
    sigma_lookup = torch.nan_to_num(
        sigma_lookup,
        nan=0.0,
        posinf=torch.finfo(sigma_lookup.dtype).max,
        neginf=0.0,
    )
    sigma_lookup = sigma_lookup.to(device=device)

# 5) Optimizer - separate optimizers for UNet (AdamW8bit) and Text Encoders (AdamW)
optimizer_unet = bnb.optim.AdamW8bit(
    [{"params": unet.parameters(), "lr": lr_unet}],
    weight_decay=optimizer_cfg["weight_decay"],
    betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
    eps=optimizer_cfg.get("eps", 1e-8),
)

# Text encoder optimizer (standard AdamW for better precision with small gradients)
optimizer_te = None
te_param_groups = []
if train_text_encoder_1:
    te_param_groups.append({"params": te1.parameters(), "lr": lr_te1})
if train_text_encoder_2:
    te_param_groups.append({"params": te2.parameters(), "lr": lr_te2})

if te_param_groups:
    optimizer_te = torch.optim.AdamW(
        te_param_groups,
        weight_decay=optimizer_cfg["weight_decay"],
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        eps=optimizer_cfg.get("eps", 1e-8),
    )
    print(f"Text Encoder Optimizer: AdamW (FP32 state) mit {len(te_param_groups)} param group(s)")
print(f"UNet Optimizer: AdamW8bit")

# 6) EMA
ema_unet = EMAModel(unet.parameters(), decay=ema_decay) if use_ema else None
if ema_unet is not None:
    _move_ema_to_device(ema_unet, device)
if resume_state_dict is not None:
    # Try new format first (separate optimizers), fall back to legacy format
    opt_unet_state = resume_state_dict.get("optimizer_unet")
    if opt_unet_state is not None:
        try:
            optimizer_unet.load_state_dict(opt_unet_state)
        except Exception as exc:
            warnings.warn(f"UNet Optimizer-Status konnte nicht geladen werden ({exc}).", stacklevel=2)
    else:
        # Legacy format: single "optimizer" key
        opt_state = resume_state_dict.get("optimizer")
        if opt_state is not None:
            warnings.warn(
                "Legacy optimizer state gefunden. Resume mit altem State-Format nicht vollständig unterstützt.",
                stacklevel=2
            )

    opt_te_state = resume_state_dict.get("optimizer_te")
    if opt_te_state is not None and optimizer_te is not None:
        try:
            optimizer_te.load_state_dict(opt_te_state)
        except Exception as exc:
            warnings.warn(f"Text Encoder Optimizer-Status konnte nicht geladen werden ({exc}).", stacklevel=2)

    ema_state = resume_state_dict.get("ema")
    if ema_state is not None and ema_unet is not None:
        try:
            ema_unet.load_state_dict(ema_state)
            _move_ema_to_device(ema_unet, device)
        except Exception as exc:
            warnings.warn(f"EMA-Status konnte nicht geladen werden ({exc}).", stacklevel=2)

lr_scheduler_cfg = training_cfg.get("lr_scheduler") or {}
if isinstance(lr_scheduler_cfg, dict) and lr_scheduler_cfg.get("type"):
    active_scheduler_cfg = lr_scheduler_cfg
else:
    active_scheduler_cfg = None

legacy_warmup_steps = int(training_cfg.get("lr_warmup_steps", 0) or 0)
if active_scheduler_cfg is None and legacy_warmup_steps > 0:
    active_scheduler_cfg = {
        "type": "constant",
        "warmup_steps": legacy_warmup_steps,
        "min_factor": 1.0,
    }

lr_controller = None
if active_scheduler_cfg is not None:
    # UNet optimizer has only one param group (index 0)
    lr_groups = [
        {"name": "unet", "idx": 0, "base_lr": lr_unet, "optimizer": "unet"},
    ]
    # Text encoder optimizer has its own param groups (TE1 at idx 0, TE2 at idx 0 or 1)
    if train_text_encoder_1:
        lr_groups.append({"name": "te1", "idx": 0, "base_lr": lr_te1, "optimizer": "te"})
    if train_text_encoder_2:
        te2_idx = 1 if train_text_encoder_1 else 0
        lr_groups.append({"name": "te2", "idx": te2_idx, "base_lr": lr_te2, "optimizer": "te"})

    total_training_steps = num_steps if num_steps is not None else total_progress_steps
    lr_controller = LearningRateController(active_scheduler_cfg, total_training_steps, lr_groups)
    print(
        f"LR Scheduler aktiv: type={lr_controller.scheduler_type} warmup={lr_controller.warmup_steps} "
        f"min_factor={lr_controller.min_factor}"
    )
    if lr_controller.scheduler_type == "cosine_restarts":
        print(
            f"  cosine_restarts: cycle_steps={lr_controller.cycle_steps} cycle_mult={lr_controller.cycle_mult}"
        )
else:
    print("LR Scheduler: deaktiviert (konstante Basis-LRs).")

eval_runner = None
if eval_cfg and (eval_cfg.get("live", {}).get("enabled") or eval_cfg.get("final", {}).get("enabled")):
    candidate = EvalRunner(
        pipeline=pipe,
        eval_cfg=eval_cfg,
        output_dir=Path(output_dir),
        device=device,
        dtype=dtype,
        ema_unet=ema_unet,
        tb_writer=tb_writer,
        run_name=run_name,
        expected_final_step=num_steps,
    )
    if candidate.has_work():
        eval_runner = candidate
    else:
        print("Eval deaktiviert: keine gültigen Prompts gefunden.")

enforce_min_sigma = MinSigmaController(sigma_lookup, min_sigma, min_sigma_warmup_steps)
if enforce_min_sigma.active:
    print(
        f"Min-Sigma Enforcement aktiv (min_sigma={min_sigma}, warmup_steps={enforce_min_sigma.warmup_steps})"
    )
elif min_sigma is not None:
    print("Min-Sigma-Konfiguration erkannt, aber Enforcement ist deaktiviert (Warmup oder Grenzwert unwirksam).")


# 9) Training Loop

trainer_modules = TrainerModules(
    pipe=pipe,
    unet=unet,
    te1=te1,
    te2=te2,
    optimizer_unet=optimizer_unet,
    optimizer_te=optimizer_te,
    lr_controller=lr_controller,
    ema_unet=ema_unet,
)

trainer_io = TrainerIO(
    train_loader=train_loader,
    encode_text=encode_text,
    prepare_latents=prepare_latents,
)

trainer_settings = TrainingLoopSettings(
    device=device,
    dtype=dtype,
    use_bf16=use_bf16,
    use_ema=use_ema,
    train_text_encoder_1=train_text_encoder_1,
    train_text_encoder_2=train_text_encoder_2,
    grad_accum_steps=grad_accum_steps,
    log_every=log_every,
    checkpoint_every=checkpoint_every,
    noise_offset=noise_offset,
    snr_gamma=snr_gamma,
    max_grad_norm=max_grad_norm,
    detect_anomaly=detect_anomaly,
    ema_update_every=ema_update_every,
    num_steps=num_steps,
    num_epochs=num_epochs,
    total_progress_steps=total_progress_steps,
    data_size=data_cfg["size"],
    tb_log_grad_norm=tb_log_grad_norm,
    tb_log_scaler=tb_log_scaler,
    batch_size=batch_size,
    prediction_type=prediction_type,
    min_timestep=min_timestep,
    max_timestep=max_timestep,
    te_freeze_step=te_freeze_step,
)

trainer_paths = TrainingPaths(
    output_dir=output_dir,
    state_save_path=state_save_path,
)

trainer_resume = ResumeState(
    state_dict=resume_state_dict,
    global_step=resume_global_step,
    epoch=resume_epoch,
)

trainer_bucket_state = BucketState(
    enabled=bucket_enabled,
    default_batch_size=bucket_default_batch_size,
    batch_size_map=bucket_batch_size_map,
    log_switches=bucket_log_switches,
)

trainer_result = run_training_loop(
    modules=trainer_modules,
    io=trainer_io,
    settings=trainer_settings,
    paths=trainer_paths,
    resume_state=trainer_resume,
    alphas_cumprod_tensor=alphas_cumprod_tensor,
    noise_scheduler=noise_scheduler,
    enforce_min_sigma=enforce_min_sigma,
    bucket_state=trainer_bucket_state,
    eval_runner=eval_runner,
    tb_writer=tb_writer,
    bucket_key_fn=bucket_key_from_target_size,
    export_cfg=export_cfg,
)
