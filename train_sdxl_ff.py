import warnings
import json
import copy
import math
import subprocess
import sys
from pathlib import Path

import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from diffusers.training_utils import EMAModel
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from dataset import SimpleCaptionDataset


CONFIG_PATH = Path("config.json")

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
        "use_bf16": True,
        "use_gradient_checkpointing": True,
        "train_text_encoders": True,
    },
    "training": {
        "output_dir": None,
        "batch_size": 4,
        "num_steps": 10_000,
        "num_epochs": None,
        "lr_unet": 5e-6,
        "lr_text_encoder": 1e-6,
        "log_every": 50,
        "checkpoint_every": 1_000,
        "grad_accum_steps": 1,
        "noise_offset": 0.1,
        "min_sigma": 0.4,
        "min_sigma_warmup_steps": 20,
        "prediction_type": "v_prediction",
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
        with path.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
        config = _deep_update(config, user_config)
    except json.JSONDecodeError as err:
        warnings.warn(f"Konfiguration konnte nicht gelesen werden ({err}). Fallback auf Defaults.", stacklevel=2)
    return config


cfg = load_config(CONFIG_PATH)

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


class FlashAttnProcessor:
    """Wrapper, der automatisch auf FlashAttention-3 fällt, falls verfügbar."""

    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(self):
        self._warned = False

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
                flash_hidden_states = flash_attn_func(
                    flash_query,
                    flash_key,
                    flash_value,
                    dropout_p=dropout_p,
                    softmax_scale=attn_scale,
                    causal=False,
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

model_id = model_cfg["id"]
output_dir_cfg = training_cfg.get("output_dir")
if output_dir_cfg:
    output_dir = str(output_dir_cfg)
else:
    if not run_name:
        raise ValueError("Setze entweder `training.output_dir` oder `run.name` in der config.")
    output_dir = str(output_root / run_name)
    training_cfg["output_dir"] = output_dir
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
lr_te = training_cfg["lr_text_encoder"]
use_ema = model_cfg["use_ema"]
use_bf16 = model_cfg["use_bf16"]
use_gradient_checkpointing = model_cfg["use_gradient_checkpointing"]
train_text_encoders = bool(model_cfg.get("train_text_encoders", True))
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
min_sigma_warmup_steps = max(0, int(training_cfg.get("min_sigma_warmup_steps", 0)))
prediction_type_override = training_cfg.get("prediction_type")

checkpoint_path_cfg = export_cfg.get("checkpoint_path")
if checkpoint_path_cfg:
    checkpoint_path = str(checkpoint_path_cfg)
else:
    if not run_name:
        raise ValueError("Setze entweder `export.checkpoint_path` oder `run.name` in der config.")
    checkpoint_path = str(checkpoint_root / f"{run_name}.safetensors")
    export_cfg["checkpoint_path"] = checkpoint_path

dtype = torch.bfloat16 if use_bf16 else torch.float16

# 1) Pipeline laden
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    use_safetensors=True,
    #variant="fp16",
)
pipe.to(device)

unet = pipe.unet
vae = pipe.vae
te1 = pipe.text_encoder
te2 = pipe.text_encoder_2

if use_gradient_checkpointing:
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    if train_text_encoders and hasattr(te1, "gradient_checkpointing_enable"):
        te1.gradient_checkpointing_enable()
    if train_text_encoders and hasattr(te2, "gradient_checkpointing_enable"):
        te2.gradient_checkpointing_enable()

# Nur VAE einfrieren (typisch für FF)
for p in vae.parameters():
    p.requires_grad_(False)

unet.requires_grad_(True)
te1.requires_grad_(train_text_encoders)
te2.requires_grad_(train_text_encoders)

# 2) FlashAttention-Processor (fällt sonst auf SDPA zurück)
unet.set_attn_processor(FlashAttnProcessor())

# 3) Tokenizer laden
tokenizer_1 = AutoTokenizer.from_pretrained(
    pipe.tokenizer.name_or_path, use_fast=False
)
tokenizer_2 = AutoTokenizer.from_pretrained(
    pipe.tokenizer_2.name_or_path, use_fast=False
)

# 4) Dataset + Dataloader
caption_dropout_prob = data_cfg.get("caption_dropout_prob", 0.0)
caption_shuffle_prob = data_cfg.get("caption_shuffle_prob", 0.0)
caption_shuffle_separator = data_cfg.get("caption_shuffle_separator", ",")
caption_shuffle_min_tokens = data_cfg.get("caption_shuffle_min_tokens", 2)

train_dataset = SimpleCaptionDataset(
    img_dir=data_cfg["image_dir"],
    tokenizer_1=tokenizer_1,
    tokenizer_2=tokenizer_2,
    size=data_cfg["size"],
    caption_dropout_prob=caption_dropout_prob,
    caption_shuffle_prob=caption_shuffle_prob,
    caption_shuffle_separator=caption_shuffle_separator,
    caption_shuffle_min_tokens=caption_shuffle_min_tokens,
)

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

# 5) Noise-Scheduler (DDPM oder dein Favorit)
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
if prediction_type_override:
    noise_scheduler.register_to_config(prediction_type=prediction_type_override)
prediction_type = noise_scheduler.config.prediction_type

sigma_lookup = None
if min_sigma is not None and min_sigma > 0:
    alpha_cumprod = noise_scheduler.alphas_cumprod.clone().detach().to(device=device, dtype=torch.float32)
    alpha_cumprod = torch.clamp(alpha_cumprod, min=1e-9)
    sigma_lookup = torch.sqrt(torch.clamp((1 - alpha_cumprod) / alpha_cumprod, min=0.0))
    sigma_lookup = torch.nan_to_num(
        sigma_lookup,
        nan=0.0,
        posinf=torch.finfo(sigma_lookup.dtype).max,
        neginf=0.0,
    )

# 6) Optimizer mit Param-Groups
param_groups = [
    {"params": unet.parameters(), "lr": lr_unet},
]
if train_text_encoders:
    param_groups.extend(
        [
            {"params": te1.parameters(), "lr": lr_te},
            {"params": te2.parameters(), "lr": lr_te},
        ]
    )

optimizer = bnb.optim.AdamW8bit(
    param_groups,
    weight_decay=optimizer_cfg["weight_decay"],
    betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
    eps=optimizer_cfg.get("eps", 1e-8),
)

# 7) EMA
ema_unet = EMAModel(unet.parameters()) if use_ema else None

# 8) Hilfsfunktionen

def encode_text(captions_batch):
    # Wir haben schon input_ids aus dem Dataset, daher hier just forward:
    input_ids_1 = captions_batch["input_ids_1"].to(device)
    attn_1 = captions_batch["attention_mask_1"].to(device)
    input_ids_2 = captions_batch["input_ids_2"].to(device)
    attn_2 = captions_batch["attention_mask_2"].to(device)

    with torch.no_grad():
        enc_1 = te1(
            input_ids_1,
            attention_mask=attn_1,
            output_hidden_states=True,
            return_dict=True,
        )
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

    # SDXL erwartet float32 für Text-Embeddings während BF16/FP16 Training
    return text_embeds.to(dtype), pooled_embeds.to(dtype)


def prepare_latents(pixel_values, generator=None):
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    # VAE encode
    with torch.no_grad():
        posterior = vae.encode(pixel_values).latent_dist
        latents = posterior.sample(generator=generator) * 0.18215
    return latents


def build_min_sigma_enforcer(sigma_lookup_tensor, min_sigma_value, warmup_steps):
    if sigma_lookup_tensor is None or min_sigma_value is None or min_sigma_value <= 0:
        return lambda timesteps, _: timesteps

    def _current_threshold(step_idx: int) -> float:
        if warmup_steps <= 0:
            return min_sigma_value
        progress = min(max(step_idx, 0) / float(warmup_steps), 1.0)
        return max(0.0, min_sigma_value * (1.0 - progress))

    def _enforce(timesteps: torch.LongTensor, step_idx: int) -> torch.LongTensor:
        threshold = _current_threshold(step_idx)
        if threshold <= 0.0:
            return timesteps
        target = torch.tensor(threshold, device=sigma_lookup_tensor.device, dtype=sigma_lookup_tensor.dtype)
        idx = torch.searchsorted(sigma_lookup_tensor, target, right=False)
        idx = torch.clamp(idx, max=sigma_lookup_tensor.shape[0] - 1)
        idx_value = int(idx.item())
        if idx_value <= 0:
            return timesteps
        limit = torch.full_like(timesteps, idx_value)
        return torch.maximum(timesteps, limit)

    return _enforce


enforce_min_sigma = build_min_sigma_enforcer(sigma_lookup, min_sigma, min_sigma_warmup_steps)


# 9) Training Loop

scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

global_step = 0
unet.train()
if train_text_encoders:
    te1.train()
    te2.train()
else:
    te1.eval()
    te2.eval()

optimizer.zero_grad(set_to_none=True)

pbar = tqdm(total=total_progress_steps, desc="SDXL Training", unit="step")
accum_counter = 0
last_loss_value = 0.0

def optimizer_step_fn(loss_value, current_step, current_accum):
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    current_accum = 0

    if ema_unet is not None:
        ema_unet.step(unet.parameters())

    current_step += 1
    display_loss = float(loss_value) if loss_value is not None else 0.0
    pbar.update(1)
    pbar.set_postfix({"loss": f"{display_loss:.4f}"})

    if log_every is not None and current_step % log_every == 0:
        print(f"step {current_step} | loss {display_loss:.4f}")

    if checkpoint_every is not None and current_step % checkpoint_every == 0:
        if ema_unet is not None:
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())

        pipe.save_pretrained(f"{output_dir}_step_{current_step}")

        if ema_unet is not None:
            ema_unet.restore(unet.parameters())

    return current_step, current_accum

epoch = 0
while True:
    if num_epochs is not None and epoch >= num_epochs:
        break

    for batch in train_loader:
        if num_steps is not None and global_step >= num_steps:
            break

        with torch.amp.autocast("cuda", dtype=dtype, enabled=not use_bf16):
            # Images -> Latents
            pixel_values = batch["pixel_values"]
            latents = prepare_latents(pixel_values)

            # Noise + Timesteps
            noise = torch.randn_like(latents)
            if noise_offset > 0:
                noise = noise + noise_offset * torch.randn(
                    (noise.shape[0], noise.shape[1], 1, 1),
                    device=device,
                    dtype=noise.dtype,
                )
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            timesteps = enforce_min_sigma(timesteps, global_step)

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text-Encodings
            prompt_embeds, pooled_embeds = encode_text(batch)

            # UNet-Forward
            add_time_ids = torch.tensor(
                [
                    data_cfg["size"],
                    data_cfg["size"],
                    0,
                    0,
                    data_cfg["size"],
                    data_cfg["size"],
                ],
                device=device,
                dtype=pooled_embeds.dtype,
            ).repeat(latents.shape[0], 1)

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
            ).sample

            # Standard MSE in noise-space
            if prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            elif prediction_type == "sample":
                target = latents
            else:
                target = noise
            raw_loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
            loss = raw_loss / grad_accum_steps

        last_loss_value = raw_loss.item()
        scaler.scale(loss).backward()
        accum_counter += 1

        if accum_counter % grad_accum_steps == 0:
            global_step, accum_counter = optimizer_step_fn(last_loss_value, global_step, accum_counter)

            if num_steps is not None and global_step >= num_steps:
                break

    if num_steps is not None and global_step >= num_steps:
        break

    if accum_counter > 0 and (num_steps is None or global_step < num_steps):
        global_step, accum_counter = optimizer_step_fn(last_loss_value, global_step, accum_counter)

        if num_steps is not None and global_step >= num_steps:
            break

    epoch += 1

pbar.close()

# Final Save (mit EMA-Gewichten, falls vorhanden)
if ema_unet is not None:
    ema_unet.store(unet.parameters())
    ema_unet.copy_to(unet.parameters())

pipe.save_pretrained(output_dir)


def _run_converter_script(script_path: Path, model_dir: Path, checkpoint_path: Path, cfg: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--model_path",
        str(model_dir),
        "--checkpoint_path",
        str(checkpoint_path),
    ]

    if cfg.get("use_safetensors", True):
        cmd.append("--use_safetensors")
    if cfg.get("half_precision", True):
        cmd.append("--half")

    extra_args = cfg.get("extra_args") or []
    if not isinstance(extra_args, (list, tuple)):
        raise TypeError("`export.extra_args` muss eine Liste von zusätzlichen Argumenten sein.")
    cmd.extend(map(str, extra_args))

    print("Starte Diffusers-Konverter:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Der Diffusers-Konverter konnte nicht erfolgreich ausgeführt werden (Exit-Code {exc.returncode})."
        ) from exc


def maybe_export_single_file(model_dir: Path, cfg: dict) -> None:
    if not cfg.get("save_single_file", False):
        print("Single-File-Export deaktiviert.")
        return

    checkpoint_path = Path(cfg.get("checkpoint_path", f"{model_dir}.safetensors")).resolve()
    converter_path = cfg.get("converter_script")
    if not converter_path:
        raise ValueError("`export.converter_script` ist nicht gesetzt. Bitte Pfad zum Konverter angeben.")
    converter_path = Path(converter_path).expanduser().resolve()
    if not converter_path.exists():
        raise FileNotFoundError(f"Konverter-Skript {converter_path} existiert nicht.")

    _run_converter_script(converter_path, Path(model_dir).resolve(), checkpoint_path, cfg)
    print(f"Single-File-Checkpoint gespeichert unter: {checkpoint_path}")


maybe_export_single_file(Path(output_dir), export_cfg)
