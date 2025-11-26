import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor
from transformers import AutoTokenizer

from config_utils import load_config
from dataset import SimpleCaptionDataset


VAE_FALLBACK_SCALING = 0.18215


class AttentionStore:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last = None  # type: Optional[Dict[str, torch.Tensor]]
        self.latent_hw = None  # (height, width) der Latents

    def set_latent_shape(self, height: int, width: int):
        """Setze die tatsächliche Latent-Shape für korrekte Aspect-Ratio."""
        self.latent_hw = (height, width)

    def maybe_store(self, attn_probs, batch_size, heads, seq_len, num_tokens):
        """
        Speichert die Attention-Map.
        
        Args:
            attn_probs: (batch*heads, seq_len, num_tokens) - die Attention-Wahrscheinlichkeiten
            batch_size: Batch-Größe
            heads: Anzahl Attention-Heads
            seq_len: Sequenzlänge (Anzahl Spatial-Positionen)
            num_tokens: Anzahl Text-Tokens
        """
        if attn_probs.ndim != 3:
            return
        
        # Reshape: (batch*heads, seq_len, tokens) -> (batch, heads, seq_len, tokens)
        try:
            attn_probs = attn_probs.view(batch_size, heads, seq_len, num_tokens)
        except Exception:
            return
            
        # Aggregiere über Heads: (batch, seq_len, tokens)
        attn_mean = attn_probs.mean(dim=1)  # (batch, seq_len, tokens)
        
        # Speichere als 1D-Sequenz, Reshape passiert später mit korrekter Aspect-Ratio
        if self.last is None:
            self.last = {"map_1d": attn_mean.detach(), "seq_len": seq_len}
            return
            
        # Speichere die Map mit der höchsten Auflösung
        prev_seq_len = self.last.get("seq_len", 0)
        if seq_len > prev_seq_len:
            self.last = {"map_1d": attn_mean.detach(), "seq_len": seq_len}

    def get_map(self, target_h: int = None, target_w: int = None) -> Optional[torch.Tensor]:
        """
        Gibt die Attention-Map zurück, reshaped zur korrekten Aspect-Ratio.
        
        Args:
            target_h: Ziel-Höhe (optional, sonst aus latent_hw)
            target_w: Ziel-Breite (optional, sonst aus latent_hw)
        """
        if self.last is None:
            return None
            
        attn_1d = self.last["map_1d"]  # (batch, seq_len, tokens)
        seq_len = self.last["seq_len"]
        batch_size, _, num_tokens = attn_1d.shape
        
        # Bestimme die Ziel-Dimensionen
        if target_h is not None and target_w is not None:
            h, w = target_h, target_w
        elif self.latent_hw is not None:
            # Skaliere proportional zur gespeicherten seq_len
            base_h, base_w = self.latent_hw
            # Die Attention-Map ist auf einer bestimmten Auflösungsstufe
            # Finde den Skalierungsfaktor
            scale = math.sqrt(seq_len / (base_h * base_w))
            h = int(round(base_h * scale))
            w = int(round(base_w * scale))
            # Korrigiere falls nötig
            while h * w < seq_len:
                if h <= w:
                    h += 1
                else:
                    w += 1
            while h * w > seq_len and h > 1 and w > 1:
                if h >= w:
                    h -= 1
                else:
                    w -= 1
        else:
            # Fallback: quadratische Approximation
            h = int(math.sqrt(seq_len))
            w = (seq_len + h - 1) // h
        
        target_size = h * w
        
        # Padding oder Truncating
        if seq_len < target_size:
            padding = torch.zeros(
                batch_size, target_size - seq_len, num_tokens,
                device=attn_1d.device, dtype=attn_1d.dtype
            )
            attn_1d = torch.cat([attn_1d, padding], dim=1)
        elif seq_len > target_size:
            attn_1d = attn_1d[:, :target_size, :]
        
        # Reshape zu 2D
        attn_map = attn_1d.view(batch_size, h, w, num_tokens)
        return attn_map



class CrossAttentionCaptureProcessor(AttnProcessor):
    def __init__(self, store: AttentionStore, name: str = ""):
        super().__init__()
        self.store = store
        self.name = name

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
            from diffusers.utils import deprecate
            deprecate(
                "scale",
                "1.0.0",
                "Please remove `scale` arguments. Pass scale via `cross_attention_kwargs` instead.",
            )

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]  # Tatsächliche Sequenzlänge

        # Prüfe BEVOR encoder_hidden_states überschrieben wird
        is_cross_attention = encoder_hidden_states is not None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross_attention:
            # attention_probs shape: (batch*heads, seq_len, num_tokens)
            num_tokens = attention_probs.shape[-1]
            self.store.maybe_store(
                attention_probs,
                batch_size=batch_size,
                heads=attn.heads,
                seq_len=seq_len,
                num_tokens=num_tokens,
            )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CrossAttentionCaptureProcessor2_0(torch.nn.Module):
    def __init__(self, store: AttentionStore, name: str = ""):
        super().__init__()
        self.store = store
        self.name = name

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
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]  # Tatsächliche Sequenzlänge

        # Prüfe BEVOR encoder_hidden_states überschrieben wird
        is_cross_attention = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key.shape[-2], batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if is_cross_attention:
            # Manuell Attention berechnen um die Wahrscheinlichkeiten zu erfassen
            scale = head_dim**-0.5
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            
            # Reshape für maybe_store: (batch*heads, seq_len, tokens)
            num_tokens = attn_probs.shape[-1]
            attn_probs_flat = attn_probs.permute(0, 1, 2, 3).contiguous().view(
                batch_size * attn.heads, seq_len, num_tokens
            )
            
            self.store.maybe_store(
                attn_probs_flat,
                batch_size=batch_size,
                heads=attn.heads,
                seq_len=seq_len,
                num_tokens=num_tokens,
            )
            hidden_states = torch.matmul(attn_probs, value)
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


def _monkeypatch_gradio_api_info():
    try:
        from gradio_client import utils as client_utils
    except Exception:
        return

    original = client_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "bool" if schema else "unknown"
        return original(schema, defs)

    try:
        client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
    except Exception:
        pass


@dataclass
class DemoSample:
    key: str
    caption: str
    target_size: torch.Tensor
    tokens: List[str]
    pixel_values: torch.Tensor
    pixel_preview: np.ndarray
    prompt_embeds: torch.Tensor
    pooled_embeds: torch.Tensor
    add_time_ids: torch.Tensor
    latents: torch.Tensor


@dataclass
class DemoContext:
    device: torch.device
    dtype: torch.dtype
    vae_dtype: torch.dtype
    scaling_factor: float
    model_id: str
    noise_offset: float
    prediction_type: str
    num_train_timesteps: int
    attn_store: "AttentionStore"
    unet: torch.nn.Module
    vae: torch.nn.Module
    scheduler: DDPMScheduler
    samples: Dict[str, DemoSample]


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


def _resolve_device(model_device: str) -> torch.device:
    if model_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(device: torch.device, prefer_bf16: bool) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if prefer_bf16 else torch.float16
    return torch.float32


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu()
    if image.ndim == 4:
        image = image[0]
    image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def _prepare_latents(
    vae: torch.nn.Module,
    pixel_values: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    scaling_factor: float,
) -> torch.Tensor:
    pixel_values = pixel_values.to(device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        posterior = vae.encode(pixel_values).latent_dist
        latents = posterior.sample() * scaling_factor
    return latents.to(dtype=dtype)


def _encode_text(
    te1,
    te2,
    sample: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def _final_ln(model, hidden_states):
        norm = getattr(getattr(model, "text_model", None), "final_layer_norm", None)
        return norm(hidden_states) if norm is not None else hidden_states

    input_ids_1 = sample["input_ids_1"].unsqueeze(0).to(device)
    attn_1 = sample["attention_mask_1"].unsqueeze(0).to(device)
    input_ids_2 = sample["input_ids_2"].unsqueeze(0).to(device)
    attn_2 = sample["attention_mask_2"].unsqueeze(0).to(device)

    with torch.no_grad():
        enc_1 = te1(
            input_ids_1,
            attention_mask=attn_1,
            output_hidden_states=False,
            return_dict=True,
        )
        enc_2 = te2(
            input_ids_2,
            attention_mask=attn_2,
            output_hidden_states=False,
            return_dict=True,
        )

    prompt_embeds_1 = _final_ln(te1, enc_1.last_hidden_state)
    prompt_embeds_2 = _final_ln(te2, enc_2.last_hidden_state)

    if prompt_embeds_1.shape[1] != 77 or prompt_embeds_2.shape[1] != 77:
        raise ValueError(
            f"Unerwartete Sequenzlängen (TE1={prompt_embeds_1.shape[1]}, TE2={prompt_embeds_2.shape[1]}). "
            "Cross-Attention benötigt exakt 77 Token."
        )

    text_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    pooled_embeds = enc_2.text_embeds
    return text_embeds.to(dtype), pooled_embeds.to(dtype)


def _build_time_ids(
    target_size: torch.Tensor,
    pooled_embeds: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    height = target_size[0].unsqueeze(0).to(device=device, dtype=pooled_embeds.dtype)
    width = target_size[1].unsqueeze(0).to(device=device, dtype=pooled_embeds.dtype)
    zeros = torch.zeros_like(width)
    return torch.stack([width, height, zeros, zeros, width, height], dim=1)


def _decode_latents(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    scaling_factor: float,
    vae_dtype: torch.dtype,
    device: torch.device,
) -> np.ndarray:
    latents = latents.to(device=device, dtype=vae_dtype) / scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample
    return _tensor_to_image(image)


def _render_attention_map(attn_map: Optional[torch.Tensor], token_idx: int, target_size: int = 512) -> np.ndarray:
    """
    Rendert eine Attention-Map als farbige Heatmap mit korrekter Aspect-Ratio.
    
    Args:
        attn_map: Attention-Map Tensor (batch, height, width, tokens)
        token_idx: Index des Tokens für das die Map angezeigt werden soll
        target_size: Maximale Größe der längeren Seite
    """
    if attn_map is None or attn_map.numel() == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    token_idx = max(0, min(token_idx, attn_map.shape[-1] - 1))
    heat = attn_map[0, :, :, token_idx].detach().cpu().float()  # (H, W)
    heat = torch.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Original Aspect-Ratio
    orig_h, orig_w = heat.shape
    
    # Berechne Zielgröße mit beibehaltener Aspect-Ratio
    if orig_h >= orig_w:
        new_h = target_size
        new_w = int(round(target_size * orig_w / orig_h))
    else:
        new_w = target_size
        new_h = int(round(target_size * orig_h / orig_w))
    
    # Normalisierung
    heat_min, heat_max = heat.min(), heat.max()
    if heat_max > heat_min:
        heat = (heat - heat_min) / (heat_max - heat_min)
    else:
        heat = torch.zeros_like(heat)

    heat = heat.clamp(0, 1).unsqueeze(0).unsqueeze(0)
    heat = torch.nn.functional.interpolate(
        heat,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    heat_np = heat.numpy()
    
    # Jet-ähnliche Colormap: blau -> cyan -> grün -> gelb -> rot
    r = np.clip(1.5 - 4 * np.abs(heat_np - 0.75), 0, 1)
    g = np.clip(1.5 - 4 * np.abs(heat_np - 0.5), 0, 1)
    b = np.clip(1.5 - 4 * np.abs(heat_np - 0.25), 0, 1)

    heat_color = np.stack([r, g, b], axis=-1)
    heat_color = (heat_color * 255).astype(np.uint8)
    return heat_color


def _prepare_samples(
    dataset: SimpleCaptionDataset,
    te1,
    te2,
    tokenizer_2,
    vae,
    device: torch.device,
    dtype: torch.dtype,
    vae_dtype: torch.dtype,
    scaling_factor: float,
    max_samples: int,
    sample_seed: Optional[int] = None,
) -> Dict[str, DemoSample]:
    samples: Dict[str, DemoSample] = {}
    total = min(len(dataset), max_samples)
    rng = random.Random(sample_seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    chosen_indices = indices[:total]

    for idx in chosen_indices:
        rng_state_before = random.getstate()
        raw = dataset[idx]
        rng_state_after = random.getstate()
        random.setstate(rng_state_before)
        caption_text = dataset._apply_caption_transforms(dataset._load_caption(dataset.files[idx]))
        random.setstate(rng_state_after)
        if "pixel_values" not in raw:
            raise ValueError(
                "Das Demo benötigt Pixelwerte. Bitte latent_cache.enabled auf false setzen."
            )
        prompt_embeds, pooled_embeds = _encode_text(te1, te2, raw, device, dtype)
        add_time_ids = _build_time_ids(raw["target_size"], pooled_embeds, device)
        latents = _prepare_latents(
            vae=vae,
            pixel_values=raw["pixel_values"],
            device=device,
            dtype=dtype,
            scaling_factor=scaling_factor,
        )
        pixel_preview = _tensor_to_image(raw["pixel_values"])
        key = f"{idx}: {dataset.files[idx].name}"
        tokens = tokenizer_2.convert_ids_to_tokens(raw["input_ids_2"].tolist())
        samples[key] = DemoSample(
            key=key,
            caption=caption_text,
            target_size=raw["target_size"],
            tokens=tokens,
            pixel_values=raw["pixel_values"],
            pixel_preview=pixel_preview,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            add_time_ids=add_time_ids,
            latents=latents,
        )
    if not samples:
        raise ValueError("Keine Samples im Dataset gefunden.")
    return samples


def load_demo_context(
    config_path: Path,
    max_samples: int = 4,
    sample_seed: Optional[int] = None,
) -> DemoContext:
    def _detect_sample_seed():
        if sample_seed is not None:
            return sample_seed
        env_seed = os.environ.get("SDXL_DEMO_SAMPLE_SEED")
        if env_seed:
            try:
                return int(env_seed)
            except (TypeError, ValueError):
                return None
        return None

    config = load_config(config_path)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    effective_sample_seed = _detect_sample_seed()

    device = _resolve_device(config.get("device", "cuda"))
    dtype = _resolve_dtype(device, model_cfg.get("use_bf16", False))
    vae_dtype = torch.float32
    noise_offset = float(training_cfg.get("noise_offset") or 0.0)

    model_path = model_cfg.get("id")
    if not model_path:
        raise ValueError("Config-Parameter model.id fehlt.")
    model_path = str(model_path)

    tokenizer_1 = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", use_fast=False)

    dataset = SimpleCaptionDataset(
        img_dir=data_cfg.get("image_dir", "./data/images"),
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        size=data_cfg.get("size", 1024),
        caption_dropout_prob=data_cfg.get("caption_dropout_prob", 0.0),
        caption_shuffle_prob=data_cfg.get("caption_shuffle_prob", 0.0),
        caption_shuffle_separator=data_cfg.get("caption_shuffle_separator", ","),
        caption_shuffle_min_tokens=data_cfg.get("caption_shuffle_min_tokens", 2),
        bucket_config=data_cfg.get("bucket", {}),
        latent_cache_config=data_cfg.get("latent_cache", {}),
        pixel_dtype=vae_dtype,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)
    unet = pipe.unet.to(device=device, dtype=dtype)
    vae = pipe.vae.to(device=device, dtype=vae_dtype)
    te1 = pipe.text_encoder.to(device=device, dtype=dtype)
    te2 = pipe.text_encoder_2.to(device=device, dtype=dtype)
    for module in (unet, vae, te1, te2):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    _wrap_vae_decode_for_dtype(vae, vae_dtype)

    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pred_override = training_cfg.get("prediction_type")
    if pred_override:
        scheduler.register_to_config(prediction_type=pred_override)
    prediction_type = scheduler.config.prediction_type

    scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", VAE_FALLBACK_SCALING))

    attn_store = AttentionStore()

    # Ersetze Cross-Attention Prozessoren
    attn_procs = {}
    replaced_count = 0
    for name, proc in unet.attn_processors.items():
        if "attn2" in name:
            proc_class = proc.__class__.__name__
            if "2_0" in proc_class or "2.0" in proc_class or proc_class == "AttnProcessor2_0":
                attn_procs[name] = CrossAttentionCaptureProcessor2_0(attn_store, name=name)
            else:
                attn_procs[name] = CrossAttentionCaptureProcessor(attn_store, name=name)
            replaced_count += 1
        else:
            attn_procs[name] = proc
    
    print(f"[INFO] Ersetzte {replaced_count} Cross-Attention Prozessoren")
    unet.set_attn_processor(attn_procs)
    
    samples = _prepare_samples(
        dataset=dataset,
        te1=te1,
        te2=te2,
        tokenizer_2=tokenizer_2,
        vae=vae,
        device=device,
        dtype=dtype,
        vae_dtype=vae_dtype,
        scaling_factor=scaling_factor,
        max_samples=max_samples,
        sample_seed=effective_sample_seed,
    )

    return DemoContext(
        device=device,
        dtype=dtype,
        vae_dtype=vae_dtype,
        scaling_factor=scaling_factor,
        model_id=model_path,
        attn_store=attn_store,
        noise_offset=noise_offset,
        prediction_type=prediction_type,
        num_train_timesteps=scheduler.config.num_train_timesteps,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        samples=samples,
    )


def run_demo_step(context: DemoContext, sample_key: str, timestep: int):
    sample = context.samples[sample_key]
    step_tensor = torch.tensor([int(timestep)], device=context.device, dtype=torch.long)
    base_latents = sample.latents

    noise = torch.randn_like(base_latents)
    if context.noise_offset > 0:
        noise = noise + context.noise_offset * torch.randn(
            (noise.shape[0], noise.shape[1], 1, 1),
            device=context.device,
            dtype=noise.dtype,
        )
    noisy_latents = context.scheduler.add_noise(base_latents, noise, step_tensor)

    context.attn_store.reset()
    # Setze die Latent-Dimensionen für korrekte Aspect-Ratio
    latent_h, latent_w = base_latents.shape[2], base_latents.shape[3]
    context.attn_store.set_latent_shape(latent_h, latent_w)
    
    with torch.no_grad():
        model_pred = context.unet(
            noisy_latents,
            step_tensor,
            encoder_hidden_states=sample.prompt_embeds,
            added_cond_kwargs={"text_embeds": sample.pooled_embeds, "time_ids": sample.add_time_ids},
        ).sample
        pred_original = context.scheduler.step(
            model_pred,
            step_tensor,
            noisy_latents,
            return_dict=True,
        ).pred_original_sample

    clean_img = sample.pixel_preview
    noisy_img = _decode_latents(
        context.vae,
        noisy_latents,
        context.scaling_factor,
        context.vae_dtype,
        context.device,
    )
    recon_img = _decode_latents(
        context.vae,
        pred_original,
        context.scaling_factor,
        context.vae_dtype,
        context.device,
    )
    attn_map = context.attn_store.get_map()
    
    # Warnung nur bei komplettem Fehlen
    if attn_map is None:
        print("Warnung: Keine Cross-Attention Map erfasst.", flush=True)
        
    return clean_img, noisy_img, recon_img, sample.caption, attn_map


def build_ui(context: DemoContext):
    sample_keys: List[str] = list(context.samples.keys())
    default_key = sample_keys[0]
    max_timestep = context.num_train_timesteps - 1

    with gr.Blocks(
        title="SDXL Training Forward Demo",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="amber"),
        css=".preview-img img {object-fit: contain;}",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "### SDXL Trainings-Forward-Pass\n"
            "Links: was der VAE im Training gesehen hat (nach Resize/Norm). "
            "Mitte: `x_t` nach Noise + Scheduler. Rechts: `x0_pred`, also die UNet-Schätzung des sauberen Bildes."
        )
        token_choices = [f"{i}: {tok}" for i, tok in enumerate(context.samples[default_key].tokens)]
        default_token = token_choices[0] if token_choices else None

        with gr.Row():
            sample_dropdown = gr.Dropdown(
                choices=sample_keys,
                value=default_key,
                label="Sample auswählen",
            )
            timestep_slider = gr.Slider(
                minimum=0,
                maximum=max_timestep,
                step=1,
                value=max_timestep // 2,
                label=f"Noise-Timestep t (0–{max_timestep})",
            )
            token_dropdown = gr.Dropdown(
                choices=token_choices,
                value=default_token,
                label="Token für Attention-Map",
            )
            run_button = gr.Button("Demo ausführen", variant="primary")

        gr.Markdown(
            f"- Modell: `{context.model_id}`\n"
            f"- Device: `{context.device}` | dtype: `{context.dtype}` (VAE: `{context.vae_dtype}`)\n"
            f"- Scheduler: `{context.scheduler.__class__.__name__}` mit `{context.num_train_timesteps}` Timesteps, "
            f"prediction_type: `{context.prediction_type}`\n"
            f"- VAE scaling_factor: {context.scaling_factor} | noise_offset: {context.noise_offset}"
        )

        with gr.Row():
            original_out = gr.Image(label="Original (train input)", type="numpy", elem_classes=["preview-img"])
            noisy_out = gr.Image(label="x_t (noisy)", type="numpy", elem_classes=["preview-img"])
            recon_out = gr.Image(label="x0_pred (UNet)", type="numpy", elem_classes=["preview-img"])
            attn_out = gr.Image(label="Cross-Attention (Token)", type="numpy", elem_classes=["preview-img"])
        caption_box = gr.Textbox(label="Caption (nach Dataset-Transforms)", interactive=False)

        def _update_tokens(selected_key):
            tokens = context.samples[selected_key].tokens
            choices = [f"{i}: {tok}" for i, tok in enumerate(tokens)]
            value = choices[0] if choices else None
            return gr.update(choices=choices, value=value)

        def _run(selected_key, t_value, token_choice):
            token_idx = 0
            if isinstance(token_choice, str) and ":" in token_choice:
                try:
                    token_idx = int(token_choice.split(":")[0].strip())
                except Exception:
                    token_idx = 0
            clean_img, noisy_img, recon_img, caption, attn_map = run_demo_step(
                context, selected_key, int(t_value)
            )
            attn_img = _render_attention_map(attn_map, token_idx)
            return clean_img, noisy_img, recon_img, attn_img, caption

        run_button.click(
            fn=_run,
            inputs=[sample_dropdown, timestep_slider, token_dropdown],
            outputs=[original_out, noisy_out, recon_out, attn_out, caption_box],
        )
        timestep_slider.release(
            fn=_run,
            inputs=[sample_dropdown, timestep_slider, token_dropdown],
            outputs=[original_out, noisy_out, recon_out, attn_out, caption_box],
        )
        sample_dropdown.change(
            fn=_update_tokens,
            inputs=[sample_dropdown],
            outputs=[token_dropdown],
        )
        sample_dropdown.change(
            fn=_run,
            inputs=[sample_dropdown, timestep_slider, token_dropdown],
            outputs=[original_out, noisy_out, recon_out, attn_out, caption_box],
        )
        token_dropdown.change(
            fn=_run,
            inputs=[sample_dropdown, timestep_slider, token_dropdown],
            outputs=[original_out, noisy_out, recon_out, attn_out, caption_box],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDXL Training Forward Demo mit Gradio UI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Pfad zur Config-Datei (gleiche Struktur wie beim Training).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Anzahl der Beispiel-Samples, die geladen werden (Default: 4).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Optionaler Seed für die zufällige Auswahl der Demo-Samples.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Erzeugt einen öffentlichen Share-Link.",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Hostname/Interface für Gradio (Default: 0.0.0.0).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port für Gradio (Default: 7860).",
    )
    parser.add_argument(
        "--show-api",
        action="store_true",
        help="Zeigt die OpenAPI/Schema-Ansicht.",
    )
    return parser.parse_args()


def main():
    _monkeypatch_gradio_api_info()
    args = parse_args()
    context = load_demo_context(
        args.config,
        max_samples=max(1, args.samples),
        sample_seed=args.sample_seed,
    )
    demo = build_ui(context)
    launch_kwargs = dict(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_api=args.show_api,
    )
    try:
        demo.launch(**launch_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "localhost is not accessible" in msg and not args.share:
            print("Warnung: localhost nicht erreichbar, starte erneut mit share=True ...")
            launch_kwargs["share"] = True
            demo.launch(**launch_kwargs)
        else:
            raise


if __name__ == "__main__":
    main()