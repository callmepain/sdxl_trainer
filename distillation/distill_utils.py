"""
Utilities for multi-teacher distillation.

Key functions:
- compute_deterministic_seed(image_id, teacher_id=None, base_seed=42) -> int
- compute_cache_path(cache_dir, image_id, teacher_id, resolution) -> Path
- load_teacher_unet_diffusers(checkpoint_path, device, dtype) -> UNet2DConditionModel
- save_teacher_cache_entry() / load_teacher_cache_entry()
- generate_deterministic_timestep(seed, min_t, max_t) -> int
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file, save_file


def compute_deterministic_seed(
    image_id: str,
    teacher_id: Optional[str] = None,
    base_seed: int = 42,
) -> int:
    """
    Generate a deterministic seed for reproducible noise generation.

    Uses SHA-256 hash of (image_id, base_seed) truncated to 32 bits.
    If teacher_id is provided, include it to create per-teacher noise.

    Args:
        image_id: Unique identifier for the image (typically relative path without extension)
        teacher_id: Optional identifier to make noise per-teacher
        base_seed: Base seed for additional randomization

    Returns:
        32-bit integer seed
    """
    if teacher_id is None:
        payload = f"{image_id}|{base_seed}"
    else:
        payload = f"{image_id}|{teacher_id}|{base_seed}"
    hash_bytes = hashlib.sha256(payload.encode('utf-8')).digest()
    # Use first 4 bytes as seed (32-bit integer)
    seed = int.from_bytes(hash_bytes[:4], byteorder='big')
    return seed


def compute_cache_path(
    cache_dir: Path,
    image_id: str,
    teacher_id: str,
    resolution: Tuple[int, int],
    cache_version: str = "1"
) -> Path:
    """
    Compute cache file path for a teacher prediction.

    Format: {cache_dir}/{teacher_id}/{hash}_{WxH}.safetensors

    Hash is based on image_id, resolution, and cache version for invalidation.

    Args:
        cache_dir: Base cache directory
        image_id: Unique identifier for the image
        teacher_id: Unique identifier for the teacher model
        resolution: (width, height) tuple
        cache_version: Version string for cache invalidation

    Returns:
        Path to the cache file
    """
    w, h = resolution
    payload = f"{image_id}|{w}x{h}|{cache_version}"
    key = hashlib.sha1(payload.encode('utf-8')).hexdigest()
    filename = f"{key}_{w}x{h}.safetensors"
    return cache_dir / teacher_id / filename


def load_teacher_unet_diffusers(
    checkpoint_path: Path,
    device: str,
    dtype: torch.dtype
):
    """
    Load UNet from a diffusers checkpoint directory.

    Args:
        checkpoint_path: Path to diffusers checkpoint (e.g., .output/my_run/)
        device: Target device
        dtype: Target dtype (torch.float16 or torch.bfloat16)

    Returns:
        Loaded UNet2DConditionModel in eval mode with gradients disabled
    """
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        checkpoint_path,
        subfolder="unet",
        torch_dtype=dtype,
    )
    unet = unet.to(device=device, dtype=dtype)
    unet.eval()
    unet.requires_grad_(False)
    return unet


def generate_deterministic_timestep(
    seed: int,
    min_timestep: int = 0,
    max_timestep: int = 1000,
) -> int:
    """
    Generate a deterministic timestep from a seed.

    Uses the seed to create a reproducible random value.

    Args:
        seed: Random seed
        min_timestep: Minimum timestep (inclusive)
        max_timestep: Maximum timestep (exclusive)

    Returns:
        Single timestep value
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    timestep = torch.randint(
        min_timestep, max_timestep, (1,), generator=rng
    ).item()
    return timestep


def generate_deterministic_noise(
    seed: int,
    shape: Tuple[int, ...],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generate deterministic noise tensor from a seed.

    Args:
        seed: Random seed
        shape: Shape of the noise tensor
        device: Target device
        dtype: Target dtype

    Returns:
        Noise tensor
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return noise


def save_teacher_cache_entry(
    path: Path,
    teacher_pred: torch.Tensor,
    noise: torch.Tensor,
    timestep: int,
    seed: int,
    resolution: Tuple[int, int],
    encoder_hidden_states: torch.Tensor,
    pooled_embeds: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> None:
    """
    Save a single teacher cache entry to disk.

    All tensors are saved in the specified dtype (default FP16) to save space.

    Args:
        path: Output file path
        teacher_pred: UNet prediction tensor [C, H, W]
        noise: Noise tensor [C, H, W]
        timestep: Timestep value
        seed: Seed used for generation
        resolution: (width, height) tuple
        encoder_hidden_states: Text embeddings [seq_len, dim]
        pooled_embeds: Pooled text embeddings [dim]
        dtype: Storage dtype
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to target dtype for storage
    data = {
        "teacher_pred": teacher_pred.to(dtype).cpu().contiguous(),
        "noise": noise.to(dtype).cpu().contiguous(),
        "timestep": torch.tensor([timestep], dtype=torch.int64),
        "seed": torch.tensor([seed], dtype=torch.int64),
        "resolution": torch.tensor(list(resolution), dtype=torch.int32),
        "encoder_hidden_states": encoder_hidden_states.to(dtype).cpu().contiguous(),
        "pooled_embeds": pooled_embeds.to(dtype).cpu().contiguous(),
    }

    save_file(data, str(path))


def load_teacher_cache_entry(path: Path, device: str = "cpu") -> dict:
    """
    Load a teacher cache entry from disk.

    Args:
        path: Path to cache file
        device: Device to load tensors to

    Returns:
        Dict with all cached tensors and metadata
    """
    data = load_file(str(path), device=device)
    return {
        "teacher_pred": data["teacher_pred"],
        "noise": data["noise"],
        "timestep": int(data["timestep"].item()),
        "seed": int(data["seed"].item()),
        "resolution": tuple(data["resolution"].tolist()),
        "encoder_hidden_states": data["encoder_hidden_states"],
        "pooled_embeds": data["pooled_embeds"],
    }


def validate_cache_entry(data: dict) -> Tuple[bool, str]:
    """
    Validate a loaded cache entry.

    Args:
        data: Dict from load_teacher_cache_entry()

    Returns:
        (is_valid, error_message)
    """
    required_keys = [
        "teacher_pred", "noise", "timestep", "seed",
        "resolution", "encoder_hidden_states", "pooled_embeds"
    ]

    for key in required_keys:
        if key not in data:
            return False, f"Missing key: {key}"

    # Check for NaN/Inf in tensor values
    tensor_keys = ["teacher_pred", "noise", "encoder_hidden_states", "pooled_embeds"]
    for key in tensor_keys:
        tensor = data[key]
        if torch.isnan(tensor).any():
            return False, f"NaN values in {key}"
        if torch.isinf(tensor).any():
            return False, f"Inf values in {key}"

    # Check teacher_pred shape (should be [4, H, W])
    teacher_pred = data["teacher_pred"]
    if teacher_pred.ndim != 3 or teacher_pred.shape[0] != 4:
        return False, f"Invalid teacher_pred shape: {teacher_pred.shape}"

    return True, ""
