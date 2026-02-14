#!/usr/bin/env python3
"""
Standalone latent cache builder for distillation.

Builds VAE latent caches without loading UNet, text encoders, or optimizer.
Reuses the same cache builder configs as build_teacher_cache.py.

Usage:
    python distillation/build_latent_cache.py --config distillation/configs/cache_v8.json
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import SimpleCaptionDataset
from distillation.distill_config import load_cache_builder_config


def build_latent_cache(config_path: Path):
    """Build VAE latent cache using a cache builder config."""
    cfg = load_cache_builder_config(config_path)

    device = cfg["device"]
    teacher_cfg = cfg["teacher"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # Determine VAE source: model.id → model.text_encoder_id → teacher.checkpoint_path
    vae_source = (
        model_cfg.get("id")
        or model_cfg.get("text_encoder_id")
        or teacher_cfg.get("checkpoint_path")
    )
    if not vae_source:
        raise ValueError(
            "No VAE source found. Set model.id, model.text_encoder_id, "
            "or teacher.checkpoint_path in config."
        )
    vae_source = Path(vae_source).expanduser()
    print(f"VAE source: {vae_source}")

    # Determine dtype for encoding (VAE always runs in FP32 for stability)
    use_bf16 = teacher_cfg.get("use_bf16", True)
    cache_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Load tokenizers (required by SimpleCaptionDataset.__init__)
    tokenizer_source = (
        model_cfg.get("text_encoder_id")
        or model_cfg.get("id")
        or teacher_cfg.get("checkpoint_path")
    )
    print(f"Loading tokenizers from: {tokenizer_source}")
    tokenizer_1 = AutoTokenizer.from_pretrained(
        tokenizer_source, subfolder="tokenizer", use_fast=False
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        tokenizer_source, subfolder="tokenizer_2", use_fast=False
    )

    # Setup dataset
    bucket_cfg = data_cfg.get("bucket", {})
    latent_cache_cfg = data_cfg.get("latent_cache", {})

    if not latent_cache_cfg.get("enabled", False):
        raise ValueError(
            "Latent cache must be enabled. "
            "Set data.latent_cache.enabled = true in config."
        )

    dataset = SimpleCaptionDataset(
        img_dir=data_cfg["image_dir"],
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        size=data_cfg["size"],
        caption_dropout_prob=0.0,
        caption_shuffle_prob=0.0,
        bucket_config=bucket_cfg,
        latent_cache_config=latent_cache_cfg,
    )
    print(f"Dataset: {len(dataset)} samples")

    # Check what's missing
    dataset.refresh_latent_cache_state()
    missing = dataset.get_missing_latent_indices()
    if not missing:
        print("Latent cache is complete. Nothing to do.")
        return

    total = len(missing)
    print(f"Missing latents: {total} / {len(dataset)}")

    # Load VAE (FP32 for encoding stability)
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        vae_source, subfolder="vae", torch_dtype=torch.float32
    )
    vae.to(device=device, dtype=torch.float32)
    vae.eval()
    vae.requires_grad_(False)
    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 0.18215)

    # Group missing indices by bucket for efficient batching
    bucket_groups = {}
    sample_buckets = getattr(dataset, "sample_buckets", None)
    for idx in missing:
        bucket_key = None
        if sample_buckets is not None and idx < len(sample_buckets):
            bucket_key = sample_buckets[idx]
        bucket_groups.setdefault(bucket_key, []).append(idx)

    # Build batch size from config (default 4)
    build_batch_size = max(1, int(latent_cache_cfg.get("build_batch_size", 4)))
    print(f"Batch size: {build_batch_size}")

    # Encode and save
    progress = tqdm(total=total, desc="Building latent cache", unit="img")
    try:
        with torch.no_grad():
            for bucket_key, idx_list in bucket_groups.items():
                for start in range(0, len(idx_list), build_batch_size):
                    batch_indices = idx_list[start : start + build_batch_size]
                    pixel_tensors = [
                        dataset.load_image_tensor_for_cache(idx)
                        for idx in batch_indices
                    ]
                    pixel_batch = torch.stack(pixel_tensors).to(
                        device=device, dtype=torch.float32
                    )
                    latents_batch = (
                        vae.encode(pixel_batch).latent_dist.sample() * scaling_factor
                    )
                    latents_batch = latents_batch.to(cache_dtype).cpu()
                    for idx, latent in zip(batch_indices, latents_batch):
                        dataset.save_latent(idx, latent)
                    progress.update(len(batch_indices))
    finally:
        progress.close()

    # Cleanup
    vae.cpu()
    del vae
    torch.cuda.empty_cache()

    # Verify
    dataset.refresh_latent_cache_state()
    still_missing = dataset.get_missing_latent_indices()
    print(f"\nDone! Built {total - len(still_missing)} latents.")
    if still_missing:
        print(f"Warning: {len(still_missing)} still missing.")
    else:
        print("Latent cache is now complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Build VAE latent cache (standalone, no UNet/optimizer needed)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to cache builder config JSON",
    )
    args = parser.parse_args()

    build_latent_cache(args.config)


if __name__ == "__main__":
    main()
