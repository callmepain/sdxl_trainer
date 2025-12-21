#!/usr/bin/env python3
"""
Build teacher prediction cache for distillation training.

Usage:
    python distillation/build_teacher_cache.py --config distillation/configs/cache_builder.json

This script:
1. Loads a teacher model from a diffusers checkpoint
2. Iterates through the dataset
3. For each sample:
   - Generates deterministic noise and timestep
   - Encodes text with text encoders
   - Runs teacher UNet forward pass
   - Saves prediction, noise, timestep, and embeddings to cache
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import SimpleCaptionDataset
from distillation.distill_config import load_cache_builder_config
from distillation.distill_utils import (
    compute_deterministic_seed,
    compute_cache_path,
    generate_deterministic_timestep,
    generate_deterministic_noise,
    save_teacher_cache_entry,
)


def build_teacher_cache(config_path: Path):
    """Main entry point for cache building."""
    cfg = load_cache_builder_config(config_path)

    device = cfg["device"]
    teacher_cfg = cfg["teacher"]
    data_cfg = cfg["data"]
    cache_cfg = cfg["cache"]
    model_cfg = cfg["model"]

    # Validate teacher checkpoint
    checkpoint_path = Path(teacher_cfg["checkpoint_path"]).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    # Derive teacher_id if not specified
    teacher_id = teacher_cfg.get("teacher_id") or checkpoint_path.name
    print(f"Building cache for teacher: {teacher_id}")
    print(f"Checkpoint: {checkpoint_path}")

    # Setup dtype
    use_bf16 = teacher_cfg.get("use_bf16", True)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    cache_dtype_str = cache_cfg.get("dtype", "fp16")
    cache_dtype = torch.bfloat16 if cache_dtype_str == "bf16" else torch.float16

    # Load tokenizers - prefer base model if specified, otherwise use teacher checkpoint
    tokenizer_source = model_cfg.get("id") or checkpoint_path
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

    dataset = SimpleCaptionDataset(
        img_dir=data_cfg["image_dir"],
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        size=data_cfg["size"],
        caption_dropout_prob=0.0,  # No augmentation for cache building
        caption_shuffle_prob=0.0,
        bucket_config=bucket_cfg,
        latent_cache_config=latent_cache_cfg,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Check latent cache
    if latent_cache_cfg.get("enabled", False):
        dataset.refresh_latent_cache_state()
        missing = dataset.get_missing_latent_indices()
        if missing:
            raise ValueError(
                f"Latent cache incomplete: {len(missing)} missing. "
                f"Run normal training first to build latent cache."
            )
        dataset.activate_latent_cache()
        print("Latent cache activated")
    else:
        raise ValueError(
            "Latent cache must be enabled for cache building. "
            "Set data.latent_cache.enabled = true in config."
        )

    # Load teacher model
    print("Loading teacher model...")
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler

    pipe = StableDiffusionXLPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)

    teacher_unet = pipe.unet
    teacher_unet.eval()
    teacher_unet.requires_grad_(False)

    te1 = pipe.text_encoder
    te2 = pipe.text_encoder_2
    te1.eval()
    te2.eval()
    te1.requires_grad_(False)
    te2.requires_grad_(False)

    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Cache parameters
    cache_dir = Path(cache_cfg["output_dir"]).expanduser()
    cache_version = str(cache_cfg.get("version", "1"))
    base_seed = int(cache_cfg.get("base_seed", 42))
    min_timestep = int(cache_cfg.get("min_timestep", 0))
    max_timestep = int(cache_cfg.get("max_timestep", 1000))
    skip_existing = bool(cache_cfg.get("skip_existing", True))

    print(f"Cache directory: {cache_dir / teacher_id}")
    print(f"Cache version: {cache_version}")
    print(f"Timestep range: [{min_timestep}, {max_timestep})")
    print(f"Skip existing: {skip_existing}")

    # Process dataset
    skipped = 0
    processed = 0
    errors = 0

    with torch.inference_mode():
        for idx in tqdm(range(len(dataset)), desc="Building cache"):
            try:
                sample = dataset[idx]
                image_id = sample.get("image_id")
                if image_id is None:
                    # Fallback: use file path relative to img_dir
                    image_id = str(
                        dataset.files[idx].relative_to(dataset.img_dir).with_suffix('')
                    )

                target_size = sample["target_size"]
                # target_size is [height, width], resolution is (width, height)
                resolution = (int(target_size[1]), int(target_size[0]))

                # Compute cache path
                cache_path = compute_cache_path(
                    cache_dir, image_id, teacher_id, resolution, cache_version
                )

                if skip_existing and cache_path.exists():
                    skipped += 1
                    continue

                # Get latents [C, H, W]
                latents = sample["latents"].unsqueeze(0).to(device=device, dtype=dtype)

                # Generate deterministic seed, noise, and timestep
                seed = compute_deterministic_seed(image_id, teacher_id, base_seed)
                noise = generate_deterministic_noise(
                    seed, latents.shape, device, dtype
                )
                timestep = generate_deterministic_timestep(seed, min_timestep, max_timestep)
                timesteps = torch.tensor([timestep], device=device, dtype=torch.long)

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text
                input_ids_1 = sample["input_ids_1"].unsqueeze(0).to(device)
                input_ids_2 = sample["input_ids_2"].unsqueeze(0).to(device)

                enc_1 = te1(
                    input_ids_1,
                    output_hidden_states=True,
                    return_dict=True,
                )
                enc_2 = te2(
                    input_ids_2,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Use penultimate hidden states (standard SDXL approach)
                prompt_embeds_1 = enc_1.hidden_states[-2]  # [1, 77, 768]
                prompt_embeds_2 = enc_2.hidden_states[-2]  # [1, 77, 1280]

                encoder_hidden_states = torch.cat(
                    [prompt_embeds_1, prompt_embeds_2], dim=-1
                )  # [1, 77, 2048]
                pooled_embeds = enc_2.text_embeds  # [1, 1280]

                # SDXL time conditioning
                height, width = resolution[1], resolution[0]
                add_time_ids = torch.tensor(
                    [[width, height, 0, 0, width, height]],
                    device=device,
                    dtype=encoder_hidden_states.dtype,
                )

                # Teacher forward pass
                teacher_pred = teacher_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(dtype),
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds.to(dtype),
                        "time_ids": add_time_ids,
                    },
                ).sample

                # Save to cache (squeeze batch dimension)
                save_teacher_cache_entry(
                    path=cache_path,
                    teacher_pred=teacher_pred.squeeze(0),
                    noise=noise.squeeze(0),
                    timestep=timestep,
                    seed=seed,
                    resolution=resolution,
                    encoder_hidden_states=encoder_hidden_states.squeeze(0),
                    pooled_embeds=pooled_embeds.squeeze(0),
                    dtype=cache_dtype,
                )
                processed += 1

            except Exception as e:
                errors += 1
                print(f"\nError processing sample {idx}: {e}")
                continue

    print(f"\nCache building complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Cache directory: {cache_dir / teacher_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Build teacher prediction cache for distillation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to cache builder config JSON",
    )
    args = parser.parse_args()

    build_teacher_cache(args.config)


if __name__ == "__main__":
    main()
