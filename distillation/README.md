# Multi-Teacher Distillation for SDXL

This module provides tools for training a student SDXL model using knowledge distillation from multiple teacher models.

## Overview

The distillation system works in two phases:

1. **Cache Building**: Pre-compute UNet predictions from teacher models for each training sample
2. **Distillation Training**: Train a student model to match the weighted average of teacher predictions

This approach allows training from multiple teachers without loading them all into memory simultaneously.

## Quick Start

### 1. Build Latent Cache

Build the VAE latent cache first. This standalone script only loads the VAE (no UNet/optimizer needed):

```bash
python distillation/build_latent_cache.py --config distillation/configs/cache_alpha.json
```

Uses the same config as `build_teacher_cache.py`. The VAE source is resolved from: `model.id` → `model.text_encoder_id` → `teacher.checkpoint_path`.

### 2. Build Teacher Caches

Build the prediction cache for each teacher model:

```bash
# Create config for teacher 1
cp distillation/configs/cache_builder.example.json distillation/configs/cache_v8.json
# Edit cache_v8.json: set teacher.checkpoint_path and teacher.teacher_id

# Build cache
python distillation/build_teacher_cache.py --config distillation/configs/cache_gamma.json

# Repeat for other teachers...
python distillation/build_teacher_cache.py --config distillation/configs/cache_chaos.json
python distillation/build_teacher_cache.py --config distillation/configs/cache_bs4.json
```

### 3. Verify Caches

Verify the caches are complete and valid:

```bash
python distillation/verify_cache.py --cache-dir ./cache/teacher_predictions/teacher_v8
python distillation/verify_cache.py --cache-dir ./cache/teacher_predictions/teacher_chaos
python distillation/verify_cache.py --cache-dir ./cache/teacher_predictions/teacher_bs4
```

### 4. Train Student Model

Configure and run distillation training:

```bash
# Create training config
cp distillation/configs/distill.example.json distillation/configs/distill.json
# Edit distill.json: set student checkpoint, teacher configs, training params

# Start training
python distillation/train_distill.py --config distillation/configs/distill.json
```

## Configuration

### Cache Builder Config (`cache_builder.json`)

```json
{
  "device": "cuda",
  "teacher": {
    "checkpoint_path": ".output/my_teacher",  // Diffusers checkpoint path
    "teacher_id": "teacher_v1",               // Unique identifier
    "use_bf16": true
  },
  "data": {
    "image_dir": "./data/images",
    "size": 1024,
    "bucket": {
      "enabled": true,
      "resolutions": [[1024, 1024], [896, 1152], ...]
    },
    "latent_cache": {
      "enabled": true,
      "cache_dir": "./cache/latents"
    }
  },
  "cache": {
    "output_dir": "./cache/teacher_predictions",
    "version": "1",
    "dtype": "fp16",
    "base_seed": 42,
    "min_timestep": 0,
    "max_timestep": 1000,
    "skip_existing": true
  },
  "model": {
    "text_encoder_id": "stabilityai/stable-diffusion-xl-base-1.0"
  }
}
```

### Distillation Training Config (`distill.json`)

```json
{
  "device": "cuda",
  "run": {
    "name": "distill_student_v1",
    "output_root": ".output/distill"
  },
  "student": {
    "checkpoint_path": "stabilityai/stable-diffusion-xl-base-1.0",
    "text_encoder_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "use_bf16": true,
    "use_gradient_checkpointing": true,
    "use_ema": true,
    "ema_decay": 0.9999
  },
  "teachers": [
    {"teacher_id": "teacher_v8", "cache_dir": "./cache/teacher_predictions", "weight": 0.333},
    {"teacher_id": "teacher_chaos", "cache_dir": "./cache/teacher_predictions", "weight": 0.333},
    {"teacher_id": "teacher_bs4", "cache_dir": "./cache/teacher_predictions", "weight": 0.334}
  ],
  "training": {
    "batch_size": 4,
    "num_steps": 10000,
    "lr": 5e-6,
    "loss_type": "mse"
  }
}
```

## Cache Format

Each cache file is a `.safetensors` file containing:

| Key | Shape | Description |
|-----|-------|-------------|
| `teacher_pred` | `[4, H/8, W/8]` | UNet prediction (FP16) |
| `noise` | `[4, H/8, W/8]` | Noise tensor (FP16) |
| `timestep` | `[1]` | Timestep value (int64) |
| `seed` | `[1]` | Random seed (int64) |
| `resolution` | `[2]` | Width, Height (int32) |
| `encoder_hidden_states` | `[77, 2048]` | Text embeddings (FP16) |
| `pooled_embeds` | `[1280]` | Pooled embeddings (FP16) |

Cache files are stored at: `{cache_dir}/{teacher_id}/{hash}_{WxH}.safetensors`

## Deterministic Noise Generation

The system uses deterministic noise and timestep generation to ensure consistency:

- **Seed**: `SHA256(image_id | base_seed)` truncated to 32 bits
- **Timestep**: Sampled using the deterministic seed
- **Noise**: Generated using the deterministic seed

This ensures:
1. Same noise/timestep for the same image across runs
2. Cache files can be regenerated identically
3. All teachers use the same noise/timestep for the same image

## Prerequisites

Before running distillation:

1. **Latent Cache Required**: Build the VAE latent cache first with `build_latent_cache.py`
2. **Teacher Checkpoints**: Teacher models must be in diffusers format (folder with `unet/`, `text_encoder/`, etc.)
3. **Matching Resolutions**: Cache building must use the same bucket resolutions as training

## Memory Requirements

- **Cache Building**: ~12GB VRAM (one teacher model at a time)
- **Distillation Training**: ~20GB VRAM (student model + gradients)
- **Disk Space**: ~380KB per sample per teacher (~4.3GB for 3800 images × 3 teachers)

## Monitoring

Training logs per-teacher and combined losses to TensorBoard:

```bash
tensorboard --logdir ./logs/tensorboard/distill
```

Metrics logged:
- `distill/loss`: Combined weighted loss
- `distill/loss_{teacher_id}`: Per-teacher loss
- `distill/lr`: Learning rate
- `distill/grad_norm`: Gradient norm

## Tips

1. **Teacher Weights**: Start with equal weights, then adjust based on per-teacher loss curves
2. **Loss Type**: MSE works well; try Huber if you see outlier issues
3. **Learning Rate**: Start with 5e-6, similar to regular fine-tuning
4. **Caching on NVMe**: Cache building is I/O bound; fast storage helps
5. **Bucketing**: Use same bucket config for cache building and training
6. **Shared Text Encoders**: Set `model.text_encoder_id` in cache builder configs and `student.text_encoder_id` in distill config so all teachers use the same TE embeddings

## File Structure

```
distillation/
├── __init__.py
├── build_latent_cache.py     # Standalone VAE latent cache builder
├── build_teacher_cache.py    # Teacher prediction cache builder
├── train_distill.py          # Distillation trainer
├── verify_cache.py           # Cache verification utility
├── distill_utils.py          # Shared utilities
├── distill_dataset.py        # Dataset for loading caches
├── distill_config.py         # Config loading
├── configs/
│   ├── cache_builder.example.json
│   └── distill.example.json
└── README.md
```
