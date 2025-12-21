#!/usr/bin/env python3
"""
Multi-teacher distillation training for SDXL.

Usage:
    python distillation/train_distill.py --config distillation/configs/distill.json

Key features:
- Loads pre-computed teacher predictions from cache
- Supports multiple teachers with configurable weights
- Logs per-teacher loss and combined loss to TensorBoard
- Reuses existing utilities (optimizer, EMA, bucketing)
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import bitsandbytes as bnb
import torch
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import SimpleCaptionDataset, BucketBatchSampler
from optim_utils import LearningRateController, compute_grad_norm
from state_utils import save_training_state

from distillation.distill_config import load_distill_config
from distillation.distill_dataset import DistillationDataset, distillation_collate_fn


def compute_distillation_loss(
    student_pred: torch.Tensor,
    teacher_predictions: Dict[str, torch.Tensor],
    teacher_weights: Dict[str, float],
    loss_type: str = "mse",
) -> tuple:
    """
    Compute weighted distillation loss from multiple teachers.

    Args:
        student_pred: Student model prediction [B, C, H, W]
        teacher_predictions: Dict mapping teacher_id -> prediction tensor
        teacher_weights: Dict mapping teacher_id -> weight
        loss_type: "mse", "huber", or "smooth_l1"

    Returns:
        (combined_loss, per_teacher_losses_dict)
    """
    total_weight = sum(teacher_weights.values())
    combined_loss = torch.tensor(
        0.0, device=student_pred.device, dtype=student_pred.dtype
    )
    per_teacher_losses = {}

    for teacher_id, teacher_pred in teacher_predictions.items():
        weight = teacher_weights.get(teacher_id, 1.0)
        normalized_weight = weight / total_weight

        # Compute loss
        if loss_type == "mse":
            loss = torch.nn.functional.mse_loss(
                student_pred.float(), teacher_pred.float(), reduction="mean"
            )
        elif loss_type == "huber":
            loss = torch.nn.functional.huber_loss(
                student_pred.float(), teacher_pred.float(), reduction="mean", delta=1.0
            )
        elif loss_type == "smooth_l1":
            loss = torch.nn.functional.smooth_l1_loss(
                student_pred.float(), teacher_pred.float(), reduction="mean"
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        per_teacher_losses[teacher_id] = loss.item()
        combined_loss = combined_loss + normalized_weight * loss

    return combined_loss, per_teacher_losses


def train_distillation(config_path: Path):
    """Main training loop for distillation."""
    cfg = load_distill_config(config_path)

    device = cfg["device"]
    run_cfg = cfg["run"]
    student_cfg = cfg["student"]
    teacher_configs = cfg["teachers"]
    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    cache_cfg = cfg["cache"]
    optimizer_cfg = cfg["optimizer"]
    export_cfg = cfg.get("export", {})

    # Validate configuration
    if not teacher_configs:
        raise ValueError(
            "No teachers configured. Add at least one teacher to 'teachers' list."
        )
    if not student_cfg.get("checkpoint_path"):
        raise ValueError("student.checkpoint_path is required")

    print("=" * 60)
    print("Multi-Teacher Distillation Training")
    print("=" * 60)
    print(f"Student: {student_cfg['checkpoint_path']}")
    print(f"Teachers ({len(teacher_configs)}):")
    for tc in teacher_configs:
        print(f"  - {tc['teacher_id']} (weight={tc.get('weight', 1.0)})")

    # Setup output directory
    run_name = run_cfg.get("name", "distill_run")
    output_root = Path(run_cfg.get("output_root", ".output/distill")).expanduser()
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Setup dtype
    use_bf16 = student_cfg.get("use_bf16", True)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Dtype: {'BF16' if use_bf16 else 'FP16'}")

    # Load student model
    student_checkpoint = Path(student_cfg["checkpoint_path"]).expanduser()
    print(f"\nLoading student model from: {student_checkpoint}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        student_checkpoint,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)

    unet = pipe.unet
    unet.train()
    unet.requires_grad_(True)

    if student_cfg.get("use_gradient_checkpointing", True):
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled")

    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Setup dataset
    print("\nSetting up dataset...")
    tokenizer_1 = AutoTokenizer.from_pretrained(
        student_checkpoint, subfolder="tokenizer", use_fast=False
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        student_checkpoint, subfolder="tokenizer_2", use_fast=False
    )

    bucket_cfg = data_cfg.get("bucket", {})
    latent_cache_cfg = data_cfg.get("latent_cache", {})

    base_dataset = SimpleCaptionDataset(
        img_dir=data_cfg["image_dir"],
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        size=data_cfg["size"],
        caption_dropout_prob=0.0,  # No augmentation for distillation
        caption_shuffle_prob=0.0,
        bucket_config=bucket_cfg,
        latent_cache_config=latent_cache_cfg,
    )

    # Activate latent cache
    if not latent_cache_cfg.get("enabled", False):
        raise ValueError("Latent cache must be enabled for distillation training.")

    base_dataset.refresh_latent_cache_state()
    missing = base_dataset.get_missing_latent_indices()
    if missing:
        raise ValueError(
            f"Latent cache incomplete: {len(missing)} missing. "
            f"Run normal training first to build latent cache."
        )
    base_dataset.activate_latent_cache()

    # Create distillation dataset
    cache_version = cache_cfg.get("version", "1")
    require_all_teachers = cache_cfg.get("require_all_teachers", True)

    distill_dataset = DistillationDataset(
        base_dataset=base_dataset,
        teacher_configs=teacher_configs,
        cache_version=cache_version,
        require_all_teachers=require_all_teachers,
    )

    if len(distill_dataset) == 0:
        raise ValueError("No samples with complete teacher caches found!")

    # Setup dataloader
    batch_size = training_cfg["batch_size"]
    bucket_enabled = bucket_cfg.get("enabled", False)

    if bucket_enabled:
        # Create bucket sampler using base dataset's buckets
        # Map distill indices back to base indices for bucket lookup
        bucket_ids = [
            base_dataset.sample_buckets[distill_dataset.valid_indices[i]]
            for i in range(len(distill_dataset))
        ]
        per_bucket_batch_sizes = bucket_cfg.get("per_resolution_batch_sizes", {})
        bucket_sampler = BucketBatchSampler(
            bucket_ids,
            batch_size=batch_size,
            shuffle=True,
            drop_last=bucket_cfg.get("drop_last", True),
            per_bucket_batch_sizes=per_bucket_batch_sizes,
        )
        train_loader = DataLoader(
            distill_dataset,
            batch_sampler=bucket_sampler,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=data_cfg.get("pin_memory", True),
            collate_fn=distillation_collate_fn,
        )
    else:
        train_loader = DataLoader(
            distill_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=data_cfg.get("pin_memory", True),
            collate_fn=distillation_collate_fn,
            drop_last=True,
        )

    print(f"Training samples: {len(distill_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Setup optimizer
    lr = training_cfg["lr"]
    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(),
        lr=lr,
        weight_decay=optimizer_cfg.get("weight_decay", 0.01),
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        eps=optimizer_cfg.get("eps", 1e-8),
    )

    # Setup EMA
    use_ema = student_cfg.get("use_ema", True)
    ema_decay = student_cfg.get("ema_decay", 0.9999)
    ema_update_every = student_cfg.get("ema_update_every", 10)
    ema_unet = EMAModel(unet.parameters(), decay=ema_decay) if use_ema else None
    if use_ema:
        print(f"EMA enabled (decay={ema_decay}, update_every={ema_update_every})")

    # Setup LR scheduler
    lr_scheduler_cfg = training_cfg.get("lr_scheduler", {})
    num_steps = training_cfg.get("num_steps")
    num_epochs = training_cfg.get("num_epochs")
    grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))

    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    if num_steps is not None:
        total_steps = num_steps
    elif num_epochs is not None:
        total_steps = steps_per_epoch * num_epochs
    else:
        total_steps = 10000  # Default

    lr_controller = None
    if lr_scheduler_cfg.get("type"):
        lr_controller = LearningRateController(
            lr_scheduler_cfg,
            total_steps,
            [{"name": "unet", "idx": 0, "base_lr": lr}],
        )
        print(f"LR scheduler: {lr_scheduler_cfg.get('type')}")

    # Setup TensorBoard
    tb_cfg = training_cfg.get("tensorboard", {})
    tb_writer = None
    if tb_cfg.get("enabled", True):
        tb_log_dir = (
            Path(tb_cfg.get("base_dir", "./logs/tensorboard/distill")).expanduser()
            / run_name
        )
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard: {tb_log_dir}")

    # Training settings
    log_every = training_cfg.get("log_every", 50)
    checkpoint_every = training_cfg.get("checkpoint_every", 1000)
    max_grad_norm = training_cfg.get("max_grad_norm")
    loss_type = training_cfg.get("loss_type", "mse")
    log_per_teacher_loss = tb_cfg.get("log_per_teacher_loss", True)
    log_grad_norm = tb_cfg.get("log_grad_norm", True)

    print(f"\nTraining configuration:")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accum: {grad_accum_steps}")
    print(f"  Effective batch: {batch_size * grad_accum_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Loss type: {loss_type}")
    print(f"  Max grad norm: {max_grad_norm}")

    # Setup AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    # Training loop
    global_step = 0
    epoch = 0

    print(f"\n{'=' * 60}")
    print("Starting training...")
    print("=" * 60)

    pbar = tqdm(total=total_steps, desc="Distillation", unit="step")
    optimizer.zero_grad(set_to_none=True)
    accum_counter = 0
    last_loss = 0.0
    last_per_teacher_losses = {}

    while True:
        if num_epochs is not None and epoch >= num_epochs:
            break

        for batch in train_loader:
            if num_steps is not None and global_step >= num_steps:
                break

            with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
                # Move batch to device
                latents = batch["latents"].to(device=device, dtype=dtype)
                noise = batch["noise"].to(device=device, dtype=dtype)
                timesteps = batch["timesteps"].to(device=device)
                encoder_hidden_states = batch["encoder_hidden_states"].to(
                    device=device, dtype=dtype
                )
                pooled_embeds = batch["pooled_embeds"].to(device=device, dtype=dtype)
                target_sizes = batch["target_sizes"].to(device=device)

                teacher_predictions = {
                    k: v.to(device=device, dtype=dtype)
                    for k, v in batch["teacher_predictions"].items()
                }
                teacher_weights = batch["teacher_weights"]

                # Add noise to latents using cached noise and timesteps
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SDXL time conditioning
                # target_sizes is [B, 2] with [height, width]
                heights = target_sizes[:, 0].to(pooled_embeds.dtype)
                widths = target_sizes[:, 1].to(pooled_embeds.dtype)
                zeros = torch.zeros_like(widths)
                add_time_ids = torch.stack(
                    [widths, heights, zeros, zeros, widths, heights],
                    dim=1,
                )

                # Student forward pass
                student_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds,
                        "time_ids": add_time_ids,
                    },
                ).sample

                # Compute distillation loss
                loss, per_teacher_losses = compute_distillation_loss(
                    student_pred, teacher_predictions, teacher_weights, loss_type
                )

                scaled_loss = loss / grad_accum_steps

            last_loss = loss.item()
            last_per_teacher_losses = per_teacher_losses
            scaler.scale(scaled_loss).backward()
            accum_counter += 1

            if accum_counter % grad_accum_steps == 0:
                # Apply LR schedule
                if lr_controller is not None:
                    lr_controller.apply(optimizer, global_step)

                # Gradient clipping
                if max_grad_norm is not None and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        unet.parameters(), max_grad_norm
                    )
                else:
                    grad_norm = compute_grad_norm(optimizer)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

                # EMA update
                if ema_unet is not None and global_step % ema_update_every == 0:
                    ema_unet.step(unet.parameters())

                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{last_loss:.4f}"})

                # Logging
                if log_every and global_step % log_every == 0:
                    teacher_loss_str = ", ".join(
                        f"{k}={v:.4f}" for k, v in last_per_teacher_losses.items()
                    )
                    print(
                        f"step {global_step} | loss {last_loss:.4f} | {teacher_loss_str}"
                    )

                if tb_writer is not None:
                    tb_writer.add_scalar("distill/loss", last_loss, global_step)
                    tb_writer.add_scalar(
                        "distill/lr", optimizer.param_groups[0]["lr"], global_step
                    )

                    if log_per_teacher_loss:
                        for teacher_id, teacher_loss in last_per_teacher_losses.items():
                            tb_writer.add_scalar(
                                f"distill/loss_{teacher_id}", teacher_loss, global_step
                            )

                    if log_grad_norm and grad_norm is not None:
                        tb_writer.add_scalar(
                            "distill/grad_norm",
                            grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                            global_step,
                        )

                # Checkpointing
                if checkpoint_every and global_step % checkpoint_every == 0:
                    ckpt_dir = output_dir / f"step_{global_step}"

                    if ema_unet is not None:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    pipe.save_pretrained(ckpt_dir)
                    save_training_state(
                        ckpt_dir / "trainer_state.pt",
                        optimizer,
                        None,  # No separate TE optimizer in distillation
                        scaler,
                        ema_unet,
                        global_step,
                        epoch,
                    )

                    if ema_unet is not None:
                        ema_unet.restore(unet.parameters())

                    print(f"Checkpoint saved: {ckpt_dir}")

                if num_steps is not None and global_step >= num_steps:
                    break

        if num_steps is not None and global_step >= num_steps:
            break

        epoch += 1

    pbar.close()

    # Final save
    print("\nSaving final checkpoint...")
    if ema_unet is not None:
        ema_unet.store(unet.parameters())
        ema_unet.copy_to(unet.parameters())

    pipe.save_pretrained(output_dir)
    save_training_state(
        output_dir / "trainer_state.pt",
        optimizer,
        None,  # No separate TE optimizer in distillation
        scaler,
        ema_unet,
        global_step,
        epoch,
    )

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    # Export single file if configured
    if export_cfg.get("save_single_file", False):
        try:
            import subprocess

            converter_script = export_cfg.get("converter_script", "./converttosdxl.py")
            half_precision = export_cfg.get("half_precision", True)

            safetensors_dir = output_dir.parent / "safetensors"
            safetensors_dir.mkdir(parents=True, exist_ok=True)
            safetensors_path = safetensors_dir / f"{run_name}.safetensors"

            cmd = [
                sys.executable,
                converter_script,
                "--model_path", str(output_dir),
                "--checkpoint_path", str(safetensors_path),
                "--use_safetensors",
            ]
            if half_precision:
                cmd.append("--half")

            print(f"Exporting single file: {safetensors_path}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Warning: Single file export failed: {e}")

    print(f"\nTraining complete!")
    print(f"Final checkpoint: {output_dir}")
    print(f"Total steps: {global_step}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-teacher distillation training for SDXL"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to distillation config JSON",
    )
    args = parser.parse_args()

    train_distillation(args.config)


if __name__ == "__main__":
    main()
