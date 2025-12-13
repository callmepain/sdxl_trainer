from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from eval_export import export_single_file
from optim_utils import compute_grad_norm, LearningRateController
from state_utils import save_training_state


@dataclass
class BucketState:
    enabled: bool
    default_batch_size: int
    batch_size_map: Dict[str, int]
    log_switches: bool


@dataclass
class TrainerModules:
    pipe: Any
    unet: Any
    te1: Any
    te2: Any
    optimizer: Any
    lr_controller: Optional[LearningRateController]
    ema_unet: Optional[Any]


@dataclass
class TrainerIO:
    train_loader: Any
    encode_text: Callable[[Dict[str, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]
    prepare_latents: Callable


@dataclass
class TrainingLoopSettings:
    device: torch.device
    dtype: torch.dtype
    use_bf16: bool
    use_ema: bool
    train_text_encoder_1: bool
    train_text_encoder_2: bool
    grad_accum_steps: int
    log_every: Optional[int]
    checkpoint_every: Optional[int]
    noise_offset: float
    snr_gamma: Optional[float]
    max_grad_norm: Optional[float]
    max_grad_norm_te_multiplier: float
    detect_anomaly: bool
    ema_update_every: int
    num_steps: Optional[int]
    num_epochs: Optional[int]
    total_progress_steps: Optional[int]
    data_size: int
    tb_log_grad_norm: bool
    tb_log_scaler: bool
    batch_size: int
    prediction_type: str
    min_timestep: Optional[int]
    max_timestep: Optional[int]


@dataclass
class TrainingPaths:
    output_dir: str
    state_save_path: Path


@dataclass
class ResumeState:
    state_dict: Optional[Dict[str, Any]]
    global_step: int
    epoch: int


@dataclass
class TrainerResult:
    global_step: int
    epoch: int


def run_training_loop(
    *,
    modules: TrainerModules,
    io: TrainerIO,
    settings: TrainingLoopSettings,
    paths: TrainingPaths,
    resume_state: ResumeState,
    alphas_cumprod_tensor: torch.Tensor,
    noise_scheduler: Any,
    enforce_min_sigma: Callable,
    bucket_state: BucketState,
    eval_runner: Optional[Any] = None,
    tb_writer: Optional[Any] = None,
    bucket_key_fn: Optional[Callable] = None,
    te1_group_idx: Optional[int] = None,
    te2_group_idx: Optional[int] = None,
    export_cfg: Optional[Dict[str, Any]] = None,
    ) -> TrainerResult:
    bucket_key_fn = bucket_key_fn or (lambda _: None)

    scaler = torch.amp.GradScaler("cuda", enabled=not settings.use_bf16)
    if resume_state.state_dict is not None:
        scaler_state = resume_state.state_dict.get("scaler")
        if scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception as exc:  # pragma: no cover - defensive
                import warnings

                warnings.warn(f"AMP-Scaler konnte nicht geladen werden ({exc}).", stacklevel=2)

    global_step = resume_state.global_step
    epoch = resume_state.epoch

    modules.unet.train()
    if settings.train_text_encoder_1:
        modules.te1.train()
    else:
        modules.te1.eval()
    if settings.train_text_encoder_2:
        modules.te2.train()
    else:
        modules.te2.eval()

    modules.optimizer.zero_grad(set_to_none=True)

    pbar_total = settings.total_progress_steps
    if pbar_total is not None and global_step > pbar_total:
        pbar_total = global_step
    pbar = tqdm(total=pbar_total, desc="SDXL Training", unit="step", initial=global_step)
    accum_counter = 0
    last_loss_value = 0.0
    last_bucket_key = None

    ema_start_step = (
        modules.lr_controller.warmup_steps if (settings.use_ema and modules.lr_controller is not None) else 0
    )

    def module_grad_norm(module) -> float:
        norms = []
        for param in module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            norms.append(grad.float().norm(2))
        if not norms:
            return 0.0
        return torch.norm(torch.stack(norms)).item()

    def optimizer_step_fn(loss_value, current_step, current_accum):
        grad_norm_value = None
        grad_norm_te1 = None
        grad_norm_te2 = None
        grad_norm_before_clip = None
        current_lr_factor = None
        if modules.lr_controller is not None:
            current_lr_factor = modules.lr_controller.apply(modules.optimizer, current_step)

        need_unscale = (
            (settings.max_grad_norm is not None and settings.max_grad_norm > 0)
            or (tb_writer is not None and settings.tb_log_grad_norm)
        )
        if need_unscale:
            scaler.unscale_(modules.optimizer)
            if tb_writer is not None and settings.tb_log_grad_norm:
                grad_norm_value = compute_grad_norm(modules.optimizer)
                if settings.train_text_encoder_1:
                    grad_norm_te1 = module_grad_norm(modules.te1)
                if settings.train_text_encoder_2:
                    grad_norm_te2 = module_grad_norm(modules.te2)
            if settings.max_grad_norm is not None and settings.max_grad_norm > 0:
                # Clip UNet separately
                unet_params = [p for p in modules.optimizer.param_groups[0]["params"] if p.grad is not None]
                if unet_params:
                    grad_norm_before_clip = clip_grad_norm_(
                        unet_params,
                        max_norm=settings.max_grad_norm,
                        norm_type=2.0,
                        error_if_nonfinite=settings.detect_anomaly,
                    )

                # Clip TEs with separate (higher) threshold to avoid over-clipping
                te_max_norm = settings.max_grad_norm * settings.max_grad_norm_te_multiplier
                if settings.train_text_encoder_1 and te1_group_idx is not None:
                    te1_params = [p for p in modules.optimizer.param_groups[te1_group_idx]["params"] if p.grad is not None]
                    if te1_params:
                        clip_grad_norm_(
                            te1_params,
                            max_norm=te_max_norm,
                            norm_type=2.0,
                            error_if_nonfinite=settings.detect_anomaly,
                        )

                if settings.train_text_encoder_2 and te2_group_idx is not None:
                    te2_params = [p for p in modules.optimizer.param_groups[te2_group_idx]["params"] if p.grad is not None]
                    if te2_params:
                        clip_grad_norm_(
                            te2_params,
                            max_norm=te_max_norm,
                            norm_type=2.0,
                            error_if_nonfinite=settings.detect_anomaly,
                        )

        scaler.step(modules.optimizer)
        scaler.update()

        modules.optimizer.zero_grad(set_to_none=True)
        current_accum = 0

        if modules.ema_unet is not None and current_step >= ema_start_step:
            if (current_step - ema_start_step) % settings.ema_update_every == 0:
                modules.ema_unet.step(modules.unet.parameters())

        current_step += 1
        display_loss = float(loss_value) if loss_value is not None else 0.0
        pbar.update(1)
        pbar.set_postfix({"loss": f"{display_loss:.4f}"})

        if settings.log_every is not None and current_step % settings.log_every == 0:
            lr_summary = ""
            if modules.lr_controller is not None:
                lr_ctrl = modules.lr_controller  # Cache to prevent theoretical None-dereferencing
                unet_lr = modules.optimizer.param_groups[0]["lr"]
                lr_parts = [f"unet={unet_lr:.3e}"]
                if settings.train_text_encoder_1 and te1_group_idx is not None:
                    lr_parts.append(f"te1={modules.optimizer.param_groups[te1_group_idx]['lr']:.3e}")
                if settings.train_text_encoder_2 and te2_group_idx is not None:
                    lr_parts.append(f"te2={modules.optimizer.param_groups[te2_group_idx]['lr']:.3e}")
                factor_value = current_lr_factor if current_lr_factor is not None else lr_ctrl.last_factor
                lr_summary = (
                    f" | lr {'/'.join(lr_parts)}"
                    f" (factor={factor_value:.4f}, "
                    f"{lr_ctrl.scheduler_type})"
                )
            print(f"step {current_step} | loss {display_loss:.4f}{lr_summary}")

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", display_loss, current_step)
            tb_writer.add_scalar("train/lr_unet", modules.optimizer.param_groups[0]["lr"], current_step)
            if settings.train_text_encoder_1 and te1_group_idx is not None:
                tb_writer.add_scalar(
                    "train/lr_text_encoder_1", modules.optimizer.param_groups[te1_group_idx]["lr"], current_step
                )
            if settings.train_text_encoder_2 and te2_group_idx is not None:
                tb_writer.add_scalar(
                    "train/lr_text_encoder_2", modules.optimizer.param_groups[te2_group_idx]["lr"], current_step
                )
            if grad_norm_value is not None:
                tb_writer.add_scalar("train/grad_norm", grad_norm_value, current_step)
            if grad_norm_before_clip is not None:
                tb_writer.add_scalar("train/grad_norm_before_clip", grad_norm_before_clip, current_step)
            if grad_norm_te1 is not None:
                tb_writer.add_scalar("train/grad_norm_te1", grad_norm_te1, current_step)
            if grad_norm_te2 is not None:
                tb_writer.add_scalar("train/grad_norm_te2", grad_norm_te2, current_step)
            if settings.tb_log_scaler:
                tb_writer.add_scalar("train/amp_scale", scaler.get_scale(), current_step)
            tb_writer.add_scalar("train/global_step", current_step, current_step)
            if modules.lr_controller is not None and current_lr_factor is not None:
                tb_writer.add_scalar("train/lr_factor", current_lr_factor, current_step)

        if settings.checkpoint_every is not None and current_step % settings.checkpoint_every == 0:
            if modules.ema_unet is not None:
                modules.ema_unet.store(modules.unet.parameters())
                modules.ema_unet.copy_to(modules.unet.parameters())

            modules.pipe.save_pretrained(f"{paths.output_dir}_step_{current_step}")
            save_training_state(paths.state_save_path, modules.optimizer, scaler, modules.ema_unet, current_step, epoch, None)

            if modules.ema_unet is not None:
                modules.ema_unet.restore(modules.unet.parameters())

        if eval_runner is not None:
            final_pending = bool(settings.num_steps is not None and current_step >= settings.num_steps)
            eval_runner.maybe_run_live(current_step, final_pending=final_pending)

        return current_step, current_accum

    while True:
        if settings.num_epochs is not None and epoch >= settings.num_epochs:
            break

        for batch in io.train_loader:
            if settings.num_steps is not None and global_step >= settings.num_steps:
                break

            if bucket_state.log_switches:
                bucket_key = bucket_key_fn(batch.get("target_size"))
                if bucket_key is not None and bucket_key != last_bucket_key:
                    effective_bucket_batch = bucket_state.batch_size_map.get(
                        bucket_key,
                        bucket_state.default_batch_size if bucket_state.enabled else settings.batch_size,
                    )
                    print(f"Bucket aktiv: {bucket_key} (batch_size={effective_bucket_batch}, step={global_step})")
                    last_bucket_key = bucket_key

            with torch.amp.autocast("cuda", dtype=settings.dtype, enabled=not settings.use_bf16):
                latents = batch.get("latents")
                if latents is not None:
                    latents = latents.to(device=settings.device, dtype=settings.dtype)
                else:
                    pixel_values = batch["pixel_values"]
                    latents = io.prepare_latents(pixel_values)

                noise = torch.randn_like(latents)
                if settings.noise_offset > 0:
                    noise = noise + settings.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1),
                        device=settings.device,
                        dtype=noise.dtype,
                    )
                min_t = settings.min_timestep if settings.min_timestep is not None else 0
                max_t = (
                    settings.max_timestep
                    if settings.max_timestep is not None
                    else noise_scheduler.config.num_train_timesteps
                )
                timesteps = torch.randint(
                    min_t,
                    max_t,
                    (latents.shape[0],),
                    device=settings.device,
                    dtype=torch.long,
                )
                sigma_original, sigma_effective = enforce_min_sigma(timesteps, global_step)
                sigma_for_loss = sigma_effective if sigma_effective is not None else sigma_original
                if tb_writer is not None:
                    tb_writer.add_scalar("train/timestep/mean", timesteps.float().mean().item(), global_step)
                    tb_writer.add_scalar("train/timestep/min", timesteps.float().min().item(), global_step)
                    tb_writer.add_scalar("train/timestep/max", timesteps.float().max().item(), global_step)
                    if sigma_original is not None:
                        tb_writer.add_scalar("train/sigma/original", sigma_original.mean().item(), global_step)
                        if sigma_effective is not None:
                            tb_writer.add_scalar("train/sigma/effective", sigma_effective.mean().item(), global_step)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_embeds, pooled_embeds = io.encode_text(batch)

                target_sizes = batch.get("target_size")
                if target_sizes is not None:
                    target_sizes = target_sizes.to(device=settings.device)
                    if target_sizes.ndim != 2 or target_sizes.shape[1] != 2:
                        raise ValueError(
                            f"target_size must have shape (batch_size, 2), got {target_sizes.shape}"
                        )
                    heights = target_sizes[:, 0].to(pooled_embeds.dtype)
                    widths = target_sizes[:, 1].to(pooled_embeds.dtype)
                else:
                    widths = torch.full(
                        (latents.shape[0],),
                        settings.data_size,
                        device=settings.device,
                        dtype=pooled_embeds.dtype,
                    )
                    heights = widths
                zeros = torch.zeros_like(widths)
                add_time_ids = torch.stack(
                    [widths, heights, zeros, zeros, widths, heights],
                    dim=1,
                )

                model_pred = modules.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
                ).sample

                if settings.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                elif settings.prediction_type == "sample":
                    target = latents
                else:
                    target = noise

                loss_dims = tuple(range(1, model_pred.ndim))
                per_example_loss = torch.mean((model_pred.float() - target.float()) ** 2, dim=loss_dims)

                if settings.snr_gamma is not None and settings.prediction_type in ("epsilon", "v_prediction"):
                    if sigma_for_loss is not None:
                        sigma_vals = sigma_for_loss.to(device=per_example_loss.device, dtype=per_example_loss.dtype)
                        snr_vals = 1.0 / torch.clamp(sigma_vals ** 2, min=1e-8)
                    else:
                        alphas_now = alphas_cumprod_tensor.index_select(0, timesteps).to(device=per_example_loss.device, dtype=per_example_loss.dtype)
                        snr_vals = alphas_now / torch.clamp(1 - alphas_now, min=1e-8)
                    gamma_tensor = torch.full_like(snr_vals, settings.snr_gamma)
                    snr_weights = torch.minimum(snr_vals, gamma_tensor) / torch.clamp(snr_vals, min=1e-8)
                    per_example_loss = per_example_loss * snr_weights

                raw_loss = per_example_loss.mean()

                if settings.detect_anomaly and not torch.isfinite(raw_loss):
                    raise FloatingPointError(
                        f"Non-finite loss detected at global_step={global_step}, "
                        f"timestep_mean={timesteps.float().mean().item():.2f}"
                    )

                loss = raw_loss / settings.grad_accum_steps

            last_loss_value = raw_loss.item()
            scaler.scale(loss).backward()
            accum_counter += 1

            if accum_counter % settings.grad_accum_steps == 0:
                global_step, accum_counter = optimizer_step_fn(last_loss_value, global_step, accum_counter)

                if settings.num_steps is not None and global_step >= settings.num_steps:
                    break

        if settings.num_steps is not None and global_step >= settings.num_steps:
            break

        if accum_counter > 0 and (settings.num_steps is None or global_step < settings.num_steps):
            global_step, accum_counter = optimizer_step_fn(last_loss_value, global_step, accum_counter)

            if settings.num_steps is not None and global_step >= settings.num_steps:
                break

        epoch += 1

    pbar.close()

    if eval_runner is not None:
        eval_runner.run_final(global_step)

    if modules.ema_unet is not None:
        modules.ema_unet.store(modules.unet.parameters())
        modules.ema_unet.copy_to(modules.unet.parameters())

    modules.pipe.save_pretrained(paths.output_dir)
    save_training_state(paths.state_save_path, modules.optimizer, scaler, modules.ema_unet, global_step, epoch, None)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    if export_cfg is not None:
        export_single_file(Path(paths.output_dir), export_cfg)

    return TrainerResult(global_step=global_step, epoch=epoch)
