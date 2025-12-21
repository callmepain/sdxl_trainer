from pathlib import Path
from typing import Optional

import torch


def save_training_state(
    state_path: Path,
    optimizer_unet,
    optimizer_te: Optional[object],
    scaler,
    ema_model,
    global_step: int,
    epoch: int,
    lr_scheduler=None,
) -> None:
    """Save training state with separate optimizers for UNet and text encoders.

    Args:
        state_path: Path to save the state file
        optimizer_unet: UNet optimizer (AdamW8bit)
        optimizer_te: Text encoder optimizer (AdamW), None if not training TEs
        scaler: AMP GradScaler
        ema_model: EMA model state
        global_step: Current training step
        epoch: Current epoch
        lr_scheduler: Optional LR scheduler
    """
    state_path = state_path.expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "optimizer_unet": optimizer_unet.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if optimizer_te is not None:
        payload["optimizer_te"] = optimizer_te.state_dict()
    if ema_model is not None:
        payload["ema"] = ema_model.state_dict()
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(payload, state_path)


__all__ = ["save_training_state"]
