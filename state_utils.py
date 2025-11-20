from pathlib import Path

import torch


def save_training_state(
    state_path: Path,
    optimizer,
    scaler,
    ema_model,
    global_step: int,
    epoch: int,
    lr_scheduler=None,
) -> None:
    state_path = state_path.expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if ema_model is not None:
        payload["ema"] = ema_model.state_dict()
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(payload, state_path)


__all__ = ["save_training_state"]
