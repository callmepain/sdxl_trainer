import math
from collections.abc import Sequence

import torch


class LearningRateController:
    def __init__(self, cfg: dict, total_steps: int | None, param_groups: Sequence[dict]):
        self.scheduler_type = (cfg.get("type") or "").strip().lower()
        valid_types = {"constant", "cosine_decay", "linear_decay", "cosine_restarts"}
        if self.scheduler_type not in valid_types:
            raise ValueError(f"Unbekannter lr_scheduler.type: {cfg.get('type')}")
        self.warmup_steps = max(0, int(cfg.get("warmup_steps", 0) or 0))
        self.min_factor = float(cfg.get("min_factor", 0.0) or 0.0)
        self.total_steps = int(total_steps) if total_steps is not None else None
        self.param_groups = list(param_groups)
        self.last_factor = 1.0
        cycle_steps_cfg = cfg.get("cycle_steps")
        inferred_cycle = None
        if (
            cycle_steps_cfg in (None, 0, "0", "")
            and self.total_steps is not None
            and self.total_steps > self.warmup_steps
        ):
            inferred_cycle = max(1, self.total_steps - self.warmup_steps)
        base_cycle = cycle_steps_cfg if cycle_steps_cfg not in (None, 0, "0", "") else inferred_cycle
        if base_cycle in (None, 0, "0", ""):
            base_cycle = 400
        self.cycle_steps = max(1, int(float(base_cycle)))
        cycle_mult_cfg = cfg.get("cycle_mult")
        if cycle_mult_cfg in (None, "", 0):
            cycle_mult_cfg = 1.0
        self.cycle_mult = float(cycle_mult_cfg)
        if self.cycle_mult <= 0:
            self.cycle_mult = 1.0

    def compute_factor(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return max((step + 1) / float(self.warmup_steps), 1e-8)

        if self.scheduler_type == "constant" or self.total_steps is None or self.total_steps <= self.warmup_steps:
            if self.scheduler_type == "cosine_restarts":
                return self._cosine_restarts_factor(step)
            return 1.0

        if self.scheduler_type == "cosine_restarts":
            return self._cosine_restarts_factor(step)

        progress_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(step - self.warmup_steps, 0) / progress_steps, 1.0)
        min_factor = max(0.0, min(self.min_factor, 1.0))

        if self.scheduler_type == "linear_decay":
            return max(min_factor, 1.0 - (1.0 - min_factor) * progress)

        if self.scheduler_type == "cosine_decay":
            cosine = (1 + math.cos(math.pi * progress)) / 2.0
            return min_factor + (1.0 - min_factor) * cosine

        return 1.0

    def apply(self, optimizer, step: int) -> float:
        factor = self.compute_factor(step)
        for group in self.param_groups:
            optimizer.param_groups[group["idx"]]["lr"] = group["base_lr"] * factor
        self.last_factor = factor
        return factor

    def _cosine_restarts_factor(self, step: int) -> float:
        effective = max(step - self.warmup_steps, 0)
        base_len = max(1, self.cycle_steps)
        cycle_mult = self.cycle_mult if self.cycle_mult is not None else 1.0
        if cycle_mult <= 1.0:
            cur_len = base_len
            pos = effective % cur_len
        else:
            cur_len = base_len
            pos = effective
            while pos >= cur_len:
                pos -= cur_len
                next_len = int(math.ceil(cur_len * cycle_mult))
                cur_len = next_len if next_len > 0 else cur_len
                if cur_len <= 0:
                    cur_len = 1
        t = pos / max(cur_len, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        min_factor = max(0.0, min(self.min_factor, 1.0))
        return min_factor + (1.0 - min_factor) * cosine


def compute_grad_norm(optimizer) -> float:
    norms = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            norms.append(grad.float().norm(2))
    if not norms:
        return 0.0
    stacked = torch.stack(norms)
    return torch.norm(stacked).item()


__all__ = ["LearningRateController", "compute_grad_norm"]
