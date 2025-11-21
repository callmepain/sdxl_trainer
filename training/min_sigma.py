from __future__ import annotations

import torch


class MinSigmaController:
    """Applies a floor to sigma after warmup to stabilise training."""

    def __init__(self, sigma_lookup_tensor, min_sigma_value, warmup_steps: int):
        self.active = (
            sigma_lookup_tensor is not None
            and min_sigma_value is not None
            and min_sigma_value > 0
            and warmup_steps > 0
        )
        self.sigma_lookup = sigma_lookup_tensor
        self.min_sigma = min_sigma_value if self.active else None
        self.warmup_steps = warmup_steps if self.active else 0
        self._min_sigma_tensor = None

    def __call__(self, timesteps: torch.LongTensor, step_idx: int):
        if not self.active:
            return None, None

        sigma_original = self.sigma_lookup.index_select(0, timesteps)

        if step_idx < self.warmup_steps:
            return sigma_original, None

        if self._min_sigma_tensor is None or self._min_sigma_tensor.device != sigma_original.device:
            self._min_sigma_tensor = torch.tensor(
                self.min_sigma,
                device=sigma_original.device,
                dtype=sigma_original.dtype,
            )
        sigma_effective = torch.maximum(sigma_original, self._min_sigma_tensor)
        return sigma_original, sigma_effective
