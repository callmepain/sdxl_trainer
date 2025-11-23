import torch
from safetensors.torch import load_file

base = load_file(".output/safetensors/Krueger3k6_0_0_1.safetensors")
finetuned = load_file(".output/safetensors/Krueger3k6_high_lr.safetensors")

PREFIXES = {
    # Matches UNet weights from the single-file SDXL export
    "unet": ["model.diffusion_model."],
    # SDXL text encoder 1 (CLIPTextModel) in our converter lands under this prefix
    "te1": ["conditioner.embedders.0.transformer."],
    # SDXL text encoder 2 (OpenCLIP with projection)
    "te2": ["conditioner.embedders.1.model."],
}


def diff_stats(prefixes):
    diffs = []
    total = 0
    for k, v in base.items():
        if not any(k.startswith(prefix) for prefix in prefixes):
            continue
        total += 1
        if k not in finetuned:
            continue
        d = (finetuned[k] - v).float()
        diffs.append(d.pow(2).mean().item())
    if total == 0:
        return None, 0, 0
    return sum(diffs) / max(len(diffs), 1), total, len(diffs)

for label, prefixes in PREFIXES.items():
    mse, total, compared = diff_stats(prefixes)
    if mse is None:
        print(f"{label.upper()} diff: keine Parameter mit Prefix(es) {prefixes} gefunden")
    else:
        print(f"{label.upper()} diff MSE: {mse} (verglichen {compared}/{total} Keys)")
