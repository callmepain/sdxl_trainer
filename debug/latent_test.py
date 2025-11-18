import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from diffusers import AutoencoderKL
from PIL import Image


def load_vae(vae_path: str, device: torch.device, use_plain_vae: bool):
    """
    vae_path:
      - entweder HF-ID (z.B. "stabilityai/stable-diffusion-xl-base-1.0")
      - oder lokaler Diffusers-Ordner
    """

    if use_plain_vae:
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=torch.float16,
        )
    else:
        # klassischer SDXL-Diffusers-Ordner: VAE liegt im "vae" Subordner
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

    vae.to(device)
    vae.eval()
    return vae


def load_latents(latent_path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Erwartet:
      - .safetensors mit entweder:
        - key "latents"
        - oder nur einem einzigen Eintrag
    Shape kann sein:
      - [4, H/8, W/8]
      - [1, 4, H/8, W/8]
    """
    data = load_file(latent_path)  # dict[str, Tensor]

    if "latents" in data:
        latents = data["latents"]
    elif len(data) == 1:
        latents = next(iter(data.values()))
    else:
        raise ValueError(
            f"{latent_path} enthält mehrere Keys: {list(data.keys())} – "
            f"ich weiß nicht, welchen ich nehmen soll."
        )

    if latents.ndim == 3:
        # [4, H/8, W/8] -> [1,4,H/8,W/8]
        latents = latents.unsqueeze(0)
    elif latents.ndim != 4:
        raise ValueError(f"Unerwartete Latent-Shape: {latents.shape}")

    latents = latents.to(device=device, dtype=dtype)
    return latents


def decode_and_save(vae: AutoencoderKL,
                    latents: torch.Tensor,
                    out_dir: Path,
                    basename: str,
                    scaling_factor: float | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # scaling_factor aus dem VAE nehmen, falls vorhanden
    if scaling_factor is None:
        scaling_factor = getattr(vae, "scaling_factor", None)
        if scaling_factor is None:
            raise ValueError(
                "Kein scaling_factor angegeben und VAE hat keinen scaling_factor-Attr. "
                "Bitte --scaling_factor explizit setzen."
            )

    # SDXL: typischerweise 0.13025
    # latents: [B,4,H/8,W/8]
    with torch.no_grad():
        latents_dec = latents / float(scaling_factor)
        images = vae.decode(latents_dec).sample  # [-1,1], Shape [B,3,H,W]

    images = (images.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0,1]
    images = images.detach().cpu()

    for idx, img in enumerate(images):
        # [3,H,W] -> [H,W,3]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255.0).round().astype("uint8")

        pil_img = Image.fromarray(img_np)
        out_path = out_dir / f"{basename}_decoded_{idx:03d}.png"
        pil_img.save(out_path)
        print(f"Gespeichert: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent_path",
        type=str,
        required=True,
        help="Pfad zur .safetensors-Datei mit den Latents",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help=(
            "HuggingFace-ID oder lokaler Diffusers-Ordner. "
            "Bei SDXL z.B. 'stabilityai/stable-diffusion-xl-base-1.0' "
            "oder '/pfad/zum/diffusers-modell'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="decoded_latents",
        help="Output-Ordner für PNGs",
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=None,
        help="Optional: überschreibt den scaling_factor des VAE (SDXL: 0.13025).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Auf CPU decodieren (langsam, aber geht immer).",
    )
    parser.add_argument(
        "--plain_vae",
        action="store_true",
        help="Setzen, wenn vae_path direkt ein VAE-Ordner ist, NICHT ein kompletter SDXL-Pipeline-Ordner.",
    )

    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Nutze Device: {device}")

    vae = load_vae(args.vae_path, device, use_plain_vae=args.plain_vae)

    latents = load_latents(args.latent_path, device, vae.dtype)

    latent_path = Path(args.latent_path)
    basename = latent_path.stem
    out_dir = Path(args.output_dir)

    decode_and_save(
        vae=vae,
        latents=latents,
        out_dir=out_dir,
        basename=basename,
        scaling_factor=args.scaling_factor,
    )


if __name__ == "__main__":
    main()
