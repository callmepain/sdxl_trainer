#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a single-file .safetensors checkpoint to a Diffusers folder."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the .safetensors file (single-file checkpoint).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for the Diffusers pipeline (default: <input>_diffusers).",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=sorted(DTYPE_MAP.keys()),
        help="Weights dtype to load and save.",
    )
    parser.add_argument(
        "--pipeline",
        default="auto",
        choices=["auto", "sdxl", "sd"],
        help="Pipeline type (auto tries SDXL, then SD).",
    )
    parser.add_argument(
        "--original-config",
        default=None,
        help="Optional original config YAML/JSON for non-SDXL checkpoints.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory if it already exists.",
    )
    return parser.parse_args()


def _resolve_optional(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    return Path(path_str).expanduser().resolve()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Eingabedatei nicht gefunden: {input_path}")

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        stem = input_path.name
        if stem.endswith(".safetensors"):
            stem = stem[: -len(".safetensors")]
        output_dir = input_path.with_name(f"{stem}_diffusers")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SystemExit(f"Ausgabepfad existiert und ist kein Ordner: {output_dir}")
        has_contents = any(output_dir.iterdir())
        if has_contents and not args.overwrite:
            raise SystemExit(f"Ausgabeordner existiert bereits: {output_dir} (nutze --overwrite)")
        if has_contents and args.overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = DTYPE_MAP[args.dtype]
    extra_kwargs = {"torch_dtype": torch_dtype}
    original_config = _resolve_optional(args.original_config)
    if original_config is not None:
        extra_kwargs["original_config_file"] = str(original_config)

    with torch.inference_mode():
        if args.pipeline == "sdxl":
            pipe = StableDiffusionXLPipeline.from_single_file(str(input_path), **extra_kwargs)
        elif args.pipeline == "sd":
            pipe = StableDiffusionPipeline.from_single_file(str(input_path), **extra_kwargs)
        else:
            try:
                pipe = StableDiffusionXLPipeline.from_single_file(str(input_path), **extra_kwargs)
            except Exception:
                pipe = StableDiffusionPipeline.from_single_file(str(input_path), **extra_kwargs)
        pipe.save_pretrained(str(output_dir), safe_serialization=True)

    print(f"Diffusers-Ordner geschrieben: {output_dir}")


if __name__ == "__main__":
    main()
