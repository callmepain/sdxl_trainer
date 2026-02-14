#!/usr/bin/env python
"""Scan caption files and report entries that exceed the 77-token limit for SDXL TE1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

from transformers import AutoTokenizer

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config {path} wurde nicht gefunden")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_tokenizer_path(cfg: dict) -> str:
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    resume_from = training_cfg.get("resume_from")
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            return str(resume_path)
    model_id = model_cfg.get("id")
    if not model_id:
        raise ValueError("Kein model.id in config gefunden")
    return model_id


def iter_images(image_dir: Path) -> Iterable[Path]:
    for entry in sorted(image_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            yield entry


def read_caption(path: Path) -> str:
    txt_path = path.with_suffix(".txt")
    if not txt_path.exists():
        return ""
    return txt_path.read_text(encoding="utf-8", errors="replace").strip()


def tokenize_length(tokenizer, text: str) -> int:
    if not text:
        return 0
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_length=True,
    )
    length_field = encoded.get("length") or encoded.get("lengths")
    if length_field is not None:
        if isinstance(length_field, (list, tuple)):
            return int(length_field[0])
        return int(length_field)
    input_ids = encoded.get("input_ids")
    if input_ids:
        return len(input_ids[0])
    return 0


def audit_captions(image_dir: Path, tokenizer, max_tokens: int) -> List[Tuple[Path, int, str]]:
    offenders: List[Tuple[Path, int, str]] = []
    for img_path in iter_images(image_dir):
        caption = read_caption(img_path)
        length = tokenize_length(tokenizer, caption)
        if length > max_tokens:
            offenders.append((img_path, length, caption))
    return offenders


def main() -> None:
    parser = argparse.ArgumentParser(description="Find captions exceeding SDXL's 77-token limit")
    parser.add_argument("--config", default="config.json", type=str, help="Pfad zur Config-Datei")
    parser.add_argument("--image-dir", default=None, type=str, help="Override für data.image_dir")
    parser.add_argument("--max-tokens", default=77, type=int, help="Token-Limit pro Caption")
    parser.add_argument("--show-snippets", action="store_true", help="Caption-Anfang mit ausgeben")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    image_dir = Path(args.image_dir or cfg.get("data", {}).get("image_dir", "./data/images")).expanduser()
    if not image_dir.exists():
        raise FileNotFoundError(f"Bild-Ordner {image_dir} nicht gefunden")

    tokenizer_root = resolve_tokenizer_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, subfolder="tokenizer", use_fast=False)

    offenders = audit_captions(image_dir, tokenizer, args.max_tokens)
    total = sum(1 for _ in iter_images(image_dir))

    print(f"Untersuchte Bilder: {total}")
    print(f"Captions > {args.max_tokens} Tokens: {len(offenders)}\n")

    for idx, (img_path, length, caption) in enumerate(offenders, 1):
        rel = img_path.relative_to(image_dir)
        snippet = caption[:100].replace("\n", " ")
        if args.show_snippets:
            print(f"{idx:04d}. {rel} -> {length} Tokens | {snippet}")
        else:
            print(f"{idx:04d}. {rel} -> {length} Tokens")

    if offenders:
        print("\nTipp: Kürze die betreffenden .txt-Dateien oder splitte die Infos auf mehrere Captions.")
    else:
        print("Alle Captions liegen innerhalb des Limits.")


if __name__ == "__main__":
    main()
