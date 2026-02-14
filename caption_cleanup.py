#!/usr/bin/env python3
"""
Caption cleanup tool for SDXL datasets.

Reads captions next to images, measures SDXL TE1 token length,
and rebuilds captions so they stay within the configured limit
without blunt truncation.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm
from transformers import AutoTokenizer

from config_utils import load_config

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

PERSON_KEYWORDS = {
    "1girl",
    "1boy",
    "2girls",
    "solo",
    "woman",
    "man",
    "girl",
    "boy",
    "female",
    "male",
    "people",
    "person",
    "portrait",
    "full body",
    "upper body",
    "sitting",
    "standing",
    "posing",
    "pose",
    "looking at viewer",
    "looking away",
}

SCENE_KEYWORDS = {
    "landscape",
    "scene",
    "backdrop",
    "environment",
    "indoors",
    "outdoors",
    "forest",
    "city",
    "street",
    "beach",
    "mountain",
    "cafe",
    "bedroom",
    "room",
    "kitchen",
    "studio",
    "sunset",
    "night",
    "day",
    "snow",
    "rain",
    "sky",
}

CLOTHING_AND_BODY_KEYWORDS = {
    "dress",
    "shirt",
    "skirt",
    "pants",
    "jeans",
    "jacket",
    "coat",
    "hoodie",
    "uniform",
    "armor",
    "kimono",
    "suit",
    "tie",
    "gloves",
    "hat",
    "cap",
    "beret",
    "boots",
    "stockings",
    "tights",
    "leggings",
    "socks",
    "scarf",
    "bikini",
    "swimsuit",
    "lingerie",
    "panties",
    "bodysuit",
    "hair",
    "fringe",
    "bangs",
    "ponytail",
    "braid",
    "eyes",
    "eyeliner",
    "eyeshadow",
    "freckles",
    "blush",
    "expression",
    "smile",
    "frown",
    "open mouth",
    "closed mouth",
    "teeth",
    "fangs",
    "hand",
    "hands",
    "fingers",
    "legs",
    "thighs",
    "hips",
    "waist",
    "shoulders",
    "back",
    "muscular",
    "fit",
    "curvy",
}

STYLE_KEYWORDS = {
    "lighting",
    "light",
    "glow",
    "bloom",
    "cinematic",
    "film",
    "grain",
    "analog",
    "digital painting",
    "oil painting",
    "watercolor",
    "sketch",
    "lineart",
    "render",
    "3d",
    "octane",
    "vray",
    "style",
    "artstyle",
    "aesthetic",
    "mood",
    "ambience",
    "camera",
    "lens",
    "bokeh",
    "depth of field",
    "dof",
    "shutter",
    "aperture",
    "f/",
    "exposure",
    "hdr",
    "photography",
    "photo",
    "film grain",
    "soft light",
    "golden hour",
    "morning light",
    "studio light",
}

QUALITY_SPAM_KEYWORDS = {
    "masterpiece",
    "best quality",
    "high quality",
    "ultra quality",
    "8k",
    "16k",
    "4k",
    "highres",
    "hires",
    "hyperrealistic",
    "hyper realistic",
    "ultra detailed",
    "very detailed",
    "incredibly detailed",
    "extremely detailed",
    "amazing",
    "perfect",
    "beautiful",
    "gorgeous",
    "award winning",
    "professional",
    "sharp focus",
    "focused",
    "best",
    "top tier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prioritized caption cleanup for SDXL datasets.")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Path to config.json")
    parser.add_argument("--max_tokens", type=int, default=77, help="Maximum token budget (default: 77)")
    parser.add_argument("--dry_run", action="store_true", help="Analyze only, do not write cleaned captions")
    parser.add_argument("--verbose", action="store_true", help="Print per-file details")
    return parser.parse_args()


def normalize_caption_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", ", ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" ,|")
    return normalized


def split_into_chunks(text: str) -> list[str]:
    if not text:
        return []
    raw_chunks = re.split(r"[|,]", text)
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]


def normalize_for_match(chunk: str) -> str:
    lowered = chunk.lower().strip()
    lowered = lowered.replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s/]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def contains_keyword(chunk_norm: str, keywords: Iterable[str]) -> bool:
    for keyword in keywords:
        if keyword in chunk_norm:
            return True
    return False


def classify_chunk(chunk: str, seen: set[str]) -> int:
    cleaned = normalize_for_match(chunk)
    if not cleaned:
        return 4
    if cleaned in seen:
        return 4

    if contains_keyword(cleaned, PERSON_KEYWORDS) or contains_keyword(cleaned, SCENE_KEYWORDS):
        seen.add(cleaned)
        return 1
    if contains_keyword(cleaned, CLOTHING_AND_BODY_KEYWORDS):
        seen.add(cleaned)
        return 2
    if contains_keyword(cleaned, STYLE_KEYWORDS):
        seen.add(cleaned)
        return 3
    if contains_keyword(cleaned, QUALITY_SPAM_KEYWORDS):
        return 4

    # Default handling: treat unknown tags as mid-priority content.
    seen.add(cleaned)
    return 2


def count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return len(token_ids)


def rebuild_caption(chunks: Sequence[str], tokenizer, max_tokens: int) -> str:
    categorized: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    seen_chunks: set[str] = set()
    for chunk in chunks:
        priority = classify_chunk(chunk, seen_chunks)
        categorized.setdefault(priority, []).append(chunk)

    new_chunks: list[str] = []
    target_min = max(1, max_tokens - 3)
    target_stop_max = max(max_tokens - 1, target_min)
    current_caption = ""
    current_tokens = 0

    for priority in (1, 2, 3, 4):
        for chunk in categorized.get(priority, []):
            if priority == 4 and current_tokens >= target_min:
                continue
            candidate_caption = chunk if not new_chunks else f"{current_caption}, {chunk}"
            token_length = count_tokens(candidate_caption, tokenizer)
            if token_length > max_tokens:
                continue
            new_chunks.append(chunk)
            current_caption = candidate_caption
            current_tokens = token_length
            if target_min <= token_length <= target_stop_max:
                return current_caption

    if not new_chunks:
        return ""
    return ", ".join(new_chunks)


def fallback_trim(text: str, tokenizer, max_tokens: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    if len(token_ids) <= max_tokens:
        return text
    trimmed_ids = token_ids[: max(1, max_tokens - 1)]
    trimmed = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    return trimmed.strip()


def resolve_tokenizer_sources(cfg: dict) -> list[str | Path]:
    training_cfg = cfg.get("training") or {}
    model_cfg = cfg.get("model") or {}
    sources: list[str | Path] = []

    resume_from = training_cfg.get("resume_from")
    if resume_from:
        resume_path = Path(resume_from).expanduser()
        sources.append(resume_path if resume_path.exists() else resume_from)

    model_id = model_cfg.get("id")
    if model_id:
        model_path = Path(model_id).expanduser()
        sources.append(model_path if model_path.exists() else model_id)

    return sources


def load_tokenizer_from_sources(sources: Sequence[str | Path]):
    last_error: Exception | None = None
    for source in sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, subfolder="tokenizer")
            print(f"[Tokenizer] SDXL TE1 geladen aus: {source}")
            return tokenizer
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[Tokenizer] Konnte {source} nicht laden: {exc}", file=sys.stderr)
    raise RuntimeError("Kein gültiger Tokenizer-Pfad gefunden.") from last_error


def gather_image_files(image_dir: Path, skip_dir: Path | None = None) -> list[Path]:
    files: list[Path] = []
    for file in image_dir.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if skip_dir is not None and skip_dir in file.parents:
            continue
        files.append(file)
    files.sort()
    return files


def process_caption_file(
    image_path: Path,
    caption_path: Path,
    dest_path: Path,
    relative_label: Path,
    tokenizer,
    max_tokens: int,
    dry_run: bool,
    verbose: bool,
) -> tuple[bool, int, int]:
    if not caption_path.exists():
        if verbose:
            print(f"[Skip] Keine Caption für {image_path.name}")
        return False, 0, 0

    raw_text = caption_path.read_text(encoding="utf-8").strip()
    normalized = normalize_caption_text(raw_text)
    if not normalized:
        if verbose:
            print(f"[Skip] Leere Caption für {relative_label}")
        return False, 0, 0

    original_caption = normalized
    original_tokens = count_tokens(original_caption, tokenizer)
    if original_tokens <= max_tokens:
        output_text = original_caption
    else:
        chunks = split_into_chunks(original_caption)
        rebuilt = rebuild_caption(chunks, tokenizer, max_tokens)
        if not rebuilt:
            rebuilt = fallback_trim(original_caption, tokenizer, max_tokens)
        output_text = rebuilt.strip()

    clean_caption = output_text
    cleaned_tokens = count_tokens(clean_caption, tokenizer)
    changed = cleaned_tokens != original_tokens or clean_caption != original_caption

    if changed:
        print(f"[CHANGED] {relative_label}: {original_tokens} → {cleaned_tokens}")
    else:
        print(f"[OK] {relative_label}: {original_tokens} → {cleaned_tokens}")

    if verbose:
        print("----- ORIGINAL -----")
        print(original_caption)
        print("----- CLEANED ------")
        print(clean_caption)
        print("--------------------")

    if not dry_run:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(clean_caption + "\n", encoding="utf-8")

    return True, int(changed), cleaned_tokens


def main() -> None:
    args = parse_args()
    cfg_path = args.config.expanduser()
    cfg = load_config(cfg_path)

    image_dir = Path(cfg["data"]["image_dir"]).expanduser()
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir {image_dir} existiert nicht.")

    tokenizer_sources = resolve_tokenizer_sources(cfg)
    if not tokenizer_sources:
        raise RuntimeError("Weder training.resume_from noch model.id gesetzt – kein Tokenizer verfügbar.")
    tokenizer = load_tokenizer_from_sources(tokenizer_sources)

    clean_root = image_dir / "captions_clean"
    image_files = gather_image_files(image_dir, skip_dir=clean_root)
    if not image_files:
        print("Keine Bilder gefunden – prüfe data.image_dir.")
        return

    total = 0
    changed = 0
    written = 0

    print(
        f"Starte Caption-Cleanup ({len(image_files)} Bilder) | "
        f"Limit={args.max_tokens} Tokens | Zielordner={clean_root}"
    )
    progress = tqdm(image_files, desc="Captions", unit="img")
    for image_path in progress:
        caption_path = image_path.with_suffix(".txt")
        relative = image_path.relative_to(image_dir)
        dest_caption = clean_root / relative
        dest_caption = dest_caption.with_suffix(".txt")
        processed, is_changed, _ = process_caption_file(
            image_path,
            caption_path,
            dest_caption,
            relative,
            tokenizer,
            args.max_tokens,
            args.dry_run,
            args.verbose,
        )
        if processed:
            total += 1
            changed += is_changed
            if not args.dry_run:
                written += 1

    progress.close()

    summary = (
        f"Fertig. Dateien verarbeitet: {total}, verändert: {changed}, "
        f"{'geschrieben' if not args.dry_run else 'simuliert'}: {written if not args.dry_run else total}"
    )
    print(summary)


if __name__ == "__main__":
    main()
