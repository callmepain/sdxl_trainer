#!/usr/bin/env python3
"""
Verify teacher prediction cache integrity and completeness.

Usage:
    python distillation/verify_cache.py --cache-dir ./cache/teacher_predictions/my_teacher
    python distillation/verify_cache.py --cache-dir ./cache/teacher_predictions/my_teacher --verbose

Checks:
- All cache files are loadable
- Required keys are present
- No NaN/Inf values in tensors
- Correct tensor shapes
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from distillation.distill_utils import validate_cache_entry


def verify_cache_entry(cache_path: Path) -> Tuple[bool, str]:
    """
    Verify a single cache entry.

    Returns:
        (is_valid, error_message)
    """
    if not cache_path.exists():
        return False, "File does not exist"

    try:
        data = load_file(str(cache_path), device="cpu")
    except Exception as e:
        return False, f"Failed to load: {e}"

    # Convert to expected format for validation
    try:
        entry = {
            "teacher_pred": data.get("teacher_pred"),
            "noise": data.get("noise"),
            "timestep": int(data["timestep"].item()) if "timestep" in data else None,
            "seed": int(data["seed"].item()) if "seed" in data else None,
            "resolution": (
                tuple(data["resolution"].tolist()) if "resolution" in data else None
            ),
            "encoder_hidden_states": data.get("encoder_hidden_states"),
            "pooled_embeds": data.get("pooled_embeds"),
        }
    except Exception as e:
        return False, f"Failed to parse data: {e}"

    return validate_cache_entry(entry)


def verify_cache_directory(cache_dir: Path, verbose: bool = False):
    """Verify all cache files in a directory."""
    if not cache_dir.exists():
        print(f"Error: Cache directory does not exist: {cache_dir}")
        return False

    cache_files = list(cache_dir.glob("*.safetensors"))
    if not cache_files:
        print(f"No cache files found in {cache_dir}")
        return False

    print(f"Verifying {len(cache_files)} cache files in {cache_dir}")

    valid_count = 0
    invalid_count = 0
    errors = []

    # Sample some files for detailed shape info
    sample_shapes = {}

    for cache_path in tqdm(cache_files, desc="Verifying"):
        is_valid, error_msg = verify_cache_entry(cache_path)
        if is_valid:
            valid_count += 1
            # Collect shape info from first valid file
            if not sample_shapes:
                try:
                    data = load_file(str(cache_path), device="cpu")
                    sample_shapes = {k: tuple(v.shape) for k, v in data.items()}
                except Exception:
                    pass
        else:
            invalid_count += 1
            errors.append((cache_path.name, error_msg))
            if verbose:
                print(f"  Invalid: {cache_path.name} - {error_msg}")

    print(f"\n{'=' * 60}")
    print(f"Verification Results")
    print(f"{'=' * 60}")
    print(f"Total files:  {len(cache_files)}")
    print(f"Valid:        {valid_count}")
    print(f"Invalid:      {invalid_count}")

    if sample_shapes:
        print(f"\nSample tensor shapes:")
        for key, shape in sample_shapes.items():
            print(f"  {key}: {shape}")

    if errors:
        print(f"\nErrors ({min(len(errors), 10)} shown):")
        for name, msg in errors[:10]:
            print(f"  {name}: {msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return invalid_count == 0


def verify_multiple_teachers(
    cache_base_dir: Path,
    teacher_ids: list,
    verbose: bool = False,
):
    """Verify caches for multiple teachers."""
    print(f"Verifying {len(teacher_ids)} teachers in {cache_base_dir}")

    all_valid = True
    for teacher_id in teacher_ids:
        print(f"\n{'=' * 60}")
        print(f"Teacher: {teacher_id}")
        print(f"{'=' * 60}")
        cache_dir = cache_base_dir / teacher_id
        if not verify_cache_directory(cache_dir, verbose):
            all_valid = False

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Verify teacher prediction cache integrity"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Path to teacher cache directory (e.g., ./cache/teacher_predictions/my_teacher)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all errors (not just first 10)",
    )
    parser.add_argument(
        "--teachers",
        type=str,
        nargs="+",
        help="If cache-dir is base directory, specify teacher IDs to verify",
    )
    args = parser.parse_args()

    if args.teachers:
        # Verify multiple teachers
        success = verify_multiple_teachers(args.cache_dir, args.teachers, args.verbose)
    else:
        # Verify single directory
        success = verify_cache_directory(args.cache_dir, args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
