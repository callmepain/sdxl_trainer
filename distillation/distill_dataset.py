"""
Dataset for multi-teacher distillation training.

Combines:
- Existing latent cache (VAE-encoded images)
- Teacher prediction caches (one per teacher)

Returns batches with:
- latents: VAE-encoded image
- teacher_predictions: dict mapping teacher_id -> prediction tensor
- noise: shared noise tensor (same across all teachers for this sample)
- timestep: shared timestep
- encoder_hidden_states: cached text embeddings
- pooled_embeds: cached pooled embeddings
- target_size: resolution tuple
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from distillation.distill_utils import compute_cache_path, load_teacher_cache_entry


class DistillationDataset(Dataset):
    """
    Dataset for distillation training.

    Loads pre-computed teacher predictions and combines them with
    the existing latent cache from the base dataset.
    """

    def __init__(
        self,
        base_dataset,  # SimpleCaptionDataset instance
        teacher_configs: List[Dict],
        cache_version: str = "1",
        require_all_teachers: bool = True,
    ):
        """
        Args:
            base_dataset: SimpleCaptionDataset with latent cache enabled and activated
            teacher_configs: List of teacher config dicts:
                [{"teacher_id": ..., "cache_dir": ..., "weight": ...}, ...]
            cache_version: Cache version string for path computation
            require_all_teachers: If True, skip samples missing any teacher cache
        """
        self.base_dataset = base_dataset
        self.teacher_configs = teacher_configs
        self.cache_version = cache_version
        self.require_all_teachers = require_all_teachers

        # Validate teacher configs
        for tc in self.teacher_configs:
            if "teacher_id" not in tc:
                raise ValueError("Each teacher config must have 'teacher_id'")
            if "cache_dir" not in tc:
                raise ValueError("Each teacher config must have 'cache_dir'")

        # Build index of valid samples (those with all/any teacher caches present)
        self.valid_indices = []
        self._build_index()

    def _get_image_id(self, idx: int) -> str:
        """Get image_id for a base dataset index."""
        img_path = self.base_dataset.files[idx]
        return str(img_path.relative_to(self.base_dataset.img_dir).with_suffix(''))

    def _get_resolution(self, idx: int) -> tuple:
        """Get resolution (width, height) for a base dataset index."""
        target_size = self.base_dataset.sample_target_sizes[idx]
        # target_size is (width, height) or [width, height]
        return (target_size[0], target_size[1])

    def _build_index(self):
        """Build index of samples that have all/any required teacher caches."""
        missing_counts = {tc["teacher_id"]: 0 for tc in self.teacher_configs}

        for idx in range(len(self.base_dataset)):
            image_id = self._get_image_id(idx)
            resolution = self._get_resolution(idx)

            # Check which teacher caches exist
            existing_teachers = []
            for teacher_cfg in self.teacher_configs:
                cache_dir = Path(teacher_cfg["cache_dir"]).expanduser()
                teacher_id = teacher_cfg["teacher_id"]
                cache_path = compute_cache_path(
                    cache_dir, image_id, teacher_id, resolution, self.cache_version
                )
                if cache_path.exists():
                    existing_teachers.append(teacher_id)
                else:
                    missing_counts[teacher_id] += 1

            # Decide if this sample is valid
            if self.require_all_teachers:
                if len(existing_teachers) == len(self.teacher_configs):
                    self.valid_indices.append(idx)
            else:
                if len(existing_teachers) > 0:
                    self.valid_indices.append(idx)

        # Report statistics
        total = len(self.base_dataset)
        valid = len(self.valid_indices)
        print(f"DistillationDataset: {valid}/{total} samples have complete teacher caches")
        if valid < total:
            print("Missing cache counts per teacher:")
            for teacher_id, count in missing_counts.items():
                print(f"  {teacher_id}: {count} missing")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        base_idx = self.valid_indices[idx]

        # Get image_id and resolution
        image_id = self._get_image_id(base_idx)
        resolution = self._get_resolution(base_idx)

        # Load latent from base dataset's latent cache
        latent = self.base_dataset._load_latent(base_idx)

        # Load teacher predictions
        teacher_predictions = {}
        teacher_weights = {}
        shared_noise = None
        shared_timestep = None
        shared_encoder_hidden_states = None
        shared_pooled_embeds = None

        for teacher_cfg in self.teacher_configs:
            cache_dir = Path(teacher_cfg["cache_dir"]).expanduser()
            teacher_id = teacher_cfg["teacher_id"]
            weight = float(teacher_cfg.get("weight", 1.0))

            cache_path = compute_cache_path(
                cache_dir, image_id, teacher_id, resolution, self.cache_version
            )

            if cache_path.exists():
                entry = load_teacher_cache_entry(cache_path)
                teacher_predictions[teacher_id] = entry["teacher_pred"]
                teacher_weights[teacher_id] = weight

                # Use first teacher's noise/timestep/embeddings as shared values
                # (they should be identical across teachers for same image_id)
                if shared_noise is None:
                    shared_noise = entry["noise"]
                    shared_timestep = entry["timestep"]
                    shared_encoder_hidden_states = entry["encoder_hidden_states"]
                    shared_pooled_embeds = entry["pooled_embeds"]

        # Build return dict
        # target_size is [height, width] to match base dataset convention
        target_size = torch.tensor(
            [resolution[1], resolution[0]], dtype=torch.int32
        )

        return {
            "latents": latent,
            "teacher_predictions": teacher_predictions,
            "teacher_weights": teacher_weights,
            "noise": shared_noise,
            "timestep": shared_timestep,
            "encoder_hidden_states": shared_encoder_hidden_states,
            "pooled_embeds": shared_pooled_embeds,
            "target_size": target_size,
            "image_id": image_id,
        }


def distillation_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DistillationDataset.

    Handles the nested teacher_predictions dict structure.
    """
    # Stack simple tensors
    latents = torch.stack([b["latents"] for b in batch])
    noise = torch.stack([b["noise"] for b in batch])
    timesteps = torch.tensor([b["timestep"] for b in batch], dtype=torch.long)
    encoder_hidden_states = torch.stack([b["encoder_hidden_states"] for b in batch])
    pooled_embeds = torch.stack([b["pooled_embeds"] for b in batch])
    target_sizes = torch.stack([b["target_size"] for b in batch])
    image_ids = [b["image_id"] for b in batch]

    # Stack teacher predictions per teacher
    # All samples in batch should have the same teachers (due to require_all_teachers)
    teacher_ids = list(batch[0]["teacher_predictions"].keys())
    teacher_predictions = {}
    teacher_weights = {}
    for teacher_id in teacher_ids:
        teacher_predictions[teacher_id] = torch.stack([
            b["teacher_predictions"][teacher_id] for b in batch
        ])
        teacher_weights[teacher_id] = batch[0]["teacher_weights"][teacher_id]

    return {
        "latents": latents,
        "teacher_predictions": teacher_predictions,
        "teacher_weights": teacher_weights,
        "noise": noise,
        "timesteps": timesteps,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_embeds": pooled_embeds,
        "target_sizes": target_sizes,
        "image_ids": image_ids,
    }
