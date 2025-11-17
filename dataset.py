from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random

from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset, Sampler
import numpy as np
import torch


class SimpleCaptionDataset(Dataset):
    def __init__(
        self,
        img_dir,
        tokenizer_1,
        tokenizer_2,
        size=1024,
        caption_dropout_prob=0.0,
        caption_shuffle_prob=0.0,
        caption_shuffle_separator=",",
        caption_shuffle_min_tokens=2,
        bucket_config=None,
        latent_cache_config=None,
    ):
        self.img_dir = Path(img_dir)
        self.size = size
        self.files = sorted(
            [
                p
                for p in self.img_dir.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
            ]
        )
        self.tok1 = tokenizer_1
        self.tok2 = tokenizer_2
        self.caption_dropout_prob = max(0.0, float(caption_dropout_prob or 0.0))
        self.caption_shuffle_prob = max(0.0, float(caption_shuffle_prob or 0.0))
        self.caption_shuffle_separator = caption_shuffle_separator or ","
        self.caption_shuffle_min_tokens = max(
            1, int(caption_shuffle_min_tokens or 1)
        )
        self.bucket_config = bucket_config or {}
        self.use_buckets = bool(self.bucket_config.get("enabled", False))
        if self.use_buckets:
            raw_resolutions = self.bucket_config.get("resolutions") or []
            self.target_resolutions = self._prepare_resolutions(raw_resolutions)
        else:
            self.target_resolutions = [(self.size, self.size)]
        self.sample_target_sizes = []
        self.sample_buckets = []
        self._prepare_samples()
        self.latent_cache_cfg = latent_cache_config or {}
        self.latent_cache_enabled = bool(self.latent_cache_cfg.get("enabled", False))
        self.latent_cache_active = False
        self.latent_cache_dir = None
        self.latent_paths = []
        self.latent_exists = []
        if self.latent_cache_enabled:
            cache_dir = self.latent_cache_cfg.get("cache_dir") or (
                self.img_dir / "latents"
            )
            self.latent_cache_dir = Path(cache_dir).expanduser()
            self.latent_cache_dir.mkdir(parents=True, exist_ok=True)
            self._initialize_latent_cache_index()

    def __len__(self):
        return len(self.files)

    def _load_caption(self, path: Path) -> str:
        txt = path.with_suffix(".txt")
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
        return ""  # notfalls leer

    def _apply_caption_transforms(self, caption: str) -> str:
        if not caption:
            return caption

        if self.caption_dropout_prob > 0 and random.random() < self.caption_dropout_prob:
            return ""

        if self.caption_shuffle_prob > 0 and random.random() < self.caption_shuffle_prob:
            tokens = [tok.strip() for tok in caption.split(self.caption_shuffle_separator) if tok.strip()]
            if len(tokens) >= self.caption_shuffle_min_tokens:
                random.shuffle(tokens)
                joiner = self.caption_shuffle_separator if self.caption_shuffle_separator else ", "
                caption = joiner.join(tokens)
        return caption

    def _prepare_resolutions(self, raw_resolutions):
        resolutions = []
        for item in raw_resolutions:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            w, h = int(item[0]), int(item[1])
            if w > 0 and h > 0:
                resolutions.append((w, h))
        if not resolutions:
            resolutions = [(self.size, self.size)]
        divisible = int(self.bucket_config.get("divisible_by", 64))
        aligned = []
        for w, h in resolutions:
            aligned.append((self._align_to_multiple(w, divisible), self._align_to_multiple(h, divisible)))
        return aligned

    def _align_to_multiple(self, value, multiple):
        if multiple <= 0:
            return value
        return int(round(value / multiple) * multiple)

    def _prepare_samples(self):
        if not self.use_buckets:
            for _ in self.files:
                self.sample_target_sizes.append((self.size, self.size))
                self.sample_buckets.append(self._bucket_key((self.size, self.size)))
            return

        for path in self.files:
            try:
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:
                width = height = self.size
            bucket_size = self._choose_bucket(width, height)
            self.sample_target_sizes.append(bucket_size)
            self.sample_buckets.append(self._bucket_key(bucket_size))
        if self.latent_cache_enabled:
            self._initialize_latent_cache_index()

    def _initialize_latent_cache_index(self):
        self.latent_paths = []
        self.latent_exists = []
        for idx in range(len(self.files)):
            path = self._latent_path_for_index(idx)
            self.latent_paths.append(path)
            self.latent_exists.append(path.exists())

    def _latent_path_for_index(self, idx):
        if not self.latent_cache_dir:
            raise ValueError("Latent cache directory is not initialized.")
        rel = self.files[idx].relative_to(self.img_dir)
        size_w, size_h = self.sample_target_sizes[idx]
        key = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()
        filename = f"{key}_{size_w}x{size_h}.safetensors"
        return self.latent_cache_dir / filename

    def refresh_latent_cache_state(self):
        if not self.latent_cache_enabled:
            return
        if not self.latent_paths:
            self._initialize_latent_cache_index()
            return
        for idx, path in enumerate(self.latent_paths):
            self.latent_exists[idx] = path.exists()

    def get_missing_latent_indices(self):
        if not self.latent_cache_enabled:
            return []
        return [idx for idx, exists in enumerate(self.latent_exists) if not exists]

    def save_latent(self, idx, latent_tensor: torch.Tensor):
        if not self.latent_cache_enabled:
            return
        path = self.latent_paths[idx]
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"latent": latent_tensor.detach().cpu()}, str(path))
        self.latent_exists[idx] = True

    def _load_latent(self, idx):
        if not self.latent_cache_enabled:
            raise ValueError("Latent cache is not enabled.")
        if not self.latent_exists[idx]:
            raise FileNotFoundError(
                f"Latent file fehlt für Index {idx}: {self.latent_paths[idx]}"
            )
        data = load_file(str(self.latent_paths[idx]), device="cpu")
        return data["latent"]

    def load_image_tensor_for_cache(self, idx):
        img_path = self.files[idx]
        target_width, target_height = self.sample_target_sizes[idx]
        return self._load_image_tensor(img_path, (target_width, target_height))

    def activate_latent_cache(self):
        if not self.latent_cache_enabled:
            return
        missing = self.get_missing_latent_indices()
        if missing:
            raise ValueError(
                f"Latent-Cache unvollständig, fehlende Einträge: {len(missing)}"
            )
        self.latent_cache_active = True

    def _choose_bucket(self, width, height):
        aspect = width / max(height, 1)
        best = None
        best_diff = None
        for w, h in self.target_resolutions:
            bucket_aspect = w / h
            diff = abs(aspect - bucket_aspect)
            if best is None or diff < best_diff:
                best = (w, h)
                best_diff = diff
        return best or (self.size, self.size)

    def _bucket_key(self, size_tuple):
        w, h = size_tuple
        return f"{w}x{h}"

    def _load_image_tensor(self, img_path: Path, target_size):
        target_width, target_height = target_size
        image = Image.open(img_path).convert("RGB")
        image = image.resize((target_width, target_height), Image.BICUBIC)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image = (image * 2.0) - 1.0  # [-1, 1]
        return image

    def __getitem__(self, idx):
        img_path = self.files[idx]
        caption = self._load_caption(img_path)
        caption = self._apply_caption_transforms(caption)

        target_width, target_height = self.sample_target_sizes[idx]

        tokens_1 = self.tok1(
            caption, truncation=True, max_length=77,
            padding="max_length", return_tensors="pt"
        )
        tokens_2 = self.tok2(
            caption, truncation=True, max_length=77,
            padding="max_length", return_tensors="pt"
        )

        sample = {
            "input_ids_1": tokens_1.input_ids[0],
            "attention_mask_1": tokens_1.attention_mask[0],
            "input_ids_2": tokens_2.input_ids[0],
            "attention_mask_2": tokens_2.attention_mask[0],
            "target_size": torch.tensor(
                [target_height, target_width], dtype=torch.int32
            ),
        }

        if self.latent_cache_active:
            sample["latents"] = self._load_latent(idx)
        else:
            image = self._load_image_tensor(img_path, (target_width, target_height))
            sample["pixel_values"] = image

        return sample


class BucketBatchSampler(Sampler):
    def __init__(
        self,
        bucket_ids,
        batch_size,
        shuffle=True,
        drop_last=True,
    ):
        self.bucket_ids = list(bucket_ids)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive for BucketBatchSampler.")

    def __iter__(self):
        bucket_to_indices = defaultdict(list)
        for idx, bucket in enumerate(self.bucket_ids):
            bucket_to_indices[bucket].append(idx)

        batches = []
        bucket_keys = list(bucket_to_indices.keys())
        if self.shuffle:
            random.shuffle(bucket_keys)

        for bucket_key in bucket_keys:
            indices = bucket_to_indices[bucket_key]
            if self.shuffle:
                random.shuffle(indices)
            full_chunks = len(indices) // self.batch_size
            for i in range(full_chunks):
                batch = indices[i * self.batch_size : (i + 1) * self.batch_size]
                batches.append(batch)
            remainder = len(indices) % self.batch_size
            if not self.drop_last and remainder:
                batches.append(indices[-remainder:])

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        counter = Counter(self.bucket_ids)
        total = 0
        for count in counter.values():
            full = count // self.batch_size
            total += full
            if not self.drop_last and (count % self.batch_size):
                total += 1
        return total
