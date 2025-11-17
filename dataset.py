from collections import Counter, defaultdict
from pathlib import Path
import random

from PIL import Image
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

    def __getitem__(self, idx):
        img_path = self.files[idx]
        caption = self._load_caption(img_path)
        caption = self._apply_caption_transforms(caption)

        target_width, target_height = self.sample_target_sizes[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((target_width, target_height), Image.BICUBIC)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image = (image * 2.0) - 1.0  # [-1, 1]

        tokens_1 = self.tok1(
            caption, truncation=True, max_length=77,
            padding="max_length", return_tensors="pt"
        )
        tokens_2 = self.tok2(
            caption, truncation=True, max_length=77,
            padding="max_length", return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids_1": tokens_1.input_ids[0],
            "attention_mask_1": tokens_1.attention_mask[0],
            "input_ids_2": tokens_2.input_ids[0],
            "attention_mask_2": tokens_2.attention_mask[0],
            "target_size": torch.tensor(
                [target_height, target_width], dtype=torch.int32
            ),
        }


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
