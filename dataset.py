from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
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
    ):
        self.img_dir = Path(img_dir)
        self.size = size
        self.files = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
        ])
        self.tok1 = tokenizer_1
        self.tok2 = tokenizer_2
        self.caption_dropout_prob = max(0.0, float(caption_dropout_prob or 0.0))
        self.caption_shuffle_prob = max(0.0, float(caption_shuffle_prob or 0.0))
        self.caption_shuffle_separator = caption_shuffle_separator or ","
        self.caption_shuffle_min_tokens = max(1, int(caption_shuffle_min_tokens or 1))

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

    def __getitem__(self, idx):
        img_path = self.files[idx]
        caption = self._load_caption(img_path)
        caption = self._apply_caption_transforms(caption)

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.BICUBIC)
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
        }
