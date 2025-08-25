from __future__ import annotations
import os
import gzip
import struct
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _open_maybe_gzip(path: str):
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    return open(path, 'rb')


def _read_idx_images(path: str) -> torch.Tensor:
    with _open_maybe_gzip(path) as f:
        data = f.read()
    magic, num, rows, cols = struct.unpack_from('>IIII', data, 0)
    if magic != 2051:
        raise ValueError(f"Invalid magic for images: {magic}")
    offset = 16
    arr = np.frombuffer(data, dtype=np.uint8, offset=offset)
    arr = arr.reshape(num, rows, cols)
    tens = torch.from_numpy(arr)
    return tens


def _read_idx_labels(path: str) -> torch.Tensor:
    with _open_maybe_gzip(path) as f:
        data = f.read()
    magic, num = struct.unpack_from('>II', data, 0)
    if magic != 2049:
        raise ValueError(f"Invalid magic for labels: {magic}")
    offset = 8
    arr = np.frombuffer(data, dtype=np.uint8, offset=offset)
    tens = torch.from_numpy(arr)
    return tens


def _prefer_uncompressed(root: str, name: str) -> str:
    # Prefer uncompressed if present, else .gz
    raw = os.path.join(root, name)
    gz = raw + '.gz'
    if os.path.exists(raw):
        return raw
    if os.path.exists(gz):
        return gz
    raise FileNotFoundError(f"Missing MNIST file: {raw}(.gz)")


class MNISTLocal(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        split = 'train' if train else 't10k'
        base = os.path.join(root, 'MNIST', 'raw')
        img_path = _prefer_uncompressed(base, f"{split}-images-idx3-ubyte")
        lbl_path = _prefer_uncompressed(base, f"{split}-labels-idx1-ubyte")

        images = _read_idx_images(img_path)  # (N, 28, 28) uint8
        labels = _read_idx_labels(lbl_path)  # (N,) uint8
        if images.shape[0] != labels.shape[0]:
            raise ValueError("Image/label count mismatch")

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]  # (28,28) uint8
        lbl = int(self.labels[idx].item())
        # ToTensor equivalent: float32 [0,1] and add channel dim
        img = img.to(torch.float32).div_(255.0).unsqueeze(0)
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl

