from __future__ import annotations
import os
import math
from typing import Tuple

import torch
import numpy as np


def _to_uint8_img(t: torch.Tensor) -> np.ndarray:
    # t: (C,H,W) in [0,1] or (H,W)
    t = t.detach().cpu()
    if t.dim() == 3 and t.size(0) in (1, 3):
        c, h, w = t.shape
        t = t.clamp(0, 1)
        arr = (t * 255.0 + 0.5).to(torch.uint8)
        if c == 1:
            return arr.squeeze(0).numpy()
        return arr.permute(1, 2, 0).numpy()
    elif t.dim() == 2:
        arr = (t.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
        return arr.numpy()
    else:
        raise ValueError("Expected (C,H,W) or (H,W) tensor")


def save_image(batch: torch.Tensor, save_path: str, nrow: int = 8, padding: int = 2) -> None:
    """Minimal save_image replacement avoiding torchvision.

    - batch: (N,C,H,W) in [0,1]
    - saves a tiled PNG grid.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if batch.dim() != 4:
        raise ValueError("batch must be 4D (N,C,H,W)")
    n, c, h, w = batch.shape
    ncol = max(1, nrow)
    nrow_count = math.ceil(n / ncol)

    grid_h = nrow_count * h + padding * (nrow_count - 1)
    grid_w = ncol * w + padding * (ncol - 1)
    grid = torch.zeros(c, grid_h, grid_w, dtype=batch.dtype)

    for idx in range(n):
        r = idx // ncol
        cidx = idx % ncol
        y = r * (h + padding)
        x = cidx * (w + padding)
        grid[:, y:y+h, x:x+w] = batch[idx]

    img_arr = _to_uint8_img(grid)
    try:
        from PIL import Image  # type: ignore
        img = Image.fromarray(img_arr)
        img.save(save_path)
    except Exception:
        # Fallback to matplotlib
        import matplotlib.pyplot as plt  # type: ignore
        plt.imsave(save_path, img_arr, cmap='gray' if img_arr.ndim == 2 else None)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_reconstructions(x, x_hat, save_path: str, n: int = 8) -> None:
    """Save a grid comparing input vs reconstruction."""
    ensure_dir(os.path.dirname(save_path))
    x = x[:n].cpu()
    x_hat = x_hat[:n].cpu()
    interleaved = torch.stack([x, x_hat], dim=1).view(-1, *x.shape[1:])
    save_image(interleaved, save_path, nrow=2, padding=2)


def save_samples(samples: torch.Tensor, save_path: str, nrow: int = 8) -> None:
    ensure_dir(os.path.dirname(save_path))
    save_image(samples.cpu(), save_path, nrow=nrow, padding=2)


def save_traversal(decoder, device, latent_dim: int, save_path: str, steps: int = 9, span: float = 3.0) -> None:
    """For latent_dim==2: traverse a grid over z1,z2 in [-span, span]."""
    if latent_dim != 2:
        return
    ensure_dir(os.path.dirname(save_path))
    zs = torch.linspace(-span, span, steps, device=device)
    grid_rows = []
    for a in zs:
        row_imgs = []
        for b in zs:
            z = torch.tensor([[a, b]], device=device, dtype=torch.float32)
            x_hat = decoder(z)
            if x_hat.dim() == 2:
                x_hat = x_hat.view(-1, 1, 28, 28)
            else:
                x_hat = x_hat
            # if logits likely present, squash for viewing
            if x_hat.min() < 0 or x_hat.max() > 1:
                x_hat = torch.sigmoid(x_hat)
            row_imgs.append(x_hat)
        grid_rows.append(torch.cat(row_imgs, dim=0))
    big = torch.cat(grid_rows, dim=0)
    save_image(big.cpu(), save_path, nrow=steps, padding=1)
