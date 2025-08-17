from __future__ import annotations
import os
import torch
from torchvision.utils import save_image


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
