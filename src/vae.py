from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VAEConfig:
    input_dim: int = 28 * 28
    hidden_dim: int = 400
    latent_dim: int = 20
    output_logits: bool = False  # True: Decoder outputs logits (for BCE-with-logits)


class VAE(nn.Module):
    """Minimal MLP VAE for MNIST.

    Paper ↔ Code mapping (Kingma & Welling, 2013/2014):
    - q_phi(z|x) = N(mu(x), diag(sigma^2(x))) → encode(x) → (mu, logvar)
    - Reparameterization: z = mu + sigma ⊙ eps, eps ~ N(0, I) → reparameterize(mu, logvar)
    - p_theta(x|z): Bernoulli likelihood (for BCE) or Gaussian (for MSE) → decode(z)
    - ELBO = E_q[log p(x|z)] − D_KL(q(z|x) || p(z)); we minimize −ELBO
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder: x -> h -> (mu, logvar)
        self.enc_fc1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.enc_fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.enc_fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        # Decoder: z -> h -> x_hat (logits or prob)
        self.dec_fc1 = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.dec_fc_out = nn.Linear(cfg.hidden_dim, cfg.input_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # q_phi(z|x)
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_fc1(x))
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    # Reparameterization trick
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    # p_theta(x|z)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        x_hat = self.dec_fc_out(h)
        if not self.cfg.output_logits:
            x_hat = torch.sigmoid(x_hat)  # Bernoulli prob for BCE
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def kld_standard_normal(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    D_KL( N(mu, sigma^2) || N(0, I) )
    = -0.5 * sum_i (1 + logvar_i - mu_i^2 - exp(logvar_i))
    Reduced over latent dims; then 'mean' or 'sum' over batch.
    """
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    if reduction == "mean":
        return kld_per_sample.mean()
    elif reduction == "sum":
        return kld_per_sample.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    loss_type: str = "bce",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    loss_type:
      - 'bce'        : x_hat is probability in (0,1), use F.binary_cross_entropy
      - 'bce_logits' : x_hat is logits (no sigmoid), use F.binary_cross_entropy_with_logits
      - 'mse'        : Gaussian assumption proxy (unit variance); MSE over pixels
    """
    if loss_type == "bce":
        loss = F.binary_cross_entropy(x_hat, x, reduction="none")
        loss = loss.view(loss.size(0), -1).sum(dim=1)
    elif loss_type == "bce_logits":
        loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction="none")
        loss = loss.view(loss.size(0), -1).sum(dim=1)
    elif loss_type == "mse":
        loss = F.mse_loss(x_hat, x, reduction="none")
        loss = loss.view(loss.size(0), -1).sum(dim=1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
