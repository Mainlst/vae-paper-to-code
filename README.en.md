# VAE: Paper ↔ Code Alignment (PyTorch examples/vae × Kingma & Welling 2013/2014)

日本語版: [README.md](./README.md)

This repository provides a minimal, readable implementation to align the original VAE paper (Kingma & Welling, 2013/2014) with PyTorch's `examples/vae` line by line.

- Paper: Auto-Encoding Variational Bayes (arXiv:1312.6114)
- Reference: https://github.com/pytorch/examples/tree/main/vae

## What's inside
- MLP VAE (MNIST)
- Loss options: BCE / BCE-with-logits / MSE
- β-VAE and KL annealing (linear / cyclic)
- Training curves (ELBO terms: recon, kl, total) exported as CSV/PNG
- Save reconstructions, random samples, and latent traversals (when z=2)

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Structured outputs (recommended): reports/<group>/<name>/...
python -m src.train \
  --epochs 20 --batch-size 128 --latent-dim 20 \
  --loss bce --beta 1.0 --beta-schedule linear \
  --lr 1e-3 --device auto --seed 42 \
  --project-dir reports --group mnist-bench --name test-run

# Compatibility mode: if --save-dir is explicitly set, write directly to that path
python -m src.train --epochs 5 --save-dir reports_legacy
```

## Outputs
Recommended structured layout:

```
reports/
  <group>/
    latest -> ./<name>          # symlink to the latest run (or latest.txt if symlink fails)
    <name>/
      run_meta.json             # snapshot of all args and config
      vae_epoch_0000.pt         # checkpoints per epoch
      curves/
        train_log.csv           # epoch, beta, recon, kl, total
        losses.png              # quick plot
      reconstructions/epoch_XXXX.png
      samples/epoch_XXXX.png
      traversals/epoch_XXXX.png # when z=2
```

## Paper ↔ Code quick map
- Approximate posterior q_φ(z|x)=N(μ,diag(σ²)) → `VAE.encode` (`mu, logvar`)
- Reparameterization z=μ+σ⊙ε, ε~N(0,I) → `VAE.reparameterize`
- Likelihood p_θ(x|z) (Bernoulli) → `loss=bce` (with `Sigmoid` outputs)
- ELBO = recon − KL → implemented as `total = recon + beta * kl` (minimize = −ELBO)

## Updates
- 2025-08-31: Introduced structured experiment directories (group/name) for repeatable runs.
  - New CLI options: `--project-dir`, `--group`, `--name`
  - Save run metadata `run_meta.json`; maintain `<group>/latest` symlink
  - Added grid runner example `scripts/run_grid.sh`

## Next steps
- ConvVAE (CIFAR-10) / Gaussian likelihood (MSE) comparison
- IWAE (importance weighted bound)
- Techniques against posterior collapse (β-annealing, free bits, etc.)
