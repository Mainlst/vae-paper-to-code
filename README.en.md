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
# A date-serial prefix (YYYYMMDD-XXX-) is automatically added to the run name
# (e.g., 20250831-001-test-run or 20250831-001-seed42). It auto-increments to avoid collisions.
python -m src.train \
  --epochs 20 --batch-size 128 --latent-dim 20 \
  --loss bce --beta 1.0 --beta-schedule linear \
  --lr 1e-3 --device auto --seed 42 \
  --project-dir reports --group mnist-bench --name test-run

# Compatibility mode: if --save-dir is explicitly set, write directly to that path
python -m src.train --epochs 5 --save-dir reports_legacy
```

### Checkpoint saving policy
By default, weights (.pt) are not saved. Use `--save-weights` to enable checkpointing per epoch.

Related options:
- `--save-weights` … save `vae_epoch_XXXX.pt` per epoch
- `--no-date-prefix` … disable auto date-serial prefixing of run names

## Outputs
Recommended structured layout:

```
reports/
  <group>/
    latest -> ./<name>          # symlink to the latest run (or latest.txt if symlink fails)
    <name>/
      run_meta.json             # snapshot of all args and config
  vae_epoch_0000.pt         # only when --save-weights is provided
      curves/
        train_log.csv           # epoch, beta, recon, kl, total
        losses.png              # quick plot
      reconstructions/epoch_XXXX.png
      samples/epoch_XXXX.png
      traversals/epoch_XXXX.png # when z=2
```

### Weights & Biases (optional)
You can log metrics and images to [Weights & Biases](https://wandb.ai/).

Key options:
- `--wandb` … enable switch
- `--wandb-project` … project name (default: `vae-paper-to-code`)
- `--wandb-entity` … team/user (optional)
- `--wandb-mode {online,offline,disabled}` … default `disabled`
- `--wandb-run-name` … override run name for W&B (defaults to `--name`)
- `--wandb-tags` … comma-separated tags

Set your API key (once):

```bash
export WANDB_API_KEY=<your_api_key>
```

Example:

```bash
python -m src.train \
  --epochs 10 --batch-size 128 --latent-dim 20 \
  --loss bce --beta 1.0 --beta-schedule linear \
  --project-dir reports --group mnist-bench --name test-run \
  --wandb --wandb-mode online --wandb-project vae-paper-to-code \
  --wandb-tags mnist,mlp,beta1
```

Logged items:
- Scalars: `loss/recon`, `loss/kl`, `loss/total`, `beta`, `epoch`
- Images: per-epoch `reconstructions`, `samples` (and `traversal` when `latent_dim==2`)

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
