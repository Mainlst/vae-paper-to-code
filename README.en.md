# VAE: Paper ↔ Code Alignment (PyTorch examples/vae × Kingma & Welling 2013/2014)

日本語版: [README.md](./README.md)

This repository now focuses on running and inspecting PyTorch's official `examples/vae` implementation while keeping close ties to the VAE paper (Kingma & Welling, 2013/2014).

- Paper: Auto-Encoding Variational Bayes (arXiv:1312.6114)
- Reference: https://github.com/pytorch/examples/tree/main/vae

## What's inside
- Wrapper to launch the official PyTorch VAE implementation (`examples/vae/main.py`) as-is
- Notebook `notebooks/vae_workflow.ipynb` with a step-by-step checklist around the official example
- Helper utilities to inspect the generated images under `examples/vae/results/`
- Montage tool for samples (`scripts/make_samples_montage.py`)

## Repository layout (main entry points)
- `examples/` – Official PyTorch example code (BSD 3-Clause). You can run `examples/vae/main.py` directly.
- `src/run_official.py` – CLI wrapper to execute the upstream example.
- `src/pytorch_examples/` – Helper utilities for reusing the official code without modifying it.
- `reports/` – Destination for experiment logs and generated artifacts.
- `notebooks/vae_workflow.ipynb` – Hands-on notebook for the official workflow and visualizations.

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run the official PyTorch VAE

```bash
python -m src.run_official -- --epochs 5 --batch-size 64
```

- `src/run_official.py` launches `examples/vae/main.py` in-process.
- When `./data` exists, a symlink to `examples/data` is created automatically (otherwise the example downloads into `examples/data`).
- Outputs are saved under `examples/vae/results/`, matching the upstream layout.
- Extra arguments for the official script pass through unchanged (e.g., `--no-accel`).

### Follow the checklist notebook

Open `notebooks/vae_workflow.ipynb` and run the cells from top to bottom to:

- Verify the environment and download MNIST
- Preview the official script and choose hyperparameters
- Launch the official example and inspect the generated images
- Optionally loop over multiple configurations

### Organize generated results

By default the official script creates PNG files under `examples/vae/results/`:

```
examples/vae/results/
    reconstruction_<epoch>.png
    sample_<epoch>.png
```

Use `scripts/make_samples_montage.py` if you want to stitch multiple samples together.

## Outputs
By default the official script stores PNG files per epoch:

```
examples/vae/results/
    reconstruction_<epoch>.png
    sample_<epoch>.png
```

The notebook automatically detects these files and plots both the latest images and progression snapshots.

## Paper ↔ Code quick map
Pointers into the official implementation (`examples/vae/main.py`):

- Approximate posterior q_φ(z|x)=N(μ,diag(σ²)) → `VAE.encode`
- Reparameterization z=μ+σ⊙ε, ε~N(0,I) → `VAE.reparameterize`
- Likelihood p_θ(x|z) (Bernoulli) → `loss_function`'s BCE term
- ELBO = recon − KL → `loss_function` returning the sum of reconstruction and KL terms

## Updates
- 2025-09-28: Retired the custom trainer (`src.train`) and focused the workflow on the official PyTorch example and helper notebook.
- 2025-08-31: Introduced structured experiment directories (group/name) for repeatable runs (legacy content retained under `reports/`).

## License & Attribution
- Code under `examples/` comes from the official PyTorch repository and remains under the BSD 3-Clause License (`examples/LICENSE`).
- Core project code (e.g., `src/`) is distributed under `LICENSE` (MIT License).
- Wrapper utilities (`src/pytorch_examples/*`, `src/run_official.py`) are provided under this repository's license.

## Next steps
- Add lightweight logging (CSV, checkpointing) around the official script if you need metrics over time
- Extend to ConvVAE / other datasets (e.g., CIFAR-10)
- Experiment with improvements mentioned in the literature (β-annealing, IWAE, free bits, etc.)

### Tips: make a montage of samples
Combine `examples/vae/results/sample_*.png` into a single image:

```bash
python scripts/make_samples_montage.py \
    --input-dir examples/vae/results \
    --pattern "sample_*.png" \
    --output examples/vae/results/samples_montage.png \
    --cols 5 --stride 1 --font-size 14
```
