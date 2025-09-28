This project reuses the official PyTorch examples VAE implementation.

## PyTorch examples / VAE (BSD 3-Clause)

- Upstream repository: https://github.com/pytorch/examples/tree/main/vae
- License: BSD 3-Clause (see `examples/LICENSE` for the full text)
- Files copied verbatim: everything under `examples/` (including `examples/vae/main.py`)
- Wrapper modules authored in this repository:
	- `src/pytorch_examples/__init__.py`
	- `src/pytorch_examples/vae.py`
	- `src/run_official.py`

The wrapper modules are provided under this repository's license and simply
invoke the upstream example without modifying its source. When redistributing
any part of the upstream example, keep the original copyright and license
notices intact.
