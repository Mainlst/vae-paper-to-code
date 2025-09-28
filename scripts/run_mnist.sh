#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to launch the official PyTorch VAE example.
# You can override EPOCHS / BATCH_SIZE via environment variables.

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EXTRA_ARGS=("${@}")

python -m src.run_official \
	--epochs "${EPOCHS}" \
	--batch-size "${BATCH_SIZE}" \
	"${EXTRA_ARGS[@]}"
