#!/usr/bin/env bash
set -euo pipefail

# 推奨: 構造化出力（reports/<group>/<name>）
python -m src.train \
	--epochs 20 \
	--batch-size 128 \
	--latent-dim 20 \
	--loss bce \
	--beta 1.0 \
	--beta-schedule linear \
	--lr 1e-3 \
	--device auto \
	--seed 42 \
	--project-dir reports \
	--group mnist-default \
	--name "$(date +%Y%m%d-%H%M%S)-seed42"
