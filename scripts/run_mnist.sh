#!/usr/bin/env bash
set -euo pipefail

python -m src.train       --epochs 20       --batch-size 128       --latent-dim 20       --loss bce       --beta 1.0       --beta-schedule linear       --lr 1e-3       --device auto       --seed 42       --save-dir reports
