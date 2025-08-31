#!/usr/bin/env bash
set -euo pipefail

# Grid search examples: vary latent dim and beta schedule
PROJECT_DIR=${1:-reports}
GROUP=${2:-mnist-grid}
SEED=${SEED:-42}

latents=(2 8 20)
losses=(bce bce_logits mse)
schedules=(none linear cyclic)

for z in "${latents[@]}"; do
  for loss in "${losses[@]}"; do
    for sched in "${schedules[@]}"; do
      NAME="z${z}-${loss}-${sched}-$(date +%Y%m%d-%H%M%S)-seed${SEED}"
      echo "Running: z=${z}, loss=${loss}, sched=${sched} -> ${PROJECT_DIR}/${GROUP}/${NAME}"
      python -m src.train \
        --epochs 20 \
        --batch-size 128 \
        --latent-dim "${z}" \
        --loss "${loss}" \
        --beta 1.0 \
        --beta-schedule "${sched}" \
        --lr 1e-3 \
        --device auto \
        --seed "${SEED}" \
        --project-dir "${PROJECT_DIR}" \
        --group "${GROUP}" \
        --name "${NAME}"
    done
  done
done
