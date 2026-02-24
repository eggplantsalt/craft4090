#!/usr/bin/env bash

set -euo pipefail

# Minimal CRaFT dry-run (3 optimizer steps) with hidden retention.
# Usage:
#   bash vla-scripts/train_craft.sh <DATA_ROOT_DIR> <ANCHOR_CACHE_DIR> [BASE_VLM_ID]

DATA_ROOT_DIR="${1:-datasets/open-x-embodiment}"
ANCHOR_CACHE_DIR="${2:-cache/craft/hidden}"
BASE_VLM_ID="${3:-openvla-7b+prismatic}"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_craft.py \
  --data_root_dir "$DATA_ROOT_DIR" \
  --vla.base_vlm "$BASE_VLM_ID" \
  --vla.data_mix libero_90 \
  --dry_run_batches 3 \
  --craft.enabled true \
  --craft.retention_mode hidden \
  --craft.pooling mean_image_tokens \
  --craft.retention_loss_type mse \
  --craft.anchor_cache_dir "$ANCHOR_CACHE_DIR" \
  --craft.anchor_batch_size 4 \
  --craft.lambda_init 0.0 \
  --craft.lambda_lr 1e-4 \
  --craft.epsilon 0.05
