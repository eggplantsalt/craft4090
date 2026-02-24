# （可选）CRaFT 训练一步到位命令清单

> 本文件为补充速查表；主文档请看 `docs/training_craft.md`。

## 单卡 3-step dry-run

```bash
export DATA_ROOT=/data/rlds
export ANCHOR_CACHE_DIR=/data/craft/hidden_cache

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_craft.py \
  --data_root_dir "$DATA_ROOT" \
  --vla.data_mix libero_90 \
  --dry_run_batches 3 \
  --craft.enabled true \
  --craft.retention_mode hidden \
  --craft.anchor_cache_dir "$ANCHOR_CACHE_DIR" \
  --craft.pooling mean_image_tokens \
  --craft.retention_loss_type mse
```

## 多卡模板

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train_craft.py \
  --data_root_dir "$DATA_ROOT" \
  --vla.data_mix libero_90 \
  --craft.anchor_cache_dir "$ANCHOR_CACHE_DIR" \
  --craft.retention_mode hidden
```
