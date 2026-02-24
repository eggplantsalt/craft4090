# CRaFT 训练指南（OpenVLA 基座）

脚本入口：`vla-scripts/train_craft.py`

## 1. 训练目标与更新规则

- `L_act`：沿用 baseline 行为克隆 loss（不改定义）
- `L_ret`：hidden retention loss（从离线 anchor cache 读取 target feature）
- 约束优化：
  - 预算：$L_{ret} \le \epsilon$
  - 对偶：$\lambda \leftarrow [\lambda + \eta_\lambda (L_{ret}-\epsilon)]_+$

梯度更新（当前实现）流程：

1. 对 `L_act` backward，得到 `g_act`
2. 对 `L_ret` backward，得到 `g_ret`
3. 计算 `dot(g_act, g_ret)` 与 `cos`
4. 若 `dot < 0`，投影 `g_act` 到 `g_ret` 正交空间
5. 合并梯度：`g = g_act_proj + λ * g_ret`
6. 写回 `param.grad`，`optimizer.step()`
7. 更新 `lambda`

## 2. 必改参数（最少）

```bash
--data_root_dir <RLDS_ROOT>
--vla.data_mix libero_90
--craft.anchor_cache_dir <ANCHOR_CACHE_DIR>
```

建议同时显式设置：

```bash
--craft.retention_mode hidden
--craft.pooling mean_image_tokens
--craft.retention_loss_type mse
--craft.hidden_layer -2
--craft.epsilon 0.05
--craft.lambda_init 0.0
--craft.lambda_lr 1e-4
--dry_run_batches 3
```

## 3. 单机单卡命令模板

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_craft.py \
  --data_root_dir /data/rlds \
  --vla.data_mix libero_90 \
  --craft.enabled true \
  --craft.retention_mode hidden \
  --craft.anchor_cache_dir /data/craft/hidden_cache \
  --craft.pooling mean_image_tokens \
  --craft.hidden_layer -2 \
  --craft.retention_loss_type mse \
  --craft.epsilon 0.05 \
  --craft.lambda_lr 1e-4 \
  --dry_run_batches 3
```

## 4. 多卡模板

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train_craft.py \
  --data_root_dir /data/rlds \
  --vla.data_mix libero_90 \
  --craft.anchor_cache_dir /data/craft/hidden_cache \
  --craft.retention_mode hidden \
  --craft.pooling mean_image_tokens
```

> 提示：多卡前先单卡 dry-run，确认 cache schema 与 hidden 维度一致。

## 5. 关键日志解释

日志字段（每 step）：

- `retention_mode`：当前 retention 类型（应为 hidden）
- `L_act`：动作损失
- `L_ret`：表示保持损失
- `lambda`：当前对偶变量
- `epsilon`：当前漂移预算（可随 schedule 变化）
- `grad_dot`：`g_act` 与 `g_ret` 点积
- `grad_cos`：`g_act` 与 `g_ret` 余弦
- `projection`：是否触发冲突投影（`dot < 0`）

## 6. 训练前检查清单

- cache 的 `hidden_layer/pooling/dtype` 与训练配置一致
- `target_features` 维度与当前模型 hidden_dim 一致
- `craft.anchor_batch_size` 不要过大（先小 batch 稳定）
- `vla.data_mix` 与你想评测的任务域一致（建议 LIBERO）
