# 数据集准备与配置（RLDS / LIBERO）

本仓库训练与评测链路建立在 RLDS 风格数据之上。

## 1. 推荐数据集

- 微调与评测优先：LIBERO（例如 `libero_10` / `libero_90`）
- 训练脚本中常见数据参数：
  - `--data_root_dir`：RLDS 根目录
  - `--dataset`（cache 构建脚本）
  - `--vla.data_mix`（训练脚本）

## 2. 路径规范

建议统一：

```bash
export DATA_ROOT=/data/rlds
# 例如：/data/rlds/libero_90
```

脚本会从 `DATA_ROOT` 下按 mix 名称读取对应数据。

## 3. 最小字段要求（用于 CRaFT hidden cache）

离线 anchor 构建脚本 `vla-scripts/build_anchor_hidden_cache.py` 使用的数据字段（由 RLDS 样本提供）：

- `observation.image_primary`
- `task.language_instruction`

脚本会构造 action-free anchor（只用 image + instruction，不读取动作监督作为 target）。

## 4. 训练数据输入（CRaFT train）

`vla-scripts/train_craft.py` 的主训练 batch 走现有 OpenVLA 数据管线，核心字段：

- `pixel_values`
- `input_ids`
- `attention_mask`
- `labels`

anchor batch 从离线 cache 读取：

- `model_inputs`（字典）
- `target_features`

## 5. 常见配置错位

- `--dataset libero_90`（用于 cache 构建）和 `--vla.data_mix libero_90`（用于训练）最好保持一致。
- `hidden_layer/pooling` 在 cache 构建和训练中必须一致，否则 `L_ret` 无意义。
