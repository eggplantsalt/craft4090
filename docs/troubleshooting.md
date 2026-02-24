# 常见报错与排查

## 1) dtype 不一致（FP16/BF16/FP32）

现象：loss 为 NaN、梯度爆炸、`expected scalar type` 报错。

排查：

1. cache 的 `dtype` 与训练期计算 dtype 是否兼容
2. `target_features` 是否在 loss 前转到 `float32` 计算（当前实现已做）
3. `pixel_values` 是否按模型期望 dtype 处理

## 2) mask 长度与 hidden 序列不对齐

现象：shape mismatch / pooling 报错。

排查：

1. 检查 `image_token_mask` 长度是否与 hidden 序列长度一致
2. 确认是否走了 fallback（由 attention_mask 推导）
3. 检查 cache 生成与训练时 `pooling` 是否一致

## 3) hidden_layer 越界

现象：`IndexError`（hidden states 层索引越界）。

排查：

1. 减小绝对值（如 `-2 -> -1`）
2. 打印 `len(output.hidden_states)` 后再配置 `hidden_layer`

## 4) cache 特征维度不一致

现象：`target_features` 与当前模型 pooled features 维度不等。

常见原因：

- teacher 与 student 主干不一致
- cache 来自旧模型版本
- pooling/hidden_layer 与训练配置不一致

处理：

- 重新用当前 teacher 配置生成 cache

## 5) DDP/多卡梯度异常

现象：梯度为 None、不同卡行为不一致。

排查：

1. 两次 backward 之间是否正确 `zero_grad`
2. 合并梯度写回时是否逐参数对齐（含 None）
3. 先单卡 `dry_run_batches=3` 跑通，再扩展多卡

## 6) epsilon 约束不收敛

现象：`L_ret` 长期 > epsilon，`lambda` 持续增大。

排查：

1. 降低学习率或 `lambda_lr`
2. 调整 epsilon（先放宽，后收紧）
3. 增大 anchor 代表性（num_samples / 数据覆盖）

## 7) 找不到评测脚本

- 当前仓库评测入口是：
  - `evaluation/craft_eval/evaluate_craft.py`
  - `experiments/robot/libero/run_libero_eval.py`
- 当前不存在：`eval_mcq_likelihood.py`
