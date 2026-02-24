# Hidden Anchor Cache：生成、格式与校验

## 1. 脚本入口

- 脚本：`vla-scripts/build_anchor_hidden_cache.py`
- 作用：离线加载冻结 teacher（θ0），前向抽取 hidden state，pooling 后存成固定长度特征。

## 2. 关键原则

- 只在离线阶段运行 teacher forward。
- 训练阶段不再在线 teacher/student 双前向。
- 不保存完整 token×dim hidden states，只保存 pooled feature（避免 cache 爆炸）。

## 3. 常用命令模板

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/build_anchor_hidden_cache.py \
  --teacher_checkpoint /path/to/teacher.pt \
  --data_root_dir /data/rlds \
  --dataset libero_90 \
  --output_dir /data/craft/hidden_cache \
  --num_samples 10000 \
  --split train \
  --seed 7 \
  --hidden_layer -2 \
  --pooling mean_image_tokens \
  --dtype float16 \
  --shard_size 2048
```

## 4. 参数说明（按当前代码）

- teacher 路径（三选一）
  - `--teacher_checkpoint`
  - `--teacher_model_path`
  - `--teacher_policy_path`
- 数据相关
  - `--data_root_dir`
  - `--dataset`
  - `--split`（默认 train）
  - `--image_aug`
  - `--shuffle_buffer_size`
- 采样与输出
  - `--output_dir`
  - `--num_samples`
  - `--shard_size`
  - `--seed`
- 表示提取
  - `--hidden_layer`
  - `--pooling`（`mean_image_tokens` / `mean_masked`）
  - `--dtype`（float16/bfloat16/float32）
- 文本提示
  - `--prompts_file`（可选）

## 5. 输出目录结构

示例：

```text
/data/craft/hidden_cache/
  hidden_anchor_shard_000000.pt
  hidden_anchor_shard_000001.pt
  ...
```

## 6. shard schema（当前实现）

每个 `.pt` 文件：

```python
{
  "records": [
    {
      "model_inputs": {
        "input_ids": LongTensor[seq],
        "attention_mask": BoolTensor[seq],
        "pixel_values": FloatTensor[...] | Dict[str, FloatTensor],
        "multimodal_indices": LongTensor[1],
        "image_token_mask": BoolTensor[expanded_seq]
      },
      "target_features": FloatTensor[hidden_dim],
      "meta": {
        "layer": int,
        "pooling": str,
        "dtype": str,
        "teacher": str,
        "split": str,
      }
    },
    ...
  ],
  "meta": {
    "schema_version": 1,
    "retention_mode": "hidden",
    "pooling": str,
    "hidden_layer": int,
    "dtype": str,
  }
}
```

## 7. image token mask 生成逻辑

优先级：

1. 直接使用已有 `image_token_mask`（若样本内已有）
2. 否则依据 OpenVLA forward 的 token 拼接布局推导：
   - image patch token 插在 BOS 后
   - 数量使用 `vision_backbone.num_patches`

## 8. 快速校验

```bash
python - <<'PY'
import torch
x = torch.load('/data/craft/hidden_cache/hidden_anchor_shard_000000.pt', map_location='cpu')
print(x.keys())
print(type(x['records']), len(x['records']))
print(x['records'][0]['target_features'].shape, x['records'][0]['target_features'].dtype)
PY
```
