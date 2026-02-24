# OpenVLA-CRaFT 快速开始（从零到可运行）

本指南面向第一次接触仓库的同学，目标是“复制粘贴命令即可跑通最小链路”。

## 0. 前置约束

- 你需要 Linux + CUDA GPU 环境。
- 本仓库训练与评测依赖 RLDS 数据格式。
- 本仓库中的 CRaFT 以 `hidden retention` 为主（非 token retention）。

## 1. 安装环境

```bash
cd /workspace
# 你也可以换成自己的路径
git clone <your_openvla_fork_url> openvla
cd openvla

# 建议 Python 3.10
conda create -n openvla-craft python=3.10 -y
conda activate openvla-craft

# 安装仓库
pip install -e .

# （可选）安装 flash-attn
pip install packaging ninja
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

## 2. 最短路径（强烈建议按顺序）

### Step A：准备数据

先准备 RLDS 数据目录，例如：

```bash
export DATA_ROOT=/data/rlds
```

数据要求详见 `docs/datasets.md`。

### Step B：生成 hidden anchor cache

```bash
export TEACHER_CKPT=/path/to/teacher/checkpoint.pt
export ANCHOR_CACHE_DIR=/data/craft/hidden_cache

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/build_anchor_hidden_cache.py \
  --teacher_checkpoint "$TEACHER_CKPT" \
  --data_root_dir "$DATA_ROOT" \
  --dataset libero_90 \
  --output_dir "$ANCHOR_CACHE_DIR" \
  --num_samples 10000 \
  --split train \
  --hidden_layer -2 \
  --pooling mean_image_tokens \
  --dtype float16 \
  --shard_size 2048
```

参数解释与输出格式详见 `docs/anchors_and_cache.md`。

### Step C：CRaFT 训练 dry-run（3 step）

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train_craft.py \
  --data_root_dir "$DATA_ROOT" \
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
```

你也可以用封装脚本：

```bash
bash vla-scripts/train_craft.sh "$DATA_ROOT" "$ANCHOR_CACHE_DIR" "openvla-7b+prismatic"
```

### Step D：评测

```bash
python evaluation/craft_eval/evaluate_craft.py \
  --anchor_cache_dir "$ANCHOR_CACHE_DIR" \
  --id_suite libero_10 \
  --ood_suites "[libero_spatial, libero_object, libero_goal]" \
  --num_trials_per_task 10 \
  --epsilon 0.05 \
  --methods "[{name: craft, checkpoint: /path/to/craft.pt}]"
```

## 3. baseline 可用性快速检查

如果你只想验证原始 OpenVLA 入口未被破坏：

```bash
python vla-scripts/train.py --help
python vla-scripts/finetune.py --help
python scripts/pretrain.py --help
```

## 4. 推荐阅读顺序

1. `docs/directory_guide.md`
2. `docs/datasets.md`
3. `docs/anchors_and_cache.md`
4. `docs/training_craft.md`
5. `docs/evaluation.md`
6. `docs/troubleshooting.md`
