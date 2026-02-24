# 工程目录指引（按当前实现）

## 1. 训练与算法核心

- `prismatic/craft/`
  - `craft_config.py`：CRaFT 配置项
  - `grad_surgery.py`：dot/projection/merge
  - `primal_dual.py`：epsilon schedule 与 lambda 更新
  - `retention_loss.py`：hidden retention loss（含 pooling 与 mask fallback）
  - `anchor_cache.py`：anchor cache dataset + collate + loader

- `vla-scripts/train_craft.py`
  - CRaFT 训练入口（L_act + L_ret、冲突投影、合并更新）

## 2. 数据与缓存

- `vla-scripts/build_anchor_hidden_cache.py`
  - 离线 hidden anchor cache 构建脚本
- `cache/...`（运行时生成）
  - hidden shard 文件

## 3. 评测

- `evaluation/craft_eval/evaluate_craft.py`
  - ID/OOD + 鲁棒性 + 表示漂移 + 对比表
- `evaluation/craft_eval/run_eval_craft.sh`
  - 评测调用模板
- `experiments/robot/libero/run_libero_eval.py`
  - 原始 LIBERO 评测入口（baseline 兼容）

## 4. baseline 训练入口（保持可用）

- `vla-scripts/train.py`（全参训练）
- `vla-scripts/finetune.py`（LoRA/PEFT）
- `scripts/pretrain.py`（VLM 预训练）

## 5. 文档

- `docs/getting_started.md`
- `docs/datasets.md`
- `docs/anchors_and_cache.md`
- `docs/training_craft.md`
- `docs/evaluation.md`
- `docs/troubleshooting.md`
- `docs/directory_guide.md`
- `docs/CHANGELOG.md`
- `docs/CONTEXT.md`（用户上下文，不修改）
