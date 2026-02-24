# OpenVLA + CRaFT（本仓库当前实现）

本仓库已在 OpenVLA 代码基线上接入 CRaFT（Constrained Representation and Fine-Tuning）主流程，包含：

- 离线 hidden anchor cache 构建
- CRaFT 训练入口（hidden retention + 梯度冲突投影 + primal-dual λ 更新）
- CRaFT 评测入口（ID/OOD、鲁棒性、表示漂移）

> 说明：本 README 聚焦**当前仓库的可复现实验路径**。更完整教程请看 `docs/`。

## 快速导航

- 新手起步：`docs/getting_started.md`
- 数据与路径规范：`docs/datasets.md`
- Hidden cache 生成：`docs/anchors_and_cache.md`
- CRaFT 训练：`docs/training_craft.md`
- 评测（ID/OOD/鲁棒性/漂移）：`docs/evaluation.md`
- 常见报错：`docs/troubleshooting.md`
- 工程目录导览：`docs/directory_guide.md`
- 文档变更记录：`docs/CHANGELOG.md`

## 当前关键脚本

- `vla-scripts/build_anchor_hidden_cache.py`
  - 离线冻结 teacher 前向，导出 pooled hidden features（不保存 token×dim）
- `vla-scripts/train_craft.py`
  - 训练中联合 `L_act + λ·L_ret`，并做冲突梯度投影
- `vla-scripts/train_craft.sh`
  - 3 step dry-run 快速入口
- `evaluation/craft_eval/evaluate_craft.py`
  - 统一评测：ID/OOD 成功率、观测/动作扰动鲁棒性、表示漂移
- `evaluation/craft_eval/run_eval_craft.sh`
  - 评测脚本调用模板

## 一句话复现顺序

1. 准备 RLDS 数据（建议 LIBERO）
2. 生成 hidden anchor cache
3. 运行 train_craft（先 dry-run，再长跑）
4. 运行 evaluate_craft，导出 json + 对比表

详细命令见 `docs/getting_started.md` 与对应专题文档。
