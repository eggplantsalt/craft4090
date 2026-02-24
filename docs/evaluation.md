# 评测指南：ID/OOD、鲁棒性、表示漂移

## 1. 评测脚本

- CRaFT 统一评测：`evaluation/craft_eval/evaluate_craft.py`
- Shell 入口：`evaluation/craft_eval/run_eval_craft.sh`
- 原仓库 LIBERO 评测：`experiments/robot/libero/run_libero_eval.py`

> 注：仓库中当前没有 `eval_mcq_likelihood.py`。如需 MCQ likelihood 评测，请新增脚本后在本文件补充。

## 2. CRaFT 统一评测内容

`evaluate_craft.py` 输出：

1. ID 成功率 SR%（默认 `libero_10`）
2. OOD 成功率 SR%（默认 `libero_spatial/object/goal`）
3. 鲁棒性下降：
   - 观察扰动（camera shift + brightness）
   - 动作扰动（高斯噪声，默认 $\sigma=0.05$）
4. 表示漂移：
   - `Lret mean/std`
   - `epsilon`
   - `epsilon_violation_rate`
5. 多方法对比表（CRaFT / Naive SFT / EWC / WiSE-FT）

## 3. 最常用命令

```bash
python evaluation/craft_eval/evaluate_craft.py \
  --anchor_cache_dir /data/craft/hidden_cache \
  --id_suite libero_10 \
  --ood_suites "[libero_spatial, libero_object, libero_goal]" \
  --num_trials_per_task 10 \
  --action_noise_sigma 0.05 \
  --epsilon 0.05 \
  --methods "[{name: craft, checkpoint: /ckpts/craft.pt}, {name: naive_sft, checkpoint: /ckpts/sft.pt}, {name: ewc, checkpoint: /ckpts/ewc.pt}, {name: wise_ft, checkpoint: /ckpts/wise.pt}]"
```

## 4. 结果文件

默认输出目录：`evaluation/craft_eval/results/`

- `craft_eval_<timestamp>.json`
- `craft_eval_table_<timestamp>.md`

## 5. 指标解读

- `ID SR%`：同域能力
- `OOD SR%`：跨任务泛化能力
- `Obs Drop / Act Drop`：扰动导致的性能下降，越小越好
- `Lret Mean`：表示漂移强度（越小越好）
- `Violation Rate`：`Lret > epsilon` 的比例（越小越好）
