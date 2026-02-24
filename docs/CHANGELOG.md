# 文档整理变更记录（本次）

## 目标

根据当前代码实现（OpenVLA + CRaFT）重整文档结构，提供可复现实验流程。

## Markdown 全盘扫描与处理决定

| 路径 | 决策 | 理由 |
|---|---|---|
| `README.md` | 重写 | 原内容超长且偏上游通用说明；已改为当前仓库 CRaFT 实装导览并指向 `docs/` |
| `docs/CONTEXT.md` | 保留（不改） | 用户明确要求不可修改 |
| `docs/IDEA.md` | 保留 | 作为方法背景草案，非操作手册 |
| `docs/getting_started.md` | 新增 | 从零到复现的最短路径 |
| `docs/datasets.md` | 新增 | 数据准备、路径、字段约束 |
| `docs/anchors_and_cache.md` | 新增 | hidden cache 参数与 shard schema |
| `docs/training_craft.md` | 新增 | 训练参数、更新规则、日志解读 |
| `docs/evaluation.md` | 新增 | 评测脚本、输出、指标解释 |
| `docs/troubleshooting.md` | 新增 | 高频报错排查 |
| `docs/directory_guide.md` | 新增 | 工程目录导览 |
| `dlimp_openvla/README.md` | 保留 | 第三方/子模块文档，非本次 CRaFT 主流程文档 |
| `dlimp_openvla/rlds_converters/README.md` | 保留 | 第三方/子模块文档 |
| `LIBERO/README.md` | 保留 | 外部基准组件文档 |
| `.pytest_cache/README.md` | 忽略 | 自动生成缓存文件，不属于工程教程 |

## 删除说明

本次未删除任何业务文档文件。

## 额外说明

- 仓库当前不存在 `eval_mcq_likelihood.py`，在 `docs/evaluation.md` 已按实际代码说明。
- CRaFT 评测实际入口为 `evaluation/craft_eval/evaluate_craft.py`。
