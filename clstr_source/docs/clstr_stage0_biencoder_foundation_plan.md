# CLSTR Stage0 Bi-Encoder Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先把 CLSTR 的大池粗召回基础重建为 SkillRouter-compatible bi-encoder，再在强 top-M 候选上训练 CLSTR 的 belief / transition / policy / act。

**Architecture:** Stage0 只承担 SkillRouter-style bi-encoder 粗召回，不接 cross-encoder/listwise reranker，不引入 `m_t`。Stage1/2/3/4 在 Stage0 top-M 候选内做 CLSTR 的多步动态选择，证明 `m_t`、transition 和 ACT 的增益，而不是让弱粗召回拖垮后续阶段。

**Tech Stack:** Python, PyTorch, Transformers, SkillRouter-Embedding-0.6B, JSONL unified training data, pytest, Slurm sbatch, H200/A800 GPU.

---

## 1. 背景与新问题

当前 v4 计划中 `Stage 1 — L_retr warmup` 实际承担的是大池粗召回，但实现和命名没有对齐 SkillRouter 的 bi-encoder 协议，导致后续 Stage2/3/4 的结果难以解释。

新判断如下：

- Stage0 应该做成 SkillRouter 的 bi-encoder 部分，即 `query_text -> encoder -> q`、`skill_text -> encoder -> s_i`、`score_i = cosine(q, s_i)`、取 top-M。
- Stage0 不应包含 SkillRouter 的 cross-encoder/listwise reranker。传统 rerank 可以作为静态强基线或上界，但不阻塞 CLSTR 主线。
- Stage2 本来就应该承担 CLSTR 的候选内动态选择，相当于我们的 learned recurrent reranker。
- 当前旧 Stage1 不能作为可靠基础。它在约 37k skill pool 上 final batch `recall_at_1=0.125`、`recall_at_50≈0.44`，弱于一个可发表的 SkillRouter-style foundation。
- 不能继续在弱 Stage0/Stage1 上训练完整 Stage2/3/4，否则即使跑完也无法证明 CLSTR 方法价值。

因此下一阶段目标是：**先暂停下游训练，把 skill pool / qrels / Stage0 bi-encoder 粗召回修到可与 SkillRouter 公平对齐，再恢复 CLSTR base/act。**

---

## 2. 当前未完成项汇总

### 2.0 当前执行状态（2026-05-26）

本计划中的 foundation repair 代码与安全入口已完成第一轮实现和单元/脚本验证：

- Task 1-5: readiness gate、skill pool/qrels audit、Stage0 bi-encoder protocol、Stage0 coarse-recall gate、same-pool SkillRouter frozen baseline 的测试已通过。
- Task 6-7: Stage1/Stage2 已改为消费 Stage0 top-M candidate set；统一 Stage1/Stage2 sbatch 默认走 CLSTR-native 入口，不再走 Qwen wrapper；相关 handoff/sbatch 测试已通过。
- Task 8: README、`docs/clstr_v4_plan.md`、`description.md` 已同步当前口径。

仍未完成的不是 foundation repair 代码，而是需要后续 GPU/benchmark 执行的实验项：Stage1 heads-init full training、Stage2 CLSTR-base full training、Stage3/4 训练、ToolRet/TRAJECT/ToolBench-G3 official evaluation matrix、AppWorld secondary combination plot。

2026-05-27 01:27 当前执行证据：旧 Stage0 baseline 作业 `77801` 与 Stage0
bi-encoder 作业 `77802` 原先提交到 `gpu_a800`，但资源显示为
`cpu=1,mem=15700M` 且长期 `PENDING (Priority)`，已取消。随后尝试在脚本内显式写
`#SBATCH --cpus-per-task=12` 会被集群 `job_submit_lua` 拒绝；正确做法是不要覆盖
GPU 队列默认 CPU 配比。现已改投 `gpu_h200`：

- `77819`: `scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh`
- `77820`: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`

`scontrol show job` 显示两者均有 `CpusPerTres=gpu:12` 与 `TresPerJob=gres/gpu:1`。
在
`outputs/clstr_stage0_skillrouter_frozen_baseline/metrics.json` 与
`outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
生成并通过 Stage0 quality gate 前，不重复提交同类作业，不进入 Stage2/3/4。

2026-05-27 01:10 补充执行证据：已接入并运行 conservative borderline review，
不自动合并候选，只把低置信 fuzzy candidates 标记为 keep-separate，高置信重复仍会继续
阻断。真实 `data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/skill_pool.jsonl`
审计结果为：

- `skill_count=37623`
- `reviewed_candidate_count=20000`
- `keep_separate_count=20000`
- `unresolved_candidate_count=0`
- review report:
  `outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json`

重跑 unified readiness 后，`skill_pool_quality.status=ok`，
`borderline_review_status=complete_by_conservative_review`，
`stage0_coarse_retriever.ready_to_submit=true`。当前 `status` 仍为
`action_required` 的原因已经不再是 skill pool dedup，而是下游产物缺失：
`missing_stage0_checkpoint`、`missing_stage2_checkpoint`、`missing_stage3_checkpoint`
以及 full rollout HRPO 尚未实现。未提交新的 GPU 作业。

2026-05-27 01:17 补充修复：Phase H evaluation matrix readiness 已接入
Stage4 quality gate，不再只看 Stage4 checkpoint 文件是否存在。Stage4 gate 现在要求
`training_metrics.jsonl` 中存在持续的 `stage4_act_loss` 与 `stage4_act_count` 记录；
若缺少 L_act 质量证据，则 Phase H report 加入 `stage4_quality_gate_not_ok`，
`official_main_table_ready=false`。这保证 Stage4 checkpoint、L_act 训练指标、
expected benchmark coverage、latest checkpoint 与 loss curve 都满足后，才允许进入
官方主表 readiness 口径。

2026-05-27 01:39 补充修复：`77819/77820` 启动后立即失败，根因为主线训练 sbatch
仍硬编码 `source activate env`，而当前集群 module 的 conda 中不存在名为 `env`
的环境；命名激活 `xzf` 也不可见。已将 Stage0/Stage1-compat/Stage2/Stage3/Stage4
主线训练 sbatch 改为默认激活绝对环境路径：
`/data/home/scyb713/run/miniconda3/envs/xzf`，并保留 `CONDA_ENV_PATH` 可覆盖。
本地验证 `source activate /data/home/scyb713/run/miniconda3/envs/xzf` 后使用
`Python 3.10.20`。已重新提交：

- `77824`: `scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh`
- `77825`: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`

两者当前为 `PENDING (Priority)`，`scontrol show job` 仍显示
`CpusPerTres=gpu:12` 与 `TresPerJob=gres/gpu:1`。在它们产出并通过 Stage0 quality
gate 前，不重复提交 Stage0，不进入 Stage2/3/4。

2026-05-27 02:05 补充状态：`77824/77825` 均已在 `gpu_h200` 运行，节点
`d1n41d29g01`，每作业 `cpu=12,mem=252000M,gres/gpu=1`。当前证据显示：

- Stage0 baseline 作业 `77824` 已生成
  `outputs/clstr_stage0_skillrouter_frozen_baseline/qrels.jsonl`，但尚未生成
  `run.tsv` 与 `metrics.json`。
- Stage0 bi-encoder 作业 `77825` 已完成 skill table rebuild，并开始训练；
  `training_metrics.jsonl` 当前 `metric_rows=1653`、`last_step=1653/5000`。
- 当前 tail-50 均值约为：`loss=9.5086`、`recall_at_20=0.4420`、
  `recall_at_50=0.5216`、`recall_at_100=0.5784`。
- `checkpoints/latest.pt` 与 `loss_curve.svg` 已持续更新，但最终 checkpoint
  `checkpoints/clstr_unified_retrieval_v2-step5000.pt` 尚不存在。

因此 Stage0 quality gate 仍缺 same-pool baseline metrics、Stage0 final
checkpoint，以及基于 final checkpoint 导出的同池 full retrieval eval metrics；
当前仍不提交 Stage2/3/4，下一次检查不应早于 15-30 分钟。

2026-05-27 03:10 补充状态：`77824/77825` 已自然完成，same-pool SkillRouter
baseline metrics、baseline run/qrels、Stage0 final checkpoint、latest checkpoint、
training metrics 与 loss curve 均已存在。重新运行 Stage0 quality gate 后，当前唯一
blocker 是缺少
`outputs/clstr_unified_stage0_biencoder/full_retrieval_eval/metrics.json`，具体为
`missing_stage0_full_eval_metrics` 以及对应 `missing_stage0_eval_Recall@20/50/100`。
已提交 full retrieval eval 作业：

- `77835`: `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh`，`gpu_h200`，
  当前 `PENDING (Priority)`。

在 `77835` 产出 full retrieval metrics 且 Stage0 quality gate 通过前，不重复提交
Stage0 full eval，不进入 Stage2/3/4。

2026-05-27 03:32 补充修复：`77835` 启动后立即失败，日志显示 Slurm 将 `$0`
重写为 `/var/spool/slurmd/job77835/slurm_script`，导致
`source "$(dirname "$0")/_clstr_gpu_env.sh"` 在 spool 目录查找共享环境脚本并失败。
这不是 Stage0 eval 本身的问题，而是主线 sbatch 入口的环境脚本定位方式不适配
Slurm 执行语义。

已将以下主线 CLSTR GPU sbatch 脚本统一改为：

```bash
PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
```

- `scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh`
- `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh`
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`
- `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
- `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh`

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_sbatch_scripts.py -q
# 39 passed

bash -n scripts/sbatch/_clstr_gpu_env.sh \
  scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh \
  scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh \
  scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh
# clean
```

已重新提交 Stage0 full retrieval eval：

- `77836`: `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh`，`gpu_h200`，
  提交后进入 `RUNNING`，日志不再出现 `_clstr_gpu_env.sh` 路径错误。

在 `77836` 产出 full retrieval metrics 且 Stage0 quality gate 通过前，不重复提交
Stage0 full eval，不进入 Stage2/3/4。

2026-05-27 04:12 补充修复：`77836` 产出了 full retrieval metrics，但 Stage0 gate
失败，Recall@20/50/100 分别为 `0.560164/0.663969/0.738568`，低于 same-pool
SkillRouter frozen baseline `0.706692/0.811626/0.884804`。继续追踪发现 full eval
导出在加载 checkpoint 后无条件 `model.rebuild_skill_table()`，会把 checkpoint 中训练后
的 `skill_table.E` 覆盖成 frozen encoder 重新编码的 skill embeddings，因此 `77836`
不是完整 Stage0 checkpoint 的有效评估。

已修复：

- `clstr/clstr_retrieval_export.py` 在 checkpoint 成功加载 `skill_table.E` 时跳过
  `rebuild_skill_table()`，report 记录
  `skill_table_rebuild=skipped_checkpoint_embeddings`。
- full retrieval export 改为 batch 级流式写 `predictions.jsonl`、`run.tsv` 与
  `progress.json`，长评估期间可观测。
- `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh` 重跑前删除旧
  `metrics.json/run.tsv/predictions.jsonl/progress.json` 等产物，避免 Stage0 gate
  误读旧评估。
- 旧错误评估结果已备份到
  `outputs/clstr_unified_stage0_biencoder/full_retrieval_eval_rebuilt_skill_table_bug_77836`。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_clstr_retrieval_export.py \
  tests/test_stage0_biencoder_protocol.py::test_stage0_biencoder_cli_help_and_sbatch_defaults \
  tests/test_sbatch_scripts.py::test_stage0_full_retrieval_eval_sbatch_exports_same_pool_eval_metrics_for_gate \
  -q
# 6 passed
```

已重新提交修复后的 Stage0 full retrieval eval：

- `77837`: `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh`，`gpu_h200`，
  当前 `RUNNING`；`progress.json` 已进入 `retrieving`，健康检查时
  `processed_queries=784/441097`。

在 `77837` 产出新的 full retrieval metrics 且 Stage0 quality gate 通过前，不进入
Stage2/3/4。

2026-05-27 补充结论：`77837` 已产出有效 full retrieval eval，`run_report.json`
确认 checkpoint 中的 `skill_table.E` 被加载且 `skill_table_rebuild` 为
`skipped_checkpoint_embeddings`。Stage0 gate 仍失败：

- 旧 CLSTR Stage0 checkpoint:
  `Recall@20/50/100 = 0.573444 / 0.682046 / 0.765536`
- same-pool SkillRouter frozen baseline:
  `Recall@20/50/100 = 0.706692 / 0.811626 / 0.884804`
- blockers:
  `stage0_recall_at_20_below_same_pool_baseline`、
  `stage0_recall_at_50_below_same_pool_baseline`、
  `stage0_recall_at_100_below_same_pool_baseline`

因此旧 Stage0 训练不是可用 foundation，禁止用
`outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
启动 Stage2/3/4。

已完成必要训练协议重构：

- `clstr/retrieval_warmup.py` 支持 `retrieval_loss_mode=multi_positive_nll`，对同一
  query 的多个 canonical positive skill 做概率质量聚合，避免把等价 positive 当负例。
- `gradient_accumulation_steps` 真实生效；`max_steps` 表示 optimizer step，
  每步可累积多个 micro-batch，并在 metrics 中记录 `optimizer_step`、
  `effective_batch_size`、`positive_label_counts`。
- Stage0 默认不再训练 frozen skill embeddings、skill bias 或 encoder projection：
  `train_skill_embeddings=false`、`train_skill_bias=false`、
  `train_encoder_projection=false`。默认只训练轻量 retrieval adapter 与
  `logit_scale_retr`，学习率降为 `2e-5`。
- `scripts/run_clstr_stage0_biencoder_train.py` 与
  `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh` 已暴露这些覆盖参数。
- 新 checkpoint / rolling latest 会写入 `stage0_protocol`，便于区分旧失败协议与新协议。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_biencoder_protocol.py \
  tests/test_skillret.py::test_skillret_retrieval_warmup_writes_checkpoint_and_report \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  tests/test_skillret.py::test_retrieval_warmup_multi_positive_loss_does_not_penalize_equivalent_positive \
  tests/test_skillret.py::test_retrieval_warmup_accumulates_gradients_inside_each_optimizer_step \
  tests/test_skillret.py::test_retrieval_warmup_batches_advance_by_batch_stride_for_full_corpus_coverage \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_is_stage0_compatibility_wrapper \
  tests/test_sbatch_scripts.py::test_main_gpu_training_sbatch_does_not_override_cluster_gpu_cpu_bundle \
  tests/test_sbatch_scripts.py::test_main_gpu_training_sbatch_uses_existing_default_conda_env_path \
  tests/test_sbatch_scripts.py::test_main_gpu_training_sbatch_sources_shared_clstr_environment \
  -q
# 16 passed
```

下一步是重新训练一个新输出目录的 Stage0 v2，例如
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE`，再做 full retrieval eval 和
Stage0 quality gate。新 gate 通过前仍不进入 Stage2。

2026-05-27 执行状态：提交前确认队列为空，且
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE` 不存在。已提交新协议 Stage0 v2
完整训练与依赖 full retrieval eval：

- `77838`: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`，
  `gpu_h200`，`PENDING (Priority)`，
  `OUTPUT_DIR=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE`。
- `77839`: `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh`，
  `gpu_h200`，`PENDING (Dependency)`，`afterok:77838`，
  `CHECKPOINT_PATH=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt`，
  `OUTPUT_DIR=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/full_retrieval_eval`。

在 `77838/77839` 自然完成并运行新 Stage0 quality gate 前，不重复提交 Stage0 v2，
不进入 Stage2。长任务检查间隔按用户要求不低于约 30 分钟。

随后检查发现 `77838` 在 `gpu_h200` 因 `Priority` pending，预计开始时间过晚；此时
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE` 尚无训练产物，说明旧链路没有实际
启动。已取消未启动的 H200 链路 `77838/77839`，改投 A800：

- `77840`: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`，
  `gpu_a800`，`RUNNING`，节点 `d1n41a12g03`，
  `OUTPUT_DIR=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE`。
- `77841`: `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh`，
  `gpu_a800`，`PENDING (Dependency)`，`afterok:77840`，
  `CHECKPOINT_PATH=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt`，
  `OUTPUT_DIR=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/full_retrieval_eval`。

当前有效链路是 `77840/77841`。不要重复提交 Stage0 v2，不进入 Stage2；下一次检查按
长任务间隔处理。

一次性健康检查：`77840` 已在 `gpu_a800` 节点 `d1n41a12g03` 运行，`77841` 仍为
`afterok:77840` 依赖等待。`setup_status.jsonl` 显示 skill table rebuild 已完成，
shape 为 `[37623, 1024]`；训练已按
`batch_size=128`、`gradient_accumulation_steps=2`、
`effective_batch_size=256`、`retrieval_loss_mode=multi_positive_nll` 启动。
`training_metrics.jsonl` 已开始写入，健康检查时最新观察到 step 22。训练 batch recall
只作为学习信号，不能替代后续 full retrieval gate。

2026-05-27 补充必要重构：Stage2/3/4 主线初始化已收敛到 gated Stage0 checkpoint，
不再混用旧 SkillRET `outputs/clstr_native_routing_init/manifest.json` 或 Qwen wrapper。
新增 `clstr/stage_checkpoint_init.py` 负责从 Stage0 checkpoint 读取 config、构建
CLSTRModel、加载 Stage0 routing weights，并在 Stage3/4 合并上游 head checkpoint。
canonical GPU 入口现在是：

- Stage2: `scripts/run_clstr_stage2_full_base_train.py`
- Stage3: `scripts/run_clstr_unified_stage3_hrpo.py`
- Stage4: `scripts/run_clstr_stage4_act_train.py`

对应 sbatch 不再暴露 `ROUTING_INIT_MANIFEST`、`QWEN_MODEL_PATH` 或
`--qwen_model_path`。Qwen 相关旧入口保留为 legacy/reference experiments，但不能作为
unified Stage2/3/4 主线训练证据。

2026-05-27 补充边界重构：`clstr.full_base_train.run_clstr_full_base_train`
现在只作为 canonical Stage2 入口，必须提供 gated Stage0
`routing_checkpoint_path`；若传入旧 `routing_init_manifest` 会直接报错。旧
SkillRET/native-routing-init 路径被收敛到显式函数
`run_legacy_clstr_full_base_train_from_routing_init`，并且旧脚本
`scripts/run_clstr_full_base_train.py`、`scripts/run_clstr_dagger_success_train.py`、
`scripts/run_clstr_loss_ablation.py` 与 `clstr/structured_train.py` 均改为调用该
legacy wrapper。这样旧实验仍可复现，但不能再被误当成 unified Stage2 主线证据。

2026-05-27 continuation audit：当前有效 Stage0 v2 链路仍是 `77840/77841`。
`77840` 在 `gpu_a800` 上运行，最近一次检查时 runtime 约 31 分钟，`77841` 仍为
`afterok:77840` 依赖等待。`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE`
已有 `latest.pt`、`training_metrics.jsonl`、`loss_curve.svg` 与
`setup_status.jsonl`，但 final checkpoint
`checkpoints/clstr_unified_retrieval_v2-step5000.pt` 和
`full_retrieval_eval/metrics.json` 尚未产生；最新观察到训练约 `step=1119/5000`。
因此 Stage0 gate 当前仍不能运行，Stage2/3/4 仍不得提交。

同次 continuation audit 对不依赖 Stage0 完成的 Stage4/Phase H 静态实现做了复验：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage4_act_train.py \
  tests/test_stage4_quality_gate.py \
  tests/test_eval_matrix_readiness.py \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_eval_matrix_readiness_sbatch_audits_without_running_benchmarks \
  -q
# 15 passed
```

这说明 Stage4 `L_act` 的 train-split next-skill builder、TRAJECT/ToolBench-G3 benchmark
filter、positive next skill injection/provenance、Stage4 quality gate、Phase H
official-vs-proxy readiness 边界在当前代码中仍成立。剩余阻塞不是这些静态实现，而是
Stage0 v2 训练、full retrieval eval 与 quality gate 尚未完成。

### 2.1 来自新讨论的未完成项

- [x] 将现有 `Stage1 retrieval warmup` 重命名或重新定位为 `Stage0 coarse retriever`。
- [x] Stage0 使用 SkillRouter-compatible bi-encoder 协议：query template、skill serialization、left padding、last-token pooling、L2 normalize、cosine scoring。
- [x] Stage0 只优化大池 top-M recall，不强行把 bi-encoder 当最终 top1 reranker。
- [x] Stage0 训练协议已修正为 multi-positive NLL + frozen SkillRouter skill table 默认策略，避免旧单正例 CE/全表微调把 frozen baseline 拉坏。
- [x] 使用新协议重跑 Stage0 v2 full training、full retrieval eval 和 Stage0 gate；当前 Stage0 gate 已为 `status=ok`。
- [x] Stage1 在 Stage0 top-M 候选上初始化 CLSTR policy / belief / transition / STOP heads。
- [x] Stage2 从 Stage1 checkpoint 继续同一个 CLSTRModel，作为更完整的 supervised CLSTR-base；内部恢复 Stage0 frozen routing foundation 只是 checkpoint restore 细节。
- [x] Stage2/3/4 canonical 初始化不再默认使用旧 SkillRET routing-init manifest 或 Qwen wrapper；canonical Stage2 函数会拒绝旧 manifest，legacy manifest 只能通过显式 legacy wrapper 使用。
- [x] SkillRouter cross-encoder/listwise reranker 只作为 baseline 或 upper bound，不作为当前主线依赖。
- [x] 当前 Stage1 弱结果不得再作为进入 Stage2 的充分依据。

### 2.2 来自 `docs/clstr_v4_plan.md` 的未完成项

- [x] 完成 partial dedup readiness/audit blocker：rule merge 后的 borderline candidates 若仍为 pending，会阻断 Stage0/Stage1/Stage2。
- [x] 完成 canonical skill id 与 multi-positive qrels 审计，避免同义 skill 被当成错误负例。
- [x] 完成 unified v2 数据集 readiness/audit 入口：TRAJECT、ToolBench-G3、ToolRet、SKILLRET 的 retrieval 与 trajectory stream 必须统一到同一 skill pool。
- [x] 完成 leakage audit v2 readiness 入口：TRAJECT/ToolBench/ToolRet/AppWorld dev/test signal 不得进入训练。
- [x] 重建 Stage0/Stage1/Stage2/Stage3/Stage4 的 sbatch 入口与 readiness gate，防止绕过数据质量门。
- [ ] 完成 Stage2 CLSTR-base 的合理训练，不再基于旧弱 retrieval checkpoint 启动。
- [x] 完成 Stage3 full simulator/on-policy HRPO，或明确只报告 offline surrogate，不能写成完整 RL 证据。
- [ ] 完成 Stage4 transition-conditioned next-skill L_act，且不局限 AppWorld。
- [ ] 完成 Phase H official evaluation matrix：ToolRet official run、TRAJECT official metrics、ToolBench-G3 StableToolBench pass rate、AppWorld secondary combination plot。

2026-05-27 当前选择的是后一种：明确只报告 offline surrogate。`stage3_offline_hrpo`
gate 标注 `on_policy_rollout_used=false`、`not_full_simulator_rollout_hrpo=true`、
`paper_evidence_ready=false`；`stage3_full_rollout_hrpo` gate 固定 not ready，blocker 为
`full_rollout_hrpo_not_implemented`。因此后续可以工程上运行 Stage3 offline surrogate，
但论文中不能把它写成 full simulator/on-policy HRPO 证据。

---

## 3. 目标阶段定义

### Pre-Stage0: Skill Pool 与 Qrels 清理

输入：

- `data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/skills.jsonl`
- `data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/retrieval.jsonl`
- `data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/trajectories.jsonl`
- manifest 中的 skill pool / borderline review metadata

输出：

- canonical `skills.jsonl`
- canonical `retrieval.jsonl`
- multi-positive qrels
- dedup report
- leakage audit report

验收：

- manifest 中 `skill_pool.borderline_review.status != pending_manual_or_llm_review`
- placeholder / null / duplicate skill records 被 gate 阻断
- 每条 retrieval positive 都能映射到 canonical skill id
- 同义 skill 不再被采样成 hard negative

### Stage0: SkillRouter-Compatible Bi-Encoder Coarse Retriever

输入：

- canonical skill pool
- canonical train-only retrieval pairs
- SkillRouter-compatible query text
- SkillRouter-compatible skill text

模型：

```text
query_text -> SkillRouter-Embedding-0.6B -> q
skill_text -> SkillRouter-Embedding-0.6B -> s_i
score_i = exp(logit_scale) * cosine(q, s_i) + skill_bias_i
top-M = argsort(score)[:M]
```

协议：

- `tokenizer.padding_side = "left"`
- pooling 使用 last token
- query/skill embedding 均 L2 normalize
- projection 与 SkillTable adapter 默认 identity init
- InfoNCE / in-batch negatives / hard negatives
- batch size 优先提高到 128-256；如显存不足使用 gradient accumulation

指标：

- primary: `Recall@20/50/100`
- secondary: `MRR@100`, `NDCG@10/20`
- 不用 `Recall@1` 作为唯一 gate，因为 Stage0 是 coarse retriever

最低验收：

- 在同一 canonical pool 上，Stage0 frozen/finetuned bi-encoder 不低于 SkillRouter frozen baseline 的 coarse recall。
- 若 Stage0 仍明显低于 SkillRouter baseline，禁止进入 Stage2。

### Stage1: CLSTR Heads Initialization on Top-M

输入：

- Stage0 checkpoint
- Stage0 top-M candidates
- train trajectory step pairs

训练：

- 初始化 BeliefGate / SkillHead / StopHead / TransitionPredictor
- 不大幅改写 Stage0 retrieval path
- policy CE 在 top-M 内学习 ground-truth next skill

验收：

- top-M candidate contains positive 的比例不退步
- candidate 内 top1 / MRR 明显高于 Stage0 raw ranking
- no-m_t、m_t、transition ablation 同时产出

### Stage2: CLSTR Base Multi-Step Supervised Training

输入：

- 通过 Stage0 quality gate 的 Stage0 checkpoint
- unified v2 trajectories
- Stage0 top-M candidate cache 或在线 top-M retrieval

训练：

- `L_policy_CE`
- `L_trans`
- `L_retr` 弱辅或冻结 routing foundation 后只作 prior regularization
- `m_t` 参与候选内动态选择
- canonical Stage2 不再从旧 SkillRET routing-init manifest 初始化；模型 config 与 routing
  weights 来自 Stage0 checkpoint。

验收：

- sequential routing 指标高于 static Stage0。
- no-m_t ablation 低于 recurrent/belief variant。
- TRAJECT / ToolBench-G3 train-dev proxy 不退步。

### Stage3: CLSTR Act / HRPO

定位：

- 若只有 offline grouped preference update，只能称为 offline HRPO-style surrogate。
- 只有接入 simulator/on-policy rollout 后，才能写成 full HRPO / RL evidence。
- canonical Stage3 从 Stage0 routing checkpoint + Stage2 head checkpoint 初始化，不使用
  Qwen external encoder wrapper。

训练：

- Stage0 retrieval path 默认冻结。
- 更新 BeliefGate / SkillHead / StopHead / TransitionPredictor / action_proj。
- 使用 routing prior KL，避免 sparse reward 把 strong retriever 拉坏。

验收：

- routing recall 不低于 Stage2。
- reward/success proxy 有稳定提升。
- full HRPO claim 必须有 simulator/on-policy rollout 证据。

### Stage4: Transition-Conditioned Next-Skill L_act

定位：

- 不局限 AppWorld。
- primary data 来自 TRAJECT sequential 与 ToolBench-G3 simulated trajectories。
- AppWorld verified pairs 只作为 secondary adaptation / combination plot。
- canonical Stage4 从 Stage0 routing checkpoint + Stage3 head checkpoint 初始化，不使用
  Qwen external encoder wrapper。

训练：

- `L_act = -log P_T(a^+_{t+1} | m_hat_{t+1}, C_{t+1})`
- 可加 pairwise ranking loss。
- `C_{t+1}` 必须包含 positive next skill；若 deterministic inject，必须记录 provenance。

验收：

- TRAJECT sequential next-skill acc / recall@K 高于 Stage3。
- ToolBench-G3 next-tool selection 不退步。
- AppWorld secondary action acc 高于 Stage3，用于 §15.5.5，不进入三主表。

---

## 4. 文件结构与职责

计划优先修改或新增这些文件。实际执行时每个任务需要先写 failing test，再改实现。

- `scripts/build_clstr_unified_pretrain.py`: 生成 canonical skill pool、qrels、dedup manifest。
- `scripts/audit_clstr_unified_training_readiness.py`: 把 skill pool borderline review / placeholder / qrels canonical 状态作为 Stage0 blocker。
- `clstr/skillret_official.py`: SkillRouter-compatible query/skill encoding 协议参考。
- `clstr/appworld_skillrouter_base.py`: SkillRouter-base scoring 参考。
- `clstr/retrieval_warmup.py`: 当前 retrieval warmup 入口，需重构为 Stage0 coarse retriever。
- `scripts/run_skillret_retrieval_warmup.py`: CLI 需改名或增加 Stage0 mode。
- `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh`: 需替换为 Stage0 bi-encoder sbatch。
- `clstr/stage1_quality_gate.py`: 改成 Stage0 coarse-recall quality gate，关注 Recall@20/50/100。
- `clstr/training_monitor.py`: 保留持续 loss/recall 曲线输出。
- `docs/clstr_v4_plan.md`: 后续同步阶段命名修订。
- `description.md`: 记录每个重大方向变更和实验结论。

---

## 5. 任务清单

### Task 1: 阻断未清理 skill pool 进入 Stage0

**Files:**

- Modify: `scripts/audit_clstr_unified_training_readiness.py`
- Modify: `tests/test_unified_training_readiness.py`

- [x] 写 failing test：当 manifest 中 `skill_pool.borderline_review.status = pending_manual_or_llm_review` 且 `candidate_count > 0` 时，Stage0/Stage1 gate 必须 blocked。
- [x] 实现 `_skill_pool_quality_report` 读取 `borderline_review.status`、`candidate_count`、`method`。
- [x] 将 blocker 命名为 `skill_pool_borderline_review_pending`。
- [x] 修改旧 placeholder skill pool 测试，让 Stage0/Stage1 也被阻断，而不是只阻断 Stage2。
- [x] 运行：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_blocks_stage1_when_skill_pool_borderline_review_pending \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_blocks_stage2_when_skill_pool_has_placeholder_records \
  -q
```

Expected: `2 passed`

Verified 2026-05-26: included in `tests/test_unified_training_readiness.py` target run; passed.

### Task 2: 完成 canonical skill pool 与 qrels 数据审计

**Files:**

- Modify: `scripts/build_clstr_unified_pretrain.py`
- Create: `scripts/audit_clstr_skill_pool_quality.py`
- Test: `tests/test_skill_pool_quality_audit.py`

- [x] 写测试：重复 skill name + 相同 param signature 必须 merge 到同一 canonical id。
- [x] 写测试：同 query 的多个等价 positive skill 必须保留为 multi-positive qrels。
- [x] 写测试：positive skill 不存在于 canonical pool 时 audit fail。
- [x] 实现 audit 输出：

```json
{
  "status": "ok",
  "raw_skill_count": 0,
  "canonical_skill_count": 0,
  "merged_group_count": 0,
  "borderline_review_status": "complete",
  "missing_positive_ref_count": 0,
  "placeholder_skill_count": 0,
  "duplicate_canonical_key_count": 0
}
```

- [x] 运行：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_skill_pool_quality_audit.py -q
```

Expected: all tests passed.

Verified 2026-05-26: `tests/test_skill_pool_quality_audit.py` passed.

Verified 2026-05-27: conservative borderline review 已接入
`scripts/audit_skill_dedup_borderline.py`、`scripts/audit_clstr_unified_training_readiness.py`
和 sbatch 入口。真实 20000 条 borderline candidates 全部保守标记为
`reviewed_keep_separate`，0 条 high-confidence unresolved duplicates。当前 readiness
报告 `outputs/clstr_unified_readiness_audit/stage0_foundation_current_readiness.json`
显示 skill pool gate 已解除，Stage0 coarse retriever 可提交；Stage2/3/4 仍等待
Stage0/Stage2/Stage3 checkpoint。

### Task 3: 新建 Stage0 SkillRouter-compatible bi-encoder 配置
可参考/data/home/scyb713/run/xzf/AAAI/autodl-tmp/skillrouter
**Files:**

- Modify: `clstr/retrieval_warmup.py`
- Modify: `scripts/run_skillret_retrieval_warmup.py`
- Create: `scripts/run_clstr_stage0_biencoder_train.py`
- Create: `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- Test: `tests/test_stage0_biencoder_protocol.py`

- [x] 写测试：Stage0 tokenizer 必须 left padding。
- [x] 写测试：Stage0 pooling 必须 last token。
- [x] 写测试：query/skill embedding 必须 L2 normalized 后再 scoring。
- [x] 写测试：projection / SkillTable adapter identity init 时，Stage0 score 与 SkillRouter frozen score 在同输入上数值接近。
- [x] 实现 Stage0 CLI 参数：

```bash
--data_root data/clstr_unified_pretrain_v2_toolbench_g3_traject_split
--output_dir outputs/clstr_unified_stage0_biencoder
--model_name_or_path .cache/hf_models/SkillRouter-Embedding-0.6B
--query_text_format skillrouter
--skill_text_format skillrouter
--encoder_pooling last_token
--tokenizer_padding_side left
--normalize_embeddings
--projection_init identity
--skill_table_adapter_init identity
--batch_size 128
--gradient_accumulation_steps 2
--max_updates 5000
--save_every 500
--plot_every 50
```

- [x] sbatch 必须使用 GPU 分区，例如：

```bash
#!/bin/bash
module load miniforge3/25.11.0-1
module load cuda/12.1
source activate env
export PYTHONUNBUFFERED=1
python scripts/run_clstr_stage0_biencoder_train.py "$@"
```

- [x] 运行：

```bash
bash -n scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_biencoder_protocol.py -q
```

Expected: shell syntax ok, tests passed.

Verified 2026-05-26: `tests/test_stage0_biencoder_protocol.py` passed; Stage0 sbatch `bash -n` passed.

### Task 4: Stage0 quality gate 改为 coarse recall gate

**Files:**

- Modify: `clstr/stage1_quality_gate.py`
- Create or Modify: `clstr/stage0_quality_gate.py`
- Keep legacy: `scripts/audit_clstr_stage1_quality.py`
- Create: `scripts/audit_clstr_stage0_quality.py`
- Test: `tests/test_stage0_quality_gate.py`

- [x] 写测试：只有 `Recall@1` 提升但 `Recall@50` 不达标时 gate fail。
- [x] 写测试：`Recall@50/100` 达标、loss 下降、source coverage 完整时 gate ok。
- [x] gate 默认检查：

```text
final checkpoint exists
training_metrics.jsonl exists
loss_curve.svg exists
max_step >= configured minimum
last-window recall_at_20/50/100 present
observed_sources include skillret/toolret_training/toolbench_g3/traject_bench
same-pool SkillRouter frozen baseline exists
Stage0 full retrieval eval metrics exists
Stage0 full retrieval Recall@20/50/100 >= baseline Recall@20/50/100 - tolerance
```

- [x] 运行：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_quality_gate.py -q
```

Expected: all tests passed.

Verified 2026-05-26: `tests/test_stage0_quality_gate.py` passed.

Current revision 2026-05-27: Stage0 gate 不再用 `training_metrics.jsonl` 的 batch
recall 与 SkillRouter baseline 比较。训练 batch recall 只作为 learning signal 写入
`coarse_recall_summary.training_batch_tail`；baseline comparison 必须使用
`outputs/clstr_unified_stage0_biencoder/full_retrieval_eval/metrics.json` 与
`outputs/clstr_stage0_skillrouter_frozen_baseline/metrics.json`。缺少 Stage0 full eval
metrics 时 gate 返回 `missing_stage0_full_eval_metrics`。

### Task 5: 建立 same-pool SkillRouter frozen baseline

**Files:**

- Create: `scripts/run_clstr_stage0_skillrouter_frozen_baseline.py`
- Create: `scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh`
- Test: `tests/test_stage0_skillrouter_baseline.py`

- [x] 写测试：baseline 使用 canonical skill pool，而不是 SkillRouter 原始 pool。
- [x] 写测试：baseline 输出 run file 与 qrels 使用同一 canonical id 空间。
- [x] 输出：

```text
outputs/clstr_stage0_skillrouter_frozen_baseline/run.tsv
outputs/clstr_stage0_skillrouter_frozen_baseline/metrics.json
outputs/clstr_stage0_skillrouter_frozen_baseline/manifest.json
```

- [x] 运行：

```bash
bash -n scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_skillrouter_baseline.py -q
```

Expected: shell syntax ok, tests passed.

Verified 2026-05-26: `tests/test_stage0_skillrouter_baseline.py` passed; baseline sbatch `bash -n` passed.

### Task 6: Stage1/2 改为消费 Stage0 top-M candidate set

**Files:**

- Create: `scripts/run_clstr_stage1_heads_init.py`
- Modify: `scripts/run_clstr_stage2_full_base_train.py`
- Create: `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh`
- Modify: `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- Modify: `clstr/full_base_train.py`
- Create: `clstr/stage_checkpoint_init.py`
- Test: `tests/test_clstr_topm_candidate_handoff.py`
- Test: `tests/test_stage_checkpoint_init.py`

- [x] 写测试：Stage1/Stage2 输入 candidate set 必须来自 Stage0 top-M 或在线 Stage0 scorer。
- [x] 写测试：当 positive 不在 top-M 时样本被 skip 或 deterministic inject，并记录 provenance。
- [x] 写测试：Stage2 不允许从全 skill pool 直接监督 top1，除非显式 `--allow_full_pool_stage2_debug`。
- [x] 写测试：Stage1 heads-init 产出 `clstr_stage1_heads-step*.pt`，且阶段标记为 `clstr_stage1_heads_init`。
- [x] 实现 handoff manifest：

```json
{
  "stage0_checkpoint": "outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt",
  "candidate_source": "stage0_topm",
  "top_m": 100,
  "positive_missing_policy": "skip_or_inject_with_provenance"
}
```

- [x] 运行：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_clstr_topm_candidate_handoff.py -q
```

Expected: all tests passed.

Verified 2026-05-26: `tests/test_clstr_topm_candidate_handoff.py` passed.

Verified 2026-05-27: Stage1 heads-init runner 已独立于 Stage2；Stage2 canonical runner
新增 `--stage1_checkpoint_path`，语义为从 Stage1 checkpoint 继续同一个 CLSTRModel。
`ROUTING_INIT_MANIFEST` 不再出现在 unified Stage1/Stage2 sbatch。`tests/test_stage_checkpoint_init.py`
和 Stage1/Stage2 handoff 相关测试已通过。

### Task 7: 更新 sbatch 安全入口

**Files:**

- Create/Modify: `scripts/submit_clstr_stage1_after_stage0_gate.sh`
- Create/Modify: `scripts/submit_clstr_stage2_after_stage1_gate.sh`
- Modify legacy alias: `scripts/submit_clstr_stage2_after_stage0_gate.sh`
- Create: `scripts/audit_clstr_stage1_heads_quality.py`
- Modify: `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- Create: `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh`
- Test: `tests/test_sbatch_scripts.py`

- [x] 写测试：Stage0 gate 不通过时不能提交 Stage1。
- [x] 写测试：Stage0→Stage2 旧入口只做兼容转发，不再作为 canonical handoff。
- [x] 写测试：Stage2 sbatch 必须检查 Stage1 heads checkpoint；Stage1/Stage2 仍显式传递 Stage0 full eval gate 信息。
- [x] 写测试：Stage2/readiness/pipeline 必须显式传递 `STAGE0_EVAL_METRICS_PATH`，
  防止绕过 Stage0 full retrieval eval gate。
- [x] 写测试：主线训练 sbatch 统一 source `_clstr_gpu_env.sh`，不能再复制旧
  `source activate env`，也不能覆盖集群 GPU/CPU 默认配比。
- [x] 写测试：主线 CLSTR GPU sbatch 不能依赖 `$(dirname "$0")` 定位共享环境脚本；
  Slurm 会把 `$0` 指向 spool 中的 `slurm_script`，因此必须通过 `PROJECT_ROOT`
  定位 `${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh`。
- [x] 写测试：Stage3/Stage4 unified sbatch 走 CLSTR-native canonical wrapper，不暴露
  Qwen wrapper、`QWEN_MODEL_PATH` 或 `--qwen_model_path`。
- [x] 写测试：Stage1/Stage2 支持 `MAX_ROWS`/`--max_rows` 小样本 smoke；submit wrapper
  在 `sbatch` 前调用 `scripts/guard_clstr_job_budget.sh`，默认 `MAX_ACTIVE_JOBS=4`。
- [x] 新增 `scripts/audit_clstr_training_health.py`，用于提交后早期检查
  `training_metrics.jsonl`、`latest.pt` 和 loss 是否正常；异常时应立即人工 `scancel`。
- [x] 运行：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_sbatch_scripts.py -q
bash -n scripts/submit_clstr_stage2_after_stage0_gate.sh \
  scripts/submit_clstr_stage1_after_stage0_gate.sh \
  scripts/submit_clstr_stage2_after_stage1_gate.sh \
  scripts/sbatch/run_clstr_unified_stage1_heads_init.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
```

Expected: tests passed, shell syntax ok.

Verified 2026-05-26: `tests/test_sbatch_scripts.py` passed; submit/sbatch `bash -n` passed.

Verified 2026-05-27: after Slurm `$0` spool failure in job `77835`, targeted red/green
check added the `PROJECT_ROOT` source assertion, then all `tests/test_sbatch_scripts.py`
passed and mainline CLSTR sbatch `bash -n` passed.

Current revision 2026-05-27: Stage0 不再直接 hand off 到 Stage2。canonical handoff
现在是：

```bash
bash scripts/submit_clstr_stage1_after_stage0_gate.sh
bash scripts/submit_clstr_stage2_after_stage1_gate.sh
```

`scripts/submit_clstr_stage2_after_stage0_gate.sh` 仅保留为 deprecated compatibility
alias，并转发到 Stage1。Stage1 是同一个 CLSTR 模型上的 heads initialization：冻结
Stage0 retrieval foundation，在 Stage0 top-M candidates 内训练 policy / belief /
transition / STOP heads。Stage2 从 Stage1 checkpoint 继续训练同一个 CLSTRModel；若内部
为了恢复冻结 routing foundation 需要读取 Stage0 checkpoint，那只是 checkpoint restore
细节，不是两个模型相加或 ensemble。

Current revision 2026-05-27: canonical Stage3/Stage4 GPU entries are
`scripts/run_clstr_unified_stage3_hrpo.py` and `scripts/run_clstr_stage4_act_train.py`;
their sbatch wrappers initialize from Stage0 routing checkpoint plus Stage2/Stage3 head
checkpoint. Qwen wrappers remain legacy/reference-only and are not mainline unified
training entries.

Current revision 2026-05-27: canonical Stage2 handoff 和 unified readiness audit 都要求
`STAGE0_EVAL_METRICS_PATH`，默认指向
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/full_retrieval_eval/metrics.json`。
Stage0 final checkpoint 默认从
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
读取。Stage0 final checkpoint 产出后，需要先运行
`scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh` 生成同池 run/eval metrics，再进入
`scripts/submit_clstr_stage1_after_stage0_gate.sh`。

Current revision 2026-05-27: 为防止下游误读旧失败 Stage0，
`scripts/run_clstr_stage0_biencoder_train.py`、Stage0 full retrieval eval、Stage2/3/4
sbatch、Stage2/3/4 submit wrapper、unified readiness audit、ToolRet/TRAJECT/ToolBench-G3
retrieval export wrapper 的 canonical 默认 Stage0 路径已统一切到
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE`。旧
`outputs/clstr_unified_stage0_biencoder` 目录只保留为历史诊断产物；若确需复现实验，必须显式
设置对应 `OUTPUT_DIR`、`STAGE0_OUTPUT_DIR` 或 `CHECKPOINT_PATH`。

Current revision 2026-05-27: 再次检查 `77840/77841` 时，`77840` 仍在运行，
`77841` 仍为依赖等待。`training_metrics.jsonl` 已写入约 1200+ optimizer steps；
最新观察约 `step=1229/5000`，tail-100 batch 指标约为
`loss=9.8338`、`recall_at_20=0.7123`、`recall_at_50=0.8006`、
`recall_at_100=0.8542`，且四类训练来源均已覆盖。该 batch 指标只说明训练仍在推进，
不能替代 full retrieval gate。当前状态报告已写入
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/stage0_quality_gate_current.json`
和 `outputs/clstr_unified_readiness_audit/stage0_v2_current_readiness.json`；两者均为
`action_required`，原因是 v2 final checkpoint 与 full retrieval eval metrics 尚未产生。
因此仍不提交 Stage2/3/4，也不重复提交 Stage0 v2 或 full retrieval eval。

Current revision 2026-05-27: Stage0 v2 full training 与 full retrieval eval 已完成，
重新运行 Stage0 quality gate 后 `status=ok`。同池 full retrieval coarse recall 为
`Recall@20/50/100 = 0.714572 / 0.831686 / 0.913444`，高于 same-pool SkillRouter
frozen baseline `0.706692 / 0.811626 / 0.884804`。因此 Stage0 foundation gate 已解除，
下一步应进入 Stage1 heads initialization，而不是直接进入 Stage2。

Current revision 2026-05-27: Stage1 job `77911` 失败后，定位到训练侧 JSONL reader 使用
`read_text().splitlines()`，会把 JSON 字符串中的 Unicode line separator（真实数据中
存在 U+0085）拆成伪行，导致合法 JSONL 训练时报 `Unterminated string`。已将
`clstr/full_base_train.py` 与 `clstr/full_base_preprocess.py` 改为逐文件行读取，并用
回归测试覆盖。后续不再直接提交 full job；full job 由用户最终确认。

### Task 8: 文档同步

**Files:**

- Modify: `docs/clstr_v4_plan.md`
- Modify: `description.md`
- Modify: `README.md`

- [x] 在 `docs/clstr_v4_plan.md` 中把训练流程改成 `Pre-Stage0 -> Stage0 coarse retriever -> Stage1 heads init -> Stage2 base -> Stage3 act/hrpo -> Stage4 L_act`。
- [x] 明确 Stage0 不含 cross-encoder/listwise reranker。
- [x] 明确 SkillRouter cross-encoder/listwise reranker 是 optional baseline / upper bound。
- [x] 在 `description.md` 追加 Stage0 重构原因、旧 Stage1 弱结果、Stage2 主线从 Qwen wrapper 切回 CLSTR-native 的原因与边界。
- [x] 在 `README.md` 写清楚当前推荐执行顺序和 sbatch 入口。
- [x] 运行：

```bash
git diff --check -- docs/clstr_v4_plan.md description.md README.md
```

Expected: no whitespace errors.

Verified 2026-05-26: `git diff --check -- docs/clstr_v4_plan.md description.md README.md` passed.

---

## 6. 第一轮执行顺序

第一轮只做 foundation repair，不提交长训练：

1. Task 1: readiness gate 阻断未清理 skill pool。
2. Task 2: skill pool / qrels audit。
3. Task 5: same-pool SkillRouter frozen baseline。
4. Task 3: Stage0 bi-encoder protocol。
5. Task 4: Stage0 coarse recall gate。
6. Task 8: 文档同步。

第一轮完成后再提交短 Stage0 smoke：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
```
没有问题可以进行stage0的完整训练，smoke不该有测试以外的任何用途，只有完整训练的stage0才能流入后续的stage

H200 空闲时可切换：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
```

短任务检查间隔不少于 15 分钟；长任务检查间隔不少于 30 分钟。

---

## 7. 进入下游训练的硬条件

只有同时满足以下条件，才能进入 Stage1：

- skill pool borderline review complete。
- no placeholder skill records。
- qrels 全部 canonical。
- same-pool SkillRouter frozen baseline 已生成。
- Stage0 full retrieval eval metrics 已生成，且同池 full retrieval
  `Recall@20/50/100` 不低于 same-pool SkillRouter baseline。
- Stage0 checkpoint、latest checkpoint、training metrics、loss/recall curve 均存在。
- Stage0 quality gate `status=ok`。

若任一条件不满足，不提交 Stage1/2/3/4。

只有 Stage1 heads-init checkpoint、`training_metrics.jsonl`、`loss_curve.svg` 与
Stage1 quality gate 均存在且 `status=ok` 后，才能进入 Stage2。Stage2 应从 Stage1
checkpoint 继续同一个 CLSTRModel，而不是绕过 Stage1 或把旧 Stage1 retrieval warmup
当作有效前置。

当前实现状态：`scripts/audit_clstr_unified_training_readiness.py` 已升级为显式
`stage1_heads_init` gate。`stage1_retrieval_warmup` 仍作为 Stage0 legacy alias
保留在报告中用于历史兼容，但 `stage2_full_base` readiness 必须同时看到可用 Stage0
checkpoint、Stage0 quality gate、Stage1 heads-init checkpoint 与 Stage1 quality
gate，不能再把 Stage0 checkpoint 当作 Stage1 前置。

执行约束：full job 只由用户最终确认提交。后续训练阶段必须先通过本地小样本 smoke 与
短作业 smoke；提交后必须早期检查日志和关键产物，异常时立即中止，不允许无人看管地长跑。

---

## 8. 论文口径

推荐论文叙述：

> Stage0 adopts a SkillRouter-compatible bi-encoder retrieval foundation to ensure fair and strong large-pool candidate recall. CLSTR is then trained on top of the recalled candidate set to perform recurrent, history-conditioned skill selection.

中文口径：

> Stage0 采用 SkillRouter 兼容的 bi-encoder 粗召回协议，保证大规模 skill pool 召回能力不弱；CLSTR 的贡献从后续阶段开始，在 top-M 候选内通过 `m_t`、transition 和 policy head 做多步动态选择。

不能写：

- CLSTR 从零训练的大池检索器已经超过 SkillRouter。
- offline HRPO-style surrogate 等同于 full on-policy HRPO。
- routing/proxy 指标等同于 TRAJECT / ToolBench-G3 official task success。
- AppWorld secondary plot 等同于三主表结果。

---

## 9. Self-Review

- Spec coverage: 本计划覆盖了新提出的 Stage0=SkillRouter bi-encoder 问题、cross-encoder 暂缓问题、Stage2 作为 CLSTR 动态 reranker 的定位，以及 v4 plan 中未完成的数据、训练、gate、eval 项。
- Placeholder scan: 文档没有遗留占位符；所有任务都有明确文件、测试、命令和验收。
- Scope check: 当前计划只要求先修 foundation，不把 Stage2/3/4 长训练混入第一轮执行。
- Safety check: 所有写入和执行路径均限定在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr`；GPU 任务使用 sbatch。
