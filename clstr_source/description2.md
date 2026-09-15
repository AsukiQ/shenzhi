## 2026-05-25 v4 Phase A implementation checkpoint

本记录对应 `docs/clstr_v4_plan.md` 的 v4 Phase A（F1 + F2-v2 + G1，并补齐 D5/D6/G2 的最小可训练实现）。本次只完成代码与单元测试级验证，尚未启动 sbatch/GPU 训练，也尚未构建 unified v2 大训练集。

### 已完成的主要改动

1. SkillTable / routing logits
   - 在 `clstr/encoders.py` 中补齐 SkillRouter 同构 retrieval logits：L2 normalize query/skill embedding、`logit_scale_retr`、`skill_bias_retr`。
   - 将 belief logits 与 retrieval logits 拆分：新增 `logit_scale_belief`、`skill_bias_belief`，`subspace_obs` 优先使用 `belief_logits`。
   - 增加 `set_retrieval_trainable()` / `set_belief_scale_trainable()`，Stage 3 可硬冻 retrieval path，同时保留 belief scale/bias 的 ablation 空间。

2. Transition / action representation
   - `CLSTRModel` 新增 `action_proj(E)`，`TransitionPredictor` 支持不再内部持有独立 `action_emb`。
   - `CLSTRModel.action_embeddings(action_ids)` 统一从 `skill_table.E` 取 skill/action embedding，再经 `action_proj` 投影；stop/out-of-range action 回退为零向量。
   - 新增 `clstr/transition_utils.py`，让 legacy 训练/评测入口在真实 CLSTRModel 上优先使用 `model.action_embeddings(labels)`，没有该接口的旧 fake/legacy model 保持原行为。
   - 已接入 `clstr/full_base_train.py`、`clstr/losses.py`、`clstr/aux_pretrain.py`、`clstr/alfworld_eval.py`。
   - `clstr/online_hrpo.py` 的 action-level controller-complete 路径不再把全 0 dummy label 传给 transition，而是直接传候选 action embedding，避免所有候选共享同一个 transition action 输入。

3. Policy head / candidate sampling / cache
   - `CLSTRModel._policy_skill_logits` 支持 v4 的 `skill_head_context=belief_residual`，即使用 `m_t - h_t`。
   - 训练时 candidate sampling 改为 top-M widen 后 multinomial no-replacement；eval 保持 deterministic top-K。
   - 增加 per-task cross-encoder cache，key 使用 `(state_text, skill_idx)`。

4. 训练配置兼容
   - `configs/model/*.yaml` 移除显式 `tau_s`，新增 retrieval/belief scale 初值、candidate widen/sampling temperature、`skill_head_context: belief_residual`。
   - `clstr/train.py`、`clstr/full_base_train.py`、`clstr/appworld_act_hrpo.py`、`clstr/online_hrpo.py` 已把 `action_proj` 纳入对应训练/冻结/reference module 逻辑。

### 新增/更新测试

- 新增 v4 测试文件：
  - `tests/test_v4_skill_table_logits.py`
  - `tests/test_v4_skill_table_freeze.py`
  - `tests/test_v4_candidate_sampling.py`
  - `tests/test_v4_action_proj_sharing.py`
  - `tests/test_v4_belief_residual.py`
  - `tests/test_v4_cross_encoder_cache.py`
- `tests/test_v4_action_proj_sharing.py` 新增回归覆盖：
  - full-base transition helper 必须优先使用 `action_embeddings(labels)`；
  - ALFWorld controller transition/trans_head candidate embedding 必须优先使用 `action_embeddings(labels)`。
- `tests/test_online_hrpo.py` 新增回归覆盖：
  - HRPO controller-complete transition 必须接收候选 action embedding，而不是常量 dummy label。

### 验证证据

使用环境：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest ...
```

已运行并通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_v4_skill_table_logits.py \
  tests/test_v4_skill_table_freeze.py \
  tests/test_v4_candidate_sampling.py \
  tests/test_v4_action_proj_sharing.py \
  tests/test_v4_belief_residual.py \
  tests/test_v4_cross_encoder_cache.py -q
# 24 passed, 2 warnings
```

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_belief.py \
  tests/test_model_pipeline.py \
  tests/test_heads.py \
  tests/test_phase1_closed_loop_gradient.py \
  tests/test_losses.py \
  tests/test_rollout.py \
  tests/test_appworld_clstr_eval.py -q
# 44 passed, 2 warnings
```

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_train_smoke.py::test_configure_stage2_base_keeps_backbones_frozen_and_heads_trainable \
  tests/test_train_smoke.py::test_configure_stage3_joint_keeps_skill_table_embeddings_frozen \
  tests/test_online_hrpo.py \
  tests/test_full_base_train.py::test_full_base_transition_skill_ce_prefers_native_trans_head_over_skill_table_logits \
  tests/test_full_base_train.py::test_full_base_loss_uses_official_replay_prefix_to_reconstruct_current_belief \
  tests/test_alfworld_eval.py::test_controller_component_scorer_uses_action_embedding_for_transition_head \
  tests/test_alfworld_eval.py::test_controller_component_scorer_uses_trained_trans_and_skill_heads \
  tests/test_aux_trajectories.py::test_aux_pretrain_next_action_ce_replaces_cosine_transition_alignment -q
# 15 passed, 2 warnings
```

```bash
git diff --check -- clstr/encoders.py clstr/belief.py clstr/model.py clstr/train.py \
  clstr/full_base_train.py clstr/online_hrpo.py clstr/appworld_act_hrpo.py \
  clstr/alfworld_eval.py clstr/losses.py clstr/aux_pretrain.py clstr/transition_utils.py \
  configs/model tests/test_v4_*.py tests/test_online_hrpo.py description2.md
# no output
```

## 2026-05-26 v4 训练可观测性与 rolling checkpoint 修复

### 背景

旧 Stage1/Stage2/Stage3/Stage4 训练循环存在共同问题：loss/metric 只在内存里累计，
最终训练结束后才写 `train_report.json` 和最终 checkpoint。长任务如果排队很久、中途
异常或需要判断是否欠拟合，无法持续看到 loss 曲线，也没有可恢复的中间 checkpoint。
这也是 `77139` 只能看到模型加载日志、没有 loss 曲线的直接原因。

### 已完成改动

1. 新增通用训练监控器 `clstr/training_monitor.py`
   - 每步追加写 `training_metrics.jsonl`；
   - 每步刷新 `loss_curve.svg`，无 matplotlib 依赖，适合集群环境；
   - 训练开始前先写 `checkpoints/latest.pt` 的 step 0 快照；
   - 每个 step/update 后用原子替换持续覆盖 `checkpoints/latest.pt`；
   - checkpoint payload 记录 `training_metrics_path` 和 `loss_curve_path`。

2. Stage1 retrieval warmup 已接入
   - 文件：`clstr/retrieval_warmup.py`
   - 覆盖 unified v2 / SKILLRET retrieval warmup；
   - `train_report.json` 新增：
     - `training_metrics_path`
     - `loss_curve_path`
     - `latest_checkpoint`

3. Stage2 full base 已接入
   - 文件：`clstr/full_base_train.py`
   - 每个 supervised step 记录总 loss 与各 loss term；
   - rolling checkpoint 仍沿用现有冻结 backbone / routing foundation 排除策略，
     不反复写入 Qwen 冻结权重。

4. Stage3 unified HRPO 已接入
   - 文件：`clstr/stage3_unified_hrpo.py`
   - 每个 update 写 `training_metrics.jsonl` 与 `loss_curve.svg`；
   - `latest.pt` 记录当前 update、HRPO metrics、freeze report 和 checkpoint init report。

5. Stage4 transition-conditioned ACT 已接入
   - 文件：`clstr/stage4_act_train.py`
   - 每个 ACT step 记录 loss / next-skill recall / MRR / candidate count；
   - `latest.pt` 持续覆盖，便于中断后检查当前训练状态。

6. StableToolBench CLSTR-routed raw generation 封装补齐
   - 新增 `scripts/sbatch/run_stabletoolbench_clstr_routed_local_raw_generation.sh`；
   - 默认读取 CLSTR top-K 派生 query：
     `outputs/toolbench_g3/stabletoolbench_clstr_topk/${TEST_SET}.json`；
   - 默认读取 CLSTR top-K 派生 toolenv：
     `outputs/toolbench_g3/stabletoolbench_clstr_topk_toolenv/tools`；
   - 不再默认使用官方 full `solvable_queries/test_instruction/${TEST_SET}.json`
     或 `StableToolBench/toolenv/tools`，避免把官方 full api_list raw generation
     误报为 CLSTR-routed SoPR。

### 作业清理

已取消重复旧链路作业：

```text
77139 77151 77156 77275 77152 77157 77276 77277 77278 77283 77284 77285
```

保留 v4 split-safe 主链：

```text
77401 Stage1 retrieval warmup
77411 Stage2 full base
77412 Stage3 HRPO
77413 Stage4 ACT
77414 ToolRet eval
77415 ToolBench-G3 routing eval
77416 TRAJECT eval
```

### 后续检查方式

训练启动后优先看以下文件，不需要频繁 `squeue`：

```bash
tail -n 5 outputs/<stage_output>/training_metrics.jsonl
ls -lh outputs/<stage_output>/loss_curve.svg
ls -lh outputs/<stage_output>/checkpoints/latest.pt
```

短任务建议 15 分钟左右检查一次，长任务建议 30 分钟或更久检查一次。

### 2026-05-26 Stage1 setup 盲区修复与 v4 主链重提

#### 发现的问题

`77401` 启动后可以看到 Qwen 权重加载和 `selected_skills.jsonl` 写出，但没有
`training_metrics.jsonl` / `latest.pt`。原因不是训练 loop 的 per-step 监控失败，
而是 Stage1 在进入训练 loop 前需要构建大规模 skill table embedding；旧代码中
`TrainingMonitor` 在 `model.rebuild_skill_table()` 之后才创建，因此 skill embedding
构建阶段没有可观察状态。

进一步检查发现 Stage1 warmup 还有一个效率问题：`CLSTRModel(...)` 默认会在构造时
构建一次 skill table embedding，随后 `run_skillret_retrieval_warmup()` 又调用
`model.rebuild_skill_table()`，等于可能对 50k+ skill pool 做两次 embedding。

#### 已完成修复

1. `clstr/retrieval_warmup.py`
   - 新增 `setup_status.jsonl`，按阶段写入：
     - `data_loaded`
     - `selected_skills_written`
     - `model_initialized`
     - `skill_table_rebuild_started`
     - `skill_table_rebuilt`
     - `training_started`
   - `TrainingMonitor` 提前到数据加载后创建；
   - 在模型初始化后、skill table rebuild 前就写入 step 0 `checkpoints/latest.pt`；
   - Stage1 warmup 内部强制 `effective_defer_skill_table_init=True`，避免构造模型时
     先构建一次 embedding、随后又 rebuild 一次。

2. `tests/test_skillret.py`
   - 增加 Stage1 setup 可观测性断言：
     - `setup_status_path` 存在；
     - setup phase 按顺序出现；
     - `training_metrics.jsonl`、`loss_curve.svg`、`checkpoints/latest.pt` 仍正常产出。

#### 作业决策

由于 `77401` 是修复前启动的，仍使用旧代码，且可能正在重复构建 skill embedding，
因此已取消修复前 v4 主链：

```text
77401 77411 77412 77413 77414 77415 77416
```

按更新后的代码重新提交 v4 主链：

```text
77445 Stage1 retrieval warmup
77446 Stage2 full base              afterok:77445
77447 Stage3 HRPO                   afterok:77446
77448 Stage4 ACT                    afterok:77447
77449 ToolRet retrieval eval         afterok:77445
77450 ToolBench-G3 routing eval      afterok:77445
77451 TRAJECT routing/proxy eval     afterok:77445
77452 StableToolBench CLSTR top-k    afterok:77445
```

`77384` TRAJECT test import 已完成，未重复提交。

#### 当前检查方式

`77445` 启动后，优先看：

```bash
tail -n 20 outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/setup_status.jsonl
tail -n 5 outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/training_metrics.jsonl
ls -lh outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/checkpoints/latest.pt
ls -lh outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/loss_curve.svg
```

注意：`setup_status.jsonl` 出现但 `training_metrics.jsonl` 为空时，说明任务仍在模型加载
或 skill table embedding 构建阶段；不是训练 loop 卡死。

### 当前未完成项

完整 v4 目标尚未完成，下一阶段应转向 Phase B/C：

1. 构建 unified CLSTR training dataset v2：
   - 统一 TRAJECT-Bench / ToolBench-G3 / ToolRet / SKILLRET / 现有轨迹数据；
   - 输出统一 skill vocab、retrieval pairs、trajectory step-pairs；
   - 严格做 leakage audit，确保各 benchmark 的 test query/task ids 不进入训练。
2. 在 unified v2 上按 Stage 1/2/3/4 训练：
   - Stage 1 retrieval warmup；
   - Stage 2 supervised transition + policy CE；
   - Stage 3 HRPO / controller training；
   - Stage 4 transition-conditioned next-skill training，不局限 AppWorld。
3. 完成主 benchmark 评测：
   - TRAJECT-Bench；
   - ToolBench-G3；
   - ToolRet；
   - AppWorld 只作为 secondary transfer / combination plot。

## 2026-05-25 v4 Phase B/C dataset-builder checkpoint

本记录对应 `docs/clstr_v4_plan.md` 的 Phase B/C 起步：先把 unified CLSTR training dataset v2 的可复现构建入口接通。由于当前本地还没有 TRAJECT-Bench / ToolBench-G3 / ToolRet 数据目录，本次没有伪造这些数据，而是在 manifest 中显式记录缺失源，保证不会误认为主战场数据已经进入训练。

### 已完成的主要改动

1. `scripts/build_clstr_unified_pretrain.py`
   - 新增可调用函数 `build_unified_pretrain(repo_root, output_dir, schema_version)`，不再只能通过 CLI 直接写死 repo 路径。
   - 默认输出改为 `data/clstr_unified_pretrain_v2`，默认 `schema_version=v2`，避免误覆盖旧 v1。
   - v2 输出文件包括：
     - `trajectories.jsonl`
     - `retrieval.jsonl`
     - `skill_pool.jsonl`
     - `source_inventory.jsonl`
     - `leakage_audit.json`
     - `manifest.json`
   - `skill_pool.jsonl` 目前使用 identity skill_id dedup，汇总已被 trajectory/retrieval 使用到的 skill，不把未使用 skill 写入训练池。
   - `source_inventory.jsonl` 显式记录 `traject_bench`、`toolbench_g3`、`toolret_training` 是否 available；当前缺失时写入 `source_path_missing_or_not_downloaded`。
   - `leakage_audit.json` 在写出后再次扫描输出，确认 AppWorld dev/test/challenge task ids 与 SKILLRET test query ids 没有进入训练输出。

2. `scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh`
   - 新增 sbatch 入口，符合集群约束。
   - 固定在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr` 内运行。
   - 设置 `HF_ENDPOINT=https://hf-mirror.com`，并把 cache 放在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache`。
   - 只构建数据，不启动训练。

3. `tests/test_unified_pretrain_v2.py`
   - 新增测试覆盖：
     - v2 builder 会输出 skill pool / source inventory / leakage audit；
     - AppWorld dev/test task 和 SKILLRET test query 会被过滤；
     - 未下载的 TRAJECT/ToolBench/ToolRet 会被显式标记 missing；
     - sbatch 脚本使用 repo-local 路径和 v2 schema。

### 验证证据

已运行并通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py -q
# 2 passed
```

```bash
python -m py_compile scripts/build_clstr_unified_pretrain.py
# exit 0
```

```bash
git diff --check -- scripts/build_clstr_unified_pretrain.py \
  scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh \
  tests/test_unified_pretrain_v2.py description2.md
# no output
```

已做 CLI 小样本 smoke，数据写在 `.codex_tmp/unified_v2_smoke/repo`，命令：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python \
  scripts/build_clstr_unified_pretrain.py \
  --repo_root .codex_tmp/unified_v2_smoke/repo \
  --output_dir data/clstr_unified_pretrain_v2 \
  --schema_version v2
```

smoke manifest 结果：

```text
status: ok
schema_version: v2
trajectory_stream.total_rows: 1
trajectory_stream.leakage_filtered: 1
retrieval_stream.total_pairs: 1
retrieval_stream.leakage_filtered: 1
skill_pool.total_skills: 3
source_inventory.missing_required_target_sources:
  - traject_bench
  - toolbench_g3
  - toolret_training
leakage_audit.status: ok
```

### 当前未完成项

- 尚未运行完整 `data/clstr_unified_pretrain_v2` 构建；应通过 sbatch 提交：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh
```

- 尚未下载/接入 TRAJECT-Bench、ToolBench-G3、ToolRet。下一步需要在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 下用 `https://ghfast.top` 或手动下载这些数据源，然后补转换器。
- 当前 `skill_pool.jsonl` 是 identity skill_id dedup，还不是 v4 计划里的 full partial-dedup（name normalize + param signature hash + borderline LLM judge）。这一步应在外部主战场数据落地后做。

## 2026-05-26 v4 Phase B/C TRAJECT + ToolRet data checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Phase B/C。当前重点是把 unified CLSTR training dataset v2 从“空骨架”推进到可消费真实主战场数据，并避免把 code-only repo 误判为可训练数据。

### 已完成的主要改动

1. TRAJECT-Bench 已接入 unified v2 builder
   - 已在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/TRAJECT-Bench/public_data` 识别到真实数据：
     - `tools/all_tools.json`: 1228 tools
     - `parallel`: 20 query files, 4000 queries
     - `sequential`: 11 query files, 1910 queries
   - `scripts/build_clstr_unified_pretrain.py` 已把 TRAJECT `tool list` 展开成 step-level trajectory rows：
     - `skill_id` 使用 `traject/<parent-tool>/<api>` 稳定生成；
     - `state_text` 包含 goal、trajectory type、domain、task 信息、previous tools；
     - `next_skill_id` / `done` / `loss_mask` 已按 step 写入；
     - `skill_pool.jsonl` 会从 `tools/all_tools.json` 写入真实 tool description，不再用 placeholder。

2. `source_inventory` 判定已收紧
   - `toolbench_g3` / `toolret_training` 不再因为只存在代码仓库而标记为 available。
   - 对目标源改为验证真实训练数据文件：
     - TRAJECT: 必须有 `public_data/tools/all_tools.json` 且有 parallel 或 sequential query 文件；
     - ToolBench-G3: 必须有完整 data/instruction + answer/toolenv/retrieval 资产，`data_example` 不算；
     - ToolRet: 必须有规范化后的 `data/toolret_training/retrieval.jsonl` 或真实下载数据文件。
   - 如果只存在 repo 但缺训练文件，`reason_if_missing` 写为 `source_exists_but_required_training_files_missing`。

3. ToolRet-Training-20w 规范化入口已新增
   - 新增 `scripts/import_toolret_training.py`：
     - 支持本地 ToolRet raw JSONL；
     - 也支持未传 `--source_path` 时通过 `datasets.load_dataset("mangopy/ToolRet-Training-20w")` 走 HF mirror；
     - 将原始 `query/prompt/pos/neg` 格式转为：
       - `data/toolret_training/retrieval.jsonl`
       - `data/toolret_training/skills.jsonl`
       - `data/toolret_training/manifest.json`
     - `pos/neg` 中的 tool text 会用 `ast.literal_eval` / JSON parse 解析，按 tool name 稳定生成 `toolret/<slug>` skill_id。
   - 新增 `scripts/sbatch/run_import_toolret_training.sh`：
     - 固定在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr` 内运行；
     - 默认输出 `data/toolret_training`；
     - 设置 `HF_ENDPOINT=https://hf-mirror.com`，cache 放在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache`；
     - 默认使用 `/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python`，因为该环境已有 `datasets`；
     - 计算节点默认要求 `SOURCE_PATH` 指向登录节点已下载好的本地 ToolRet raw JSONL；只有显式 `ALLOW_HF_STREAMING=1` 时才会尝试在线 HF streaming，避免违反“计算节点不能联网”的集群约束；
     - 只做数据导入/规范化，不启动训练。

4. unified v2 builder 已消费规范化 ToolRet
   - `build_retrieval_pairs()` 现在会读取 `data/toolret_training/retrieval.jsonl`。
   - `build_skill_pool()` 现在会读取 `data/toolret_training/skills.jsonl`。
   - ToolRet positive 和 negative skill 都会进入 `used_skill_ids`，保证后续 retrieval/ranking 训练能构造候选池。

### 已完成的真实数据构建

已通过 sbatch 提交并完成：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh
# Submitted batch job 77095
```

`data/clstr_unified_pretrain_v2/manifest.json` 当前结果：

```text
status: ok
trajectory_stream.total_rows: 273269
trajectory_stream.traject_bench.available: True
trajectory_stream.traject_bench.total_rows: 38094
trajectory_stream.traject_bench.query_count: 5870
retrieval_stream.total_pairs: 127190
skill_pool.total_skills: 11164
source_inventory.missing_required_target_sources:
  - toolbench_g3
  - toolret_training
leakage_audit.status: ok
excluded_appworld_task_id_count: 642
excluded_skillret_query_id_count: 4997
```

解释：
- TRAJECT-Bench 已进入 unified v2；
- SKILLRET train retrieval 已进入 unified v2，并过滤 4997 个 SKILLRET test qid；
- AppWorld dev/test/challenge 642 task id 没有进入 trajectory 输出；
- ToolBench-G3 仍只有 code/example，缺完整训练/评测数据；
- ToolRet benchmark repo 已有，但 ToolRet-Training-20w 真实训练数据尚未下载到本地，所以主构建仍标 missing。

### 验证证据

已运行并通过：

```bash
python -m py_compile scripts/build_clstr_unified_pretrain.py scripts/import_toolret_training.py
# exit 0
```

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py -q
# 5 passed
```

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolret_training_import.py -q
# 3 passed
```

ToolRet importer 小样本 CLI smoke 已跑通：

```bash
python scripts/import_toolret_training.py \
  --source_path .codex_tmp/toolret_smoke/raw.jsonl \
  --output_dir .codex_tmp/toolret_smoke/data/toolret_training
# status: ok; retrieval_pairs: 1; skills: 2

python scripts/build_clstr_unified_pretrain.py \
  --repo_root .codex_tmp/toolret_smoke \
  --output_dir data/clstr_unified_pretrain_v2 \
  --schema_version v2
# status: ok; retrieval_stream.total_pairs: 1; skill_pool.total_skills: 2
# source_inventory.missing_required_target_sources: [traject_bench, toolbench_g3]
```

ToolRet HF mirror 小样本探测：

```text
HF_ENDPOINT=https://hf-mirror.com
dataset: mangopy/ToolRet-Training-20w
streaming first sample readable: yes
observed fields: query, id, prompt, positive, negative
example id: train_0
```

注意：`conda run -n reasoning_trap python -c ...` 在拿到第一条样本并打印后发生过一次 `code 134` abort，因此目前不把“计算节点在线 streaming 完整导入”作为默认路径。推荐路径仍然是：登录节点先下载/导出本地 JSONL，再通过 `SOURCE_PATH=... sbatch ...` 在计算节点规范化。

### 当前未完成项

完整 v4 目标尚未完成。下一步应继续 Phase B/C：

1. 在登录节点下载真实 ToolRet-Training-20w 到 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp`，导出本地 JSONL 后再运行：

```bash
SOURCE_PATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/<toolret_train>.jsonl \
  sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_import_toolret_training.sh
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh
```

2. 获取 ToolBench-G3 完整数据（当前 repo 只有 `data_example`，不能作为训练源）。
3. 在 ToolRet / ToolBench 都落地后，推进 partial-dedup vocab v2，而不是继续使用 identity skill_id dedup。
4. 然后进入 Stage 1/2/3/4 训练与三个主 benchmark 评测。

## 2026-05-26 v4 Phase B/C ToolBench-G3 normalization checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Phase B/C。当前补齐 ToolBench-G3 完整数据下载后的规范化入口，并把规范化输出接入 unified v2 builder。注意：当前本地 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench` 仍只有 repo + `data_example`，不是完整官方 `data.zip`，因此主构建仍不会把 ToolBench-G3 标记为 available。

### 已完成的主要改动

1. 新增 `scripts/import_toolbench_g3.py`
   - 输入：完整 ToolBench `data/` 目录，要求至少包含 `instruction/G3_query.json`。
   - 输出：
     - `data/toolbench_g3/retrieval.jsonl`
     - `data/toolbench_g3/trajectories.jsonl`
     - `data/toolbench_g3/skills.jsonl`
     - `data/toolbench_g3/manifest.json`
   - 从 `instruction/G3_query.json` 中读取：
     - `query_id`
     - `query`
     - `api_list`
     - `relevant APIs`
   - skill_id 稳定规则：
     - `toolbench-g3/<tool_name_slug>/<api_name_slug>`
   - retrieval pair 规则：
     - `relevant APIs` 中的 tool/api 作为 positive；
     - 同 query 的其它 `api_list` skill 作为 negatives。
   - trajectory 规则：
     - 从 `answer/G3_answer/<query_id>_*.json` 的 DFS tree 第一条分支抽取 `Action -> Action Input -> observation`；
     - 写成 step-level trajectory rows；
     - `state_text` 包含 goal、ToolBench-G3 标记、previous tools；
     - `next_skill_id` / `done` / `loss_mask` 按 step 写入。

2. 新增 `scripts/sbatch/run_import_toolbench_g3.sh`
   - 符合集群约束，固定在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr` 内运行。
   - 默认输出 `data/toolbench_g3`。
   - 默认 Python 使用 `/data/home/scyb713/run/miniconda3/envs/reasoning_trap/bin/python`。
   - 计算节点不联网，因此必须通过 `SOURCE_ROOT` 指向登录节点已经下载/解压好的完整 ToolBench `data/` 目录。
   - 只做数据导入/规范化，不启动训练。

3. `scripts/build_clstr_unified_pretrain.py` 已消费规范化 ToolBench-G3
   - `build_trajectories()` 读取 `data/toolbench_g3/trajectories.jsonl`。
   - `build_retrieval_pairs()` 读取 `data/toolbench_g3/retrieval.jsonl`。
   - `build_skill_pool()` 读取 `data/toolbench_g3/skills.jsonl`。
   - `source_inventory` 对规范化 ToolBench-G3 的 available 判定已收紧为三件套必须齐全：
     - `trajectories.jsonl`
     - `retrieval.jsonl`
     - `skills.jsonl`
   - 只存在 `ToolBench` 代码仓库或 `data_example` 不会被标为 available。

### ToolBench-G3 当前数据缺口

官方 README 指向完整数据：

```text
Google Drive: https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL
Tsinghua Cloud: https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/
```

完整数据结构应包含：

```text
ToolBench/data/
  instruction/G3_query.json
  answer/G3_answer/*.json
  toolenv/
  retrieval/
  test_instruction/
  test_query_ids/
  retrieval_test_query_ids/
```

当前本地只有 `ToolBench/data_example`，可用于 smoke，但不能作为训练源。

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py \
  tests/test_toolbench_g3_import.py \
  tests/test_toolret_training_import.py -q
# 12 passed
```

```bash
python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  scripts/import_toolbench_g3.py \
  scripts/import_toolret_training.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_import_toolbench_g3.sh \
  scripts/sbatch/run_import_toolret_training.sh \
  scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh
# exit 0
```

```bash
git diff --check -- \
  scripts/build_clstr_unified_pretrain.py \
  scripts/import_toolbench_g3.py \
  scripts/import_toolret_training.py \
  scripts/sbatch/run_import_toolbench_g3.sh \
  scripts/sbatch/run_import_toolret_training.sh \
  tests/test_unified_pretrain_v2.py \
  tests/test_toolbench_g3_import.py \
  tests/test_toolret_training_import.py \
  description2.md
# no output
```

### 当前未完成项

完整 v4 目标仍未完成。下一步应继续 Phase B/C：

1. 获取完整 ToolBench `data.zip` 并解压到 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data`。
2. 运行：

```bash
SOURCE_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data \
  sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_import_toolbench_g3.sh
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh
```

3. 获取真实 ToolRet-Training-20w 本地 JSONL 后运行 ToolRet import，再重建 unified v2。
4. 两个目标源都落地后，推进 partial-dedup vocab v2 与 Stage 1/2 训练。

## 2026-05-26 v4 Phase B/C partial-dedup skill vocab checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 D11-v2 / D15-v2。当前已把
`scripts/build_clstr_unified_pretrain.py` 的 skill vocab 从纯
`identity_skill_id_v2` 推进到保守的规则版 partial dedup，并把训练流里的
skill 引用重写到 canonical id。该步骤不依赖外部数据下载，可以先改善
Stage 1/2 的 skill embedding 输入质量。

### 已完成的主要改动

1. `skill_pool.jsonl` 现在使用 `partial_rule_name_param_v2`
   - 规则 key：规范化后的 skill/tool name + 参数签名。
   - canonical id 优先级：`toolbench-g3/` > `traject/` > `toolret/` > 其它来源。
   - 每个 canonical skill 保留：
     - `alias_skill_ids`
     - `alternate_descriptions`
     - `source_files`
     - `provenance.partial_dedup`
   - 同名但参数签名不同的工具不会合并，避免过度去重破坏 benchmark 可信度。

2. 新增 `skill_aliases.jsonl`
   - 记录每个 raw skill id 到 canonical skill id 的映射。
   - `manifest.files.skill_aliases` 已加入文件清单。

3. unified stream 会同步重写到 canonical id
   - `trajectories.jsonl` 的 `skill_id` / `next_skill_id` 会重写。
   - `retrieval.jsonl` 的 `positive_skill_id` / `negative_skill_ids` 会重写。
   - `manifest.skill_pool.stream_remap` 记录重写行数和变更引用数。

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_partial_dedup_remaps_duplicate_tool_skills \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_partial_dedup_keeps_same_name_with_different_params_separate -q
# 2 passed
```

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py \
  tests/test_toolbench_g3_import.py \
  tests/test_toolret_training_import.py -q
# 16 passed
```

```bash
python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  scripts/import_toolbench_g3.py \
  scripts/import_toolret_training.py \
  scripts/download_toolret_training_parquet.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_import_toolbench_g3.sh \
  scripts/sbatch/run_import_toolret_training.sh \
  scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh
# exit 0
```

### 当前未完成项

完整 v4 目标仍未完成：

1. 需要重新运行 `scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh`，让真实
   `data/clstr_unified_pretrain_v2` 产物切换到 partial-dedup 版本。
2. ToolRet-Training-20w 仍需在登录节点下载 parquet shard，再通过 sbatch import。
3. ToolBench-G3 仍需要完整官方 `ToolBench/data`，当前本地只有 repo /
   `data_example`。
4. 后续还需要把 Stage 1/2 训练脚本切到 unified v2 + canonical skill vocab。

## 2026-05-26 v4 Phase B/C ToolRet landing + Stage 1 entrypoint checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Phase B/C 与 Stage 1。当前已经把
ToolRet-Training-20w 的真实 parquet shard 下载到本地，并补齐 unified v2
retrieval warmup 的训练入口。该阶段仍然不是完整训练结果，只是让 Stage 1
可以直接读取 unified v2 的 `retrieval.jsonl` + canonical `skill_pool.jsonl`。

### 已完成的主要改动

1. ToolRet-Training-20w parquet 已下载到本地
   - 目录：
     `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_training/ToolRet-Training-20w`
   - 文件：
     `train-00000-of-00006.parquet` 到 `train-00005-of-00006.parquet`
   - 总大小约 878MB。
   - 下载使用 `https://hf-mirror.com`，符合“计算节点不能联网，登录节点先下载”的约束。

2. 修复 `scripts/import_toolret_training.py` 的真实数据编码问题
   - 真实 parquet 中存在 lone Unicode surrogate，之前会在写
     `retrieval.jsonl` / `skills.jsonl` 时触发：
     `UnicodeEncodeError: surrogates not allowed`。
   - 现在写出前递归清洗 JSON payload，将非法 surrogate 替换为 `?`。
   - 该修复避免 ToolRet import 在大约 20w rows 后留下半成品。

3. 新增 unified v2 Stage 1 retrieval warmup 入口
   - `clstr/retrieval_warmup.py` 新增 `data_format="unified_v2"`。
   - `scripts/run_skillret_retrieval_warmup.py` 新增 `--data_format {skillret,unified_v2}`。
   - 新增 sbatch：
     `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh`
   - 默认数据：
     `DATA_ROOT=data/clstr_unified_pretrain_v2`
   - 默认模型：
     `BASE_MODEL_NAME=models/Qwen3-8B`
   - 默认输出：
     `outputs/clstr_unified_stage1_retrieval_warmup`
   - 该入口复用现有 CLSTR retrieval warmup 训练循环，不改变旧 SKILLRET-only 默认行为。

4. 新增 unified v2 Stage 2 full-base 训练入口
   - 新增 sbatch：
     `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
   - 默认 trajectory 数据：
     `TRAIN_PATH=data/clstr_unified_pretrain_v2/trajectories.jsonl`
   - 默认 skill pool：
     `SKILLS_PATH=data/clstr_unified_pretrain_v2/skill_pool.jsonl`
   - 默认模型：
     `QWEN_MODEL_PATH=models/Qwen3-8B`
   - 该入口显式传入 `--local_files_only`，避免计算节点联网。
   - 该入口只是训练包装，未在当前 checkpoint 启动完整 Stage 2 训练；应等
     ToolRet import + unified v2 重建完成后再提交。

5. 真实 unified v2 已完成一次 partial-dedup 重建
   - sbatch job: `77122`
   - `data/clstr_unified_pretrain_v2/manifest.json` 当前显示：
     - `status: ok`
     - `skill_pool.dedup_method: partial_rule_name_param_v2`
     - `skill_pool.raw_skill_count: 11164`
     - `skill_pool.canonical_skill_count: 10158`
     - `skill_pool.merged_group_count: 576`
     - `skill_pool.changed_alias_count: 1006`
     - `leakage_audit.status: ok`
   - 由于这次重建发生在 ToolRet import 完成前，manifest 中
     `toolret_training` 仍然 missing；ToolRet import 完成后需要再次重建。

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolret_training_import.py::test_import_toolret_training_sanitizes_lone_unicode_surrogates \
  tests/test_toolret_training_import.py -q
# 6 passed
```

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_uses_unified_v2_data_and_local_cache \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool -q
# 3 passed
```

```bash
python -m py_compile \
  scripts/import_toolret_training.py \
  scripts/download_toolret_training_parquet.py \
  clstr/retrieval_warmup.py \
  scripts/run_skillret_retrieval_warmup.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_import_toolret_training.sh \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# exit 0
```

### 当前未完成项

完整 v4 目标仍未完成：

1. ToolRet import 重试作业 `77127` 已提交，需要等它完成后确认
   `data/toolret_training/manifest.json`、`retrieval.jsonl`、`skills.jsonl`。
2. ToolRet import 完成后，需要再次提交
   `scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh`，让 unified v2 manifest
   中 `toolret_training` 从 missing 变为 available。
3. ToolBench-G3 仍需要完整官方 `ToolBench/data`，当前本地只有 repo /
   `data_example`。
4. Stage 1 入口已具备，但尚未提交完整训练作业；应在 unified v2 包含 ToolRet
   后再启动。

## 2026-05-26 v4 Phase B/C ToolRet included in unified v2

ToolRet import 修复后已完整跑通，并已重新构建 unified v2。

### 真实数据结果

ToolRet import 重试作业：

```text
sbatch job: 77127
status: ok
seen_rows: 208826
retrieval_pairs: 208826
skills: 28466
skipped_rows: 0
source_path: /data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_training/ToolRet-Training-20w
```

unified v2 重建作业：

```text
sbatch job: 77134
status: ok
retrieval_stream.total_pairs: 336016
retrieval_stream.counts:
  skillret_train_pair: 127190
  toolret_training_pair: 208826
skill_pool.dedup_method: partial_rule_name_param_v2
skill_pool.raw_skill_count: 37239
skill_pool.canonical_skill_count: 36183
skill_pool.merged_group_count: 621
skill_pool.changed_alias_count: 1056
source_inventory.available_source_count: 5
source_inventory.missing_required_target_sources:
  - toolbench_g3
leakage_audit.status: ok
```

对应文件行数：

```text
data/clstr_unified_pretrain_v2/retrieval.jsonl: 336016
data/clstr_unified_pretrain_v2/skill_pool.jsonl: 36183
data/clstr_unified_pretrain_v2/skill_aliases.jsonl: 37239
```

### 当前未完成项

完整 v4 目标仍未完成：

1. TRAJECT-Bench 已进入 trajectory stream，但还未进入 retrieval stream；下一步应补
   TRAJECT step-level retrieval pairs。
2. ToolBench-G3 仍需要完整官方 `ToolBench/data`。
3. Stage 1/2 unified v2 入口已具备，但尚未提交完整训练作业；建议等 TRAJECT
   retrieval stream 补齐后再启动 Stage 1。

## 2026-05-26 v4 Phase B/C TRAJECT retrieval stream checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Phase B/C。此前 TRAJECT-Bench
只进入了 trajectory stream，Stage 1 retrieval warmup 仍只含 SKILLRET /
ToolRet。当前已补齐 TRAJECT step-level retrieval pairs，让 TRAJECT sequential /
parallel 的状态到工具选择监督也进入 unified v2 的 retrieval stream。

### 已完成的主要改动

1. `scripts/build_clstr_unified_pretrain.py` 新增 TRAJECT retrieval 构建
   - 复用 TRAJECT public_data 的 step 序列。
   - 每个 step 生成一个 retrieval pair：
     - `query_text`: 当前 step 的 state text，包含 goal、trajectory_type、domain、
       task_name、task_description、previous_tools。
     - `positive_skill_id`: 当前 step 使用的 tool。
     - `negative_skill_ids`: 同一条 trajectory 里其它 step 的 tool，按出现顺序去重。
   - `manifest.retrieval_stream.counts.traject_bench_pair` 记录 TRAJECT retrieval
     pair 数。
   - `manifest.retrieval_stream.traject_bench` 记录 available、total_pairs、query_count。

2. 设计理由
   - 这比只用原始 task query 更贴近 CLSTR 的论文主张：在多步状态下选择下一个
     skill。
   - negatives 只来自同一条 trajectory，避免跨任务随机构造过多弱负样本。
   - 该监督仍是 benchmark-agnostic 的 stepwise routing，不写入 AppWorld 特化逻辑。

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py \
  tests/test_toolbench_g3_import.py \
  tests/test_toolret_training_import.py \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_uses_unified_v2_data_and_local_cache \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool -q
# 20 passed
```

```bash
python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  scripts/import_toolbench_g3.py \
  scripts/import_toolret_training.py \
  scripts/download_toolret_training_parquet.py \
  clstr/retrieval_warmup.py \
  scripts/run_skillret_retrieval_warmup.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh \
  scripts/sbatch/run_import_toolbench_g3.sh \
  scripts/sbatch/run_import_toolret_training.sh \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# exit 0
```

### 当前未完成项

完整 v4 目标仍未完成：

1. ToolBench-G3 仍需要完整官方 `ToolBench/data`。
2. 等 Stage 1 retrieval warmup 完成后，再进入 Stage 2 full-base 训练。

## 2026-05-26 v4 Stage 1 readiness checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Stage 1。当前 unified v2 已经包含
SKILLRET + TRAJECT-Bench + ToolRet 三个 retrieval 来源，且 Stage 1 训练入口已修正为
真正的 L_retr warmup。

### 真实 unified v2 manifest

最新重建作业：

```text
sbatch job: 77135
status: ok
retrieval_stream.total_pairs: 374110
retrieval_stream.counts:
  skillret_train_pair: 127190
  traject_bench_pair: 38094
  toolret_training_pair: 208826
retrieval_stream.traject_bench.available: true
retrieval_stream.traject_bench.total_pairs: 38094
retrieval_stream.traject_bench.query_count: 5870
trajectory_stream.total_rows: 273269
skill_pool.dedup_method: partial_rule_name_param_v2
skill_pool.raw_skill_count: 37239
skill_pool.canonical_skill_count: 36183
skill_pool.merged_group_count: 621
skill_pool.changed_alias_count: 1056
source_inventory.missing_required_target_sources:
  - toolbench_g3
leakage_audit.status: ok
```

对应文件行数：

```text
data/clstr_unified_pretrain_v2/retrieval.jsonl: 374110
data/clstr_unified_pretrain_v2/trajectories.jsonl: 273269
data/clstr_unified_pretrain_v2/skill_pool.jsonl: 36183
data/clstr_unified_pretrain_v2/skill_aliases.jsonl: 37239
```

### Stage 1 入口修正

1. `clstr/retrieval_warmup.py`
   - `data_format="unified_v2"` 读取
     `data/clstr_unified_pretrain_v2/retrieval.jsonl` 和 canonical
     `skill_pool.jsonl`。
   - 训练目标记录为 `L_retr_full_pool_cross_entropy`。
   - 优化器现在包含：
     - `encoder.proj`
     - `skill_table.W`
     - `skill_table.E`
     - `skill_table.logit_scale_retr`
     - `skill_table.skill_bias_retr`
   - 这与 v4 计划的 Stage 1 “trainable: W, logit_scale_retr,
     skill_bias_retr, E” 对齐。

2. `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh`
   - 默认 `DATA_ROOT=data/clstr_unified_pretrain_v2`。
   - 默认 `BASE_MODEL_NAME=models/Qwen3-8B`。
   - 默认 `USE_CROSS_ENCODER=0`，提交时会传 `--disable_cross_encoder`。
   - 这样 Stage 1 不会为了 reranker 额外加载一份 Qwen3-8B，避免 GPU 显存和时间浪费。

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_uses_unified_v2_data_and_local_cache -q
# 2 passed
```

```bash
python -m py_compile clstr/retrieval_warmup.py scripts/run_skillret_retrieval_warmup.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/retrieval_warmup.py \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  tests/test_skillret.py \
  tests/test_sbatch_scripts.py \
  scripts/build_clstr_unified_pretrain.py \
  tests/test_unified_pretrain_v2.py \
  description2.md
# no output
```

### 当前未完成项

完整 v4 目标仍未完成：

1. 提交并等待 Stage 1 retrieval warmup 完整训练。
2. ToolBench-G3 仍需要完整官方 `ToolBench/data`。
3. Stage 1 完成后，提交 Stage 2 full-base 训练。

### Stage 1 作业提交记录

已提交 unified v2 Stage 1 retrieval warmup：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh
# Submitted batch job 77136
```

默认配置：

```text
DATA_ROOT=data/clstr_unified_pretrain_v2
DATA_FORMAT=unified_v2
BASE_MODEL_NAME=models/Qwen3-8B
OUTPUT_DIR=outputs/clstr_unified_stage1_retrieval_warmup
MAX_STEPS=5000
BATCH_SIZE=16
MODEL_DIM=1024
TOP_K=50
USE_CROSS_ENCODER=0
```

注意：该作业仍在运行/等待结果时，不应提交 Stage 2；Stage 2 需要以 Stage 1
checkpoint 作为更合理的 routing warmup 初始化。

## 2026-05-26 v4 Stage 1/2 continuity fix checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Stage 1 → Stage 2 训练连续性。前一次
Stage 1 作业 `77136` / `77137` 均快速失败，日志报
`json.decoder.JSONDecodeError: Unterminated string`，位置在读取
`data/clstr_unified_pretrain_v2/skill_pool.jsonl`。后续用 Python 对当前
`skill_pool.jsonl`、`retrieval.jsonl`、`trajectories.jsonl` 全量解析均通过，
说明当前文件已经稳定，失败更可能是作业在 unified v2 重建写入完成前读到了半写入文件，
或日志来自同一旧失败状态。

### 本次修复

1. Stage 1 retrieval warmup 输入与 checkpoint 修复
   - `clstr/retrieval_warmup.py` 的 JSONL 读取改为流式逐行解析；
     JSON 损坏时会报告具体文件名和行号，例如 `retrieval.jsonl line 2`，
     避免深层 `JSONDecodeError` 无法定位。
   - Stage 1 warmup checkpoint 不再保存冻结的 Qwen backbone 和
     `skill_table.encoder_fn.*` 引用，只保存 Stage 2 需要的 routing/proj 权重。
   - checkpoint 报告新增：
     - `checkpoint_excludes_frozen_backbone: true`
     - `checkpoint_state_key_count`
     - `excluded_state_key_prefixes`

2. Stage 2 从 Stage 1 初始化
   - `train_clstr_full_base_with_model(...)` 新增 `routing_checkpoint_path`。
   - 加载点放在设备迁移和 deferred skill table rebuild 之后，避免 warmup 的
     `skill_table.E/W/scale/bias` 被后续 skill table rebuild 覆盖。
   - 只加载兼容的 routing foundation keys：
     - `encoder.proj.*`
     - `skill_table.W.*`
     - `skill_table.E`
     - `skill_table.logit_scale_retr`
     - `skill_table.skill_bias_retr`
     - `skill_table.logit_scale_belief`
     - `skill_table.skill_bias_belief`
   - 明确跳过 backbone 和随机初始化 head，例如 `encoder.backbone.*`、
     `skill_head.*`，避免 Stage 1 retrieval warmup 污染 Stage 2 的 ACT/head
     训练目标。
   - 报告和 checkpoint 中新增 `routing_checkpoint` 审计字段，记录：
     - `loaded`
     - `path`
     - `stage`
     - `loaded_keys`
     - `skipped_keys`
     - `shape_mismatched`
     - `partial_load_mode`

3. Stage 2 CLI / sbatch 接入
   - `scripts/run_clstr_qwen3_full_base_train.py` 新增
     `--routing_checkpoint_path`。
   - `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh` 默认使用：

```text
ROUTING_CHECKPOINT_PATH=outputs/clstr_unified_stage1_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt
```

   - 这意味着 Stage 1 未完成时，Stage 2 会明确阻塞，而不是静默用随机 routing
     foundation 进入 full-base 训练。
   - `clstr/qwen_full_base_train.py` 会在加载 Qwen3-8B 前检查
     `routing_checkpoint_path` 是否存在；缺失时直接写 blocker report，避免先占用
     GPU/加载 8B 模型后才失败。
   - `train_clstr_full_base_with_model(...)` 会先读取 Stage 1 checkpoint 元信息；
     如果其中包含形状兼容的 `skill_table.E`，则跳过 deferred `rebuild_skill_table()`，
     直接加载 Stage 1 的 routing foundation。这样 Stage 2 不会再用 Qwen3-8B 重新编码
     3.6 万个 skill 后又被 checkpoint 覆盖。
   - 新增 Stage 2 preflight：
     - `clstr/stage_preflight.py`
     - `scripts/run_clstr_stage2_preflight.py`
   - `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh` 在加载 Qwen3-8B 前先检查
     Stage 1 checkpoint 是否存在、`skill_table.E` 是否为 rank-2、skill 数是否与
     `skill_pool.jsonl` 对齐、embedding dim 是否等于 `MODEL_DIM`，并写出
     `outputs/clstr_unified_stage2_full_base/stage2_preflight.json`。
   - Stage 2 full-base 新增 `embedding_cache_mode` / `embedding_cache_max_rows`：
     - `auto`: 默认策略，小数据仍做 embedding cache；Qwen3-8B 外部编码器 + 大数据
       超过阈值时跳过全量预编码。
     - `always`: 强制训练前预编码。
     - `never`: 强制不预编码。
   - `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh` 默认：

```text
EMBEDDING_CACHE_MODE=auto
EMBEDDING_CACHE_MAX_ROWS=20000
```

   - 这避免 Stage 2 在 27 万轨迹行上先用 Qwen3-8B 预编码所有
     state/action/next/candidate embedding 后才开始训练；大数据正式训练会改为按 batch
     编码，报告中记录 `embedding_cache_policy`。

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  tests/test_skillret.py::test_unified_v2_retrieval_loader_reports_corrupt_jsonl_line \
  tests/test_full_base_train.py::test_full_base_training_loads_stage1_routing_checkpoint_after_skill_table_rebuild \
  tests/test_full_base_train.py::test_full_base_training_skips_deferred_skill_table_rebuild_when_stage1_checkpoint_has_embeddings \
  tests/test_qwen_external_encoder.py::test_qwen_full_base_train_forwards_stage1_routing_checkpoint \
  tests/test_qwen_external_encoder.py::test_qwen_full_base_train_blocks_before_loading_qwen_when_stage1_checkpoint_missing \
  tests/test_stage2_preflight.py \
  tests/test_full_base_train.py::test_qwen_external_training_uses_small_embedding_cache_batches \
  tests/test_full_base_train.py::test_qwen_external_training_skips_full_dataset_embedding_cache_for_large_stage2_data \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_uses_unified_v2_data_and_local_cache \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool -q
# 12 passed
```

```bash
python -m py_compile \
  clstr/retrieval_warmup.py \
  clstr/full_base_train.py \
  clstr/qwen_full_base_train.py \
  scripts/run_clstr_qwen3_full_base_train.py \
  scripts/run_skillret_retrieval_warmup.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/retrieval_warmup.py \
  clstr/full_base_train.py \
  clstr/qwen_full_base_train.py \
  scripts/run_clstr_qwen3_full_base_train.py \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  tests/test_skillret.py \
  tests/test_full_base_train.py \
  tests/test_qwen_external_encoder.py \
  tests/test_sbatch_scripts.py
# no output
```

### 当前未完成项

1. 重新提交 Stage 1 retrieval warmup，等待 `clstr_unified_retrieval_v2-step5000.pt`。
2. Stage 1 checkpoint 生成并验证后，再提交 Stage 2 full-base 训练。
3. ToolBench-G3 完整官方数据仍未落地，主表 #2 仍是数据 blocker。

## 2026-05-26 v4 ToolBench-G3 official data download/verify entrypoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Phase B 数据准备。由于
ToolBench-G3 是主表 #2 的必要训练/评测源，本次补齐官方 `data.zip` 的登录节点下载、
解压和完整性校验入口；当前仍不把 `ToolBench/data_example` 当作可训练数据。

### 本次新增

1. 新增 `scripts/download_toolbench_data.py`
   - 默认写入范围限制在：

```text
/data/home/scyb713/run/xzf/AAAI/autodl-tmp
```

   - 默认目标：

```text
zip_path=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data.zip
extract_root=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench
data_root=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data
```

   - 支持：
     - `--dry_run`：只写 manifest，不下载/解压；
     - `--download --source google`：尝试 Google Drive 官方 file id；
     - `--download --source tsinghua`：尝试清华云盘下载 URL；
     - `--verify_only`：只校验 `ToolBench/data` 是否包含完整 G3 资产；
     - 安全 zip 解压：拒绝 `../` 路径穿越；
     - 下载时先写 `data.zip.part`，完成后再替换为 `data.zip`；
     - 若下载得到 HTML/坏 zip，则标记 `invalid_zip` 并把本次下载产物隔离到
       `data.zip.invalid`，避免后续误解压。

2. 校验策略
   - `verify_toolbench_data(...)` 至少要求：

```text
instruction/G3_query.json
answer/G3_answer/*.json
```

   - `data_example/instruction/G3_query.json` 只能触发 `missing_required_files`，
     不会被视为完整 ToolBench-G3。

3. 当前官方源探测结果
   - Google Drive：

```text
python scripts/download_toolbench_data.py --download \
  --manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/download_manifest.json
# status: download_failed
# error: Network is unreachable to drive.google.com
```

   - Tsinghua Cloud：

```text
python scripts/download_toolbench_data.py --download --source tsinghua \
  --manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/download_tsinghua_manifest.json
# status: invalid_zip
# content_type: text/html; charset=utf-8
# downloaded_bytes: 7221
# invalid_zip_path: /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data.zip.invalid
```

   - 结论：当前登录节点无法自动拿到官方 ToolBench `data.zip`。下一步需要手动从
     Google Drive / 清华云盘下载官方 `data.zip` 到上述 `zip_path`，然后运行：

```bash
python scripts/download_toolbench_data.py

SOURCE_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data \
  sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_import_toolbench_g3.sh
```

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py -q
# 9 passed
```

```bash
python -m py_compile scripts/download_toolbench_data.py scripts/import_toolbench_g3.py
# exit 0
```

```bash
python scripts/download_toolbench_data.py --dry_run --source tsinghua \
  --manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/download_tsinghua_dry_run_manifest.json
# status: dry_run
```

```bash
python scripts/download_toolbench_data.py --verify_only \
  --manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/verify_manifest.json
# status: missing_required_files
# missing: instruction/G3_query.json, answer/G3_answer/*.json
```

```bash
git diff --check -- scripts/download_toolbench_data.py tests/test_toolbench_g3_import.py
# no output
```

### 当前未完成项

1. ToolBench-G3 官方 `data.zip` 仍未落地；主表 #2 仍是数据 blocker。
2. 拿到 `ToolBench/data` 后，需要提交 `run_import_toolbench_g3.sh`，然后重建
   `data/clstr_unified_pretrain_v2`。
3. 如果 ToolBench-G3 在短期内无法取得，v4 仍可先继续 Stage 1/2 on
   SKILLRET + TRAJECT + ToolRet，但最终主表 #2 不能省略。

## 2026-05-26 v4 Stage 4 generic L_act pipeline checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Stage 4。之前 `L_act` 主要绑定
AppWorld `VerifiedPair`，不利于 v4 中“Stage 4 不局限 AppWorld”的设定。本次新增
generic transition-conditioned next-skill 训练入口，使 Stage 4 可以直接消费 unified
`trajectories.jsonl` 中的 train split 轨迹行。

### 本次修复

1. 新增 `clstr/stage4_act_train.py`
   - 从 unified trajectory rows 构造 Stage 4 样本：

```text
state_text/history at t
action_text / skill_id at t
next_observation_text
candidate_next_skill_ids / candidate_next_skill_indices
positive_next_skill_id / positive_next_skill_idx
```

   - 只接收 train split：
     - `train`
     - `training`
     - `train_or_released_g3`
     - `released_train`
   - dev/test 或无 split 行会计入 `skipped_reasons.split_not_train`，不会进入训练。
   - 若候选集合不包含正例 `next_skill_id`，会 deterministic inject，并记录
     `positive_injected=True` / `positive_injected_rows`。
   - 负例补齐使用 stable random index sampling，不会对 3.6 万到 15 万 skill pool
     每行全量 shuffle。
   - L_act 形式：

```text
L_act = CE(TransHead(m_hat_{t+1}, action_proj(E)[C_{t+1}]), positive_next_skill_position)
```

   - 默认冻结 routing foundation 和 BeliefGate，只训练：
     - `trans_head`
     - `action_proj`
   - 可通过 `train_transition=True` 同时训练 `TransitionPredictor`。

2. 新增 `clstr/qwen_stage4_act_train.py`
   - Qwen3-8B 仍作为 frozen external encoder。
   - 在加载 Qwen3-8B 之前先检查两个必要 checkpoint：

```text
routing_checkpoint_path: Stage 1 retrieval warmup checkpoint
head_checkpoint_path: Stage 2 或 Stage 3 head checkpoint
```

   - 缺任一 checkpoint 时直接写 blocker report，不占用 GPU 加载 8B 模型。
   - 初始化策略：

```text
merged_state = Stage1 routing state + Stage2/Stage3 head state
model.load_state_dict(merged_state, strict=False)
```

   - 写出 `stage4_checkpoint_init.json`，记录两个 checkpoint 的 stage/key count、
     missing/unexpected keys 和合并方式。

3. 新增运行入口
   - CLI:

```text
scripts/run_clstr_qwen3_stage4_act_train.py
```

   - sbatch:

```text
scripts/sbatch/run_clstr_unified_stage4_act_train.sh
```

   - 默认配置：

```text
TRAJECTORIES_PATH=data/clstr_unified_pretrain_v2/trajectories.jsonl
SKILLS_PATH=data/clstr_unified_pretrain_v2/skill_pool.jsonl
ROUTING_CHECKPOINT_PATH=outputs/clstr_unified_stage1_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt
HEAD_CHECKPOINT_PATH=outputs/clstr_unified_stage2_full_base/checkpoints/clstr_full_base-step10000.pt
OUTPUT_DIR=outputs/clstr_unified_stage4_act
CANDIDATE_COUNT=64
MAX_STEPS=2000
BATCH_SIZE=8
```

   - Stage 3 完成后，可用 `HEAD_CHECKPOINT_PATH` 覆盖为 Stage 3 HRPO checkpoint。

4. Qwen external CLSTR config 修复
   - `clstr/qwen_external_encoder.py` 中 `d_a` 从 `max(64, d//4)` 改为 `d`。
   - 这是对齐 v4 决策 D6：`action_proj(E)` 与 state/belief 同维，避免
     TransitionPredictor/TransHead 仍停留在旧的窄动作 embedding 设定。

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stage4_act_train.py \
  tests/test_qwen_external_encoder.py::test_build_qwen_external_clstr_config_uses_identity_projection_when_dim_matches \
  tests/test_qwen_external_encoder.py::test_qwen_stage4_act_train_blocks_before_loading_qwen_when_checkpoint_missing \
  tests/test_qwen_external_encoder.py::test_qwen_stage4_act_train_loads_stage1_and_head_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints -q
# 8 passed
```

```bash
python -m py_compile \
  clstr/stage4_act_train.py \
  clstr/qwen_stage4_act_train.py \
  clstr/qwen_external_encoder.py \
  scripts/run_clstr_qwen3_stage4_act_train.py
# exit 0
```

```bash
bash -n scripts/sbatch/run_clstr_unified_stage4_act_train.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/stage4_act_train.py \
  clstr/qwen_stage4_act_train.py \
  clstr/qwen_external_encoder.py \
  scripts/run_clstr_qwen3_stage4_act_train.py \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  tests/test_stage4_act_train.py \
  tests/test_qwen_external_encoder.py \
  tests/test_sbatch_scripts.py
# no output
```

### 当前未完成项

1. Stage 4 不能在 Stage 1/2/3 完成前正式提交；当前只是补齐 pipeline。
2. Stage 3 HRPO 训练入口仍需后续接到 unified TRAJECT/ToolBench 数据上。
3. ToolBench-G3 官方数据仍未落地，Stage 4 当前只能先使用 unified v2 已有 train
   trajectories；ToolBench-G3 轨迹需等数据下载/导入完成后重建 unified v2。

## 2026-05-26 v4 Stage 3 unified HRPO-style pipeline checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Stage 3。现有 `online_hrpo.py` 主要绑定
ALFWorld 环境 rollout，而 v4 的主线需要 TRAJECT / ToolBench / ToolRet 方向的 unified
训练入口。由于 TRAJECT / ToolBench simulator 仍未完全接入，本次先补齐一个明确标注为
**offline train-split HRPO-style action policy update** 的 Stage 3 pipeline，保证后续
Stage 2 checkpoint 产出后有可提交、可审计的统一入口；它不伪装成最终 simulator
rollout HRPO。

### 本次新增

1. 新增 `clstr/stage3_unified_hrpo.py`
   - 从 unified `trajectories.jsonl` 构造 Stage 3 action-level 样本：

```text
state_text
candidate_actions / admissible_actions
expert_action / selected_index
reward
group_id = trajectory_id or task_id
```

   - 只接收 train split：
     - `train`
     - `training`
     - `train_or_released_g3`
     - `released_train`
   - dev/test 或无 split 行会计入 `skipped_reasons.split_not_train`。
   - 若 expert action 不在候选动作集合里，会 deterministic inject，并记录
     `expert_injected=True` / `expert_injected_rows`。
   - 使用 grouped reward advantages + KL-to-reference 的 HRPO-style loss：

```text
loss = -log pi(expert_action | state, candidates) * grouped_advantage
       + beta_kl * KL(pi || pi_ref)
```

   - 默认冻结 routing foundation，只训练 controller/head 相关模块：
     - `skill_head`
     - `transition`
     - `gate`
     - `stop_head`
     - `action_proj`
   - 报告中明确：

```text
training_regime=offline_train_split_grouped_preference_update
on_policy_rollout_used=False
not_full_simulator_rollout_hrpo=True
```

2. 新增 `clstr/qwen_stage3_unified_hrpo.py`
   - Qwen3-8B 仍作为 frozen external encoder。
   - 在加载 Qwen3-8B 前先检查必要 checkpoint：

```text
routing_checkpoint_path: Stage 1 retrieval warmup checkpoint
head_checkpoint_path: Stage 2 full-base checkpoint
```

   - 缺任一 checkpoint 时直接写 blocker report，不加载 8B 模型。
   - 初始化策略与 Stage 4 对齐：

```text
merged_state = Stage1 routing state + Stage2 head state
model.load_state_dict(merged_state, strict=False)
```

3. 新增公共 checkpoint 初始化工具
   - `clstr/qwen_checkpoint_init.py`
   - Stage 3 / Stage 4 共同复用：
     - `checkpoint_payload(...)`
     - `checkpoint_state(...)`
     - `merge_routing_and_head_state(...)`
     - `load_routing_and_head_checkpoints(...)`
   - 避免 Stage 3/4 各自维护一套 Stage1+head checkpoint 合并逻辑。

4. 新增运行入口
   - CLI:

```text
scripts/run_clstr_qwen3_unified_stage3_hrpo.py
```

   - sbatch:

```text
scripts/sbatch/run_clstr_unified_stage3_hrpo.sh
```

   - 默认配置：

```text
TRAJECTORIES_PATH=data/clstr_unified_pretrain_v2/trajectories.jsonl
SKILLS_PATH=data/clstr_unified_pretrain_v2/skill_pool.jsonl
ROUTING_CHECKPOINT_PATH=outputs/clstr_unified_stage1_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt
HEAD_CHECKPOINT_PATH=outputs/clstr_unified_stage2_full_base/checkpoints/clstr_full_base-step10000.pt
OUTPUT_DIR=outputs/clstr_unified_stage3_hrpo
UPDATES=500
BATCH_SIZE=8
BETA_KL=0.15
```

5. Stage 4 wrapper cleanup
   - `clstr/qwen_stage4_act_train.py` 已改为复用 `clstr/qwen_checkpoint_init.py`，
     不再保留重复的 checkpoint 合并实现。

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stage3_unified_hrpo.py \
  tests/test_stage4_act_train.py \
  tests/test_qwen_external_encoder.py::test_qwen_unified_stage3_hrpo_blocks_before_loading_qwen_when_checkpoint_missing \
  tests/test_qwen_external_encoder.py::test_qwen_unified_stage3_hrpo_loads_stage1_and_stage2_checkpoints \
  tests/test_qwen_external_encoder.py::test_qwen_stage4_act_train_blocks_before_loading_qwen_when_checkpoint_missing \
  tests/test_qwen_external_encoder.py::test_qwen_stage4_act_train_loads_stage1_and_head_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage3_hrpo_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints -q
# 13 passed
```

```bash
python -m py_compile \
  clstr/stage3_unified_hrpo.py \
  clstr/qwen_stage3_unified_hrpo.py \
  clstr/qwen_checkpoint_init.py \
  clstr/stage4_act_train.py \
  clstr/qwen_stage4_act_train.py \
  scripts/run_clstr_qwen3_unified_stage3_hrpo.py \
  scripts/run_clstr_qwen3_stage4_act_train.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/stage3_unified_hrpo.py \
  clstr/qwen_stage3_unified_hrpo.py \
  clstr/qwen_checkpoint_init.py \
  clstr/stage4_act_train.py \
  clstr/qwen_stage4_act_train.py \
  scripts/run_clstr_qwen3_unified_stage3_hrpo.py \
  scripts/run_clstr_qwen3_stage4_act_train.py \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  tests/test_stage3_unified_hrpo.py \
  tests/test_stage4_act_train.py \
  tests/test_qwen_external_encoder.py \
  tests/test_sbatch_scripts.py
# no output
```

### 当前未完成项

1. 当前 Stage 3 是 offline HRPO-style 入口，不是最终 TRAJECT/ToolBench simulator
   rollout HRPO；论文中不能把它当作 on-policy simulator result。
2. Stage 3 正式提交仍需等待 Stage 1/2 checkpoint 完成。
3. 真正 v4 Stage 3 还应继续接入 TRAJECT / ToolBench simulator reward 或稳定 judge
   reward，当前入口先保证 unified train split policy update 可运行、可审计。

## 2026-05-26 v4 unified training benchmark-filter checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的主战场切换。当前 unified v2 数据中
ALFWorld / ScienceWorld / AgentGym 行数远大于 TRAJECT-Bench，若 Stage 2/3/4
默认直接消费全量 `trajectories.jsonl`，训练会被辅助环境数据主导，和 v4 的
TRAJECT / ToolBench 主线不一致。因此本次补齐了可审计的 benchmark 过滤链路。

### 本次改动

1. Stage 2 full-base 训练新增 `allowed_benchmarks`
   - `clstr/full_base_train.py` 新增 benchmark 过滤工具。
   - `train_clstr_full_base_with_model(...)` / `run_clstr_full_base_train(...)`
     支持 `allowed_benchmarks`。
   - 训练报告和 checkpoint payload 写入：

```text
benchmark_filter.enabled
benchmark_filter.allowed_benchmarks
benchmark_filter.source_rows
benchmark_filter.retained_rows
benchmark_filter.skipped_rows
benchmark_filter.skipped_by_benchmark
```

   - 若过滤后没有训练行，会显式报错，不会静默训练空数据。

2. Qwen Stage 2 CLI / wrapper / sbatch 接入过滤
   - `clstr/qwen_full_base_train.py`
   - `scripts/run_clstr_qwen3_full_base_train.py`
   - `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
   - sbatch 默认：

```text
ALLOWED_BENCHMARKS=traject_bench,toolbench_g3
```

   可通过提交前设置环境变量覆盖；设为空字符串可回到全量 unified train。

3. Stage 3 / Stage 4 数据报告补齐过滤审计
   - `clstr/stage3_unified_hrpo.py`
   - `clstr/stage4_act_train.py`
   - builder 原本已支持 `allowed_benchmarks`，本次补充报告字段：

```text
benchmark_filter_enabled
allowed_benchmarks
skipped_benchmarks
skipped_reasons.benchmark_not_allowed
```

4. Qwen Stage 3 / Stage 4 CLI / wrapper / sbatch 接入过滤
   - `clstr/qwen_stage3_unified_hrpo.py`
   - `clstr/qwen_stage4_act_train.py`
   - `scripts/run_clstr_qwen3_unified_stage3_hrpo.py`
   - `scripts/run_clstr_qwen3_stage4_act_train.py`
   - `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`
   - `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
   - Stage 3/4 sbatch 默认同样为：

```text
ALLOWED_BENCHMARKS=traject_bench,toolbench_g3
```

### 设计含义

这不是把 CLSTR 做成 AppWorld 特化方法，而是相反：默认训练入口转向 v4 主战场的
trajectory/tool-calling 数据。AppWorld 后续只作为 secondary transfer /
combination plot。当前 ToolBench-G3 数据仍未落地，因此默认过滤会先保留
TRAJECT-Bench；ToolBench-G3 导入 unified v2 后会自动进入同一训练入口。

### 验证证据

已先写失败测试确认功能缺失，再实现并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_full_base_train.py::test_full_base_training_filters_to_allowed_benchmarks_and_records_audit \
  tests/test_stage3_unified_hrpo.py::test_unified_hrpo_builder_filters_allowed_benchmarks_and_reports_skips \
  tests/test_stage4_act_train.py::test_stage4_builder_filters_allowed_benchmarks_and_reports_skips \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  tests/test_sbatch_scripts.py::test_unified_stage3_hrpo_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints -q
# 6 passed
```

### 当前未完成项

1. Stage 1 unified retrieval warmup job `77139` 仍在运行，本次未中断。
2. Stage 2 正式训练仍需等待 Stage 1 checkpoint，并先通过 Stage 2 preflight。
3. ToolBench-G3 官方完整数据仍缺失；导入后需要重建 unified v2，让默认过滤同时
   覆盖 TRAJECT-Bench 和 ToolBench-G3。

## 2026-05-26 v4 unified readiness audit + train-safe default checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Phase B/C/D。前一版 Stage 2/3/4
sbatch 默认使用 `ALLOWED_BENCHMARKS=traject_bench,toolbench_g3`，这在逻辑上符合
主战场切换，但 readiness audit 暴露出一个关键问题：当前本地 TRAJECT-Bench
`public_data/` 是 benchmark/evaluation public data，unified v2 中的 split 被标为
`public_train_or_eval_unlabeled`，不是明确 train split。若把这批数据直接作为
Stage 2/3/4 train，会削弱 leakage audit 和论文可信度。

### 本次改动

1. 新增 unified readiness audit
   - 新增 `scripts/audit_clstr_unified_training_readiness.py`。
   - 新增 sbatch 包装：

```text
scripts/sbatch/run_clstr_unified_readiness_audit.sh
```

   - 只读审计，不训练、不提交其他作业。
   - 输出默认写入：

```text
outputs/clstr_unified_readiness_audit/readiness_report.json
```

   - 报告拆分三条数据线：
     - `retrieval_filter`: Stage 1 retrieval warmup 可用 train-safe retrieval rows；
     - `base_train_filter`: Stage 2 full-base 可用 train-safe trajectory rows；
     - `benchmark_filter`: Stage 3/4 target benchmark train rows。

2. Stage 1 retrieval warmup 默认排除 public/eval-like retrieval rows
   - `clstr/retrieval_warmup.py` 的 unified v2 loader 现在默认排除：

```text
dev, eval, evaluation, public_train_or_eval_unlabeled, test, valid, validation
```

   - 当前真实 unified v2 上，TRAJECT retrieval rows 被排除：

```text
retrieval_filter.source_retrieval_rows: 374110
retrieval_filter.retained_train_retrieval_rows: 336016
retrieval_filter.skipped_by_source.traject_bench: 38094
retrieval_filter.skipped_by_split.public_train_or_eval_unlabeled: 38094
```

   - 这意味着当前运行中的 Stage 1 job `77139` 是在本次修正前提交的；若它完成，
     其 checkpoint 需要标记为“可能包含 TRAJECT public unlabeled retrieval”并谨慎使用。
     推荐重新提交 Stage 1，以使用修正后的 train-safe retrieval loader。

3. Stage 2 full-base 默认改回 train-safe unified base
   - `clstr/full_base_train.py` 新增 train split filter。
   - Stage 2 默认接受明确 train split：

```text
train, training, train_or_released_g3, released_train
```

   - 默认排除 `public_train_or_eval_unlabeled`。
   - `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh` 默认：

```text
ALLOWED_BENCHMARKS=
```

   即 Stage 2 默认训练 train-safe unified base，不再 target-only 过滤。后续若拿到明确的
   TRAJECT/ToolBench train split，可显式设置 `ALLOWED_BENCHMARKS=traject_bench,toolbench_g3`
   做 target-only ablation。

4. Stage 3/4 保持 target benchmark train-only gate
   - Stage 3/4 sbatch 仍默认 `ALLOWED_BENCHMARKS=traject_bench,toolbench_g3`。
   - 但 readiness audit 当前显示 target train rows 为 0，因此 Stage 3/4 不能正式提交：

```text
benchmark_filter.retained_train_trajectory_rows: 0
stage_gates.stage3_offline_hrpo.blockers:
  - missing_stage1_checkpoint
  - missing_stage2_checkpoint
  - no_stage3_candidate_rows
stage_gates.stage4_act.blockers:
  - missing_stage1_checkpoint
  - missing_stage2_or_stage3_checkpoint
  - no_stage4_next_skill_rows
```

### 当前真实 readiness audit 结果

已在当前 unified v2 上生成：

```bash
python scripts/audit_clstr_unified_training_readiness.py \
  --data_root data/clstr_unified_pretrain_v2 \
  --allowed_benchmarks traject_bench,toolbench_g3 \
  --stage1_checkpoint outputs/clstr_unified_stage1_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt \
  --stage2_checkpoint outputs/clstr_unified_stage2_full_base/checkpoints/clstr_full_base-step10000.pt \
  --output_path outputs/clstr_unified_readiness_audit/readiness_report.json
```

关键结论：

```text
status: action_required
leakage_audit.status: ok
missing_required_target_sources: [toolbench_g3]
stage1_retrieval_warmup.ready_to_submit: true
stage1_retrieval_warmup.data_rows: 336016
stage2_full_base.ready_to_submit: false
stage2_full_base.blockers: [missing_stage1_checkpoint]
stage2_full_base.data_rows: 235175
stage3_offline_hrpo.ready_to_submit: false
stage3_offline_hrpo.data_rows: 0
stage4_act.ready_to_submit: false
stage4_act.data_rows: 0
paper_readiness.full_three_benchmark_data_ready: false
```

### 设计含义

这次修正避免了一个会影响论文说服力的问题：不能为了尽快训练而把 TRAJECT public
benchmark rows 当作 train split。当前合理路线是：

1. 重新提交 Stage 1 train-safe retrieval warmup；
2. Stage 1 完成后提交 Stage 2 train-safe unified base；
3. 继续获取/构造明确 train split 的 ToolBench-G3 / TRAJECT train 数据，再启用 Stage 3/4 target training；
4. TRAJECT public_data 先保留为 eval/dev-style target benchmark，不作为默认训练数据。

### 验证证据

新增/更新测试已覆盖：

```text
tests/test_unified_training_readiness.py
tests/test_skillret.py::test_unified_v2_retrieval_loader_excludes_public_unlabeled_rows_by_default
tests/test_full_base_train.py::test_full_base_training_excludes_public_unlabeled_rows_by_default
tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool
```

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_training_readiness.py \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  tests/test_skillret.py::test_unified_v2_retrieval_loader_reports_corrupt_jsonl_line \
  tests/test_skillret.py::test_unified_v2_retrieval_loader_excludes_public_unlabeled_rows_by_default \
  tests/test_full_base_train.py::test_train_clstr_full_base_with_model_respects_loss_masks_and_writes_checkpoint \
  tests/test_full_base_train.py::test_full_base_training_filters_to_allowed_benchmarks_and_records_audit \
  tests/test_full_base_train.py::test_full_base_training_excludes_public_unlabeled_rows_by_default \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_uses_unified_v2_data_and_local_cache \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  tests/test_sbatch_scripts.py::test_unified_stage3_hrpo_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints -q
# 13 passed, 2 warnings
```

```bash
python -m py_compile \
  clstr/full_base_train.py \
  clstr/qwen_full_base_train.py \
  clstr/retrieval_warmup.py \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/run_skillret_retrieval_warmup.py \
  scripts/run_clstr_qwen3_full_base_train.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/full_base_train.py \
  clstr/qwen_full_base_train.py \
  clstr/retrieval_warmup.py \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/run_skillret_retrieval_warmup.py \
  scripts/run_clstr_qwen3_full_base_train.py \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  tests/test_unified_training_readiness.py \
  tests/test_skillret.py \
  tests/test_full_base_train.py \
  tests/test_sbatch_scripts.py \
  description2.md
# no output
```

## 2026-05-26 v4 train-safe Stage 1/2 gate verification checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 的 Phase B/C/D。当前策略已经明确：
先在 unified CLSTR training dataset 上完成 CLSTR 的通用 routing / belief / act
训练，再回到 AppWorld 做 secondary transfer / combination plot。AppWorld 不再作为
当前训练主战场，避免在 CLSTR base/act 尚未完整训练时用单一 benchmark 反复调参。

### 本次确认的关键状态

1. unified v2 当前是真实可消费的训练集骨架
   - `data/clstr_unified_pretrain_v2/manifest.json` 状态为 `ok`。
   - 当前总量：

```text
trajectory_rows: 273269
retrieval_pairs: 374110
skill_count: 36183
```

   - 已进入训练池的安全来源：
     - Stage 1 retrieval: SKILLRET train + ToolRet-Training-20w train，共 336016 rows；
     - Stage 2 base trajectories: ALFWorld + ScienceWorld train，共 235175 rows。
   - TRAJECT public rows 当前 split 为 `public_train_or_eval_unlabeled`，默认不进入训练。
   - ToolBench-G3 官方完整数据仍缺失，因此三主 benchmark 数据尚未齐。

2. Stage 1 checkpoint 现在写入 train-safety 元数据
   - `clstr/retrieval_warmup.py` 对 `data_format="unified_v2"` 写入：

```text
train_safety.unified_v2_train_safe_retrieval: true
train_safety.public_train_or_eval_unlabeled_excluded: true
train_safety.excluded_retrieval_splits:
  dev, eval, evaluation, public_train_or_eval_unlabeled, test, valid, validation
```

   - 这保证后续可以区分“修复后 train-safe Stage 1”与“修复前可能吃到 TRAJECT public
     unlabeled rows 的 legacy Stage 1”。

3. Stage 2 sbatch 已强制 preflight
   - `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh` 在启动大模型训练前会先运行：

```text
scripts/run_clstr_stage2_preflight.py
```

   - preflight 检查：
     - Stage 1 checkpoint 是否存在；
     - `skill_table.E` 行数是否等于当前 `skill_pool.jsonl`；
     - `model_dim` 是否匹配；
     - unified Stage 1 checkpoint 是否带 train-safe retrieval metadata。
   - 旧的 `stage="clstr_unified_retrieval_v2"` checkpoint 如果缺少 train-safety 元数据，会被拒绝。

4. 当前 readiness audit 结论

已重新生成：

```bash
python scripts/audit_clstr_unified_training_readiness.py \
  --data_root data/clstr_unified_pretrain_v2 \
  --allowed_benchmarks traject_bench,toolbench_g3 \
  --stage1_checkpoint outputs/clstr_unified_stage1_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt \
  --stage2_checkpoint outputs/clstr_unified_stage2_full_base/checkpoints/clstr_full_base-step10000.pt \
  --output_path outputs/clstr_unified_readiness_audit/readiness_report.json
```

关键结果：

```text
status: action_required
leakage_audit.status: ok
missing_required_target_sources: [toolbench_g3]

retrieval_filter.retained_train_retrieval_rows: 336016
retrieval_filter.skipped_by_split.public_train_or_eval_unlabeled: 38094

base_train_filter.retained_train_trajectory_rows: 235175
base_train_filter.retained_by_benchmark:
  alfworld: 141741
  scienceworld: 93434

benchmark_filter.retained_train_trajectory_rows: 0

stage1_retrieval_warmup.ready_to_submit: true
stage2_full_base.ready_to_submit: false
stage2_full_base.blockers: [missing_stage1_checkpoint]
stage3_offline_hrpo.ready_to_submit: false
stage4_act.ready_to_submit: false
paper_readiness.full_three_benchmark_data_ready: false
```

### 执行含义

下一步不是继续在 AppWorld 上做小样本 smoke，而是：

1. 重新提交 train-safe Stage 1 retrieval warmup；
2. Stage 1 完成后通过 Stage 2 preflight，再提交 Stage 2 full-base；
3. 继续补 ToolBench-G3 官方完整数据，并获取/确认 TRAJECT 或 ToolBench 的明确 train split；
4. 只有当 target train rows 非 0 且 Stage 1/2 checkpoint 齐备时，才正式提交 Stage 3/4；
5. AppWorld 放到 Stage 4 后作为 secondary transfer / combination plot，不作为当前统一训练集的主驱动。

### 验证证据

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stage2_preflight.py \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  tests/test_skillret.py::test_unified_v2_retrieval_loader_reports_corrupt_jsonl_line \
  tests/test_skillret.py::test_unified_v2_retrieval_loader_excludes_public_unlabeled_rows_by_default \
  tests/test_full_base_train.py::test_full_base_training_excludes_public_unlabeled_rows_by_default \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  tests/test_unified_training_readiness.py -q
# 11 passed, 2 warnings
```

```bash
python -m py_compile \
  clstr/stage_preflight.py \
  clstr/retrieval_warmup.py \
  clstr/full_base_train.py \
  scripts/run_clstr_stage2_preflight.py \
  scripts/run_skillret_retrieval_warmup.py \
  scripts/audit_clstr_unified_training_readiness.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/stage_preflight.py \
  clstr/retrieval_warmup.py \
  clstr/full_base_train.py \
  scripts/run_clstr_stage2_preflight.py \
  scripts/run_skillret_retrieval_warmup.py \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  tests/test_stage2_preflight.py \
  tests/test_skillret.py \
  tests/test_full_base_train.py \
  tests/test_sbatch_scripts.py \
  tests/test_unified_training_readiness.py \
  description2.md
# no output
```

## 2026-05-26 v4 train-safe Stage 1 path isolation + readiness audit hardening

本记录继续对应 `docs/clstr_v4_plan.md` 的 Phase B/C/D。前一版已经让 Stage 2
preflight 拒绝缺少 train-safety metadata 的 legacy unified Stage 1 checkpoint；
本次进一步修正 readiness audit 和默认 sbatch 路径，避免旧作业产物被后续流程误用。

### 本次修正

1. readiness audit 与 Stage 2 preflight 对齐
   - `scripts/audit_clstr_unified_training_readiness.py` 不再只检查 Stage 1
     checkpoint 文件是否存在。
   - 若 Stage 1 checkpoint 存在，audit 会复用
     `clstr.stage_preflight.validate_stage2_routing_checkpoint()` 校验：
     - `skill_table.E` 行数与 `skill_pool.jsonl` 一致；
     - checkpoint 维度合法；
     - `stage="clstr_unified_retrieval_v2"` 时必须包含 train-safe retrieval metadata。
   - 若 checkpoint 缺少 train-safety，Stage 2/3/4 gate 会阻塞为：

```text
stage1_checkpoint_not_train_safe
```

2. readiness audit CLI 入口修复
   - 直接运行 `python scripts/audit_clstr_unified_training_readiness.py` 时，会先把
     repo root 加入 `sys.path`，不再依赖外部 `PYTHONPATH`。
   - preflight 校验改为懒加载：只有 Stage 1 checkpoint 文件存在、需要真实检查内容时才
     import `torch`。因此“checkpoint 缺失”的只读 audit 可以在不含 torch 的轻量
     Python 环境中完成。

3. train-safe Stage 1 默认输出目录与旧作业隔离
   - 旧 Stage 1 job `77139` 是在 train-safe retrieval filter 修复前提交的，仍在写旧目录：

```text
outputs/clstr_unified_stage1_retrieval_warmup
```

   - 为避免混用 legacy/unsafe checkpoint，新的 Stage 1 sbatch 默认输出改为：

```text
outputs/clstr_unified_stage1_train_safe_retrieval_warmup
```

   - Stage 2/3/4 sbatch 和 readiness audit 的默认 Stage 1 checkpoint 也同步切到：

```text
outputs/clstr_unified_stage1_train_safe_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt
```

   - 旧作业不被中断，但其 checkpoint 不再是后续默认输入。

### 当前真实 readiness audit 结果

已用新的默认 train-safe Stage 1 路径重新生成：

```bash
python scripts/audit_clstr_unified_training_readiness.py \
  --data_root data/clstr_unified_pretrain_v2 \
  --allowed_benchmarks traject_bench,toolbench_g3 \
  --output_path outputs/clstr_unified_readiness_audit/readiness_report.json
```

关键结果：

```text
status: action_required
leakage_audit.status: ok
manifest.trajectory_rows: 273269
manifest.retrieval_pairs: 374110
manifest.skill_count: 36183
missing_required_target_sources: [toolbench_g3]

retrieval_filter.retained_train_retrieval_rows: 336016
retrieval_filter.skipped_by_split.public_train_or_eval_unlabeled: 38094

base_train_filter.retained_train_trajectory_rows: 235175
base_train_filter.retained_by_benchmark:
  alfworld: 141741
  scienceworld: 93434

benchmark_filter.retained_train_trajectory_rows: 0

stage1_retrieval_warmup.ready_to_submit: true
stage2_full_base.ready_to_submit: false
stage2_full_base.blockers: [missing_stage1_checkpoint]
stage2_full_base.stage1_checkpoint:
  outputs/clstr_unified_stage1_train_safe_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt

stage3_offline_hrpo.ready_to_submit: false
stage4_act.ready_to_submit: false
paper_readiness.full_three_benchmark_data_ready: false
```

### 验证证据

已先写失败测试确认：
- readiness audit 不能把缺少 train-safety 的 legacy Stage 1 checkpoint 视为可用；
- readiness audit CLI 在无 `PYTHONPATH` 时也能从 repo root 直接运行；
- 缺 Stage 1 checkpoint 的 audit 不应依赖 `torch`；
- Stage 1 train-safe 输出目录必须与旧作业目录隔离，Stage 2/3/4 默认读取新目录。

已运行并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_training_readiness.py \
  tests/test_stage2_preflight.py \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  tests/test_skillret.py::test_unified_v2_retrieval_loader_reports_corrupt_jsonl_line \
  tests/test_skillret.py::test_unified_v2_retrieval_loader_excludes_public_unlabeled_rows_by_default \
  tests/test_full_base_train.py::test_full_base_training_excludes_public_unlabeled_rows_by_default \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_uses_unified_v2_data_and_local_cache \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  tests/test_sbatch_scripts.py::test_unified_stage3_hrpo_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints -q
# 17 passed, 2 warnings
```

```bash
python -m py_compile \
  scripts/audit_clstr_unified_training_readiness.py \
  clstr/stage_preflight.py \
  clstr/retrieval_warmup.py \
  clstr/full_base_train.py \
  scripts/run_clstr_stage2_preflight.py \
  scripts/run_skillret_retrieval_warmup.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh
# exit 0
```

```bash
git diff --check -- \
  scripts/audit_clstr_unified_training_readiness.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py \
  description2.md \
  clstr/stage_preflight.py \
  clstr/retrieval_warmup.py \
  clstr/full_base_train.py \
  scripts/run_clstr_stage2_preflight.py \
  scripts/run_skillret_retrieval_warmup.py \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  tests/test_stage2_preflight.py \
  tests/test_skillret.py \
  tests/test_full_base_train.py
# no output
```

### 已提交的新 train-safe Stage 1 作业

旧作业 `77139` 未中断，仍视为 legacy/unsafe 路径作业。新的 train-safe Stage 1
已经提交到独立输出目录：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh
# Submitted batch job 77147
```

默认输出：

```text
outputs/clstr_unified_stage1_train_safe_retrieval_warmup
```

后续检查不要频繁轮询；短任务约 15 分钟、长任务约 30 分钟再看一次即可。Stage 2
必须等待该目录下的 train-safe checkpoint 生成，并通过 Stage 2 preflight 后再提交。

## 2026-05-26 v4 ToolBench-G3 official data blocker + static fallback guard

本记录继续对应 `docs/clstr_v4_plan.md` 的 Phase B/C。当前 ToolBench-G3 仍是
unified CLSTR training dataset 的主要数据缺口。本次没有把任何替代数据并入正式训练流，
只增强下载/验证诊断，避免把 ToolBench-Static 或 `data_example` 误当作官方 G3。

### 当前本地 ToolBench 状态

1. 本地已有 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench` 代码仓库和
   `data_example`，但没有官方完整：

```text
ToolBench/data/instruction/G3_query.json
ToolBench/data/answer/G3_answer/*.json
```

2. 官方 Google Drive 下载再次失败：

```bash
python scripts/download_toolbench_data.py \
  --download \
  --source google \
  --manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/download_google_retry_manifest.json
```

失败原因为登录节点无法访问 `drive.google.com`：

```text
Network is unreachable
```

3. 清华云链接之前下载到的是 HTML，不是 zip。已确认 HTML 内容为：

```text
Link does not exist.
```

因此当前无法自动获取官方 ToolBench-G3 `data.zip`。仍需要手动提供官方数据：

```text
/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data.zip
```

或直接解压后的：

```text
/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data
```

### ToolBench-Static 处理原则

查到 ModelScope OSS 有可访问的 ToolBench-Static zip：

```text
https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/toolbench-static/data.zip
```

HEAD 探测结果：

```text
status: 200
content-length: 1519465
content-type: application/zip
```

但该 zip 只有：

```text
data/toolbench_static/in_domain.json
data/toolbench_static/out_of_domain.json
```

它没有官方 G3 所需的 `instruction/G3_query.json` 和 `answer/G3_answer` trajectory
tree，因此只能作为静态 retrieval/schema plumbing smoke，不允许替代官方
ToolBench-G3 进入主训练集或论文主表。

### 本次改动

1. `scripts/download_toolbench_data.py`
   - 在 manifest 中新增 `static_smoke_fallback`：

```text
paper_role: smoke_only_not_official_g3
may_replace_official_toolbench_g3: false
```

   - `verify_toolbench_data()` 现在会报告 `static_smoke_assets` 是否存在，但即使存在也仍
     返回 `missing_required_files`，除非官方 G3 的 `instruction/G3_query.json` 和
     `answer/G3_answer/*.json` 都存在。

2. `tests/test_toolbench_g3_import.py`
   - 新增回归覆盖：
     - dry-run manifest 必须记录 static fallback 的 smoke-only 角色；
     - 只有 `toolbench_static/in_domain.json` / `out_of_domain.json` 时，不能通过官方 G3 验证。

### 验证证据

已先写失败测试，再实现并通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py -q
# 10 passed
```

```bash
python -m py_compile scripts/download_toolbench_data.py scripts/import_toolbench_g3.py
# exit 0
```

```bash
bash -n scripts/sbatch/run_import_toolbench_g3.sh
# exit 0
```

```bash
git diff --check -- scripts/download_toolbench_data.py tests/test_toolbench_g3_import.py description2.md
# no output
```

### unified builder 防误接入确认

进一步补充 `tests/test_unified_pretrain_v2.py` 回归，确认即使本地存在：

```text
ToolBench/data/toolbench_static/in_domain.json
ToolBench/data/toolbench_static/out_of_domain.json
```

`scripts/build_clstr_unified_pretrain.py` 也不会把它识别为官方 ToolBench-G3。当前判定仍为：

```text
toolbench_g3.available: false
toolbench_g3.reason_if_missing: source_exists_but_required_training_files_missing
```

只有以下任一条件满足时，ToolBench-G3 才能进入 unified v2：

```text
data/toolbench_g3/{trajectories.jsonl,retrieval.jsonl,skills.jsonl}
```

或官方完整数据根目录包含：

```text
instruction/G3_query.json
answer/G3_answer/*.json
```

并通过 import 规范化。

补充验证：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py \
  tests/test_toolbench_g3_import.py -q
# 20 passed
```

## 2026-05-26 v4 Stage 2 afterok dependency submission

为减少人工轮询并避免 Stage 2 在 train-safe Stage 1 完成前误启动，已提交 Slurm
依赖作业：

```bash
sbatch --dependency=afterok:77147 --gpus=1 -p gpu_h200 scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# Submitted batch job 77148
```

含义：

```text
77147: train-safe Stage 1 retrieval warmup
77148: Stage 2 full-base train, only starts after 77147 exits successfully
```

Stage 2 脚本内部仍会先执行：

```text
scripts/run_clstr_stage2_preflight.py
```

因此即使依赖满足，也会再次检查：
- Stage 1 checkpoint 是否存在；
- `skill_table.E` 与当前 `skill_pool.jsonl` 行数是否匹配；
- unified Stage 1 checkpoint 是否包含 train-safe retrieval metadata。

后续仍按约定不要频繁检查队列；长作业约 30 分钟后再看一次即可。

## 2026-05-26 v4 generic retrieval evaluator checkpoint

本记录对应 `docs/clstr_v4_plan.md` 的 Phase H / 主表评测基础设施补齐。
当前主表 #3 ToolRet 以及 TRAJECT / ToolBench-G3 的 routing-style 中间评测都需要
统一的 run/qrels 指标口径，因此先新增一个不依赖 GPU、不依赖 `pytrec_eval` 的通用
retrieval evaluator。它不是最终主表结果，只是后续 CLSTR checkpoint 导出 run 后的
标准评测入口。

### 已完成的主要改动

1. `clstr/retrieval_metrics.py`
   - 新增通用 `qrels.jsonl` loader，兼容 `skill_id` / `tool_id` / `doc_id` 等字段；
   - 新增 JSONL prediction loader，兼容 `ranked_skill_ids` / `ranked_tool_ids` 等字段；
   - 新增 TREC run loader，兼容 `query_id Q0 doc_id rank score run_name`；
   - 新增纯 Python 指标：
     - `NDCG@K`
     - `Recall@K`
     - `MAP@K`
     - `Precision@K`
     - `evaluated_queries`
     - `missing_run_queries`
   - 新增 `evaluate_retrieval_run()`，统一写出 `metrics.json`。

2. `scripts/evaluate_retrieval_run.py`
   - 新增 CLI：

```bash
python scripts/evaluate_retrieval_run.py \
  --qrels_path path/to/qrels.jsonl \
  --run_path path/to/run.tsv \
  --output_dir outputs/... \
  --run_format trec \
  --k_values 5 10 \
  --benchmark toolret \
  --method clstr
```

   - 该入口可直接用于 ToolRet 主表指标，也可用于 TRAJECT / ToolBench-G3 的
     offline routing recall/next-tool ranking 评测。

3. `tests/test_generic_retrieval_eval.py`
   - 新增回归覆盖：
     - ToolRet-style `NDCG/Recall/MAP` 数值计算；
     - unified qrels + JSONL ranked prediction loader；
     - TREC run CLI 与 programmatic API 都能写出 `metrics.json`。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
ModuleNotFoundError: No module named 'clstr.retrieval_metrics'
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_generic_retrieval_eval.py -q
# 3 passed
```

## 2026-05-26 v4 retrieval qrels normalization checkpoint

本记录继续补齐 `docs/clstr_v4_plan.md` 的 Phase H / 主表评测基础设施。
上一节已经有通用 evaluator，但后续 ToolRet / TRAJECT / ToolBench-G3 的 routing
评测还需要把 unified retrieval rows 稳定转成标准 `qrels.jsonl`。本次新增的工具只做
schema 规范化，不做模型打分，不引入任务特异性规则。

### 已完成的主要改动

1. `clstr/retrieval_qrels.py`
   - 新增 `export_retrieval_qrels()`；
   - 支持从 CLSTR unified `retrieval.jsonl` 中读取：
     - `query_id`
     - `positive_skill_id` / `positive_skill_ids`
     - `negative_skill_ids`
     - `source`
     - `provenance.split`
   - 输出标准 qrels rows：

```json
{"query_id": "...", "skill_id": "...", "relevance": 1, "source": "...", "split": "train"}
```

   - 可选写出负例 `relevance: 0`；
   - 支持 `allowed_sources` / `allowed_splits` 过滤，避免把
     `public_train_or_eval_unlabeled` 等不安全 split 误用于训练/评测 qrels；
   - report 会记录 `skipped_by_source` / `skipped_by_split` / `skipped_missing_fields`。

2. `scripts/export_retrieval_qrels.py`
   - 新增 CLI：

```bash
python scripts/export_retrieval_qrels.py \
  --retrieval_path data/clstr_unified_pretrain_v2/retrieval.jsonl \
  --output_path outputs/.../qrels.jsonl \
  --report_path outputs/.../qrels_export_report.json \
  --allowed_sources toolret_training \
  --allowed_splits train
```

   - 该输出可直接接入上一节的 `scripts/evaluate_retrieval_run.py`。

3. `tests/test_export_retrieval_qrels.py`
   - 新增回归覆盖：
     - 正例/负例 qrels 规范化；
     - source 与 split filter；
     - CLI 写出 report。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
ModuleNotFoundError: No module named 'clstr.retrieval_qrels'
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_export_retrieval_qrels.py tests/test_generic_retrieval_eval.py -q
# 6 passed
```

```bash
python -m py_compile \
  clstr/retrieval_qrels.py \
  clstr/retrieval_metrics.py \
  scripts/export_retrieval_qrels.py \
  scripts/evaluate_retrieval_run.py
# exit 0
```

```bash
git diff --check -- \
  clstr/retrieval_qrels.py \
  clstr/retrieval_metrics.py \
  scripts/export_retrieval_qrels.py \
  scripts/evaluate_retrieval_run.py \
  tests/test_export_retrieval_qrels.py \
  tests/test_generic_retrieval_eval.py \
  description2.md
# no output
```

## 2026-05-26 v4 ToolRet eval data normalization checkpoint

本记录对应 `docs/clstr_v4_plan.md` 的 Phase B/H：ToolRet 不只有
ToolRet-Training-20w 训练集，还需要官方评测侧的 `ToolRet-Queries` /
`ToolRet-Tools` 规范化入口，才能最终产出主表 #3 的 `NDCG@5/10`、
`Recall@5/10`、`MAP@10`。

### 已完成的主要改动

1. `clstr/toolret_eval_import.py`
   - 新增 `import_toolret_eval()`；
   - 支持本地 JSONL `query_source` / `tool_source`；
   - 未提供本地源时可通过 `datasets.load_dataset()` 访问：
     - `mangopy/ToolRet-Queries`
     - `mangopy/ToolRet-Tools`
   - 默认使用 `HF_ENDPOINT=https://hf-mirror.com`；
   - 输出：
     - `queries.jsonl`
     - `skills.jsonl`
     - `qrels.jsonl`
     - `manifest.json`
   - `labels` 支持官方字符串 JSON 和 list[dict] 两种形式；
   - manifest 会记录 `missing_qrel_skill_ids`，避免 qrels 指向不在 tool pool 里的
     tool 时静默通过。

2. `scripts/import_toolret_eval.py`
   - 新增 CLI：

```bash
python scripts/import_toolret_eval.py \
  --query_source path/to/toolret_queries.jsonl \
  --tool_source path/to/toolret_tools.jsonl \
  --output_dir data/toolret_eval \
  --split test
```

3. `scripts/sbatch/run_import_toolret_eval.sh`
   - 新增 sbatch 包装；
   - 默认要求 `QUERY_SOURCE` 和 `TOOL_SOURCE` 指向登录节点已下载好的本地文件；
   - 只有显式 `ALLOW_HF_STREAMING=1` 时才允许无本地源尝试 HF；
   - cache 固定在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache` 下；
   - 工作目录固定在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr`。

4. 测试
   - 新增 `tests/test_toolret_eval_import.py`；
   - 扩展 `tests/test_sbatch_scripts.py` 覆盖 ToolRet eval import sbatch。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
ModuleNotFoundError: No module named 'clstr.toolret_eval_import'
```

补 sbatch 测试时初始失败为：

```text
FileNotFoundError: scripts/sbatch/run_import_toolret_eval.sh
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolret_eval_import.py \
  tests/test_sbatch_scripts.py::test_toolret_eval_import_sbatch_requires_local_sources_and_uses_hf_mirror -q
# 4 passed
```

```bash
python -m py_compile clstr/toolret_eval_import.py scripts/import_toolret_eval.py
# exit 0
```

```bash
bash -n scripts/sbatch/run_import_toolret_eval.sh
# exit 0
```

## 2026-05-26 v4 generic CLSTR retrieval export checkpoint

本记录对应 `docs/clstr_v4_plan.md` 的 Phase H：前面已经补齐 ToolRet eval
数据规范化、qrels 规范化和通用 evaluator；本次补齐从 CLSTR checkpoint 到
`predictions.jsonl` / TREC `run.tsv` 的通用导出层。这样 Stage 1/2/3 checkpoint
可以在 ToolRet / TRAJECT / ToolBench-G3 的 routing-style qrels 上复用同一套评测链路。

### 已完成的主要改动

1. `clstr/clstr_retrieval_export.py`
   - 新增 `export_clstr_retrieval_run()`；
   - 输入：
     - `queries.jsonl`
     - `skills.jsonl`
     - CLSTR checkpoint（可选）
     - base model path
   - 输出：
     - `predictions.jsonl`
     - `run.tsv`
     - `run_report.json`
   - 默认使用 `query_text`，没有时回退到 `instruction + query`；
   - 复用现有 `CLSTRModel` / `SkillTable.retrieval_logits`；
   - checkpoint config 解析复用 `resolve_clstr_export_config()`，避免与 SKILLRET exporter
     形成两套模型初始化口径。

2. `scripts/export_clstr_retrieval_run.py`
   - 新增 CLI：

```bash
python scripts/export_clstr_retrieval_run.py \
  --queries_path data/toolret_eval/queries.jsonl \
  --skills_path data/toolret_eval/skills.jsonl \
  --output_dir outputs/toolret_eval/clstr_retrieval \
  --base_model_name models/Qwen3-8B \
  --checkpoint_path outputs/clstr_unified_stage1_train_safe_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt \
  --top_k 50 \
  --model_dim 1024 \
  --local_files_only \
  --disable_cross_encoder
```

3. `scripts/sbatch/run_export_clstr_retrieval_run.sh`
   - 新增 sbatch 包装；
   - 默认面向 ToolRet eval：
     - `data/toolret_eval/queries.jsonl`
     - `data/toolret_eval/skills.jsonl`
     - `data/toolret_eval/qrels.jsonl`
   - 默认 checkpoint 为 train-safe Stage 1：
     `outputs/clstr_unified_stage1_train_safe_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
   - `RUN_EVAL=1` 时会自动调用 `scripts/evaluate_retrieval_run.py`，产出
     `metrics.json`；
   - 也可通过环境变量替换成 TRAJECT / ToolBench-G3 的同构
     `queries/skills/qrels`。

4. 测试
   - 新增 `tests/test_clstr_retrieval_export.py`；
   - 扩展 `tests/test_sbatch_scripts.py` 覆盖通用 CLSTR retrieval export sbatch。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
ModuleNotFoundError: No module named 'clstr.clstr_retrieval_export'
```

补 sbatch 测试时初始失败为：

```text
FileNotFoundError: scripts/sbatch/run_export_clstr_retrieval_run.sh
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_clstr_retrieval_export.py \
  tests/test_sbatch_scripts.py::test_clstr_retrieval_export_sbatch_exports_and_optionally_evaluates_generic_run -q
# 3 passed, 2 warnings
```

```bash
python -m py_compile clstr/clstr_retrieval_export.py scripts/export_clstr_retrieval_run.py
# exit 0
```

```bash
bash -n scripts/sbatch/run_export_clstr_retrieval_run.sh
# exit 0
```

## 2026-05-26 v4 TRAJECT split-safe training protocol checkpoint

本次修复一个会直接影响 v4 论文主张的问题：当前旧
`data/clstr_unified_pretrain_v2_toolbench_g3` 虽然包含 TRAJECT public_data，但
readiness 审计显示所有 TRAJECT rows 都被标为 `public_train_or_eval_unlabeled` 并在
训练过滤器中跳过。因此旧 Stage 2/3/4 实际只会用 ToolBench-G3 目标轨迹，不会真正训练
TRAJECT sequential belief/m_t，这会削弱“trajectory-aware / sequential routing”的核心论证。

### 根因

TRAJECT-Bench public_data 当前没有官方 train/dev/test split。旧实现为避免泄露，保守地将
TRAJECT trajectory/retrieval rows 标成：

```text
public_train_or_eval_unlabeled
```

`audit_clstr_unified_training_readiness.py` 和 `clstr/retrieval_warmup.py` 会把这个 split
当成 unsafe split 跳过。这是安全的，但导致 TRAJECT 只可评测、不可训练。

### 已完成的主要改动

1. `clstr/traject_split.py`
   - 新增 `assign_traject_split()`；
   - 使用稳定 SHA256 hash，对 `trajectory_id` 做 trajectory-level deterministic split；
   - 默认比例：
     - train: 80%
     - dev: 10%
     - test: 10%
   - 同一条 trajectory 的所有 step 永远进入同一个 split，避免 step-level leakage。

2. `scripts/build_clstr_unified_pretrain.py`
   - TRAJECT Stream A / Stream B 只写入 `assign_traject_split(trajectory_id) == "train"` 的 rows；
   - 写入 provenance：

```json
{
  "split": "train",
  "split_policy": "deterministic_hash_trajectory_level_v1"
}
```

   - 对 dev/test rows 只计入 manifest skip 统计，不进入训练文件；
   - 修复直接运行脚本时 `clstr` import 失败的问题，在脚本入口加入 repo root 到 `sys.path`。

3. `clstr/traject_eval_import.py` / `scripts/import_traject_eval.py`
   - 新增 `split_partition` / `--split_partition`；
   - 可导出 deterministic `train/dev/test` 任一 held-out partition；
   - `scripts/sbatch/run_import_traject_eval.sh` 默认：

```bash
SPLIT=test
SPLIT_PARTITION=test
```

   - 因此后续 `data/traject_eval` 默认只包含与 train partition disjoint 的 TRAJECT held-out test。

4. 新 split-safe 默认流水线
   - 为避免覆盖当前旧作业正在用的正式数据目录，后续默认切到新的数据线：

```text
data/clstr_unified_pretrain_v2_toolbench_g3_traject_split
outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup
outputs/clstr_unified_stage2_toolbench_g3_traject_split_full_base
outputs/clstr_unified_stage3_toolbench_g3_traject_split_hrpo
outputs/clstr_unified_stage4_toolbench_g3_traject_split_act
```

   - 已更新相关 sbatch 默认路径：
     - `scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh`
     - `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh`
     - `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
     - `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`
     - `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
     - `scripts/sbatch/run_clstr_unified_readiness_audit.sh`
     - retrieval export eval scripts 的默认 Stage 1 checkpoint。

### 真实 TRAJECT public_data split 分布

只读扫描 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/TRAJECT-Bench/public_data` 得到：

```text
trajectory_counts:
  train: 4699
  dev: 602
  test: 569
step_counts:
  train: 30526
  dev: 3945
  test: 3623
by_type:
  parallel:
    train: 3190
    dev: 424
    test: 386
  sequential:
    train: 1509
    dev: 178
    test: 183
```

这意味着新 unified 数据重建后，TRAJECT 将贡献约 30.5k train step rows，
其中 sequential train trajectories 约 1.5k 条；held-out test 约 3.6k step rows，
与 train trajectory disjoint。

### 注意

- 当前旧 `data/clstr_unified_pretrain_v2_toolbench_g3` 没有被覆盖；
- 当前已运行的旧 Stage 1 作业仍基于旧数据线；
- 后续要证明 v4 方法价值，应使用新 split-safe 数据线重新构建并重新提交 Stage 1/2/3/4；
- 旧数据线结果只能作为历史参考，不能作为 TRAJECT sequential training 结论。

### 验证证据

已按 TDD 先写失败测试，初始失败包括：

```text
ModuleNotFoundError: No module named 'clstr.traject_split'
```

以及直接运行 builder 时：

```text
ModuleNotFoundError: No module named 'clstr'
```

实现后通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_traject_split.py \
  tests/test_traject_eval_import.py::test_import_traject_eval_can_filter_deterministic_heldout_split \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_only_uses_train_partition_of_traject_public_data \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_converts_traject_bench_steps_and_tools -q
# 5 passed
```

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_cli_defaults_use_toolbench_g3_stage_paths \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_uses_unified_v2_data_and_local_cache \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  tests/test_sbatch_scripts.py::test_unified_stage3_hrpo_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_toolbench_g3_unified_pipeline_sbatch_runs_import_build_and_readiness_gates \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_cli_help_runs_from_repo_root -q
# 7 passed
```

```bash
python -m py_compile \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/build_clstr_unified_pretrain.py \
  clstr/traject_split.py \
  clstr/traject_eval_import.py \
  clstr/traject_sequence_proxy.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  scripts/sbatch/run_import_traject_eval.sh \
  scripts/sbatch/run_export_traject_retrieval_run.sh
# exit 0
```

## 2026-05-26 v4 TRAJECT sequence proxy evaluator checkpoint

本次在 TRAJECT routing eval 后补齐一个 **selection-only trajectory sequence proxy**，
用于 Stage 1/2/3 导出的 `run.tsv` 产出轨迹级选择指标。这个报告只衡量 CLSTR 是否选对
skill/tool 序列，不计算官方 TRAJECT 的参数使用、LLM 满意度或最终答案正确性。

### 已完成的主要改动

1. `clstr/traject_sequence_proxy.py`
   - 新增 `evaluate_traject_sequence_proxy()`；
   - 输入：
     - `data/traject_eval/queries.jsonl`
     - `data/traject_eval/qrels.jsonl`
     - retrieval `run.tsv` 或 jsonl prediction；
   - 按 `trajectory_id + step_index` 重构每条 trajectory 的 GT skill 序列和 top-1 predicted skill 序列；
   - 输出：
     - `step_top1_accuracy`
     - `ordered_exact_match`
     - `unordered_exact_match`
     - `inclusion`
     - `by_trajectory_type`，分别统计 `parallel` / `sequential`。

2. `scripts/evaluate_traject_sequence_proxy.py`
   - 新增 CLI：

```bash
python scripts/evaluate_traject_sequence_proxy.py \
  --queries_path data/traject_eval/queries.jsonl \
  --qrels_path data/traject_eval/qrels.jsonl \
  --run_path outputs/traject_eval/clstr_retrieval/run.tsv \
  --output_dir outputs/traject_eval/clstr_retrieval \
  --run_format trec \
  --method clstr_unified_retrieval
```

3. `scripts/sbatch/run_export_traject_retrieval_run.sh`
   - 新增 `RUN_SEQUENCE_PROXY_EVAL=1` 默认后处理；
   - 在普通 retrieval metric eval 后自动运行 `scripts/evaluate_traject_sequence_proxy.py`；
   - 输出 `outputs/traject_eval/clstr_retrieval/traject_sequence_proxy_metrics.json`；
   - 如果 `queries/qrels/run.tsv` 缺失，只跳过 proxy eval，不影响已有 retrieval eval 逻辑。

### 指标边界

- 该 evaluator 可以支持论文内部分析：
  - Stage 2/3 是否提升 sequential trajectory skill sequence；
  - sequential vs parallel 的 belief/m_t 贡献是否稳定；
  - ordered/unordered proxy 是否和 stepwise recall 同向。
- 该 evaluator **不能替代** 官方 TRAJECT：
  - `Usage`：仍需要下游 LM executor 输出带参数的 tool call；
  - `Traj-Satisfy`：仍需要官方 LLM judge；
  - `Acc`：仍需要执行/生成 final answer 后做官方答案评测。

### 验证证据

已按 TDD 先写失败测试，初始失败包括：

```text
ModuleNotFoundError: No module named 'clstr.traject_sequence_proxy'
```

以及 sbatch 回归：

```text
assert 'scripts/evaluate_traject_sequence_proxy.py' in script
```

实现后通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_traject_sequence_proxy.py -q
# 2 passed
```

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py::test_traject_retrieval_export_sbatch_exports_and_evaluates_traject_run -q
# 1 passed
```

```bash
python -m py_compile clstr/traject_sequence_proxy.py scripts/evaluate_traject_sequence_proxy.py
# exit 0
```

```bash
bash -n scripts/sbatch/run_export_traject_retrieval_run.sh
# exit 0
```

## 2026-05-26 v4 TRAJECT native metric readiness checkpoint

本次补齐 TRAJECT-Bench eval 数据的主表指标可用性审计，目的是避免把当前
stepwise routing eval / trajectory proxy 误写成官方完整 EM / Inclusion / Usage /
Traj-Satisfy / Acc。

### 已完成的主要改动

1. `clstr/traject_eval_audit.py`
   - 在原有 qrels/skills/queries 完整性检查基础上新增：
     - `trajectory_reconstruction`
     - `paper_metric_readiness`
   - 审计会检查 `trajectory_id` 与 `step_index` 是否覆盖所有 positive qrel query；
   - 将 TRAJECT 结果明确分成三层：
     - `routing_eval: ready`：可用现有 CLSTR retrieval run 计算 Recall@K / NDCG@K / MAP@K；
     - `trajectory_sequence_proxy: ready`：可按 `trajectory_id + step_index` 重构 top-1 skill sequence，作为轨迹级 selection proxy；
     - 官方主指标边界：
       - `official_em_inclusion: partial_proxy_ready`，只能做 selection-only 近似；
       - `official_usage: blocked`，需要下游 executor 生成带参数的 tool call；
       - `official_traj_satisfy: blocked`，需要官方 TRAJECT judge / LLM judge；
       - `official_acc: blocked`，需要 executor 输出 final answer 再做官方答案评测。

2. `tests/test_traject_eval_import.py`
   - 新增回归测试 `test_audit_traject_eval_data_reports_native_metric_readiness_boundaries`；
   - 该测试用极小 stepwise TRAJECT 数据确认审计报告不会把 proxy 指标误标为官方完整主指标。

### 真实数据审计结果

已在当前本地数据上运行：

```bash
conda run -n reasoning_trap env PYTHONPATH=. python \
  scripts/audit_traject_eval_data.py \
  --data_dir data/traject_eval \
  --output_path outputs/traject_eval/traject_eval_audit.json \
  --fail_on_action_required
```

结果摘要：

```text
status: ok
query_count: 38094
skill_count: 689
qrel_count: 38094
positive_qrel_count: 38094
trajectory_count: 5870
trajectory_type_counts:
  parallel: 4000
  sequential: 1870
max_steps_per_trajectory: 13
ordered_sequence_proxy_ready: true
missing_positive_qrel_skill_ids: []
missing_positive_qrel_query_ids: []
duplicate_skill_ids: []
```

结论：

- 当前 TRAJECT normalized eval 数据本身已经 ready；
- CLSTR 可以立刻在该数据上做 stepwise routing eval 和 trajectory sequence proxy；
- 若论文主表严格采用 TRAJECT 官方 `Usage / Traj-Satisfy / Acc`，还必须接入下游 LM executor / 官方 judge；
- 在 executor 未完成前，TRAJECT 主表不能只用当前 routing proxy 冒充官方完整 task-success 指标。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
KeyError: 'paper_metric_readiness'
```

实现后通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_traject_eval_import.py::test_audit_traject_eval_data_reports_native_metric_readiness_boundaries -q
# 1 passed
```

完整 TRAJECT eval import/audit 测试通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_traject_eval_import.py -q
# 8 passed
```

语法检查通过：

```bash
python -m py_compile clstr/traject_eval_audit.py scripts/audit_traject_eval_data.py
# exit 0
```

## 2026-05-26 v4 StableToolBench virtual API readiness checkpoint

本记录继续服务于 ToolBench-G3 官方 SoPR 链路。前一阶段已经确认：
`StableToolBench/toolenv/tools` 已由 `G3_instruction.json` 构建完成，raw generation 仍被
`missing_service_url_for_stabletoolbench_virtual_api` 和
`missing_openai_key_for_chatgpt_function` 阻塞。本次补齐 virtual API server 的可审计入口，
并修复 StableToolBench mirrorapi 配置中的旧机器路径。

### 已完成的主要改动

1. 新增 virtual API readiness audit
   - `clstr/stabletoolbench_virtual_api.py`
   - `scripts/audit_stabletoolbench_virtual_api.py`
   - 支持 `mirrorapi` / `mirrorapi_cache` / `gpt_cache` 三种 StableToolBench server 模式；
   - 检查 server script、YAML config、`tools_folder`、tool json 数量、端口、模型后端字段；
   - 明确拒绝 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 以外的 `tools_folder`；
   - 明确拒绝 `data_example` / `solvable_queries_example`；
   - 对 `/root/...` 这类不可访问路径只输出 blocker，不再因 `PermissionError` 崩溃；
   - 输出 `service_url`，例如 `http://127.0.0.1:12001/virtual`。

2. 新增 sbatch wrapper
   - `scripts/sbatch/run_stabletoolbench_virtual_api_server.sh`
   - 默认 `RUN_SERVER=0`，只做 readiness audit；
   - 显式 `RUN_SERVER=1` 时才通过 sbatch 启动 server；
   - 保持 cache / HF mirror 路径都在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 下。

3. 修复 StableToolBench mirrorapi 配置
   - `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/server/config_mirrorapi.yml`
   - `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/server/config_mirrorapi_cache.yml`
   - 将 `tools_folder` 从旧的 `/root/gzc/...` 改为：

```text
/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolenv/tools
```

### 真实 readiness 状态

已重新审计 `mirrorapi`：

```bash
python scripts/audit_stabletoolbench_virtual_api.py \
  --stabletoolbench_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench \
  --mode mirrorapi \
  --autodl_tmp_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp \
  --output_path outputs/toolbench_g3/stabletoolbench_virtual_api_readiness.json
```

结果：

```text
status: ok
can_start_virtual_api: true
service_url: http://127.0.0.1:12001/virtual
tool_json_file_count: 7
blockers: []
```

已重新审计 `mirrorapi_cache`：

```text
status: ok
can_start_virtual_api: true
service_url: http://127.0.0.1:12001/virtual
tool_json_file_count: 7
blockers: []
```

随后把该 `service_url` 接回 raw generation readiness：

```text
status: action_required
can_generate_raw_answers: false
blockers:
  - missing_openai_key_for_chatgpt_function
service_url: http://127.0.0.1:12001/virtual
tool_json_file_count: 7
```

解释：virtual API 的工具目录 gate 已经通过；官方 raw answer generation 现在不再缺
`SERVICE_URL`，但仍缺 downstream executor 的 `OPENAI_KEY`。此外，`127.0.0.1` 只在
virtual API server 与 raw generation 位于同一个 sbatch 作业/同一节点时成立；若拆成两个
作业，必须显式传入 server 所在节点可访问的 `SERVICE_URL`。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
ModuleNotFoundError: No module named 'clstr.stabletoolbench_virtual_api'
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stabletoolbench_virtual_api.py \
  tests/test_sbatch_scripts.py::test_stabletoolbench_virtual_api_sbatch_audits_before_server_start -q
# 6 passed
```

完整相关 StableToolBench readiness 测试通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stabletoolbench_toolenv.py \
  tests/test_stabletoolbench_raw_generation.py \
  tests/test_stabletoolbench_answer_conversion.py \
  tests/test_stabletoolbench_pass_rate.py \
  tests/test_stabletoolbench_virtual_api.py \
  tests/test_sbatch_scripts.py::test_stabletoolbench_toolenv_build_sbatch_uses_official_solvable_queries \
  tests/test_sbatch_scripts.py::test_stabletoolbench_raw_generation_sbatch_audits_before_generation \
  tests/test_sbatch_scripts.py::test_stabletoolbench_answer_conversion_sbatch_audits_raw_answers_before_convert \
  tests/test_sbatch_scripts.py::test_stabletoolbench_pass_rate_sbatch_audits_before_official_sopr \
  tests/test_sbatch_scripts.py::test_stabletoolbench_virtual_api_sbatch_audits_before_server_start -q
# 25 passed
```

```bash
bash -n scripts/sbatch/run_stabletoolbench_virtual_api_server.sh \
  scripts/sbatch/run_stabletoolbench_raw_generation.sh \
  scripts/sbatch/run_convert_stabletoolbench_answers.sh \
  scripts/sbatch/run_stabletoolbench_pass_rate.sh \
  scripts/sbatch/run_build_stabletoolbench_toolenv.sh
# exit 0
```

```bash
conda run -n reasoning_trap env PYTHONPATH=. python -m py_compile \
  clstr/stabletoolbench_virtual_api.py \
  scripts/audit_stabletoolbench_virtual_api.py
# exit 0
```

```bash
git diff --check -- clstr/stabletoolbench_virtual_api.py \
  scripts/audit_stabletoolbench_virtual_api.py \
  scripts/sbatch/run_stabletoolbench_virtual_api_server.sh \
  tests/test_stabletoolbench_virtual_api.py \
  tests/test_sbatch_scripts.py description2.md
# exit 0
```

```bash
git -C /data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench diff --check -- \
  server/config_mirrorapi.yml server/config_mirrorapi_cache.yml
# exit 0
```

## 2026-05-26 v4 skill dedup borderline review checkpoint

本记录继续对应 `docs/clstr_v4_plan.md` 中 D11-v2 / D15-v2 的 unified vocab v2。
当前真实 `data/clstr_unified_pretrain_v2_toolbench_g3` 已经不是旧的 identity-only
skill pool，而是规则版 partial dedup：

```text
data/clstr_unified_pretrain_v2_toolbench_g3/manifest.json
status: ok
trajectory_stream.total_rows: 282839
retrieval_stream.total_pairs: 448665
skill_pool.dedup_method: partial_rule_name_param_v2
skill_pool.raw_skill_count: 38875
skill_pool.canonical_skill_count: 37623
skill_pool.merged_group_count: 785
skill_pool.changed_alias_count: 1252
skill_pool.stream_remap.trajectory_changed_skill_refs: 1351
skill_pool.stream_remap.retrieval_changed_skill_refs: 27836
source_inventory.missing_required_target_sources: []
leakage_audit.status: ok
```

这说明 v4 的 first-pass rule dedup 已经落地。缺口是 D15-v2 的 second-pass
borderline review：相似名称、参数部分重叠但未达到自动合并条件的技能需要被列出，
供人工或 LLM judge 审核。为避免污染当前训练，本次只新增候选审计，不自动改写训练数据。

### 已完成的主要改动

1. `scripts/build_clstr_unified_pretrain.py`
   - 新增 `skill_dedup_borderline_candidates.jsonl` 输出；
   - 规则：
     - fuzzy name similarity in `[0.6, 0.95)`；
     - 或 input param name 有部分重叠但不完全相同；
   - 已经属于同一 canonical group 的 alias 不再列为候选；
   - 候选记录包含：
     - `left_skill_id` / `right_skill_id`
     - `left_name` / `right_name`
     - `name_similarity`
     - `param_overlap`
     - `reasons`
     - `review_status: pending_manual_or_llm`
   - manifest 新增：
     - `files.skill_dedup_borderline_candidates`
     - `skill_pool.borderline_review`

2. `scripts/audit_skill_dedup_borderline.py`
   - 新增独立审计 CLI；
   - 只读取已有 `skill_pool.jsonl`，输出候选 JSONL 与 report；
   - 不调用 `build_clstr_unified_pretrain.py` 重建数据；
   - 不自动合并候选 pair。

3. `scripts/sbatch/run_skill_dedup_borderline_audit.sh`
   - 新增 sbatch 入口，默认读取：

```text
data/clstr_unified_pretrain_v2_toolbench_g3/skill_pool.jsonl
```

   - 默认输出：

```text
outputs/clstr_unified_readiness_audit/skill_dedup_borderline_candidates.jsonl
outputs/clstr_unified_readiness_audit/skill_dedup_borderline_report.json
```

   - 保持 cache / HF mirror 均在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 下；
   - 不启动训练，不重建 unified dataset。

### 验证证据

已按 TDD 先写失败测试，初始失败包括：

```text
FileNotFoundError: skill_dedup_borderline_candidates.jsonl
can't open file 'scripts/audit_skill_dedup_borderline.py': [Errno 2] No such file or directory
FileNotFoundError: scripts/sbatch/run_skill_dedup_borderline_audit.sh
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py -q
# 13 passed
```

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py::test_skill_dedup_borderline_audit_cli_generates_candidates_for_existing_skill_pool \
  tests/test_sbatch_scripts.py::test_skill_dedup_borderline_audit_sbatch_uses_unified_skill_pool_without_rebuilding_data -q
# 2 passed
```

```bash
bash -n scripts/sbatch/run_skill_dedup_borderline_audit.sh
# exit 0
```

```bash
conda run -n reasoning_trap env PYTHONPATH=. python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  scripts/audit_skill_dedup_borderline.py
# exit 0
```

```bash
git diff --check -- scripts/build_clstr_unified_pretrain.py \
  scripts/audit_skill_dedup_borderline.py \
  scripts/sbatch/run_skill_dedup_borderline_audit.sh \
  tests/test_unified_pretrain_v2.py \
  tests/test_sbatch_scripts.py
# exit 0
```

### 已提交作业

真实 full skill pool 的 borderline review audit 已通过 sbatch 提交：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_skill_dedup_borderline_audit.sh
# Submitted batch job 77347
```

该作业完成后应检查：

```text
outputs/clstr_unified_readiness_audit/skill_dedup_borderline_report.json
outputs/clstr_unified_readiness_audit/skill_dedup_borderline_candidates.jsonl
```

注意：这个 audit 只生成候选，不改变 Stage 1 正在使用的
`data/clstr_unified_pretrain_v2_toolbench_g3`，因此不会影响当前训练作业。

## 2026-05-26 v4 readiness checkpoint path alignment

本记录修复一个会影响后续 Stage gate 判断的路径一致性问题。当前正式训练链路已经切到：

```text
data/clstr_unified_pretrain_v2_toolbench_g3
outputs/clstr_unified_stage1_toolbench_g3_retrieval_warmup
outputs/clstr_unified_stage2_toolbench_g3_full_base
```

但仍有两个入口保留了旧的 `query_tool_fix` / `train_safe` 默认 checkpoint 路径：

- `scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh`
- `scripts/audit_clstr_unified_training_readiness.py`

这会导致 Stage 1 训练即使在当前 `toolbench_g3` output dir 完成，readiness gate 也可能继续误报
`missing_stage1_checkpoint`。

### 已完成的主要改动

1. `scripts/audit_clstr_unified_training_readiness.py`
   - `--stage1_checkpoint` 默认值改为：

```text
outputs/clstr_unified_stage1_toolbench_g3_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt
```

   - `--stage2_checkpoint` 默认值改为：

```text
outputs/clstr_unified_stage2_toolbench_g3_full_base/checkpoints/clstr_full_base-step10000.pt
```

2. `scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh`
   - 同步改为上述 `toolbench_g3` Stage 1/2 checkpoint 默认路径；
   - 回归测试明确禁止该 pipeline 再出现 `query_tool_fix` 默认路径。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
assert 'outputs/clstr_unified_stage1_toolbench_g3_retrieval_warmup/...' in script
assert 'outputs/clstr_unified_stage1_toolbench_g3_retrieval_warmup/...' in audit CLI source
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py::test_toolbench_g3_unified_pipeline_sbatch_runs_import_build_and_readiness_gates \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_cli_defaults_use_toolbench_g3_stage_paths \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_sbatch_entrypoint_is_audit_only_and_local -q
# 3 passed
```

```bash
bash -n scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh
# exit 0
```

```bash
conda run -n reasoning_trap env PYTHONPATH=. python -m py_compile \
  scripts/audit_clstr_unified_training_readiness.py
# exit 0
```

```bash
git diff --check -- scripts/audit_clstr_unified_training_readiness.py \
  scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh \
  tests/test_sbatch_scripts.py \
  tests/test_unified_training_readiness.py
# exit 0
```

## 2026-05-26 v4 StableToolBench same-node raw generation entrypoint

前一阶段已经把 StableToolBench virtual API 的 `tools_folder` 修到本地 G3 toolenv，并确认：

```text
stabletoolbench_virtual_api_readiness.status: ok
stabletoolbench_raw_generation_readiness.status: action_required
stabletoolbench_raw_generation_readiness.blockers: [missing_openai_key_for_chatgpt_function]
```

但原始 raw generation 仍存在一个运行编排风险：`SERVICE_URL=http://127.0.0.1:12001/virtual`
只有在 virtual API server 和 raw generation executor 位于同一个 sbatch 作业/同一节点时才成立。
如果分两个作业运行，`127.0.0.1` 会指向各自节点，raw generation 可能连不上 server。

### 已完成的主要改动

1. 新增 `scripts/sbatch/run_stabletoolbench_local_raw_generation.sh`
   - 在同一个 sbatch 作业内：
     1. audit virtual API readiness；
     2. audit raw generation readiness；
     3. 显式 `RUN_GENERATION=1` 时后台启动 StableToolBench virtual API；
     4. 等待 `http://127.0.0.1:${VIRTUAL_API_PORT}/docs` 可访问；
     5. 运行 official `toolbench/inference/qa_pipeline_multithread.py`；
     6. `trap cleanup EXIT` 清理后台 server。
   - 默认 `RUN_GENERATION=0`，只做 readiness audit，不启动 server / raw generation。
   - 默认：

```text
MODE=mirrorapi
VIRTUAL_API_PORT=12001
SERVICE_URL=http://127.0.0.1:12001/virtual
TOOL_ROOT_DIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolenv/tools
RAW_ANSWER_PATH=outputs/toolbench_g3/stabletoolbench_raw_answers
CANDIDATE_MODEL=clstr_toolbench_g3
TEST_SET=G3_instruction
METHOD=CLSTR@1
```

2. 新增测试
   - `tests/test_sbatch_scripts.py::test_stabletoolbench_local_raw_generation_sbatch_runs_server_and_generation_on_same_node`
   - 覆盖：
     - 默认 `RUN_GENERATION=0`；
     - `SERVICE_URL` 默认使用同节点 `127.0.0.1:${VIRTUAL_API_PORT}`；
     - 有 `VIRTUAL_API_PID` 和 `trap cleanup EXIT`；
     - 同时包含 virtual API audit、raw generation audit、server start、official
       `qa_pipeline_multithread.py`。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
FileNotFoundError: scripts/sbatch/run_stabletoolbench_local_raw_generation.sh
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py::test_stabletoolbench_local_raw_generation_sbatch_runs_server_and_generation_on_same_node -q
# 1 passed
```

```bash
bash -n scripts/sbatch/run_stabletoolbench_local_raw_generation.sh
# exit 0
```

```bash
git diff --check -- scripts/sbatch/run_stabletoolbench_local_raw_generation.sh \
  tests/test_sbatch_scripts.py
# exit 0
```

注意：该入口仍不会绕过 `OPENAI_KEY` 要求。要真正生成 official raw answers，必须用
`RUN_GENERATION=1` 且提供 downstream executor 的 `OPENAI_KEY`。当前没有 key 时只应运行
readiness audit。

## 2026-05-26 v4 StableToolBench official SoPR readiness gate checkpoint

本次补齐 StableToolBench 官方 Solvable Pass Rate 前置审计。结论是：下载侧已经不是当前瓶颈；
ToolBench-G3、StableToolBench、TRAJECT-Bench、ToolRet eval/training 均已在
`/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 下落盘，但 StableToolBench 官方 SoPR 还不能直接跑。

### 当前数据准备状态

- ToolBench-G3 HF 下载 manifest:
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/hf_g3_download_manifest.json`
  - `status: ok`
  - `failed_download_count: 0`
  - 因 `hf-mirror.com` 高并发会 429，当前稳定参数是 `workers=2`,
    `retry_backoff_seconds=20.0`, `retry_backoff_multiplier=1.5`。
- Normalized ToolBench-G3:
  `data/toolbench_g3/manifest.json`
  - `status: ok`
  - `answer_files: 5000`
  - `retrieval_pairs: 74555`
  - `trajectory_rows: 9570`
  - `skills: 1600`
- ToolBench-G3 static routing eval:
  `data/toolbench_g3_routing_eval/manifest.json`
  - `status: ok`
  - `query_count: 25709`
  - 该指标只用于 normalized ToolBench-G3 relevant API routing/retrieval recall，
    不能当作 StableToolBench pass-rate。
- ToolRet eval:
  `data/toolret_eval/manifest.json`
  - `status: ok`
  - `query_count: 7961`
  - `skill_count: 44453`
  - `qrel_count: 14106`
- TRAJECT eval:
  `data/traject_eval/manifest.json`
  - `status: ok`
  - `query_count: 38094`
  - `skill_count: 689`
  - `qrel_count: 38094`
- Unified CLSTR v2 + ToolBench-G3:
  `data/clstr_unified_pretrain_v2_toolbench_g3/manifest.json`
  - `status: ok`
  - `trajectory_stream.total_rows: 282839`
  - `retrieval_stream.total_pairs: 448665`
  - `skill_pool.total_skills: 37623`
  - `leakage_audit.status: ok`
- Unified readiness:
  `outputs/clstr_unified_readiness_audit/toolbench_g3_readiness_report.json`
  - data side says `paper_readiness.full_three_benchmark_data_ready: true`
  - `toolret_eval_ready: true`
  - `traject_eval_ready: true`
  - `toolbench_g3_data_ready: true`
  - 当前 `status: action_required` 主要是因为新的 Stage 1/2 checkpoints 还在作业队列中，
    不是因为三数据集缺失。

### StableToolBench official SoPR 当前 blocker

新增审计输出：

`outputs/toolbench_g3/stabletoolbench_pass_rate_readiness.json`

当前结果：

```text
status: action_required
can_run_pass_rate: false
blockers:
  - missing_converted_answer_file
  - api_pool_has_no_valid_api_key
metric_scope: StableToolBench Solvable Pass Rate; not static routing recall.
```

具体含义：

- StableToolBench repo 已下载：
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench`
  commit `aa4ed9f4737ad98bd706663f01d63623c3427812`。
- 官方 `eval_pass_rate.py` 存在：
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolbench/tooleval/eval_pass_rate.py`
- 官方 `convert_to_answer_format.py` 存在：
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolbench/tooleval/convert_to_answer_format.py`
- G3 solvable test ids 存在：
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/solvable_queries/test_query_ids/G3_instruction.json`
- 但 CLSTR 尚未产生 StableToolBench converted answer:
  `outputs/toolbench_g3/stabletoolbench_converted/clstr_toolbench_g3/G3_instruction.json`
- `StableToolBench/openai_key.json` 目前没有有效 evaluator key，`valid_api_key_count: 0`。

因此：当前可以跑 CLSTR training 和 static routing/retrieval eval；不能把 ToolBench-G3 routing recall
报告成 StableToolBench SoPR。官方 SoPR 需要先完成 CLSTR/下游 executor 对 G3 solvable queries 的
answer generation，再转换为 StableToolBench answer format，并提供 evaluator key。

### 已完成的主要改动

1. `clstr/stabletoolbench_pass_rate.py`
   - 新增 `audit_stabletoolbench_pass_rate()`；
   - 检查 StableToolBench repo、官方 eval/convert 脚本、G3 test ids、
     converted answer file、evaluator config、API pool key；
   - 输出 `status: ok` 或 `status: action_required`；
   - 明确 `metric_scope` 为 StableToolBench Solvable Pass Rate，不是 static routing recall；
   - `reproducible_command` 使用 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr`
     逻辑路径，避免 `cd toolbench/tooleval` 后相对路径失效。

2. `scripts/audit_stabletoolbench_pass_rate.py`
   - 新增 CLI，可写 readiness report：

```bash
python scripts/audit_stabletoolbench_pass_rate.py \
  --stabletoolbench_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench \
  --converted_answer_path outputs/toolbench_g3/stabletoolbench_converted \
  --api_pool_file /data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/openai_key.json \
  --candidate_model clstr_toolbench_g3 \
  --test_set G3_instruction \
  --save_path outputs/toolbench_g3/stabletoolbench_pass_rate \
  --command_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr \
  --output_path outputs/toolbench_g3/stabletoolbench_pass_rate_readiness.json
```

3. `scripts/sbatch/run_stabletoolbench_pass_rate.sh`
   - 新增官方 SoPR sbatch wrapper；
   - 默认先跑 readiness audit，并带 `--fail_on_action_required`；
   - readiness 不为 `ok` 时不会进入官方 `eval_pass_rate.py`；
   - 默认路径均在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 下；
   - 默认 `MAX_EVAL_THREADS=1`，避免 evaluator API 并发失控。

4. 测试
   - `tests/test_stabletoolbench_pass_rate.py` 覆盖：
     - 缺 converted answer 和有效 key 时返回 `action_required`；
     - 输入齐全时返回 `ok`；
     - CLI 写报告并在 `--fail_on_action_required` 下返回 2；
     - 可复现命令使用绝对/逻辑 CLSTR 路径。
   - `tests/test_sbatch_scripts.py` 覆盖 StableToolBench pass-rate sbatch wrapper。

### 验证证据

已按 TDD 先写失败测试，初始失败包括：

```text
ModuleNotFoundError: No module named 'clstr.stabletoolbench_pass_rate'
```

以及新增 `command_root` 行为时的预期失败：

```text
TypeError: audit_stabletoolbench_pass_rate() got an unexpected keyword argument 'command_root'
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stabletoolbench_pass_rate.py \
  tests/test_sbatch_scripts.py::test_stabletoolbench_pass_rate_sbatch_audits_before_official_sopr -q
# 6 passed
```

```bash
bash -n scripts/sbatch/run_stabletoolbench_pass_rate.sh
python -m py_compile clstr/stabletoolbench_pass_rate.py scripts/audit_stabletoolbench_pass_rate.py
git diff --check -- \
  clstr/stabletoolbench_pass_rate.py \
  scripts/audit_stabletoolbench_pass_rate.py \
  scripts/sbatch/run_stabletoolbench_pass_rate.sh \
  tests/test_stabletoolbench_pass_rate.py \
  tests/test_sbatch_scripts.py
# exit 0
```

## 2026-05-26 v4 StableToolBench answer conversion gate checkpoint

在 official SoPR readiness gate 之后，本次继续补齐其上游环节：StableToolBench raw executor
answers 到官方 converted answer JSON 的转换门禁。该环节只负责调用 StableToolBench 官方
`convert_to_answer_format.py`，不会伪造模型执行结果；如果 CLSTR/Qwen 尚未生成 raw answer
文件，会返回 `action_required` 并阻止进入转换。

### 当前真实 readiness 结果

新增审计输出：

`outputs/toolbench_g3/stabletoolbench_answer_conversion_readiness.json`

当前结果：

```text
status: action_required
can_convert_answers: false
blockers:
  - missing_raw_answer_dir
  - no_raw_answer_files_matching_method
raw_answer_file_count: 0
matching_raw_answer_file_count: 0
metric_scope: StableToolBench answer format conversion; prerequisite for SoPR, not pass-rate.
```

具体含义：

- 官方转换脚本存在：
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolbench/tooleval/convert_to_answer_format.py`
- 但当前没有 CLSTR 的 raw executor answer 目录：
  `outputs/toolbench_g3/stabletoolbench_raw_answers/clstr_toolbench_g3/G3_instruction`
- 因此无法生成：
  `outputs/toolbench_g3/stabletoolbench_converted/clstr_toolbench_g3/G3_instruction.json`
- 这也解释了上一个 SoPR readiness gate 的 `missing_converted_answer_file` blocker。

本阶段结论：StableToolBench official SoPR 的链路应当是：

```text
CLSTR/Qwen executor raw answers
  -> scripts/sbatch/run_convert_stabletoolbench_answers.sh
  -> outputs/toolbench_g3/stabletoolbench_converted/clstr_toolbench_g3/G3_instruction.json
  -> scripts/sbatch/run_stabletoolbench_pass_rate.sh
  -> official StableToolBench SoPR
```

当前卡点在第一步：还没有 `G3_instruction` raw executor answers；不是 conversion 脚本缺失。

### 已完成的主要改动

1. `clstr/stabletoolbench_answer_conversion.py`
   - 新增 `audit_stabletoolbench_answer_conversion()`；
   - 检查 StableToolBench repo、官方 `convert_to_answer_format.py`、raw answer dir、
     method 匹配的 raw answer JSON；
   - 输出 `status: ok` 或 `status: action_required`；
   - 明确 `metric_scope` 为 answer format conversion，不是 pass-rate；
   - 生成可复现命令：

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolbench/tooleval && \
python convert_to_answer_format.py \
  --answer_dir /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/outputs/toolbench_g3/stabletoolbench_raw_answers/clstr_toolbench_g3/G3_instruction \
  --method CLSTR@1 \
  --output /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/outputs/toolbench_g3/stabletoolbench_converted/clstr_toolbench_g3/G3_instruction.json
```

2. `scripts/convert_stabletoolbench_answers.py`
   - 新增 CLI；
   - 默认只 audit；
   - 加 `--run_convert` 时才调用官方转换脚本；
   - 加 `--fail_on_action_required` 时 readiness 不为 `ok` 返回 2。

3. `scripts/sbatch/run_convert_stabletoolbench_answers.sh`
   - 新增转换 sbatch wrapper；
   - 默认路径均在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 下；
   - 默认 `RAW_ANSWER_PATH=outputs/toolbench_g3/stabletoolbench_raw_answers`；
   - 默认 `CONVERTED_ANSWER_PATH=outputs/toolbench_g3/stabletoolbench_converted`；
   - 默认 `CANDIDATE_MODEL=clstr_toolbench_g3`、`TEST_SET=G3_instruction`、`METHOD=CLSTR@1`；
   - 先审计，缺 raw answer 时不会写空 converted 文件。

4. 测试
   - `tests/test_stabletoolbench_answer_conversion.py` 覆盖：
     - raw answer 目录缺失时返回 `action_required`；
     - raw answer 文件存在且文件名包含 method 时返回 `ok`；
     - CLI 写 readiness report 并在 `--fail_on_action_required` 下返回 2。
   - `tests/test_sbatch_scripts.py` 覆盖转换 sbatch wrapper 的默认路径和门禁参数。

### 验证证据

已按 TDD 先写失败测试，初始失败：

```text
ModuleNotFoundError: No module named 'clstr.stabletoolbench_answer_conversion'
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stabletoolbench_answer_conversion.py \
  tests/test_sbatch_scripts.py::test_stabletoolbench_answer_conversion_sbatch_audits_raw_answers_before_convert -q
# 4 passed
```

```bash
bash -n scripts/sbatch/run_convert_stabletoolbench_answers.sh
python -m py_compile clstr/stabletoolbench_answer_conversion.py scripts/convert_stabletoolbench_answers.py
git diff --check -- \
  clstr/stabletoolbench_answer_conversion.py \
  scripts/convert_stabletoolbench_answers.py \
  scripts/sbatch/run_convert_stabletoolbench_answers.sh \
  tests/test_stabletoolbench_answer_conversion.py \
  tests/test_sbatch_scripts.py
# exit 0
```

## 2026-05-26 v4 StableToolBench raw answer generation gate checkpoint

在 answer conversion gate 之后，本次继续补齐更上游的 official StableToolBench raw answer
generation readiness gate。该 gate 不运行模型、不连接虚拟 API，只审计 official inference
脚本在 sbatch 里真正启动前的硬前置条件，避免把缺 toolenv / service / key 的状态误判成
可以跑官方 SoPR。

### 当前真实 readiness 结果

新增审计输出：

`outputs/toolbench_g3/stabletoolbench_raw_generation_readiness.json`

当前结果：

```text
status: action_required
can_generate_raw_answers: false
blockers:
  - missing_tool_root_dir
  - missing_service_url_for_stabletoolbench_virtual_api
  - missing_openai_key_for_chatgpt_function
tool_json_file_count: 0
metric_scope: StableToolBench raw answer generation; prerequisite for answer conversion and SoPR.
```

具体含义：

- 官方 raw answer 生成入口存在：
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolbench/inference/qa_pipeline_multithread.py`
- G3 solvable query 文件存在：
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/solvable_queries/test_instruction/G3_instruction.json`
- 但 official toolenv 目录不存在：
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolenv/tools`
- 当前没有设置 StableToolBench virtual API `SERVICE_URL`；
- 当前没有可用于 `chatgpt_function` downstream executor 的 `OPENAI_KEY`。

注意：审计显式拒绝 `data_example/toolenv/tools` 这类示例工具目录，blocker 名称为
`tool_root_dir_is_data_example_not_official`，避免把 smoke resource 当成 ToolBench-G3
主表环境。

### 当前 official StableToolBench SoPR 链路状态

```text
1. raw answer generation:
   outputs/toolbench_g3/stabletoolbench_raw_answers/clstr_toolbench_g3/G3_instruction
   status: action_required
   blockers: missing toolenv/tools, SERVICE_URL, OPENAI_KEY

2. answer conversion:
   outputs/toolbench_g3/stabletoolbench_converted/clstr_toolbench_g3/G3_instruction.json
   status: action_required
   blockers: missing raw answer dir, no method-matching raw JSON

3. official SoPR:
   outputs/toolbench_g3/stabletoolbench_pass_rate/clstr_toolbench_g3
   status: action_required
   blockers: missing converted answer file, no valid evaluator key
```

因此当前 ToolBench-G3 可用于 static routing/retrieval eval，但还不能报告 official
StableToolBench Solvable Pass Rate。

### 已完成的主要改动

1. `clstr/stabletoolbench_raw_generation.py`
   - 新增 `audit_stabletoolbench_raw_generation()`；
   - 检查 StableToolBench repo、official `qa_pipeline_multithread.py`、G3 solvable query、
     official toolenv/tools、virtual API `SERVICE_URL`、downstream model key/path；
   - 对 `chatgpt_function` 检查 `OPENAI_KEY`；
   - 对 `toolllama*` 检查 `MODEL_PATH`；
   - 输出 `status: ok` 或 `status: action_required`；
   - `reproducible_command` 中用 `${OPENAI_KEY}` / `${TOOLBENCH_KEY}` 占位，不泄露 key。

2. `scripts/audit_stabletoolbench_raw_generation.py`
   - 新增 CLI；
   - 默认从环境读取 `OPENAI_KEY`、`SERVICE_URL`、`TOOLBENCH_KEY`；
   - 支持 `--fail_on_action_required`，readiness 不为 `ok` 返回 2。

3. `scripts/sbatch/run_stabletoolbench_raw_generation.sh`
   - 新增 sbatch wrapper；
   - 默认 `RUN_GENERATION=0`，只做 readiness audit，不启动推理；
   - 只有显式设置 `RUN_GENERATION=1` 且 audit 通过后，才调用 official
     `toolbench/inference/qa_pipeline_multithread.py`；
   - 默认输出 raw answers 到：
     `outputs/toolbench_g3/stabletoolbench_raw_answers/clstr_toolbench_g3/G3_instruction`
   - 默认路径均限制在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 下。

4. 测试
   - `tests/test_stabletoolbench_raw_generation.py` 覆盖：
     - 缺 official toolenv / key 时返回 `action_required`；
     - toolenv + key + service_url 齐全时返回 `ok`；
     - CLI 写 readiness report 并在 `--fail_on_action_required` 下返回 2；
     - 拒绝 `data_example/toolenv/tools`。
   - `tests/test_sbatch_scripts.py` 覆盖 raw generation sbatch wrapper 的默认路径、
     `RUN_GENERATION=0` 安全默认值和关键环境变量。

### 验证证据

已按 TDD 先写失败测试，初始失败：

```text
ModuleNotFoundError: No module named 'clstr.stabletoolbench_raw_generation'
```

针对拒绝 `data_example/toolenv/tools` 的回归测试初始失败：

```text
AssertionError: assert 'ok' == 'action_required'
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stabletoolbench_raw_generation.py \
  tests/test_sbatch_scripts.py::test_stabletoolbench_raw_generation_sbatch_audits_before_generation -q
# 5 passed
```

```bash
bash -n scripts/sbatch/run_stabletoolbench_raw_generation.sh
python -m py_compile clstr/stabletoolbench_raw_generation.py scripts/audit_stabletoolbench_raw_generation.py
git diff --check -- \
  clstr/stabletoolbench_raw_generation.py \
  scripts/audit_stabletoolbench_raw_generation.py \
  scripts/sbatch/run_stabletoolbench_raw_generation.sh \
  tests/test_stabletoolbench_raw_generation.py \
  tests/test_sbatch_scripts.py
# exit 0
```

## 2026-05-26 v4 StableToolBench G3 toolenv subset checkpoint

为消除 raw answer generation 链路最前面的本地数据 blocker，本次新增从 official
StableToolBench solvable queries 构建 G3 toolenv 子集的脚本。该脚本只使用：

`/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/solvable_queries/test_instruction/G3_instruction.json`

不会使用 `solvable_queries_example` 或 `data_example/toolenv`。这点很重要，因为
`data_example` 只能用于 smoke，不能进入 ToolBench-G3 主表链路。

### 当前真实构建结果

输出目录：

`/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolenv/tools`

manifest：

`/data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolenv/toolenv_G3_instruction_manifest.json`

当前结果：

```text
status: ok
query_count: 61
api_ref_count: 352
tool_count: 7
api_count: 44
written_tool_files: 7
```

已写入的工具文件包括：

```text
StableToolBench/toolenv/tools/Advertising/url_link_shortener.json
StableToolBench/toolenv/tools/Entertainment/watchmode.json
StableToolBench/toolenv/tools/Media/vimeo.json
StableToolBench/toolenv/tools/Movies/ott_details.json
StableToolBench/toolenv/tools/Movies/streaming_availability.json
StableToolBench/toolenv/tools/Tools/bitly.json
StableToolBench/toolenv/tools/Tools/ytstream_download_youtube_videos.json
```

刷新 raw generation readiness 后，`missing_tool_root_dir` 已消失。当前 raw generation 剩余 blocker：

```text
status: action_required
can_generate_raw_answers: false
blockers:
  - missing_service_url_for_stabletoolbench_virtual_api
  - missing_openai_key_for_chatgpt_function
tool_json_file_count: 7
```

因此：StableToolBench G3 的本地 toolenv 子集已准备好；official raw answer generation
下一步需要 virtual API `SERVICE_URL` 和下游 executor key/model。

### 已完成的主要改动

1. `clstr/stabletoolbench_toolenv.py`
   - 新增 `build_stabletoolbench_toolenv_from_queries()`；
   - 从 official `solvable_queries/test_instruction/<test_set>.json` 读取 `api_list`；
   - 按 StableToolBench/ToolBench inference 约定写入：
     `toolenv/tools/<category>/<standardized_tool_name>.json`；
   - 对同一 tool 的重复 API 按 standardized API name 去重；
   - 拒绝 `solvable_queries_example` 和 `data_example` 路径。

2. `scripts/build_stabletoolbench_toolenv.py`
   - 新增 CLI：

```bash
python scripts/build_stabletoolbench_toolenv.py \
  --query_file /data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/solvable_queries/test_instruction/G3_instruction.json \
  --output_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolenv/tools \
  --manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/toolenv/toolenv_G3_instruction_manifest.json \
  --fail_on_action_required
```

3. `scripts/sbatch/run_build_stabletoolbench_toolenv.sh`
   - 新增 sbatch wrapper；
   - 默认使用 official `StableToolBench/solvable_queries/test_instruction/G3_instruction.json`；
   - 默认输出到 official `StableToolBench/toolenv/tools`；
   - 显式不使用 `data_example`。

4. 测试
   - `tests/test_stabletoolbench_toolenv.py` 覆盖：
     - 从 official query schema 构建 tool JSON；
     - 去重重复 API；
     - 拒绝 example query source；
     - CLI 写 manifest。
   - `tests/test_sbatch_scripts.py` 覆盖 toolenv 构建 sbatch wrapper。

### 验证证据

已按 TDD 先写失败测试，初始失败：

```text
ModuleNotFoundError: No module named 'clstr.stabletoolbench_toolenv'
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stabletoolbench_toolenv.py \
  tests/test_sbatch_scripts.py::test_stabletoolbench_toolenv_build_sbatch_uses_official_solvable_queries -q
# 4 passed
```

```bash
bash -n scripts/sbatch/run_build_stabletoolbench_toolenv.sh
python -m py_compile clstr/stabletoolbench_toolenv.py scripts/build_stabletoolbench_toolenv.py
git diff --check -- \
  clstr/stabletoolbench_toolenv.py \
  scripts/build_stabletoolbench_toolenv.py \
  scripts/sbatch/run_build_stabletoolbench_toolenv.sh \
  tests/test_stabletoolbench_toolenv.py \
  tests/test_sbatch_scripts.py
# exit 0
```

## 2026-05-26 v4 ToolBench-G3 official download completion checkpoint

本记录对应 v4 主表 #2 ToolBench-G3 官方主表数据准备。此前 HF mirror 下载在
`workers=16` 时触发 `429 Too Many Requests`，导致完整下载 manifest 为
`download_failed`，本地仅有部分 answer 文件。经排查，脚本本身和 HF mirror 可用，
根因是并发过高触发限流。

### 已完成的主要改动

1. `scripts/download_toolbench_g3_from_hf.py`
   - 新增 `--retry_backoff_seconds`；
   - 新增 `--retry_backoff_multiplier`；
   - `_download_file_with_retries()` 在失败后按指数退避等待再重试；
   - manifest 记录退避参数，便于复盘下载设置。

2. `tests/test_toolbench_g3_import.py`
   - 新增回归 `test_download_toolbench_g3_from_hf_backs_off_between_retry_attempts`；
   - 覆盖 transient failure 下的等待序列和 manifest 参数记录。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
AttributeError: module 'download_toolbench_g3_from_hf' has no attribute 'time'
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_backs_off_between_retry_attempts \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_retries_transient_file_failures \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_records_persistent_file_failures_without_crashing \
  -q
# 3 passed
```

```bash
python -m py_compile scripts/download_toolbench_g3_from_hf.py
# exit 0
```

```bash
git diff --check -- scripts/download_toolbench_g3_from_hf.py tests/test_toolbench_g3_import.py
# no output
```

### 完整下载结果

最终使用低并发 + 长退避续传：

```bash
python scripts/download_toolbench_g3_from_hf.py \
  --output_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data \
  --api_manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/adorg_toolbench_api.json \
  --manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/hf_g3_download_manifest.json \
  --workers 2 \
  --max_retries 8 \
  --retry_backoff_seconds 20 \
  --retry_backoff_multiplier 1.5 \
  --timeout 180
```

结果：

```text
G3_answer_files=5000
part_files=0
manifest.status=ok
selected_answer_file_count=5000
verification.status=ok
verification.g3_answer_files=5000
failed_download_count=0
skipped_existing_file_count=3840
downloaded_file_count=5001
```

### 后处理 pipeline

完整下载后已提交 ToolBench-G3 后处理 pipeline：

```bash
cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh
# Submitted batch job 77248
```

该 pipeline 会执行：

1. raw verify；
2. normalized ToolBench-G3 import；
3. normalized audit；
4. 重建 `data/clstr_unified_pretrain_v2_toolbench_g3`；
5. unified readiness audit。

后续必须检查：

- `data/toolbench_g3/manifest.json`
- `outputs/toolbench_g3/toolbench_g3_audit.json`
- `data/clstr_unified_pretrain_v2_toolbench_g3/manifest.json`
- `outputs/clstr_unified_readiness_audit/toolbench_g3_readiness_report.json`

## 2026-05-26 v4 ToolBench-G3 normalized import + unified v2 readiness checkpoint

后处理作业 `77248` 已完成，预期产物均已生成：

```text
data/toolbench_g3/manifest.json
outputs/toolbench_g3/toolbench_g3_audit.json
data/clstr_unified_pretrain_v2_toolbench_g3/manifest.json
outputs/clstr_unified_readiness_audit/toolbench_g3_readiness_report.json
```

### ToolBench-G3 normalized 结果

```text
manifest.status=ok
queries=25709
answer_files=5000
retrieval_pairs=74555
trajectory_rows=9570
skills=1600
skipped_unresolved_answer_files=187
unresolved_action_count=271
```

审计结果：

```text
outputs/toolbench_g3/toolbench_g3_audit.json status=ok
blockers=[]
```

### unified v2 + ToolBench-G3 结果

```text
data/clstr_unified_pretrain_v2_toolbench_g3/trajectories.jsonl 282839
data/clstr_unified_pretrain_v2_toolbench_g3/retrieval.jsonl     448665
data/clstr_unified_pretrain_v2_toolbench_g3/skill_pool.jsonl     37623
```

readiness audit 中三大主表数据已 ready：

```text
paper_readiness.full_three_benchmark_data_ready=true
paper_readiness.toolret_eval_ready=true
paper_readiness.traject_eval_ready=true
paper_readiness.toolbench_g3_data_ready=true
paper_readiness.missing_required_target_sources=[]
benchmark_eval_readiness.toolbench_g3.may_use_for_paper_main_table=true
```

当前 `status=action_required` 的原因不是数据缺失，而是该新 unified dataset 对应的训练
checkpoint 还没产出：

```text
stage1_retrieval_warmup.ready_to_submit=true
stage2_full_base.blockers=["missing_stage1_checkpoint"]
stage3_offline_hrpo.blockers=["missing_stage1_checkpoint", "missing_stage2_checkpoint"]
stage4_act.blockers=["missing_stage1_checkpoint", "missing_stage2_or_stage3_checkpoint"]
```

因此下一步应把 Stage 1/2/3/4 训练脚本的默认数据路径切到
`data/clstr_unified_pretrain_v2_toolbench_g3`，产物也使用新的 output dir，先提交 Stage 1
retrieval warmup，Stage 2/3/4 按 checkpoint 依赖顺序推进。

## 2026-05-26 v4 three-benchmark training path switch checkpoint

为避免后续误用旧的 `data/clstr_unified_pretrain_v2` 或 query-tool-fix 中间数据，本次把
v4 主训练链路默认切到完整 ToolBench-G3 后的 three-benchmark unified 数据集：

```text
data/clstr_unified_pretrain_v2_toolbench_g3
```

### 已完成的主要改动

1. Stage 1 retrieval warmup
   - 默认 `DATA_ROOT=data/clstr_unified_pretrain_v2_toolbench_g3`
   - 默认 `OUTPUT_DIR=outputs/clstr_unified_stage1_toolbench_g3_retrieval_warmup`

2. Stage 2 full base
   - 默认 `TRAIN_PATH=data/clstr_unified_pretrain_v2_toolbench_g3/trajectories.jsonl`
   - 默认 `SKILLS_PATH=data/clstr_unified_pretrain_v2_toolbench_g3/skill_pool.jsonl`
   - 默认 `OUTPUT_DIR=outputs/clstr_unified_stage2_toolbench_g3_full_base`
   - 默认读取 Stage 1:
     `outputs/clstr_unified_stage1_toolbench_g3_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt`

3. Stage 3 HRPO / Stage 4 ACT
   - 默认使用同一个 ToolBench-G3 unified 轨迹与 skill pool；
   - 默认 Stage 1/2 checkpoint 均指向新的 `toolbench_g3` output dir。

4. Readiness / ToolRet export / TRAJECT export
   - 默认 checkpoint 也同步到新的 Stage 1 output dir；
   - readiness 默认输出：
     `outputs/clstr_unified_readiness_audit/readiness_report_toolbench_g3.json`。

### 验证证据

先更新测试断言，确认脚本仍指向旧路径时出现预期失败：

```text
4 failed: Stage 1/2/3/4 sbatch scripts still referenced old unified paths
3 failed: readiness/export scripts still referenced old checkpoint paths
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest tests/test_sbatch_scripts.py -q
# 18 passed
```

```bash
bash -n scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh
bash -n scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
bash -n scripts/sbatch/run_clstr_unified_stage3_hrpo.sh
bash -n scripts/sbatch/run_clstr_unified_stage4_act_train.sh
bash -n scripts/sbatch/run_clstr_unified_readiness_audit.sh
bash -n scripts/sbatch/run_export_clstr_retrieval_run.sh
bash -n scripts/sbatch/run_export_traject_retrieval_run.sh
# exit 0
```

```bash
python -m py_compile scripts/download_toolbench_g3_from_hf.py
# exit 0
```

```bash
git diff --check -- \
  scripts/download_toolbench_g3_from_hf.py \
  tests/test_toolbench_g3_import.py \
  tests/test_sbatch_scripts.py \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  scripts/sbatch/run_export_clstr_retrieval_run.sh \
  scripts/sbatch/run_export_traject_retrieval_run.sh \
  description2.md
# no output
```

下一步提交 Stage 1 retrieval warmup。Stage 2/3/4 必须等待新 Stage 1/2 checkpoint 生成后再依赖推进。

## 2026-05-26 v4 three-benchmark Stage 1-4 sbatch submission checkpoint

已在 `gpu_a800` 队列提交新的 three-benchmark CLSTR 训练链路，输入数据统一为：

```text
data/clstr_unified_pretrain_v2_toolbench_g3
```

提交命令与 job id：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh
# stage1_job=77275

sbatch --dependency=afterok:77275 --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# stage2_job=77276

sbatch --dependency=afterok:77276 --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage3_hrpo.sh
# stage3_job=77277

sbatch --dependency=afterok:77277 --gpus=1 -p gpu_a800 \
  --export=ALL,HEAD_CHECKPOINT_PATH=outputs/clstr_unified_stage3_toolbench_g3_hrpo/checkpoints/clstr_unified_stage3_hrpo-step500.pt \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh
# stage4_job=77278
```

依赖关系：

```text
77275 Stage 1 retrieval warmup
  -> 77276 Stage 2 full base
    -> 77277 Stage 3 offline HRPO
      -> 77278 Stage 4 ACT, initialized from Stage 3 checkpoint
```

提交时 `77275` 状态为 `PD (Priority)`，后续不要高频检查。短期建议约 15 分钟检查一次，
如果进入长时间训练阶段则按 30 分钟或更长间隔检查。

## 2026-05-26 v4 ToolBench-G3 static routing eval entrypoint checkpoint

v4 计划中 ToolBench-G3 主指标最终需要 StableToolBench simulation pass-rate；但在完整
pass-rate executor 接入前，论文成功条件也明确需要 ToolBench-G3 routing recall@5。因此本次补齐
一个不冒充 pass-rate 的 **ToolBench-G3 static routing eval** 入口，供 Stage 1 retrieval
checkpoint 产出后直接评估 routing recall/NDCG/MAP。

### 已完成的主要改动

1. `clstr/toolbench_g3_routing_eval.py`
   - 新增 `prepare_toolbench_g3_routing_eval()`；
   - 从 normalized `data/toolbench_g3/retrieval.jsonl` 生成：
     - `queries.jsonl`
     - `qrels.jsonl`
     - `manifest.json`
   - 对同一个 ToolBench-G3 query 去重；
   - 对同一个 query 的多个 positive API 聚合为多个 positive qrels；
   - 明确记录：
     `metric_scope=static routing/retrieval recall over normalized ToolBench-G3 relevant APIs; not StableToolBench pass-rate.`

2. `scripts/prepare_toolbench_g3_routing_eval.py`
   - 新增 CLI，默认输出到：
     `data/toolbench_g3_routing_eval`

3. `scripts/sbatch/run_toolbench_g3_routing_eval.sh`
   - 新增 sbatch 入口，流程为：
     1. `scripts/audit_toolbench_g3_data.py`
     2. `scripts/prepare_toolbench_g3_routing_eval.py`
     3. `scripts/export_clstr_retrieval_run.py`
     4. `scripts/evaluate_retrieval_run.py`
   - 默认 checkpoint：
     `outputs/clstr_unified_stage1_toolbench_g3_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
   - 默认输出：
     `outputs/toolbench_g3/clstr_routing_eval`

4. 测试
   - `tests/test_toolbench_g3_routing_eval.py`
   - `tests/test_sbatch_scripts.py::test_toolbench_g3_routing_eval_sbatch_prepares_exports_and_evaluates_run`

### 验证证据

先写红灯测试，初始失败为：

```text
ModuleNotFoundError: No module named 'clstr.toolbench_g3_routing_eval'
FileNotFoundError: scripts/sbatch/run_toolbench_g3_routing_eval.sh
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py tests/test_toolbench_g3_routing_eval.py -q
# 21 passed
```

```bash
python -m py_compile clstr/toolbench_g3_routing_eval.py scripts/prepare_toolbench_g3_routing_eval.py
bash -n scripts/sbatch/run_toolbench_g3_routing_eval.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/toolbench_g3_routing_eval.py \
  scripts/prepare_toolbench_g3_routing_eval.py \
  scripts/sbatch/run_toolbench_g3_routing_eval.sh \
  tests/test_toolbench_g3_routing_eval.py \
  tests/test_sbatch_scripts.py \
  description2.md
# no output
```

### 实际准备数据结果

已在登录节点只做轻量 JSONL 准备：

```bash
python scripts/prepare_toolbench_g3_routing_eval.py \
  --retrieval_path data/toolbench_g3/retrieval.jsonl \
  --output_dir data/toolbench_g3_routing_eval
```

结果：

```text
status=ok
source_rows=74555
query_count=25709
positive_qrels=73478
query_text_conflicts=0
queries.jsonl lines=25709
qrels.jsonl lines=73478
```

### Stage 1 后评估作业

已把三路 routing/retrieval eval 挂到 Stage 1 `77275` 之后：

```bash
sbatch --dependency=afterok:77275 --gpus=1 -p gpu_a800 scripts/sbatch/run_export_clstr_retrieval_run.sh
# toolret_eval_job=77283

sbatch --dependency=afterok:77275 --gpus=1 -p gpu_a800 scripts/sbatch/run_export_traject_retrieval_run.sh
# traject_eval_job=77284

sbatch --dependency=afterok:77275 --gpus=1 -p gpu_a800 scripts/sbatch/run_toolbench_g3_routing_eval.sh
# toolbench_g3_routing_eval_job=77285
```

这三路作业只证明 Stage 1 retrieval/routing 能力；ToolBench-G3 official pass-rate 仍需后续
StableToolBench simulation executor/verifier 接入，不能用 static routing eval 替代。

## 2026-05-26 07:29 CST — ToolBench-G3 HF 镜像续传与并发下载器

### 背景

`docs/clstr_v4_plan.md` 的 Phase B/C 需要官方 ToolBench-G3 数据进入
`data/toolbench_g3`，再参与 unified v2 重建。官方 Google/Tsinghua
`data.zip` 当前不可稳定获取，已采用 Hugging Face 镜像数据集
`Adorg/ToolBench` 作为 `official_g3_mirror`。

### 当前原始数据状态

- `ToolBench/data/instruction/G3_query.json` 已存在，大小约 300MB；
- `ToolBench/data/answer/G3_answer/*.json` 当前已下载 851 / 5000；
- `data/toolbench_g3` normalized export 仍未生成；
- 因此 ToolBench-G3 仍是 Phase B 未完成项，不能作为主表数据使用。

### 代码改动

`scripts/download_toolbench_g3_from_hf.py` 新增并发续传能力：

- 新增 `workers: int = 1` 参数和 CLI `--workers`；
- `workers > 1` 时使用 `ThreadPoolExecutor` 并发下载所选 G3 文件；
- 保留原有 skip existing file 逻辑，已下载的 query/answer 文件不会重复下载；
- manifest 记录 `workers`，便于复现实验环境；
- 保留 allowed root gate，输出仍限制在
  `/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 下。

### 验证证据

先验证新增 workers 测试在实现前失败：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_workers_download_all_selected_files -q
# failed: TypeError: download_toolbench_g3_from_hf() got an unexpected keyword argument 'workers'
```

实现后通过目标回归：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_dry_run_selects_only_g3_required_files \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_downloads_selected_files_and_verifies \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_workers_download_all_selected_files \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_tolerates_move_race_when_target_exists -q
# 4 passed
```

```bash
python -m py_compile scripts/download_toolbench_g3_from_hf.py
git diff --check scripts/download_toolbench_g3_from_hf.py tests/test_toolbench_g3_import.py
# exit 0
```

### 2026-05-26 08:02 CST — ToolBench-G3 importer 轨迹质量修正

检查已下载的真实 ToolBench-G3 answer tree 后，发现原 importer 有三类会污染
Stage 2/4 supervised trajectory 的问题：

1. `Finish` 节点被当成普通 tool action，可能生成
   `toolbench-g3/unknown/finish`；
2. `win=False` 的失败 answer tree 会被当作正例轨迹写入
   `trajectories.jsonl`；
3. 真实 DFS tree 不是单一路径，旧实现只取第一个 child，可能选中失败分支并错过后面的
   `Finish/give_answer` 成功分支。

### 代码改动

`scripts/import_toolbench_g3.py`：

- `Finish` 现在作为终止信号，不再生成 skill row / trajectory row；
- 只有 `answer["win"] is True` 的 answer tree 会生成 supervised trajectories；
  `win=False` 只保留 retrieval pairs，并在 manifest 中计数：
  - `answer_files`
  - `successful_answer_files`
  - `failed_answer_files`
  - `skipped_failed_answer_files`
- action alias 增加确定性变体，覆盖 ToolBench 常见命名差异：
  - tool name 侧增加 `get_<tool>` 变体；
  - api name 侧允许去掉开头 `get_`；
  - 例如 `latest_coupons_for_get_27coupons` 可映射到
    `toolbench-g3/27coupons/latest-coupons`；
  - `polygon_balance_from_specific_network_for_cryptocurrency_balance` 可映射到
    `toolbench-g3/cryptocurrency-balance/get-polygon-balance-from-specific-network`。
- 成功 answer tree 中仍无法解析到 `api_list` 的 action 不再生成
  `toolbench-g3/unknown/*`，而是跳过该 answer 的 supervised trajectory，并在 manifest 中记录：
  - `skipped_unresolved_answer_files`
  - `unresolved_action_count`
  - `unresolved_action_samples`
- DFS tree traversal 改为枚举候选路径：
  - 优先选择走到 `Finish/give_answer` 的路径；
  - 多个成功路径同时存在时，优先选择所有 action 都能解析到 `api_list` 的路径；
  - 若仍不可解析，则跳过该 answer trajectory，不污染训练。

### 真实样本抽查

在当前已下载的前 300 个 G3 answer 文件中抽查 `win=True` 样本：

```text
win_files_sample: 176
all_resolved_paths: 168
mapped_rows: 551
unresolved_actions: 14
unresolved_rate: 0.0248
```

未解析样本从修正前约 4.3% 降到约 2.5%。剩余未解析大多是 action 只写 tool 名、
拼写缺字，或同一 query 下存在多 API 歧义；当前策略是不把它们硬映射成任务特异规则，
而是跳过对应 supervised trajectory，保留 retrieval pairs。

### 验证证据

新增 TDD 回归覆盖：

- `test_import_toolbench_g3_treats_finish_as_terminal_not_unknown_skill`
- `test_import_toolbench_g3_skips_failed_answer_trees_for_supervised_trajectories`
- `test_import_toolbench_g3_resolves_common_action_alias_variants`
- `test_import_toolbench_g3_skips_successful_tree_with_unresolved_actions`
- `test_import_toolbench_g3_prefers_successful_finish_branch_over_first_failed_branch`
- `test_import_toolbench_g3_prefers_resolved_success_branch_over_unresolved_success_branch`

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py -q
# 30 passed
```

```bash
python -m py_compile \
  scripts/import_toolbench_g3.py \
  scripts/download_toolbench_g3_from_hf.py \
  clstr/toolbench_g3_audit.py \
  scripts/audit_toolbench_g3_data.py \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/build_clstr_unified_pretrain.py

git diff --check \
  scripts/import_toolbench_g3.py \
  scripts/download_toolbench_g3_from_hf.py \
  clstr/toolbench_g3_audit.py \
  scripts/audit_toolbench_g3_data.py \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/build_clstr_unified_pretrain.py \
  scripts/sbatch/run_import_toolbench_g3.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  tests/test_toolbench_g3_import.py \
  tests/test_unified_pretrain_v2.py \
  tests/test_unified_training_readiness.py \
  description2.md
# exit 0
```

### 2026-05-26 08:03 CST — ToolBench-G3 下载低频检查与续传重启

按用户要求约半小时检查一次长下载。07:29 启动的旧下载进程检查结果：

- `G3_answer/*.json`: 851 / 5000；
- `hf_g3_download_manifest.json`: 不存在；
- `hf_g3_download_stdout.log`: 无有效输出；
- `.part` 文件数: 0；
- pid 文件中的旧进程无可见 `ps` 输出。

判断：旧下载未继续推进，且未留下 manifest。已用新版带单文件重试的下载器重启续传：

```bash
nohup python scripts/download_toolbench_g3_from_hf.py \
  --api_manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/adorg_toolbench_api.json \
  --output_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data \
  --manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/hf_g3_download_manifest.json \
  --timeout 60 \
  --workers 16 \
  --max_retries 3 \
  > /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/hf_g3_download_stdout.log 2>&1 &
```

新 PID：

```text
1192795
```

下一次下载状态检查仍按低频节奏执行，不做高频轮询。

### 2026-05-26 08:06 CST — ToolBench-G3 完整下载后的标准 unified v2 pipeline

为避免 ToolBench-G3 下载完成后手工漏掉某个 gate，新增标准 sbatch 入口：

```text
scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh
```

该脚本不启动训练，只串联数据准备和审计：

1. raw ToolBench-G3 verify
   - `scripts/download_toolbench_data.py --verify_only`
   - 默认 `EXPECTED_G3_ANSWER_FILES=5000`
2. normalized ToolBench-G3 import
   - `scripts/import_toolbench_g3.py`
   - 默认输出 `data/toolbench_g3`
3. normalized ToolBench-G3 audit
   - `scripts/audit_toolbench_g3_data.py`
   - 继续要求 raw answer 数达到 5000
4. unified v2 rebuild
   - `scripts/build_clstr_unified_pretrain.py`
   - 默认输出 `data/clstr_unified_pretrain_v2_toolbench_g3`
5. unified readiness audit
   - `scripts/audit_clstr_unified_training_readiness.py`
   - 默认检查 `traject_bench,toolbench_g3`
   - 默认继续要求 `EXPECTED_G3_ANSWER_FILES=5000`

推荐提交方式：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh
```

如 H200 队列拥挤，可换用已有权限的其他 GPU 队列，例如：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh
```

### 验证证据

先写失败测试，旧状态失败在脚本不存在：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py::test_toolbench_g3_unified_pipeline_sbatch_runs_import_build_and_readiness_gates -q
# failed before implementation:
# FileNotFoundError: scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py::test_toolbench_g3_unified_pipeline_sbatch_runs_import_build_and_readiness_gates -q
# 1 passed
```

```bash
bash -n scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh
git diff --check scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh tests/test_sbatch_scripts.py
# exit 0
```

### 运行中任务

已在登录节点启动轻量网络下载进程（非 GPU 训练）：

```bash
nohup python scripts/download_toolbench_g3_from_hf.py \
  --api_manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/adorg_toolbench_api.json \
  --output_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data \
  --manifest_path /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/hf_g3_download_manifest.json \
  --timeout 60 \
  --workers 16 \
  > /data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/hf_g3_download_stdout.log 2>&1 &
```

PID 写入：

```text
/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/hf_g3_download.pid
```

后续按用户要求对长下载低频检查，约 30 分钟检查一次，不做高频轮询。

### 2026-05-26 补充 — 防止半成品 ToolBench-G3 被误放行

发现原始数据 verify 之前只检查 `answer/G3_answer/*.json` 是否存在，可能把
851/5000 这种半成品误判为可用。已收紧 gate：

- `scripts/download_toolbench_g3_from_hf.py`
  - `_verify_g3_output()` 新增 `expected_answer_files`；
  - HF manifest 中记录 `selected_answer_file_count`；
  - 下载结束后若 answer 数小于 selected count，则返回
    `status: incomplete_answer_files`，不再标记为 `ok`。
- `scripts/download_toolbench_data.py`
  - `verify_toolbench_data()` 新增 `expected_answer_files`；
  - CLI 新增 `--expected_answer_files`；
  - 显式传入期望数量时，answer 文件不足会返回
    `status: incomplete_answer_files`。
- `scripts/sbatch/run_import_toolbench_g3.sh`
  - normalized import 前先运行 raw preflight：

```bash
scripts/download_toolbench_data.py \
  --verify_only \
  --extract_root "${SOURCE_ROOT}/.." \
  --manifest_path "${RAW_VERIFY_OUTPUT}" \
  --expected_answer_files "${EXPECTED_G3_ANSWER_FILES}"
```

  - 默认 `EXPECTED_G3_ANSWER_FILES=5000`；
  - 因此当前 851/5000 的目录不会进入 `data/toolbench_g3` import。

### 验证证据

新增失败测试先确认旧实现问题：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_requires_all_selected_answer_files \
  tests/test_toolbench_g3_import.py::test_verify_toolbench_data_rejects_incomplete_answer_count_when_expected -q
# 2 failed before implementation:
# - downloader status was incorrectly "ok"
# - verify_toolbench_data had no expected_answer_files parameter
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py::test_toolbench_g3_import_sbatch_entrypoint_uses_local_source_path \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_dry_run_selects_only_g3_required_files \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_downloads_selected_files_and_verifies \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_requires_all_selected_answer_files \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_workers_download_all_selected_files \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_tolerates_move_race_when_target_exists \
  tests/test_toolbench_g3_import.py::test_verify_toolbench_data_accepts_full_g3_assets \
  tests/test_toolbench_g3_import.py::test_verify_toolbench_data_rejects_incomplete_answer_count_when_expected -q
# 8 passed
```

```bash
python -m py_compile \
  scripts/download_toolbench_g3_from_hf.py \
  scripts/download_toolbench_data.py \
  scripts/import_toolbench_g3.py \
  scripts/audit_toolbench_g3_data.py

bash -n scripts/sbatch/run_import_toolbench_g3.sh

git diff --check \
  scripts/download_toolbench_g3_from_hf.py \
  scripts/download_toolbench_data.py \
  scripts/sbatch/run_import_toolbench_g3.sh \
  tests/test_toolbench_g3_import.py \
  description2.md
# exit 0
```

### 2026-05-26 补充 — ToolBench-G3 完整性 gate 扩展到 normalized/readiness 层

进一步检查发现：即使 HF 下载器和 import 前 raw verify 已经要求
`EXPECTED_G3_ANSWER_FILES=5000`，`data/toolbench_g3` normalized export
和 unified readiness audit 仍可能只根据 “存在任意 G3 answer 文件” 或
“normalized 三个 JSONL 存在” 做弱判断。为避免半成品 G3 数据进入论文主表路径，
已把完整性 gate 扩展到四层：

1. raw 下载层
   - `scripts/download_toolbench_g3_from_hf.py`
   - 下载结束后用 HF manifest 的 `selected_answer_file_count` 校验 answer 数；
   - answer 不足返回 `incomplete_answer_files`。

2. raw verify / import 前层
   - `scripts/download_toolbench_data.py`
   - `verify_toolbench_data(..., expected_answer_files=...)`；
   - `scripts/sbatch/run_import_toolbench_g3.sh` 默认
     `EXPECTED_G3_ANSWER_FILES=5000`。

3. normalized ToolBench-G3 audit 层
   - `clstr/toolbench_g3_audit.py`
   - `audit_toolbench_g3_data(..., expected_answer_files=...)`；
   - raw source answer 数不足时加入 blocker：
     `source_root_incomplete_g3_answer_files`；
   - `scripts/audit_toolbench_g3_data.py` 新增 CLI 参数
     `--expected_answer_files`；
   - `scripts/sbatch/run_import_toolbench_g3.sh` 在 import 后的 normalized
     audit 也显式传入同一个 `EXPECTED_G3_ANSWER_FILES`。

4. unified readiness 层
   - `scripts/audit_clstr_unified_training_readiness.py`
   - 新增参数 `expected_toolbench_g3_answer_files`，并传给
     `audit_toolbench_g3_data()`；
   - `scripts/sbatch/run_clstr_unified_readiness_audit.sh` 默认
     `EXPECTED_TOOLBENCH_G3_ANSWER_FILES=5000` 并传入
     `--expected_toolbench_g3_answer_files`。

同时修正 unified builder 的 source inventory 语义：

- `scripts/build_clstr_unified_pretrain.py`
  - `_is_toolbench_g3_training_data()` 现在只把 normalized
    `data/toolbench_g3/{trajectories,retrieval,skills}.jsonl` 视为
    unified training source；
  - raw `ToolBench/data` 只作为 import 输入，不再让 source inventory
    提前标记 `toolbench_g3.available=True`；
  - 这避免当前 851/5000 的 raw 目录在 unified manifest 中被误认为
    ToolBench-G3 训练源已准备好。

### 验证证据

关键 TDD 失败点：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_does_not_mark_raw_toolbench_g3_as_training_ready_before_import -q
# failed before implementation:
# assert by_id["toolbench_g3"]["available"] is False
# actual: True
```

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py::test_audit_toolbench_g3_data_rejects_incomplete_raw_answer_count \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_blocks_toolbench_g3_when_raw_answer_count_incomplete -q
# failed before implementation:
# - audit_toolbench_g3_data() did not accept expected_answer_files
# - audit_unified_training_readiness() did not accept expected_toolbench_g3_answer_files
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_does_not_mark_toolbench_static_as_official_g3 \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_does_not_mark_raw_toolbench_g3_as_training_ready_before_import \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_includes_normalized_toolbench_g3_trajectories_retrieval_and_skills \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_requires_complete_normalized_toolbench_g3_files \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_blocks_toolbench_g3_when_raw_answer_count_incomplete \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_reports_toolbench_g3_data_blockers \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_sbatch_entrypoint_is_audit_only_and_local -q
# 29 passed
```

```bash
python -m py_compile \
  clstr/toolbench_g3_audit.py \
  scripts/audit_toolbench_g3_data.py \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/build_clstr_unified_pretrain.py \
  scripts/download_toolbench_data.py \
  scripts/download_toolbench_g3_from_hf.py

bash -n \
  scripts/sbatch/run_import_toolbench_g3.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh

git diff --check \
  clstr/toolbench_g3_audit.py \
  scripts/audit_toolbench_g3_data.py \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/build_clstr_unified_pretrain.py \
  scripts/download_toolbench_data.py \
  scripts/download_toolbench_g3_from_hf.py \
  scripts/sbatch/run_import_toolbench_g3.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  tests/test_toolbench_g3_import.py \
  tests/test_unified_pretrain_v2.py \
  tests/test_unified_training_readiness.py \
  description2.md
# exit 0
```

### 2026-05-26 07:45 CST — ToolBench-G3 HF 下载器加入单文件重试

背景：ToolBench-G3 还有数千个小 answer 文件需要从 HF mirror 续传。原并发实现中，
任意单个请求如果因为镜像瞬断失败，`ThreadPoolExecutor.map()` 会让整次下载提前抛异常退出。
这不利于长时间低频检查和断点续传。

### 代码改动

`scripts/download_toolbench_g3_from_hf.py` 新增：

- `max_retries: int = 2` 参数与 CLI `--max_retries`；
- `_download_file_with_retries()` 包装单文件下载；
- 临时失败会按 `max_retries + 1` 次尝试；
- 持续失败不会让整个下载器未记录地崩溃，而是在 manifest 写入：
  - `status: download_failed`
  - `failed_download_count`
  - `failed_downloads`
  - 每个失败文件的 `remote_file` / `attempts` / `error`
- 成功文件仍写入 `downloads`，已存在文件仍走 skip existing，后续可直接续传。

注意：07:29 启动的后台下载进程早于该重试逻辑。如果旧进程已失败或后续失败，
下一次续传会使用新版脚本，不会重新下载已有文件。

### 验证证据

先写失败测试，旧实现失败在缺少 `max_retries` 参数：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_retries_transient_file_failures \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_records_persistent_file_failures_without_crashing -q
# failed before implementation:
# TypeError: download_toolbench_g3_from_hf() got an unexpected keyword argument 'max_retries'
```

实现后通过下载器相关回归：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_dry_run_selects_only_g3_required_files \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_downloads_selected_files_and_verifies \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_requires_all_selected_answer_files \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_workers_download_all_selected_files \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_retries_transient_file_failures \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_records_persistent_file_failures_without_crashing \
  tests/test_toolbench_g3_import.py::test_download_toolbench_g3_from_hf_tolerates_move_race_when_target_exists -q
# 7 passed
```

```bash
python -m py_compile scripts/download_toolbench_g3_from_hf.py
git diff --check scripts/download_toolbench_g3_from_hf.py tests/test_toolbench_g3_import.py
# exit 0
```

## 2026-05-26 v4 TRAJECT query-only skill records + skill_pool quality gate

本记录继续推进 `docs/clstr_v4_plan.md` 的 unified CLSTR training dataset v2。
这次修复的核心问题是：TRAJECT query 的 `tool list` 中有一批工具没有出现在
`public_data/tools/all_tools.json`，旧 unified builder 会把这些被 trajectory /
retrieval 引用的 TRAJECT skills 写成 placeholder：

```text
provenance.missing_source_record = true
description = skill_id
```

这会直接损害 skill embedding 的输入质量。当前默认旧数据
`data/clstr_unified_pretrain_v2/skill_pool.jsonl` 经新 audit 检查发现：

```text
skill_pool_quality.status: action_required
missing_source_record_count: 665
missing_source_record_by_prefix: {traject: 665}
placeholder_description_count: 665
```

### 已完成的主要改动

1. `scripts/build_clstr_unified_pretrain.py`
   - `build_skill_pool()` 现在不仅读取 TRAJECT `tools/all_tools.json`，还会扫描
     TRAJECT query files 中实际被使用的 `tool list`；
   - 当某个 TRAJECT skill 被 trajectory/retrieval 引用、但不在 `all_tools.json`
     中时，会从 query tool record 补齐真实 `tool name` / `tool description` /
     parameter schema；
   - 这类记录的 provenance 标为：

```text
dedup_method = traject_query_tool_identity_v2
```

   - `parent tool name` 与 `API name` 都为 `unknown` 时继续使用 `tool name`
     回退生成 skill id，避免不同 query-only tools collapse 到
     `traject/unknown/unknown`。

2. `scripts/audit_clstr_unified_training_readiness.py`
   - 新增 `skill_pool_quality` 报告；
   - 检查 `provenance.missing_source_record=true`、空 description、以及
     `description == skill_id` 的 placeholder 情况；
   - Stage 2/3/4 gate 会在发现 placeholder skill records 时阻塞：

```text
skill_pool_has_placeholder_records
```

   - Stage 1 retrieval warmup 不因该项直接阻塞，因为当前 Stage 1 train-safe
     retrieval filter 只保留 SKILLRET + ToolRet train rows，TRAJECT public rows
     仍按 `public_train_or_eval_unlabeled` 排除。

3. `clstr/stage_preflight.py`
   - Stage 2 preflight 现在会解析 `skill_pool.jsonl`；
   - 如果发现 `provenance.missing_source_record=true`，会直接拒绝启动 Stage 2，
     防止绕过 readiness audit 后在低质量 skill embeddings 上训练 full-base。

4. 作业状态处理
   - 旧 fixed 前提交的 Stage 2 作业 `77152` 已 hold：

```text
77152: PENDING (JobHeldUser)
```

   - 这样不会中断仍在跑的旧 Stage 1 作业 `77151`，但也不会让旧 Stage 2
     在旧 `skill_pool` 上自动启动。

### 修复后数据构建

已通过 sbatch 构建独立数据目录，没有覆盖旧默认目录：

```bash
OUTPUT_DIR=data/clstr_unified_pretrain_v2_query_tool_fix \
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh
# job 77155
```

`slurm-77155.out` 显示构建完成，manifest 写入：

```text
data/clstr_unified_pretrain_v2_query_tool_fix/manifest.json
```

关键结果：

```text
status: ok
trajectory_stream.total_rows: 273269
retrieval_stream.total_pairs: 374110
skill_pool.total_skills: 36129
skill_pool.raw_skill_count: 37275
skill_pool.canonical_skill_count: 36129
skill_pool.missing_source_record_count: 0
skill_pool.placeholder_description_count: 0
skill_pool.stream_remap.trajectory_changed_skill_refs: 805
skill_pool.stream_remap.retrieval_changed_skill_refs: 18536
leakage_audit.status: ok
source_inventory.missing_required_target_sources: [toolbench_g3]
```

对修复后数据运行 readiness：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python \
  scripts/audit_clstr_unified_training_readiness.py \
  --data_root data/clstr_unified_pretrain_v2_query_tool_fix \
  --stage1_checkpoint outputs/clstr_unified_stage1_query_tool_fix_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt \
  --stage2_checkpoint outputs/clstr_unified_stage2_query_tool_fix_full_base/checkpoints/clstr_full_base-step10000.pt \
  --output_path outputs/clstr_unified_readiness_audit/readiness_report_query_tool_fix.json
```

结果：

```text
skill_pool_quality.status: ok
missing_source_record_count: 0
placeholder_description_count: 0
stage1_retrieval_warmup.ready_to_submit: true
stage2_full_base.blockers: [missing_stage1_checkpoint]
toolret_eval_ready: true
traject_eval_ready: true
toolbench_g3_data_ready: false
```

### 新 fixed-data Stage 1/2 作业

已基于修复后的数据目录提交新的 Stage 1：

```bash
sbatch \
  --export=ALL,DATA_ROOT=data/clstr_unified_pretrain_v2_query_tool_fix,OUTPUT_DIR=outputs/clstr_unified_stage1_query_tool_fix_retrieval_warmup \
  --gpus=1 -p gpu_a800 \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh
# job 77156
```

并提交依赖 `afterok:77156` 的 Stage 2：

```bash
sbatch \
  --dependency=afterok:77156 \
  --export=ALL,TRAIN_PATH=data/clstr_unified_pretrain_v2_query_tool_fix/trajectories.jsonl,SKILLS_PATH=data/clstr_unified_pretrain_v2_query_tool_fix/skill_pool.jsonl,OUTPUT_DIR=outputs/clstr_unified_stage2_query_tool_fix_full_base,ROUTING_CHECKPOINT_PATH=outputs/clstr_unified_stage1_query_tool_fix_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt \
  --gpus=1 -p gpu_a800 \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# job 77157
```

当前作业关系：

```text
77151: legacy/default-data Stage 1, still running; do not use for final Stage 2
77152: legacy/default-data Stage 2, held by user
77156: query-tool-fix Stage 1, running
77157: query-tool-fix Stage 2, pending afterok:77156
```

后续不要频繁轮询；Stage 1 属长作业，按约 30 分钟粒度检查即可。

### 验证证据

已按 TDD 先写失败测试，初始失败包括：

```text
tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_keeps_traject_query_only_unknown_tools_distinct
AssertionError: description == skill_id placeholder
```

以及：

```text
tests/test_unified_training_readiness.py::test_unified_training_readiness_blocks_stage2_when_skill_pool_has_placeholder_records
KeyError: 'skill_pool_quality'
```

和：

```text
tests/test_stage2_preflight.py::test_stage2_preflight_rejects_skill_pool_placeholder_records
Failed: DID NOT RAISE ValueError
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py \
  tests/test_unified_training_readiness.py \
  tests/test_stage2_preflight.py \
  tests/test_sbatch_scripts.py -q
# 43 passed
```

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_traject_eval_import.py \
  tests/test_unified_pretrain_v2.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py -q
# 45 passed
```

```bash
python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  scripts/audit_clstr_unified_training_readiness.py \
  clstr/stage_preflight.py \
  scripts/run_clstr_stage2_preflight.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh
# exit 0
```

```bash
git diff --check -- \
  scripts/build_clstr_unified_pretrain.py \
  tests/test_unified_pretrain_v2.py \
  scripts/audit_clstr_unified_training_readiness.py \
  tests/test_unified_training_readiness.py \
  clstr/stage_preflight.py \
  tests/test_stage2_preflight.py
# no output
```

## 2026-05-26 v4 TRAJECT eval normalization and audit checkpoint

本记录继续推进 `docs/clstr_v4_plan.md` 的 Phase B/H：TRAJECT-Bench 现在不仅能被
unified v2 builder 消费，也有了独立的标准 retrieval eval 数据格式和审计 gate。这样
Stage 1/2/3 checkpoint 产出后，可以直接对 TRAJECT 的 stepwise routing 做
NDCG/Recall/MAP 评测，并作为主表 #1 的 routing-side 证据之一。

### 已完成的主要改动

1. `clstr/traject_eval_import.py`
   - 新增 `import_traject_eval()`；
   - 输入 TRAJECT-Bench `public_data`；
   - 输出标准三件套：
     - `queries.jsonl`
     - `skills.jsonl`
     - `qrels.jsonl`
   - query 粒度为 trajectory step：
     - `query_id = traject::<trajectory_type>::<domain>::<file_stem>::<query_idx>::<step_idx>`；
     - `query_text` 包含 goal、trajectory type、domain、task 信息和 previous tools；
     - qrels 正例为该 step 的 ground-truth tool。
   - skill pool 使用 `tools/all_tools.json` 的完整工具池；
   - qrels 的 skill_id 通过 tool name 映射回 `all_tools.json`，避免 query 文件缺少
     `parent tool name` / `API name` 时产生 skill_id 不一致。

2. `clstr/traject_eval_audit.py`
   - 新增 `audit_traject_eval_data()`；
   - 复用通用 retrieval eval 完整性规则，检查：
     - required files 是否存在；
     - qrels 正例 query 是否都在 `queries.jsonl`；
     - qrels 正例 skill 是否都在 `skills.jsonl`；
     - skill id 是否重复；
     - 是否有 positive qrels。

3. CLI / sbatch 入口
   - 新增 `scripts/import_traject_eval.py`；
   - 新增 `scripts/audit_traject_eval_data.py`；
   - 新增 `scripts/sbatch/run_import_traject_eval.sh`：
     - 默认读取 `../TRAJECT-Bench/public_data`；
     - 默认输出 `data/traject_eval`；
     - 导入完成后立即写 `outputs/traject_eval/traject_eval_audit.json`；
     - audit 不是 `ok` 时退出，避免无效 eval 数据进入后续主表。
   - 新增 `scripts/sbatch/run_export_traject_retrieval_run.sh`：
     - 默认读取 `data/traject_eval/{queries,skills,qrels}.jsonl`；
     - 默认输出 `outputs/traject_eval/clstr_retrieval`；
     - 默认先跑 TRAJECT eval audit gate；
     - 然后导出 CLSTR retrieval run 并用 `scripts/evaluate_retrieval_run.py`
       计算 `NDCG@5/10`、`Recall@5/10`、`MAP@5/10`、`Precision@5/10`。

4. Unified readiness audit
   - `scripts/audit_clstr_unified_training_readiness.py` 新增 `--traject_eval_dir`；
   - `benchmark_eval_readiness` 现在同时报告：
     - `toolret`
     - `traject`
   - `paper_readiness` 新增 `traject_eval_ready`；
   - `scripts/sbatch/run_clstr_unified_readiness_audit.sh` 新增 `TRAJECT_EVAL_DIR`
     并传入 readiness CLI。

### 验证证据

已按 TDD 先写失败测试，初始失败包括：

```text
ModuleNotFoundError: No module named 'clstr.traject_eval_audit'
```

以及 sbatch 入口缺失：

```text
FileNotFoundError: scripts/sbatch/run_export_traject_retrieval_run.sh
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_traject_eval_import.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py \
  tests/test_generic_retrieval_eval.py \
  tests/test_clstr_retrieval_export.py -q
# 35 passed, 2 warnings
```

```bash
python -m py_compile \
  clstr/traject_eval_import.py \
  clstr/traject_eval_audit.py \
  scripts/import_traject_eval.py \
  scripts/audit_traject_eval_data.py \
  scripts/audit_clstr_unified_training_readiness.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_import_traject_eval.sh \
  scripts/sbatch/run_export_traject_retrieval_run.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh
# exit 0
```

### 当前未完成项

- 还需要通过 sbatch 在集群上运行完整 TRAJECT eval 导入：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_import_traject_eval.sh
```

- Stage 1 checkpoint 产出后，运行：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_export_traject_retrieval_run.sh
```

- TRAJECT 原生 EM / Inclusion / Usage / Traj-Satisfy / Acc 仍需要后续 executor /
  official-style evaluation 接入；本次完成的是 retrieval/routing eval gate，不等价于
  完整 TRAJECT task-success 主表。

## 2026-05-26 v4 ToolBench-G3 official-data audit gate checkpoint

本记录继续推进 `docs/clstr_v4_plan.md` 的 Phase B/H。当前本地状态确认：

- `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench` 只有代码仓库和
  `data_example`；
- `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data.zip.invalid` 存在，
  说明之前下载到的压缩包不是有效官方数据；
- `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/verify_manifest.json`
  显示官方完整 G3 文件缺失：
  - `instruction/G3_query.json`
  - `answer/G3_answer/*.json`
- 因此 ToolBench-G3 仍是论文主表的真实数据 blocker，不能用 repo、`data_example`
  或 static smoke 替代。

### 已完成的主要改动

1. `clstr/toolbench_g3_audit.py`
   - 新增 `audit_toolbench_g3_data()`；
   - 只有下面两类情况才可能 `status: ok`：
     - `data/toolbench_g3/{skills,retrieval,trajectories}.jsonl` 三件套完整；
     - 且 normalized 数据内部引用一致。
   - 显式拒绝：
     - 缺 normalized 三件套；
     - `source_root` 指向 `data_example`；
     - static smoke 资产替代官方 G3；
     - retrieval positive / negative skill 引用不在 `skills.jsonl`；
     - trajectory `skill_id` / `next_skill_id` 引用不在 `skills.jsonl`。
   - 输出 `may_use_for_paper_main_table`，确保后续不会误把 smoke 数据放进主表。

2. `scripts/audit_toolbench_g3_data.py`
   - 新增 CLI：

```bash
python scripts/audit_toolbench_g3_data.py \
  --data_dir data/toolbench_g3 \
  --source_root ../ToolBench/data \
  --output_path outputs/toolbench_g3/toolbench_g3_audit.json \
  --fail_on_action_required
```

3. `scripts/sbatch/run_import_toolbench_g3.sh`
   - 在 import 后新增审计 gate；
   - 默认写：

```text
outputs/toolbench_g3/toolbench_g3_audit.json
```

   - audit 不是 `ok` 时退出，避免不完整 ToolBench-G3 进入 unified v2 或主表。

4. `scripts/audit_clstr_unified_training_readiness.py`
   - 新增参数：
     - `--toolbench_g3_data_dir`
     - `--toolbench_g3_source_root`
   - `benchmark_eval_readiness` 新增 `toolbench_g3`；
   - `paper_readiness` 新增 `toolbench_g3_data_ready`；
   - `scripts/sbatch/run_clstr_unified_readiness_audit.sh` 同步传入：
     - `TOOLBENCH_G3_DATA_DIR`
     - `TOOLBENCH_G3_SOURCE_ROOT`

### 验证证据

已按 TDD 先写失败测试，初始失败包括：

```text
ModuleNotFoundError: No module named 'clstr.toolbench_g3_audit'
```

以及 readiness 未接入时：

```text
TypeError: audit_unified_training_readiness() got an unexpected keyword argument 'toolbench_g3_data_dir'
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolbench_g3_import.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py -q
# 41 passed
```

```bash
python -m py_compile \
  clstr/toolbench_g3_audit.py \
  scripts/audit_toolbench_g3_data.py \
  scripts/audit_clstr_unified_training_readiness.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_import_toolbench_g3.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/toolbench_g3_audit.py \
  scripts/audit_toolbench_g3_data.py \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/sbatch/run_import_toolbench_g3.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  tests/test_toolbench_g3_import.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py
# no output
```

### 当前未完成项

- ToolBench-G3 官方完整数据仍未落地。需要在登录节点手动或通过可用镜像下载官方
  `data.zip`，使以下文件存在：

```text
/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data/instruction/G3_query.json
/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data/answer/G3_answer/*.json
```

- 官方数据到位后，再提交：

```bash
SOURCE_ROOT=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/ToolBench/data \
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_import_toolbench_g3.sh
```

- 在此之前，`paper_readiness.full_three_benchmark_data_ready` 和
  `paper_readiness.toolbench_g3_data_ready` 应保持 false。

### ToolRet full eval data landing

完整 ToolRet eval 原始数据已在登录节点通过 HF mirror 下载到：

```text
/data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_eval
```

命令：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python \
  scripts/download_toolret_eval_data.py \
  --output_dir /data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_eval
```

结果：

```text
status: ok
query_count: 7961
tool_count: 44453
query_configs: 35 ToolRet tasks
tool_configs: code/customized/web
hf_endpoint: https://hf-mirror.com
```

下载过程中 HF mirror 出现若干 `HTTP Error 429` retry，但最终成功完成。

随后已提交规范化作业，将完整原始数据转成 CLSTR 本地 eval 格式：

```bash
QUERY_SOURCE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_eval/queries.jsonl \
TOOL_SOURCE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_eval/tools.jsonl \
OUTPUT_DIR=data/toolret_eval \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_import_toolret_eval.sh
# Submitted batch job 77149
```

该作业完成后应检查：

```bash
python scripts/audit_toolret_eval_data.py \
  --data_dir data/toolret_eval \
  --output_path outputs/toolret_eval/toolret_eval_audit.json \
  --fail_on_action_required
```

若 audit 为 `ok`，即可用：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_export_clstr_retrieval_run.sh
```

导出 CLSTR ToolRet run 并评测 `NDCG@5/10`、`Recall@5/10`、`MAP@10`。

ToolRet eval 规范化作业调度记录：

```bash
# 初始提交到 H200 后长时间 pending，且未开始写 data/toolret_eval，因此取消。
scancel 77149

# 改投 A800，避免继续占 H200 队列。
QUERY_SOURCE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_eval/queries.jsonl \
TOOL_SOURCE=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_eval/tools.jsonl \
OUTPUT_DIR=data/toolret_eval \
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_import_toolret_eval.sh
# Submitted batch job 77150
```

`77150` 已完成，`data/toolret_eval/manifest.json`：

```text
status: ok
query_count: 7961
skill_count: 44453
qrel_count: 14106
missing_qrel_skill_ids: []
```

完整 audit 已通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python \
  scripts/audit_toolret_eval_data.py \
  --data_dir data/toolret_eval \
  --output_path outputs/toolret_eval/toolret_eval_audit.json \
  --fail_on_action_required
```

结果：

```text
status: ok
blockers: []
query_count: 7961
unique_query_count: 7961
skill_count: 44453
unique_skill_count: 44453
qrel_count: 14106
positive_qrel_count: 14106
missing_positive_qrel_skill_ids: []
missing_positive_qrel_query_ids: []
duplicate_skill_ids: []
```

结论：ToolRet 主表 #3 的 eval 数据已经从“未接入”推进到“本地完整、可审计、
可用于 CLSTR run export/evaluation”的状态。下一步等 train-safe Stage 1 checkpoint
完成后，可运行 `scripts/sbatch/run_export_clstr_retrieval_run.sh` 产出 ToolRet
`NDCG@5/10`、`Recall@5/10`、`MAP@10`。

### Stage 1/2 改投 A800

由于 H200 队列持续 pending，且 `77147/77148` 尚未启动、对应输出目录尚未写入新产物，
已按“如果 H200 用的人太多可以试试其他显卡”的原则改投 A800。

```bash
scancel 77147 77148
```

重新提交：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh
# Submitted batch job 77151
```

Stage 2 依赖同步重建到新的 Stage 1：

```bash
sbatch --dependency=afterok:77151 --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# Submitted batch job 77152
```

当前含义：

```text
77151: train-safe Stage 1 retrieval warmup on A800
77152: Stage 2 full-base train on A800, afterok:77151
```

后续不要频繁轮询；Stage 1 属长作业，约 30 分钟粒度检查即可。

### 本地回归

在提交 `77149` 后，已先跑完整本地回归，验证新增 ToolRet eval / generic retrieval
链路代码：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolret_eval_import.py \
  tests/test_clstr_retrieval_export.py \
  tests/test_generic_retrieval_eval.py \
  tests/test_export_retrieval_qrels.py \
  tests/test_sbatch_scripts.py -q
# 30 passed, 2 warnings
```

```bash
python -m py_compile \
  clstr/toolret_eval_import.py \
  clstr/toolret_eval_audit.py \
  clstr/clstr_retrieval_export.py \
  clstr/retrieval_metrics.py \
  clstr/retrieval_qrels.py \
  scripts/download_toolret_eval_data.py \
  scripts/import_toolret_eval.py \
  scripts/audit_toolret_eval_data.py \
  scripts/export_clstr_retrieval_run.py \
  scripts/evaluate_retrieval_run.py \
  scripts/export_retrieval_qrels.py
# exit 0
```

```bash
bash -n scripts/sbatch/run_import_toolret_eval.sh scripts/sbatch/run_export_clstr_retrieval_run.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/clstr_retrieval_export.py \
  scripts/export_clstr_retrieval_run.py \
  scripts/sbatch/run_export_clstr_retrieval_run.sh \
  tests/test_clstr_retrieval_export.py \
  tests/test_sbatch_scripts.py \
  description2.md
# no output
```

## 2026-05-26 v4 ToolRet eval HF config expansion fix

本记录是 ToolRet eval 数据规范化入口的后续修复。官方
`tool-retrieval-benchmark/toolret/eval.py` 里的 `all` 不是 HF dataset 的真实
config，而是需要展开成所有 ToolRet task；ToolRet tools 同理，需要展开成
`code/web/customized` 三个 category。若直接调用
`load_dataset("mangopy/ToolRet-Queries", "all")`，后续登录节点下载官方 eval
数据会失败。

### 已完成的主要改动

1. `clstr/toolret_eval_import.py`
   - 新增官方 ToolRet task/category 常量；
   - `_iter_hf_queries(tasks=None)` 默认展开为全部官方 ToolRet tasks；
   - `_iter_hf_tools(categories=None)` 默认展开为官方 categories：
     `code`、`customized`、`web`；
   - 显式传入 task/category 时仍只加载指定子集。

2. `tests/test_toolret_eval_import.py`
   - 新增回归：默认 HF 路径不能把 `"all"` 传给 `load_dataset`；
   - 确认 `apigen` 和 `web` 等官方 config 会被使用。

### 验证证据

已先写失败测试，初始失败为：

```text
AssertionError: ('mangopy/ToolRet-Queries', 'all', 'queries') not in calls
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolret_eval_import.py -q
# 4 passed
```

```bash
python -m py_compile clstr/toolret_eval_import.py scripts/import_toolret_eval.py
# exit 0
```

```bash
git diff --check -- clstr/toolret_eval_import.py tests/test_toolret_eval_import.py description2.md
# no output
```

## 2026-05-26 v4 ToolRet eval raw download entrypoint checkpoint

本记录继续对应 ToolRet 主表 #3 的数据准备。前面已经有
`scripts/import_toolret_eval.py`，但它消费的是本地 JSONL 或在线 HF dataset；为了符合
“登录节点下载、计算节点不联网”的集群约束，本次新增一个登录节点专用的官方
ToolRet eval 原始数据落地脚本。

### 已完成的主要改动

1. `scripts/download_toolret_eval_data.py`
   - 新增登录节点下载/导出入口；
   - 从 HF mirror 读取：
     - `mangopy/ToolRet-Queries`
     - `mangopy/ToolRet-Tools`
   - 将官方 queries/tools 分别写成普通 JSONL：
     - `queries.jsonl`
     - `tools.jsonl`
     - `download_manifest.json`
   - 默认输出位置：
     `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/toolret_eval`
   - 支持 `--dry_run`，用于无网络/不下载时检查 config 展开和本地输出路径；
   - 支持 `--tasks`、`--categories`、`--max_queries_per_config`、`--max_tools_per_config`
     做小样本 smoke。

2. `tests/test_toolret_eval_import.py`
   - 新增 dry-run 回归：
     - 官方 query configs 必须包含 `apigen`；
     - tool configs 必须为 `code/customized/web`；
     - 输出文件名必须为 `queries.jsonl` / `tools.jsonl`；
     - `HF_ENDPOINT` 必须为 `https://hf-mirror.com`。

### 验证证据

已按 TDD 先写失败测试，初始失败为：

```text
FileNotFoundError: scripts/download_toolret_eval_data.py
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolret_eval_import.py -q
# 5 passed
```

```bash
python -m py_compile \
  clstr/toolret_eval_import.py \
  scripts/import_toolret_eval.py \
  scripts/download_toolret_eval_data.py
# exit 0
```

```bash
git diff --check -- \
  clstr/toolret_eval_import.py \
  scripts/download_toolret_eval_data.py \
  tests/test_toolret_eval_import.py \
  description2.md
# no output
```

### ToolRet eval smoke 结果

登录节点小样本 smoke 已跑通，用 HF mirror 只拉取：

```text
queries: apigen, max_queries_per_config=2
tools: web, max_tools_per_config=5
```

命令：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python \
  scripts/download_toolret_eval_data.py \
  --output_dir .codex_tmp/toolret_eval_smoke_raw \
  --tasks apigen \
  --categories web \
  --max_queries_per_config 2 \
  --max_tools_per_config 5
```

结果：

```text
status: ok
query_count: 2
tool_count: 5
hf_endpoint: https://hf-mirror.com
```

随后用本地 JSONL import：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python \
  scripts/import_toolret_eval.py \
  --query_source .codex_tmp/toolret_eval_smoke_raw/queries.jsonl \
  --tool_source .codex_tmp/toolret_eval_smoke_raw/tools.jsonl \
  --output_dir .codex_tmp/toolret_eval_smoke_import \
  --split test
```

结果：

```text
status: ok
query_count: 2
skill_count: 5
qrel_count: 4
missing_qrel_skill_ids:
  - apigen_tool_1530
  - apigen_tool_1591
  - apigen_tool_1970
  - apigen_tool_289
```

解释：这是预期的 smoke 现象，因为只截取了前 5 个 `web` tools，qrels 正例不一定在
小样本 tool pool 中。完整 ToolRet eval 主表必须使用完整 `ToolRet-Tools`，不能用该
smoke 结果。

## 2026-05-26 v4 ToolRet eval audit gate checkpoint

为避免上面的小样本问题污染主表，本次新增 ToolRet eval 完整性审计 gate。该 gate 会在
CLSTR retrieval run 导出前检查：

- `queries.jsonl` / `skills.jsonl` / `qrels.jsonl` 是否存在；
- qrels 正例 skill 是否都在 `skills.jsonl`；
- qrels 正例 query 是否都在 `queries.jsonl`；
- skill id 是否重复；
- 是否至少存在一个 positive qrel。

### 已完成的主要改动

1. `clstr/toolret_eval_audit.py`
   - 新增 `audit_toolret_eval_data()`；
   - 输出 `status: ok` 或 `status: action_required`；
   - action_required 时列出 blockers，例如：
     - `missing_positive_qrel_skill_ids`
     - `missing_positive_qrel_query_ids`
     - `duplicate_skill_ids`
     - `no_positive_qrels`
     - `missing_required_files`

2. `scripts/audit_toolret_eval_data.py`
   - 新增 CLI：

```bash
python scripts/audit_toolret_eval_data.py \
  --data_dir data/toolret_eval \
  --output_path outputs/toolret_eval/audit.json \
  --fail_on_action_required
```

3. `scripts/sbatch/run_export_clstr_retrieval_run.sh`
   - 新增 `RUN_PREFLIGHT_AUDIT=1` 默认 gate；
   - 默认先跑：

```bash
scripts/audit_toolret_eval_data.py \
  --data_dir "${TOOLRET_DATA_DIR}" \
  --output_path "${OUTPUT_DIR}/toolret_eval_audit.json" \
  --fail_on_action_required
```

   - 如果 audit 不是 `ok`，脚本会在导出 CLSTR run 前退出，避免产生无效 ToolRet
     主表指标；
   - 如用于非 ToolRet 的同构 qrels，可显式设置 `RUN_PREFLIGHT_AUDIT=0`。

4. 测试
   - `tests/test_toolret_eval_import.py` 新增审计库函数和 CLI 回归；
   - `tests/test_sbatch_scripts.py` 新增 export sbatch preflight audit 覆盖。

### 验证证据

已按 TDD 先写失败测试，初始失败包括：

```text
ModuleNotFoundError: No module named 'clstr.toolret_eval_audit'
```

以及：

```text
can't open file 'scripts/audit_toolret_eval_data.py': [Errno 2] No such file or directory
```

实现后通过：

```bash
TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.codex_tmp/pytest_tmp \
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_toolret_eval_import.py::test_audit_toolret_eval_cli_writes_report \
  tests/test_sbatch_scripts.py::test_clstr_retrieval_export_sbatch_exports_and_optionally_evaluates_generic_run -q
# 2 passed
```

```bash
python -m py_compile clstr/toolret_eval_audit.py scripts/audit_toolret_eval_data.py
# exit 0
```

```bash
bash -n scripts/sbatch/run_export_clstr_retrieval_run.sh
# exit 0
```

## 2026-05-26 v4 split-safe TRAJECT held-out eval 默认路径对齐

### 背景

v4 训练线已经把 TRAJECT public_data 做成确定性 trajectory-level hash split：

- unified train 数据只消费 `assigned_traject_split == train`；
- 后续 TRAJECT eval 应只使用 held-out `test` partition；
- 旧默认目录 `data/traject_eval` 容易和历史未切分导入混淆。

因此将未来默认的 TRAJECT held-out eval 数据和输出路径改为独立目录：

- 数据目录：`data/traject_eval_traject_split_test`
- eval 输出：`outputs/traject_eval_traject_split_test`

### 已完成改动

1. `scripts/sbatch/run_import_traject_eval.sh`
   - 默认 `OUTPUT_DIR=data/traject_eval_traject_split_test`；
   - 默认 `SPLIT=test` 且 `SPLIT_PARTITION=test`；
   - 默认 audit 输出 `outputs/traject_eval_traject_split_test/traject_eval_audit.json`。

2. `scripts/sbatch/run_export_traject_retrieval_run.sh`
   - 默认读取 `data/traject_eval_traject_split_test/{queries,skills,qrels}.jsonl`；
   - 默认输出 `outputs/traject_eval_traject_split_test/clstr_retrieval`；
   - 继续保留 TRAJECT retrieval metrics 和 sequence proxy eval。

3. `scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh`
   - readiness gate 默认 `TRAJECT_EVAL_DIR=data/traject_eval_traject_split_test`。

4. `scripts/sbatch/run_clstr_unified_readiness_audit.sh`
   - readiness gate 默认 `TRAJECT_EVAL_DIR=data/traject_eval_traject_split_test`。

5. CLI / library 默认值同步：
   - `scripts/import_traject_eval.py`
   - `scripts/audit_traject_eval_data.py`
   - `scripts/audit_clstr_unified_training_readiness.py`
   - `clstr/traject_eval_import.py`
   - `clstr/traject_eval_audit.py`

### 验证证据

先写路径期望测试并确认失败：

```text
3 failed
```

失败点均为 sbatch 脚本仍指向旧 `data/traject_eval` 或 `outputs/traject_eval`。实现路径切换后通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py::test_traject_eval_import_sbatch_uses_local_public_data_and_repo_cache \
  tests/test_sbatch_scripts.py::test_unified_readiness_sbatch_includes_traject_eval_gate \
  tests/test_sbatch_scripts.py::test_traject_retrieval_export_sbatch_exports_and_evaluates_traject_run -q
# 3 passed
```

完整相关回归：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py \
  tests/test_traject_eval_import.py \
  tests/test_traject_split.py \
  tests/test_traject_sequence_proxy.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_uses_unified_v2_data_and_local_cache \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  tests/test_sbatch_scripts.py::test_unified_stage3_hrpo_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_toolbench_g3_unified_pipeline_sbatch_runs_import_build_and_readiness_gates \
  tests/test_sbatch_scripts.py::test_traject_eval_import_sbatch_uses_local_public_data_and_repo_cache \
  tests/test_sbatch_scripts.py::test_traject_retrieval_export_sbatch_exports_and_evaluates_traject_run -q
# 49 passed
```

静态检查：

```bash
python -m py_compile \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/import_traject_eval.py \
  scripts/audit_traject_eval_data.py \
  clstr/traject_eval_import.py \
  clstr/traject_eval_audit.py \
  clstr/traject_split.py \
  clstr/traject_sequence_proxy.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  scripts/sbatch/run_import_traject_eval.sh \
  scripts/sbatch/run_export_traject_retrieval_run.sh \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh
# exit 0
```

## 2026-05-26 Stage 4 L_act transition condition 修正

### 问题

v4 计划要求 Stage 4 是 transition-conditioned next-skill objective：

```text
L_act = -log P_T(a+_{t+1} | m_hat_{t+1}, C_{t+1})
transition input = state/history at t + action_t + observation_t
```

实现审计时发现 `clstr/stage4_act_train.py::_compute_stage4_act_loss()` 虽然已经读取了
`next_observation_text` 并编码成 `obs_emb`，但调用 `_transition_prediction()` 时仍把
`action_text` embedding 作为第三个 transition 输入传入。结果 Stage 4 的 L_act 实际更接近
`current belief + action` 条件，而不是严格的 `(m_t, action_t, observation_t)` transition condition。

### 修正

1. `clstr/stage4_act_train.py`
   - `_compute_stage4_act_loss()` 现在将 `obs_emb` 传入 `_transition_prediction()`；
   - action 输入仍由 `_transition_prediction()` 内部通过 `transition_action_input(model, labels, like=obs_emb)`
     从当前 `skill_idx` 生成，符合 v4 的 `action_proj(E)[a_t]` 设计；
   - 修正范围只限 Stage 4 L_act 数据流，不改变数据格式、不引入 AppWorld 特异逻辑。

2. `tests/test_stage4_act_train.py`
   - 新增 `test_stage4_act_loss_conditions_transition_on_next_observation_embedding`；
   - 用 recording transition 直接断言 Stage 4 transition 第三个输入来自
     `next_observation_text` embedding，而不是 action text embedding。

### TDD 证据

先写失败测试，当前实现失败：

```text
expected transition obs arg: tensor([[0., 1., 0.]])
actual transition obs arg:   tensor([[1., 0., 0.]])
1 failed
```

修正后通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stage4_act_train.py -q
# 6 passed
```

相关回归：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stage4_act_train.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_toolbench_g3_unified_pipeline_sbatch_runs_import_build_and_readiness_gates \
  tests/test_sbatch_scripts.py::test_traject_eval_import_sbatch_uses_local_public_data_and_repo_cache \
  tests/test_sbatch_scripts.py::test_traject_retrieval_export_sbatch_exports_and_evaluates_traject_run -q
# 23 passed
```

静态检查：

```bash
python -m py_compile \
  clstr/stage4_act_train.py \
  tests/test_stage4_act_train.py \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/import_traject_eval.py \
  scripts/audit_traject_eval_data.py \
  clstr/traject_eval_import.py \
  clstr/traject_eval_audit.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  scripts/sbatch/run_import_traject_eval.sh \
  scripts/sbatch/run_export_traject_retrieval_run.sh
# exit 0
```

## 2026-05-26 Stage 4 默认接 Stage 3 HRPO checkpoint

### 问题

v4 训练链路应为：

```text
Stage 1 retrieval warmup
  -> Stage 2 supervised full-base
  -> Stage 3 HRPO / routing-policy update
  -> Stage 4 transition-conditioned next-skill L_act
```

审计时发现 `scripts/sbatch/run_clstr_unified_stage4_act_train.sh` 的默认
`HEAD_CHECKPOINT_PATH` 仍指向 Stage 2：

```text
outputs/clstr_unified_stage2_toolbench_g3_traject_split_full_base/checkpoints/clstr_full_base-step10000.pt
```

这会让 Stage 4 绕过 Stage 3 HRPO 的更新，削弱“先 policy/belief，再 L_act”的完整 CLSTR 论证链路。

### 修正

1. `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
   - 默认 `HEAD_CHECKPOINT_PATH` 改为 Stage 3 输出：

```text
outputs/clstr_unified_stage3_toolbench_g3_traject_split_hrpo/checkpoints/clstr_unified_stage3_hrpo-step500.pt
```

2. `scripts/audit_clstr_unified_training_readiness.py`
   - 新增 `stage3_checkpoint` / `--stage3_checkpoint`；
   - Stage 3 gate 仍检查 Stage 2 checkpoint；
   - Stage 4 gate 改为检查 Stage 3 checkpoint；
   - Stage 4 blocker 从含糊的 `missing_stage2_or_stage3_checkpoint` 改为明确的
     `missing_stage3_checkpoint`；
   - readiness 输出新增 `stage3_checkpoint`、`stage3_checkpoint_exists`，并让
     `head_checkpoint_exists` 表示 Stage 4 实际 head checkpoint 是否存在。

3. `scripts/sbatch/run_clstr_unified_readiness_audit.sh`
   - 新增 `STAGE3_CHECKPOINT` 默认值并传入 readiness audit。

4. `scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh`
   - 新增 `STAGE3_CHECKPOINT` 默认值并传入 readiness audit。

5. 测试
   - `tests/test_sbatch_scripts.py`
     - Stage 3 仍要求默认接 Stage 2；
     - Stage 4 要求默认接 Stage 3；
     - readiness/pipeline sbatch 要求包含 Stage 3 checkpoint。
   - `tests/test_unified_training_readiness.py`
     - 新增 `test_unified_training_readiness_stage4_requires_stage3_checkpoint`；
     - CLI 默认路径测试要求包含 Stage 3 checkpoint。

### TDD 证据

先改测试后，当前实现失败：

```text
TypeError: audit_unified_training_readiness() got an unexpected keyword argument 'stage3_checkpoint'
assert Stage 3 checkpoint path in Stage 4 sbatch/readiness/pipeline scripts
5 failed
```

实现修正后通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_stage4_requires_stage3_checkpoint \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_cli_defaults_use_toolbench_g3_stage_paths \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_readiness_sbatch_includes_traject_eval_gate \
  tests/test_sbatch_scripts.py::test_toolbench_g3_unified_pipeline_sbatch_runs_import_build_and_readiness_gates -q
# 5 passed
```

静态检查：

```bash
python -m py_compile scripts/audit_clstr_unified_training_readiness.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh
# exit 0
```

## 2026-05-26 split-safe v4 Stage 1-4 依赖链提交

### 数据准备状态

`77377` 已完成 split-safe unified v2 数据构建，产物：

```text
data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/
```

关键 manifest 摘要：

```text
manifest.status: ok
trajectory_stream.total_rows: 275271
trajectory_stream.traject_bench.total_rows: 30526
trajectory_stream.toolbench_g3.total_rows: 9570
retrieval_stream.total_pairs: 448665
retrieval_stream.traject_bench.total_pairs: 30526
skill_pool.total_skills: 37623
leakage_audit.status: ok
traject_split_skipped: dev=3945, test=3623
```

readiness 摘要：

```text
benchmark_filter.retained_train_trajectory_rows: 40096
benchmark_filter.stage4_next_skill_rows: 32518
stage1_retrieval_warmup.ready_to_submit: true
paper_readiness.full_three_benchmark_data_ready: true
```

### 已提交作业

使用 `gpu_a800`，按 `afterok` 依赖链提交：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh
# 77401

sbatch --dependency=afterok:77401 --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# 77411

sbatch --dependency=afterok:77411 --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage3_hrpo.sh
# 77412

sbatch --dependency=afterok:77412 --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage4_act_train.sh
# 77413
```

说明：

- Stage 2 等 Stage 1 成功后启动；
- Stage 3 等 Stage 2 成功后启动；
- Stage 4 等 Stage 3 成功后启动；
- 这样可以避免频繁人工轮询；
- Stage 4 默认 head checkpoint 已改为 Stage 3 HRPO 输出：

```text
outputs/clstr_unified_stage3_toolbench_g3_traject_split_hrpo/checkpoints/clstr_unified_stage3_hrpo-step500.pt
```

### 验证

提交依赖链前已通过相关回归：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_pretrain_v2.py \
  tests/test_traject_eval_import.py \
  tests/test_traject_split.py \
  tests/test_traject_sequence_proxy.py \
  tests/test_stage3_unified_hrpo.py \
  tests/test_stage4_act_train.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py -q
# 79 passed
```

静态检查：

```bash
python -m py_compile \
  scripts/audit_clstr_unified_training_readiness.py \
  scripts/import_traject_eval.py \
  scripts/audit_traject_eval_data.py \
  clstr/traject_eval_import.py \
  clstr/traject_eval_audit.py \
  clstr/traject_split.py \
  clstr/traject_sequence_proxy.py \
  clstr/stage3_unified_hrpo.py \
  clstr/stage4_act_train.py \
  tests/test_stage4_act_train.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  scripts/sbatch/run_import_traject_eval.sh \
  scripts/sbatch/run_export_traject_retrieval_run.sh \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_export_clstr_retrieval_run.sh \
  scripts/sbatch/run_toolbench_g3_routing_eval.sh
# exit 0
```

## 2026-05-26 Stage 1 后 routing eval 依赖作业提交

### 当前队列状态

一次性检查队列时看到：

```text
77384 PENDING (Priority)    # TRAJECT held-out test import
77401 PENDING (Priority)    # Stage 1 retrieval warmup
77411 PENDING (Dependency)  # Stage 2 afterok:77401
77412 PENDING (Dependency)  # Stage 3 afterok:77411
77413 PENDING (Dependency)  # Stage 4 afterok:77412
```

尚未产生 Stage 1-4 checkpoint 产物；训练链路还在等待资源。

### 新增依赖评估作业

为避免频繁人工轮询，在 Stage 1 checkpoint 完成后自动跑 routing eval：

```bash
sbatch --dependency=afterok:77401 --gpus=1 -p gpu_a800 \
  --export=ALL,OUTPUT_DIR=outputs/toolret_eval/clstr_unified_stage1_traject_split_retrieval \
  scripts/sbatch/run_export_clstr_retrieval_run.sh
# 77414
```

```bash
sbatch --dependency=afterok:77401 --gpus=1 -p gpu_a800 \
  --export=ALL,OUTPUT_DIR=outputs/toolbench_g3/clstr_unified_stage1_traject_split_routing_eval \
  scripts/sbatch/run_toolbench_g3_routing_eval.sh
# 77415
```

```bash
sbatch --dependency=afterok:77401:77384 --gpus=1 -p gpu_a800 \
  --export=ALL,OUTPUT_DIR=outputs/traject_eval_traject_split_test/clstr_unified_stage1_retrieval \
  scripts/sbatch/run_export_traject_retrieval_run.sh
# 77416
```

说明：

- `77414` / `77415` 只依赖 Stage 1 checkpoint；
- `77416` 同时依赖 Stage 1 和 TRAJECT held-out test import，避免新 eval 目录未生成时误跑；
- 这些 eval 都是 routing/retrieval 层，不声称完成官方 TRAJECT Usage / Traj-Satisfy / Acc 或 ToolBench official SoPR pass rate。

## 2026-05-26 StableToolBench CLSTR-routed SoPR 输入链路修补

### 问题

审计发现已有 StableToolBench raw generation sbatch 默认直接读取官方
`StableToolBench/solvable_queries/test_instruction/G3_instruction.json`，即使用官方
full `api_list`。这只能说明 StableToolBench 基础设施可运行，不能说明 CLSTR routing
预测进入了 ToolBench-G3 SoPR 链路。

### 已完成的主要改动

1. 新增 `clstr/stabletoolbench_clstr_queries.py`
   - 支持把 StableToolBench solvable query 导出为 CLSTR retrieval query JSONL；
   - 支持读取 CLSTR `run.tsv` 和 `data/toolbench_g3/skills.jsonl`，生成
     CLSTR top-K API 列表版 `G3_instruction.json`；
   - report 显式记录：
     - `query_count`
     - `retained_query_count`
     - `missing_run_query_count`
     - `empty_api_list_query_count`
     - `missing_skill_id_count`
     - `average_api_list_size`
   - `query_id` 兼容 `455` 和 `toolbench-g3-455`，避免 run/query id 前缀差异导致误判。

2. 新增 CLI / sbatch
   - `scripts/build_stabletoolbench_clstr_queries.py`
   - `scripts/sbatch/run_build_stabletoolbench_clstr_queries.sh`
   - 默认输入：
     - original query: `StableToolBench/solvable_queries/test_instruction/${TEST_SET}.json`
     - skills: `data/toolbench_g3/skills.jsonl`
     - run: `outputs/toolbench_g3/stabletoolbench_solvable_clstr_retrieval/run.tsv`
   - 默认输出：
     - `outputs/toolbench_g3/stabletoolbench_clstr_topk/${TEST_SET}.json`
     - `outputs/toolbench_g3/stabletoolbench_clstr_topk/${TEST_SET}_build_report.json`

3. raw generation audit/sbatch 已支持 derived query file
   - `scripts/audit_stabletoolbench_raw_generation.py` 新增 `--input_query_file`；
   - `scripts/sbatch/run_stabletoolbench_raw_generation.sh` 和
     `scripts/sbatch/run_stabletoolbench_local_raw_generation.sh` 新增 `INPUT_QUERY_FILE`；
   - audit report 新增：
     - `official_input_query_file`
     - `input_query_source`
     - `uses_official_full_api_list`

### 后续使用注意

CLSTR-routed ToolBench-G3 SoPR 需要按以下顺序运行：

```bash
# 1. 用 StableToolBench solvable query 导出 CLSTR retrieval queries
python scripts/build_stabletoolbench_clstr_queries.py \
  --export_retrieval_queries \
  --query_file /data/home/scyb713/run/xzf/AAAI/autodl-tmp/StableToolBench/solvable_queries/test_instruction/G3_instruction.json \
  --output_path data/stabletoolbench_g3_solvable_queries.jsonl

# 2. 对 data/stabletoolbench_g3_solvable_queries.jsonl 导出 CLSTR run.tsv
#    该步需用 sbatch 跑 export_clstr_retrieval_run.py，不能在登录节点直接跑模型。

# 3. 用 CLSTR run.tsv 生成 derived G3_instruction.json
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_build_stabletoolbench_clstr_queries.sh

# 4. 由 derived query file 重建 StableToolBench toolenv/tools
#    否则 CLSTR 预测出官方 full api_list 外的工具时，executor 可能找不到工具 JSON。

# 5. raw generation 必须显式传入 derived query file
sbatch --gpus=1 -p gpu_a800 \
  --export=ALL,INPUT_QUERY_FILE=outputs/toolbench_g3/stabletoolbench_clstr_topk/G3_instruction.json \
  scripts/sbatch/run_stabletoolbench_local_raw_generation.sh
```

只有第 3-5 步后再跑 answer conversion + official SoPR，才能报告
“CLSTR-routed ToolBench-G3 StableToolBench SoPR”。不能把当前 static routing recall
或官方 full `api_list` raw generation 结果写成 CLSTR SoPR。

### 验证

已通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stabletoolbench_clstr_queries.py \
  tests/test_stabletoolbench_raw_generation.py \
  tests/test_sbatch_scripts.py::test_stabletoolbench_raw_generation_sbatch_audits_before_generation \
  tests/test_sbatch_scripts.py::test_stabletoolbench_local_raw_generation_sbatch_runs_server_and_generation_on_same_node \
  tests/test_sbatch_scripts.py::test_stabletoolbench_clstr_query_build_sbatch_uses_clstr_run_and_skills -q
# 11 passed
```

静态检查：

```bash
python -m py_compile \
  clstr/stabletoolbench_clstr_queries.py \
  clstr/stabletoolbench_raw_generation.py \
  scripts/build_stabletoolbench_clstr_queries.py \
  scripts/audit_stabletoolbench_raw_generation.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_build_stabletoolbench_clstr_queries.sh \
  scripts/sbatch/run_stabletoolbench_raw_generation.sh \
  scripts/sbatch/run_stabletoolbench_local_raw_generation.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/stabletoolbench_clstr_queries.py \
  clstr/stabletoolbench_raw_generation.py \
  scripts/build_stabletoolbench_clstr_queries.py \
  scripts/audit_stabletoolbench_raw_generation.py \
  scripts/sbatch/run_build_stabletoolbench_clstr_queries.sh \
  scripts/sbatch/run_stabletoolbench_raw_generation.sh \
  scripts/sbatch/run_stabletoolbench_local_raw_generation.sh \
  tests/test_stabletoolbench_clstr_queries.py \
  tests/test_stabletoolbench_raw_generation.py \
  tests/test_sbatch_scripts.py
# no output
```

## 2026-05-26 Stage2-4 训练可观测性与提交纪律修正

### 背景

用户指出：Stage2-4 代码尚未补完时不应提前提交完整训练链。该批依赖作业
`77446`、`77447`、`77448` 已取消；Stage1 `77445` 保留，因为它对应
retrieval warmup 路径已单独通过聚焦测试，且不依赖这次 Stage2-4 未完成改动。
后续 Stage2-4 训练链必须在代码、测试、静态检查和文档记录完成后再重新提交。

### 已完成的主要改动

1. Stage2/3/4 core train 均补齐 `setup_status.jsonl`
   - Stage2 `clstr/full_base_train.py` 记录：
     - `data_loaded`
     - `rows_filtered`
     - `model_moved_to_device`
     - `skill_table_rebuild_started` / `skill_table_rebuilt` / `skill_table_rebuild_skipped`
     - `routing_checkpoint_loaded`
     - `embedding_cache_started`
     - `embedding_cache_prepared`
     - `training_started`
   - Stage3 `clstr/stage3_unified_hrpo.py` 记录：
     - `stage3_rows_built`
     - `model_moved_to_device`
     - `stage3_freeze_applied`
     - `training_started`
     - blocked 时写 `stage3_blocked`
   - Stage4 `clstr/stage4_act_train.py` 记录：
     - `skills_loaded`
     - `stage4_rows_built`
     - `model_moved_to_device`
     - `stage4_freeze_applied`
     - `training_started`
     - blocked 时写 `stage4_blocked`

2. Stage2/3/4 checkpoint/report 均暴露可观测路径
   - `train_report.json` / `train_stdout.json` 新增 `setup_status_path`；
   - final checkpoint payload 新增 `setup_status_path`；
   - rolling `checkpoints/latest.pt` 初始 step 0 和每次 step/update 覆盖写入时均携带
     `setup_status_path`；
   - 继续保留 `training_metrics.jsonl`、`loss_curve.svg`、`checkpoints/latest.pt`。

3. Qwen Stage2/3/4 wrapper 补齐加载前状态链路
   - `clstr/qwen_full_base_train.py`
   - `clstr/qwen_stage3_unified_hrpo.py`
   - `clstr/qwen_stage4_act_train.py`
   - wrapper 在加载 Qwen 前先 reset 并写同一个 `setup_status.jsonl`；
   - checkpoint preflight 失败时不会加载 Qwen，并写：
     - `qwen_stage2_blocked`
     - `qwen_stage3_blocked`
     - `qwen_stage4_blocked`
   - 成功路径记录：
     - `qwen_stage*_started`
     - `qwen_stage*_checkpoint_preflight_ok`
     - `qwen_model_load_started`
     - `qwen_model_loaded`
     - Stage3/4 额外记录 `stage3_checkpoint_init_loaded` /
       `stage4_checkpoint_init_loaded`
   - wrapper 将同一个 `setup_status_path` 传入 core train，core 不再重置 wrapper
     早期状态。

### 验证

聚焦测试已通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_full_base_train.py::test_train_clstr_full_base_with_model_respects_loss_masks_and_writes_checkpoint \
  tests/test_stage3_unified_hrpo.py::test_train_unified_hrpo_with_model_writes_auditable_checkpoint \
  tests/test_stage4_act_train.py::test_train_stage4_act_with_model_writes_checkpoint_and_manifest \
  tests/test_qwen_external_encoder.py::test_qwen_full_base_train_forwards_stage1_routing_checkpoint \
  tests/test_qwen_external_encoder.py::test_qwen_full_base_train_blocks_before_loading_qwen_when_stage1_checkpoint_missing \
  tests/test_qwen_external_encoder.py::test_qwen_stage4_act_train_blocks_before_loading_qwen_when_checkpoint_missing \
  tests/test_qwen_external_encoder.py::test_qwen_stage4_act_train_loads_stage1_and_head_checkpoints \
  tests/test_qwen_external_encoder.py::test_qwen_unified_stage3_hrpo_blocks_before_loading_qwen_when_checkpoint_missing \
  tests/test_qwen_external_encoder.py::test_qwen_unified_stage3_hrpo_loads_stage1_and_stage2_checkpoints \
  --basetemp=.tmp_pytest -q
# 9 passed, 2 warnings
```

静态检查已通过：

```bash
python -m py_compile \
  clstr/training_monitor.py \
  clstr/full_base_train.py \
  clstr/stage3_unified_hrpo.py \
  clstr/stage4_act_train.py \
  clstr/qwen_full_base_train.py \
  clstr/qwen_stage3_unified_hrpo.py \
  clstr/qwen_stage4_act_train.py
# exit 0
```

```bash
bash -n \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_stabletoolbench_clstr_routed_local_raw_generation.sh
# exit 0
```

```bash
git diff --check -- \
  clstr/training_monitor.py \
  clstr/full_base_train.py \
  clstr/stage3_unified_hrpo.py \
  clstr/stage4_act_train.py \
  clstr/qwen_full_base_train.py \
  clstr/qwen_stage3_unified_hrpo.py \
  clstr/qwen_stage4_act_train.py \
  tests/test_full_base_train.py \
  tests/test_stage3_unified_hrpo.py \
  tests/test_stage4_act_train.py \
  tests/test_qwen_external_encoder.py \
  scripts/sbatch/run_stabletoolbench_clstr_routed_local_raw_generation.sh \
  description2.md
# no output
```

## 2026-05-26 Stage1 skill table rebuild 进度可观测性补丁

### 背景

Stage1 `77445` 运行期间，`setup_status.jsonl` 已写到
`skill_table_rebuild_started`，但 `training_metrics.jsonl` 仍为空、`loss_curve.svg`
仍是占位图。这说明训练 loop 尚未开始，时间主要花在 37623 个 skill 的 frozen
Qwen embedding table rebuild 上。此前该 rebuild 阶段只有 started/rebuilt 两个事件，
中间没有批次级进度，因此长时间看不到 loss 或 setup 变化。

### 已完成的主要改动

1. `clstr/encoders.py`
   - `SkillTable` 新增可选参数 `embedding_progress_callback`；
   - `build_embeddings()` 每完成一个 embedding batch 后回调：
     - `batch_index`
     - `batch_count`
     - `start`
     - `end`
     - `total`
   - 默认值为 `None`，不改变任何已有 embedding 数值、参数初始化或训练目标。

2. `clstr/retrieval_warmup.py`
   - Stage1 在 `model.rebuild_skill_table()` 前临时挂载 progress callback；
   - 每个 batch 写入 `setup_status.jsonl`：
     - `phase=skill_table_rebuild_progress`
     - `encoded_skill_count`
     - `total_skill_count`
     - `batch_index`
     - `batch_count`
   - rebuild 结束后恢复原 callback，避免影响后续训练逻辑。

### 重要说明

该补丁不影响已经启动的 `77445` 进程；它只会影响后续重新启动的 Stage1 /
retrieval warmup 作业。当前正在运行的 Stage1 若仍使用旧进程代码，setup 文件仍可能在
rebuild 完成前只显示 `skill_table_rebuild_started`。

### 验证

聚焦测试已通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_v4_skill_table_freeze.py::test_skill_table_reports_embedding_build_progress_batches \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  --basetemp=.tmp_pytest -q
# 2 passed, 2 warnings
```

## 2026-05-26 Phase A 本地实现层验收

### 背景

在等待 Stage1 完整 checkpoint 期间，先复核 v4 Phase A 中不依赖训练产物的实现层 gate：
F1/F2-v2/G1/G2、belief residual、action_proj sharing、candidate sampling、
cross-encoder cache、SkillTable freeze 等本地单元测试必须先稳定，避免后续训练把实现错误
误判为方法性能问题。

### 验证

已通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_v4_*.py \
  --basetemp=.tmp_pytest -q
# 25 passed, 2 warnings
```

该结果只证明 v4 Phase A 的本地实现层测试通过，不代表 Stage1/2/3/4 完整训练已完成，
也不代表 TRAJECT / ToolBench-G3 / ToolRet 主表指标已经达标。

## 2026-05-26 v4 数据与评估工具 readiness 本地验收

### 背景

Stage1 已经进入训练 loop，`training_metrics.jsonl` 与 `loss_curve.svg` 开始更新；
但最终 Stage1 checkpoint 尚未产出，因此仍不提交 Stage2-4 训练链。等待期间先验证不依赖
训练产物的 unified v2 数据、Stage2 preflight、TRAJECT / ToolBench-G3 /
StableToolBench 转换与评估工具。

### 验证

已通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_training_readiness.py \
  tests/test_stage2_preflight.py \
  tests/test_unified_pretrain_v2.py \
  tests/test_traject_split.py \
  tests/test_traject_eval_import.py \
  tests/test_traject_sequence_proxy.py \
  tests/test_toolbench_g3_routing_eval.py \
  tests/test_generic_retrieval_eval.py \
  tests/test_stabletoolbench_clstr_queries.py \
  tests/test_stabletoolbench_toolenv.py \
  --basetemp=.tmp_pytest -q
# 58 passed
```

该结果只证明数据转换、preflight、routing/proxy eval 与 StableToolBench derived query/toolenv
工具链的本地行为通过测试；不代表 Stage2-4 训练已经完成，也不能替代官方 task success /
SoPR / TRAJECT 原生指标。

## 2026-05-26 sbatch 与 StableToolBench 包装层本地验收

### 背景

在重新提交 Stage2-4 训练链前，先验证 sbatch 包装脚本和 StableToolBench 执行链的本地
行为，避免后续因为脚本默认路径、derived query/toolenv、raw generation readiness gate
或 answer conversion/pass-rate 包装问题浪费 GPU 队列时间。

### 验证

已通过：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py \
  tests/test_stabletoolbench_raw_generation.py \
  tests/test_stabletoolbench_answer_conversion.py \
  tests/test_stabletoolbench_pass_rate.py \
  tests/test_stabletoolbench_virtual_api.py \
  tests/test_stabletoolbench_raw_generation.py \
  --basetemp=.tmp_pytest -q
# 47 passed
```

该结果只证明脚本包装和本地 readiness/format conversion 行为通过测试；真正的
StableToolBench SoPR 仍需要 CLSTR-routed raw executor answers、answer conversion 和
official pass-rate / judge 链路全部完成后才能报告。

## 2026-05-26 v4 Python / sbatch 全量语法 gate

### 背景

在 Stage1 最终 checkpoint 出现前，不提交 Stage2-4 训练链；等待期间先做更宽的本地
语法 gate，降低后续 sbatch 作业因为 Python 语法错误或 shell 语法错误而浪费队列时间的风险。

### 验证

已通过：

```bash
python -m compileall -q clstr scripts
# exit 0
```

```bash
find scripts/sbatch -name '*.sh' -exec bash -n {} \;
# exit 0
```

该结果只证明 Python/sbatch 语法层通过，不代表训练、评估或官方 benchmark 指标完成。

## 2026-05-26 当前 readiness audit 与 Stage2-4 提交前置条件

### 背景

为避免再次在上游产物未满足时提交下游训练链，本次运行当前统一训练 readiness audit，
只生成本地报告，不提交 sbatch 作业。

### 命令

```bash
python scripts/audit_clstr_unified_training_readiness.py \
  --data_root data/clstr_unified_pretrain_v2_toolbench_g3_traject_split \
  --allowed_benchmarks traject_bench,toolbench_g3 \
  --stage1_checkpoint outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt \
  --stage2_checkpoint outputs/clstr_unified_stage2_toolbench_g3_traject_split_full_base/checkpoints/clstr_full_base-step10000.pt \
  --stage3_checkpoint outputs/clstr_unified_stage3_toolbench_g3_traject_split_hrpo/checkpoints/clstr_unified_stage3_hrpo-step500.pt \
  --toolret_eval_dir data/toolret_eval \
  --traject_eval_dir data/traject_eval_traject_split_test \
  --toolbench_g3_data_dir data/toolbench_g3 \
  --toolbench_g3_source_root ../ToolBench/data \
  --expected_toolbench_g3_answer_files 5000 \
  --output_path outputs/clstr_unified_readiness_audit/readiness_report_toolbench_g3_traject_split_current.json
# status: action_required
```

### 当前结论

数据与评估侧 readiness：
- `manifest.status = ok`
- `leakage_audit.status = ok`
- `skill_pool_quality.status = ok`
- `paper_readiness.full_three_benchmark_data_ready = true`
- `toolret_eval_ready = true`
- `traject_eval_ready = true`
- `toolbench_g3_data_ready = true`

Stage gate：
- Stage1 retrieval warmup: `ready_to_submit = true`
- Stage2 full base: `ready_to_submit = false`
  - blocker: `missing_stage1_checkpoint`
- Stage3 offline HRPO: `ready_to_submit = false`
  - blockers: `missing_stage1_checkpoint`, `missing_stage2_checkpoint`
- Stage4 ACT: `ready_to_submit = false`
  - blockers: `missing_stage1_checkpoint`, `missing_stage3_checkpoint`

### Stage2-4 重新提交前置条件

重新提交训练链前必须同时满足：

1. Stage1 最终 checkpoint 存在：
   `outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
2. Stage2 preflight 通过，且报告写入：
   `outputs/clstr_unified_stage2_toolbench_g3_traject_split_full_base/stage2_preflight.json`
3. Stage2 完成后 checkpoint 存在：
   `outputs/clstr_unified_stage2_toolbench_g3_traject_split_full_base/checkpoints/clstr_full_base-step10000.pt`
4. Stage3 完成后 checkpoint 存在：
   `outputs/clstr_unified_stage3_toolbench_g3_traject_split_hrpo/checkpoints/clstr_unified_stage3_hrpo-step500.pt`

在第 1 条满足前，不再提交 Stage2/3/4；在第 3 条满足前，不提交 Stage3；
在第 4 条满足前，不提交 Stage4。

### 后续重新提交流程

Stage1 完成后按以下顺序执行，不能跳步：

1. 检查 Stage1 最终 checkpoint 是否存在：

```bash
test -s outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt
```

2. 运行 Stage1 quality gate，确认这次 retrieval warmup 不是退化训练：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/audit_clstr_stage1_quality.py \
  --output_dir outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup \
  --checkpoint_path outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt \
  --output_path outputs/clstr_unified_stage2_toolbench_g3_traject_split_full_base/stage1_quality_gate.json \
  --fail_on_action_required
```

该 gate 检查：

- 最终 checkpoint 存在；
- checkpoint 记录 `sampling_strategy=batch_stride` 与 `shuffle_queries=true`；
- `training_metrics.jsonl` 有足够 step；
- first/last window loss 有下降；
- last-window recall 不是退化为 0；
- 每步 `query_sources` 覆盖 SKILLRET / ToolRet / ToolBench-G3 / TRAJECT。

该 gate 只是 Stage1→Stage2 工程质量门，不能替代正式 benchmark。

3. 运行 Stage2 preflight：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_clstr_stage2_preflight.py \
  --checkpoint_path outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt \
  --skills_path data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/skill_pool.jsonl \
  --model_dim 1024 \
  --output_path outputs/clstr_unified_stage2_toolbench_g3_traject_split_full_base/stage2_preflight.json
```

4. 只有第 1-3 步都通过后，提交 Stage2：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
```

5. Stage2 结束并确认 `clstr_full_base-step10000.pt` 存在后，提交 Stage3：

```bash
sbatch --gpus=1 -p gpu_a800 \
  --dependency=afterok:<STAGE2_JOB_ID> \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh
```

6. Stage3 结束并确认 `clstr_unified_stage3_hrpo-step500.pt` 存在后，提交 Stage4：

```bash
sbatch --gpus=1 -p gpu_a800 \
  --dependency=afterok:<STAGE3_JOB_ID> \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh
```

若 H200 空闲且预算允许，可把 `-p gpu_a800` 改为 `-p gpu_h200`；但每次提交前仍必须先确认
上游 checkpoint 文件存在且 readiness/preflight gate 通过。训练期间按用户要求低频查看：
短任务约 15 分钟一次，长任务约 30 分钟一次，不做高频轮询。

## 2026-05-26 16:35 — Stage1 loss 异常诊断与修复

用户查看 Stage1 `loss_curve.svg` 后指出 loss 一直跳动。重新核验后结论如下：

1. 单步 loss 抖动本身正常，因为 Stage1 batch size 只有 16，且每个 query 的难度不同。
2. 当前异常不只是抖动，而是 rolling mean 基本不下降：
   - 约 4000 step 时 loss 仍在 10.2 左右；
   - 37623 个 skill 的随机 full-pool CE 约为 `log(37623)=10.535`；
   - `recall_at_50` 后段窗口接近 0。
3. 定位到两个实现/配置问题：
   - `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh` 默认使用 `models/Qwen3-8B`，偏离 v4 plan 中的 `SkillRouter-Emb-0.6B` 初始化；
   - `clstr/retrieval_warmup.py` 训练 batch 使用滑动一行采样，5000 step、batch size 16 时只覆盖约 5015 个连续 query，并且由于 unified retrieval 文件按 source 排序，实际主要训练到了文件头部 SKILLRET，没覆盖 ToolRet/ToolBench-G3/TRAJECT 的全 corpus。

处理：

- 取消异常 Stage1 作业 `77445` 及其依赖导出作业 `77449-77452`；
- 将异常输出归档到：
  `outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup_bad_qwen_sliding_20260526_1628/`
- 修复 Stage1 warmup：
  - query 默认 deterministic shuffle；
  - batch 采样改为 batch-stride，避免只滑动一行；
  - `training_metrics.jsonl` 增加每步 `query_ids` / `query_sources`，便于检查 source 覆盖；
  - report/checkpoint 记录 `sampling_strategy=batch_stride` 与 `shuffle_queries`；
  - CLI 增加 `--no_shuffle_queries` 便于测试；
  - sbatch 默认 backbone 改为本地 `.cache/hf_models/SkillRouter-Embedding-0.6B`。

验证：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_skillret.py::test_retrieval_warmup_batches_advance_by_batch_stride_for_full_corpus_coverage \
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream \
  tests/test_sbatch_scripts.py::test_unified_stage1_retrieval_warmup_sbatch_uses_unified_v2_data_and_local_cache \
  -q --basetemp=.tmp_pytest
# 3 passed

bash -n scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh
python -m py_compile clstr/retrieval_warmup.py scripts/run_skillret_retrieval_warmup.py
```

结论：之前的 Stage1 不应作为 Stage2 前置；必须用修复后的 Stage1 重新训练，等新的最终 checkpoint 存在后再运行 Stage2 preflight。

## 2026-05-26 16:55 — Stage1→Stage2 quality gate 接入

为避免退化 Stage1 checkpoint 再次流入 Stage2，本轮新增 Stage1 quality gate：

- 新增 `clstr/stage1_quality_gate.py`
- 新增 CLI `scripts/audit_clstr_stage1_quality.py`
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh` 现在会先运行：

```bash
scripts/audit_clstr_stage1_quality.py --fail_on_action_required
```

只有 Stage1 gate `status=ok` 后，才继续运行 `scripts/run_clstr_stage2_preflight.py`
和 Stage2 训练。gate 检查内容包括：

- 最终 Stage1 checkpoint 是否存在；
- checkpoint stage 是否为 `clstr_unified_retrieval_v2`；
- checkpoint 是否记录 `sampling_strategy=batch_stride` 与 `shuffle_queries=true`；
- `training_metrics.jsonl` 是否达到最小 step；
- first/last window loss 是否有下降；
- last-window recall 是否超过退化阈值；
- `query_sources` 是否覆盖 `skillret`、`toolret_training`、`toolbench_g3`、`traject_bench`。

验证：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_stage1_quality_gate.py \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  -q --basetemp=.tmp_pytest
# 4 passed

python -m py_compile clstr/stage1_quality_gate.py scripts/audit_clstr_stage1_quality.py
bash -n scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
```

注意：该 gate 是训练质量门，不是论文主表指标；正式 SKILLRET / TRAJECT / ToolRet
评测仍需在 Stage1/Stage2/Stage3 完成后单独跑。

## 2026-05-26 17:10 — readiness audit 接入 Stage1 quality gate

为保证总 readiness 审计和 Stage2 sbatch 使用同一套前置条件，本轮把 Stage1
quality gate 接入 `scripts/audit_clstr_unified_training_readiness.py`：

- `audit_unified_training_readiness(...)` 新增 `stage1_output_dir` 参数；
- CLI 新增 `--stage1_output_dir`，默认指向
  `outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup`；
- `scripts/sbatch/run_clstr_unified_readiness_audit.sh` 新增 `STAGE1_OUTPUT_DIR` 并传给审计脚本；
- 当 Stage1 checkpoint 存在时，readiness audit 会运行 `clstr.stage1_quality_gate`；
- 若 Stage1 quality gate 不是 `status=ok`，则 Stage2/Stage3/Stage4 gate 都加入
  `stage1_quality_gate_not_ok` blocker；
- 若 Stage1 checkpoint 缺失，readiness audit 仍保持轻量路径，不强制 import torch-heavy
  quality gate，保留缺 checkpoint 情况下的快速审计能力。

验证：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_unified_training_readiness.py -q --basetemp=.tmp_pytest
# 15 passed

python -m py_compile scripts/audit_clstr_unified_training_readiness.py
bash -n scripts/sbatch/run_clstr_unified_readiness_audit.sh
git diff --check -- scripts/audit_clstr_unified_training_readiness.py \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh \
  tests/test_unified_training_readiness.py
```

当前修复后的 Stage1 作业 `77497` 已在运行，早期 metrics 已显示 `query_sources`
覆盖 `skillret/toolret_training/toolbench_g3/traject_bench`，且 `recall_at_50` 已非退化。
仍需等待最终 checkpoint，再运行 Stage1 quality gate 与 Stage2 preflight；本轮没有提交 Stage2。

## 2026-05-26 17:25 — Stage1 完成后安全提交 Stage2 入口

为减少人工提交时绕过 gate 的风险，本轮新增一个只负责 Stage1→Stage2 的安全提交入口：

```bash
bash scripts/submit_clstr_stage2_after_stage1_gate.sh
```

该脚本做且只做以下步骤：

1. 检查 `STAGE1_CHECKPOINT` 是否存在且非空；
2. 运行 `scripts/audit_clstr_stage1_quality.py --fail_on_action_required`；
3. 运行 `scripts/run_clstr_stage2_preflight.py`；
4. 只有上述步骤都通过后，执行：

```bash
sbatch --gpus="${GPUS}" -p "${PARTITION}" \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
```

默认参数：

- `PARTITION=gpu_a800`
- `GPUS=1`
- `STAGE1_OUTPUT_DIR=outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup`
- `STAGE1_CHECKPOINT=${STAGE1_OUTPUT_DIR}/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
- `STAGE2_OUTPUT_DIR=outputs/clstr_unified_stage2_toolbench_g3_traject_split_full_base`

该脚本**不会提交 Stage3/Stage4**。Stage3 仍需等 Stage2 最终 checkpoint
`clstr_full_base-step10000.pt` 存在后再单独提交。

验证：

```bash
conda run -n reasoning_trap env PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py::test_submit_stage2_after_stage1_gate_checks_gates_and_only_submits_stage2 \
  -q --basetemp=.tmp_pytest
# 1 passed

bash -n scripts/submit_clstr_stage2_after_stage1_gate.sh
git diff --check -- scripts/submit_clstr_stage2_after_stage1_gate.sh tests/test_sbatch_scripts.py
```

注意：当前 Stage1 `77497` 尚未产出最终 checkpoint，因此现在不能运行该提交入口。
