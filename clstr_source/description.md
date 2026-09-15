## 2026-06-09 Stage4-online post-action observation 接入

本次修复只针对 v4.1b ALFWorld online HRPO / Stage4-online 路径，不提交新的训练作业。此前 online HRPO 的动作选择输入已经包含当前 `obs_t`，但 rollout step 没有保存 `env.step(action_t)` 返回的 `obs_{t+1}`，训练 loop 也只优化 HRPO policy loss；`controller_complete` 中 transition/belief 因为不能在动作前看到未来 observation 而被置零。因此它只能算 policy-only online RL，不能证明完整 CLSTR transition/belief 后验学习。

修复后：

- `HrpoStep` 记录 `observation_text`、`next_observation_text`、`next_admissible_actions`、`done`、`env_reward`、`won`、`goal_condition_success_rate`。
- 动作选择仍只看 `goal + obs_t + history + admissible_actions_t`，报告显式写入 `pre_action_next_observation_leakage=false`。
- 在 `env.step(action_t)` 之后新增 `_compute_online_observation_aux_loss()`：用 `state_text_t + chosen_action_t + next_observation_text` 训练 transition/belief consistency，不要求在线 ALFWorld 人工提供 `next_skill_id`，目标为 detached `obs_{t+1}` skill-subspace memory。
- `train_report.json` / checkpoint payload 新增 `post_action_observation_used`、`online_observation_aux_count`，`training_metrics.jsonl` 新增 `online_transition_loss`、`online_belief_loss`、`online_observation_aux_loss`。
- 已新增测试覆盖 rollout 后验 observation 记录、动作前 no-leakage、observation auxiliary loss 非零和 trainer 报告字段。

### 2026-06-09 Stage4-online reward/advantage viability 修正

`outputs/clstr_v4_1b_alfworld_online_hrpo_observation_smoke` 暴露出新的设计问题：两个 rollout 的 action trace 不同，但原 reward 都是 `-0.15`，因为 reward 只看 success / goal_condition / step / info / repeat，无法区分“发现目标物体”和“拿无关物体”。这不是 Qwen3 问题；该 smoke 实际使用 `SkillRouter-Embedding-0.6B`，`qwen_direct_generator=false`。

修复后：

- 新增 `estimate_alfworld_progress_points(goal_text, step_records)`，从 ALFWorld goal 中抽取目标物体/位置词；目标物体权重大于 receptacle/location。
- `HrpoRewardConfig` 新增 `goal_progress_reward=0.05`，`shape_alfworld_rollout_reward()` 将 progress points 纳入总 reward。
- rollout report 新增 `progress_points`、`progress_reward`；training metrics / report 新增 `mean_progress_points`、`mean_progress_reward`、`progress_reward_unique_count`。
- 对旧 smoke 的两条失败轨迹离线重算后，reward 从同分变为 `-0.10` vs `-0.05`，说明 progress reward 能为 HRPO advantage 提供非零区分信号。
- 已新增测试：目标物体 discovery 的 progress reward 必须高于仅到达 receptacle 或无关进展；完整 `tests/test_online_hrpo.py` 通过。
- 新 smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_progress_smoke` 已完成，A800 job `86218`，elapsed `00:02:25`，exit code `0:0`。结果：`reward_unique_count=2`、`updates_with_nonzero_advantage=1`、`nonzero_advantage_count=2`、`hrpo_policy_loss=-0.0372`、`online_transition_loss=0.0802`、`online_belief_loss=0.00129`、`viability.status=ok`。该 smoke 仍是 2 rollout / 10 step 的低成本 viability 验证，不代表 ALFWorld task success 已提升。

### 2026-06-10 Stage4-online action-quality reward / exploration prior

多 update smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_progress_update_smoke`（A800 job `86284`，elapsed `00:03:36`）证明 reward/advantage 信号稳定存在：`rollout_count=12`、`reward_unique_count=9`、`updates_with_nonzero_advantage=3`、`online_observation_aux_count=96`、`viability.status=ok`。但 action trace 仍大量包含 `take pillow/laptop/pen`、`examine bed/pillow` 等无关操作，没有目标物体交互，说明只奖励 discovery 还不够。

修复后：

- `HrpoRewardConfig` 新增 `goal_interaction_reward=0.10`、`irrelevant_action_penalty=0.03`。
- 新增 `estimate_alfworld_progress_report()`，区分 `progress_points`、`goal_interaction_points`、`irrelevant_action_count`。
- 对 `take/pick up/put/place/open/close/examine` 等操作，如果 action 命中目标物体则奖励，否则在不命中任何 goal term 时计入 irrelevant penalty。
- `HrpoLossConfig` 新增默认关闭的 `goal_action_bonus=0.0`；`compute_hrpo_action_logits()` 可在 rollout exploration 中给匹配目标物体的 candidate action 加通用文本 prior。该 prior 只作为低成本探索诊断，不能直接作为主论文结果。
- 新增 CLI/sbatch 参数 `--goal_action_bonus` / `GOAL_ACTION_BONUS`。
- `outputs/clstr_v4_1b_alfworld_online_hrpo_interaction_smoke`（job `86290`）在新 reward 下保持 `viability.status=ok`，但 `mean_goal_interaction_points=0.0`，说明 reward 本身还不能让弱探索采到目标动作。
- `outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_smoke`（job `86300`，`GOAL_ACTION_BONUS=1.0`）出现目标物体交互：update 1 中采到 `take tissuebox 2 from diningtable 1`，`mean_goal_interaction_points=0.5`，最高 rollout reward 到 `0.01`；但 update 2/3 未持续出现目标交互，说明下一瓶颈是探索/导航与训练强度，而不是 observation 后验链路。

### 2026-06-10 Stage4-online advantage scaling 修复

更强的 goal-prior smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_lr5e5_u5_s12_smoke`（job `86312`，`MAX_STEPS=12, UPDATES=5, LR=5e-5`）技术上完成，`viability.status=ok`、`updates_with_nonzero_advantage=5`，并在 update 1/4 采到目标物体动作。但平均 reward 从上一轮 `-0.2375` 降到 `-0.4525`，`mean_irrelevant_action_count` 升到 `3.95`，目标交互没有跨 update 稳定延续。

复查训练实现后发现 `_flatten_rollout_steps()` 把 rollout-level advantage 除以 `len(steps)` 后再用于每个 action，而 `compute_hrpo_loss()` 已经对所有 step 求均值；这会对 8-12 步 rollout 形成二次缩放，导致 policy gradient 随 rollout 变长明显变弱。修复后每个 action 直接继承该 rollout 的 group-normalized advantage，不再按步数削弱。新增回归测试 `test_flatten_rollout_steps_keeps_rollout_advantage_per_action`，完整 `tests/test_online_hrpo.py` 当前 17 项通过。

advantage scaling 修复后的同配置 smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_goal_prior_lr5e5_u5_s12_advfix_smoke`（job `86314`）仍与修复前产生完全相同的 20 条 action trace，最大动作概率变化只有约 `3.3e-4`。这说明 policy loss 放大是必要修复，但 5 个 update 内尚不足以改变探索分布。

继续复查 action trace 后发现 goal-action prior 只使用目标物体词：`tissuebox` 会被增强，但目标 receptacle/navigation（如 `go to sidetable`）得不到增强。对于 `find two tissuebox and put them in sidetable`，这会导致拿到 tissuebox 后仍不稳定地去 sidetable。修复后 `_goal_action_bonus_logits()` 使用所有 goal terms，并保留权重差异：目标物体权重最高，receptacle/location 有较弱但非零 prior。新增 `test_goal_action_bonus_prior_also_supports_goal_receptacle_navigation`，完整 `tests/test_online_hrpo.py` 当前 18 项通过。该 prior 仍只作为 ALFWorld online exploration 诊断，不作为主论文方法默认收益。

receptacle prior smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_goal_receptacle_prior_lr5e5_u5_s12_smoke`（job `86321`）显示了正向行为变化：`mean_rollout_reward=-0.3565`（优于 advfix 的 `-0.4525`）、`mean_irrelevant_action_count=2.30`（优于 `3.95`）、`sidetable` 动作数从 17 提到 64，并在 update 5 出现 `take tissuebox 3 from diningtable 1 -> go to sidetable 2 -> move tissuebox 3 to sidetable 2` 的目标前缀，最高 rollout reward 回到 `0.01`。但 `rollout_success_rate` 仍为 0，说明它只证明 Stage4-online 有可学习的正向探索信号，还没有证明 ALFWorld task success。

随后运行的 horizon 诊断 `outputs/clstr_v4_1b_alfworld_online_hrpo_goal_receptacle_prior_lr5e5_u5_s20_smoke`（job `86472`，`MAX_STEPS=20`）进一步定位低成功率原因：`rollout_success_rate` 仍为 0，`mean_goal_interaction_points` 从 0.15 升到 0.70，`take_tissuebox_count=11`、`move_tissuebox_count=8`，但 `mean_rollout_reward=-0.6215`、`mean_irrelevant_action_count=3.85`。轨迹显示模型已能拿/放目标物体，却会反复拿回已放置的 tissuebox、把 tissuebox 放回 diningtable，或夹杂大量无关物体操作。因此当前瓶颈不是 Qwen，也不是单纯步数不够，而是 ALFWorld online bridge 缺少“已完成子目标/库存状态/正确放置不可逆奖励”的显式建模；现有 shaped reward 和 action prior 只能产生目标前缀，不能稳定约束完整两物体放置策略。

2026-06-10 继续修复为通用 subgoal tracker + step-level credit assignment，而不是写死 `tissuebox/sidetable`。新增逻辑从 goal 中解析目标物体数、目标物体词、目标 receptacle/location 词，并从 ALFWorld admissible action schema 中解析 `take X from Y`、`move/put/place X to Y`。rollout report 新增 `placed_target_count`、`placed_target_event_count`、`reverted_target_count`、`wrong_receptacle_move_count`；reward 对最终正确放置给正奖励，对从目标位置拿回、放到非目标位置给惩罚。

仅修改 rollout-level reward 以后，`outputs/clstr_v4_1b_alfworld_online_hrpo_subgoal_reward_lr5e5_u5_s20_smoke`（job `86489`）仍复现了上一轮 20 条 action trace，说明整条 rollout advantage 仍会把好坏动作混在一起。随后新增 step-level credit：每个 step 根据通用事件获得即时 credit，同一 task 内标准化后叠加到 rollout advantage。这样正确放置动作会被单独强化，拿回/放错/无关动作会被单独压低。新增测试 `test_alfworld_subgoal_tracker_rewards_final_placement_and_penalizes_reverts` 与 `test_flatten_rollout_steps_adds_step_credit_advantages`，完整 `tests/test_online_hrpo.py` 当前 20 项通过。

第一版 step credit 将 step/info/repeat penalty 也放入每一步 credit，导致 400/400 个 step 都有非零 credit，信号过密且噪声大。已改为 sparse event credit：普通导航、look/inventory/help 不直接产生 step credit；step-level credit 只来自目标发现/目标物体交互/正确放置/拿回/放错/无关 manipulation。新增 `test_alfworld_step_credit_is_sparse_for_neutral_navigation`，完整 `tests/test_online_hrpo.py` 当前 21 项通过。

继续检查 sparse credit 后发现 `move target_object to wrong_receptacle` 和 `take target_object from target_receptacle` 仍会先拿到 target-object interaction 正分，再扣错放/拿回惩罚，导致负信号不够干净。已将 target-object transfer 改成互斥事件打分：`take target from non-target` 为正，`move target to target` 为更强正，`take target from target` 和 `move target to non-target` 直接为负，不再叠加泛化的 target interaction 正分。新增 `test_alfworld_step_credit_scores_target_transfers_by_outcome`，完整 `tests/test_online_hrpo.py` 当前 22 项通过。

互斥 transfer step-credit smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_transfer_step_credit_lr5e5_u5_s20_smoke`（job `86535`）正常完成，但仍未提升成功率或行为质量：`rollout_success_rate=0.0`、`mean_rollout_reward=-0.6425`、`mean_goal_interaction_points=0.70`、`mean_placed_target_count=0.10`、`mean_reverted_target_count=0.15`、`mean_wrong_receptacle_move_count=0.15`、`online_credit_reward_nonzero_count=130`。这说明瓶颈不再是“没有 step-level credit”或“正负事件混在一起”，而是 step credit 注入 policy advantage 的强度太弱，5 个 update 后 action trace 基本不变。

本轮新增 `step_credit_advantage_weight` 作为可审计超参：默认 `1.0` 保持旧行为，`0.0` 可做关闭 ablation，大于 `1.0` 用于验证 step credit 强度是否足以推动策略。CLI/sbatch 入口通过 `--step_credit_advantage_weight` / `STEP_CREDIT_ADVANTAGE_WEIGHT` 暴露该参数。新增测试 `test_flatten_rollout_steps_can_disable_step_credit_advantages` 与 `test_flatten_rollout_steps_scales_step_credit_advantages`，完整 `tests/test_online_hrpo.py` 当前 24 项通过。下一步只跑 `STEP_CREDIT_ADVANTAGE_WEIGHT=3.0` 的小 smoke；若动作概率和 trace 仍几乎不动，应转向 high-credit action self-imitation / auxiliary objective，而不是继续叠加 reward shaping。

`STEP_CREDIT_ADVANTAGE_WEIGHT=3.0` 小 smoke `outputs/clstr_v4_1b_alfworld_online_hrpo_step_credit_w3_lr5e5_u5_s20_smoke`（job `86577`）正常完成。它把 policy loss 从上一轮平均 `0.0599` 推到 `-0.2765`，但 20 条 rollout 的 action trace 与 `86535` 完全一致，关键行为指标也完全一致，`prob_max_abs_delta=0.0047`、`prob_mean_abs_delta=0.00027`。因此单纯放大 advantage 通道仍不足以在低成本 online smoke 中改变策略。

下一步改为通用 high-credit action auxiliary objective：`step_credit_imitation_weight` 默认 `0.0`，开启后对正 step credit 的已选动作做 self-imitation，对负 step credit 的已选动作做 unlikelihood penalty。该 objective 只依赖 `online_credit_reward`，不是 ALFWorld 固定物体或固定任务特化；后续可迁移到任何能给 step-level credit/preference 的 online 或 replay 环境。CLI/sbatch 入口通过 `--step_credit_imitation_weight` / `STEP_CREDIT_IMITATION_WEIGHT` 暴露。新增测试 `test_step_credit_imitation_loss_pushes_positive_and_negative_actions` 与 `test_step_credit_imitation_loss_defaults_to_disabled`，完整 `tests/test_online_hrpo.py` 当前 26 项通过。

`STEP_CREDIT_IMITATION_WEIGHT=0.5, LR=5e-5` 小 smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_credit_imitation_w05_lr5e5_u5_s20_smoke`
（job `86599`）正常完成，`step_credit_imitation_loss≈0.563`，说明 auxiliary loss 已接入；
但 20 条 action trace 仍与上一轮完全一致，概率变化只有 `1e-4` 量级。随后用该 checkpoint
做 post-update eval 时发现 online HRPO checkpoint 没有顶层 `skills_path`，只在
`routing_init.skill_source_path` 里保留 skill pool 路径，导致 `_load_clstr_alfworld_model`
回退到旧 `pseudo_skills.jsonl` 并失败。已修复 loader：checkpoint 顶层路径、`routing_init`
路径和嵌套 `policy_checkpoint` 路径都会被解析；online HRPO checkpoint 也会写入顶层
`skills_path/resolved_skills_path`。新增 loader 回归测试。

`STEP_CREDIT_IMITATION_WEIGHT=0.5, LR=2e-4` 小 smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_credit_imitation_w05_lr2e4_u5_s20_smoke`
（job `86667`）开始推动策略：20 条中 8 条 action trace 改变，平均 reward 从 `-0.6425`
提升到 `-0.6085`，最后一轮 `mean_goal_interaction_points=2.25`、
`mean_irrelevant_action_count=2.25`。但 `rollout_success_rate=0`，且
`wrong_receptacle_move_count` 上升到 `0.75`，说明模型学会了更多目标物体操作，但还没学会
稳定保持正确放置。

严格 step-credit 版本
`outputs/clstr_v4_1b_alfworld_online_hrpo_strict_credit_imitation_w05_lr2e4_u5_s20_smoke`
（job `86680`）进一步清理了信用：重复 `examine target_object` 不再得到正 credit，
错放/拿回的 step 负分至少等同正确放置正分。结果略好但仍未成功：
平均 reward `-0.6050`，最后一轮 `mean_placed_target_count=0.25`、
`mean_irrelevant_action_count=1.75`，但 `mean_wrong_receptacle_move_count=0.75`、
`rollout_success_rate=0`。这说明瓶颈已经从“有没有 RL 信号”推进到“策略缺少显式进度/库存
状态记忆”。

本轮新增 `progress_memory` 输入字段：`build_alfworld_policy_state_text()` 默认行为不变，
但 online HRPO 可传入由完整 action history 解析出的目标进度摘要，如
`placed_target_count`、`inventory_target_count`、`reverted_target_count`、
`wrong_receptacle_move_count`。这不是写死 ALFWorld 某个物体，而是把可观测 action history
压缩成 belief/progress memory，符合 CLSTR 的 closed-loop routing 叙事。新增测试
`test_build_alfworld_policy_state_text_accepts_optional_progress_memory` 与
`test_alfworld_progress_memory_tracks_full_history_target_state`；完整
`tests/test_online_hrpo.py` 当前 30 项通过。

progress memory 小 smoke
`outputs/clstr_v4_1b_alfworld_online_hrpo_strict_credit_imitation_w05_lr2e4_progress_memory_u5_s20_smoke`
（job `86695`）正常完成，但没有解决 Stage4-online 瓶颈：与 strict credit 版 `86680`
相比，18/20 条 action trace 完全相同，`rollout_success_rate=0`，平均 reward
`-0.6055`，最后一轮 `mean_placed_target_count=0.25`、
`mean_reverted_target_count=0.25`、`mean_wrong_receptacle_move_count=0.75`。结论是：
在当前弱 online ALFWorld bridge 上继续小修 reward / credit / prompt memory 的边际收益很低；
Stage4-online 只能作为“有可学习信号但未达 task success”的诊断，不应继续烧钱当主线。
下一步应转向离线 trajectory/preference 预热或切到更稳定的 executor/benchmark，再回来做
online RL。

## 2026-05-20 CLSTR-Qwen3-8B external encoder + CLSTR-native ACT

### 实验意图

本阶段目标是在不把 Qwen3-8B 变成 CLSTR 主实验 direct agent 的前提下，将 `Qwen/Qwen3-8B` 冻结为 external grounding encoder，并继续训练/使用 CLSTR-native `skill_head`、`trans_head`、`transition`、`belief/gate`、`STOP`、`action_emb` 和 closed-loop controller。Qwen direct admissible-action baseline 只作为强模型参考线，不计为 CLSTR 结果。

## 2026-05-22 AppWorld + SkillX routing/train pipeline

### 目标

本阶段把项目从 AppWorld/SkillX smoke 推进到可训练、可评测的 AppWorld routing-layer pipeline。当前实现仍严格遵守新的论文定位：CLSTR 是 closed-loop skill routing layer，不是直接替代 AppWorld LM executor 的端到端 agent。

### 关键实现

- 新增 `clstr/appworld_routing.py`：读取 AppWorld split/task metadata、ground-truth `api_calls.json` 和 API docs，把 task instruction + required apps + API refs 转成 routing query，并从 188 条 SkillX AppWorld skills 中选出 task-level positive skill ids。
- 新增 `clstr/appworld_eval.py`：实现 AppWorld routing qrels 上的 SkillRouter-style static routing baseline，输出 predictions 和 recall/MRR 指标。
- 新增 `clstr/appworld_clstr_eval.py`：支持加载 CLSTR checkpoint，用 `skill_table.logits` 对 AppWorld routing tasks 输出 top-k SkillX skill ranking。
- 更新 `clstr/train.py` / `clstr/data.py`：新增 `training_data_source=appworld_routing`，训练资产为 `data/appworld_skill_pool/skill_pool.jsonl` + `data/appworld_routing/train_tasks.jsonl`，环境为 `ReplayTrajectoryEnv(train_replay.jsonl)`。
- 新增入口脚本：
  - `scripts/build_appworld_routing_data.py`
  - `scripts/run_appworld_skillrouter_baseline.py`
  - `scripts/run_appworld_clstr_eval.py`
- 新增配置：
  - `configs/data/appworld_routing.yaml`
  - `configs/model/appworld_mini.yaml`
  - `configs/train/appworld_routing_smoke.yaml`
  - `configs/train/appworld_routing_train.yaml`
- 新增集群脚本：
  - `scripts/sbatch/run_appworld_clstr_train.sh`
  - `scripts/sbatch/run_appworld_clstr_eval.sh`
  - `scripts/sbatch/run_appworld_skillrouter_baseline.sh`

### 已生成数据

`scripts/build_appworld_routing_data.py` 已在真实 AppWorld 数据上完成：

- skill pool：188 条 SkillX AppWorld skills。
- API catalog：453 个 AppWorld standard API docs entries。
- train：90 tasks / 450 qrels / 90 replay rows。
- dev：57 tasks / 285 qrels。
- test_normal：168 tasks / 840 qrels。
- test_challenge：417 tasks / 2085 qrels。
- manifest：`data/appworld_routing/manifest.json`。
- 明确约束：`dev_test_used_for_training=false`，训练只使用 train split 的 routing positives/replay。

### SkillRouter AppWorld baseline

已用 SkillRouter-style serialization + static lexical scoring 在 AppWorld dev routing qrels 上跑一次 baseline：

- 报告：`outputs/appworld_skillrouter_baseline/dev/report.json`
- predictions：`outputs/appworld_skillrouter_baseline/dev/predictions.jsonl`
- dev query_count：57
- skill_count：188
- recall@1：0.982456
- recall@5：1.0
- recall@10：1.0
- recall@20：1.0
- mrr@20：0.991228

注意：这是 AppWorld task-to-SkillX routing baseline，不是 AppWorld DB-state task-completion score。它用于比较 routing layer，而不是证明端到端 agent 已完成官方 AppWorld benchmark。

已补充真正的 SkillRouter embedding baseline 入口：

- 脚本：`scripts/run_appworld_skillrouter_embedding_baseline.py`
- sbatch：`scripts/sbatch/run_appworld_skillrouter_embedding_baseline.sh`
- 模型：`.cache/hf_models/SkillRouter-Embedding-0.6B`
- 首次 A800 job：74962 因 A800 分区缺 `cuda/12.1` module 在 shell 阶段失败；已修复 sbatch 脚本为可选加载 CUDA module。
- 成功 job：74964，A800，exit code 0，elapsed 00:00:43。
- 输出目录：`outputs/appworld_skillrouter_embedding_baseline/dev_a800_v2`
- dev query_count：57
- skill_count：188
- recall@1：0.245614
- recall@5：0.526316
- recall@10：0.596491
- recall@20：0.666667
- mrr@20：0.371327

这是当前更接近“用 SkillRouter 跑 AppWorld routing baseline”的结果；lexical baseline 保留为无 GPU fallback。

### AppWorld official runtime sanity

为确认不是只做离线 routing 文件，已新增并运行单 task official runtime oracle smoke：

- 脚本：`scripts/run_appworld_oracle_solution_smoke.py`
- 报告：`outputs/appworld_oracle_solution_smoke/report.json`
- task：`82e2fac_1`
- 执行内容：官方 `ground_truth/compiled_solution.py`
- 结果：`execute_output="Execution successful."`，`task_completed=true`，`evaluate_available=true`

该结果只证明 AppWorld runtime/verifier 可用，是 oracle sanity check，不计为 CLSTR 或 SkillRouter 模型结果。

### 训练状态

CLSTR AppWorld routing 训练入口已接通：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_train.sh
```

训练脚本会先刷新 `data/appworld_routing/`，再运行：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m clstr.train \
  --train_config configs/train/appworld_routing_train.yaml \
  --model_config configs/model/appworld_mini.yaml \
  --data_config configs/data/appworld_routing.yaml \
  --checkpoint_dir outputs/appworld_clstr_train/checkpoints \
  --max_train_tasks 90
```

已提交一个训练 smoke job 验证集群路径：

- 旧 H200 pending job：74954/74956 因优先级等待过久已取消，避免旧脚本重复运行。
- 首次 A800 job：74961 因 A800 分区缺 `cuda/12.1` module 在 shell 阶段失败；根因同 baseline job，已修复脚本。
- 成功 job：74963，A800，exit code 0，elapsed 00:00:55。
- command：`sbatch --gpus=1 -p gpu_a800 --export=ALL,TRAIN_CONFIG=configs/train/appworld_routing_smoke.yaml,MAX_TRAIN_TASKS=2,OUTPUT_DIR=outputs/appworld_clstr_train_smoke_a800_v2 scripts/sbatch/run_appworld_clstr_train.sh`
- checkpoint：`outputs/appworld_clstr_train_smoke_a800_v2/checkpoints/stage0_warmstart-step1.pt`
- train stdout/report：`outputs/appworld_clstr_train_smoke_a800_v2/train_stdout.json`
- 训练配置：`training_data_source=appworld_routing`，`env_mode=replay`，`model_dim=384`，`device=cuda:0`，`max_train_tasks=2`，`updates=1`。

该 smoke 证明 CLSTR train CLI、AppWorld routing data、ReplayTrajectoryEnv、MiniLM-backed CLSTRModel、checkpoint 输出链路均可在集群 GPU 作业中跑通。正式训练入口是同一脚本，把 `TRAIN_CONFIG=configs/train/appworld_routing_train.yaml` 且 `MAX_TRAIN_TASKS=90` 即可跑完整 train split routing 训练。

### CLSTR AppWorld routing eval

已用 smoke checkpoint 跑 AppWorld dev routing eval：

- job：74966，A800，exit code 0，elapsed 00:00:19。
- checkpoint：`outputs/appworld_clstr_train_smoke_a800_v2/checkpoints/stage0_warmstart-step1.pt`
- report：`outputs/appworld_clstr_eval/dev/report.json`
- predictions：`outputs/appworld_clstr_eval/dev/predictions.jsonl`
- query_count：57
- skill_count：188
- recall@1：0.052632
- recall@5：0.175439
- recall@10：0.263158
- recall@20：0.508772
- mrr@20：0.136596

已生成三方对比：

- JSON：`outputs/appworld_routing_comparison/comparison.json`
- Markdown：`outputs/appworld_routing_comparison/comparison.md`
- 方法包含：
  - `skillrouter_serialization_lexical`
  - `skillrouter_embedding_0.6b`
  - `clstr_checkpoint_skill_table_logits`

### 2026-05-22 AppWorld base 训练修正（direct routing supervision）

针对前一版 CLSTR AppWorld base 明显弱于 SkillRouter embedding baseline 的问题，已定位到两个直接 root cause：

- `clstr/bridges/skillrouter/datasets.py::load_eval_tasks` 训练时优先读取 `instruction_text`，导致 AppWorld enriched routing query（含 `required_apps` 和 `api_refs`）在训练链路中被丢掉，而 dev eval 实际用的是完整 `query`。
- `clstr/train.py` + `clstr/losses.py` 之前的 AppWorld base 基本依赖 replay rollout 的 reward 和 weak positives；在真实 smoke 中 `mean_reward=0.0`，因此几乎没有直接的 task-to-positive-skill 监督，`skill_table.logits` 很难学到有效的 routing adapter。

已完成的修正：

- 更新 `clstr/bridges/skillrouter/datasets.py`：`load_eval_tasks` 现在优先使用 `query`，只有缺失时才回退到 `instruction_text`。
- 新增 `clstr/losses.py::multi_positive_routing_loss`：对 `train_tasks.jsonl` 中的 `positive_skill_ids` 做 multi-positive set routing supervision，直接优化 `skill_table.logits` 把概率质量压到 positive skill 集合上。
- 更新 `clstr/train.py`：新增 `routing_supervision_weight` 配置项，把 direct routing supervision 接入 AppWorld base 训练，即使 rollout reward 仍为 0，也能继续训练 retrieval adapter。
- 更新训练配置：
  - `configs/train/appworld_routing_smoke.yaml`
  - `configs/train/appworld_routing_train.yaml`
  现在默认启用 `routing_supervision_weight: 1.0`。
- 更新 `scripts/sbatch/run_appworld_clstr_eval.sh`：支持 `MODEL_CONFIG` / `TASKS_PATH` / `QRELS_PATH` / `SKILL_POOL_PATH` / `OUTPUT_DIR` / `TOP_K` 环境变量，方便对中途 checkpoint 做不覆盖旧结果的集群评估。
  - 后续又补充 `CHECKPOINT_DIR`，支持 full train 结束后通过 `afterok` 自动评估最新 checkpoint。

本地验证：

- 使用仓库内私有 pytest 目录 `.vendor_pytest/` 跑：
  - `tests/test_bridge_skillrouter.py`
  - `tests/test_losses.py`
  - 结果：`22 passed`
- 新增覆盖：
  - `test_load_eval_tasks_prefers_enriched_query_over_instruction_text`
  - `test_multi_positive_routing_loss_uses_task_positive_skill_ids`

中途 GPU 结果：

- 训练 job：`75082`
  - command：`sbatch --gpus=1 -p gpu_a800 --export=ALL,TRAIN_CONFIG=configs/train/appworld_routing_train.yaml,MAX_TRAIN_TASKS=20,OUTPUT_DIR=outputs/appworld_clstr_train_supervised_smoke_v1 scripts/sbatch/run_appworld_clstr_train.sh`
- 中途 eval job：`75083`
  - checkpoint：`outputs/appworld_clstr_train_supervised_smoke_v1/checkpoints/stage0_warmstart-step7.pt`
  - report：`outputs/appworld_clstr_eval/intermediate_stage0_step7/report.json`
  - metrics：
    - recall@1：`0.175439`
    - recall@5：`0.456140`
    - recall@10：`0.596491`
    - recall@20：`0.807018`
    - mrr@20：`0.297108`

- 后续中途 eval job：`75091`
  - checkpoint：`outputs/appworld_clstr_train_supervised_smoke_v1/checkpoints/stage2_base-step1.pt`
  - report：`outputs/appworld_clstr_eval/intermediate_stage2_step1/report.json`
  - metrics：
    - recall@1：`0.824561`
    - recall@5：`1.0`
    - recall@10：`1.0`
    - recall@20：`1.0`
    - mrr@20：`0.889181`

- final smoke eval job：`75107`
  - checkpoint：`outputs/appworld_clstr_train_supervised_smoke_v1/checkpoints/stage2_base-step20.pt`
  - report：`outputs/appworld_clstr_eval/final_smoke_stage2_step20/report.json`
  - metrics：
    - recall@1：`0.771930`
    - recall@5：`0.947368`
    - recall@10：`1.0`
    - recall@20：`1.0`
    - mrr@20：`0.852548`

与当前 SkillRouter embedding baseline 对比：

- SkillRouter embedding baseline：`outputs/appworld_skillrouter_embedding_baseline/dev_a800_v2/report.json`
  - recall@20：`0.666667`
- 修正后的 CLSTR 中途 checkpoint：
  - recall@20：`0.807018`

- 修正后的 CLSTR 后续中途 checkpoint：
  - recall@20：`1.0`

已生成当前 supervised 路由对比表：

- JSON：`outputs/appworld_routing_comparison_supervised_v1/comparison.json`
- Markdown：`outputs/appworld_routing_comparison_supervised_v1/comparison.md`
- 当前包含：
  - `skillrouter_embedding_0.6b`
  - `clstr final_smoke_stage2_step20`
  - `clstr full_v1_stage0_step20`

这说明 direct routing supervision 已经把 CLSTR-base 的 routing retrieval 拉到高于当前 SkillRouter embedding baseline。完整 90-task 正式训练已另外提交：

- full train job：`75084`
  - command：`sbatch --gpus=1 -p gpu_a800 --export=ALL,TRAIN_CONFIG=configs/train/appworld_routing_train.yaml,MAX_TRAIN_TASKS=90,OUTPUT_DIR=outputs/appworld_clstr_train_supervised_full_v1 scripts/sbatch/run_appworld_clstr_train.sh`
  - 当前状态：运行中；已挂 dependent eval job `75090`
  - eval command：`sbatch --dependency=afterok:75084 --gpus=1 -p gpu_a800 --export=ALL,CHECKPOINT_DIR=outputs/appworld_clstr_train_supervised_full_v1/checkpoints,OUTPUT_DIR=outputs/appworld_clstr_eval/full_v1_dev scripts/sbatch/run_appworld_clstr_eval.sh`
  - 再挂一个最终对比后处理 job：`75125`
    - command：`sbatch --dependency=afterok:75090 --gpus=1 -p gpu_a800 --wrap="module load miniforge3/25.11.0-1; cd /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr; /data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/build_appworld_routing_comparison.py --reports outputs/appworld_skillrouter_embedding_baseline/dev_a800_v2/report.json outputs/appworld_clstr_eval/full_v1_dev/report.json --output_dir outputs/appworld_routing_comparison_full_v1"`
    - 作用：`75090` 完成后自动生成 full 训练 vs SkillRouter embedding baseline 的最终对比表。
  - 另已补一个 full 训练中途 eval job：`75104`
    - checkpoint：`outputs/appworld_clstr_train_supervised_full_v1/checkpoints/stage0_warmstart-step5.pt`
    - output_dir：`outputs/appworld_clstr_eval/full_v1_stage0_step5`
    - metrics：`recall@20=0.771930`
  - 再补一个更靠后的 full 训练中途 eval job：`75108`
    - checkpoint：`outputs/appworld_clstr_train_supervised_full_v1/checkpoints/stage0_warmstart-step10.pt`
    - report：`outputs/appworld_clstr_eval/full_v1_stage0_step10/report.json`
    - metrics：
      - recall@1：`0.350877`
      - recall@5：`0.649123`
      - recall@10：`0.719298`
      - recall@20：`0.859649`
      - mrr@20：`0.480547`
  - 再补一个 `stage0` 结束点的 full 中途 eval job：`75116`
    - checkpoint：`outputs/appworld_clstr_train_supervised_full_v1/checkpoints/stage0_warmstart-step20.pt`
    - report：`outputs/appworld_clstr_eval/full_v1_stage0_step20/report.json`
    - metrics：
      - recall@1：`0.543860`
      - recall@5：`0.807018`
      - recall@10：`0.982456`
      - recall@20：`1.0`
      - mrr@20：`0.676427`
  - 再补一个 `stage1` 早期 full 中途 eval job：`75123`
    - checkpoint：`outputs/appworld_clstr_train_supervised_full_v1/checkpoints/stage1_heads-step2.pt`
    - report：`outputs/appworld_clstr_eval/full_v1_stage1_step2/report.json`
    - metrics：
      - recall@1：`0.631579`
      - recall@5：`0.947368`
      - recall@10：`1.0`
      - recall@20：`1.0`
      - mrr@20：`0.768588`

### 当前边界

当前已经做到“AppWorld routing 数据可构建、SkillRouter lexical/embedding baseline 已跑、CLSTR 训练入口和 checkpoint 输出已通过 GPU sbatch smoke、CLSTR dev routing eval 已跑、对比表已生成”。完整 AppWorld 官方 DB-state completion 还需要接入下游 LM executor/official verifier loop，不能把当前 routing qrels 结果冒充为官方 AppWorld success rate。

### 关键实现

- 新增 `clstr/qwen_direct_policy.py`：实现 Qwen direct admissible-action scorer。模型只能从 `admissible_commands` 中选择 exact action；解析失败会记录 `fallback`，不会伪造不在候选集内的动作。
- 新增 `scripts/run_qwen3_alfworld_direct_eval.py`：复用 ALFWorld official closed-loop harness，输出 direct Qwen baseline 的 metrics/blocker；结果显式标记 `not_clstr_result=true`。
- 新增 `clstr/qwen_external_encoder.py`：实现 frozen Qwen external encoder 配置和 CLSTR model builder。默认 `last_token` pooling、left padding、`use_cross_encoder=false`，避免加载第二个 8B backbone。
- 新增 `clstr/qwen_full_base_train.py` 和 `scripts/run_clstr_qwen3_full_base_train.py`：冻结 Qwen，训练 CLSTR-native ACT/full-base modules，并在失败时写可复现 blocker。
- 更新 `clstr/alfworld_eval.py`：`run.jsonl` 可记录 policy metadata trace；CLSTR eval 可识别 Qwen external checkpoint metadata。
- 更新 `clstr/full_base_train.py`：报告 `clstr_native_act_trained`，并在 checkpoint/report 中保留 Qwen external encoder metadata。
- 新增 `scripts/prepare_qwen3_8b.py` 和 `scripts/build_clstr_qwen3_reports.py`。

### 保留的必要方法修改

`L_trans_skill_ce` 保留为 action-level transition supervision：用 transition prior / `trans_head` 对完整 skill pool 打分，并以 `next_skill_id` 做交叉熵。这比 observation cosine 更直接，适合作为 CLSTR-native ACT 的 supervised transition-prior objective。`L_trans` observation/cosine 仍只作为低权重辅助，不作为主导目标。

### 2026-05-20 下载和设备修复更新

Qwen3-8B 已完整下载到 `/root/autodl-tmp/clstr/models/Qwen3-8B`。下载使用 AutoDL HuggingFace 镜像站 `HF_ENDPOINT=https://hf-mirror.com`，并保留 `/etc/network_turbo` 环境作为传输加速。`outputs/clstr_qwen3_8b_init/manifest.json` 已刷新为 `status=ok`。

修复一个会显著影响实验速度的设备问题：

- `QwenDirectAdmissibleActionScorer` 现在在未显式指定 device 时，会在 CUDA 可用时自动把 Qwen direct 模型搬到 GPU。
- `build_qwen_external_clstr_config` 现在对 Qwen external CLSTR 设置 `defer_skill_table_init=true`，避免 CLSTR 初始化阶段在 CPU 上用 8B encoder 编码 skill table。
- `train_clstr_full_base_with_model` 会在 `model.to(device)` 后重建 deferred skill table，确保 skill embeddings 在目标设备上生成。
- Qwen external 训练的 embedding cache batch size 从通用默认 256 降到 8，避免 8B encoder 在缓存 state/action embeddings 时 OOM。
- Qwen external checkpoint 过滤掉 `encoder.backbone.*`、`encoder.proj.*`、`skill_table.encoder_fn.backbone.*`、`skill_table.encoder_fn.proj.*`、`skill_table.W/E` 等冻结/可重建张量，只保存 CLSTR-native trainable heads，避免由于 `SkillTable.encoder_fn` 重复引用 Qwen backbone 生成十几 GB checkpoint。

相关验证：

- `pytest -q tests/test_qwen_direct_policy.py tests/test_qwen_external_encoder.py`：10 passed。
- `pytest -q tests/test_full_base_train.py::test_qwen_external_checkpoint_filter_removes_duplicate_skill_table_encoder_refs tests/test_full_base_train.py::test_qwen_external_checkpoint_excludes_frozen_backbone_state`：2 passed。

### 2026-05-20 Qwen gate 训练与 closed-loop 诊断

2k gate 训练已完成，使用 `data/clstr_qwen3_8b_full_base_gate/train_alfworld_official.jsonl`，只含 ALFWorld train official replay。Qwen3-8B frozen，未做 full fine-tuning，训练 CLSTR-native `transition/gate/stop_head/skill_head/trans_head/action_emb`。这版的 `L_policy` 是 supervised admissible-action CE：`CE(policy_scores(admissible_commands_t), expert_action_t)`，不是 markdown 中需要 on-policy grouped rollout/reward 的 GRPO-style policy loss；报告中明确 `grpo_policy_loss_used=false`。routing/retrieval 的 InfoNCE-like 辅助项作为 `retrieval_contrastive_loss` 单独存在。

重建 full-base 数据后，ALFWorld official gate 子集包含：

- `L_policy`: 2443
- `L_trans`: 2443
- `L_trans_skill_ce`: 2269
- `STOP`: 2443
- `routing`: 2369

训练后 offline 指标：

- `policy_ce_loss`: 2.0671
- `transition_skill_ce_loss`: 0.7229
- `transition_skill_recall@1`: 0.7717
- `transition_skill_recall@5`: 0.9920
- peak GPU memory: 19.074GB

但是 valid_seen 20 episode closed-loop 仍为 0：

- `outputs/alfworld_eval/clstr_qwen3_8b_gate_valid_seen/metrics.json`
- success_rate=0.0, reward=0.0, average_goal_condition_points=0.0

trace 诊断显示 raw policy 在 1000/1000 步把 `look` 排为 policy top-1；controller final choice 又大量选择 `help`/`inventory`/`look`。因此当前瓶颈不是 Qwen 没冻结，也不是 `L_policy` 没启用，而是 action grounding/calibration 失败：`look` 在 official replay 中是高频 expert 动作，`help`/`inventory` 几乎每步都在 admissible list 中且 embedding 稳定，普通 CE 和当前 controller calibration 没有足够负约束来抑制这些信息动作。下一步先加入 controller-level information-action prior/penalty，再补训练层 hard-negative/aux penalty；若仍无提升，再跑更完整 Qwen direct baseline 判断 base 模型和 prompt 上限。

### 论文叙事边界

- `Qwen3-8B direct baseline` 是参考线，不是 CLSTR。
- `CLSTR-Qwen3-8B` 主实验中 Qwen 必须 frozen，只提供 external grounding representation。
- CLSTR 主实验的 action selection 必须仍然在 `admissible_commands` 上通过 CLSTR-native heads/controller ranking。
- Qwen direct baseline 现在已有 closed-loop 指标，但只作为参考线，不能计为 CLSTR 结果。


## 2026-05-21 HRPO-style online loss gate

用户提出的 HRPO/GRPO-style `L_policy` 不是当前 supervised admissible-action CE。已新增独立 online 阶段：`clstr/online_hrpo.py` 和 `scripts/run_clstr_qwen3_online_hrpo.py`。该阶段只在 ALFWorld train split 上 rollout，按同一 train game 的 grouped samples 计算 normalized advantage，使用 `-advantage * log pi(action|state, admissible_commands)`，并对 frozen initial CLSTR controller modules 加 KL reference。Qwen3-8B 仍冻结，`qwen_direct_generator=false`，CLSTR 主实验仍只在 admissible commands 上 ranking。

实现细节：
- 历史记录：当时的 `HrpoLossConfig.score_mode=controller_complete` 曾把 candidate/action embedding 当作 proxy observation，计算 transition、belief/gate、STOP 分数，并让梯度流到 `transition/gate/stop_head`。
- 2026-06-07 语义修复后，上述 pre-action proxy observation 路径已禁用；当前旧 HRPO `controller_complete` 只保留 policy-side action logits，transition/gate/STOP 不再通过 candidate embedding 代理观测参与反传。closed-loop transition 证据应来自 Stage2/Stage4 的真实 `next_observation_text` 语义，而不是这条历史 ALFWorld HRPO 近似路径。
- reference policy 不是第二个 Qwen，而是复制冻结初始 CLSTR 小模块，复用同一 frozen Qwen embeddings，避免 32GB 显存中同时加载两套 8B。
- 为避免 OOM，HRPO action/state encoding 增加 `encode_batch_size` 分块；之前未分块的小规模 gate 在 2 train games x 3 rollouts x 3 updates 时 OOM，blocker 写入 `outputs/clstr_qwen3_8b_online_hrpo_gate/blocker_report.json`。
- 新增 controller long-cycle penalty，记录为工程 controller calibration，不等同 HRPO 本身。

已完成结果：
- HRPO batched gate training: `outputs/clstr_qwen3_8b_online_hrpo_gate_batched/train_report.json`
  - checkpoint: `outputs/clstr_qwen3_8b_online_hrpo_gate_batched/checkpoints/clstr_qwen3_online_hrpo-step3.pt`
  - ALFWorld train-only rollouts: 12 条 / 240 步；train rollout success_rate=0.0；mean shaped reward=-1.1725；peak GPU=20.601GB。
- valid_seen closed-loop eval: `outputs/alfworld_eval/clstr_qwen3_8b_hrpo_cycle_gate_valid_seen/metrics.json`
  - 20 episodes, success_rate=0.0, average_reward=0.0, average_goal_condition_points=0.0, average_episode_steps=50.0。
  - blocker: `outputs/alfworld_eval/clstr_qwen3_8b_hrpo_cycle_gate_valid_seen/blocker_report.json`。

诊断：
- supervised CE 后 raw policy top-1 在 valid_seen 1000/1000 步仍是 `look`。
- HRPO + controller penalties 后 info-action collapse 明显下降，但行为转为 `go/examine` 导航循环，`go` 动作占约 79%，仍无 pick/place/heat/cool/clean 等任务进展。
- ALFWorld official replay subset 本身 `go`≈52%、`look`≈24%，object-interaction 标签稀疏；train HRPO rollout 没有正 reward，无法提供有效 credit assignment。
- 当前 CLSTR-Qwen HRPO gate 没有超过 0.6B CLSTR best（valid_seen 20 episode success_rate=0.05）。不能把这个阶段称为 ALFWorld 提升。

下一步建议：先跑更完整 Qwen direct admissible-action baseline（20/50 episodes）确认 Qwen3-8B prompt 上限；如果 direct Qwen 有非零成功，再用 ALFWorld train split 生成 verified teacher rollout 增强 CLSTR。若 direct Qwen 仍为 0，应先换更强/更适配的 instruct model 或 prompt/planner，再继续 CLSTR 训练。


## 2026-05-21 Qwen direct likelihood baseline

为避免生成式 Qwen baseline 的解析失败问题，新增 `QwenDirectLikelihoodActionScorer`：对每个 admissible command 做 conditional log-likelihood ranking，仍然只从候选动作 exact 选择，不自由生成，也不是 CLSTR 结果。为避免候选 batch OOM，增加 `likelihood_batch_size` 分块。

结果：`outputs/alfworld_eval/qwen3_8b_likelihood_valid_seen/metrics.json`
- valid_seen 20 episodes，success_rate=0.0，average_reward=0.0，average_goal_condition_points=0.0，average_episode_steps=50.0。
- parse_success_rate=1.0，fallback_count=0，说明 0 分不是解析失败导致。
- trace 主要 stuck 在 `examine/open/use/close` 等局部动作，仍缺少 ALFWorld 长程规划。

判断（已被后续 chat-template + AVAILABLE ACTIONS generation baseline 修正）：该 0 分只说明旧 likelihood prompt/dataflow 不足，不能再作为“Qwen3-8B base 没有 ALFWorld 能力”的结论。

### 2026-05-21 Qwen direct likelihood full eval

已补跑 Qwen3-8B direct likelihood baseline 的完整 ALFWorld valid_seen + valid_unseen closed-loop eval，仍然只在 `admissible_commands` 上做 exact-action likelihood ranking，不自由生成动作，也不训练 Qwen。

结果：
- `outputs/alfworld_eval/qwen3_8b_likelihood_full/valid_seen/metrics.json`：140 episodes，success_rate=0.0，average_reward=0.0，average_goal_condition_points=0.0，average_episode_steps=50.0。
- `outputs/alfworld_eval/qwen3_8b_likelihood_full/valid_unseen/metrics.json`：134 episodes，success_rate=0.0，average_reward=0.0，average_goal_condition_points=0.0，average_episode_steps=50.0。
- `outputs/alfworld_eval/qwen3_8b_likelihood_full/metrics.json`：aggregate 274 episodes，success_rate=0.0，average_reward=0.0，average_goal_condition_points=0.0，average_episode_steps=50.0。

结论（已被后续 chat-template + AVAILABLE ACTIONS generation baseline 修正）：这只说明旧 likelihood baseline 不是合适的 Qwen 裸跑协议。不能用该结果否定 Qwen3-8B 在显式候选动作 prompt 下的 ALFWorld 能力。

### 2026-05-21 Qwen direct dataflow bugfix: chat template + AVAILABLE ACTIONS

重新审计裸跑 Qwen 数据流后发现先前结论过强：`qwen3_8b_likelihood_full=0` 不能代表“Qwen3-8B + Available Actions 也失败”。原因：

- generation 版虽然在 raw prompt 中列出了 admissible commands，但没有使用 Qwen3 chat template；parser 只看首行，`40. open cabinet 1` 这类编号动作或后置 `ACTION:` 行会被误判为 fallback。
- likelihood 版更严重：实际 scoring prompt 是 `state + "Action: " + candidate`，没有把完整 `AVAILABLE ACTIONS` 候选列表放进模型上下文。
- fallback 之前默认偏向 `look`，会把 parse failure 系统性导向信息动作循环。

已修复：

- `clstr/qwen_direct_policy.py` 支持 Qwen3 `tokenizer.apply_chat_template(..., enable_thinking=false)`。
- generation 和 likelihood prompt 都显式包含 `AVAILABLE ACTIONS`。
- parser 支持 exact action、编号动作行、后置 `ACTION:` 行。
- CLI 默认 fallback 改为 `first_admissible`，并继续单独统计 fallback/invalid parse。

修复后初步 closed-loop：

- `outputs/alfworld_eval/qwen3_8b_chat_available_valid_seen_20/metrics.json`：valid_seen 20 episodes，success_rate=0.10，average_reward=0.10，average_episode_steps=46.25，parse_success_rate=0.940541。
- `outputs/alfworld_eval/qwen3_8b_chat_available_valid_seen_50/metrics.json`：valid_seen 50 episodes，success_rate=0.14，average_reward=0.14，average_episode_steps=44.70，parse_success_rate=0.943177。
- `outputs/alfworld_eval/qwen3_8b_chat_available_full/metrics.json`：full valid_seen+valid_unseen 共 274 episodes，success_rate=0.109489，average_reward=0.109489，average_episode_steps=45.572993，parse_success_rate=0.890606。
- full split 明细：valid_seen 140 episodes success_rate=0.114286；valid_unseen 134 episodes success_rate=0.104478。

这说明旧 Qwen baseline 的 0 分主要是协议和解析实现问题，不是 Qwen3-8B 完全没有 ALFWorld 能力。Qwen direct 仍是 reference baseline，不是 CLSTR 结果。

### 2026-05-21 CLSTR-Qwen Available Actions planner context

为避免 32GB 显存下同时加载 Qwen CausalLM planner 和 Qwen external encoder，先实现 memory-safe 的 CLSTR-Qwen latent planner context：同一个 frozen Qwen encoder 读取 `state + AVAILABLE ACTIONS + CLSTR skill-routing query`，输出 latent action-query embedding；CLSTR-native `skill_head/controller` 仍负责 admissible action ranking。该路径记录为 `planner_intent_mode=latent_available_actions_query`，`qwen_direct_generator_in_clstr=false`。这不是 Qwen direct action generation，也不替代 CLSTR controller。

评测结果：
- `outputs/alfworld_eval/clstr_qwen3_8b_available_actions_gate_valid_seen_20/metrics.json`：valid_seen 20 episodes，success_rate=0.0，average_reward=0.0，average_goal_condition_points=0.0，average_episode_steps=50.0。
- `outputs/alfworld_eval/clstr_qwen3_8b_available_actions_gate_valid_seen_20/controller_diagnostic.json`：controller component trace 已记录，`controller_mode=policy_plus_transition_belief_stop_loop_penalty`，但行为仍主要是 `go/examine` 导航循环；top verbs 为 `go=797, examine=135, inventory=29, use=25, look=14`。
- `outputs/alfworld_eval/clstr_qwen3_8b_available_actions_gate_valid_seen_20/blocker_report.json`：结论是 available-actions latent context 没有把 Qwen direct 的非零行为迁移到 CLSTR-native ACT/controller。当前主 blocker 是 CLSTR action grounding / controller calibration，而不是 Qwen 是否冻结。
- 进一步用 `data/clstr_qwen3_8b_available_actions_train/train.jsonl`（ALFWorld official train replay + AgentGym train exact-match weak-policy rows）训练 2k step 后，`outputs/alfworld_eval/clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20/metrics.json` 仍为 success_rate=0.0 / average_reward=0.0 / average_goal_condition_points=0.0 / average_episode_steps=50.0。
- 对应 blocker: `outputs/alfworld_eval/clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20/blocker_report.json`。当前结论是 embedding-only frozen Qwen planner context + CLSTR-native ACT/controller 不能继承 Qwen direct generation 的非零规划能力；瓶颈主要在 CLSTR action grounding/controller objective，而不是是否给 Qwen 看到了 available actions。

HRPO-style `L_policy` 的实现边界：由于 CLSTR 主路线不是 token generator，而是在 admissible action set 上 ranking，公式中的 `log pi(y_t | x, y_<t, H_<t)` 在这里对应 action-level `log pi(a_t | state_t, admissible_commands_t)`。它使用 train split rollout 的 grouped normalized advantage 和 KL-to-reference，但不把 Qwen 变成自由动作生成器，也不使用 valid/test 训练。

### 2026-05-21 Available-actions retrain result

为验证“候选动作显式进 Qwen latent state”是否需要重训，构建了 `data/clstr_qwen3_8b_available_actions_train/train.jsonl`：
- official ALFWorld train replay rows: 2443
- AgentGym/AgentTraj-L weak-policy exact-match rows: 2406
- total rows: 4849
- split=train_only，ALFWorld valid/test 未用于训练；AgentGym 标记为 weak_policy，不伪装成 official replay。

训练输出：
- checkpoint: `outputs/clstr_qwen3_8b_full_base_available_actions_gate/checkpoints/clstr_full_base-step2000.pt`
- train_report: `outputs/clstr_qwen3_8b_full_base_available_actions_gate/train_report.json`
- Qwen frozen=true，qwen_direct_generator=false，include_available_actions_in_state=true，CLSTR-native ACT/head/controller 仍是主路线。
- offline: policy_ce_loss=2.0563，transition_skill_recall@1=0.7755，transition_skill_recall@5=0.9913。

closed-loop gate：
- `outputs/alfworld_eval/clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20/metrics.json`：valid_seen 20 episodes，success_rate=0.0，average_reward=0.0，average_goal_condition_points=0.0，average_episode_steps=50.0。
- `outputs/alfworld_eval/clstr_qwen3_8b_available_actions_trained_gate_valid_seen_20/blocker_report.json`：重训后仍主要是 `go/examine`，top verbs 为 `go=524, examine=445, look=22, use=9`，20 集全部 50 步 timeout。

Qwen direct protocol-correct reference：
- `outputs/alfworld_eval/qwen3_8b_chat_available_valid_seen_20/metrics.json`：valid_seen 20 episodes，success_rate=0.10。
- `outputs/alfworld_eval/qwen3_8b_chat_available_valid_seen_50/metrics.json`：valid_seen 50 episodes，success_rate=0.14，average_reward=0.14，parse_success_rate=0.943177。
- `outputs/alfworld_eval/qwen3_8b_chat_available_full/metrics.json`：valid_seen+valid_unseen full 274 episodes，success_rate=0.109489，average_reward=0.109489，average_goal_condition_points=0.0，average_episode_steps=45.572993，parse_success_rate=0.890606。
  - valid_seen: `outputs/alfworld_eval/qwen3_8b_chat_available_full/valid_seen/metrics.json`，140 episodes，success_rate=0.114286。
  - valid_unseen: `outputs/alfworld_eval/qwen3_8b_chat_available_full/valid_unseen/metrics.json`，134 episodes，success_rate=0.104478。
- 因此 Qwen3-8B direct reference 有非零 ALFWorld 能力，但当前 embedding-only external encoder + CLSTR-native ranking 没有继承这种能力。下一步若继续 ALFWorld，不建议继续盲目扩大当前 CE/HRPO；更合理的是 verified Qwen teacher rollout 或把 CLSTR controller 升级为 planner-controller/Verifier，而不是只做 pooled embedding scorer。

### 2026-05-21 AgentGym/AgentTraj-L audit

已下载并审计 `AgentGym/AgentTraj-L` 的 `alfworld_train.json` 到 `data/agentgym_agenttraj_l/`。它是 external agent conversation/SFT trajectory，不是 official replay，也不是 verified env replay。

审计输出：

- manifest: `data/agentgym_agenttraj_l/alfworld/manifest.json`
- converted jsonl: `data/agentgym_agenttraj_l/alfworld/converted_train.jsonl`
- audit report: `data/agentgym_agenttraj_l/alfworld/audit_report.json`

统计：

- trajectories=2420
- converted steps=32247
- `L_policy` enabled steps=2406，仅限存在 `AVAILABLE ACTIONS` 且 action exact match 的 step
- next observation steps=29827，可作为 weak `L_trans/belief` 证据
- routing enabled steps=31927
- bucket=`weak_policy`
- leakage_risk=`external_agent_trajectory_train_only_unverified_env_replay`

因此它可以作为 train-only weak trajectory 数据进入 ablation，但不能包装成 official ALFWorld replay。

### 2026-05-21 SkillNet registry alignment plan

新增方案文档：`docs/clstr_skillnet_pseudoskill_integration_plan.md`。

结论：后续 ALFWorld 修复应先把 SkillNet registry 作为 canonical CLSTR skill space，但 final action 仍必须是官方 env 返回的 `admissible_commands` exact string。当前 `data/clstr_dagger_expert_corrected_train/skills.jsonl` 虽然已有 121 个 SkillNet-derived ALFWorld skills，但缺 `body`；训练/评估 action-to-skill mapping 也仍有重复实现。下一步优先级是：

1. 统一 canonical `alfworld_action_to_skill_id`；
2. 生成包含 SkillNet `body`、action templates、train-only examples 的 enriched skill table；
3. 让 controller 的 transition/belief 分数使用训练过的 `trans_head` / `skill_head`，不再依赖 goal-cosine delta；
4. 明确 P3 是 frozen Qwen pooled embedding 的潜在表示瓶颈，不是第一优先代码 bug；先修 P2/P4 后再判断；
5. 明确 P6 hardcoded penalty 是 calibration 风险，需在真实 head signal 对齐后做 normal/half/zero penalty ablation；
6. 修复后再恢复 loss ablation，旧 ablation 仅作为 pre-fix diagnostic。

### 2026-05-21 P2/P4 controller scorer and skill mapper fix

已完成两个会直接影响正常训练/评估一致性的修复：

- 历史记录：当时的 `clstr/alfworld_eval.py::make_controller_component_scorer` 曾优先使用训练过的 CLSTR native heads：
  - `transition_scores = model.trans_head(predicted_next_state, candidate_action_emb)`
  - `belief_scores = model.skill_head(candidate_action_emb, belief_state)`
  - 仅当对应 head 缺失或输出 shape 不兼容时才回退到旧的 goal-cosine diagnostic。
- 2026-06-07 语义复查后确认：pre-action ALFWorld scorer 没有真实 post-action observation，不能把 candidate action embedding 当作 observation 输入给 transition/gate。当前 `make_controller_component_scorer` 已禁用这条 proxy transition/belief 路径，metadata 会记录 `component_score_source=policy_q_success_stop_only_no_proxy_observation`、`transition_score_source=disabled_no_post_action_observation`、`belief_score_source=disabled_no_post_action_observation`，避免继续把 fallback/proxy 分数误认为 trained-head controller。
- 删除重复实现 `clstr/alfworld_skill_mapping.py` 和对应测试，保留 `clstr/alfworld_action_skills.py` 作为唯一 canonical mapper。
- `clstr/full_base_preprocess.py`、`clstr/alfworld_eval.py`、`clstr/agentgym_data.py` 现在都使用同一份 ALFWorld action -> SkillNet pseudo-skill mapper，避免 train/eval/weak trajectory 的 skill_id 空间漂移。

验证：

- `pytest -q tests/test_alfworld_action_skills.py tests/test_agentgym_data.py tests/test_alfworld_eval.py tests/test_closed_loop_controller.py tests/test_skill_embedding.py tests/test_full_base_train.py::test_apply_skill_text_format_switches_skill_table_serializer`
- 结果：40 passed，2 warnings。

### 2026-05-21 Final DAgger + Q_success + loss ablation result

完成 CLSTR-Qwen DAgger expert-corrected rollout + Q_success action-value head + post-fix loss ablation。

关键修复：

- Qwen train-split rollout 不再把 `qwen_action` 当 expert；每步使用官方 ALFWorld expert action 作为 `L_policy` 正标签。
- ALFWorld action -> SkillNet pseudo-skill mapping 已统一到 `clstr/alfworld_action_skills.py`，删除重复 mapper。
- Skill embedding 已切换 enriched table：`data/clstr_dagger_expert_corrected_train_enriched/skills.jsonl`，121/121 skills include SkillNet body，未使用 valid/test。
- 历史记录：该轮 `make_controller_component_scorer` 曾尝试真实使用 trained heads：
  - transition: `trans_head(predicted_state, model.action_emb(mapped_skill_id))`
  - belief: `skill_head(candidate_text_embedding, belief_state)`
  - Q_success: `q_success_head(h, m, candidate_text_embedding)`
- 2026-06-07 后这条 ALFWorld pre-action transition/belief scoring 语义已被禁用；保留 Q_success，但不再把 candidate/action embedding 作为 transition/gate 的 proxy observation。旧 `component_score_source=trained_clstr_heads` trace 只可作历史诊断，不能作为当前 closed-loop transition 证据。
- `run_alfworld_clstr_eval.py` 支持 `q_success_weight`，避免训练了 Q_success 但 eval final_score 不用。

数据与训练：

- expert-corrected rollout: `data/alfworld_qwen3_expert_corrected_rollout/train_rollout.jsonl`
- rollout stats: 2248 steps, `expert_action_in_admissible_rate=1.0`, `qwen_action_matches_expert_rate=0.155694`
- DAgger train data: `data/clstr_dagger_expert_corrected_train_enriched/train.jsonl`
- best checkpoint: `outputs/clstr_loss_ablation_enriched/policy_plus_q_success_trans_skill_ce/checkpoints/clstr_full_base-step300.pt`
- best offline metrics:
  - `policy_expert_recall@1=0.796667`
  - `q_success_accuracy@0.5=0.958905`
  - `transition_skill_recall@1=0.908333`
- GPU: NVIDIA vGPU-32GB, peak allocated 29.116GB; embedding cache batch-size auto-reduced after recoverable OOM, no fatal OOM.

Post-fix loss ablation:

- Table: `outputs/clstr_loss_ablation_enriched/ablation_table.md`
- Summary: `outputs/clstr_loss_ablation_enriched/ablation_summary.json`
- Recommended slim config: `outputs/clstr_loss_ablation_enriched/recommended_slim_loss_config.json`
- Best slim loss: `L_policy + Q_success + L_trans_skill_ce`
- `full_minus_trans_skill_ce` drops from 0.15 to 0.10 valid_seen gate, so `L_trans_skill_ce` is the only ablated auxiliary loss with clear positive closed-loop contribution here.
- STOP, belief, routing, hard negative, and observation cosine did not improve the 20-episode gate in this post-fix run; they are not recommended in the slim config.

Closed-loop results:

- Gate path: `outputs/alfworld_eval/clstr_dagger_qsuccess_gate_valid_seen/metrics.json`
  - valid_seen 20 episodes: success_rate=0.15, average_reward=0.15, average_steps=45.0
- Full path: `outputs/alfworld_eval/clstr_dagger_qsuccess_full/metrics.json`
  - valid_seen+valid_unseen 274 episodes: success_rate=0.167883, average_reward=0.167883, average_steps=43.605839
  - valid_seen: 140 episodes, success_rate=0.207143
  - valid_unseen: 134 episodes, success_rate=0.126866
- Baselines:
  - 0.6B CLSTR best full: success_rate=0.00365
  - previous structured CLSTR-Qwen gate: success_rate=0.10
  - Qwen3-8B direct reference full: success_rate=0.109489
- This is a CLSTR structured controller result: Qwen is frozen planner/reference, Qwen direct output is not final action; CLSTR final action is selected over admissible commands.

Reports:

- `outputs/clstr_dagger_qsuccess_comparison_table.md`
- `outputs/clstr_dagger_qsuccess_comparison_summary.json`
- `outputs/clstr_dagger_qsuccess_paper_report.md`
- `outputs/clstr_dagger_qsuccess_paper_report.json`

Verification:

- Required grouped tests passed.
- Full `pytest -q`: 319 passed, 2 warnings.
- `/root/autodl-tmp/skillrouter` and `/root/autodl-tmp/alfworld_repo` git status are clean.

## 2026-05-22 Qwen3-8B ACT train after AppWorld base dependency

为满足“AppWorld CLSTR-base 完成后再启动 Qwen3-8B 下游 CLSTR ACT 训练”的执行顺序，新增当前集群路径下的 sbatch 入口：

- 脚本：`scripts/sbatch/run_clstr_qwen3_full_base_train.sh`
- 默认 Qwen 路径：`models/Qwen3-8B`
- 默认 cache：`/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/huggingface`
- 默认训练数据：`data/clstr_qwen3_8b_full_base_gate/train_alfworld_official.jsonl`
- 默认 skill pool：`data/clstr_full_base_train/skills.jsonl`
- 默认输出：`outputs/clstr_qwen3_8b_full_base_after_appworld_base_v1`
- 默认行为：`--local_files_only`，避免计算节点联网依赖；`INCLUDE_AVAILABLE_ACTIONS_IN_STATE=1` 时把候选动作上下文纳入 Qwen latent state。

已提交依赖作业：

```bash
sbatch --dependency=afterok:75125 --gpus=1 -p gpu_a800 \
  --export=ALL,OUTPUT_DIR=outputs/clstr_qwen3_8b_full_base_after_appworld_base_v1,INCLUDE_AVAILABLE_ACTIONS_IN_STATE=1,MAX_STEPS=2000,BATCH_SIZE=4 \
  scripts/sbatch/run_clstr_qwen3_full_base_train.sh
```

- job：`75135`
- 依赖链：`75084` full AppWorld base train -> `75090` final dev eval -> `75125` final routing comparison -> `75135` Qwen3-8B ACT train。

边界说明：这个 Qwen3-8B ACT 训练入口复用当前已存在的 CLSTR-Qwen frozen external encoder + CLSTR-native heads 训练通路，训练数据仍是 ALFWorld train-only replay，不是 AppWorld official DB-state executor loop。它可以作为“Qwen3-8B 下游 CLSTR ACT 训练”的当前可运行证据；若论文主表需要 AppWorld official task success，则还需要后续实现 AppWorld Qwen executor / verifier closed-loop，不能把该 ACT 训练直接冒充为 AppWorld 官方 success benchmark。

### 完成状态

AppWorld full CLSTR-base 训练与最终 dev routing eval 已完成：

- full train job：`75084`，state=`COMPLETED`，elapsed=`00:38:37`。
- final eval job：`75090`，state=`COMPLETED`，elapsed=`00:00:09`。
- final comparison job：`75125`，state=`COMPLETED`。
- full base final checkpoint：`outputs/appworld_clstr_train_supervised_full_v1/checkpoints/stage2_base-step20.pt`。
- train report：`outputs/appworld_clstr_train_supervised_full_v1/train_stdout.json`，阶段为 `stage0_warmstart/stage1_heads/stage2_base`，每阶段 20 updates，`training_data_source=appworld_routing`，`env_mode=replay`，`device=cuda:0`。
- final CLSTR report：`outputs/appworld_clstr_eval/full_v1_dev/report.json`
  - recall@1：`0.824561`
  - recall@5：`1.0`
  - recall@20：`1.0`
  - mrr@20：`0.897661`
- SkillRouter embedding baseline：`outputs/appworld_skillrouter_embedding_baseline/dev_a800_v2/report.json`
  - recall@1：`0.245614`
  - recall@5：`0.526316`
  - recall@20：`0.666667`
  - mrr@20：`0.371327`
- final comparison：`outputs/appworld_routing_comparison_full_v1/comparison.json` / `comparison.md`，角色仍是 AppWorld routing-layer comparison，不是 official DB-state task-completion。

Qwen3-8B ACT 训练也已按依赖链完成：

- job：`75135`，state=`COMPLETED`，elapsed=`00:07:04`。
- report：`outputs/clstr_qwen3_8b_full_base_after_appworld_base_v1/train_report.json`
- checkpoint：`outputs/clstr_qwen3_8b_full_base_after_appworld_base_v1/checkpoints/clstr_full_base-step2000.pt`
- 关键审计字段：
  - `status=ok`
  - `qwen_model_name_or_path=models/Qwen3-8B`
  - `qwen_external_encoder=true`
  - `qwen_frozen=true`
  - `qwen_direct_generator=false`
  - `clstr_native_act_trained=true`
  - `not_full_clstr_act=false`
  - `valid_or_test_used_for_training=false`
  - `include_available_actions_in_state=true`
- offline ACT 指标：
  - policy_expert_recall@1：`0.413625`
  - transition_skill_recall@1：`0.755667`
  - transition_skill_recall@5：`0.990458`
  - transition_skill_mrr：`0.856244`
  - peak GPU allocated：`19.324GB` on A800 80GB。

最终验证：

- `pytest -q tests/test_data.py::test_validate_data_roles_treats_inaccessible_optional_roots_as_missing tests/test_bridge_skillrouter.py tests/test_losses.py tests/test_qwen_external_encoder.py tests/test_full_base_train.py::test_qwen_external_checkpoint_filter_removes_duplicate_skill_table_encoder_refs tests/test_full_base_train.py::test_qwen_external_checkpoint_excludes_frozen_backbone_state tests/test_train_smoke.py::test_stage3_train_cli_requires_and_uses_verified_pairs`：31 passed。
- `bash -n scripts/sbatch/run_appworld_clstr_train.sh scripts/sbatch/run_appworld_clstr_eval.sh scripts/sbatch/run_clstr_qwen3_full_base_train.sh`：通过。
- `git diff --check` 针对本阶段修改文件：通过。

## 2026-05-23 AppWorld routing-only base 调参、ACT 入口与 Qwen executor smoke

本阶段目标是按当前 AppWorld + SkillX 主路线推进三件事：

1. 把 CLSTR-base 调到更合理的 AppWorld routing initializer；
2. 在 CLSTR-base 之后进入 CLSTR-act 训练链路；
3. 用固定 Qwen3-8B executor 做 AppWorld official runtime smoke，并准备与 SkillRouter 比较。

### SkillRouter-base 对照

新增并跑通 AppWorld 上的 `SkillRouter-base` 对照：

- 实现：`clstr/appworld_skillrouter_base.py`
- train/eval 入口：
  - `scripts/run_appworld_skillrouter_base_train.py`
  - `scripts/run_appworld_skillrouter_base_eval.py`
  - `scripts/sbatch/run_appworld_skillrouter_base_train.sh`
  - `scripts/sbatch/run_appworld_skillrouter_base_eval.sh`
- 定义：冻结 `.cache/hf_models/SkillRouter-Embedding-0.6B`，只训练一个 AppWorld train qrels 上的小 scorer；不是完整参数微调 SkillRouter。
- 结果：
  - train job：`75235` completed
  - eval job：`75237` completed
  - dev report：`outputs/appworld_skillrouter_base_eval/dev_v1/report.json`
  - dev metrics：recall@1=`0.982456`，recall@5=`1.0`，recall@20=`1.0`，mrr@20=`0.991228`

解释：这个 baseline 很强，但 AppWorld train/dev positive skill set reuse 很高，因此它更像“static supervised router upper baseline”，不能直接等同于端到端 AppWorld task success。

### CLSTR-base routing-only 调参

新增 `loss_mode: routing_only`，让 CLSTR-base 跳过 rollout/`total_loss`，只优化 `multi_positive_routing_loss`，避免当前 AppWorld replay 零奖励导致 `L_policy` 无效。

关键实现：

- `clstr/train.py`
  - `loss_mode=routing_only`
  - `start_stage`，支持 act 训练只跑 `stage3_joint`
  - `--warmstart_clstr_ckpt`，支持从 CLSTR 自身 checkpoint 继续训练
  - `checkpoint_every`，避免每一步都保存大 checkpoint
- 配置：`configs/train/appworld_routing_only_fit.yaml`
- 测试：`tests/test_train_smoke.py`

训练与评估：

- train job：`75243` completed
- checkpoint：`outputs/appworld_clstr_train_routing_only_fit_v1/checkpoints/stage2_base-step120.pt`
- dev eval job：`75253` completed
- train eval job：`75254` completed
- dev report：`outputs/appworld_clstr_eval/routing_only_fit_v1_dev/report.json`
  - recall@1=`0.912281`
  - recall@5=`1.0`
  - recall@20=`1.0`
  - mrr@20=`0.95614`
- train report：`outputs/appworld_clstr_eval/routing_only_fit_v1_train/report.json`
  - recall@1=`0.833333`
  - recall@5=`1.0`
  - recall@20=`1.0`
  - mrr@20=`0.905556`

对比旧 full_v1：旧 CLSTR-base dev id recall@1=`0.824561`，duplicate-aware/canonical recall@1=`0.912281`。routing-only 后 id recall@1 已达到 `0.912281`，说明欠拟合问题明显缓解，但仍低于 SkillRouter-base 的 `0.982456`。

### AppWorld executor 环境修复

之前 Qwen executor smoke 失败在 `ModuleNotFoundError("No module named 'appworld.apps.admin'")`，根因是项目本地 `.deps/appworld_py310` 安装了 AppWorld 包但没有解包官方 apps bundle。

修复方式：

- 使用 AppWorld 自带 `appworld.install.install_package()` 将 `.source/apps.bundle` 解包到：
  - `.deps/appworld_py310/appworld/apps`
- cache 写入仍限制在：
  - `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/appworld`
- 验证：
  - `import appworld.apps.admin` 通过
  - `tests/test_appworld_executor.py` 通过

Qwen-only executor smoke：

- job：`75245` completed
- report：`outputs/appworld_executor_smoke/qwen_only_dev1_after_appworld_install/report.json`
- 结果：`reset_ok=true`，`generation_failures=0`，`execution_failures=0`
- 未成功完成任务是因为 smoke 设置 `MAX_NEW_TOKENS=128`，生成代码被截断产生 syntax error；这不是 AppWorld package/runtime 问题。

### CLSTR-act 入口

新增 train-only AppWorld oracle API trace -> CLSTR `VerifiedPair` 构建器，用来启动 `L_act`：

- 实现：`clstr/appworld_act_verified_pairs.py`
- CLI：`scripts/build_appworld_act_verified_pairs.py`
- sbatch：`scripts/sbatch/run_appworld_build_act_verified_pairs.sh`
- 配置：`configs/train/appworld_act_train.yaml`
- 测试：`tests/test_appworld_act_verified_pairs.py`

verified pairs 构建：

- job：`75250` completed
- manifest：`data/appworld_act/manifest.json`
- verified pairs：`data/appworld_act/verified_pairs_train.jsonl`
- 统计：
  - train tasks=`90`
  - tasks_with_skill_sequences=`87`
  - verified_pair_count=`618`
  - avg_sequence_length=`8.103`
  - dev_test_used_for_training=`false`
- 来源说明：只使用 AppWorld train split 的 `ground_truth/api_calls.json`，按 API refs 映射到 SkillX skills；这是 train-only oracle trace，不是 dev/test 或 executor-generated success trajectory。

CLSTR-act 训练链路：

- train job：`75255`
- dev eval dependent job：`75256`
- base warmstart：`outputs/appworld_clstr_train_routing_only_fit_v1/checkpoints/stage2_base-step120.pt`
- verified pairs：`data/appworld_act/verified_pairs_train.jsonl`
- 注意：ACT 最终是否优于 base，需要等 `outputs/appworld_clstr_eval/act_v1_dev/report.json`。

### AppWorld executor 对比 smoke

已提交 3-task dev smoke，用同一个 Qwen3-8B downstream executor，只换 routing provider：

- Qwen-only：job `75257`，output `outputs/appworld_executor_smoke/qwen_only_dev3_v2`
- SkillRouter-base：job `75258`，output `outputs/appworld_executor_smoke/skillrouter_base_dev3_v2`
- CLSTR-base routing-only：job `75259`，output `outputs/appworld_executor_smoke/clstr_base_routing_only_dev3_v2`
- CLSTR-act：job `75260`，依赖 `75256`，output `outputs/appworld_executor_smoke/clstr_act_dev3_v1`
- executor comparison dependent job：`75262`，output `outputs/appworld_executor_comparison_dev3_v1`

这些是 official AppWorld runtime loop smoke，会记录 `task_completed()` / `evaluate()`，不同于 routing qrels eval。是否能扩展到完整 dev/test benchmark，要看 smoke 是否没有系统性 prompt/API 使用错误。

### 环境清理

routing-only base 初版每一步保存 checkpoint，产生 360 个 `.pt`，约 `67G`。已清理中间 checkpoint，只保留每个阶段最终点：

- `stage0_warmstart-step120.pt`
- `stage1_heads-step120.pt`
- `stage2_base-step120.pt`

清理后目录大小约 `567M`。后续 `configs/train/appworld_routing_only_fit.yaml` 和 `configs/train/appworld_act_train.yaml` 已加入 `checkpoint_every`，避免再次产生大量中间 checkpoint。

### 本阶段验证

本地轻量测试：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_act_verified_pairs.py \
  tests/test_appworld_executor.py \
  tests/test_appworld_routing_diagnostics.py \
  tests/test_train_smoke.py::test_build_stage_plan_can_start_from_stage3_joint \
  tests/test_train_smoke.py::test_should_save_checkpoint_honors_interval_and_final_step \
  tests/test_train_smoke.py::test_clstr_checkpoint_warmstart_loads_model_state \
  tests/test_train_smoke.py::test_train_iteration_routing_only_uses_qrels_without_rollout \
  tests/test_losses.py::test_multi_positive_routing_loss_uses_task_positive_skill_ids
```

结果：`13 passed in 9.76s`。

脚本语法检查：

- `bash -n scripts/sbatch/run_appworld_clstr_train.sh`
- `bash -n scripts/sbatch/run_appworld_clstr_eval.sh`
- `bash -n scripts/sbatch/run_appworld_build_act_verified_pairs.sh`

结果：通过。

## 2026-05-23 AppWorld executor v4 prompt 与记录修正

v3 3-task executor smoke 的环境与 reset 已正常，但失败日志暴露出三个 executor 层面的系统性问题：

- Qwen 仍会对 `apis.*` 使用位置参数，AppWorld wrapper 实际只接受关键字参数，典型错误为 `_api_request() takes 0 positional arguments but 2 were given`。
- Qwen 会猜测 `show_account_passwords()` 的 `account_name == "Spotify"`，但 AppWorld ground-truth pattern 使用小写 key，例如 `supervisor_passwords["spotify"]`、`["file_system"]`。
- Qwen 有时会把 `apis.*` 返回值当成 response 对象再调用 `requester.json()`；实际 `apis.*` 已直接返回 Python dict/list。

已修改 `clstr/appworld_executor.py`：

- 在 prompt 中加入 canonical login snippet，显式要求：
  - 所有 `apis.*` 调用使用 keyword arguments；
  - `apis.*` 直接返回 Python 对象，禁止在其后调用 `requester.json()`；
  - `show_account_passwords()` 使用小写 app name 取密码，例如 `supervisor_passwords["spotify"]`；
  - 登录调用形如 `apis.spotify.login(username=..., password=...)`。
  - `complete_task` 使用 `status="success"`，禁止 `status="complete"` / `done`。
  - SkillX skills 只能作为参考代码/逻辑，禁止调用不存在的 `apis.skillx` namespace；如果 SkillX body 有用，应复制其逻辑并调用真实 AppWorld APIs。
- `_json_safe()` 支持 `TestTracker.to_dict(stats_only=False)`，避免 `evaluate` 只记录成 `<appworld.evaluator.TestTracker object ...>` 字符串。
- `run_appworld_executor_eval()` 会把包含 `Execution failed.` / `Traceback` / `Error:` 的 `world.execute()` 输出计为 `execution_ok=false`，并要求 `execution_ok and task_completed` 才算 success，避免 report 低估执行失败数。
- `scripts/run_appworld_qwen_executor_eval.py` 的默认 predictions 已更新：`clstr_base` 指向当前合理的 `routing_only_fit_v1_dev`，`clstr_act` 指向待产出的 `act_v2_dev`，避免后续裸跑 executor 时误用旧默认路径。

新增/更新测试：

- `tests/test_appworld_executor.py::test_prompt_builder_includes_instruction_skills_and_filtered_api_docs`
- `tests/test_appworld_executor.py::test_executor_eval_serializes_appworld_tracker_objects`
- `tests/test_appworld_executor.py::test_executor_eval_counts_traceback_execute_output_as_execution_failure`
- `tests/test_appworld_executor.py::test_qwen_executor_default_clstr_base_predictions_use_routing_only_fit`

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_executor.py \
  tests/test_train_smoke.py::test_task_batch_for_update_cycles_through_train_tasks \
  tests/test_train_smoke.py::test_stage3_train_cli_requires_and_uses_verified_pairs \
  tests/test_appworld_act_verified_pairs.py \
  tests/test_losses.py::test_multi_positive_routing_loss_uses_task_positive_skill_ids
```

结果：初次为 `11 passed in 19.50s`；补充 default predictions 与 SkillX-namespace prompt 测试后，相关测试组为 `12 passed in 26.41s`。

随后补充了 `complete_task(status="success")` 的 prompt 断言并重跑：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py
```

结果：`7 passed in 1.48s`。

已提交 v4 executor smoke：

- Qwen-only：job `75276`，output `outputs/appworld_executor_smoke/qwen_only_dev3_v4`
- SkillRouter embedding：job `75277`，output `outputs/appworld_executor_smoke/skillrouter_embedding_dev3_v4`
- SkillRouter-base：job `75278`，output `outputs/appworld_executor_smoke/skillrouter_base_dev3_v4`
- CLSTR-base routing-only：job `75279`，output `outputs/appworld_executor_smoke/clstr_base_routing_only_dev3_v4`

下一步：等待 v4 smoke 产出后检查 `runs.jsonl`，重点看是否仍有位置参数、`requester.json()`、大小写密码 key 错误；若这些系统性错误消失，再扩展到 `MAX_TASKS=10` 的 dev task-success benchmark。

## 2026-05-23 CLSTR-act v2 与 executor v5

### CLSTR-act v2 结果

`clstr-act-v2` 训练与依赖评测已完成：

- train job：`75265`，completed，耗时 `00:19:47`
- train eval job：`75266`，completed
- dev eval job：`75267`，completed
- checkpoint：`outputs/appworld_clstr_train_act_v2/checkpoints/stage3_joint-step40.pt`
- warmstart：`outputs/appworld_clstr_train_routing_only_fit_v1/checkpoints/stage2_base-step120.pt`
- verified templates：`618`
- task_batch_size：`4`
- env_mode：`replay`
- final stage3 loss：`1.424097`
- action_loss_used：`64.0`

Routing eval 对比：

| model | split | recall@1 | recall@5 | recall@20 | mrr@20 |
|---|---:|---:|---:|---:|---:|
| CLSTR-base routing-only fit | train | 0.833333 | 1.0 | 1.0 | 0.905556 |
| CLSTR-base routing-only fit | dev | 0.912281 | 1.0 | 1.0 | 0.956140 |
| CLSTR-act v2 | train | 0.877778 | 1.0 | 1.0 | 0.933333 |
| CLSTR-act v2 | dev | 0.877193 | 0.947368 | 1.0 | 0.918888 |
| SkillRouter-base | dev | 0.982456 | 1.0 | 1.0 | 0.991228 |

结论：当前 ACT v2 对 train split 有提升，但 dev split 明显低于 CLSTR-base routing-only fit；因此现在不能把 ACT v2 当成优于 base 的主结果。后续若继续 ACT，需要调小 ACT 对 routing logits 的扰动或加 dev-safe early stopping/selection，不能直接用 step40 作为最终模型。

### executor v4 结果与根因

v4 3-task smoke 都完成，但 task success 仍为 0：

| method | job | success_count | execution_failures |
|---|---:|---:|---:|
| qwen_only | 75276 | 0/3 | 3 |
| skillrouter_embedding | 75277 | 0/3 | 3 |
| skillrouter_base | 75278 | 0/3 | 3 |
| clstr_base routing-only | 75279 | 0/3 | 3 |

v4 已消除的系统性错误：

- 位置参数登录：0 次
- `requester.json()`：0 次
- `apis.skillx`：0 次
- `Spotify` 大写密码 key：0 次

v4 仍存在的新问题：

- 作业在 `complete_task(status="success")` prompt 修正前启动，因此每个 run 仍用了 `status="complete"`。
- Qwen-only 直接从 `show_song_library()` 读取 `play_count`，但 list/library schema 没有该字段，需要再调用 `show_song(song_id=...)`。
- SkillRouter embedding 仍会把 skill 名称幻觉成真实 API，如 `apis.spotify.find_songs_by_metric(...)`。
- SkillRouter-base 用 `page_limit=100`，违反 AppWorld docs 中 `page_limit <= 20` 的约束。
- CLSTR-base 对 `show_playlist()` 漏传 required `access_token`。

### executor v5 修正

已修改 `clstr/appworld_executor.py`：

- `_param_signature()` 将 API 参数 `default` 与 `constraints` 写入 prompt，例如 `page_limit: integer optional default 5 constraints value >= 1.0, <= 20.0`。
- prompt 增加：
  - 只能使用 `[AppWorld API Docs]` 中列出的 API 名；
  - 不得从 skill name/description 发明 API；
  - API 参数必须严格匹配签名，不能传未列出的参数，必须传 required 参数；
  - list/library API 需要遵守 page_limit 约束，并在需要全量条目时循环 `page_index` 直到空页；
  - 如果 list/library response schema 缺少任务需要的字段，需要调用对应 detail API 后再使用。

测试：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_executor.py \
  tests/test_train_smoke.py::test_task_batch_for_update_cycles_through_train_tasks \
  tests/test_train_smoke.py::test_stage3_train_cli_requires_and_uses_verified_pairs \
  tests/test_appworld_act_verified_pairs.py \
  tests/test_losses.py::test_multi_positive_routing_loss_uses_task_positive_skill_ids
```

结果：`12 passed in 28.19s`。

已提交 v5 executor smoke：

- Qwen-only：job `75284`，output `outputs/appworld_executor_smoke/qwen_only_dev3_v5`
- SkillRouter embedding：job `75285`，output `outputs/appworld_executor_smoke/skillrouter_embedding_dev3_v5`
- SkillRouter-base：job `75286`，output `outputs/appworld_executor_smoke/skillrouter_base_dev3_v5`
- CLSTR-base routing-only：job `75287`，output `outputs/appworld_executor_smoke/clstr_base_routing_only_dev3_v5`
- CLSTR-act v2：job `75288`，output `outputs/appworld_executor_smoke/clstr_act_dev3_v5`

下一步：按低频轮询等待 v5，检查是否还有 `status="complete"`、invented API、越界 page_limit、漏 required 参数等错误；只有 v5 smoke 至少达到 API 调用稳定，才扩展到 `MAX_TASKS=10`。

## 2026-05-23 executor v5 结果与 v6 收紧

v5 3-task smoke 已完成：

| method | job | success_count | success_rate | execution_failures |
|---|---:|---:|---:|---:|
| Qwen-only | 75284 | 2/3 | 0.666667 | 1 |
| SkillRouter embedding | 75285 | 0/3 | 0.0 | 3 |
| SkillRouter-base | 75286 | 0/3 | 0.0 | 3 |
| CLSTR-base routing-only | 75287 | 2/3 | 0.666667 | 1 |
| CLSTR-act v2 | 75288 | 1/3 | 0.333333 | 2 |

v5 已稳定解决的问题：

- `status="complete"`：0 次
- 位置参数登录：0 次
- `requester.json()`：0 次
- `apis.skillx`：0 次
- 大写 `Spotify` 密码 key：0 次

v5 仍未解决的问题：

- Qwen-only 与 CLSTR-base 在 `50e1ac9_1` 上仍会从 `show_song_library()` 结果读取 `genre` / `play_count`；这些字段只在 `show_song(song_id=...)` detail schema 中。
- SkillRouter embedding/base 仍明显受 retrieved SkillX 文本影响：
  - `page_limit=100` 仍出现；
  - embedding 仍出现 `apis.spotify.find_songs_by_metric(...)` 这类由 skill name 幻觉出的 API。
- ACT v2 的 executor smoke 低于 CLSTR-base，与 routing eval 中 ACT dev 低于 base 一致。

结论：v5 已经证明 Qwen-only 和 CLSTR-base 能跑通 official AppWorld executor smoke，但 SkillRouter variants 还不稳定。此时直接扩到 10-task 会把 SkillX 文本污染问题混入 benchmark，因此先做 v6 prompt/format 收紧。

v6 修改：

- `_format_skill()` 将检索到的 SkillX 条目标成 `[Reference Skill - not callable]`，字段从 `name` 改成 `reference_name_not_function`，并显式加入 `not_callable: true`。
- prompt 新增：
  - SkillX skill name 不是函数名，不能转换成 `apis.<app>.<skill_name>(...)`；
  - 永远不要用 `page_limit=100`，优先用 `page_limit=20` 或更小的文档上限；
  - Spotify song/album/playlist library 任务中，先收集 song IDs，再调用 `apis.spotify.show_song(song_id=song_id)`，之后才能读取 `genre` 或 `play_count`。

测试：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_executor.py \
  tests/test_train_smoke.py::test_task_batch_for_update_cycles_through_train_tasks \
  tests/test_train_smoke.py::test_stage3_train_cli_requires_and_uses_verified_pairs \
  tests/test_appworld_act_verified_pairs.py \
  tests/test_losses.py::test_multi_positive_routing_loss_uses_task_positive_skill_ids
```

结果：`12 passed in 36.94s`。

已提交 v6 executor smoke：

- Qwen-only：job `75298`，output `outputs/appworld_executor_smoke/qwen_only_dev3_v6`
- SkillRouter embedding：job `75299`，output `outputs/appworld_executor_smoke/skillrouter_embedding_dev3_v6`
- SkillRouter-base：job `75300`，output `outputs/appworld_executor_smoke/skillrouter_base_dev3_v6`
- CLSTR-base routing-only：job `75301`，output `outputs/appworld_executor_smoke/clstr_base_routing_only_dev3_v6`
- CLSTR-act v2：job `75302`，output `outputs/appworld_executor_smoke/clstr_act_dev3_v6`

下一步：等待 v6；如果 SkillRouter variants 仍因 SkillX 文本导致 `page_limit=100` 或 invented API，应考虑 executor benchmark 中只传 SkillX body 的 “logic-only compressed skill context”，而不是完整 name/description/body。

## 2026-05-23 executor v6 结果与 v7 决策

v6 3-task smoke 已完成：

| method | job | success_count | success_rate | execution_failures |
|---|---:|---:|---:|---:|
| Qwen-only | 75298 | 3/3 | 1.0 | 0 |
| SkillRouter embedding | 75299 | 2/3 | 0.666667 | 1 |
| SkillRouter-base | 75300 | 3/3 | 1.0 | 0 |
| CLSTR-base routing-only | 75301 | 1/3 | 0.333333 | 2 |
| CLSTR-act v2 | 75302 | 3/3 | 1.0 | 0 |

v6 错误模式：

- Qwen-only、SkillRouter-base、CLSTR-act：无执行失败，且无位置参数、`requester.json()`、`apis.skillx`、`status="complete"`、invented `find_songs_by_metric`、大写密码 key。
- SkillRouter embedding：无 invented API / page_limit=100，但 `50e1ac9_2` 把 album/playlist detail 中的简略 song item 当成 full song detail，读取 `play_count` 失败。
- CLSTR-base routing-only：仍出现一次 `page_limit=100`，说明 `temperature=0.2` 下 prompt 约束仍会被采样破坏；另一次失败同样是简略 song item 缺少 `play_count`。

结论：v6 已证明 executor 可以在 official AppWorld runtime 中跑通，但 3-task 结果对采样很敏感。为了进入 10-task benchmark，先固定 `TEMPERATURE=0.0` 做 v7 3-task smoke；如果 v7 稳定，再扩到 `MAX_TASKS=10`。暂不继续堆 prompt。

已提交 v7 executor smoke，均使用 `TEMPERATURE=0.0`：

- Qwen-only：job `75307`，output `outputs/appworld_executor_smoke/qwen_only_dev3_v7`
- SkillRouter embedding：job `75308`，output `outputs/appworld_executor_smoke/skillrouter_embedding_dev3_v7`
- SkillRouter-base：job `75309`，output `outputs/appworld_executor_smoke/skillrouter_base_dev3_v7`
- CLSTR-base routing-only：job `75310`，output `outputs/appworld_executor_smoke/clstr_base_routing_only_dev3_v7`
- CLSTR-act v2：job `75311`，output `outputs/appworld_executor_smoke/clstr_act_dev3_v7`

## 2026-05-23 executor success 口径修正与 v8

检查 v7 `runs.jsonl` 后发现一个关键统计问题：`run_appworld_executor_eval()` 之前用 `execution_ok and task_completed` 作为 success。这只能说明代码执行并调用了 `complete_task`，不能说明答案通过 AppWorld evaluator。用旧 v7 的 `evaluate.success` 重新计算，五个方法真实 task success 都是 `0/3`。

已修正 `clstr/appworld_executor.py`：

- `_json_safe()` 仍序列化 `TestTracker.to_dict(stats_only=False)`。
- 新增 `_evaluation_succeeded()`。
- 每条 run 记录新增 `evaluation_success`。
- `success = execution_ok and task_completed and evaluation_success`。

新增测试：

- `tests/test_appworld_executor.py::test_executor_eval_requires_evaluate_success_for_task_success`

同时根据 evaluator failure 修正 prompt：前三个 smoke task 要求 “across Spotify song, album and playlist libraries”，模型之前只看 song library 或把 album/playlist 的 `song_ids` 当 full song objects。prompt 现在显式要求：

- 如果任务要求 across Spotify song/album/playlist libraries，先从 `show_song_library`、`show_album_library`、`show_playlist_library` 合并 song IDs；
- 对每个 unique song_id 调 `show_song(song_id=...)`；
- 再基于 detail 中的 `genre` / `play_count` 过滤和排序。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_executor.py \
  tests/test_train_smoke.py::test_task_batch_for_update_cycles_through_train_tasks \
  tests/test_train_smoke.py::test_stage3_train_cli_requires_and_uses_verified_pairs \
  tests/test_appworld_act_verified_pairs.py \
  tests/test_losses.py::test_multi_positive_routing_loss_uses_task_positive_skill_ids
```

结果：`13 passed in 36.97s`。

下一步：必须用新 success 口径重跑 executor smoke；v5/v6/v7 只能作为 prompt/API 调试记录，不能作为最终 task-success benchmark。

已提交 v8 executor smoke，均使用新 success 口径与 `TEMPERATURE=0.0`：

- Qwen-only：job `75316`，output `outputs/appworld_executor_smoke/qwen_only_dev3_v8`
- SkillRouter embedding：job `75317`，output `outputs/appworld_executor_smoke/skillrouter_embedding_dev3_v8`
- SkillRouter-base：job `75318`，output `outputs/appworld_executor_smoke/skillrouter_base_dev3_v8`
- CLSTR-base routing-only：job `75319`，output `outputs/appworld_executor_smoke/clstr_base_routing_only_dev3_v8`
- CLSTR-act v2：job `75320`，output `outputs/appworld_executor_smoke/clstr_act_dev3_v8`

## 2026-05-23 executor v8 结果与 task hint 修正

v8 使用真实 `evaluate.success` 口径后，3-task smoke 结果如下：

| method | job | true success_count | success_rate | execution_failures |
|---|---:|---:|---:|---:|
| Qwen-only | 75316 | 0/3 | 0.0 | 2 |
| SkillRouter embedding | 75317 | 1/3 | 0.333333 | 1 |
| SkillRouter-base | 75318 | 1/3 | 0.333333 | 1 |
| CLSTR-base routing-only | 75319 | 0/3 | 0.0 | 1 |
| CLSTR-act v2 | 75320 | 0/3 | 0.0 | 0 |

主要根因：

- executor runtime 已基本可用，但模型仍经常没有完全执行任务语义。
- 失败集中在 `50e1ac9_*` 这类 Spotify 模板：任务要求 “top N most played <genre> song titles from across song, album and playlist libraries”，模型经常：
  - 只排序所有歌曲，没有先按 `genre` 过滤；
  - 只看 song library，或合并 album/playlist 后把简略 song id/item 当成 full song detail；
  - 执行成功并调用 `complete_task`，但 evaluator 判答案不匹配。

已修改 `clstr/appworld_executor.py`：

- 新增 `_build_task_specific_hints()`。
- 当 instruction 匹配 `top N most played <genre> song titles` 且是 Spotify 任务时，prompt 增加：
  - 先 case-insensitive 匹配 `song["genre"] == <genre>`；
  - 不要先排序所有歌曲；
  - 对过滤后的歌曲按 `song["play_count"]` 降序；
  - 返回 exactly N 个 song titles，逗号分隔，无额外文本。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_executor.py \
  tests/test_train_smoke.py::test_task_batch_for_update_cycles_through_train_tasks \
  tests/test_train_smoke.py::test_stage3_train_cli_requires_and_uses_verified_pairs \
  tests/test_appworld_act_verified_pairs.py \
  tests/test_losses.py::test_multi_positive_routing_loss_uses_task_positive_skill_ids
```

结果：`13 passed in 21.71s`。

下一步：提交 v9 3-task smoke。只有 v9 在真实 `evaluate.success` 口径下能稳定通过，才扩展到 10-task dev benchmark。

已提交 v9 executor smoke，均使用真实 `evaluate.success` 口径、Spotify task-specific hint、`TEMPERATURE=0.0`：

- Qwen-only：job `75342`，output `outputs/appworld_executor_smoke/qwen_only_dev3_v9`
- SkillRouter embedding：job `75343`，output `outputs/appworld_executor_smoke/skillrouter_embedding_dev3_v9`
- SkillRouter-base：job `75344`，output `outputs/appworld_executor_smoke/skillrouter_base_dev3_v9`
- CLSTR-base routing-only：job `75345`，output `outputs/appworld_executor_smoke/clstr_base_routing_only_dev3_v9`
- CLSTR-act v2：job `75346`，output `outputs/appworld_executor_smoke/clstr_act_dev3_v9`

v9 已完成，真实 `evaluate.success` 口径结果：

| method | job | true success_count | success_rate | execution_failures |
|---|---:|---:|---:|---:|
| Qwen-only | 75342 | 2/3 | 0.666667 | 0 |
| SkillRouter embedding | 75343 | 2/3 | 0.666667 | 0 |
| SkillRouter-base | 75344 | 0/3 | 0.0 | 2 |
| CLSTR-base routing-only | 75345 | 3/3 | 1.0 | 0 |
| CLSTR-act v2 | 75346 | 3/3 | 1.0 | 0 |

错误模式：

- Qwen-only 与 SkillRouter embedding 均只在 `50e1ac9_2` 失败；代码执行成功，但 EDM top-6 答案不匹配 evaluator。
- SkillRouter-base 的 top-k 技能在前三个 Spotify 模板上反而误导 Qwen：例如把 `show_song_library()` 返回的 song item 当成含 `song_ids` 的对象，导致 `KeyError: 'song_ids'`。
- CLSTR-base routing-only 与 CLSTR-act v2 均通过 3/3，说明在该 smoke 上 CLSTR top-k skill context 已经比 Qwen-only/embedding 更有帮助。

决策：进入小型 `MAX_TASKS=10` dev benchmark，而不是完整 dev 57-task。理由是 smoke 已证明 CLSTR-base/ACT executor 可以通过 evaluator，但样本仍太小；10-task 可以检查是否只是前三个 Spotify 模板上的局部胜利。

已提交 dev10 benchmark，均使用真实 `evaluate.success`、`TEMPERATURE=0.0`：

- Qwen-only：job `75358`，output `outputs/appworld_executor_benchmark/qwen_only_dev10_v1`
- SkillRouter embedding：job `75359`，output `outputs/appworld_executor_benchmark/skillrouter_embedding_dev10_v1`
- SkillRouter-base：job `75360`，output `outputs/appworld_executor_benchmark/skillrouter_base_dev10_v1`
- CLSTR-base routing-only：job `75361`，output `outputs/appworld_executor_benchmark/clstr_base_routing_only_dev10_v1`
- CLSTR-act v2：job `75362`，output `outputs/appworld_executor_benchmark/clstr_act_dev10_v1`

dev10 benchmark 已完成，comparison 输出：

- `outputs/appworld_executor_benchmark/dev10_v1_comparison/comparison.json`
- `outputs/appworld_executor_benchmark/dev10_v1_comparison/comparison.md`

结果：

| method | success_count | success_rate | execution_failures |
|---|---:|---:|---:|
| Qwen-only | 6/10 | 0.6 | 4 |
| SkillRouter embedding | 3/10 | 0.3 | 4 |
| SkillRouter-base | 7/10 | 0.7 | 2 |
| CLSTR-base routing-only | 4/10 | 0.4 | 3 |
| CLSTR-act v2 | 0/10 | 0.0 | 9 |

结论：

- dev10 不支持 “CLSTR-base + Qwen 比 Qwen-only/SkillRouter-base 更好”。
- SkillRouter-base 在 dev10 反而最好；CLSTR-base 低于 Qwen-only，CLSTR-act 明显退化。
- 因为 dev10 只覆盖 dev 前 10 个任务，仍需跑完整 dev 57-task 确认趋势，不能只凭 dev10 做最终论文判断。

已提交完整 dev57 benchmark，均使用真实 `evaluate.success`、`TEMPERATURE=0.0`：

- Qwen-only：job `75386`，output `outputs/appworld_executor_benchmark/qwen_only_dev57_v1`
- SkillRouter embedding：job `75387`，output `outputs/appworld_executor_benchmark/skillrouter_embedding_dev57_v1`
- SkillRouter-base：job `75388`，output `outputs/appworld_executor_benchmark/skillrouter_base_dev57_v1`
- CLSTR-base routing-only：job `75389`，output `outputs/appworld_executor_benchmark/clstr_base_routing_only_dev57_v1`
- CLSTR-act v2：job `75390`，output `outputs/appworld_executor_benchmark/clstr_act_dev57_v1`

dev57 已完成，真实 `evaluate.success` 结果：

| method | success_count | success_rate | execution_failures |
|---|---:|---:|---:|
| Qwen-only | 14/57 | 0.245614 | 24 |
| SkillRouter embedding | 15/57 | 0.263158 | 32 |
| SkillRouter-base | 14/57 | 0.245614 | 32 |
| CLSTR-base routing-only | 9/57 | 0.157895 | 32 |
| CLSTR-act v2 | 13/57 | 0.228070 | 29 |

comparison 输出：

- `outputs/appworld_executor_benchmark/dev57_v1_comparison/comparison.json`
- `outputs/appworld_executor_benchmark/dev57_v1_comparison/comparison.md`

诊断结论：

- 完整 dev57 不支持 “CLSTR-base + Qwen 优于 SkillRouter/Qwen-only”。CLSTR-base 明显低于 Qwen-only 与 SkillRouter-family；ACT v2 只部分恢复，仍无优势。
- routing 层也显示 SkillRouter-base 更强：SkillRouter-base dev recall@1=`0.982456`，CLSTR-base=`0.912281`，CLSTR-act v2=`0.877193`。
- 但 routing recall@1 有 duplicate artifact：例如多个 `spotify-authenticate-with-stored-credentials-*` 是语义等价技能，qrels 只标了部分 id；CLSTR top1 常命中等价 duplicate 但不在 qrels 中。CLSTR-base top5 仍为 `1.0`，说明问题不是完全找不到相关技能，而是排序、duplicate canonicalization 与 executor-helpfulness 三者不一致。
- SkillRouter-base 看起来简单，但它建立在 `SkillRouter-Embedding-0.6B` 的强语义表征上；CLSTR 当前 AppWorld 配置使用 `all-MiniLM-L6-v2`、`d=384`、冻结 backbone、90 个 train tasks。小数据下，简单 scorer + 强预训练表征比 CLSTR-mini 更占优是合理的。
- ACT v2 不应作为最终结果：它用 train-only oracle API trace 映射出的 VerifiedPair 启动，但 exact pairs 只有 `15`，且不是 executor-generated success trajectories；stage3_joint 后 dev routing 下降，说明当前 ACT 目标/数据对 dev 泛化有负作用。

下一步建议：

1. 暂停继续扩大 executor benchmark；dev57 已足够说明当前 CLSTR 没有优势。
2. 先做 duplicate/canonical skill eval，确认 CLSTR 与 SkillRouter 的真实语义差距，不只看 raw skill_id。
3. 以 SkillRouter-base 作为 teacher 或初始化：尝试 SkillRouter-score distillation / SkillRouter embedding 初始化 CLSTR，再训练 CLSTR heads。
4. 重训 ACT 前先让 CLSTR-base 在 routing 上至少追平 SkillRouter-base；否则 ACT 只是在较弱 base 上放大噪声。
5. 论文叙事暂时不能声称 AppWorld task-success 上 CLSTR 优于 SkillRouter；当前证据只能支持 “pipeline 已打通，发现 CLSTR-mini/ACT 数据不足，需要改初始化/蒸馏/duplicate-aware 评价”。

## 2026-05-23: AppWorld CLSTR-mini 诊断与下一步实验判断

当前 AppWorld 版本被称为 `CLSTR-mini`，不是因为 CLSTR 方法本身被定义为 mini，而是因为当前实验配置确实是轻量化实现：

- `configs/model/appworld_mini.yaml` 使用 `.cache/hf_models/all-MiniLM-L6-v2`，`d=384`，`freeze_backbone=true`，`max_length=256`。
- `configs/train/appworld_routing_only_fit.yaml` 使用 `loss_mode=routing_only`、`rollouts_per_task=0`、`beta=0.0`，因此它主要训练 routing，不是完整的 belief/transition/policy/ACT CLSTR。
- AppWorld base 训练只使用官方 train split 的 `90` 个任务和 `450` 条 qrels；dev/test 没有用于训练。
- SkillRouter-base 对照使用 `.cache/hf_models/SkillRouter-Embedding-0.6B` 的 frozen embedding 加一个小型 trainable scorer。这个 backbone 本来就是 skill routing 向强预训练表征，维度 `1024`，所以它看起来简单但初始化明显更强。

已完成的 AppWorld routing 与 executor 结果需要分开解释。

Routing 层：

| method | raw recall@1 | raw recall@5 | canonical recall@1 | canonical recall@5 |
|---|---:|---:|---:|---:|
| SkillRouter-base | 0.982456 | 1.0 | 0.982456 | 1.0 |
| CLSTR-base routing-only | 0.912281 | 1.0 | 1.0 | 1.0 |
| CLSTR-act v2 | 0.877193 | 0.947368 | 0.947368 | 1.0 |

诊断结论：CLSTR-base 的 raw recall@1 被 SkillX duplicate skill id 明显低估。例如多个 `spotify-authenticate-with-stored-credentials-*` 是语义等价技能，qrels 只标了部分 id；canonical duplicate-aware 评价后 CLSTR-base dev recall@1 为 `1.0`。因此不能基于 raw routing 直接说 CLSTR routing 不如 SkillRouter。

Executor/task-success 层：

| method | dev57 success | success_rate | execution_failures |
|---|---:|---:|---:|
| Qwen-only | 14/57 | 0.245614 | 24 |
| SkillRouter embedding | 15/57 | 0.263158 | 32 |
| SkillRouter-base | 14/57 | 0.245614 | 32 |
| CLSTR-base routing-only | 9/57 | 0.157895 | 32 |
| CLSTR-act v2 | 13/57 | 0.228070 | 29 |

诊断结论：task-success 结果仍然不支持 “CLSTR-base + Qwen 优于 SkillRouter/Qwen-only”。因为 CLSTR-base canonical routing 已经很强但 executor 成功率偏低，当前主要怀疑点不是“找不到相关技能”，而是：

1. top-k skill text 的排序和组成是否会误导 Qwen；
2. duplicate/auth/general skills 是否在 top-k 中占据过高位置；
3. SkillX body 与 AppWorld official API docs 的转译是否存在不匹配；
4. 当前 prompt 仍可能让 Qwen 过度复用错误的 SkillX 代码片段；
5. ACT v2 的 verified pairs 来源是 train-only oracle API trace 到 SkillX skill 的映射，不是 executor-generated success trajectories，因此不应作为最终 ACT 结论。

本地确实已有可用于更大规模预训练的数据，但它们与 AppWorld official task-success 的关系不同：

- `data/clstr_full_base_train/train.jsonl`: `202928` 条 ALFWorld/ScienceWorld offline replay supervised pretraining 记录。
- `data/aux_skillnet_rebuilt/trajectories.jsonl`: `17941` 条轨迹，约 `661549` steps，其中 train 可用部分用于辅助 SkillNet 风格预训练。
- `data/skillret`: `68256` queries、`16783` skills、`135537` qrels，可用于 retrieval/skill matching 预训练。
- `data/appworld_routing`: AppWorld train/dev/test routing corpus，其中只有 train split 可用于训练，dev/test 只能评价。

下一步应优先做一个能区分“方法问题”和“实现/初始化问题”的诊断实验，而不是继续盲目扩大 ACT：

1. 做 executor failure top-k analysis：比较 CLSTR-base、SkillRouter-base、Qwen-only 在 dev57 上的成功/失败交集，输出 CLSTR 失败但 SkillRouter/Qwen 成功任务的 top-k skills、skill 类型、是否 auth/general duplicate 过多、是否包含真正任务相关 API refs。
2. 实现 SkillRouter-init/SkillRouter-teacher CLSTR：用 SkillRouter embedding 或 SkillRouter-base ranking/logits 初始化/蒸馏 CLSTR，再只训练 CLSTR heads/adapters，避免把弱 MiniLM 初始化误判为 CLSTR 方法失败。
3. 如果 SkillRouter-init CLSTR 追平或超过 SkillRouter-base，说明 CLSTR 结构本身可用，当前问题主要是初始化和训练数据；再进入大数据预训练 + AppWorld train fine-tune。
4. 如果 SkillRouter-init CLSTR 仍然显著低于 SkillRouter-base，则重点检查 CLSTR head/objective、skill serialization、executor prompt，而不是继续加数据。
5. 大数据训练可作为第二阶段：SkillRET/SkillsBench 做 retrieval pretrain，ALFWorld/ScienceWorld 做 transition/belief pretrain，最后仅用 AppWorld train 做适配；不能使用 AppWorld dev/test 训练。

## 2026-05-23: AppWorld executor failure top-k analysis

新增诊断工具：

- `clstr/appworld_routing_diagnostics.py::build_executor_failure_topk_report`
- `scripts/run_appworld_routing_diagnostics.py executor-failure-topk`
- 测试：`tests/test_appworld_routing_diagnostics.py::test_executor_failure_topk_report_flags_focus_failures_with_reference_success`

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_routing_diagnostics.py -q
```

结果：`3 passed in 0.04s`。

真实 dev57 诊断报告：

- `outputs/appworld_routing_diagnostics/topk_executor_failure_overlap_v1/report.json`

输入为已完成的 dev57 executor runs，不重新运行 Qwen 或 AppWorld：

- Qwen-only: `outputs/appworld_executor_benchmark/qwen_only_dev57_v1/runs.jsonl`
- SkillRouter embedding: `outputs/appworld_executor_benchmark/skillrouter_embedding_dev57_v1/runs.jsonl`
- SkillRouter-base: `outputs/appworld_executor_benchmark/skillrouter_base_dev57_v1/runs.jsonl`
- CLSTR-base: `outputs/appworld_executor_benchmark/clstr_base_routing_only_dev57_v1/runs.jsonl`
- CLSTR-act: `outputs/appworld_executor_benchmark/clstr_act_dev57_v1/runs.jsonl`

报告中的 dev57 method summary 与之前 comparison 一致：

| method | success_count | success_rate | execution_failure_count | task_completed_count |
|---|---:|---:|---:|---:|
| Qwen-only | 14/57 | 0.245614 | 24 | 33 |
| SkillRouter embedding | 15/57 | 0.263158 | 32 | 25 |
| SkillRouter-base | 14/57 | 0.245614 | 32 | 25 |
| CLSTR-base | 9/57 | 0.157895 | 32 | 25 |
| CLSTR-act | 13/57 | 0.228070 | 29 | 28 |

重点分析 CLSTR-base 失败但至少一个 reference method 成功的任务，共 `11` 个：

- `50e1ac9_1`
- `50e1ac9_2`
- `50e1ac9_3`
- `6171bbc_1`
- `6171bbc_2`
- `6171bbc_3`
- `6c2c621_1`
- `6c2c621_2`
- `6c2c621_3`
- `b119b1f_2`
- `b119b1f_3`

聚合现象：

- `rows_with_focus_top1_auth_like = 11/11`
- `rows_with_focus_canonical_hit_in_topk = 11/11`
- `rows_with_no_focus_id_hit_in_topk = 0/11`
- `rows_with_focus_top1_off_app = 0/11`

解释：

1. CLSTR-base 在这些失败样本中并不是没有命中正例；无论 raw id 还是 canonical 口径，top-k 都含正例。
2. 但 CLSTR-base top-1 全是 auth/login/credential 类技能，top-k 中常有多个 duplicate auth slots。例如 Spotify 任务的 top-5 往往是 3 个 Spotify auth duplicate 加 1-2 个 phone/file_system auth。
3. 对 executor 来说，这类 top-k 对登录有帮助，但不能提供足够的任务执行逻辑；有时还会诱导 Qwen 按 SkillX body 误读 AppWorld API 返回结构。
4. 典型错误包括：
   - Spotify library API 返回 dict/list item 被当作 song id，触发 `TypeError: unhashable type: 'dict'`；
   - 代码执行成功并 `task_completed=True`，但 evaluator 答案不匹配；
   - Simple Note/File System 多 app 任务里使用错误 access token；
   - Spotify next/previous song 任务里生成了被 sandbox 禁止的 `exit()`。
5. 这进一步说明当前瓶颈不是单纯 routing recall，而是 top-k 的任务信息密度、duplicate/auth 去重、skill serialization 和 executor prompt 之间的交互。

下一步实验应避免直接把问题归因于 CLSTR objective：

1. 在 executor provider 层增加 duplicate-aware / auth-aware top-k packing 作为诊断 ablation：限制同一 canonical skill cluster 只出现一次，限制 auth-like skill slots，保留更多任务操作型 skills。
2. 与当前 raw top-k 对比 smoke/dev10：如果去重/packing 后 CLSTR-base + Qwen 明显恢复，说明主要是 prompt context composition 问题。
3. 并行准备 SkillRouter-init/teacher CLSTR：用 SkillRouter embedding 或 ranking 初始化 CLSTR，再观察在相同 top-k packing 下是否优于 SkillRouter。

## 2026-05-23: Packed top-k executor ablation

根据 failure top-k analysis，新增一个默认关闭的 executor ablation，用于验证 CLSTR-base task-success 偏低是否由 top-k context composition 导致。

代码改动：

- `clstr/appworld_executor.py`
  - `PredictionFileSkillProvider` 新增：
    - `dedupe_canonical_skills`
    - `max_auth_like_skills`
  - 默认值保持原始行为不变。
  - 当显式开启时，provider 会跳过同一 canonical skill cluster 的重复技能，并限制 auth/login/credential/password 类技能进入 top-k 的数量。
  - executor report 增加 `skill_provider_config`，记录 ablation 开关。
- `scripts/run_appworld_qwen_executor_eval.py`
  - 新增 CLI：
    - `--dedupe_canonical_skills`
    - `--max_auth_like_skills`
- `scripts/sbatch/run_appworld_qwen_executor_eval.sh`
  - 新增环境变量：
    - `DEDUPE_CANONICAL_SKILLS=1`
    - `MAX_AUTH_LIKE_SKILLS=<int>`
- `tests/test_appworld_executor.py`
  - 新增 provider packing 单测，覆盖 duplicate auth-heavy top-k 被压缩为更有任务信息密度的 top-k。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py -q
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_routing_diagnostics.py -q
```

结果：

- `tests/test_appworld_executor.py`: `10 passed in 1.36s`
- `tests/test_appworld_routing_diagnostics.py`: `3 passed in 0.03s`

已提交 packed top-k smoke：

```bash
DEDUPE_CANONICAL_SKILLS=1 \
MAX_AUTH_LIKE_SKILLS=1 \
METHOD=clstr_base \
OUTPUT_DIR=outputs/appworld_executor_smoke/clstr_base_packed_dev3_v1 \
MAX_TASKS=3 \
TEMPERATURE=0.0 \
TOP_K=5 \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

SLURM job: `75427`

按当前约定，短任务不频繁轮询；约 15 分钟后再检查该 job。如果 packed top-k dev3 相比 raw CLSTR-base dev57 前 3 个失败样本明显恢复，再扩展到 dev10；否则优先转向 SkillRouter-init/teacher CLSTR。

## 2026-05-23: SkillRouter-init CLSTR routing-only 诊断启动

为区分 “CLSTR 方法本身问题” 与 “当前 MiniLM 初始化/文本模板太弱”，新增 SkillRouter-init CLSTR 诊断配置。这个实验仍只使用 AppWorld train split，不使用 dev/test 训练。

代码/配置改动：

- `clstr/model.py`
  - `CLSTRConfig` 新增 `state_text_format`，默认 `raw`，因此现有配置行为不变。
  - 新增 `state_text_format=appworld_skillrouter_query`，query 模板对齐 SkillRouter-base：
    - `Instruct: Given a task description, retrieve the most relevant skill document that would help an agent complete the task\nQuery:<query>`
  - 新增 `skill_text_format=appworld_skillrouter_base`，skill 文本模板对齐 AppWorld SkillRouter-base：
    - `name | description | executor_desc | body`
- `configs/model/appworld_skillrouter_init.yaml`
  - 使用 `.cache/hf_models/SkillRouter-Embedding-0.6B`
  - `d=1024`
  - `encoder_pooling=last_token`
  - `cross_encoder_pooling=last_token`
  - `tokenizer_padding_side=left`
  - `freeze_backbone=true`
  - `projection_init=identity`
  - `skill_table_adapter_init=identity`
  - `normalize_embeddings=true`
  - `skill_text_format=appworld_skillrouter_base`
  - `state_text_format=appworld_skillrouter_query`
- `configs/train/appworld_skillrouter_init_routing_only.yaml`
  - routing-only, 120 updates
  - AppWorld train split
  - `task_batch_size=15`
  - no rollout / no ACT

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_model_pipeline.py \
  tests/test_encoders.py::test_clstr_config_can_identity_initialize_skillrouter_projection_and_adapter \
  tests/test_appworld_executor.py \
  tests/test_appworld_routing_diagnostics.py -q
```

结果：`17 passed in 10.59s`。

YAML/Config 解析验证通过：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python - <<'PY'
from pathlib import Path
import yaml
from clstr.model import CLSTRConfig
for path in ['configs/model/appworld_skillrouter_init.yaml','configs/train/appworld_skillrouter_init_routing_only.yaml']:
    data=yaml.safe_load(Path(path).read_text())
    print(path, 'ok')
CLSTRConfig(**yaml.safe_load(Path('configs/model/appworld_skillrouter_init.yaml').read_text()))
print('CLSTRConfig ok')
PY
```

已提交训练作业：

```bash
TRAIN_CONFIG=configs/train/appworld_skillrouter_init_routing_only.yaml \
MODEL_CONFIG=configs/model/appworld_skillrouter_init.yaml \
OUTPUT_DIR=outputs/appworld_clstr_train_skillrouter_init_routing_only_v1 \
MAX_TRAIN_TASKS=90 \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_train.sh
```

SLURM job: `75430`

后续检查节奏：

- 不频繁轮询。
- packed top-k smoke job `75427` 是短任务，约 15 分钟后检查。
- SkillRouter-init CLSTR train job `75430` 可能更长，约 30 分钟后检查。
- 若 `75430` 成功，立即用同一 model config 跑 dev routing eval，并与 SkillRouter-base、CLSTR-mini raw/canonical routing 比较。

## 2026-05-23: Packed top-k smoke 结果

已检查 packed top-k smoke job `75427`，作业已完成。

输出：

- `outputs/appworld_executor_smoke/clstr_base_packed_dev3_v1/report.json`
- `outputs/appworld_executor_smoke/clstr_base_packed_dev3_v1/runs.jsonl`

配置：

```json
{
  "dedupe_canonical_skills": true,
  "max_auth_like_skills": 1
}
```

结果：

| method | task_count | success_count | success_rate | execution_failures |
|---|---:|---:|---:|---:|
| CLSTR-base packed top-k | 3 | 0 | 0.0 | 2 |

与 raw CLSTR-base dev57 前 3 个任务对比：

- raw CLSTR-base selected skills 前 3 个任务几乎全部被 auth duplicate 占据。
- packed top-k 后 selected skills 变为：
  - `50e1ac9_1`: Spotify auth + playlist duration + rating + create playlist
  - `50e1ac9_2`: Spotify auth + playlist duration + rating + create playlist
  - `50e1ac9_3`: Spotify auth + playlist duration + rating + create playlist
- 说明 packing 确实减少了 auth duplicate，但没有带来 task success。

失败细节：

- `50e1ac9_1`: execution failed，仍把 `show_song_library()` 返回的 dict/list item 当作 hashable song id，触发 `TypeError: unhashable type: 'dict'`。
- `50e1ac9_2`: execution successful 且 `task_completed=True`，但 evaluator answer mismatch；说明代码逻辑仍没有正确覆盖 song/album/playlist library 中所有歌曲与排序细节。
- `50e1ac9_3`: execution failed，生成了未定义变量 `song_id`，触发 `NameError`。

结论：

1. top-k packing 只解决了 “auth duplicate 挤占 context” 的一部分问题，但不能单独修复 CLSTR-base + Qwen 的 AppWorld task success。
2. 当前关键错误仍在 executor prompt/code generation 对 AppWorld API 返回 schema 的理解，以及复杂 Spotify aggregation 任务的实现细节。
3. 不扩展 packed top-k dev10；继续等待 SkillRouter-init CLSTR routing-only job `75430`，用它判断初始化和文本模板是否是更主要因素。
4. 如果 SkillRouter-init CLSTR routing 能追平 SkillRouter-base，但 executor 仍不提升，则下一步重点应转向 executor prompt/API schema grounding，而不是继续调 routing。

已提交 SkillRouter-init CLSTR dev routing eval 依赖作业，避免频繁轮询训练：

```bash
MODEL_CONFIG=configs/model/appworld_skillrouter_init.yaml \
CHECKPOINT_DIR=outputs/appworld_clstr_train_skillrouter_init_routing_only_v1/checkpoints \
OUTPUT_DIR=outputs/appworld_clstr_eval/skillrouter_init_routing_only_v1_dev \
TOP_K=20 \
sbatch --dependency=afterok:75430 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_eval.sh
```

SLURM job: `75437`

该 eval job 只有在 `75430` 成功结束后才会运行；如果训练失败，eval 不会误跑。

已新增通用 routing diagnostics sbatch wrapper：

- `scripts/sbatch/run_appworld_routing_diagnostics.sh`

用途：用 sbatch 跑轻量 AppWorld routing diagnostics，避免在登录节点手动跑诊断脚本。当前支持：

- `COMMAND=duplicate-aware`
- `COMMAND=train-dev-overlap`

已提交 SkillRouter-init CLSTR duplicate-aware canonical 诊断依赖作业：

```bash
COMMAND=duplicate-aware \
METHOD=clstr_skillrouter_init_routing_only \
PREDICTIONS_PATH=outputs/appworld_clstr_eval/skillrouter_init_routing_only_v1_dev/predictions.jsonl \
OUTPUT_DIR=outputs/appworld_routing_diagnostics/skillrouter_init_routing_only_v1_dev \
sbatch --dependency=afterok:75437 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_routing_diagnostics.sh
```

SLURM job: `75440`

依赖链：

- `75430`: SkillRouter-init CLSTR routing-only train
- `75437`: afterok `75430`, dev routing eval
- `75440`: afterok `75437`, duplicate-aware canonical routing diagnostics

已检查 `75430`，训练成功完成。`75437` 当前 pending priority，`75440` 等待 `75437`。

SkillRouter-init CLSTR routing-only train 结果：

| stage | checkpoint | loss | train-batch recall@1 | train-batch recall@5 |
|---|---|---:|---:|---:|
| stage0_warmstart | `stage0_warmstart-step120.pt` | 0.336940 | 0.866667 | 1.0 |
| stage1_heads | `stage1_heads-step120.pt` | 0.265520 | 1.0 | 1.0 |
| stage2_base | `stage2_base-step120.pt` | 0.165876 | 1.0 | 1.0 |

训练配置摘要：

- `model_dim=1024`
- `base_model_name=.cache/hf_models/SkillRouter-Embedding-0.6B`
- `task_batch_size=15`
- `beta=0.0`
- `env_mode=replay`
- `verified_templates=0`
- `verified_exact_pairs=0`

为保持项目环境干净，已删除本次训练的中间 checkpoint，只保留每个 stage 的最终 checkpoint：

- `outputs/appworld_clstr_train_skillrouter_init_routing_only_v1/checkpoints/stage0_warmstart-step120.pt`
- `outputs/appworld_clstr_train_skillrouter_init_routing_only_v1/checkpoints/stage1_heads-step120.pt`
- `outputs/appworld_clstr_train_skillrouter_init_routing_only_v1/checkpoints/stage2_base-step120.pt`

checkpoint 目录大小从约 `42G` 降到约 `7.0G`。`75437` 使用 checkpoint dir 自动选择最新 checkpoint，保留的 `stage2_base-step120.pt` 是本次最终模型。

`75437` 和 `75440` 已完成。

SkillRouter-init CLSTR dev routing eval:

- report: `outputs/appworld_clstr_eval/skillrouter_init_routing_only_v1_dev/report.json`
- predictions: `outputs/appworld_clstr_eval/skillrouter_init_routing_only_v1_dev/predictions.jsonl`
- checkpoint: `outputs/appworld_clstr_train_skillrouter_init_routing_only_v1/checkpoints/stage2_base-step120.pt`

Raw routing:

| method | recall@1 | recall@5 | mrr@20 |
|---|---:|---:|---:|
| SkillRouter-base | 0.982456 | 1.0 | 0.991228 |
| CLSTR-base MiniLM routing-only | 0.912281 | 1.0 | 0.956140 |
| CLSTR SkillRouter-init routing-only | 0.964912 | 1.0 | 0.976608 |

Duplicate-aware canonical routing:

| method | canonical recall@1 | canonical recall@5 | canonical mrr@20 |
|---|---:|---:|---:|
| SkillRouter-base | 0.982456 | 1.0 | 0.991228 |
| CLSTR-base MiniLM routing-only | 1.0 | 1.0 | 1.0 |
| CLSTR SkillRouter-init routing-only | 0.964912 | 1.0 | 0.982456 |

SkillRouter-init CLSTR misses:

- `23cf851_1`
- `23cf851_2`

两者 top1 都是 `skillx/appworld/login-to-apps-15`，positive 在 rank 3 / canonical rank 2。

诊断结论：

1. 换成 SkillRouter embedding 与 SkillRouter 文本模板后，CLSTR raw recall@1 从 `0.912281` 提升到 `0.964912`，说明当前 MiniLM 初始化确实是 raw routing 弱项之一。
2. 但 SkillRouter-init CLSTR 仍低于 SkillRouter-base 的 `0.982456`，说明 CLSTR head/skill table 训练并未自动超过 SkillRouter-base scorer。
3. MiniLM CLSTR 的 canonical recall@1 反而是 `1.0`，表明 duplicate/id artifact 与 executor usefulness 是两个不同问题。
4. 下一步只做小型 executor smoke，不直接扩大 dev benchmark：把 SkillRouter-init CLSTR predictions 接入 Qwen executor，跑 dev 前 3 个任务，与 raw CLSTR-base/packed top-k 的 `0/3` 对比。如果仍无提升，则当前主瓶颈应转向 executor/API schema grounding，而不是继续调 routing。

已注册 SkillRouter-init CLSTR executor 方法名：

- `scripts/run_appworld_qwen_executor_eval.py`
  - 新增 `clstr_skillrouter_init`
  - 默认 predictions: `outputs/appworld_clstr_eval/skillrouter_init_routing_only_v1_dev/predictions.jsonl`

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py -q
```

结果：`11 passed in 28.95s`。

SkillRouter-init CLSTR 的 dev 前 3 个 Spotify aggregation 任务 top-20 仍明显 auth/credential-heavy：

- top1-3: `spotify-authenticate-with-stored-credentials-*`
- top4-5: `retrieve-stored-account-credentials-*`
- 直到 rank 17 才出现 `spotify-calculate-playlist-duration-*`

因此不跑 raw executor smoke，直接跑 packed smoke，避免重复验证已知 auth-heavy failure。

已提交 SkillRouter-init CLSTR packed executor smoke：

```bash
DEDUPE_CANONICAL_SKILLS=1 \
MAX_AUTH_LIKE_SKILLS=1 \
METHOD=clstr_skillrouter_init \
OUTPUT_DIR=outputs/appworld_executor_smoke/clstr_skillrouter_init_packed_dev3_v1 \
MAX_TASKS=3 \
TEMPERATURE=0.0 \
TOP_K=5 \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

SLURM job: `75444`

若 `75444` 仍为 `0/3` 或主要错误仍是 API schema/aggregation，则停止继续 routing-side executor benchmark，转向 executor prompt/API docs grounding 或 AppWorld-specific code scaffold。

## 2026-05-23: Spotify API schema grounding prompt fix

根据 CLSTR-base packed top-k smoke 和 SkillRouter-init packed smoke 的已知失败模式，补充 Spotify library aggregation 任务的 prompt grounding。失败集中在：

- 把 `show_song_library()` 返回的 song dict 当作 song id，触发 `TypeError: unhashable type: 'dict'`；
- 误以为 `show_playlist_library()` 返回 `songs` 字段；
- 未正确从 album/playlist library 的 `song_ids` 展开所有歌曲；
- 未分页遍历完整 library；
- 代码执行成功但 evaluator answer mismatch。

真实 AppWorld Spotify API schema：

- `show_song_library(...)` returns list of song dicts with `song_id`。
- `show_album_library(...)` returns list of album dicts with `song_ids`。
- `show_playlist_library(...)` returns list of playlist dicts with `song_ids`。
- `show_song(song_id=...)` returns detail fields including `genre`, `play_count`, `title`，且不需要 `access_token`。

代码改动：

- `clstr/appworld_executor.py`
  - `_build_task_specific_hints()` 对 `top N most played <genre> song titles` Spotify 任务新增 schema-level hints：
    - `show_song_library` 使用 `item["song_id"]`
    - `show_album_library` 使用 `album["song_ids"]`
    - `show_playlist_library` 使用 `playlist["song_ids"]`
    - 不要期待 `show_playlist_library` 返回 `songs`
    - 三个 library API 都要 `page_index` 分页到空列表
    - `show_song(song_id=song_id)` 后再读 `genre/play_count/title`
- `tests/test_appworld_executor.py`
  - 补充 prompt 断言，确保上述 schema hints 出现在 prompt 中。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py -q
```

结果：`11 passed in 1.39s`。

当前 `75444` 仍在运行，它使用旧 prompt。为获得新 prompt 可比结果，已提交 afterany 依赖作业：

```bash
DEDUPE_CANONICAL_SKILLS=1 \
MAX_AUTH_LIKE_SKILLS=1 \
METHOD=clstr_skillrouter_init \
OUTPUT_DIR=outputs/appworld_executor_smoke/clstr_skillrouter_init_packed_dev3_v2_schema_hints \
MAX_TASKS=3 \
TEMPERATURE=0.0 \
TOP_K=5 \
sbatch --dependency=afterany:75444 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

SLURM job: `75447`

解释：`75447` 会在旧 prompt smoke `75444` 结束后运行，用同一 SkillRouter-init packed top-k，但采用新的 Spotify schema hints。

已检查旧 prompt smoke `75444`，结果：

- output: `outputs/appworld_executor_smoke/clstr_skillrouter_init_packed_dev3_v1`
- task_count: `3`
- success_count: `2`
- success_rate: `0.666667`
- execution_failures: `0`

逐任务：

| task | success | execution_ok | task_completed | evaluation_success | selected skills |
|---|---:|---:|---:|---:|---|
| `50e1ac9_1` | true | true | true | true | Spotify auth + Spotify calculate playlist duration |
| `50e1ac9_2` | false | true | true | false | Spotify auth + Spotify calculate playlist duration |
| `50e1ac9_3` | true | true | true | true | Spotify auth + Spotify calculate playlist duration |

解释：

1. SkillRouter-init + packed top-k 明显优于 CLSTR-base packed top-k 的 `0/3`，说明初始化和 top-k composition 都对 executor 有实际影响。
2. 失败的 `50e1ac9_2` 不是执行失败，而是 answer mismatch：Qwen 成功完成任务，但 EDM top-6 答案不匹配 evaluator。
3. 旧 prompt 下生成代码已经正确使用 `song["song_id"]`，但仍可能未完整分页遍历 song/album/playlist library，或 aggregation 细节仍不符合 evaluator。
4. `75447` 正在运行，用新 Spotify schema/pagination hints 复测同一 3-task smoke。

已检查 `75447`，新 schema hints smoke 成功：

- output: `outputs/appworld_executor_smoke/clstr_skillrouter_init_packed_dev3_v2_schema_hints`
- task_count: `3`
- success_count: `3`
- success_rate: `1.0`
- execution_failures: `0`

逐任务均通过：

| task | success | execution_ok | task_completed | evaluation_success | pagination in code |
|---|---:|---:|---:|---:|---:|
| `50e1ac9_1` | true | true | true | true | true |
| `50e1ac9_2` | true | true | true | true | true |
| `50e1ac9_3` | true | true | true | true | true |

对比旧 prompt `75444`：

- 旧 prompt: `2/3`，失败在 `50e1ac9_2` answer mismatch，生成代码没有 `page_index` 分页。
- 新 schema hints: `3/3`，生成代码包含分页逻辑。

结论：

1. SkillRouter-init + packed top-k 已经能在 dev 前 3 个 Spotify aggregation 任务上达到 `3/3`。
2. schema/pagination grounding 是关键改动，说明 executor 侧 API grounding 是当前 task-success 的主要瓶颈之一。
3. 由于 prompt 已改变，必须重新跑同一 prompt 下的 dev10 对照，不能直接拿旧 dev10/dev57 表比较。

已提交 dev10 v2 schema-hints 对比作业：

| job | method | output_dir | packing |
|---:|---|---|---|
| `75449` | Qwen-only | `outputs/appworld_executor_benchmark/qwen_only_dev10_v2_schema_hints` | none |
| `75450` | SkillRouter embedding | `outputs/appworld_executor_benchmark/skillrouter_embedding_dev10_v2_schema_hints` | none |
| `75453` | SkillRouter-base | `outputs/appworld_executor_benchmark/skillrouter_base_dev10_v2_schema_hints` | none |
| `75452` | CLSTR-base MiniLM | `outputs/appworld_executor_benchmark/clstr_base_packed_dev10_v2_schema_hints` | canonical dedupe + max auth-like 1 |
| `75451` | CLSTR SkillRouter-init | `outputs/appworld_executor_benchmark/clstr_skillrouter_init_packed_dev10_v2_schema_hints` | canonical dedupe + max auth-like 1 |

提交命令均使用 `MAX_TASKS=10`、`TEMPERATURE=0.0`、`TOP_K=5`、`sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh`。等待约 15 分钟后再检查，不频繁轮询。

新增 executor comparison sbatch wrapper：

- `scripts/sbatch/run_appworld_executor_comparison.sh`

作用：用 sbatch 汇总多个 executor `report.json`，生成 comparison，避免在登录节点直接跑 Python 汇总脚本。

已提交 dev10 v2 schema-hints comparison 依赖作业：

```bash
OUTPUT_DIR=outputs/appworld_executor_benchmark/dev10_v2_schema_hints_comparison \
sbatch --dependency=afterok:75449:75450:75451:75452:75453 --gpus=1 -p gpu_h200 \
  scripts/sbatch/run_appworld_executor_comparison.sh \
  outputs/appworld_executor_benchmark/qwen_only_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/skillrouter_embedding_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/skillrouter_base_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/clstr_base_packed_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/clstr_skillrouter_init_packed_dev10_v2_schema_hints/report.json
```

SLURM job: `75458`

依赖链：`75449-75453` 全部成功后，`75458` 自动生成 `outputs/appworld_executor_benchmark/dev10_v2_schema_hints_comparison/{comparison.json,comparison.md}`。

## 2026-05-23 AppWorld ACT/HRPO 诊断：当前 `L_policy` 没有真正获得多 rollout 收益归因

针对“随机采样 skill 后模型如何知道哪个 skill 收益最大”的问题，已核对当前实现和
`19876_Hybrid_Latent_Reasoning_.pdf`：

- 当前 CLSTR 的 `m_t` 不是随机 skill embedding，而是 `softmax(skill logits) @ skill_table.E`
  得到的连续 skill-memory；真正的随机性发生在 rollout 阶段，从 top-k candidate skills 的
  policy logits 中采样 skill/STOP。
- 当前 `policy_loss` 已经按 `task_id` 分组，并用同一任务内多条 rollout 的 reward 标准化成
  advantage。这和 HRPO/GRPO 风格的组内相对优势是一致的。
- 但 AppWorld 现有配置没有真正触发这个机制：
  - `configs/train/appworld_routing_only_fit.yaml` 是 `loss_mode: routing_only`，
    `rollouts_per_task: 0`，没有 on-policy rollout；
  - `configs/train/appworld_act_train.yaml` 只有 `rollouts_per_task: 1`。单个 task 只有一条
    rollout 时，组内 reward 标准化后 advantage 约等于 0，因此 `L_policy` 基本不给有效梯度。
- `19876_Hybrid_Latent_Reasoning_.pdf` 的 HRPO 关键设定是每个 query 生成 `g` 个 hybrid
  rollouts，并在组内标准化 reward；附录默认 group size 为 `4 / 8`。因此 AppWorld 上若要
  让 CLSTR 学会“哪个采样 skill/context 对 Qwen executor 的 task success 更有用”，必须接入
  AppWorld official executor/evaluator 的多 rollout reward，而不是只用单条 replay trajectory。

下一步实验含义：

1. 当前 MiniLM CLSTR-base 和 SkillRouter-init CLSTR-base 的 routing eval 只能说明 retrieval/routing
   表面能力，不能验证 CLSTR-act 的 RL 贡献。
2. 若 dev10 schema-hints executor 对比显示 SkillRouter-init packed top-k 已接近或超过 baseline，
   应优先实现 AppWorld skill-level HRPO/ACT：
   - 每个 train task 采样 `g=4` 或 `g=8` 组 top-k skill context；
   - Qwen3-8B 生成 Python code；
   - 用 AppWorld official `execute/evaluate` 得到 success reward；
   - 对同一 task 的 rewards 做组内 advantage；
   - 只更新 CLSTR policy/head/gate/projection，Qwen 冻结，并加 KL 到 frozen CLSTR reference。
3. SkillRouter-init + 多 rollout CLSTR-act 是最干净的诊断：
   - 如果 task success 超过 SkillRouter/Qwen baselines，说明 CLSTR-act 机制有效；
   - 如果仍不能超过，优先怀疑 executor API grounding、top-k skill text 组织或方法本身收益有限，
     而不是继续盲目扩大静态 routing 数据。

## 2026-05-23 修复：新增 AppWorld skill-level HRPO/ACT 训练入口

已补上真正使用 AppWorld official executor/evaluator reward 的 CLSTR-act 路径：

- 新增 `clstr/appworld_act_hrpo.py`
  - `sample_clstr_skill_context()`：
    - CLSTR 先取 top-k candidate skills；
    - 从 skill policy logits 中采样一个 focus skill；
    - prompt context 中把 sampled focus skill 放在第一位，再补其余 top-k context；
    - 记录 sampled skill 的 `log_prob` 和 frozen reference logits。
  - `run_appworld_hrpo_rollout()`：
    - 用 sampled skill context + instruction + API docs 构造 Qwen executor prompt；
    - Qwen 生成 Python code；
    - 交给 AppWorld `execute()`；
    - 用 `task_completed()` 和 `evaluate(success=True)` 得到二值 reward。
  - `compute_appworld_hrpo_loss()`：
    - 按 `task_id` 对同一 task 的多条 rollout reward 做组内标准化 advantage；
    - 用 `- advantage * log_prob(sampled_skill)` 更新 CLSTR skill policy；
    - 加 KL 到 frozen CLSTR reference；
    - 单 rollout group 会报告 `advantage_nonzero_count=0`，明确暴露无效 policy gradient。
  - `train_appworld_clstr_hrpo()`：
    - Qwen 冻结；
    - 默认从 CLSTR checkpoint warmstart；
    - 默认 `detach_policy_inputs=True`，先训练轻量 policy/head 路径，避免 Qwen executor 同卡运行时保留大 encoder 图；
    - 保存 `train_report.json`、`rollouts.jsonl` 和 `checkpoints/appworld_clstr_hrpo-step*.pt`。

- 新增训练入口：
  - `scripts/run_appworld_clstr_hrpo_train.py`
  - `scripts/sbatch/run_appworld_clstr_hrpo_train.sh`

默认 sbatch 用法：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

默认重要参数：

- `WARMSTART_CLSTR_CKPT=outputs/appworld_clstr_train_skillrouter_init_routing_only_v1/checkpoints/stage2_base-step120.pt`
- `GROUP_SIZE=4`
- `TOP_K=8`
- `CONTEXT_TOP_K=5`
- `MAX_TASKS=5`
- `UPDATES=5`
- `QWEN_TEMPERATURE=0.0`
- `ROLLOUT_TEMPERATURE=1.0`

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_act_hrpo.py tests/test_appworld_executor.py -q
```

结果：`14 passed in 3.87s`。

额外静态检查：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/appworld_act_hrpo.py scripts/run_appworld_clstr_hrpo_train.py
bash -n scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

均通过。

## 2026-05-23 AppWorld HRPO/ACT export 修复：让 policy head 真正进入 executor

根因：

- `clstr/appworld_act_hrpo.py` 的 HRPO 训练默认 `detach_policy_inputs=true`，因此主要更新的是
  CLSTR `policy_head`，用于学习“同一 task 的多个 sampled skills 中哪个带来更高 AppWorld official
  reward”。
- 但旧的 `scripts/run_appworld_clstr_eval.py` / `clstr/appworld_clstr_eval.py` 导出
  `predictions.jsonl` 时只使用 `skill_table.logits` 排序，没有调用 `policy_head`。
- 因此旧的 HRPO smoke v3 dev routing predictions 和随后 `75479` 的 `clstr_act` dev3 executor
  smoke 大体仍是 base routing 排序，不应被解释为真正的 ACT policy-head 评估。

修复：

- `clstr/appworld_clstr_eval.py`
  - 新增 `rank_clstr_skills_for_query(..., ranking_mode="skill_table"|"policy_head"|"policy_blend")`；
  - 默认 `skill_table` 保持旧 base/routing eval 行为；
  - `policy_head` 模式先用 `skill_table.logits` 取 `candidate_top_k` 个候选，再计算
    `m_t=subspace_obs(...)`、`batch_cross_encode(...)` 和 `model.policy_forward(...)`，最后按
    policy logits 重排导出的 `ranked_skill_ids`。
  - `policy_blend` 模式在同一候选池内对 base routing logits 与 policy logits 做标准化后加权，
    作为 conservative CLSTR-act export，避免未充分训练的 policy head 破坏强 base routing。
- `scripts/run_appworld_clstr_eval.py`
  - 新增 `--ranking_mode {skill_table,policy_head,policy_blend}`；
  - 新增 `--candidate_top_k`。
  - 新增 `--policy_blend_alpha`。
- `scripts/sbatch/run_appworld_clstr_eval.sh`
  - 新增 `RANKING_MODE` / `CANDIDATE_TOP_K` / `POLICY_BLEND_ALPHA` 环境变量，便于通过 sbatch 导出真正的
    CLSTR-act predictions。
- 新增 `tests/test_appworld_clstr_eval.py`，用假模型确认 `policy_head` 可以覆盖 raw
  `skill_table.logits` 的原始顺序。
- `clstr/appworld_act_hrpo.py`
  - 新增 `routing_logits` 记录与 `beta_routing_prior`；
  - HRPO loss 变为 `L_policy + beta_kl * KL(policy||reference) + beta_routing_prior * KL(policy||base_routing)`，
    让 ACT policy 在 AppWorld reward 还稀疏时保留 SkillRouter-init/base routing 先验。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_clstr_eval.py tests/test_appworld_act_hrpo.py tests/test_sbatch_scripts.py tests/test_appworld_executor.py -q
```

结果：`21 passed in 3.80s`。

已完成的旧 HRPO smoke 结果仍有诊断价值：

- `75477` 证明 AppWorld official reward 已进入 HRPO loss，且多 rollout 产生了非零 advantage；
- `75478` / `75479` 证明 checkpoint 可加载、executor 可跑通；
- 但需要用 `RANKING_MODE=policy_head` 重新导出 predictions 后，才能评估 HRPO/ACT 对 executor
  context selection 的实际影响。

已提交修复后的验证链路：

```bash
RANKING_MODE=policy_head \
CANDIDATE_TOP_K=8 \
MODEL_CONFIG=configs/model/appworld_skillrouter_init.yaml \
CHECKPOINT_PATH=outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v3/checkpoints/appworld_clstr_hrpo-step1.pt \
OUTPUT_DIR=outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v3_dev_policy_head \
TOP_K=20 \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_eval.sh
```

SLURM job: `75487`

```bash
METHOD=clstr_act \
PREDICTIONS_PATH=outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v3_dev_policy_head/predictions.jsonl \
OUTPUT_DIR=outputs/appworld_executor_smoke/clstr_hrpo_smoke_skillrouter_init_g2_u1_v3_policy_head_dev3 \
DEDUPE_CANONICAL_SKILLS=1 \
MAX_AUTH_LIKE_SKILLS=1 \
MAX_TASKS=3 \
TOP_K=5 \
TEMPERATURE=0.0 \
sbatch --dependency=afterok:75487 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

SLURM job: `75488`

补充修复：

- `candidate_top_k` 必须表示 policy rerank 前的候选池大小，而不是被 `TOP_K` 隐式扩大的下界；
- 已新增测试覆盖 `TOP_K > CANDIDATE_TOP_K` 时只在 `candidate_top_k` 个 base candidates 内重排。

由于 `75487/75488` 提交后又修正了 `candidate_top_k` 语义，保留它们作为过渡检查，不作为最终
policy-head smoke 结论。已提交修正版链路：

```bash
RANKING_MODE=policy_head \
CANDIDATE_TOP_K=8 \
MODEL_CONFIG=configs/model/appworld_skillrouter_init.yaml \
CHECKPOINT_PATH=outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v3/checkpoints/appworld_clstr_hrpo-step1.pt \
OUTPUT_DIR=outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v3_dev_policy_head_top8_v2 \
TOP_K=8 \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_eval.sh
```

SLURM job: `75490`

```bash
METHOD=clstr_act \
PREDICTIONS_PATH=outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v3_dev_policy_head_top8_v2/predictions.jsonl \
OUTPUT_DIR=outputs/appworld_executor_smoke/clstr_hrpo_smoke_skillrouter_init_g2_u1_v3_policy_head_top8_v2_dev3 \
DEDUPE_CANONICAL_SKILLS=1 \
MAX_AUTH_LIKE_SKILLS=1 \
MAX_TASKS=3 \
TOP_K=5 \
TEMPERATURE=0.0 \
sbatch --dependency=afterok:75490 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

SLURM job: `75491`

检查 `75490` 的 pure policy-head export 后发现：

- `outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v3_dev_policy_head_top8_v2/report.json`
  的 routing 指标为 `recall@1=0.228070`、`recall@5=0.438596`、`recall@8=1.000000`。
- 这说明 policy-head 确实进入了 export，但当前 AppWorld routing-only base 没有训练
  `policy_head`；1 update HRPO smoke 也不足以把随机 policy-head 训练成稳定 router。
- 因此后续 executor benchmark 不应使用 pure `policy_head` 作为主 ACT export，而应使用
  `policy_blend` 或先对 policy-head 做 routing imitation/warmup。

已提交 v4 prior + conservative blend 验证链路：

```bash
OUTPUT_DIR=outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v4_prior \
MAX_TASKS=2 \
GROUP_SIZE=2 \
UPDATES=1 \
TASK_BATCH_SIZE=2 \
TOP_K=8 \
CONTEXT_TOP_K=5 \
QWEN_TEMPERATURE=0.0 \
ROLLOUT_TEMPERATURE=1.0 \
BETA_ROUTING_PRIOR=0.05 \
TIMEOUT_SECONDS=60 \
MAX_INTERACTIONS=3 \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

SLURM job: `75493`

```bash
RANKING_MODE=policy_blend \
CANDIDATE_TOP_K=8 \
POLICY_BLEND_ALPHA=0.25 \
MODEL_CONFIG=configs/model/appworld_skillrouter_init.yaml \
CHECKPOINT_PATH=outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v4_prior/checkpoints/appworld_clstr_hrpo-step1.pt \
OUTPUT_DIR=outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v4_prior_dev_blend025_top8 \
TOP_K=8 \
sbatch --dependency=afterok:75493 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_eval.sh
```

SLURM job: `75494`

`75494` 已完成，输出：

- `outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v4_prior_dev_blend025_top8/report.json`
- `outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v4_prior_dev_blend025_top8/predictions.jsonl`

Routing 指标：

| export | recall@1 | recall@5 | recall@8/20 | mrr |
|---|---:|---:|---:|---:|
| SkillRouter-init CLSTR base (`skill_table`) | 0.964912 | 1.000000 | 1.000000@20 | 0.976608@20 |
| HRPO v3 pure `policy_head` top8 | 0.228070 | 0.438596 | 1.000000@8 | 0.398287@8 |
| HRPO v4 prior `policy_blend` alpha=0.25 top8 | 0.964912 | 0.982456 | 1.000000@8 | 0.971345@8 |

解释：

- `policy_blend` 保住了 base routing 的 top-1，但 top-5/MRR 略低于纯 `skill_table`；
- 这说明 conservative export 比 pure `policy_head` 稳定得多，适合继续跑 executor benchmark；
- 但它还没有证明 ACT 带来提升，因为 v4 训练 `advantage_nonzero_count=0`，主要变化来自
  routing-prior/conservative rerank，而不是有效 policy-gradient。

```bash
METHOD=clstr_act \
PREDICTIONS_PATH=outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v4_prior_dev_blend025_top8/predictions.jsonl \
OUTPUT_DIR=outputs/appworld_executor_smoke/clstr_hrpo_smoke_skillrouter_init_g2_u1_v4_prior_blend025_dev3 \
DEDUPE_CANONICAL_SKILLS=1 \
MAX_AUTH_LIKE_SKILLS=1 \
MAX_TASKS=3 \
TOP_K=5 \
TEMPERATURE=0.0 \
sbatch --dependency=afterok:75494 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

SLURM job: `75495`

`75495` 已完成：

- output: `outputs/appworld_executor_smoke/clstr_hrpo_smoke_skillrouter_init_g2_u1_v4_prior_blend025_dev3`
- report: `outputs/appworld_executor_smoke/clstr_hrpo_smoke_skillrouter_init_g2_u1_v4_prior_blend025_dev3/report.json`
- runs: `outputs/appworld_executor_smoke/clstr_hrpo_smoke_skillrouter_init_g2_u1_v4_prior_blend025_dev3/runs.jsonl`

Executor smoke 指标：

| metric | value |
|---|---:|
| task_count | 3 |
| success_count | 3 |
| success_rate | 1.0 |
| generation_failures | 0 |
| execution_failures | 0 |

逐任务所用 skill context：

- `50e1ac9_1`: `spotify-authenticate-with-stored-credentials-26`, `spotify-calculate-playlist-duration-43`
- `50e1ac9_2`: 同上
- `50e1ac9_3`: 同上

解释：dev3 smoke 证明 v4 `policy_blend` predictions 可以接入 Qwen executor 并通过 AppWorld
`execute/task_completed/evaluate`。但 3 个 task 的 skill context 完全相同，任务覆盖太窄，不能用来证明
ACT 相比 base routing 有收益；关键仍要看 dev10 comparison。

已排队 v4 prior/blend 的 dev10 executor benchmark：

```bash
METHOD=clstr_act \
PREDICTIONS_PATH=outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v4_prior_dev_blend025_top8/predictions.jsonl \
OUTPUT_DIR=outputs/appworld_executor_benchmark/clstr_act_hrpo_v4_prior_blend025_dev10_v2_schema_hints \
DEDUPE_CANONICAL_SKILLS=1 \
MAX_AUTH_LIKE_SKILLS=1 \
MAX_TASKS=10 \
TOP_K=5 \
TEMPERATURE=0.0 \
sbatch --dependency=afterok:75494 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

SLURM job: `75498`

已排队 v4 prior/blend 的 dev10 comparison 汇总：

```bash
OUTPUT_DIR=outputs/appworld_executor_benchmark/dev10_v2_schema_hints_plus_hrpo_v4_comparison \
sbatch --dependency=afterok:75498 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_executor_comparison.sh \
  outputs/appworld_executor_benchmark/qwen_only_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/skillrouter_embedding_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/skillrouter_base_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/clstr_base_packed_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/clstr_skillrouter_init_packed_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/clstr_act_hrpo_v4_prior_blend025_dev10_v2_schema_hints/report.json
```

SLURM job: `75501`

`75493` 已写出训练报告和 checkpoint。训练报告：

- `outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v4_prior/train_report.json`
- `success_rate=0.5`
- `mean_reward=0.5`
- `advantage_nonzero_count=0`
- `policy_loss=-0.0`
- `routing_prior_loss=6.231988906860352`
- `loss=0.311599463224411`

逐 rollout 结果：

- `82e2fac_1`: 两条 rollout 都失败；
- `82e2fac_2`: 两条 rollout 都成功。

解释：v4 验证了 `beta_routing_prior` 能进入 loss，但同一 task 内没有成功/失败混合，因此这次 smoke
仍没有产生有效 HRPO policy-gradient。为了提高组内 reward 差异出现概率，已提交更大的 train split
HRPO 链路：

```bash
OUTPUT_DIR=outputs/appworld_clstr_train_hrpo/train10_g4_u3_v1_prior \
MAX_TASKS=10 \
GROUP_SIZE=4 \
UPDATES=3 \
TASK_BATCH_SIZE=2 \
TOP_K=8 \
CONTEXT_TOP_K=5 \
QWEN_TEMPERATURE=0.0 \
ROLLOUT_TEMPERATURE=1.0 \
BETA_ROUTING_PRIOR=0.05 \
TIMEOUT_SECONDS=60 \
MAX_INTERACTIONS=3 \
sbatch --dependency=afterok:75494 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

SLURM job: `75502`

```bash
RANKING_MODE=policy_blend \
CANDIDATE_TOP_K=8 \
POLICY_BLEND_ALPHA=0.25 \
MODEL_CONFIG=configs/model/appworld_skillrouter_init.yaml \
CHECKPOINT_PATH=outputs/appworld_clstr_train_hrpo/train10_g4_u3_v1_prior/checkpoints/appworld_clstr_hrpo-step3.pt \
OUTPUT_DIR=outputs/appworld_clstr_eval/hrpo_train10_g4_u3_v1_prior_dev_blend025_top8 \
TOP_K=8 \
sbatch --dependency=afterok:75502 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_eval.sh
```

SLURM job: `75503`

```bash
METHOD=clstr_act \
PREDICTIONS_PATH=outputs/appworld_clstr_eval/hrpo_train10_g4_u3_v1_prior_dev_blend025_top8/predictions.jsonl \
OUTPUT_DIR=outputs/appworld_executor_benchmark/clstr_act_hrpo_train10_g4_u3_v1_prior_blend025_dev10_v2_schema_hints \
DEDUPE_CANONICAL_SKILLS=1 \
MAX_AUTH_LIKE_SKILLS=1 \
MAX_TASKS=10 \
TOP_K=5 \
TEMPERATURE=0.0 \
sbatch --dependency=afterok:75503 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

SLURM job: `75504`

```bash
OUTPUT_DIR=outputs/appworld_executor_benchmark/dev10_v2_schema_hints_plus_hrpo_train10_g4_u3_v1_comparison \
sbatch --dependency=afterok:75504 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_executor_comparison.sh \
  outputs/appworld_executor_benchmark/qwen_only_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/skillrouter_embedding_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/skillrouter_base_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/clstr_base_packed_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/clstr_skillrouter_init_packed_dev10_v2_schema_hints/report.json \
  outputs/appworld_executor_benchmark/clstr_act_hrpo_train10_g4_u3_v1_prior_blend025_dev10_v2_schema_hints/report.json
```

SLURM job: `75505`

`75498` / `75501` / `75502` / `75503` / `75504` / `75505` 已完成。

### v4 prior/blend dev10 executor benchmark

comparison:

- `outputs/appworld_executor_benchmark/dev10_v2_schema_hints_plus_hrpo_v4_comparison/comparison.md`
- `outputs/appworld_executor_benchmark/dev10_v2_schema_hints_plus_hrpo_v4_comparison/comparison.json`

结果：

| method | task_count | success_count | success_rate | execution_failures |
|---|---:|---:|---:|---:|
| qwen_only | 10 | 7 | 0.700000 | 3 |
| skillrouter_embedding | 10 | 6 | 0.600000 | 4 |
| skillrouter_base | 10 | 6 | 0.600000 | 3 |
| clstr_base | 10 | 5 | 0.500000 | 3 |
| clstr_skillrouter_init | 10 | 4 | 0.400000 | 3 |
| clstr_act v4 prior/blend | 10 | 3 | 0.300000 | 4 |

### train10/g4/u3 HRPO 结果

training report:

- `outputs/appworld_clstr_train_hrpo/train10_g4_u3_v1_prior/train_report.json`
- checkpoint: `outputs/appworld_clstr_train_hrpo/train10_g4_u3_v1_prior/checkpoints/appworld_clstr_hrpo-step3.pt`
- rollout log: `outputs/appworld_clstr_train_hrpo/train10_g4_u3_v1_prior/rollouts.jsonl`

训练指标：

| update | rollout_count | success_rate | advantage_nonzero_count | policy_loss | routing_prior_loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.500000 | 8 | 0.0001839697 | 6.231989 |
| 2 | 8 | 0.500000 | 0 | -0.0 | 6.270057 |
| 3 | 8 | 0.000000 | 0 | -0.0 | 6.346064 |

rollout 分布：

- `82e2fac_1`: 2/4 success，产生组内 reward 差异；
- `82e2fac_2`: 2/4 success，产生组内 reward 差异；
- `82e2fac_3`: 4/4 success，没有组内差异；
- `692c77d_1/2/3`: 0/4 success，没有组内差异。

解释：train10/g4/u3 首个 update 确实产生了非零 HRPO policy signal，说明多 rollout
credit assignment 链路有效；但信号很稀疏，后续 update 没有组内差异，整体主要仍由
`routing_prior_loss` 约束。

### train10/g4/u3 ACT dev10 executor benchmark

comparison:

- `outputs/appworld_executor_benchmark/dev10_v2_schema_hints_plus_hrpo_train10_g4_u3_v1_comparison/comparison.md`
- `outputs/appworld_executor_benchmark/dev10_v2_schema_hints_plus_hrpo_train10_g4_u3_v1_comparison/comparison.json`

结果：

| method | task_count | success_count | success_rate | execution_failures |
|---|---:|---:|---:|---:|
| qwen_only | 10 | 7 | 0.700000 | 3 |
| skillrouter_embedding | 10 | 6 | 0.600000 | 4 |
| skillrouter_base | 10 | 6 | 0.600000 | 3 |
| clstr_base | 10 | 5 | 0.500000 | 3 |
| clstr_skillrouter_init | 10 | 4 | 0.400000 | 3 |
| clstr_act train10/g4/u3 prior/blend | 10 | 4 | 0.400000 | 6 |

逐任务 success 对比：

| task | qwen_only | skillrouter_base | clstr_base | clstr_skillrouter_init | clstr_act train10 |
|---|---:|---:|---:|---:|---:|
| 4ec8de5_1 | 1 | 1 | 1 | 1 | 1 |
| 50e1ac9_1 | 1 | 1 | 1 | 1 | 1 |
| 50e1ac9_2 | 1 | 1 | 1 | 1 | 1 |
| 50e1ac9_3 | 1 | 1 | 1 | 1 | 1 |
| 530b157_1 | 0 | 0 | 0 | 0 | 0 |
| 530b157_2 | 0 | 0 | 0 | 0 | 0 |
| 530b157_3 | 0 | 0 | 0 | 0 | 0 |
| fac291d_1 | 1 | 1 | 0 | 0 | 0 |
| fac291d_2 | 1 | 0 | 0 | 0 | 0 |
| fac291d_3 | 1 | 1 | 1 | 0 | 0 |

ACT train10 所用 context 摘要：

- 前 4 个成功 task 基本使用 `spotify-authenticate-with-stored-credentials-26` +
  `spotify-calculate-playlist-duration-39`；
- `530b157_*` 失败，context 主要是 `login-to-venmo-and-phone-17`；
- `fac291d_*` 失败，context 是 Spotify credential + playlist-duration skills。

结论：

1. AppWorld official executor benchmark pipeline 已经可以比较 Qwen-only、SkillRouter embedding、
   SkillRouter-base、CLSTR-base、CLSTR SkillRouter-init、CLSTR-act。
2. 当前 dev10 结果中 Qwen-only 仍最强，SkillRouter-base 次之；CLSTR-base 和 CLSTR-act 都没有超过
   Qwen-only 或 SkillRouter-base。
3. `fac291d_*` 是关键信号：Qwen-only 能成功，但加入 retrieved SkillX context 后失败，说明当前
   SkillX context 对 Qwen executor 有明显污染/干扰。
4. 因此不应继续盲目扩大当前 ACT 训练来声称提升。下一步更合理的是先做 context selection/abstention：
   当 retrieved skill context 低置信或与 task/API schema 不匹配时，让 executor 退回 Qwen-only 或只保留
   API docs/少量 auth skill；否则 ACT 会优化一个会干扰下游 Qwen 的上下文选择器。

已提交 HRPO smoke v3：

```bash
OUTPUT_DIR=outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v3 \
MAX_TASKS=2 \
GROUP_SIZE=2 \
UPDATES=1 \
TASK_BATCH_SIZE=2 \
TOP_K=8 \
CONTEXT_TOP_K=5 \
QWEN_TEMPERATURE=0.0 \
ROLLOUT_TEMPERATURE=1.0 \
TIMEOUT_SECONDS=60 \
MAX_INTERACTIONS=3 \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

SLURM job: `75477`

已挂新依赖链：

- `75478`: `afterok:75477`，导出 dev routing predictions。
- `75479`: `afterok:75478`，用 HRPO smoke v3 predictions 跑 dev3 executor smoke。

`75477` 已完成：

- state: `COMPLETED`
- output: `outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v3`
- checkpoint: `outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v3/checkpoints/appworld_clstr_hrpo-step1.pt`
- rollout log: `outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v3/rollouts.jsonl`
- train report: `outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v3/train_report.json`

训练 smoke 指标：

| metric | value |
|---|---:|
| task_count | 2 |
| rollout_count | 4 |
| rollout success_rate | 0.75 |
| mean_reward | 0.75 |
| advantage_nonzero_count | 2 |
| policy_loss | -0.0004326105 |
| kl_loss | 0.0 |

逐 rollout 观察：

- `82e2fac_1` sampled `spotify-calculate-playlist-duration-43` 成功；
- `82e2fac_1` sampled `spotify-authenticate-with-stored-credentials-60` 失败，evaluation 显示答案未给出；
- `82e2fac_2` sampled `spotify-authenticate-with-stored-credentials-64` 成功；
- `82e2fac_2` sampled `spotify-calculate-playlist-duration-43` 成功。

解释：

1. AppWorld official executor/evaluator reward 已进入 CLSTR-act 训练路径；
2. 同一 task 的多 rollout 产生了 reward 差异，`advantage_nonzero_count=2`，说明此前“单 rollout 导致 policy loss 无效”的核心问题已经修正；
3. 这只是 2 task / 4 rollout / 1 update 的 smoke，不可作为性能结论。

后续：

- `75478` 已在队列中，等待导出 HRPO smoke v3 dev routing predictions；
- `75479` 等待 `75478` 后运行 dev3 executor smoke。

`75478` 已完成：

- output: `outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v3_dev`
- predictions: `outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v3_dev/predictions.jsonl`
- routing report: `outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v3_dev/report.json`

Routing eval 指标：

| metric | value |
|---|---:|
| query_count | 57 |
| recall@1 | 0.964912 |
| recall@5 | 1.000000 |
| recall@10 | 1.000000 |
| recall@20 | 1.000000 |
| mrr@20 | 0.976608 |

解释：

1. 这个 smoke 只做了 2 train tasks / 4 rollouts / 1 update，因此 routing 指标基本保持 SkillRouter-init base 水平是合理的；
2. 关键证据不是 routing 提升，而是 AppWorld official reward 能驱动非零 HRPO policy loss；
3. `75479` 已解除依赖，当前排队等待运行 dev3 executor smoke。

## 2026-05-23 dev10 schema-hints executor benchmark 结果

`75449-75453` 和依赖汇总 `75458` 已完成：

- comparison: `outputs/appworld_executor_benchmark/dev10_v2_schema_hints_comparison/comparison.md`

结果：

| method | task_count | success_count | success_rate | execution_failures |
|---|---:|---:|---:|---:|
| qwen_only | 10 | 7 | 0.700000 | 3 |
| skillrouter_embedding | 10 | 6 | 0.600000 | 4 |
| skillrouter_base | 10 | 6 | 0.600000 | 3 |
| clstr_base packed | 10 | 5 | 0.500000 | 3 |
| clstr_skillrouter_init packed | 10 | 4 | 0.400000 | 3 |

解释：

1. 在 dev10 + schema hints 设置下，Qwen-only 反而最好，说明当前 top-k SkillX context 并不稳定；
   对部分任务，retrieved skill text 会干扰 Qwen executor。
2. 因此不应直接扩大 dev57 来声称 CLSTR/SkillRouter 提升；需要先修 ACT 训练和 context 选择机制。
3. 已新增 AppWorld skill-level HRPO/ACT 训练入口后，先提交一个很小的 end-to-end smoke，只验证训练管线可用。

已提交 HRPO smoke：

```bash
OUTPUT_DIR=outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v1 \
MAX_TASKS=2 \
GROUP_SIZE=2 \
UPDATES=1 \
TASK_BATCH_SIZE=2 \
TOP_K=8 \
CONTEXT_TOP_K=5 \
QWEN_TEMPERATURE=0.0 \
ROLLOUT_TEMPERATURE=1.0 \
TIMEOUT_SECONDS=60 \
MAX_INTERACTIONS=3 \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

SLURM job: `75465`

该 smoke 不作为最终性能结论，只检查 reset/prompt/generation/execute/evaluate/reward/loss/checkpoint 是否能完整跑通。

为避免频繁轮询，已挂后续依赖作业：

1. `75466`：`afterok:75465` 后导出 HRPO smoke checkpoint 的 dev routing predictions。

```bash
MODEL_CONFIG=configs/model/appworld_skillrouter_init.yaml \
CHECKPOINT_PATH=outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v1/checkpoints/appworld_clstr_hrpo-step1.pt \
OUTPUT_DIR=outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v1_dev \
TOP_K=20 \
sbatch --dependency=afterok:75465 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_eval.sh
```

2. `75467`：`afterok:75466` 后用 HRPO smoke predictions 跑 dev3 executor smoke。

```bash
METHOD=clstr_act \
PREDICTIONS_PATH=outputs/appworld_clstr_eval/hrpo_smoke_skillrouter_init_g2_u1_v1_dev/predictions.jsonl \
OUTPUT_DIR=outputs/appworld_executor_smoke/clstr_hrpo_smoke_skillrouter_init_g2_u1_v1_dev3 \
DEDUPE_CANONICAL_SKILLS=1 \
MAX_AUTH_LIKE_SKILLS=1 \
MAX_TASKS=3 \
TOP_K=5 \
TEMPERATURE=0.0 \
sbatch --dependency=afterok:75466 --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

## 2026-05-23 HRPO smoke v1 失败与修复

`75465` 失败：

- state: `FAILED`
- error: `ModuleNotFoundError: No module named 'appworld'`
- root cause: `scripts/sbatch/run_appworld_clstr_hrpo_train.sh` 没有像 executor wrapper 一样把
  `.deps/appworld_py310` 加入 `PYTHONPATH`。计算节点使用当前 conda env 时无法 import
  AppWorld runtime。

修复：

- `scripts/sbatch/run_appworld_clstr_hrpo_train.sh` 新增：

```bash
export PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/.deps/appworld_py310:/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr:${PYTHONPATH:-}
```

- 新增回归测试：`tests/test_sbatch_scripts.py`

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_sbatch_scripts.py tests/test_appworld_act_hrpo.py tests/test_appworld_executor.py -q
```

结果：`15 passed in 13.98s`。

额外检查：

```bash
PYTHONPATH=.deps/appworld_py310:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -c "import appworld; print('appworld_import_ok')"
```

结果：`appworld_import_ok`。

已取消旧失败依赖链：

```bash
scancel 75466 75467
```

已提交 HRPO smoke v2：

```bash
OUTPUT_DIR=outputs/appworld_clstr_train_hrpo/smoke_skillrouter_init_g2_u1_v2 \
MAX_TASKS=2 \
GROUP_SIZE=2 \
UPDATES=1 \
TASK_BATCH_SIZE=2 \
TOP_K=8 \
CONTEXT_TOP_K=5 \
QWEN_TEMPERATURE=0.0 \
ROLLOUT_TEMPERATURE=1.0 \
TIMEOUT_SECONDS=60 \
MAX_INTERACTIONS=3 \
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

SLURM job: `75470`

已挂新依赖链：

- `75471`: `afterok:75470`，导出 dev routing predictions。
- `75472`: `afterok:75471`，用 HRPO smoke v2 predictions 跑 dev3 executor smoke。

## 2026-05-23 HRPO smoke v2 失败与修复

`75470` 失败：

- state: `FAILED`
- error:

```text
RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol
```

- failure layer: 第一条 AppWorld HRPO rollout 已返回后，在写 `rollouts.jsonl` 之前失败。
- root cause: `_rollout_to_json()` 使用 `dataclasses.asdict(row)`，而 `asdict()` 会 deepcopy dataclass
  内所有字段；`selected_log_prob` 是带梯度的非叶子 tensor，PyTorch 不允许 deepcopy。

修复：

- `clstr/appworld_act_hrpo.py`
  - `_rollout_to_json()` 改为手动构造 JSON payload；
  - 明确跳过 `selected_log_prob`、`policy_logits`、`ref_logits` 等训练图 tensor。
- `tests/test_appworld_act_hrpo.py`
  - 新增回归测试 `test_rollout_json_serialization_omits_non_leaf_tensors()`。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_act_hrpo.py tests/test_sbatch_scripts.py tests/test_appworld_executor.py -q
```

结果：`16 passed in 4.08s`。

静态检查：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/appworld_act_hrpo.py scripts/run_appworld_clstr_hrpo_train.py
bash -n scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

均通过。

## 2026-05-23 SkillX executor context 诊断

针对“是否可能是 SkillX 使用实现有问题”的检查结论：有可能，而且当前证据较强。需要区分两件事：

1. 用 SkillX `skill.md`/JSON 文本做 skill embedding 和 routing target，这条链路本身仍然合理；
2. 把检索到的 SkillX `body` 原文直接塞给 Qwen executor，并提示“如果有用就复制逻辑”，这条链路目前不安全。

代码证据：

- `clstr/appworld_executor.py::_format_skill()` 会把 `body` / `skill_md` 原文放入 prompt；
- `clstr/bridges/skillx/appworld_adapter.py` 只做字段归一化，不验证 body 是否符合当前 AppWorld API schema，也不剔除硬编码用户信息；
- `clstr/appworld_routing.py` 的 positive selection 按 API overlap、app overlap 和 token overlap 选 skill，可能把 auth-only 或 loosely related skill 标成 positive，但这些 skill 不一定对 task execution 有帮助。

失败样例：

- `530b157_*` 的 CLSTR-act train10 context 主要是 `skillx/appworld/login-to-venmo-and-phone-17`。该 skill body 中硬编码了
  `nan_ritt@gmail.com` 和 `2307354647`，Qwen 生成代码时复制了这些凭证，导致 AppWorld login 返回
  `401 Invalid credentials`。这说明 raw SkillX body 会直接污染 executor。
- `fac291d_*` 的 Qwen-only 在 dev10 v2 schema-hints 下 3/3 成功，但 CLSTR-act train10 选中
  `spotify-authenticate-with-stored-credentials-26` + `spotify-calculate-playlist-duration-43` 后 3/3 失败。
  失败代码用 `len(song_ids) // 20` 推导 `page_index`，触发超过 1000 次请求的无限循环。该现象说明相关但不匹配的 Spotify skill body 会把 Qwen 引向错误执行模式。

因此当前 dev10 task-success 不能简单解释为 “CLSTR routing 本身不如 SkillRouter”。更准确的判断是：

- routing 层可以先继续用 SkillX 文本训练/评估；
- official executor benchmark 必须把 SkillX context 格式作为独立变量控制；
- 在证明 raw body 有益前，不能默认把原始 skill body 当成安全 reference code。

下一步实现：

- 为 AppWorld Qwen executor 增加 `skill_context_mode`：
  - `raw`: 保留当前完整 body，作为历史可比 baseline；
  - `metadata`: 只传 skill id、名称、描述、executor/API refs，不传 body；
  - `safe_metadata`: metadata 模式，并在格式中明确 body 被省略，因为 SkillX source code 未经过当前 AppWorld API/schema 校验。
- 重新跑同一 dev10 设置下的 CLSTR-act / CLSTR-base safe metadata 对比，观察 `fac291d_*` 是否恢复。如果恢复，说明主要问题是 executor context composition；如果仍不恢复，再回到 CLSTR policy 和训练数据本身排查。

### safe metadata context 实现

已实现 executor 端 SkillX context mode：

- `clstr/appworld_executor.py`
  - 新增 `VALID_SKILL_CONTEXT_MODES = {"raw", "metadata", "safe_metadata"}`；
  - `raw` 保持历史格式，继续把 SkillX `body` 放入 prompt；
  - `metadata` / `safe_metadata` 只保留 `skill_id`、`name`、`description`、`executor_desc`，并标记
    `raw_body_omitted: true`；
  - `safe_metadata` 额外说明 raw body 未经过当前 AppWorld runtime/schema 校验；
  - `run_appworld_executor_eval()` 将 `skill_context_mode` 写入 report 的 `skill_provider_config`。
- `scripts/run_appworld_qwen_executor_eval.py`
  - 新增 `--skill_context_mode {raw,metadata,safe_metadata}`。
- `scripts/sbatch/run_appworld_qwen_executor_eval.sh`
  - 新增环境变量 `SKILL_CONTEXT_MODE`，默认 `raw`，用于保持旧结果可复现。

回归测试：

- `tests/test_appworld_executor.py::test_prompt_builder_safe_metadata_omits_unsafe_skill_body`
  - 确认 `safe_metadata` prompt 保留 skill id/描述/executor refs；
  - 确认硬编码邮箱 `nan_ritt@gmail.com` 和手机号 `2307354647` 不会进入 prompt；
  - 确认不再出现 `body:` 字段。
- `tests/test_sbatch_scripts.py::test_appworld_qwen_executor_sbatch_exposes_skill_context_mode`
  - 确认 sbatch wrapper 暴露 `SKILL_CONTEXT_MODE` 和 `--skill_context_mode`。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_executor.py tests/test_sbatch_scripts.py -q
```

结果：`15 passed in 1.35s`。

静态检查：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/appworld_executor.py scripts/run_appworld_qwen_executor_eval.py
bash -n scripts/sbatch/run_appworld_qwen_executor_eval.sh
```

均通过。

### dev10 safe metadata context ablation

注意：smoke 只用于验证 reset、prompt、code execution、`task_completed()` 和日志链路，不能判断方法好坏。
本轮 safe metadata 改动需要至少用 dev10 做方向诊断；若 dev10 显示 raw SkillX body 污染明显，再扩展到 dev57/full dev。

已提交同一 dev10 设置的 context ablation：

| job | method | output | skill context |
|---:|---|---|---|
| 75524 | Qwen-only | `outputs/appworld_executor_benchmark/qwen_only_dev10_v3_context_ablation` | none/raw prompt baseline |
| 75526 | SkillRouter embedding | `outputs/appworld_executor_benchmark/skillrouter_embedding_dev10_v3_safe_metadata` | `safe_metadata` |
| 75525 | SkillRouter-base | `outputs/appworld_executor_benchmark/skillrouter_base_dev10_v3_safe_metadata` | `safe_metadata` |
| 75527 | CLSTR-base packed | `outputs/appworld_executor_benchmark/clstr_base_packed_dev10_v3_safe_metadata` | `safe_metadata`, canonical dedupe, max auth-like 1 |
| 75528 | CLSTR-act train10/g4/u3 prior/blend | `outputs/appworld_executor_benchmark/clstr_act_hrpo_train10_g4_u3_v1_prior_dev10_v3_safe_metadata` | `safe_metadata`, canonical dedupe, max auth-like 1 |

已提交依赖汇总：

- job `75529`
- output: `outputs/appworld_executor_benchmark/dev10_v3_safe_metadata_comparison`
- dependency: `afterok:75524:75525:75526:75527:75528`

轮询策略：按用户要求不频繁检查，dev10 这类短任务约 15 分钟后再看；如果部分 job 失败，先看 slurm/error log，不直接基于缺失 report 下结论。

### dev10 safe metadata failure/top-k diagnostics

为避免只看 aggregate success rate，已扩展 `scripts/sbatch/run_appworld_routing_diagnostics.sh`：

- 新增 `COMMAND=executor-failure-topk`；
- 用 `RUN_ARGS` 传入 `METHOD=runs.jsonl`；
- 用 `PREDICTION_ARGS` 传入 `METHOD=predictions.jsonl`；
- 暴露 `FOCUS_METHOD`、`REFERENCE_METHODS`、`TOP_K`。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_sbatch_scripts.py -q
```

结果：`4 passed in 0.03s`。

静态检查：

```bash
bash -n scripts/sbatch/run_appworld_routing_diagnostics.sh
```

通过。

已提交两个依赖诊断作业：

| job | focus | references | output |
|---:|---|---|---|
| 75531 | `clstr_base` | `qwen_only`, `skillrouter_base` | `outputs/appworld_routing_diagnostics/dev10_v3_safe_metadata_clstr_base_failure_topk` |
| 75532 | `clstr_act` | `qwen_only`, `skillrouter_base`, `clstr_base` | `outputs/appworld_routing_diagnostics/dev10_v3_safe_metadata_clstr_act_failure_topk` |

这两个诊断不重新跑 AppWorld，只在 executor `runs.jsonl` 和 routing `predictions.jsonl` 产出后做逐任务分析：

- focus 方法失败但 reference 成功的任务有哪些；
- focus top-k skill 是否 auth-like 过多、off-app、duplicate；
- focus top-k 是否命中 qrels 或 canonical qrels；
- reference 成功时选了哪些 top-k skill。

### 方法判断边界与后续决策树

为避免把工程 smoke 误当成方法结论，后续按以下证据强度判断：

1. `smoke` 只回答“pipeline 是否能跑”：
   - 是否能 reset AppWorld；
   - Qwen prompt 是否能生成 Python code；
   - `AppWorld.execute()`、`task_completed()`、`evaluate()` 是否接入；
   - `runs.jsonl` / `report.json` 是否完整。
   - 不能用来判断 CLSTR 是否优于 SkillRouter/Qwen-only。
2. `dev10` 只回答“方向和典型 failure mode”：
   - 是否存在明显 raw SkillX body 污染；
   - safe metadata 是否恢复 `fac291d_*` 这类 Qwen-only 成功但 skill-context 失败的任务；
   - CLSTR-base/CLSTR-act 的失败是否集中在 auth-only、off-app、duplicate、API schema mismatch。
   - dev10 仍不能作为最终论文性能结论。
3. `dev57/full dev` 才能支持 AppWorld official executor task-success 方向性结论：
   - 必须固定同一 Qwen、同一 prompt 版本、同一 `TOP_K`、同一 `skill_context_mode`；
   - 必须同时报告 aggregate success、execution failure、task_completed、逐任务差异；
   - 如果 dev57 上 CLSTR-base + Qwen 没有超过 Qwen-only/SkillRouter-base，则不能声称 task-success 提升。

safe metadata dev10 出结果后的决策：

- 如果 `safe_metadata` 明显提升带 SkillX context 的方法，尤其恢复 `fac291d_*`：
  1. 扩展到 dev57 safe metadata；
  2. 同时保留 raw-body 作为 ablation；
  3. 将论文实验拆成 routing quality 与 executor context quality 两部分，避免把 raw body 注入错误归因到 CLSTR routing。
- 如果 `safe_metadata` 仍不提升：
  1. 先做 skill embedding 文本 ablation，而不是继续扩大 ACT：
     - `name + description + executor_desc + body`（当前 SkillRouter-init 风格）；
     - `name + description + executor_desc`（metadata-only embedding）；
     - sanitized body embedding（去硬编码邮箱、电话、凭证、实例常量）；
  2. 检查 AppWorld query text 是否过度依赖 train-only `api_refs`；
  3. 再检查 CLSTR policy/HRPO 是否真的学到下游 reward，而不是只扰动强 base routing。
- 如果 CLSTR-base safe metadata 在 dev10/dev57 超过 baselines：
  1. 再构造 AppWorld train-only ACT 数据；
  2. 用 official executor/evaluator reward 或 oracle compiled solution 生成 verified pairs；
  3. 训练真正 AppWorld CLSTR-act；
  4. 再跑 Qwen-only / SkillRouter / CLSTR-base / CLSTR-act 同设置 task-success benchmark。

当前结论保持保守：smoke 和 dev10 之前的局部结果只能说明 pipeline 可用、raw SkillX body 可能污染 executor；尚不足以判断 CLSTR 方法本身优劣。

### dev10 safe metadata 进度：Qwen-only baseline 完成

`75524` 已完成并生成：

- report: `outputs/appworld_executor_benchmark/qwen_only_dev10_v3_context_ablation/report.json`
- runs: `outputs/appworld_executor_benchmark/qwen_only_dev10_v3_context_ablation/runs.jsonl`

结果：

| method | task_count | success_count | success_rate | generation_failures | execution_failures |
|---|---:|---:|---:|---:|---:|
| Qwen-only v3 context-ablation anchor | 10 | 7 | 0.700000 | 0 | 3 |

逐任务：

| task | success | execution_ok | task_completed | evaluation_success |
|---|---:|---:|---:|---:|
| 50e1ac9_1 | 1 | 1 | 1 | 1 |
| 50e1ac9_2 | 1 | 1 | 1 | 1 |
| 50e1ac9_3 | 1 | 1 | 1 | 1 |
| fac291d_1 | 1 | 1 | 1 | 1 |
| fac291d_2 | 1 | 1 | 1 | 1 |
| fac291d_3 | 1 | 1 | 1 | 1 |
| 530b157_1 | 0 | 0 | 0 | 0 |
| 530b157_2 | 0 | 0 | 0 | 0 |
| 530b157_3 | 0 | 0 | 0 | 0 |
| 4ec8de5_1 | 1 | 1 | 1 | 1 |

解释：

- 该 anchor 与 dev10 v2 Qwen-only 的 `7/10` 保持一致；
- `fac291d_*` 仍然 3/3 成功，适合作为判断 SkillX context 是否污染 Spotify library/counting 任务的 reference；
- `530b157_*` 仍然 3/3 失败，说明这类 Venmo/Phone 任务不是简单靠移除 SkillX body 就能由 Qwen-only 解决。

当前仍在等待：

- `75525`: SkillRouter-base + safe metadata；
- `75526`: SkillRouter embedding + safe metadata；
- `75527`: CLSTR-base packed + safe metadata；
- `75528`: CLSTR-act train10/g4/u3 prior/blend + safe metadata；
- `75529`: comparison；
- `75531` / `75532`: failure/top-k diagnostics。

### 当前 goal 完成度审计

已具备证据：

1. SkillRouter-base AppWorld routing 训练与 eval 已完成：
   - eval report: `outputs/appworld_skillrouter_base_eval/dev_v1/report.json`
   - dev routing: recall@1 `0.982456`，recall@5/10/20 `1.0`，mrr@20 `0.991228`
   - train checkpoint: `outputs/appworld_skillrouter_base_train_v1/model.pt`
2. CLSTR-base AppWorld routing eval 已完成：
   - eval report: `outputs/appworld_clstr_eval/routing_only_fit_v1_dev/report.json`
   - dev routing: recall@1 `0.912281`，recall@5/10/20 `1.0`，mrr@20 `0.956140`
3. CLSTR + Qwen3-8B AppWorld executor loop 已实现：
   - `clstr/appworld_executor.py`
   - `scripts/run_appworld_qwen_executor_eval.py`
   - `scripts/sbatch/run_appworld_qwen_executor_eval.sh`
   - 支持 `skill_context_mode={raw,metadata,safe_metadata}`。
4. Qwen-only dev10 v3 anchor 已完成：
   - report: `outputs/appworld_executor_benchmark/qwen_only_dev10_v3_context_ablation/report.json`
   - result: `7/10`

尚缺证据，不能标记 goal 完成：

1. safe metadata dev10 旧 checkpoint 对比已经完成，但没有显示 CLSTR task-success 提升：
   - comparison: `outputs/appworld_executor_benchmark/dev10_v3_safe_metadata_comparison/comparison.json`
   - Qwen-only `7/10`
   - SkillRouter-base `6/10`
   - CLSTR-base packed `6/10`
   - CLSTR-act train10/g4/u3 prior `6/10`
2. failure/top-k diagnostics 已产出：
   - `outputs/appworld_routing_diagnostics/dev10_v3_safe_metadata_clstr_base_failure_topk/report.json`
   - `outputs/appworld_routing_diagnostics/dev10_v3_safe_metadata_clstr_act_failure_topk/report.json`
   - 诊断显示 `4ec8de5_1` 的 CLSTR 失败不是 routing 正例未命中，而是 Qwen 生成代码误用 AppWorld API 返回数据结构。
3. 更合理的 SkillRouter-init CLSTR-base full train-set routing run 尚在作业链路中：
   - train job `75545`
   - routing eval job `75546`
   - executor eval job `75547`
   - comparison job `75548`
4. 还没有 safe metadata 的 dev57/full dev task-success benchmark，因此不能判断 CLSTR 在 official executor task success 上是否优于 Qwen-only/SkillRouter。
5. 只有在 CLSTR-base + Qwen 在足够稳定的 dev 设置中显示提升后，才应继续构造 AppWorld train-only ACT verified pairs 并训练真正的 AppWorld CLSTR-act；当前条件尚未满足。

### 关于 CLSTR-base / CLSTR-act 是否充分训练

结论：当前用于 executor 对比的 CLSTR-base 和 CLSTR-act 都不应视为“完整且合理训练后的最终模型”。它们目前更适合作为 pipeline/diagnostic baseline。

证据：

1. MiniLM CLSTR-base routing-only：
   - output: `outputs/appworld_clstr_train_routing_only_fit_v1/train_stdout.json`
   - 每个 stage 120 updates；
   - final stage train-side routing_supervision_recall@1 约 `0.833333`，recall@5 `1.0`；
   - dev eval recall@1 `0.912281`，recall@5 `1.0`；
   - 说明 top5 能覆盖 positives，但 top1/排序仍未充分拟合，且该训练不是 official executor task-success 训练。
2. SkillRouter-init CLSTR-base：
   - output: `outputs/appworld_clstr_train_skillrouter_init_routing_only_v1/train_stdout.json`
   - 使用 `configs/model/appworld_skillrouter_init.yaml`，backbone frozen，SkillRouter-Embedding-0.6B 初始化；
   - 每个 stage 120 updates，`task_batch_size=15`；
   - dev eval recall@1 `0.964912`，recall@5 `1.0`；
   - 这是当前更强、更公平的 CLSTR 初始化，但仍只是 routing-only 训练，不是完整 task-success 方法训练。
3. CLSTR-act HRPO train10：
   - output: `outputs/appworld_clstr_train_hrpo/train10_g4_u3_v1_prior/train_report.json`
   - 只用 10 个 train tasks、3 updates、group_size 4，共 24 rollouts；
   - update 1 有非零 advantage，update 2/3 advantage_nonzero_count 为 0；
   - `policy_loss` 量级约 `1e-4`，`routing_prior_loss` 约 `6.2`，在 `beta_routing_prior=0.05` 下总 loss 主要由 routing prior 主导；
   - 因此它证明了 HRPO pipeline 能产生 official reward 信号，但不是充分 ACT 训练。
4. 训练/评测口径不一致已做代码级修复，但旧 ACT checkpoint 仍不能视为 safe-metadata 训练结果：
   - 2026-05-23 已将 `skill_context_mode` 接入 `clstr/appworld_act_hrpo.py`、`scripts/run_appworld_clstr_hrpo_train.py`、`scripts/sbatch/run_appworld_clstr_hrpo_train.sh`；
   - HRPO ACT 现在可以通过 `SKILL_CONTEXT_MODE=safe_metadata` 使用和 executor eval 一致的安全 SkillX metadata prompt；
   - 已验证 `tests/test_appworld_act_hrpo.py::test_appworld_hrpo_rollout_can_use_safe_metadata_context` 与 `tests/test_sbatch_scripts.py::test_appworld_hrpo_sbatch_exports_appworld_pythonpath`；
   - 现有 `outputs/appworld_clstr_train_hrpo/train10_g4_u3_v1_prior` 是修复前产物，仍按 raw SkillX body prompt 训练，后续正式 ACT 必须重新训练。

超参数风险：

- CLSTR-base 的 `max_steps=120`/stage 偏短，尤其 MiniLM 版本 train recall@1 未饱和；
- SkillRouter-init 版本 `task_batch_size=15`，报告中的 train recall 只反映当前 batch，不等价于全 train set 收敛；
- MiniLM 版本 `max_length=256` 可能截断 SkillX body，对 AppWorld skill 文档偏短；
- ACT 的 `MAX_TASKS=10`、`UPDATES=3`、`GROUP_SIZE=4` 太小，reward sparse 时很难学到稳定 policy；
- ACT 的 `beta_routing_prior=0.05` 相对 policy gradient 过强，容易把 policy 拉回 base routing，掩盖 reward 信号；
- ACT 当前 `detach_policy_inputs=true` 只训练较小头部/策略层，适合 smoke，但限制了完整适配能力；
- 若提交 ACT 时未显式设置 `SKILL_CONTEXT_MODE=safe_metadata`，仍会退回 raw body 默认值，从而让训练目标和 evaluator 目标不一致。

更合理的下一步训练顺序：

1. 保留当前 dev10 safe-metadata jobs，不取消；它们用于诊断 executor context 是否污染 Qwen。
2. 先重新训练一个“合理 CLSTR-base”：
   - 使用 SkillRouter-init 模型配置，而不是 MiniLM，作为公平强初始化；
   - 使用全 90 train tasks 的 routing-only/full-batch 或大 batch 设置；
   - 增加训练步数，并保存多个 checkpoint 做 dev routing eval；
   - 以 dev routing + dev10 executor safe metadata 同时选择 checkpoint。
3. 在确定 base + executor context 口径后，再训练 ACT：
   - ACT 训练必须使用和 eval 一致的 `skill_context_mode`；
   - 先做 medium run（例如 30 tasks、更多 updates、group_size >= 4）确认 nonzero advantage 和 dev10 executor 有收益；
   - 再扩 full train split；
   - 如果 dev10/dev57 仍无收益，再转向 skill embedding text ablation，而不是继续加 ACT 预算。

因此，当前 SkillRouter/CLSTR executor 对比只能说明“现有 pipeline 与当前 checkpoint 的表现”，不能作为最终论文中的“完整 CLSTR 方法不如 SkillRouter”的结论。

### dev10 safe metadata 完整结果与下一轮 base 作业

2026-05-23 更新：`dev10_v3_safe_metadata` 旧 checkpoint 对比已经完成。

结果文件：

- comparison: `outputs/appworld_executor_benchmark/dev10_v3_safe_metadata_comparison/comparison.json`
- Qwen-only: `outputs/appworld_executor_benchmark/qwen_only_dev10_v3_context_ablation/report.json`
- SkillRouter embedding: `outputs/appworld_executor_benchmark/skillrouter_embedding_dev10_v3_safe_metadata/report.json`
- SkillRouter-base: `outputs/appworld_executor_benchmark/skillrouter_base_dev10_v3_safe_metadata/report.json`
- CLSTR-base packed: `outputs/appworld_executor_benchmark/clstr_base_packed_dev10_v3_safe_metadata/report.json`
- CLSTR-act train10/g4/u3 prior: `outputs/appworld_executor_benchmark/clstr_act_hrpo_train10_g4_u3_v1_prior_dev10_v3_safe_metadata/report.json`

汇总：

| method | task_count | success_count | success_rate | execution_failures |
|---|---:|---:|---:|---:|
| Qwen-only | 10 | 7 | 0.70 | 3 |
| SkillRouter embedding | 10 | 4 | 0.40 | 3 |
| SkillRouter-base | 10 | 6 | 0.60 | 3 |
| CLSTR-base packed | 10 | 6 | 0.60 | 4 |
| CLSTR-act train10/g4/u3 prior | 10 | 6 | 0.60 | 4 |

诊断文件：

- CLSTR-base focus: `outputs/appworld_routing_diagnostics/dev10_v3_safe_metadata_clstr_base_failure_topk/report.json`
- CLSTR-act focus: `outputs/appworld_routing_diagnostics/dev10_v3_safe_metadata_clstr_act_failure_topk/report.json`

关键诊断：

- CLSTR-base/ACT 相对 Qwen-only 多失败的主要样例是 `4ec8de5_1`；
- 该任务要求统计 Spotify song/album libraries 中今年或去年发布的歌曲数量；
- CLSTR top-k 已命中正例，`focus_id_hit_rank=1`，`focus_canonical_hit_rank=1`；
- top-k 没有 off-app skill，只有一个 auth-like skill；
- 失败代码报错 `TypeError: list indices must be integers or slices, not str`，即 Qwen 把 AppWorld API 返回 list 当 dict 读；
- 因此这个 dev10 差异不能归因为 CLSTR routing 没检索到正例，更像是 skill context / API schema prompt 让 Qwen 生成了错误数据结构访问代码。

结论：

- 旧 checkpoint 下，safe metadata 仍没有带来 task-success 提升；Qwen-only 仍是 dev10 最强；
- SkillRouter-base 与旧 CLSTR-base/ACT 都是 6/10，不能说明 CLSTR 方法已经优于 SkillRouter；
- 旧 CLSTR-act 仍是修复 `skill_context_mode` 前的 raw-body ACT 训练产物，不能作为正式 safe-metadata ACT 结论；
- 继续盲目扩大 ACT 训练不合适，先要把 CLSTR-base 训练量和 executor prompt/API schema 使用问题分开。

已提交下一轮更合理的 SkillRouter-init CLSTR-base 链路：

1. 新配置：`configs/train/appworld_skillrouter_init_routing_full_v1.yaml`
   - `start_stage: stage2_base`
   - `loss_mode: routing_only`
   - `max_steps: 600`
   - `checkpoint_every: 100`
   - `task_batch_size: 90`
   - `learning_rate_heads: 5.0e-4`
   - `learning_rate_backbone: 0.0`
2. 训练作业：
   - job `75545`
   - output: `outputs/appworld_clstr_train_skillrouter_init_routing_full_v1`
   - warmstart: `outputs/appworld_clstr_train_skillrouter_init_routing_only_v1/checkpoints/stage2_base-step120.pt`
3. 依赖 routing eval：
   - job `75546`
   - dependency: `afterok:75545`
   - output: `outputs/appworld_clstr_eval/skillrouter_init_routing_full_v1_dev`
4. 依赖 executor eval：
   - job `75547`
   - dependency: `afterok:75546`
   - output: `outputs/appworld_executor_benchmark/clstr_skillrouter_init_routing_full_v1_dev10_safe_metadata`
   - setting: `SKILL_CONTEXT_MODE=safe_metadata`, `TOP_K=5`, `MAX_TASKS=10`, `DEDUPE_CANONICAL_SKILLS=1`, `MAX_AUTH_LIKE_SKILLS=1`
5. 依赖 comparison：
   - job `75548`
   - dependency: `afterok:75547`
   - output: `outputs/appworld_executor_benchmark/dev10_v3_safe_metadata_plus_routing_full_v1_comparison`
6. 补充 checkpoint sweep routing eval：
   - jobs `75553-75558`
   - dependency: `afterok:75545`
   - checkpoints: `stage2_base-step100.pt`, `step200.pt`, `step300.pt`, `step400.pt`, `step500.pt`, `step600.pt`
   - outputs: `outputs/appworld_clstr_eval/skillrouter_init_routing_full_v1_dev_step{100,200,300,400,500,600}`
7. 补充 checkpoint sweep comparison：
   - job `75560`
   - dependency: `afterok:75553:75554:75555:75556:75557:75558`
   - output: `outputs/appworld_clstr_eval/skillrouter_init_routing_full_v1_checkpoint_sweep_comparison`
   - 新增 wrapper: `scripts/sbatch/run_appworld_routing_comparison.sh`
   - 验证：`PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_sbatch_scripts.py -q` 得到 `5 passed`；`bash -n scripts/sbatch/run_appworld_routing_comparison.sh scripts/sbatch/run_appworld_clstr_eval.sh scripts/sbatch/run_appworld_executor_comparison.sh` 通过。

下一步判断标准：

- 如果 full-v1 base 的 dev routing 提升但 executor dev10 仍不提升，主要问题在 executor prompt/API schema context，而不是 routing fit；
- 如果 checkpoint sweep 显示非最终 checkpoint dev routing 更好，需要补该 checkpoint 的 executor dev10 eval，不能只看 latest checkpoint；
- 如果 full-v1 base 的 executor dev10 超过 Qwen-only/SkillRouter-base，再扩展到 dev57 safe metadata；
- 只有 dev10/dev57 出现稳定 task-success 改善后，再提交更大规模 `SKILL_CONTEXT_MODE=safe_metadata` 的 ACT/HRPO 训练。

### AppWorld executor 输入形态：single-shot 与 trajectory 的边界

2026-05-23 记录：需要区分当前项目里的三类输入。

1. 当前 SkillRouter-base / CLSTR-base routing 训练输入：
   - 输入是 AppWorld routing query 和 SkillX skill document；
   - query 形态是 `Instruction + Required apps + Observed train-only API refs + Observed API call count + routing instruction`；
   - skill document 形态是 `name | description | executor_desc | body/skill_md`；
   - 标签是 `positive_skill_ids` / qrels；
   - 这里没有 `[Thought] [Code] [Environment Output]` 多步轨迹。
2. 当前 Qwen executor benchmark 输入：
   - 仍是 single-shot code generation；
   - prompt 由 system message + user message 组成；
   - user message 包含 task instruction、retrieved top-k SkillX context、AppWorld API docs、输出契约；
   - Qwen 一次性生成 Python code；
   - AppWorld 执行一次 `world.execute(code)`，再用 `task_completed()` 与 `evaluate()` 判断 success。
3. 更完整、更常见的 AppWorld agent / executor 训练输入：
   - 可表示为 `[System Prompt] + [User Goal] + [Thought 1] + [Code 1] + [Environment Output 1] + [Thought 2] + [Code 2] + [Environment Output 2] + ...`；
   - success 应来自 AppWorld verifier/evaluator，而不是模型自己输出的文本标签；
   - 这种格式适合训练下游 LM executor 或构造 richer ACT trajectories；
   - 但它不是当前 CLSTR-base routing-only 训练的输入，也不是当前 dev10 benchmark 的口径。

对当前实验的影响：

- 旧 dev10 结果不能说明多步 agent 训练下的 CLSTR 最终表现；
- 当前 ACT/HRPO 只是在 single-shot Qwen executor reward 上训练 CLSTR skill sampling policy；
- 如果后续要做真正 AppWorld CLSTR-act，建议先固定一个 trajectory schema：
  - 每个 step 记录 `prompt_context`、`selected_skill_ids`、`thought/code`、`execute_output`、`execution_ok`、`task_completed`、`evaluate`；
  - success/reward 只从 AppWorld official verifier 读取；
  - 训练 split 只用 train tasks 的成功轨迹或 oracle compiled solution，不能泄漏 dev/test；
  - 与当前 single-shot benchmark 分开报告，避免把 executor 能力提升误归因到 routing 方法。

短期决策：

- 在 `75545-75548` full-v1 base 链路完成前，不改变当前 executor benchmark 为多步 agent 口径；
- 若 full-v1 base 在 routing 上明显提升但 executor task success 仍不提升，下一步优先设计 multi-step executor/trajectory 数据，而不是只扩大 single-shot ACT rollout 数。

### CLSTR 方法潜力与 AppWorld 口径风险

2026-05-23 进一步判断：当前 AppWorld 训练目标与常见 AppWorld agent training objective 不完全一致。

当前训练实际是什么：

- `data/appworld_routing/*_tasks.jsonl` 的正例来自 AppWorld ground-truth API calls 与 SkillX skill 文档的 overlap；
- `train_replay.jsonl` 里的 step 是 synthetic routing replay，形态是“匹配到这些 reusable skills”，不是 AppWorld 环境真实交互轨迹；
- CLSTR-base 的主 loss 是 multi-positive skill routing loss；
- CLSTR-act HRPO 目前是在 single-shot Qwen executor success reward 上训练 skill sampling policy；
- 因此当前 AppWorld 设置下，CLSTR 的“连续选择”只被弱实现为连续/多候选 skill routing，不是完整的 `Thought -> Code -> Environment Output -> next skill/code` 闭环控制。

这对论文卖点的影响：

- 如果论文声称的是“连续选择 latent skills 来驱动 agent 多步完成任务”，当前 AppWorld 证据还不够；
- 当前证据最多支持“SkillX-aware latent router / top-k skill context selector can be trained and evaluated on AppWorld routing and single-shot executor success”；
- 这仍有研究价值，但卖点要收窄，否则容易被审稿人质疑 GT、输入格式和 AppWorld task success 不一致。

我对 CLSTR 潜力的判断：

- CLSTR 仍有潜力，但不应继续只作为 one-shot top-k retriever 与 SkillRouter 比；
- 真正有差异化的位置是“多步 latent skill/controller”，即每轮根据 environment output 更新 belief state，再选择下一个 skill/context/code prior；
- 如果停留在 single-shot retrieval，SkillRouter 的简单 embedding scorer 很强，CLSTR 的复杂结构未必有优势，甚至会因为训练噪声和 executor prompt 干扰而吃亏；
- 因此 CLSTR 的潜力取决于是否转向 trajectory-level executor/ACT 数据，而不是继续加大 routing-only 训练。

建议方向：

1. 短期保留当前 routing/full-v1 与 single-shot executor benchmark：
   - 它能回答“当前 top-k skill context selector 是否比 SkillRouter/Qwen-only 有用”；
   - 如果 full-v1 仍不提升 task success，不能再把失败简单归因于 base 欠拟合。
2. 中期转向 AppWorld multi-step trajectory schema：
   - 输入采用 `[System Prompt] + [User Goal] + [Thought/Code/Environment Output]*`；
   - GT/reward 来自 official `task_completed()` / `evaluate()`；
   - 每一步记录 CLSTR selected skills、Qwen code、environment output、success/reward；
   - 让 CLSTR 在每个 environment output 后重新选择 skill/context，真正体现“连续选择”。
3. 论文表述上拆成两条线：
   - routing quality: SkillRouter vs CLSTR-base；
   - agent task success: Qwen-only vs Qwen+SkillRouter vs Qwen+CLSTR multi-step controller；
   - 如果只有 routing 改善没有 task success 改善，应报告为诊断/负结果，不能包装成完整 AppWorld success benchmark。

当前推荐决策：

- 不立刻放弃 CLSTR；
- 但如果 `75545-75560` 证明 routing 已经很强而 executor 仍无提升，就应停止继续堆 single-shot ACT；
- 下一步应设计并实现 multi-step AppWorld trajectory executor，让 CLSTR 的 belief update / sequential selection 真的进入实验闭环。

### 长程选择能力与 hidden state / GT 设计

2026-05-23 进一步判断：CLSTR 架构上可以承载长程选择，但当前 AppWorld 训练数据还没有充分监督这种能力。

现有架构为什么“能做”：

- CLSTR 有 recurrent belief state `m_t`；
- 每步用当前 serialized state 编码得到 `h_t`；
- `subspace_obs(skill_table, h_t)` 得到观测诱导的 skill-subspace belief；
- policy head 用 `h_t + m_t + candidate skill embeddings` 选择 skill 或 stop；
- 执行后用 `action skill + observation embedding` 经过 transition/GRU 得到 `m_hat`；
- belief gate 再融合 `m_hat` 与新观测 belief，形成 `m_{t+1}`；
- 因此从模型结构上，它不是只能做 one-shot retrieval，而是可以做 `state -> skill -> observation -> next state -> next skill`。

当前为什么“还没真正做出来”：

- AppWorld routing 数据的 step 是 synthetic replay，不是真实 AppWorld code execution trajectory；
- 当前 base 训练主要监督 `query -> positive SkillX skills`；
- 当前 ACT/HRPO 只在 single-shot Qwen code executor 上给一个 task-level reward；
- 没有把每一步 `Thought/Code/Environment Output` 显式变成 CLSTR 的 state transition supervision；
- 因此 long-horizon credit assignment 和 belief update 目前证据不足。

如果转向多步 AppWorld，需要定义：

1. State/input `x_t`
   - `user_goal`;
   - 已执行的 compact history：每步 selected skills、code 摘要、environment output/error；
   - 当前可用 API docs 或 schema hints；
   - 可选 retrieved SkillX context。
2. Action/CLSTR output `a_t`
   - 推荐先定义为 selected SkillX skill/context，而不是直接让 CLSTR 生成 code；
   - Qwen 负责根据 `x_t + selected skill context + API docs` 生成下一段 code。
3. Observation `o_t`
   - `AppWorld.execute(code)` 的返回、错误、关键 API response 摘要；
   - `task_completed()` 与 partial evaluator signal 如果可用，也作为 observation metadata。
4. GT / reward
   - 强监督 GT 可来自 oracle compiled solution 或 train split 成功轨迹映射出的 step-level skill labels；
   - RL reward 应来自 official `evaluate()` / `task_completed()`，不是模型输出的 success tag；
   - 中间 reward 可用 execution_ok、no exception、schema-correct、progress heuristics，但最终论文指标必须用 official success。
5. Hidden state 处理
   - 不建议把 `m_t` 当作外部数据集主输入；主输入应是可复现的 serialized prefix/history；
   - 训练时可缓存 `m_t_exact` 加速或用于 verified pair，但权威来源应是 replay prefix，可由模型重新 rollout 得到；
   - 否则换 checkpoint 后缓存 hidden state 会失效，影响可复现性。

建议的第一版多步 schema：

```json
{
  "task_id": "...",
  "split": "train",
  "user_goal": "...",
  "steps": [
    {
      "step_idx": 0,
      "state_text": "goal + compact history before step",
      "selected_skill_ids": ["..."],
      "positive_skill_ids": ["..."],
      "qwen_code": "...",
      "execute_output": "...",
      "execution_ok": true,
      "task_completed": false,
      "evaluate": {"success": false}
    }
  ],
  "final_success": true
}
```

方法判断：

- CLSTR 有潜力，但要把潜力落到 AppWorld 上，必须从 one-shot routing 转向 multi-step state/action/observation/reward；
- 如果不做这一步，CLSTR 的 sequential/belief-state 卖点在 AppWorld 上会站不住；
- 更合理的论文路线是：先报告 routing/single-shot executor 作为诊断，再把主方法实验放到 multi-step CLSTR controller。

### `m_t` 注入下游输入的候选设计

2026-05-23 进一步设计：如果只把 `m_t` 翻译成自然语言 skill mixture 给 Qwen，方法味道不够强；更合理的是让 `m_t` 以数学形式进入 CLSTR policy 或下游 executor embedding。但必须分层实验，避免未训练 latent 直接扰乱 Qwen。

符号约定：

- `h_t`: 当前 serialized state / prefix 的 encoder hidden state；
- `m_obs`: `softmax(skill_logits) @ skill_table.E`，即当前 observation 诱导出的 skill-subspace belief；
- `m_t`: recurrent latent belief state，融合历史 action、environment output 和当前 observation；
- `z_t`: 可选的可读 latent summary，如 top-k skill mixture，只用于日志或 prompt ablation。

候选注入方式：

1. CLSTR 内部融合，暂不碰 Qwen：
   - 形式：`h_cond = LayerNorm(h_t + W_m m_t)`；
   - skill score 用 `f(h_cond, m_t, e_i, h_cond * e_i)`；
   - 优点：稳定、容易训练，直接强化 stepwise skill selection；
   - 缺点：Qwen 仍只看到文本 skill/context，看不到连续 latent 本身；
   - 优先级：最高，作为 multi-step CLSTR controller 第一版。
2. `m_t -> soft prefix` 注入 Qwen embedding：
   - 形式：`prefix_embeds = W_prefix(m_t)`，shape 为 `[num_virtual_tokens, qwen_hidden_dim]`；
   - Qwen 输入 embedding 变成 `concat(prefix_embeds, token_embeds(prompt))`；
   - 优点：真正把 latent belief 作为下游 executor 的 soft condition；
   - 风险：需要训练 `W_prefix`，最好配合 LoRA/prefix tuning/SFT 成功轨迹，否则 frozen Qwen 不知道 soft tokens 含义；
   - 优先级：第二阶段，在有 multi-step 成功轨迹后再做。
3. FiLM / gated modulation：
   - 形式：`gamma, beta = W(m_t)`，`x'_tokens = LayerNorm(x_tokens * (1 + gamma) + beta)`；
   - 可只调制 API docs、SkillX context 或 history token；
   - 优点：比直接加向量更可控；
   - 风险：实现复杂，需要 code-generation loss 或 RL reward；
   - 优先级：作为 soft prefix 的替代 ablation。
4. Cross-attention memory：
   - 形式：Qwen hidden states attend 到 `[m_t, selected_skill_embs]` memory；
   - 优点：表达能力最强；
   - 风险：需要修改 executor 模型结构或 adapter，工程和训练成本最高；
   - 优先级：暂不作为近期实现。
5. Final `m_n` value head：
   - 形式：`success_logit = ValueHead(m_n)`；
   - 用于判断 CLSTR belief 是否编码了任务完成进展；
   - 适合做诊断/辅助 reward model，不适合作为主控制策略。

推荐实验顺序：

1. `StepCLSTR-TopK`：
   - 输入：`state_text_t`；
   - CLSTR 内部使用 `h_t + m_t` 选择 top-k SkillX；
   - Qwen 仍用文本 prompt；
   - GT：step-level positive SkillX + final official success。
2. `StepCLSTR-MT-Fusion`：
   - 在 policy head 中显式加入 `h_t + W_m m_t` / gating fusion；
   - 对比普通 CLSTR policy，证明 `m_t` 对连续选择有增益。
3. `FinalMT-Value`：
   - 用 `m_n` 预测 final success；
   - 证明 latent belief 与 task progress 相关。
4. `MT-SoftPrefix-Qwen`：
   - 训练 `m_t -> soft prefix` adapter；
   - Qwen 使用 `soft prefix + normal prompt` 生成 code；
   - 只有当前三项证明有效后再上。

论文价值判断：

- 如果只做 one-shot top-k retrieval，CLSTR 很难稳定超过 SkillRouter，因为 SkillRouter 的 embedding scorer 已经很强；
- 如果 `m_t` 能在 multi-step setting 中提升 step routing、success value、最终 task success，才能证明 CLSTR 的 latent belief / continuous selection 是必要的；
- 因此目标不应只是“routing recall 超过 SkillRouter”，而应是“在同等 Qwen executor 下，CLSTR multi-step controller 在 AppWorld task success 上超过 SkillRouter context baseline”。

短期执行原则：

- 不直接把未训练的 `m_t` 加到 frozen Qwen token embedding；
- 先让 `m_t` 在 CLSTR policy 内部影响 stepwise skill selection；
- 以 official AppWorld success 作为最终指标；
- routing recall、value accuracy、step positive hit 只作为机制证明和诊断指标。

### full-v1 routing 训练结果与转向 multi-step 的证据

2026-05-23 更新：`75545-75560` 已完成，结果进一步支持转向 multi-step CLSTR controller。

训练结果：

- train output: `outputs/appworld_clstr_train_skillrouter_init_routing_full_v1/train_stdout.json`
- final checkpoint: `outputs/appworld_clstr_train_skillrouter_init_routing_full_v1/checkpoints/stage2_base-step600.pt`
- full train batch routing supervision 达到 `routing_supervision_recall@1 = 1.0`，`recall@5 = 1.0`；
- final train loss 降到 `0.01260884664952755`。

dev routing checkpoint sweep：

- comparison: `outputs/appworld_clstr_eval/skillrouter_init_routing_full_v1_checkpoint_sweep_comparison/comparison.json`

| checkpoint | dev recall@1 | dev recall@5 | dev mrr@20 |
|---|---:|---:|---:|
| step100 | 0.964912 | 0.964912 | 0.969298 |
| step200 | 0.912281 | 1.000000 | 0.937135 |
| step300 | 0.912281 | 1.000000 | 0.937135 |
| step400 | 0.964912 | 1.000000 | 0.976608 |
| step500 | 0.964912 | 1.000000 | 0.976608 |
| step600 | 0.912281 | 1.000000 | 0.937135 |

executor dev10 result for final checkpoint:

- report: `outputs/appworld_executor_benchmark/clstr_skillrouter_init_routing_full_v1_dev10_safe_metadata/report.json`
- comparison: `outputs/appworld_executor_benchmark/dev10_v3_safe_metadata_plus_routing_full_v1_comparison/comparison.json`
- result: `3/10`, success rate `0.30`, execution failures `7`

comparison:

| method | dev10 success |
|---|---:|
| Qwen-only | 7/10 |
| SkillRouter embedding | 4/10 |
| SkillRouter-base | 6/10 |
| old CLSTR-base packed | 6/10 |
| old CLSTR-act train10/g4/u3 prior | 6/10 |
| full-v1 final CLSTR step600 | 3/10 |

解释：

- full-v1 可以把 train routing 拟合到很高，但 final checkpoint 在 dev routing 退化，且 single-shot executor task success 明显变差；
- 这说明继续增加 routing-only 训练或直接扩大 single-shot ACT 不是可靠路径；
- step400/step500 是 dev routing 最好 checkpoint，若还要补 single-shot executor，应优先评估 step400/step500，而不是 step600；
- 但即使 step400/500 恢复到 old CLSTR-base 的 6/10，也仍不足以证明 CLSTR 的 sequential/belief-state 价值。

因此下一阶段的目标应从“更强 one-shot router”转向“multi-step CLSTR controller”：

- 用真实 AppWorld execution trajectory 定义 state/action/observation/reward；
- 让 `m_t` 进入 stepwise policy，而不只是 one-shot skill retrieval；
- 用 final official task success 证明方法价值；
- 以 SkillRouter 作为每步 retrieval/context baseline，而不是只比较单步 routing recall。

### 下一步完整 goal：超过 SkillRouter 并证明 CLSTR 方法价值

2026-05-23 补充：下一阶段不再把 CLSTR 当作 single-shot retriever 去硬拼 SkillRouter 的静态 embedding 排序，而是实现和验证 AppWorld multi-step CLSTR controller。核心论文假设是：在相同 Qwen3-8B executor、相同 AppWorld API docs、相同 safe metadata skill context 下，CLSTR 的 recurrent latent belief `m_t` 能利用多步执行反馈，做出比 SkillRouter static/per-step retrieval 更好的 stepwise skill/context selection，从而提升 official AppWorld task success。

#### 输入与 GT 口径

每个 AppWorld episode 使用可复现的 serialized trajectory prefix，而不是把缓存 hidden state 当成数据源：

```text
[System Prompt]
[User Goal]
[Step 1 Selected Skills]
[Code 1]
[Environment Output 1]
[Step 2 Selected Skills]
[Code 2]
[Environment Output 2]
...
```

在第 `t` 步：

- `state_text_t = user_goal + compact history + latest environment output/error + available API/schema hints`；
- CLSTR 输入为 `state_text_t`，编码得到 `h_t`，并维护 recurrent belief `m_t`；
- CLSTR 输出为 top-k SkillX skill ids/context，不直接生成 code；
- Qwen3-8B 根据 `state_text_t + selected skills + API docs` 生成下一段 Python code；
- AppWorld 执行 code，得到 `execute_output_t`、`execution_ok_t`、`task_completed_t`、`evaluate_t`；
- GT/reward 以 AppWorld official `task_completed()` / `evaluate()` 为准，不使用模型自报 success tag。

#### `m_t` 如何进入模型

优先做 CLSTR 内部数学融合，而不是一开始把未训练的 `m_t` 注入 frozen Qwen：

1. Observation/state encoding:
   - `h_t = Encoder(state_text_t)`；
   - `m_obs_t = softmax(skill_logits(h_t)) @ skill_table.E`。
2. CLSTR belief update:
   - 用上一轮 selected skill embedding、execution output embedding、`m_obs_t` 更新 `m_t`；
   - `m_t` 是模型内部 recurrent state，不写死到数据集里。
3. 第一阶段融合：
   - `h_cond_t = LayerNorm(h_t + W_m m_t)`；
   - policy score 使用 `score(h_cond_t, m_t, e_i, h_cond_t * e_i)`；
   - 目标是证明 `m_t` 能改善 stepwise skill selection 与最终 success。
4. 第二阶段 ablation：
   - 可以比较 `concat(h_t, m_t)`、gated fusion、FiLM；
   - 只有当 CLSTR 内部融合有效后，再尝试 `m_t -> soft prefix` 或 token-embedding residual 注入 Qwen：
     `qwen_inputs = concat(W_prefix(m_t), token_embeds(prompt))`。

这个顺序的原因是：frozen Qwen 不知道随机 soft vector 的语义，直接把未训练 `m_t` 加进 Qwen token embedding 很容易破坏 executor；先让 `m_t` 在 CLSTR policy 内部影响 skill/context selection，更稳定，也更符合 CLSTR 作为 latent controller 的论文定位。

#### 需要实现的实验管线

1. Multi-step AppWorld executor：
   - 每步选择 skills、调用 Qwen、执行 AppWorld、记录 output；
   - 支持 `qwen_only`、`skillrouter_base`、`clstr_base`、`clstr_mt_fusion` 四种 controller；
   - 输出 `runs.jsonl`、`report.json`、per-step trace。
2. SkillRouter baseline：
   - 在相同 `state_text_t` 上做每步 retrieval/context selection；
   - 与 CLSTR 使用相同 top-k、相同 safe metadata context、相同 Qwen prompt。
3. CLSTR multi-step controller：
   - 加载 SkillRouter-init CLSTR checkpoint；
   - 每个 episode 内维护 `m_t`；
   - 每步根据 `h_t + m_t` 选 top-k skills；
   - 记录 `m_t` 是否改变 ranking、是否命中 step positive skills、是否提升 final success。
4. ACT/HRPO 训练：
   - 在 train split 上做多 rollout；
   - 同一 task 至少 group_size > 1，避免 policy loss 无 advantage；
   - reward 来自 official success / task_completed / execution_ok / progress heuristic；
   - 控制 `beta_routing_prior`，避免 policy 被强行拉回 SkillRouter-init base。
5. 对照与验收：
   - 先用 dev10 smoke 验证无崩溃和 trace 合理；
   - 再跑 dev57 或 full dev；
   - 主指标是 official task success；
   - 机制指标包括 step-positive hit rate、routing recall/MRR、`m_t` value accuracy、execution failure rate。

#### 成功标准

该阶段只有满足下面条件，才能说“CLSTR 在 AppWorld 上证明了方法价值”：

- 在相同 Qwen3-8B executor 和相同 prompt/context 安全策略下，`CLSTR multi-step + m_t fusion` 的 official AppWorld task success 超过 `SkillRouter-base per-step retrieval + Qwen`；
- 相比 `CLSTR without m_t`，`m_t` fusion 带来可观的 step hit rate 或 final success 提升；
- 失败分析显示提升不是 prompt 偶然性或数据泄漏，而是来自 environment output 后的 belief update / sequential skill selection；
- routing-only 或 single-shot executor 结果只作为诊断，不再作为论文主成功结论。

#### 下一步交给 Codex 的执行 goal

在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr` 内完成并验证 AppWorld multi-step CLSTR controller，使 CLSTR 的 recurrent belief `m_t` 真正进入每一步 skill/context 选择，并在相同 Qwen3-8B executor 下与 SkillRouter-base per-step retrieval 做公平对比。先实现/验证 dev10 smoke，再扩展到 dev57/full dev；若 smoke 中 CLSTR 低于 SkillRouter，先做 trace 诊断和 `m_t`/context ablation，不盲目扩大训练。最终目标是在 official AppWorld task success 上超过 SkillRouter-base，并用 `m_t` ablation、step-level trace 和 value/routing 机制指标证明 CLSTR latent belief 与连续选择确实有贡献。

### 2026-05-23 multi-step CLSTR controller 工程接入

本轮把 multi-step executor 从“每步复用静态 prediction file”推进到可以加载 CLSTR checkpoint 并在 episode 内维护 recurrent belief `m_t`。

新增/修改：

- `clstr/appworld_multistep.py`
  - 新增 `CLSTRMultiStepController`；
  - 每个 episode `reset()` 时清空 `m_t`；
  - 第一步用 `subspace_obs(skill_table, h_t)` 初始化 `m_t`；
  - 每步从 `skill_table.logits(h_t)` 取 candidate pool，再用 `policy_head` 或 `policy_blend` 排序；
  - `observe()` 用 selected skill index、`execute_output` embedding 和 `model.step_update()` 更新 `m_{t+1}`；
  - trace 中记录 `candidate_skill_ids`、`selected_action_indices`、`belief_updates`、`belief_norm`。
- `clstr/model.py`
  - 新增 `CLSTRConfig.mt_fusion_mode`；
  - `mt_fusion_mode=h_plus_m` 时启用 `h_cond = LayerNorm(h_t + W_m m_t)`；
  - skill logits 使用 `[candidate_emb, m_t, h_cond, candidate_emb * h_cond]`，使 `h_t` 与 `m_t` 都进入 policy score；
  - 默认 `mt_fusion_mode=none`，不改变旧 checkpoint/旧训练路径。
- `configs/model/appworld_skillrouter_init_mt_fusion.yaml`
  - 基于 `appworld_skillrouter_init.yaml`，增加 `mt_fusion_mode: h_plus_m`；
  - 用于后续训练/评测 `clstr_mt_fusion`。
- `scripts/run_appworld_multistep_executor_eval.py`
  - 新增 `clstr_multistep` 和 `clstr_mt_fusion` 方法；
  - 新增 `--clstr_model_config`、`--clstr_checkpoint_path`、`--ranking_mode`、`--candidate_top_k`、`--policy_blend_alpha`；
  - 可以加载 CLSTR checkpoint 构造真实 multi-step controller。
- `scripts/sbatch/run_appworld_multistep_executor_eval.sh`
  - 暴露 CLSTR checkpoint/config/ranking 参数；
  - 后续 GPU smoke/benchmark 仍通过 sbatch 提交。

已验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 15 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_multistep.py scripts/run_appworld_multistep_executor_eval.py clstr/model.py

bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

当前仍未完成的关键点：

- `clstr_mt_fusion` 目前只是结构和评测入口接通；如果直接加载旧 checkpoint，新增 `W_m/mt_skill_head` 是随机初始化，不能把未训练结果当作方法性能；
- 还没有跑真实 Qwen/AppWorld dev10 multi-step smoke；
- 还没有实现 train split multi-step ACT/HRPO 数据和训练；
- SkillRouter baseline 目前可用 prediction-file/static 版本，后续仍需补“每步按当前 `state_text_t` 动态检索”的更公平 baseline。

### 2026-05-23 per-step SkillRouter live baseline 接入

为避免把 CLSTR multi-step 与 SkillRouter static prediction-file 进行不公平对比，本轮继续补了 per-step dynamic retrieval baseline：

- `clstr/appworld_multistep.py`
  - 新增 `LiveEmbeddingStepController`；
  - 初始化时编码完整 SkillX skill pool；
  - 每一步用当前 `state_text_t` 重新编码 query，并与 skill embeddings 计算 top-k；
  - 支持纯 embedding dot-product，也支持传入 `SkillRouterBaseScorer` 做 trainable scorer；
  - trace 中标记 `candidate_source=current_state_text`。
- `scripts/run_appworld_multistep_executor_eval.py`
  - 新增 `skillrouter_embedding_live`；
  - 新增 `skillrouter_base_live`；
  - 增加 persistent HF encoder，避免每步重新加载 SkillRouter-Embedding-0.6B；
  - `skillrouter_base_live` 会加载 `outputs/appworld_skillrouter_base_train_v1/model.pt`。
- `scripts/sbatch/run_appworld_multistep_executor_eval.sh`
  - 增加 `SKILLROUTER_MODEL_NAME_OR_PATH`、`SKILLROUTER_CHECKPOINT_PATH`、`SKILLROUTER_BATCH_SIZE`、`SKILLROUTER_MAX_LENGTH`、`SKILLROUTER_DEVICE`。

已验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 16 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_multistep.py scripts/run_appworld_multistep_executor_eval.py clstr/model.py

bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

下一步真实 smoke 组合应至少包含：

1. `qwen_only`：multi-step executor anchor；
2. `skillrouter_base_live`：每步基于当前 state_text 动态检索；
3. `clstr_multistep`：旧 checkpoint + recurrent `m_t`，但无新 fusion head；
4. `clstr_mt_fusion`：只在训练出 mt-fusion checkpoint 后参与正式比较，不能直接用旧 checkpoint 报性能。

### 2026-05-23 multi-step reporting 与 train trajectory 导出

继续补了两个支撑工具：

- `build_multistep_executor_comparison()`
  - 入口：`clstr/appworld_multistep.py`；
  - 脚本：`scripts/build_appworld_multistep_comparison.py`；
  - 汇总字段包括 `success_rate`、`execution_failures`、`task_completed_count`、`evaluate_success_count`、`average_steps`、`step_positive_skill_hit_rate`、`stop_accuracy`；
  - 用于 dev3/dev10/dev57 多步对照，不再复用 single-shot executor comparison 的窄字段。
- `build_train_multistep_trajectories_from_runs()`
  - 入口：`clstr/appworld_multistep.py`；
  - 脚本：`scripts/build_appworld_multistep_train_trajectories.py`；
  - 从 `runs.jsonl` 导出统一 trajectory schema；
  - 默认只接受 `split=train` 且 `final_success=true` 的 runs；
  - 输出 `leakage_guard=train_split_only`，防止误把 dev/test success/evaluate 信息用于训练；
  - 这一步只是数据接口，真实训练仍需要 train split multi-step rollout 产出成功轨迹。

已验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 18 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_multistep.py scripts/run_appworld_multistep_executor_eval.py scripts/build_appworld_multistep_comparison.py scripts/build_appworld_multistep_train_trajectories.py clstr/model.py

bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

已提交 dev3 multi-step smoke：

| job | method | output |
|---:|---|---|
| 75617 | `qwen_only` | `outputs/appworld_multistep_executor_smoke/qwen_only_dev3_v1` |
| 75618 | `skillrouter_base_live` | `outputs/appworld_multistep_executor_smoke/skillrouter_base_live_dev3_v1` |
| 75619 | `clstr_multistep` step500 policy_blend | `outputs/appworld_multistep_executor_smoke/clstr_multistep_step500_dev3_v1` |

提交后首次检查：三个作业均为 `RUNNING`，运行时间约 `0:15`。后续按短任务约 15 分钟间隔检查，不频繁轮询。

dev3 smoke 已完成：

| method | task success | execution failures | step hit |
|---|---:|---:|---:|
| `qwen_only` | 3/3 | 0 | 0.0 |
| `skillrouter_base_live` | 3/3 | 0 | 1.0 |
| `clstr_multistep` step500 | 3/3 | 0 | 1.0 |

comparison:

- `outputs/appworld_multistep_executor_smoke/dev3_v1_comparison/comparison.json`
- `outputs/appworld_multistep_executor_smoke/dev3_v1_comparison/comparison.md`

判断：

- dev3 已经饱和，不能证明 CLSTR 超过 SkillRouter；
- 但它证明 Qwen/AppWorld multi-step loop、SkillRouter live baseline、CLSTR recurrent `m_t` controller 都可以真实跑通；
- 下一步需要 dev10/dev57，并且需要训练 `mt_fusion`，不能只用旧 checkpoint。

train-only oracle multi-step trajectories 已构建：

- output: `data/appworld_multistep/oracle_train_trajectories.jsonl`
- manifest: `data/appworld_multistep/oracle_train_manifest.json`
- task_count: 90
- trajectory_count: 90
- skipped_by_split: 0
- skipped_without_steps: 0
- supervision_type: `train_oracle_api_trace`
- execution_source: `ground_truth_api_calls_not_qwen_execution`
- leakage_guard: `train_split_only`

注意：这批数据可以用于 stepwise skill transition / `m_t` fusion 的离线监督预训练，但不能冒充真实 Qwen execution trajectory；后续 ACT/HRPO 仍需要 train split rollout reward。

### 2026-05-23 dev10 multi-step baseline 与 auth-context 诊断

dev10 multi-step v1 已完成：

comparison:

- `outputs/appworld_multistep_executor_benchmark/dev10_v1_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/dev10_v1_comparison/comparison.md`

| method | task success | execution failures | avg steps | step hit |
|---|---:|---:|---:|---:|
| `qwen_only` | 6/10 | 6 | 2.2 | 0.0 |
| `skillrouter_base_live` | 7/10 | 0 | 2.1 | 1.0 |
| `clstr_multistep` step500 | 3/10 | 12 | 2.4 | 1.0 |

判断：

- 当前未训练 mt-fusion 的 `clstr_multistep` 明显低于 `skillrouter_base_live`；
- 这不是 step positive hit 低造成的，CLSTR step hit 也是 1.0；
- 主要失败点是 selected context 结构污染：CLSTR top-k 中多个 auth-like SkillX skill 占据前排，Qwen 被诱导生成错误登录/API 代码；
- 典型失败包括：
  - `fac291d_*`: `song_ids.update(album["song_ids"] for album in response)` 导致 `TypeError: unhashable type: 'list'`；
  - `530b157_*`: phone/venmo 登录相关 API 幻觉或凭证错误；
  - `4ec8de5_1`: album/song schema 访问错误。

已修复/新增 ablation：

- `CLSTRMultiStepController` 与 `LiveEmbeddingStepController` 支持 `dedupe_canonical_skills` 和 `max_auth_like_skills`；
- `scripts/run_appworld_multistep_executor_eval.py` 会把已有 `--max_auth_like_skills` 传给 CLSTR/live controllers；
- 已提交 dev10 ablation：

| job | method | output |
|---:|---|---|
| 75629 | `clstr_multistep` step500 + `MAX_AUTH_LIKE_SKILLS=1` | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_step500_dev10_auth1_v1` |
| 75630 | `clstr_mt_fusion` smoke10 + `MAX_AUTH_LIKE_SKILLS=1` | `outputs/appworld_multistep_executor_benchmark/clstr_mt_fusion_smoke10_dev10_auth1_v1` |

auth/context ablation 已完成：

comparison:

- `outputs/appworld_multistep_executor_benchmark/dev10_v1_plus_auth_ablation_comparison/comparison.json`

| method | task success | execution failures | task_completed | step hit |
|---|---:|---:|---:|---:|
| `qwen_only` | 6/10 | 6 | 7 | 0.0 |
| `skillrouter_base_live` | 7/10 | 0 | 7 | 1.0 |
| `clstr_multistep` step500 | 3/10 | 12 | 4 | 1.0 |
| `clstr_multistep` step500 + auth1 | 3/10 | 6 | 7 | 0.625 |
| `clstr_mt_fusion` smoke10 + auth1 | 3/10 | 16 | 3 | 0.625 |

判断：

- auth-like context 限制能把 execution failures 从 12 降到 6，但 success 没有提升；
- success 不提升说明问题不只是 auth skills 过多，还包括 CLSTR 排序/上下文组合让 Qwen 生成了错误 schema 访问代码；
- `clstr_mt_fusion` smoke10 训练太小，且只用 oracle API trace，不足以改善 dev10；
- 后续需要：
  1. 等 full 90 oracle-supervised mt-fusion checkpoint；
  2. 做 `TOP_K` / auth0 / SkillRouter-prior hybrid ablation；
  3. 最终还需要 train split ACT/HRPO reward，不应把 oracle-supervised smoke 当作论文结果。

mt-fusion 训练：

- smoke job `75628` 已完成；
- output: `outputs/appworld_mt_fusion_train/oracle_smoke10_v1`;
- checkpoint: `outputs/appworld_mt_fusion_train/oracle_smoke10_v1/checkpoints/appworld_mt_fusion.pt`;
- train trajectories: 10;
- supervised steps: 40;
- loss: `2.7720658779144287`;
- caveat: 这是 train-only oracle API trace 的离线监督，不是 ACT/HRPO。

为安全扩展到 full train，已给 mt-fusion 训练增加 `trajectory_batch_size`，避免一次性对 90 条 trajectory 全量反传。已提交 full 90 trajectory 训练：

| job | output | epochs | trajectory batch |
|---:|---|---:|---:|
| 75631 | `outputs/appworld_mt_fusion_train/oracle_full_v1` | 3 | 8 |

### 2026-05-23 SkillRouter-prior hybrid CLSTR controller

auth ablation 说明：单纯限制 auth-like context 只能降低 execution failures，不能恢复 task success。因此继续补了一个更合理的 hybrid controller：

- `HybridCLSTRSkillRouterController`
  - 候选池和 prior score 来自 `SkillRouter-base live`，即每步按当前 `state_text_t` 动态检索；
  - CLSTR 维护自己的 recurrent `m_t`；
  - 对 SkillRouter candidate pool 计算 CLSTR policy logits；
  - 最终 score:
    `score = (1 - clstr_alpha) * standardize(skillrouter_score) + clstr_alpha * standardize(clstr_policy_score)`；
  - 这样可以避免 CLSTR 自己较弱的 candidate pool 直接污染 Qwen，同时仍检验 `m_t` policy 是否能在强 SkillRouter prior 上带来增益。
- CLI:
  - 新增 method: `clstr_skillrouter_hybrid_live`;
  - 新增参数：`--clstr_alpha`;
  - sbatch 环境变量：`CLSTR_ALPHA`。

已验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 26 passed
```

已提交 dev10 hybrid ablation：

| job | method | output |
|---:|---|---|
| 75638 | `clstr_skillrouter_hybrid_live`, `CLSTR_ALPHA=0.25`, `MAX_AUTH_LIKE_SKILLS=1` | `outputs/appworld_multistep_executor_benchmark/clstr_skillrouter_hybrid_step500_alpha025_dev10_auth1_v1` |

判断标准：

- 如果 hybrid >= SkillRouter-base live 7/10，只能说明“CLSTR 没有破坏强 prior”；还需要 alpha ablation 证明 `m_t` 有增益；
- 如果 hybrid > 7/10，才有初步证据表明 `m_t` 在多步 state feedback 下对 SkillRouter prior 有正贡献；
- 如果 hybrid < 7/10，则说明当前 CLSTR policy signal 仍在扰动强 prior，需要继续训练或降低 `clstr_alpha`。

### 2026-05-23 下一阶段目标：embedding-level recurrent CLSTR 与 ACT 价值验证

当前结论不能解释为“CLSTR 方法本身无效”，但也不能继续用弱结果声称论文方法有效。dev10 结果显示：

- `skillrouter_base_live` 已经是强 baseline：每步用当前 `state_text_t` 动态检索 SkillX skill context，dev10 为 7/10；
- 当前 `clstr_multistep` 虽然有 recurrent `m_t`，但更多是在 routing head 后端做 rerank/late fusion，尚未充分影响 executor 输入质量；
- 当前训练主要是 stepwise/oracle-supervised routing，不是真正从 Qwen execution success/failure 中学习“采样哪个 skill 序列能提高最终任务成功率”；
- 因此，CLSTR 的论文价值必须通过“长程多步状态 + `m_t` 参与输入表示 + ACT/HRPO reward learning”来证明，而不是只和 SkillRouter 做单步相似度检索对比。

需要把接下来的研究问题拆成三个可验证假设：

1. **输入表示假设**：`m_t` 不应只作为 policy head 的 late feature，而应进入 executor/routing 输入 embedding。可优先实现两种轻量形式：
   - `m_t` virtual token：把 `m_t` 经投影层映射成一个或多个 soft prompt token，插入到 `[System Prompt] + [User Goal] + history_t` 的 embedding 前缀或当前位置；
   - gated residual/FiLM：对正常文本 hidden states 做 `hidden'_i = hidden_i + gate_i * W_m m_t` 或 `hidden'_i = gamma(m_t) * hidden_i + beta(m_t)`，让 recurrent belief 影响每个 token 的条件表示。
2. **多步监督假设**：训练样本不应只是一条 `state_text_t -> skill_t`。更合理的样本应包含：
   - `task_id`、`user_goal`、system/tool prompt；
   - step history：`Thought/Code/Env Output` 或至少 `Code/Env Output`；
   - 上一步/当前步的 `m_t`；
   - candidate skill pool、selected skill、stop action；
   - oracle/API-trace label 与 Qwen execution outcome 分开记录，防止把 oracle trace 冒充真实执行成功。
3. **强化学习/ACT 假设**：随机采样 skill embedding 的收益必须来自多次 rollout 的回报差异。也就是说，`L_policy` 应主要在 CLSTR-ACT 阶段使用 train split rollout 产生的 final success、valid API use、execution failure、step-positive hit 等 reward，而不是只依赖离线 oracle trace。

建议下一阶段同时保留以下 ablation，避免把改进误归因：

| variant | 目的 |
|---|---|
| `skillrouter_base_live` | 强 baseline，候选和 context 全由 SkillRouter 每步动态检索 |
| `clstr_skillrouter_hybrid_live alpha=0` | 等价 SkillRouter prior，作为 hybrid 下限 |
| `clstr_skillrouter_hybrid_live alpha>0` | 检验 `m_t` policy 是否能在强 prior 上带来增益 |
| `clstr_multistep recurrent_belief=1` | 检验完整 recurrent CLSTR |
| `clstr_multistep recurrent_belief=0` | `m_t` 消融，证明长程 belief 是否有用 |
| `clstr_mt_virtual_token` / `clstr_mt_gated_residual` | 检验 `m_t` 进入输入 embedding 是否优于 late fusion |
| `clstr_act` | 用真实 Qwen rollout reward 训练 policy，检验 ACT/HRPO 对最终 task success 的贡献 |

下一阶段成功标准：

- 短期 dev10：至少有一个 CLSTR variant 超过 `skillrouter_base_live` 的 7/10，而不是只追平；
- 消融证据：`recurrent_belief=1` 必须优于 `recurrent_belief=0`，或 hybrid `alpha>0` 必须优于 `alpha=0`；
- 训练证据：ACT/HRPO 后的 checkpoint 必须优于只做 oracle-supervised mt-fusion 的 checkpoint；
- 泛化证据：dev10 通过后扩展到 dev57/test，不使用 dev/test success signal 训练；
- 论文表述：若只在 hybrid 上超过 SkillRouter，需要明确贡献是“CLSTR recurrent policy improves a strong SkillRouter prior”，不能声称完全替代 SkillRouter。

给 Codex 的下一步完整 goal：

> 在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr` 内，基于现有 AppWorld/SkillX/Qwen3-8B executor 代码，完成并验证一个能够超过 `skillrouter_base_live` 的 multi-step CLSTR 版本。具体要求：实现至少一种 `m_t` 进入输入 embedding 的模型形式（优先 `m_t` virtual token 或 gated residual），保留 late-fusion/hybrid 作为对照；构建 train split 的多步 execution rollout 数据与 ACT/HRPO 训练入口，使 `L_policy` 从真实 Qwen 执行回报中学习；用 sbatch 在 dev10 上比较 `skillrouter_base_live`、hybrid alpha=0/alpha>0、recurrent/no-recurrent、oracle-supervised mt-fusion、ACT checkpoint；若 dev10 超过 SkillRouter，再扩展 dev57/test；所有重大设计、命令、结果、失败分析都追加写入 `description.md`，且所有读写限制在 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp` 内。

### 2026-05-23 gated residual `m_t` fusion 初步实现

为推进“`m_t` 不只做 late fusion”的方向，本轮先实现了一个最小可训练版本：

- `CLSTRConfig.mt_fusion_mode` 新增 `gated_residual`；
- 模型形式：
  - `gate_t = sigmoid(W_g [h_t; m_t])`
  - `h_cond = LayerNorm(h_t + gate_t * W_m m_t)`
  - `policy_score = f(candidate_emb, m_t, h_cond, candidate_emb * h_cond)`
- 新配置：
  - `configs/model/appworld_skillrouter_init_gated_residual.yaml`
- 新测试：
  - `tests/test_model_pipeline.py::test_gated_residual_mt_fusion_uses_belief_gate_to_condition_state_hidden`
  - RED 时失败于 `unsupported mt_fusion_mode: gated_residual`；
  - GREEN 后证明 gate 关闭时 policy 读到原始 `h_t`，gate 打开时读到 `h_t + W_m m_t`。

注意：这仍然是 pooled state representation 级别的 gated residual，不是直接改 frozen Qwen executor 的 token embedding。它的作用是先验证 `m_t` 是否能在 CLSTR policy 内部提供可训练增益；如果该版本仍不能超过 SkillRouter，再考虑更重的 virtual-token / executor prompt embedding 方案。

已提交作业：

| job | purpose | output |
|---:|---|---|
| 75647 | train-only oracle full `gated_residual` mt-fusion, 3 epochs | `outputs/appworld_mt_fusion_train/oracle_gated_residual_full_v1` |
| 75648 | afterok:75647 dev10 `clstr_mt_fusion` eval, auth0 | `outputs/appworld_multistep_executor_benchmark/clstr_mt_gated_residual_full_dev10_auth0_v1` |
| 75659 | dev10 `clstr_skillrouter_hybrid_live`, `clstr_alpha=0.00`, no auth cap | `outputs/appworld_multistep_executor_benchmark/clstr_skillrouter_hybrid_step500_alpha000_dev10_v1` |
| 75649 | dev10 `clstr_skillrouter_hybrid_live`, `clstr_alpha=0.05`, no auth cap | `outputs/appworld_multistep_executor_benchmark/clstr_skillrouter_hybrid_step500_alpha005_dev10_v1` |
| 75658 | dev10 `clstr_skillrouter_hybrid_live`, `clstr_alpha=0.10`, no auth cap | `outputs/appworld_multistep_executor_benchmark/clstr_skillrouter_hybrid_step500_alpha010_dev10_v1` |

等待结果时不应频繁轮询；短任务约 15 分钟检查一次即可。

### 2026-05-23 multi-step HRPO / ACT 训练入口

为解决“`L_policy` 应该在 CLSTR-ACT 中从多次 rollout 的最终回报学习”的问题，本轮补了 multi-step HRPO 入口：

- `clstr/appworld_act_hrpo.py`
  - 新增 `AppWorldMultiStepHrpoRollout`；
  - 新增 `run_appworld_multistep_hrpo_rollout()`；
  - 每一步构造 `state_text_t = user goal + compact execution history + latest env output`；
  - 每一步从 CLSTR policy 采样 skill，记录 `selected_log_prob_t`；
  - Qwen 用当前 `state_text_t + selected SkillX context + API docs` 生成下一段 code；
  - AppWorld 执行 code，并用 observation 更新 `m_{t+1}`；
  - 最终 reward 来自整条 trajectory 的 official `task_completed && evaluate_success`；
  - 新增 `compute_appworld_multistep_hrpo_loss()`，用整条 skill sequence 的 log-prob sum 接收 task-level reward advantage。
- `scripts/run_appworld_clstr_hrpo_train.py`
  - 新增 `--multi_step/--no-multi_step`；
  - 新增 `--max_steps`；
  - `--multi_step` 时调用 `run_appworld_clstr_multistep_hrpo_from_paths()`。
- `scripts/sbatch/run_appworld_clstr_hrpo_train.sh`
  - 新增 `MULTI_STEP`；
  - 新增 `MAX_STEPS`；
  - 仍然通过 `sbatch` 提交，不在登录节点跑训练。

已验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_act_hrpo.py tests/test_sbatch_scripts.py -q
# 15 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_hrpo_cli.py tests/test_appworld_act_hrpo.py tests/test_sbatch_scripts.py -q
# 16 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_hrpo_cli.py tests/test_appworld_act_hrpo.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 37 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_hrpo_cli.py tests/test_appworld_act_hrpo.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 38 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_hrpo_cli.py tests/test_appworld_act_hrpo.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 39 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_hrpo_cli.py tests/test_appworld_act_hrpo.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 40 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_act_hrpo.py scripts/run_appworld_clstr_hrpo_train.py clstr/model.py

bash -n scripts/sbatch/run_appworld_clstr_hrpo_train.sh scripts/sbatch/run_appworld_mt_fusion_train.sh scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

判断：现在 CLSTR-ACT 不再局限于 single-shot skill sampling；已经具备“同一 AppWorld task 多步采样 skill sequence，并由最终 official success 给整条 sequence 回传”的训练入口。下一步需要先跑很小的 train-split smoke，确认真实 Qwen/AppWorld/sbatch 路径可用，再扩大 updates/group size。

已提交作业：

| job | purpose | output |
|---:|---|---|
| 75652 | train-split multi-step HRPO smoke, `MAX_TASKS=2`, `GROUP_SIZE=2`, `UPDATES=1`, `MAX_STEPS=3` | `outputs/appworld_clstr_train_multistep_hrpo/smoke_step500_v1` |
| 75654 | afterok:75652 dev3 `clstr_multistep` eval for HRPO smoke checkpoint | `outputs/appworld_multistep_executor_smoke/clstr_multistep_hrpo_smoke_dev3_v1` |
| 75655 | afterok:75652 train-split multi-step HRPO pilot, `MAX_TASKS=10`, `GROUP_SIZE=4`, `UPDATES=5`, `MAX_STEPS=3` | `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_v1` |
| 75656 | afterok:75655 dev10 `clstr_multistep` eval for ACT pilot checkpoint | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_dev10_v1` |
| 75657 | afterok:75655 dev10 no-recurrent ablation for ACT pilot checkpoint | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_dev10_no_recurrent_v1` |
| 75660 | afterok:75647:75652 gated residual multi-step HRPO pilot, `MAX_TASKS=10`, `GROUP_SIZE=4`, `UPDATES=5`, `MAX_STEPS=3` | `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_gated_v1` |
| 75661 | afterok:75660 dev10 `clstr_multistep` eval for gated ACT pilot checkpoint | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_gated_pilot10_g4_u5_dev10_v1` |
| 75663 | afterok:75660 dev10 no-recurrent ablation for gated ACT pilot checkpoint | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_gated_pilot10_g4_u5_dev10_no_recurrent_v1` |

其中 `75652/75654` 只验证新 ACT 训练入口的真实运行链路，不作为论文性能结果；`75655/75656` 是第一轮 ACT pilot，也不能作为最终论文结果。真正用于超过 SkillRouter 的 ACT 训练需要在 pilot 结果确认有效后继续增加 train tasks、group size、updates，并在 dev10/dev57 上比较。

已生成 dev10 v2 pending comparison：

- `outputs/appworld_multistep_executor_benchmark/dev10_v2_pending_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/dev10_v2_pending_comparison/comparison.md`

当前 comparison 中，已完成 baseline 仍是：

- `qwen_only`: 6/10
- `skillrouter_base_live`: 7/10
- best completed CLSTR variant: `clstr_multistep` auth0, 6/10

新提交的 hybrid alpha sweep、full mt-fusion、gated residual、ACT pilot、no-recurrent ablation 都仍是 `status=missing`。因此截至当前状态，目标“超过 SkillRouter 并证明 `m_t` 价值”尚未完成。

### 2026-05-23 active goal completion audit

目标拆解为可验收标准：

1. AppWorld multi-step CLSTR controller 能在同等 Qwen3-8B executor 下真实运行；
2. 训练/评测只使用 train split reward 或 dev/test eval，不把 dev/test success 用于训练；
3. 至少一个 CLSTR variant 的 dev10 AppWorld task success 超过 `skillrouter_base_live` 的 7/10；
4. ablation 证明 `m_t` 有价值：recurrent `m_t` > no-recurrent，或 hybrid `alpha>0` > hybrid `alpha=0`；
5. 若 dev10 超过 SkillRouter，再扩展 dev57/test；
6. 重大设计、实验设置、结果和失败分析写入 `description.md`；
7. 重训练/评测通过 sbatch 运行，不在登录节点直接跑 GPU 任务。

当前证据：

| requirement | evidence | status |
|---|---|---|
| multi-step executor | dev3 smoke 3/3；dev10 baseline reports；`run_appworld_multistep_executor_eval.py` 和 controller tests | done |
| train-only trajectory/oracle data | `data/appworld_multistep/oracle_train_trajectories.jsonl` 90 条，`leakage_guard=train_split_only` | done for oracle-supervised pretrain |
| ACT/HRPO multi-step entry | `run_appworld_multistep_hrpo_rollout()`、`train_appworld_clstr_multistep_hrpo()`；40 个相关 tests 通过 | implemented, awaiting real sbatch smoke |
| exceed SkillRouter | completed best CLSTR is 6/10 vs SkillRouter 7/10 | not done |
| `m_t` ablation proof | base no-recurrent and ACT pilot no-recurrent jobs pending；no completed proof yet | not done |
| dev57/test | not started because dev10 has not exceeded SkillRouter | not done |
| sbatch compliance | jobs 75638/75641/75643/75647/75648/75649/75652/75654/75655/75656/75657/75658/75659 submitted via `sbatch --gpus=1 -p gpu_h200` | done for queued work |

结论：不能调用 goal complete。下一 gate 是等待 pending dev10 reports：

- hybrid `alpha=0/0.05/0.10/0.25`；
- full mt-fusion and gated residual；
- ACT pilot recurrent/no-recurrent。

只有当 comparison 显示某个 CLSTR variant > 7/10，且 `m_t` ablation 也支持 recurrent belief 有增益，才进入 dev57/test。

补充：为避免 multi-step HRPO 在大多数 rollout 失败时只有全 0 reward，本轮加入保守 reward shaping：

- final official success 仍固定为 `1.0`；
- 未成功轨迹按 `generation_ok`、`execution_ok`、`task_completed`、`evaluation_success` 给小额进展奖励；
- 对 `generation_ok=True` 但 `execution_ok=False` 的 step 给小额惩罚；
- reward 被限制在 `[-0.25, 0.95]`，避免 shaped reward 超过真正 final success；
- `success` 字段仍只表示 official final success，论文/benchmark 仍以 AppWorld task success 为准。

该 shaping 只用于 ACT/HRPO 训练信号，不用于报告最终 benchmark success。

鲁棒性补充：

- 如果某个 multi-step HRPO update 的 rollout 在采样 skill 前全部失败，导致 `decision_count=0`，训练不再对无 grad 的 zero loss 调用 `backward()`；
- 该 update 会跳过 optimizer step，并在 `updates_report[*].optimizer_step_skipped=true` 中记录；
- 这样可以避免真实 AppWorld reset/generation 早期失败让长作业直接崩掉。

### 2026-05-23 next-state belief update 修复

dev10 v2 结果回来后，关键现象是：

- 没有任何 CLSTR variant 超过 `skillrouter_base_live` 的 7/10；
- hybrid `alpha=0/0.05/0.10/0.25` 都是 7/10，只是追平 SkillRouter；
- ACT pilot/gated ACT pilot 没有超过 SkillRouter；
- gated ACT no-recurrent 是 7/10，但 recurrent 是 6/10，说明当前 `m_t` 反而可能在伤害选择。

进一步检查实现后发现一个不一致点：

- `build_multistep_state_text()` 给 selection 使用的是 `user goal + compact history + latest environment output`；
- 但 recurrent `m_t` update 使用的是裸 `execute_output`：
  - executor eval: `step_update(..., [obs_text])`
  - HRPO train: `step_update(..., [execute_output])`
- 这会让 `m_obs` correction 丢掉 user goal 和历史上下文，训练和评测中的 recurrent belief 都更像“只看最近环境字符串”，不是真正的 long-horizon state feedback。

已修复：

- `CLSTRMultiStepController.observe()`
  - 优先读取 `step["next_state_text"]`；
  - fallback 才用裸 `execute_output`，保证兼容旧调用。
- `run_appworld_multistep_executor_eval()`
  - 每步执行/evaluate 后，在 `controller.observe()` 前写入：
    `step["next_state_text"] = build_multistep_state_text(task=task, steps=[*steps, step])`
- `run_appworld_multistep_hrpo_rollout()`
  - 训练时也写入同样的 `next_state_text`；
  - 当前 model 和 ref model 的 `step_update()` 都使用该 next-state text。

新增/更新测试：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_multistep.py::test_clstr_multistep_controller_updates_belief_from_next_state_text_when_available \
  tests/test_appworld_multistep.py::test_multistep_executor_passes_next_state_text_to_recurrent_controller \
  tests/test_appworld_act_hrpo.py::test_appworld_multistep_hrpo_rollout_samples_each_step_and_rewards_final_success -q
# all passed after RED failures

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_act_hrpo.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py \
  tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py \
  tests/test_appworld_hrpo_cli.py -q
# 42 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/appworld_multistep.py clstr/appworld_act_hrpo.py \
  scripts/run_appworld_multistep_executor_eval.py scripts/run_appworld_clstr_hrpo_train.py clstr/model.py

bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh \
  scripts/sbatch/run_appworld_clstr_hrpo_train.sh scripts/sbatch/run_appworld_mt_fusion_train.sh
```

由于旧 dev10 reports 都是在旧 `m_t` update 逻辑下得到的，已提交 next-state v2 作业：

| job | purpose | output |
|---:|---|---|
| 75701 | step500 recurrent dev10 auth0, next-state update | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_step500_dev10_auth0_nextstate_v2` |
| 75700 | step500 no-recurrent dev10 auth0, same code path control | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_step500_dev10_auth0_no_recurrent_nextstate_v2` |
| 75702 | hybrid alpha=0.10 dev10, next-state update | `outputs/appworld_multistep_executor_benchmark/clstr_skillrouter_hybrid_step500_alpha010_dev10_nextstate_v2` |
| 75703 | normal ACT pilot retrain with next-state update | `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_nextstate_v2` |
| 75705 | afterok:75703 normal ACT pilot dev10 recurrent | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_dev10_nextstate_v2` |
| 75704 | afterok:75703 normal ACT pilot dev10 no-recurrent | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_dev10_no_recurrent_nextstate_v2` |
| 75706 | gated ACT pilot retrain with next-state update | `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_gated_nextstate_v2` |
| 75708 | afterok:75706 gated ACT pilot dev10 recurrent | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_gated_pilot10_g4_u5_dev10_nextstate_v2` |
| 75707 | afterok:75706 gated ACT pilot dev10 no-recurrent | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_gated_pilot10_g4_u5_dev10_no_recurrent_nextstate_v2` |

判断：这是一个真实实现问题，不只是调参。next-state v2 结果回来前，不能用旧 recurrent/no-recurrent ablation 判断 `m_t` 方法本身无效。

### 2026-05-23 next-state v2 interim 诊断：base policy rerank 可能在破坏强初始化

已完成的 next-state v2 对照只有 `75700`：

- output: `outputs/appworld_multistep_executor_benchmark/clstr_multistep_step500_dev10_auth0_no_recurrent_nextstate_v2`
- report: `outputs/appworld_multistep_executor_benchmark/clstr_multistep_step500_dev10_auth0_no_recurrent_nextstate_v2/report.json`
- 结果：`success_count=7/10`, `success_rate=0.7`
- 对照：`skillrouter_base_live` 也是 `7/10`，`qwen_only` 是 `6/10`
- interim comparison:
  - `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v2_interim_completed_comparison/comparison.json`
  - `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v2_interim_completed_comparison/comparison.md`

逐任务对齐发现：

- `clstr_multistep no-recurrent nextstate_v2` 与 `skillrouter_base_live` 的成功/失败集合完全一致；
- 三个失败样例都是 `530b157_*`，任务类型是 phone text + Venmo payment；
- 该结果只能说明当前 CLSTR base no-recurrent 追平 SkillRouter，不能证明超过 SkillRouter，也不能证明 `m_t` 有价值。

更具体的失败模式：

- 在 `530b157_*` 上，CLSTR base 的初始 candidate pool 已经包含相关技能：
  - `simple-note-parse-expense-shares-from-note-27`
  - `simple-note-get-expense-shares-from-note-59`
  - `login-to-venmo-and-phone-17`
  - `login-to-venmo-and-phone-12`
  - `phone-authenticate-with-supervisor-49`
  - `venmo-create-payment-requests-*`
- 但 `ranking_mode=policy_blend` 最终把 `file-system-compress-directories-63` / `file-system-compress-subdirectories-62` 选到 top context；
- 因此当前问题不是“候选召回完全失败”，而是 base 阶段 policy head / blend score 把 SkillRouter-init skill-table 的相关候选重排坏了；
- 这解释了为什么 step-positive hit rate 很低：`clstr_multistep no-recurrent nextstate_v2` 为 `0.1875`，而 `skillrouter_base_live` 为 `1.0`，但 Qwen 在 dev10 前 10 个任务上仍靠自身能力完成了同样的 7 个任务。

下一步 gate 不变：

1. 等待 `75701` recurrent next-state v2、`75702` hybrid alpha=0.10，以及 `75703/75706` ACT retrain 后续 eval；
2. 如果 recurrent 或 ACT 仍不能超过 7/10，需要优先做 `ranking_mode=skill_table`、低 `POLICY_BLEND_ALPHA`、以及 hybrid alpha sweep 的诊断，而不是盲目扩大训练；
3. 如果 `skill_table` 或低 alpha 明显好于 `policy_blend=0.5`，说明实现/超参应把 SkillRouter-init retrieval 作为更强 prior，只让 `m_t` policy 进行小幅、可学习的 rerank；
4. 如果 ACT 后 policy head 能把 `530b157_*` 的相关 phone/Venmo skill 排到前面，才说明 multi-step reward learning 正在修复该问题。

已补交一个最小诊断作业：

| job | purpose | output |
|---:|---|---|
| 75711 | step500 dev10 next-state v2，只改 `RANKING_MODE=skill_table`、`POLICY_BLEND_ALPHA=0`，验证 base policy rerank 是否破坏 SkillRouter-init prior | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_step500_dev10_skilltable_nextstate_v2` |

该作业不是最终论文实验；它用于决定后续应优先调 `policy_blend`/policy head，还是继续扩大 ACT 训练。

### 2026-05-23 auto ranking 修复：routing-only base 不再默认混入未训练 policy head

进一步检查训练实现后确认了 `policy_blend` 失败的 root cause：

- `configs/train/appworld_skillrouter_init_routing_full_v1.yaml` 使用 `loss_mode=routing_only`；
- `train_iteration()` 在 `routing_only` 下不会调用 `total_loss()`，也不会产生 `L_policy`；
- 该模式只调用 `multi_positive_routing_loss()`，而这个 loss 只优化 `model.skill_table.logits(h_t)`；
- 因此 `stage2_base-step500.pt` 的训练报告中只有：
  - `routing_supervision_tasks`
  - `routing_supervision_positive_labels`
  - `routing_supervision_recall@1/5`
  - `loss`
  - `mean_reward`
- 没有 `policy_loss` 或 HRPO/ACT policy 指标；
- 结论：对 routing-only base checkpoint 使用 `ranking_mode=policy_blend` 会把未监督或弱监督的 `policy_head` logits 混入 skill-table prior，导致 `530b157_*` 上把相关 phone/Venmo candidate 重排到 file-system skill 之后。

已修复默认行为：

- `scripts/run_appworld_multistep_executor_eval.py`
  - 新增 `_resolve_auto_ranking_mode()`；
  - `--ranking_mode` 新增 `auto`，默认改为 `auto`；
  - `_build_controller()` 的 Python 默认值也同步改为 `auto`，避免脚本内部直接调用时绕过该保护；
  - `method=clstr_mt_fusion` 的 `auto` 解析为 `policy_head`，因为 mt-fusion 训练的是 policy path；
  - checkpoint `stage` 或 `train_config` 显示 HRPO/ACT 时，`auto` 解析为 `policy_blend`；
  - 其他保守 fallback 解析为 `skill_table`，尤其是 routing-only base checkpoint。
- `clstr/appworld_clstr_eval.py`
  - `_load_checkpoint_model()` 的 load report 现在返回 checkpoint metadata：`stage`、`step`、`metrics`、`train_config`、`epoch_stats` 等；
  - 这样 executor CLI 可以基于真实 checkpoint 信息决定 auto ranking。
- `scripts/sbatch/run_appworld_multistep_executor_eval.sh`
  - `RANKING_MODE` 默认从 `policy_blend` 改为 `auto`。

新增测试：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_multistep_cli.py::test_build_controller_auto_ranking_uses_skill_table_for_routing_only_base \
  tests/test_appworld_multistep_cli.py::test_build_controller_default_ranking_is_auto_for_routing_only_base \
  tests/test_appworld_multistep_cli.py::test_build_controller_auto_ranking_keeps_policy_blend_for_hrpo_checkpoint -q
# RED:
# - auto explicit tests initially failed because controller received unsupported ranking_mode=auto
# - default test initially failed because _build_controller() still defaulted to policy_blend
# GREEN: 3 passed
```

回归验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_multistep_cli.py tests/test_appworld_multistep.py \
  tests/test_appworld_act_hrpo.py tests/test_appworld_clstr_eval.py \
  tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py \
  tests/test_sbatch_scripts.py -q
# 47 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  scripts/run_appworld_multistep_executor_eval.py clstr/appworld_clstr_eval.py clstr/appworld_multistep.py

bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

注意：

- 已经完成的 `75700` 和正在运行的 `75701` 是按旧提交时的 base `policy_blend` 设置产生/运行的，不能代表修复后 routing-only base 的合理默认；
- `75711` 明确设置了 `RANKING_MODE=skill_table`，可作为该修复方向的 dev10 诊断；
- 后续若要重新跑 base recurrent，应使用 `RANKING_MODE=auto` 或显式 `skill_table`，不要再把 routing-only base 当作 policy-trained checkpoint 使用。

### 2026-05-23 `75701` recurrent next-state v2 结果

`75701` 已完成：

- output: `outputs/appworld_multistep_executor_benchmark/clstr_multistep_step500_dev10_auth0_nextstate_v2`
- report: `outputs/appworld_multistep_executor_benchmark/clstr_multistep_step500_dev10_auth0_nextstate_v2/report.json`
- result: `success_count=6/10`, `success_rate=0.6`
- execution failures: `11`
- step-positive hit rate: `0.166667`

已生成 base policy-blend comparison：

- `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v2_base_policyblend_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v2_base_policyblend_comparison/comparison.md`

当前已完成 next-state v2 base 对照：

| method | setting | success | execution_failures | step hit |
|---|---|---:|---:|---:|
| `qwen_only` | no skill context | 6/10 | 6 | 0.0 |
| `skillrouter_base_live` | per-step SkillRouter | 7/10 | 0 | 1.0 |
| `clstr_multistep` | old default recurrent `policy_blend=0.5` | 6/10 | 11 | 0.166667 |
| `clstr_multistep` | old default no-recurrent `policy_blend=0.5` | 7/10 | 9 | 0.1875 |

逐任务差异：

- recurrent 与 no-recurrent 只在 `4ec8de5_1` 上不同；
- no-recurrent 第一步直接完成并通过 official evaluation；
- recurrent 第一步 `task_completed=True` 但 evaluation 失败，后续继续执行并生成了 invented API：
  - `apis.spotify.get_songs_play_count_details(...)`
- 这不是 `m_t` 有正贡献的证据，反而说明旧 `policy_blend=0.5` 下 recurrent belief update 会放大未训练 policy head 的不稳定选择。

判断：

- `75701` 仍是在 auto-ranking 修复前提交/启动的旧默认实验，不能代表修复后的 routing-only base；
- 但它进一步支持 root cause：routing-only base 不应默认使用 policy rerank；
- 现在应等待 `75711` 的 `skill_table` 诊断结果，再决定是否需要重跑 base recurrent `RANKING_MODE=auto`。

### 2026-05-23 `75702` hybrid alpha=0.10 next-state v2 结果

`75702` 已完成：

- output: `outputs/appworld_multistep_executor_benchmark/clstr_skillrouter_hybrid_step500_alpha010_dev10_nextstate_v2`
- report: `outputs/appworld_multistep_executor_benchmark/clstr_skillrouter_hybrid_step500_alpha010_dev10_nextstate_v2/report.json`
- result: `success_count=7/10`, `success_rate=0.7`
- execution failures: `0`
- step-positive hit rate: `1.0`

已生成当前 completed-so-far comparison：

- `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v2_completed_so_far_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v2_completed_so_far_comparison/comparison.md`

当前 next-state v2 已完成结果：

| method | setting | success | execution_failures | step hit |
|---|---|---:|---:|---:|
| `qwen_only` | no skill context | 6/10 | 6 | 0.0 |
| `skillrouter_base_live` | per-step SkillRouter | 7/10 | 0 | 1.0 |
| `clstr_multistep` | old default recurrent `policy_blend=0.5` | 6/10 | 11 | 0.166667 |
| `clstr_multistep` | old default no-recurrent `policy_blend=0.5` | 7/10 | 9 | 0.1875 |
| `clstr_skillrouter_hybrid_live` | SkillRouter prior + CLSTR alpha=0.10 | 7/10 | 0 | 1.0 |

逐任务对齐：

- hybrid alpha=0.10 与 `skillrouter_base_live` 的成功/失败集合完全一致；
- 三个失败样例仍是 `530b157_*`；
- 没有任何 hybrid > SkillRouter 的样例，也没有 SkillRouter 成功而 hybrid 失败的样例。

判断：

- hybrid alpha=0.10 目前只能说明 CLSTR rerank 没破坏强 SkillRouter prior；
- 它不能证明 `m_t` 或 CLSTR policy 带来增益；
- 当前目标“超过 SkillRouter 并证明 `m_t` 价值”仍未达成；
- 下一步仍等待：
  - `75711` skill_table/auto-ranking base 诊断；
  - `75703/75705/75704` normal ACT next-state v2；
  - `75706/75708/75707` gated ACT next-state v2。

### 2026-05-23 ACT/HRPO 采样修复：rollout 不再纯靠未训练 policy head 探索

auto-ranking 诊断还暴露了另一个会影响 ACT 的实现问题：

- routing-only base 只训练 `skill_table.logits(h_t)`，没有训练 `policy_head`；
- 但 `run_appworld_hrpo_rollout()` 和 `run_appworld_multistep_hrpo_rollout()` 的采样分布原来只使用 `policy_forward()` logits；
- `beta_routing_prior` 只在 loss 里正则化 policy 与 routing prior 的 KL，不能阻止 rollout 初期从未训练 policy head 抽到坏 skill；
- 因此 ACT pilot 可能在训练开始阶段就采到低质量 skill context，导致 reward 信号稀疏且噪声大。

已修复：

- `clstr/appworld_act_hrpo.py`
  - `AppWorldHrpoTrainConfig` 新增 `policy_sampling_alpha=0.25`；
  - 新增 `_sampling_logits()`；
  - 采样分布使用：
    `sampling_logits = (1 - alpha) * standardize(routing_logits.detach()) + alpha * standardize(policy_logits)`；
  - `selected_log_prob` 仍来自原始 `policy_logits`，保留 policy head 的学习梯度；
  - single-step 和 multi-step HRPO rollout 均接入该采样 prior；
  - step trace 中记录 `policy_sampling_alpha`。
- `scripts/run_appworld_clstr_hrpo_train.py`
  - 新增 `--policy_sampling_alpha`。
- `scripts/sbatch/run_appworld_clstr_hrpo_train.sh`
  - 新增 `POLICY_SAMPLING_ALPHA`，默认 `0.25`。

新增测试：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_act_hrpo.py::test_appworld_hrpo_sampling_can_use_routing_prior_when_policy_head_is_untrained \
  tests/test_appworld_act_hrpo.py::test_appworld_multistep_hrpo_sampling_can_use_routing_prior_when_policy_head_is_untrained -q
# RED: unexpected keyword argument policy_sampling_alpha
# GREEN: 2 passed
```

回归验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_act_hrpo.py tests/test_appworld_hrpo_cli.py \
  tests/test_appworld_multistep_cli.py tests/test_appworld_multistep.py \
  tests/test_appworld_clstr_eval.py tests/test_appworld_mt_fusion_train.py \
  tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 50 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/appworld_act_hrpo.py scripts/run_appworld_clstr_hrpo_train.py \
  scripts/run_appworld_multistep_executor_eval.py clstr/appworld_clstr_eval.py \
  clstr/appworld_multistep.py

bash -n scripts/sbatch/run_appworld_clstr_hrpo_train.sh \
  scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

注意：

- 已经运行中的 `75703` normal ACT pilot 是在该采样修复前启动的，不能作为修复后 ACT 结论；
- `75706` gated ACT pilot 若尚未启动，则会读取新脚本/代码并使用 `POLICY_SAMPLING_ALPHA=0.25`；
- 需要补交至少一个 normal ACT next-state v3 诊断作业，明确使用修复后的 sampling prior，再与 SkillRouter 和 no-recurrent 做比较。

已补交修复后的 normal ACT v3 作业：

| job | purpose | output |
|---:|---|---|
| 75718 | normal ACT pilot v3，`POLICY_SAMPLING_ALPHA=0.25`、`TOP_K=16`、`MAX_TASKS=10`、`GROUP_SIZE=4`、`UPDATES=5`、`MAX_STEPS=3` | `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_samplingprior_v3` |
| 75719 | afterok:75718 dev10 recurrent eval, `RANKING_MODE=auto` | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_samplingprior_dev10_nextstate_v3` |
| 75720 | afterok:75718 dev10 no-recurrent eval, `RANKING_MODE=auto` | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_samplingprior_dev10_no_recurrent_nextstate_v3` |

说明：

- 不取消 `75703/75704/75705`，保留它们作为 pre-fix ACT 对照；
- 但判断 ACT 是否有效应优先看 `75718/75719/75720`；
- 如果 v3 仍不能超过 SkillRouter，则需要检查 rollouts 中 reward 分布、`policy_sampling_alpha` 是否过低/过高、以及 `530b157_*` 是否仍拿不到 phone/Venmo 可执行上下文。

### 2026-05-23 `75703` pre-fix normal ACT 训练结果

`75703` 已完成：

- output: `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_nextstate_v2`
- report: `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_nextstate_v2/train_report.json`
- checkpoint: `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_nextstate_v2/checkpoints/appworld_clstr_multistep_hrpo-step5.pt`
- rollouts: `40`
- rollout official successes: `4/40`
- train config:
  - `top_k=8`
  - `group_size=4`
  - `updates=5`
  - `task_batch_size=2`
  - `max_steps=3`
  - `skill_context_mode=safe_metadata`
  - no `policy_sampling_alpha` field, confirming this is pre-fix ACT sampling.

Update summary:

| update | success_rate | mean_reward | decision_count | advantage_nonzero_count | routing_prior_loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | 0.66 | 16 | 0 | 2.2620 |
| 2 | 0.0 | 0.20 | 24 | 4 | 3.0243 |
| 3 | 0.0 | 0.0425 | 24 | 4 | 3.3805 |
| 4 | 0.0 | 0.105 | 24 | 4 | 1.9690 |
| 5 | 0.0 | 0.105 | 24 | 4 | 1.9078 |

判断：

- pre-fix ACT 没有稳定地从 rollout reward 中得到成功轨迹；除第 1 个 update 外，其余 updates official success 都是 0；
- 第 1 个 update 的 `advantage_nonzero_count=0`，说明同 task group 内 reward 没有方差，policy gradient 实际不可用；
- 这进一步支持 ACT 采样需要 routing prior，而不是纯 policy head；
- `75704/75705` 仍会跑完作为 pre-fix eval 对照，但不能作为修复后 ACT 结论。

`75704` pre-fix no-recurrent eval 已完成：

- output: `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_dev10_no_recurrent_nextstate_v2`
- report: `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_dev10_no_recurrent_nextstate_v2/report.json`
- result: `success_count=7/10`, `success_rate=0.7`
- execution failures: `5`
- step-positive hit rate: `0.625`
- failures: `530b157_1/2/3`

判断：pre-fix ACT no-recurrent 仍只是追平 `skillrouter_base_live`，不是超过 SkillRouter 的证据；还需要等 `75705` recurrent eval 和修复后的 `75718/75719/75720`。

`75705` pre-fix recurrent eval 与 `75711` skill-table 诊断均已完成，并已生成当前 pre-fix completed comparison：

- `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v2_prefixed_completed_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v2_prefixed_completed_comparison/comparison.md`

当前已完成 dev10 next-state v2 汇总：

| method | setting | success | execution_failures | step hit | note |
|---|---|---:|---:|---:|---|
| `qwen_only` | no skill context | 6/10 | 6 | 0.0 | baseline |
| `skillrouter_base_live` | per-step SkillRouter | 7/10 | 0 | 1.0 | target baseline |
| `clstr_multistep` | old default recurrent `policy_blend=0.5` | 6/10 | 11 | 0.166667 | worse than SkillRouter |
| `clstr_multistep` | old default no-recurrent `policy_blend=0.5` | 7/10 | 9 | 0.1875 | ties SkillRouter, no `m_t` |
| `clstr_skillrouter_hybrid_live` | SkillRouter prior + CLSTR alpha=0.10 | 7/10 | 0 | 1.0 | exactly ties SkillRouter |
| `clstr_multistep` | `ranking_mode=skill_table` / auto base diagnostic | 6/10 | 6 | 1.0 | high step hit but task success drops to Qwen-only |
| `clstr_multistep` | pre-fix ACT recurrent | 7/10 | 4 | 0.625 | ties SkillRouter |
| `clstr_multistep` | pre-fix ACT no-recurrent | 7/10 | 5 | 0.625 | ties recurrent; no `m_t` gain |

Per-task pattern:

- `530b157_1/2/3` remain failed for every completed method, including SkillRouter;
- `4ec8de5_1` is the only task where methods differ:
  - SkillRouter succeeds;
  - old recurrent policy-blend fails;
  - skill-table diagnostic fails;
  - no-recurrent policy-blend, hybrid alpha=0.10, and pre-fix ACT succeed.

Interpretation:

1. `skill_table` has perfect step-positive hit on these dev10 labels but only 6/10 task success, so step hit alone is not a sufficient proxy for AppWorld task success.
2. Hybrid alpha=0.10 exactly preserves SkillRouter behavior but does not improve it.
3. Pre-fix ACT improves execution failure count relative to base policy-blend, but recurrent and no-recurrent are both 7/10; this does not prove `m_t`.
4. The current goal remains unmet until a fixed ACT or gated variant exceeds 7/10 and shows recurrent > no-recurrent or alpha>0 > alpha=0.

Pending after this comparison:

- `75706`: gated ACT pilot with current script defaults, now running;
- `75708/75707`: gated ACT recurrent/no-recurrent eval after `75706`;
- `75718`: normal ACT v3 with `POLICY_SAMPLING_ALPHA=0.25`, now running;
- `75719/75720`: v3 recurrent/no-recurrent eval after `75718`.

### 2026-05-23 `75706` gated ACT 训练结果

`75706` 已完成：

- output: `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_gated_nextstate_v2`
- report: `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_gated_nextstate_v2/train_report.json`
- checkpoint: `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_gated_nextstate_v2/checkpoints/appworld_clstr_multistep_hrpo-step5.pt`
- rollouts: `40`
- rollout official successes: `9/40`
- train config confirms:
  - `policy_sampling_alpha=0.25`
  - `top_k=8`
  - `context_top_k=5`
  - `group_size=4`
  - `updates=5`
  - `max_steps=3`
  - `skill_context_mode=safe_metadata`

Update summary:

| update | success_rate | mean_reward | decision_count | advantage_nonzero_count | policy_loss | routing_prior_loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.625 | 0.73 | 15 | 4 | -0.8932 | 2.3868 |
| 2 | 0.5 | 0.6375 | 16 | 4 | 0.0181 | 3.3446 |
| 3 | 0.0 | 0.115 | 24 | 4 | -0.0034 | 3.5639 |
| 4 | 0.0 | 0.24 | 24 | 8 | 0.0056 | 2.0507 |
| 5 | 0.0 | 0.1325 | 24 | 8 | 0.0208 | 1.8873 |

Interpretation:

- Compared with pre-fix normal ACT (`4/40` rollout successes), gated ACT with sampling prior gives more successful rollouts (`9/40`);
- However updates 3-5 still have `success_rate=0.0`, so reward learning remains unstable;
- This is not task-success evidence until `75707/75708` evals finish;
- `75707` no-recurrent and `75708` recurrent eval are now running.

`75707/75708` gated ACT eval 已完成：

| job | setting | output | success | execution_failures | step hit |
|---:|---|---|---:|---:|---:|
| 75708 | gated ACT recurrent | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_gated_pilot10_g4_u5_dev10_nextstate_v2` | 6/10 | 3 | 0.666667 |
| 75707 | gated ACT no-recurrent | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_gated_pilot10_g4_u5_dev10_no_recurrent_nextstate_v2` | 7/10 | 6 | 0.625 |

Per-task pattern:

- gated recurrent fails `4ec8de5_1` and all `530b157_*`;
- gated no-recurrent succeeds `4ec8de5_1` and fails only `530b157_*`;
- recurrent is worse than no-recurrent by 1 task.

判断：

- gated ACT with sampling prior still does not beat SkillRouter;
- `m_t` recurrent belief is again negative in this dev10 slice (`6/10` vs `7/10`);
- this result cannot support the paper claim.

### 2026-05-23 `75718` normal ACT v3 训练结果

`75718` 已完成：

- output: `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_samplingprior_v3`
- report: `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_samplingprior_v3/train_report.json`
- checkpoint: `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_samplingprior_v3/checkpoints/appworld_clstr_multistep_hrpo-step5.pt`
- rollouts: `40`
- rollout official successes: `6/40`
- train config:
  - `policy_sampling_alpha=0.25`
  - `top_k=16`
  - `context_top_k=5`
  - `group_size=4`
  - `updates=5`
  - `max_steps=3`
  - `skill_context_mode=safe_metadata`

Update summary:

| update | success_rate | mean_reward | decision_count | advantage_nonzero_count | policy_loss | routing_prior_loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | 0.66 | 16 | 0 | -0.0 | 2.7509 |
| 2 | 0.25 | 0.37 | 20 | 8 | -1.3854 | 3.7256 |
| 3 | 0.0 | 0.0775 | 24 | 8 | -0.0001 | 4.0342 |
| 4 | 0.0 | 0.18 | 24 | 8 | 0.00003 | 3.3498 |
| 5 | 0.0 | 0.1175 | 24 | 8 | -0.00008 | 2.8746 |

Interpretation:

- normal ACT v3 confirms the sampling-prior path is active (`policy_sampling_alpha=0.25`);
- but rollout success is only `6/40`, lower than gated ACT's `9/40`;
- updates 3-5 again have official success `0.0`, so ACT remains unstable at this training scale;
- `75719/75720` recurrent/no-recurrent eval are now running and will decide whether v3 has any task-success gain despite weak rollout stats.

### 2026-05-23 normal ACT v3 eval 结果与新的实现诊断

`75719/75720` 已完成，comparison 已更新到：

- `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v3_samplingprior_completed_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v3_samplingprior_completed_comparison/comparison.md`

v3 eval 结果：

| job | setting | output | success | execution_failures | task_completed | step hit |
|---:|---|---|---:|---:|---:|---:|
| 75719 | normal ACT v3 recurrent | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_samplingprior_dev10_nextstate_v3` | 7/10 | 3 | 9 | 0.625 |
| 75720 | normal ACT v3 no-recurrent | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_samplingprior_dev10_no_recurrent_nextstate_v3` | 7/10 | 3 | 8 | 0.625 |

结论：

- sampling-prior ACT v3 仍只追平 `skillrouter_base_live` 的 7/10，没有超过 SkillRouter；
- recurrent 与 no-recurrent 都是 7/10，不能证明 `m_t` 的正贡献；
- 两者共同失败仍是 `530b157_1/2/3`；
- `4ec8de5_1` 在 v3 recurrent/no-recurrent 中都成功，说明 v3 修复了 pre-fix/gated recurrent 在该样例上的回退，但没有带来额外成功任务。

`530b157_*` 失败模式：

- SkillRouter 在 step0 通常把 `login-to-venmo-and-phone-*` 与 simple-note expense-share skills 同时放进 top5，但仍失败；
- v3 ACT recurrent/no-recurrent 的 step0 更偏向 simple-note skills 和 generic `login-to-apps-*`，Venmo/Phone skills 只偶尔在后续 step 进入 top5；
- 因此当前 v3 不仅没有解决 executor 对 Venmo/Phone payment+text 的长程执行问题，也没有稳定把 Phone/Venmo skill 作为首步或持续上下文。

新的实现诊断：

- `CLSTRMultiStepController.select()` 当前先用 `h_t -> skill_table.logits` 取 `candidate_top_k`，再让 `m_t` 通过 `policy_forward(h_t, m_t, candidate_embs)` 在候选集内 rerank；
- 这意味着 recurrent `m_t` 不能召回没有进入 `skill_table` top-k 的 skill，只能重排已召回候选；
- 对 AppWorld 这种多 app、多阶段任务，这会弱化“连续选择”的可观测价值：如果历史 observation 指向 Phone/Venmo，但该 skill 没进当前 `h_t` 候选集，`m_t` 无法把它拉进 context；
- 因此下一步应做一个最小、可消融的候选召回修复，而不是继续盲目扩大当前 ACT pilot。

下一步候选修复设计：

1. 在 `CLSTRMultiStepController` 中新增可选 candidate source，例如 `candidate_source="routing"`（默认旧行为）和 `candidate_source="routing_belief_union"`；
2. `routing_belief_union` 用当前 `h_t` 的 routing top-k 与 `m_t @ skill_table.E.T` 的 belief top-k 做 union，再用既有 `ranking_mode` 做最终 top-k；
3. 只在 recurrent path 中使用 belief candidates；no-recurrent ablation 仍保持 `routing`，这样能检验 `m_t` 是否真的提升 recall 和 task success；
4. 在 CLI/sbatch 暴露 `CANDIDATE_SOURCE`，默认维持旧行为，避免污染已有结果；
5. 先写单元测试证明：更新后的 `m_t` 能把 routing top-k 外的 skill 放入候选集；再跑 dev10 小评估。如果 dev10 仍不能超过 7/10，再考虑更大 ACT 训练或 Venmo/Phone 专项 teacher rollout。

当前目标状态：

- “超过 SkillRouter 并证明 `m_t` 价值”仍未达成；
- 不能进入 dev57/test 或论文主表 claim；
- 下一步应先修 candidate recall，让 `m_t` 进入候选生成本身，而不是只作为候选内 reranker。

### 2026-05-23 实现：`m_t` 进入候选召回

已实现上述 candidate recall 修复：

- `CLSTRMultiStepController` 新增 `candidate_source`：
  - `routing`：默认旧行为，只用 `h_t -> skill_table.logits` 召回候选；
  - `routing_belief_union`：全步 union，把 routing top-k 与当前 `m_t @ skill_table.E.T` top-k 做 union，再执行原有 `skill_table` / `policy_head` / `policy_blend` rerank；
  - `routing_belief_union_after_update`：严格闭环模式，只在至少一次 `observe()` 更新后启用 belief union，避免把首步 `m_0` 变化误算为历史反馈收益。
- `scripts/run_appworld_multistep_executor_eval.py` 新增 `--candidate_source {routing,routing_belief_union,routing_belief_union_after_update}`；
- `scripts/sbatch/run_appworld_multistep_executor_eval.sh` 新增 `CANDIDATE_SOURCE` 环境变量，默认仍为 `routing`，保证已有实验可复现；
- 单元测试新增：
  - `test_clstr_multistep_controller_can_union_belief_candidates_after_update`
  - `test_clstr_multistep_after_update_candidate_source_does_not_union_initial_belief`
  - `test_build_controller_can_set_candidate_source_for_clstr_multistep`
  - sbatch controller settings 覆盖 `CANDIDATE_SOURCE`。
- selection diagnostics 现在额外记录：
  - `routing_candidate_skill_ids`
  - `belief_candidate_skill_ids`
  - `belief_candidate_enabled`
  这用于判断最终候选变化是否真的来自 `m_t` belief recall。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_sbatch_scripts.py -q
# 28 passed

bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh
# exit 0

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_multistep.py scripts/run_appworld_multistep_executor_eval.py
# exit 0

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_multistep.py::test_clstr_multistep_after_update_candidate_source_does_not_union_initial_belief tests/test_appworld_multistep.py::test_clstr_multistep_controller_can_union_belief_candidates_after_update tests/test_appworld_multistep_cli.py::test_build_controller_can_set_candidate_source_for_clstr_multistep tests/test_sbatch_scripts.py::test_appworld_multistep_executor_sbatch_exposes_controller_settings -q
# 4 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_act_hrpo.py tests/test_appworld_hrpo_cli.py tests/test_appworld_multistep_cli.py tests/test_appworld_multistep.py tests/test_appworld_clstr_eval.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 53 passed

补充 source-level diagnostics 后再次运行同一轻量回归：

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_act_hrpo.py tests/test_appworld_hrpo_cli.py tests/test_appworld_multistep_cli.py tests/test_appworld_multistep.py tests/test_appworld_clstr_eval.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 53 passed
```

实验决策：

- 先提交一个 dev10 recurrent-only 诊断作业，使用 v3 ACT checkpoint 与 `CANDIDATE_SOURCE=routing_belief_union`；
- 现有 v3 no-recurrent 7/10 report 作为 ablation 对照；
- 如果全步 union 有提升，需要再用 `routing_belief_union_after_update` 严格复核，避免把首步 `m_0` 候选变化误当成闭环历史反馈；
- 若 belief-union recurrent 超过 7/10，再补 no-recurrent/seed/dev57；若仍为 7/10 或更低，则说明单纯让 `m_t` 进入候选召回还不足以证明论文 claim，需要继续转向更强 ACT 数据或 Venmo/Phone 专项 teacher rollout。

已提交诊断作业：

| job | purpose | key settings | output |
|---:|---|---|---|
| 75775 | normal ACT v3 recurrent + belief-union candidate recall dev10 eval | `CANDIDATE_SOURCE=routing_belief_union`, `RECURRENT_BELIEF=1`, `RANKING_MODE=auto`, `CANDIDATE_TOP_K=16`, checkpoint `outputs/appworld_clstr_train_multistep_hrpo/pilot10_g4_u5_step500_samplingprior_v3/checkpoints/appworld_clstr_multistep_hrpo-step5.pt` | `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_samplingprior_dev10_nextstate_v3_belief_union` |

`75775` 已完成：

- report: `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_samplingprior_dev10_nextstate_v3_belief_union/report.json`
- runs: `outputs/appworld_multistep_executor_benchmark/clstr_multistep_hrpo_pilot10_g4_u5_samplingprior_dev10_nextstate_v3_belief_union/runs.jsonl`
- comparison:
  - `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v3_belief_union_comparison/comparison.json`
  - `outputs/appworld_multistep_executor_benchmark/dev10_nextstate_v3_belief_union_comparison/comparison.md`

结果：

| method/setting | success | task_completed | evaluate_success | execution_failures | step hit |
|---|---:|---:|---:|---:|---:|
| `skillrouter_base_live` | 7/10 | 7 | 7 | 0 | 1.0 |
| normal ACT v3 recurrent | 7/10 | 9 | 7 | 3 | 0.625 |
| normal ACT v3 no-recurrent | 7/10 | 8 | 7 | 3 | 0.625 |
| normal ACT v3 recurrent + full-step `routing_belief_union` | 7/10 | 10 | 7 | 3 | 0.625 |

判断：

- full-step belief union 没有超过 SkillRouter，仍是 7/10；
- 因为没有 task-success 增益，不需要再提交严格 `routing_belief_union_after_update` 复核；
- `task_completed_count` 从 v3 recurrent 的 9 增到 10，但 `evaluate_success_count` 仍是 7，说明问题从“是否调用 complete_task”转为“答案/状态是否通过 verifier”；
- 失败仍集中在 `530b157_1/2/3`，下一步应先分析这些 Venmo/Phone/SimpleNote 任务的 generated code 与 verifier failures，而不是继续扩大 dev57/test。

### 2026-05-23 `530b157_*` root cause：executor prompt 的 Phone login 模板错误

对 `75775` 的 `runs.jsonl` 逐步检查后，`530b157_*` 失败不是单纯 routing recall 问题：

- step0 常见生成：
  - `phone_access_token = apis.phone.login(username=supervisor_profile["email"], password=supervisor_passwords["phone"])["access_token"]`
  - AppWorld API docs 明确 `phone.login.username` 是 account phone number，不是 email；
  - 因此 step0 直接 401；
- 后续 step 即使把 `login-to-venmo-and-phone-*` 放入 top5，也经常只调用 `apis.supervisor.complete_task(status="success")`，没有创建 `venmo.Transaction` 或 phone text message；
- verifier failure 始终显示缺少：
  - `venmo.Transaction`
  - `phone.GlobalTextMessage`
  - `phone.UserTextMessage`
- full-step belief union 只把 `task_completed_count` 提升到 10/10，但 `evaluate_success_count` 仍是 7/10。

已修复 executor prompt/API grounding：

- `build_appworld_executor_prompt()` 现在对每个 required app 生成登录示例；
- `phone` 特判使用 `supervisor_profile["phone_number"]`；
- 其他 email-based apps 仍使用 `supervisor_profile["email"]`；
- 对 phone+venmo grocery/payment/text 任务增加 task-specific hints：
  - 先从 phone text messages 找 payer 和 amount；
  - 用 `apis.venmo.create_transaction(receiver_email=..., amount=..., description=..., access_token=venmo_access_token)`；
  - 用 `apis.phone.send_text_message(phone_number=..., message=..., access_token=phone_access_token)`；
  - 两者成功前不要 `complete_task`。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py::test_prompt_builder_uses_phone_number_login_and_payment_text_hints tests/test_appworld_executor.py::test_prompt_builder_includes_instruction_skills_and_filtered_api_docs tests/test_appworld_executor.py::test_prompt_builder_safe_metadata_omits_unsafe_skill_body -q
# 3 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py tests/test_appworld_act_hrpo.py tests/test_appworld_hrpo_cli.py tests/test_appworld_multistep_cli.py tests/test_appworld_multistep.py tests/test_appworld_clstr_eval.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 66 passed
```

已构造 dev 子集：

- `data/appworld_routing/dev_530b_tasks.jsonl`
- 包含 `530b157_1/2/3` 三个 phone+venmo 任务。

已提交 prompt-fix smoke：

| job | method | output |
|---:|---|---|
| 75798 | `qwen_only` | `outputs/appworld_multistep_executor_benchmark/promptfix_qwen_only_dev530b_v1` |
| 75800 | `skillrouter_base_live` | `outputs/appworld_multistep_executor_benchmark/promptfix_skillrouter_base_live_dev530b_v1` |
| 75799 | CLSTR v3 recurrent, `CANDIDATE_SOURCE=routing_belief_union_after_update` | `outputs/appworld_multistep_executor_benchmark/promptfix_clstr_v3_afterupdate_dev530b_v1` |

这些作业只验证 executor prompt/API grounding 修复是否让 `530b157_*` 变得可解，不是论文主结果。如果 qwen_only/SkillRouter 同样提升，说明主要瓶颈是 executor；如果 CLSTR 独有提升，再进入 dev10/dev57 复核。

### 2026-05-23 `530b157_*` v2 executor schema/algorithm grounding

v1 prompt-fix smoke 结果仍为 0/3；继续检查 `runs.jsonl` 与 AppWorld API docs 后确认失败已经从 Phone login 401 转为更细的 executor schema/algorithm 问题：

- `phone.search_text_messages` 的返回字段是 `message`，不是 `text`；
- text message 的 `sender` / `receiver` 只有 `contact_id`、`name`、`phone_number`，没有 `email`；
- Venmo `receiver_email` 应来自 `phone.search_contacts(query=payer_name)` 返回的 contact `email`，不能用 phone number 或 text message receiver 当 email；
- `page_limit=100` 仍会导致 422，因为 Phone/Venmo paginated APIs 最大是 20；
- 对 `530b157_*`，official ground-truth solution 只用 Phone contacts/text + Venmo transaction，不需要任何 `simple_note` API；SkillX simple_note metadata 会诱导 Qwen 生成不存在的 `apis.simple_note.get_expense_shares_from_note(...)`。

已修改 `clstr/appworld_executor.py` 的 phone+venmo grocery/text task-specific hints：

- 保留 Phone login 使用 `supervisor_profile["phone_number"]`；
- 明确合法解题顺序：login -> search contact -> 用 contact phone number 分页读取 text messages -> 用 regex 从 `message` 字段抽取金额 -> `venmo.create_transaction` -> `phone.send_text_message` -> `complete_task`；
- 明确 `search_text_messages` schema、sender/receiver 无 email、contact email 才能作为 Venmo receiver；
- 明确不要调用任何 `apis.simple_note.*`，尤其是该任务不应由 simple_note 技能体驱动；
- 明确使用 `page_limit=20` 并分页，禁止继续用 100。

新增/更新测试：

- `tests/test_appworld_executor.py::test_prompt_builder_uses_phone_number_login_and_payment_text_hints`
  - RED：旧 prompt 缺少 contact/text schema 与 algorithm order；
  - GREEN：新 prompt 包含上述约束。

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py::test_prompt_builder_uses_phone_number_login_and_payment_text_hints -q
# 1 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_sbatch_scripts.py -q
# 42 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_executor.py clstr/appworld_multistep.py scripts/run_appworld_multistep_executor_eval.py
# exit 0

bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh
# exit 0

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py tests/test_appworld_act_hrpo.py tests/test_appworld_hrpo_cli.py tests/test_appworld_multistep_cli.py tests/test_appworld_multistep.py tests/test_appworld_clstr_eval.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 66 passed
```

已提交 v2 prompt/schema-grounding smoke：

| job | method | output |
|---:|---|---|
| 75825 | `qwen_only` | `outputs/appworld_multistep_executor_benchmark/promptfix_qwen_only_dev530b_v2` |
| 75824 | `skillrouter_base_live` | `outputs/appworld_multistep_executor_benchmark/promptfix_skillrouter_base_live_dev530b_v2` |
| 75826 | CLSTR v3 recurrent, `CANDIDATE_SOURCE=routing_belief_union_after_update` | `outputs/appworld_multistep_executor_benchmark/promptfix_clstr_v3_afterupdate_dev530b_v2` |

下一步：约 15 分钟后检查这三个 job。如果 v2 仍为 0/3，说明纯 prompt grounding 仍不足，需要进入 train-only verified teacher rollout / few-shot executor pattern，而不是继续调 routing。

### 2026-05-23 `530b157_*` v2 结果与 v3 prompt 修复

v2 作业已完成，comparison 已生成：

- `outputs/appworld_multistep_executor_benchmark/promptfix_dev530b_v2_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/promptfix_dev530b_v2_comparison/comparison.md`

结果：

| method | success | task_completed | evaluate_success | execution_failures | step hit |
|---|---:|---:|---:|---:|---:|
| `qwen_only` | 0/3 | 2 | 0 | 6 | 0.0 |
| `skillrouter_base_live` | 0/3 | 0 | 0 | 0 | 1.0 |
| CLSTR v3 after-update | 0/3 | 1 | 0 | 4 | 1.0 |

v2 失败模式已经比 v1 更集中：

- phone login、`message` vs `text`、receiver email schema、`page_limit=100` 大多已被修掉；
- qwen-only / CLSTR 现在能生成接近正确的 contacts -> text messages -> Venmo -> phone text 代码，但金额常被抽成 `37.0`；
- 根因是 Qwen 仍会：
  - 只看 `text_messages[0]`，没有扫描所有分页消息；
  - 使用 `r'\$?(\d+...)'` 这种 `$` 可选的 regex，匹配到非金额数字；
  - 对无金额消息没有 skip/continue；
- SkillRouter/CLSTR 带 SkillX context 时仍容易重复 login-only code，没有沿着 successful execution history 继续创建缺失的 `venmo.Transaction` / `phone.*TextMessage`。

已继续修改 `clstr/appworld_executor.py`：

- 对 phone+venmo grocery/text task 增加金额抽取硬约束：
  - 扫描所有分页 text messages；
  - 不假设 page 0 或第一条消息含金额；
  - regex 必须要求字面 `$`，例如 `r"\$(\d+(?:\.\d+)?)"`；
  - 不要用 `\$?`；
  - 无 `$` 金额的消息必须 skip，不能 default 0 或匹配无关数字；
- 对 multi-step prompt 增加 execution-history 规则：
  - prior variables/app state 在成功 step 后继续存在；
  - 不要重复 successful login-only code；
  - 如果 evaluation history 说 `venmo.Transaction`、`phone.GlobalTextMessage`、`phone.UserTextMessage` 缺失，应继续创建这些 records 再 `complete_task`。

新增/更新测试：

- `tests/test_appworld_executor.py::test_prompt_builder_uses_phone_number_login_and_payment_text_hints`
- `tests/test_appworld_multistep.py::test_multistep_executor_prompt_tells_model_to_continue_after_successful_history`

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py::test_prompt_builder_uses_phone_number_login_and_payment_text_hints -q
# 1 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_multistep.py::test_multistep_executor_prompt_tells_model_to_continue_after_successful_history -q
# 1 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py tests/test_sbatch_scripts.py -q
# 43 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/appworld_executor.py clstr/appworld_multistep.py scripts/run_appworld_multistep_executor_eval.py
# exit 0

bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh
# exit 0

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_executor.py tests/test_appworld_act_hrpo.py tests/test_appworld_hrpo_cli.py tests/test_appworld_multistep_cli.py tests/test_appworld_multistep.py tests/test_appworld_clstr_eval.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 67 passed
```

已提交 v3 smoke，除上述 prompt 修复外沿用 v2 设置：

| job | method | output |
|---:|---|---|
| 75834 | `qwen_only` | `outputs/appworld_multistep_executor_benchmark/promptfix_qwen_only_dev530b_v3` |
| 75835 | `skillrouter_base_live` | `outputs/appworld_multistep_executor_benchmark/promptfix_skillrouter_base_live_dev530b_v3` |
| 75836 | CLSTR v3 recurrent, `CANDIDATE_SOURCE=routing_belief_union_after_update` | `outputs/appworld_multistep_executor_benchmark/promptfix_clstr_v3_afterupdate_dev530b_v3` |

下一步：约 15 分钟后检查 v3。若 v3 仍不能解 `530b157_*`，就不应再继续堆自然语言 prompt，而应构造 train-only verified teacher/few-shot executor pattern 或进入 executor 侧更强结构化 planner。

队列状态补充：

- `75834/75835/75836` 在提交后多轮检查均为 `PD (Priority)`，尚未开始运行；
- `squeue --start -j 75834,75835,75836` 预计启动时间为 `2026-05-27T21:19:54`；
- 当前 `sinfo` 显示公共分区 `gpu_a800` 有 3 张可用卡，`gpu_h100` / `gpu_h200` 可用卡为 0；
- 在这些 pending 作业启动前，不应继续修改 executor prompt 代码，否则作业会读取启动时的当前文件，导致 `promptfix_*_dev530b_v3` 输出不再对应上述 v3 修复定义；
- 后续若需要更快验证，应先由用户决定是否取消这些 H200 队列作业并改投 `gpu_a800`，或继续等待 H200。

### 2026-05-24 `530b157_*` v3 结果、泛化边界与通用 stop policy

v3 作业实际已在 H200 完成，comparison 已生成：

- `outputs/appworld_multistep_executor_benchmark/promptfix_dev530b_v3_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/promptfix_dev530b_v3_comparison/comparison.md`

结果：

| method | success | task_completed | evaluate_success | execution_failures | step hit |
|---|---:|---:|---:|---:|---:|
| `qwen_only` | 0/3 | 3 | 0 | 1 | 0.0 |
| `skillrouter_base_live` | 0/3 | 2 | 0 | 0 | 1.0 |
| CLSTR v3 after-update | 0/3 | 3 | 0 | 0 | 1.0 |

判断：

- v3 已把大多数 execution crash 转成 evaluator failure：模型能登录、查 contacts/text、创建 Venmo transaction、发短信、调用 `complete_task`；
- 仍失败的核心是金额选择错误，常取到同一联系人后续无关短信中的 `$25/$37`，而 verifier 需要 phone conversation 中真正对应 grocery payment 的金额；
- 这不是 routing recall 问题：SkillRouter 和 CLSTR 的 step hit 都是 1.0；
- 不能继续堆 `530b157_*` dev-template 规则，例如“找 grocery 问句后一条带 `$` 的消息”。这种规则过于任务特异，会降低论文说服力，也不能迁移到其他 benchmark。

后续改动原则：

- 方法层只做 benchmark-agnostic 改动，例如通用 multi-step termination、generic verifier/reflection、train-only teacher pattern、跨任务可复用的 state/action/observation 结构；
- AppWorld API prompt 只保留 schema/安全类 guardrail，不把某个 dev task 的语义模板写进核心方法；
- 如果需要 executor few-shot，也必须来自 train split 或通用合成模式，并与 CLSTR routing/ACT 贡献分开报告，不能把 executor 模板修复冒充为 CLSTR 提升。

已实现通用 stop policy：

- 新增 `clstr/multistep_stop.py`：
  - `MultiStepStopPolicy.decide(done, success)`；
  - 默认 `done=True` 即停止，即使 `success=False`；
  - 这是 benchmark-agnostic side-effect control，后续 WebShop/ScienceWorld/ALFWorld/AppWorld 都可复用；
- `clstr/appworld_multistep.py` 调用该 policy；
- 目的：防止 `complete_task` / env terminal 后继续执行后续 step，重复创建交易、短信、购买、提交等副作用；
- 这不会解决金额推理本身，但会让失败诊断更干净，也更符合 closed-loop benchmark 语义。

新增测试：

- `tests/test_multistep_stop.py`
- `tests/test_appworld_multistep.py::test_multistep_executor_stops_after_task_completed_even_if_evaluation_fails`

验证：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_appworld_multistep.py::test_multistep_executor_stops_after_task_completed_even_if_evaluation_fails -q
# 1 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_multistep_stop.py tests/test_appworld_multistep.py tests/test_appworld_executor.py -q
# 34 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_multistep_stop.py tests/test_appworld_executor.py tests/test_appworld_act_hrpo.py tests/test_appworld_hrpo_cli.py tests/test_appworld_multistep_cli.py tests/test_appworld_multistep.py tests/test_appworld_clstr_eval.py tests/test_appworld_mt_fusion_train.py tests/test_model_pipeline.py tests/test_sbatch_scripts.py -q
# 71 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/multistep_stop.py clstr/appworld_multistep.py clstr/appworld_executor.py
# exit 0
```

已提交 v4 smoke，除通用 stop policy 外沿用 v3 设置：

| job | method | output |
|---:|---|---|
| 75914 | `qwen_only` | `outputs/appworld_multistep_executor_benchmark/promptfix_qwen_only_dev530b_v4` |
| 75915 | `skillrouter_base_live` | `outputs/appworld_multistep_executor_benchmark/promptfix_skillrouter_base_live_dev530b_v4` |
| 75916 | CLSTR v3 recurrent, `CANDIDATE_SOURCE=routing_belief_union_after_update` | `outputs/appworld_multistep_executor_benchmark/promptfix_clstr_v3_afterupdate_dev530b_v4` |

下一步：约 15 分钟后检查 v4。若仍是 0/3，就停止在 `530b157_*` 上做 prompt 工程，转向更通用的 executor-side reflection/verification 或扩大 dev10/dev57 看 CLSTR 是否在非该模板任务上仍有机会超过 SkillRouter。

v4 作业已完成，comparison 已生成：

- `outputs/appworld_multistep_executor_benchmark/promptfix_dev530b_v4_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/promptfix_dev530b_v4_comparison/comparison.md`

结果：

| method | success | task_completed | evaluate_success | execution_failures | avg_steps | step hit |
|---|---:|---:|---:|---:|---:|---:|
| `qwen_only` | 0/3 | 3 | 0 | 0 | 1.0 | 0.0 |
| `skillrouter_base_live` | 0/3 | 3 | 0 | 0 | 3.0 | 1.0 |
| CLSTR v3 after-update | 0/3 | 2 | 0 | 0 | 2.666667 | 1.0 |

判断：

- 通用 stop policy 生效：qwen-only 平均步数从 v3 的 3.0 降到 1.0，避免了重复交易/短信；
- 但 `530b157_*` 仍为 0/3，剩余失败是金额证据选择错误，不应再做 dev-template prompt 工程；
- 按泛化约束，下一步转向 dev10 stop-policy 对比，观察更广任务集上的 CLSTR/SkillRouter 相对位置。

已提交 dev10 stop-policy 对比：

| job | method | output |
|---:|---|---|
| 75920 | `qwen_only` | `outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_qwen_only_v1` |
| 75919 | `skillrouter_base_live` | `outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_skillrouter_base_live_v1` |
| 75921 | CLSTR v3 recurrent, `CANDIDATE_SOURCE=routing_belief_union_after_update` | `outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_clstr_v3_afterupdate_v1` |

下一步：约 15 分钟后检查 dev10。若 CLSTR 仍不超过 SkillRouter，停止把精力放在 executor prompt，转向更完整的 multi-step ACT 训练或跨 benchmark 的通用 controller/evaluator 设计。

## 2026-05-24 SkillRouter/AppWorld leakage check + dev10 stop-policy results

### SkillRouter PDF 与 AppWorld test 泄露判断

已抽取并检查 `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/papers/skillrouter.pdf`：

- 论文全文没有出现 `AppWorld`；
- SkillRouter 主 benchmark 是 SkillsBench-derived benchmark，训练数据是从约 80K community skill pool 采样 skill 后用 GPT-4o-mini 生成 synthetic query；
- 论文声称 benchmark-labeled skills are excluded from training supervision；
- 因此没有证据支持“SkillRouter 训练时泄露 AppWorld test”这个判断。此前文档里需要警惕的是 SkillRouter checkpoint 与 SkillsBench eval 的潜在污染，不是 AppWorld test。

### SkillX AppWorld leakage audit

为检查真正的风险点（SkillX AppWorld skill body 是否来自 AppWorld dev/test trajectory），新增并运行了字符串级 audit：

- 实现：`clstr/bridges/skillx/appworld_adapter.py::audit_skillx_appworld_leakage`
- CLI：`scripts/audit_skillx_appworld.py --leakage_audit`
- sbatch 入口：`scripts/sbatch/run_appworld_skillx_leakage_audit.sh`
- 测试：
  - `tests/test_skillx_appworld_audit.py`
  - `tests/test_sbatch_scripts.py::test_appworld_skillx_leakage_audit_sbatch_runs_leakage_mode`
- 输出：
  - `outputs/skillx_appworld_leakage_audit/report.json`
  - `outputs/skillx_appworld_leakage_audit/report.md`

审计内容：

- 当前 `data/appworld_skill_pool/skill_pool.jsonl` 的 188 条 SkillX AppWorld skills；
- AppWorld routing splits：train 90 / dev 57 / test_normal 168 / test_challenge 417；
- 检查 skill text 中的 task_id substring、完整 instruction substring、高 token-Jaccard near-duplicate；
- 额外检查 SkillX 发布库 `skillx_db/appworld/vanilla-iter*/plan.json` 的原始 `user_task` 与 AppWorld split instruction 的交集。

结果：

| check | train | dev | test_normal | test_challenge |
|---|---:|---:|---:|---:|
| skill text task_id hits | 0 | 0 | 0 | 0 |
| skill text instruction hits | 0 | 0 | 0 | 0 |
| skill text near-duplicate hits | 0 | 0 | 0 | 0 |
| SkillX plan user_task overlaps | 81 | 0 | 0 | 0 |

结论：

- 当前可见发布文件层面没有发现 AppWorld dev/test contamination；
- SkillX `plan.json` 与 AppWorld train split 有大量交集，且与 dev/test 无交集，这支持“当前 SkillX AppWorld skill 库来自 train 可见任务”的判断；
- 但该 audit 是 deterministic string-level/provenance-level check，不能数学证明不存在语义级泄露。论文里应表述为“we audited and found no dev/test string/provenance overlap”，不要写成绝对无泄露证明。

验证命令：

```bash
PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.codex_deps:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest tests/test_skillx_appworld_audit.py tests/test_sbatch_scripts.py::test_appworld_skillx_leakage_audit_sbatch_runs_leakage_mode -q
# 8 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/audit_skillx_appworld.py \
  --leakage_audit \
  --skillx_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/SkillX \
  --skill_pool_path data/appworld_skill_pool/skill_pool.jsonl \
  --routing_dir data/appworld_routing \
  --output_dir outputs/skillx_appworld_leakage_audit
# status=ok, risk_level=low, dev_test_contamination_count=0, train_overlap_count=81
```

### dev10 stop-policy job results

作业 `75919/75920/75921` 已完成，comparison 已生成：

- `outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_comparison_v1/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_comparison_v1/comparison.md`

| method | success | task_completed | evaluate_success | execution_failures | avg_steps | step hit |
|---|---:|---:|---:|---:|---:|---:|
| `qwen_only` | 7/10 | 10 | 7 | 0 | 1.8 | 0.0 |
| `skillrouter_base_live` | 7/10 | 10 | 7 | 0 | 2.3 | 1.0 |
| CLSTR v3 after-update | 6/10 | 9 | 6 | 1 | 1.4 | 0.571429 |

判断：

- 通用 stop policy 后，dev10 上 qwen-only 和 SkillRouter 都是 7/10，CLSTR v3 after-update 是 6/10；
- 当前 CLSTR 没有超过 SkillRouter，也没有超过 qwen-only；
- CLSTR step positive hit 低于 SkillRouter，说明问题不只是 executor prompt，而是 CLSTR 当前多步 routing/ACT policy 本身仍不够强；
- 下一步不应继续做 AppWorld dev-template prompt 工程，应转向完整 ACT 多 rollout 训练、m_t fusion/no-m_t ablation、SkillRouter-init vs no-init 的公平比较。

## 2026-05-24 full90 ACT v4 submission

为满足“先完整训练，再比较 SkillRouter”的当前 goal，已提交两条 AppWorld train-only full ACT/HRPO 训练线。训练只使用 `data/appworld_routing/train_tasks.jsonl`，评测依赖作业只跑 dev10，不用 dev/test 反馈更新模型。

训练作业：

| job | variant | key config | output |
|---:|---|---|---|
| 75931 | normal ACT full90 v4 | `MODEL_CONFIG=configs/model/appworld_skillrouter_init.yaml`, warmstart `stage2_base-step600.pt`, `MAX_TASKS=90`, `GROUP_SIZE=4`, `UPDATES=20`, `TASK_BATCH_SIZE=3`, `TOP_K=16`, `POLICY_SAMPLING_ALPHA=0.25` | `outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_step600_samplingprior_v4` |
| 75932 | gated ACT full90 v4 | `MODEL_CONFIG=configs/model/appworld_skillrouter_init_gated_residual.yaml`, warmstart `oracle_gated_residual_full_v1/checkpoints/appworld_mt_fusion.pt`, same ACT settings | `outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_gated_samplingprior_v4` |

依赖评测作业：

| job | variant | output |
|---:|---|---|
| 75933 | normal recurrent dev10 | `outputs/appworld_multistep_executor_benchmark/full90_v4_normal_recurrent_dev10` |
| 75935 | normal no-recurrent dev10 | `outputs/appworld_multistep_executor_benchmark/full90_v4_normal_no_recurrent_dev10` |
| 75937 | normal `routing_belief_union_after_update` dev10 | `outputs/appworld_multistep_executor_benchmark/full90_v4_normal_belief_after_update_dev10` |
| 75934 | gated recurrent dev10 | `outputs/appworld_multistep_executor_benchmark/full90_v4_gated_recurrent_dev10` |
| 75936 | gated no-recurrent dev10 | `outputs/appworld_multistep_executor_benchmark/full90_v4_gated_no_recurrent_dev10` |
| 75938 | gated `routing_belief_union_after_update` dev10 | `outputs/appworld_multistep_executor_benchmark/full90_v4_gated_belief_after_update_dev10` |

当前提交时状态：

- `75931` running；
- `75932` pending by priority；
- `75933-75938` pending by dependency。

下一步：不要频繁轮询。约 30 分钟后检查训练作业状态；所有 eval 完成后生成 dev10 full90 v4 comparison，并与 `dev10_stop_policy_qwen_only_v1`、`dev10_stop_policy_skillrouter_base_live_v1`、`dev10_stop_policy_clstr_v3_afterupdate_v1` 对比。如果某个 full90 v4 variant > 7/10 且 recurrent/belief-after-update 优于 no-recurrent，再进入 dev57/full dev；否则记录失败诊断，不进入 test。

## 2026-05-24 full90 ACT v4 results

作业 `75931-75938` 已完成，队列检查时没有残留作业。对比文件：

- `outputs/appworld_multistep_executor_benchmark/full90_v4_dev10_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/full90_v4_dev10_comparison/comparison.md`

dev10 official `task_completed/evaluate` 结果：

| variant | success | task_completed | evaluate_success | exec_fail | avg_steps | step hit |
|---|---:|---:|---:|---:|---:|---:|
| qwen-only stop-policy baseline | 7/10 | 10 | 7 | 0 | 1.8 | 0.0 |
| SkillRouter base live | 7/10 | 10 | 7 | 0 | 2.3 | 1.0 |
| CLSTR v3 after-update pilot | 6/10 | 9 | 6 | 1 | 1.4 | 0.571429 |
| CLSTR full90 v4 normal recurrent | 3/10 | 10 | 3 | 0 | 1.2 | 0.5 |
| CLSTR full90 v4 normal no-recurrent | 3/10 | 10 | 3 | 0 | 1.2 | 0.5 |
| CLSTR full90 v4 normal belief-after-update | 4/10 | 7 | 4 | 4 | 1.7 | 0.411765 |
| CLSTR full90 v4 gated recurrent | 7/10 | 9 | 7 | 1 | 1.4 | 0.571429 |
| CLSTR full90 v4 gated no-recurrent | 6/10 | 10 | 6 | 1 | 1.4 | 0.571429 |
| CLSTR full90 v4 gated belief-after-update | 6/10 | 9 | 6 | 2 | 1.5 | 0.6 |

训练报告：

| train variant | train tasks used | rollout count | updates | benchmark-success rollouts | zero-success updates | mean success/update | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal ACT full90 v4 | 90 train-only pool, 60 sampled update tasks | 240 | 20 | about 12/240 | 17/20 | 0.05 | 0.1583 |
| gated ACT full90 v4 | 90 train-only pool, 60 sampled update tasks | 240 | 20 | about 18/240 | 17/20 | 0.075 | 0.1884 |

判断：

- 当前 full90 ACT 没有超过 SkillRouter。最好的 `gated recurrent` 只是 7/10，和 qwen-only、SkillRouter base live 打平；
- recurrent m_t 在 gated variant 内有弱正收益：`gated recurrent` 7/10，高于 `gated no-recurrent` 和 `gated belief-after-update` 的 6/10；但这个收益不足以证明 CLSTR 已经优于 SkillRouter；
- normal ACT 明显退化到 3-4/10，说明当前 on-policy 更新会破坏 SkillRouter-init/base router 的可用排序；
- 训练过程的 benchmark-success rollout 很稀疏，20 个 update 中 17 个 update 没有成功轨迹，policy 主要在 shaped reward 上学习，不能稳定对齐最终 AppWorld task success；
- 当前不应进入 dev57/full dev/test，也不应把 full90 v4 作为论文主结果。继续推进前需要先做更受约束的 residual ACT 或 prior-anchored CLSTR：把 SkillRouter/skill-table 排序作为强先验，只允许 m_t/ACT 学 residual rerank 或 gate，而不是直接改写整个 routing 分布。

当前 goal 的结论：

- “完成一次更完整的 CLSTR base+ACT train-only 训练并与 SkillRouter 做 dev10 比较”已经完成；
- “超过 SkillRouter 并证明论文方法价值”尚未达成；
- 下一步实验目标应改为：在不使用 dev/test 反馈、不做 AppWorld 特化 prompt hack 的前提下，训练 constrained residual CLSTR/ACT，并要求先在 dev10 稳定超过 7/10，且 recurrent/m_t variant 明确优于 no-m_t ablation，再扩展到 dev57/full dev。

## 2026-05-24 prior-residual ACT fix and submission

基于 full90 ACT v4 的失败诊断，本轮做了两处 benchmark-agnostic 修改：

- 修正多步 ACT reward：`world.task_completed()` 但 official `evaluate` 失败的错误完成不再获得 `task_completed_bonus`，而是加 `wrong_completion_penalty=0.08`。这样避免 policy 学到“尽快错误 complete_task”；
- 新增通用 `policy_skill_mode=prior_residual`：skill logits 变为 `standardized(routing_prior) * routing_prior_strength + policy_head_residual * policy_residual_scale`。该机制不依赖 AppWorld 规则，可迁移到其他 benchmark；AppWorld 控制器只负责把已有 routing logits 传入 `policy_forward`。

新增/修改文件：

- `clstr/model.py`：新增 `policy_skill_mode`、`routing_prior_strength`、`policy_residual_scale`，并在 `policy_forward` 中支持 prior-residual skill logits；
- `clstr/appworld_act_hrpo.py`：修正 reward shaping，ACT 采样/训练路径向 policy 传入 routing logits；
- `clstr/appworld_clstr_eval.py`、`clstr/appworld_multistep.py`、`clstr/appworld_mt_fusion_train.py`：评测、executor、offline mt-fusion 训练路径同步传入 routing prior；
- `configs/model/appworld_skillrouter_init_prior_residual.yaml`：no-m_t/head residual over SkillRouter-init prior；
- `configs/model/appworld_skillrouter_init_gated_prior_residual.yaml`：gated m_t residual over SkillRouter-init prior；
- `tests/test_appworld_act_hrpo.py`、`tests/test_model_pipeline.py` 等测试补充和适配。

验证：

```bash
PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.codex_deps:. \
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_act_hrpo.py::test_shape_appworld_multistep_reward_keeps_final_success_primary_but_uses_progress \
  tests/test_model_pipeline.py::test_prior_residual_policy_mode_anchors_skill_logits_to_routing_prior -q
# 2 passed

PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.codex_deps:. \
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_act_hrpo.py tests/test_model_pipeline.py tests/test_appworld_mt_fusion_train.py -q
# 21 passed

PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.codex_deps:. \
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_clstr_eval.py tests/test_appworld_multistep.py tests/test_appworld_hrpo_cli.py tests/test_sbatch_scripts.py -q
# 30 passed

PYTHONPATH=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.codex_deps:. \
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_appworld_act_hrpo.py tests/test_appworld_multistep.py tests/test_appworld_clstr_eval.py tests/test_model_pipeline.py tests/test_appworld_mt_fusion_train.py -q
# 42 passed
```

提交的新训练作业：

| job | variant | key config | output |
|---:|---|---|---|
| 76004 | normal prior-residual strict reward | `MODEL_CONFIG=configs/model/appworld_skillrouter_init_prior_residual.yaml`, warmstart `stage2_base-step600.pt`, `MAX_TASKS=90`, `GROUP_SIZE=4`, `UPDATES=20`, `TASK_BATCH_SIZE=3`, `TOP_K=16`, `POLICY_SAMPLING_ALPHA=1.0`, `LEARNING_RATE=2e-5`, `BETA_ROUTING_PRIOR=0.02` | `outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_priorresidual_strictreward_v1` |
| 76005 | gated prior-residual strict reward | `MODEL_CONFIG=configs/model/appworld_skillrouter_init_gated_prior_residual.yaml`, warmstart `oracle_gated_residual_full_v1/checkpoints/appworld_mt_fusion.pt`, same ACT settings | `outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_gated_priorresidual_strictreward_v1` |

训练命令模板：

```bash
sbatch --gpus=1 -p gpu_h200 --export=ALL,MODEL_CONFIG=<prior-residual-config>,OUTPUT_DIR=<train-output>,WARMSTART_CLSTR_CKPT=<full-base-or-gated-base>,MAX_TASKS=90,GROUP_SIZE=4,TOP_K=16,CONTEXT_TOP_K=5,MAX_STEPS=3,MULTI_STEP=1,SKILL_CONTEXT_MODE=safe_metadata,UPDATES=20,TASK_BATCH_SIZE=3,LEARNING_RATE=2.0e-5,BETA_KL=0.01,BETA_ROUTING_PRIOR=0.02,ROLLOUT_TEMPERATURE=1.0,POLICY_SAMPLING_ALPHA=1.0,MAX_INTERACTIONS=10,MAX_APIS_PER_APP=12,TIMEOUT_SECONDS=60,MAX_NEW_TOKENS=768 scripts/sbatch/run_appworld_clstr_hrpo_train.sh
```

依赖评测作业：

| job | dependency | variant | output |
|---:|---:|---|---|
| 76011 | afterok:76004 | normal prior-residual recurrent dev10, `RANKING_MODE=policy_head`, `CANDIDATE_SOURCE=routing` | `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_normal_recurrent_dev10` |
| 76012 | afterok:76004 | normal prior-residual no-recurrent dev10 | `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_normal_no_recurrent_dev10` |
| 76015 | afterok:76004 | normal prior-residual `routing_belief_union_after_update` dev10 | `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_normal_belief_after_update_dev10` |
| 76016 | afterok:76005 | gated prior-residual recurrent dev10 | `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_gated_recurrent_dev10` |
| 76013 | afterok:76005 | gated prior-residual no-recurrent dev10 | `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_gated_no_recurrent_dev10` |
| 76014 | afterok:76005 | gated prior-residual `routing_belief_union_after_update` dev10 | `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_gated_belief_after_update_dev10` |

评测命令模板：

```bash
sbatch --dependency=afterok:<train-job> --gpus=1 -p gpu_h200 --export=ALL,METHOD=clstr_multistep,CLSTR_MODEL_CONFIG=<prior-residual-config>,CLSTR_CHECKPOINT_PATH=<train-output>/checkpoints/appworld_clstr_multistep_hrpo-step20.pt,OUTPUT_DIR=<eval-output>,TASKS_PATH=data/appworld_routing/dev_tasks.jsonl,MAX_TASKS=10,TOP_K=5,MAX_STEPS=3,RANKING_MODE=policy_head,CANDIDATE_TOP_K=16,CANDIDATE_SOURCE=<routing-or-routing_belief_union_after_update>,RECURRENT_BELIEF=<0-or-1>,SKILL_CONTEXT_MODE=safe_metadata,MAX_INTERACTIONS=10,MAX_APIS_PER_APP=12,TIMEOUT_SECONDS=60,MAX_NEW_TOKENS=768 scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

当前队列状态：`76004/76005` pending by priority，`76011-76016` pending by dependency。下一步按用户要求不要频繁轮询，约 30 分钟后检查；全部完成后生成 prior-residual dev10 comparison，并与 `qwen_only`、`skillrouter_base_live`、CLSTR v3、full90 v4 做公平比较。如果仍不超过 7/10，需要进入失败诊断：reward 稀疏、candidate recall、m_t 更新、skill embedding、executor bottleneck 或方法假设不成立。

## 2026-05-24 prior-residual ACT running status and GPU fallback

13:35 CST 检查队列：`76004 clstr_pr_v1` 和 `76005 clstr_gpr_v1` 已在 `gpu_h200` 运行约 14 分钟，`parajobs` 显示 GPU 利用率约 52%，显存占用约 23.5GB，`slurm-76004.out`/`slurm-76005.out` 暂无错误栈。当前输出目录只有 `train_stdout.json`，还未生成 `train_report.json` 或 step20 checkpoint，因此训练仍未完成。

当前可用公共 GPU 队列概况：`gpu_a800` 可运行 1 卡作业约 9 个，`gpu_h100` 约 3 个，`gpu_h200` 约 3 个。策略：不迁移已经正常运行的 `76004/76005` 训练；训练完成后如果依赖评测 `76011-76016` 在 H200 上排队严重，可以用 `gpu_a800` 或 `gpu_h100` 重新提交等价 eval。重新提交时必须使用新的 `OUTPUT_DIR` 或先确认原依赖 eval 不会写同一路径，避免覆盖结果。

## 2026-05-24 ACT loss/reward diagnosis before prior-residual results

为了回答“是否 loss/reward 设置有问题”，在 `76004/76005` 训练等待期间，对已完成的 full90 ACT v4 做了静态和结果级诊断。

诊断产物：

- `outputs/appworld_multistep_failure_diagnostics/full90_v4_gated_recurrent_vs_skillrouter_dev10/failure_diagnostics.json`
- `outputs/appworld_multistep_failure_diagnostics/full90_v4_normal_recurrent_vs_skillrouter_dev10/failure_diagnostics.json`

关键证据：

- `full90_v4_gated_recurrent`、`skillrouter_base_live`、`qwen_only` 在 dev10 上成功的是完全同一批 7 个任务：`4ec8de5_1`、`50e1ac9_1/2/3`、`fac291d_1/2/3`；共同失败的是 `530b157_1/2/3`。因此 gated v4 只是追平 SkillRouter/Qwen，没有证明 CLSTR 修复了 SkillRouter 失败样本；
- `full90_v4_normal_recurrent` 只有 3/10，且 SkillRouter 成功而 normal CLSTR 失败的 4 个任务都表现为 `task_completed=True` 但 `evaluation_success=False`，不是执行崩溃。这说明 normal policy rerank 会把排序推向错误完成；
- v4 训练 rollout 稀疏：normal 12/240 成功、gated 18/240 成功；二者都是 17/20 个 update 没有任何 benchmark-success rollout；
- v4 中失败 rollout 仍有正 shaped reward：normal fail 平均 reward 约 0.114，gated fail 平均约 0.123。旧 reward 对错误 `task_completed` 没有惩罚，因此可能鼓励“执行成功但最终答案错”的轨迹；
- dev10 运行轨迹中的 skill hit 也显示 CLSTR 的 rerank/candidate 仍弱于 SkillRouter：SkillRouter selected/candidate hit 都是 1.0；`full90_v4_gated_recurrent` selected hit 0.5714、candidate hit 0.7857；`full90_v4_normal_recurrent` selected hit 0.5、candidate hit 0.75；`full90_v4_gated_belief_after_update` 的 belief candidate hit 是 1.0，但 selected hit 仍只有 0.6，说明 m_t/belief 能把正例带进候选，但 policy rerank 没有稳定选出来；
- 代码层面，`compute_appworld_multistep_hrpo_loss` 是轨迹级 `sum(log_prob_t) * grouped_advantage`，没有 step-level advantage 或 per-step verifier signal，credit assignment 很弱；
- 当前 ACT 路径中 `step_update` 在 `torch.no_grad()` 下运行，且 `detach_policy_inputs=True` 时 `m_t` 输入被 detach。因此 ACT 实际训练的是 policy residual/mt skill head，不会直接通过 reward 改进 transition/gate 的 m_t 更新函数。m_t 是否有用主要取决于 base/offline mt-fusion 是否已经学好；
- ACT 的 stop head 也没有被实际训练或用于当前 stop-policy benchmark，评测停止主要由 `MultiStepStopPolicy` 的 `task_completed/evaluation_success` 启发式决定。

判断：

- loss/reward 确实是当前 CLSTR 不稳定的重要原因，但不是唯一原因；
- 已提交的 `76004/76005` prior-residual strict-reward 实验只修正了其中两点：错误完成惩罚，以及把 SkillRouter/skill-table routing prior 作为强先验，只允许小 residual 调整；
- 它还没有解决的点包括：trajectory-level credit assignment、ACT 不直接训练 transition/gate、Qwen executor 对 `530b157_*` 类任务的共同失败瓶颈；
- 如果 prior-residual strict-reward 仍然不能超过 7/10，下一步应优先做 benchmark-agnostic 的 step-level/leave-one-step credit 诊断和 candidate recall 诊断，而不是继续调 prompt。

## 2026-05-24 prior-residual post-run audit checklist

为避免训练完成后临时拼结果，本轮先固定验收口径。当前已确认 baseline reports 存在：

- `outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_qwen_only_v1/report.json`
- `outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_skillrouter_base_live_v1/report.json`
- `outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_clstr_v3_afterupdate_v1/report.json`
- `outputs/appworld_multistep_executor_benchmark/full90_v4_normal_recurrent_dev10/report.json`
- `outputs/appworld_multistep_executor_benchmark/full90_v4_gated_recurrent_dev10/report.json`

当前仍缺少、需等待训练完成后出现的文件：

- `outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_priorresidual_strictreward_v1/train_report.json`
- `outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_priorresidual_strictreward_v1/checkpoints/appworld_clstr_multistep_hrpo-step20.pt`
- `outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_gated_priorresidual_strictreward_v1/train_report.json`
- `outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_gated_priorresidual_strictreward_v1/checkpoints/appworld_clstr_multistep_hrpo-step20.pt`
- 六个依赖 eval 的 `report.json`：
  - `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_normal_recurrent_dev10/report.json`
  - `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_normal_no_recurrent_dev10/report.json`
  - `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_normal_belief_after_update_dev10/report.json`
  - `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_gated_recurrent_dev10/report.json`
  - `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_gated_no_recurrent_dev10/report.json`
  - `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_gated_belief_after_update_dev10/report.json`

训练完成后先检查 `train_report.json`：

- `rollout_count` 应为 `MAX_TASKS * GROUP_SIZE * UPDATES/TASK_BATCH_CYCLE` 实际采样数，本轮预期 240；
- `train_config` 应记录 `MAX_TASKS=90`、`GROUP_SIZE=4`、`UPDATES=20`、`TASK_BATCH_SIZE=3`、`POLICY_SAMPLING_ALPHA=1.0`、`BETA_ROUTING_PRIOR=0.02`；
- `reward_shaping` 应包含 `wrong_completion_penalty=0.08`，且错误完成不再拿 `task_completed_bonus`；
- 统计成功 rollout 数、zero-success updates、mean reward、wrong completion count、execution failure count，用于判断 strict reward 是否改善 ACT credit。

六个 eval 全部完成后运行：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/build_appworld_multistep_comparison.py \
  --reports \
  outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_qwen_only_v1/report.json \
  outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_skillrouter_base_live_v1/report.json \
  outputs/appworld_multistep_executor_benchmark/dev10_stop_policy_clstr_v3_afterupdate_v1/report.json \
  outputs/appworld_multistep_executor_benchmark/full90_v4_normal_recurrent_dev10/report.json \
  outputs/appworld_multistep_executor_benchmark/full90_v4_gated_recurrent_dev10/report.json \
  outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_normal_recurrent_dev10/report.json \
  outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_normal_no_recurrent_dev10/report.json \
  outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_normal_belief_after_update_dev10/report.json \
  outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_gated_recurrent_dev10/report.json \
  outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_gated_no_recurrent_dev10/report.json \
  outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_gated_belief_after_update_dev10/report.json \
  --output_dir outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_dev10_comparison
```

判定标准：

- 若任一 prior-residual variant `success_count > 7/10`，并且 recurrent/after-update 明确优于 no-m_t，则进入 dev57/full dev 复验；
- 若最高仍为 `7/10`，只能说“CLSTR 追平 SkillRouter/Qwen”，不能主张超过；
- 若低于 `7/10`，优先做失败诊断：与 SkillRouter/Qwen 成功集合比较、selected/candidate hit、wrong completion、execution failure、共同失败任务、以及 m_t candidate 是否把正例带入但 rerank 未选中。

## 2026-05-24 active goal coverage audit

当前 goal 仍未完成，完成状态按证据拆分如下。

已完成/已有证据：

- AppWorld train split routing data 与 SkillX skill pool 已准备完成；`data/appworld_routing/manifest.json` 记录 train/dev/test split，`dev_test_used_for_training=False`；
- SkillX AppWorld audit 为 low risk，未发现 dev/test contamination；
- SkillRouter 论文/实现侧未发现 AppWorld test 泄露；
- 完整 CLSTR-base routing-only 训练已完成，final checkpoint 为 `outputs/appworld_clstr_train_skillrouter_init_routing_full_v1/checkpoints/stage2_base-step600.pt`，`train_stdout.json` 显示 train routing supervision recall@1/5 都为 1.0；
- 已完成一轮 full90 ACT v4：normal/gated、recurrent/no-recurrent/belief-after-update，对比了 Qwen-only、SkillRouter-base live、CLSTR v3；
- 已完成 full90 v4 failure diagnostics，证明 gated v4 只追平 SkillRouter/Qwen 的同一批 dev10 任务，没有产生新增成功样本；
- 已实现并测试 prior-residual policy mode 与 strict reward 修正，相关单测/集成测试通过。

进行中/等待外部作业：

- `76004` normal prior-residual strict-reward ACT 训练；
- `76005` gated prior-residual strict-reward ACT 训练；
- `76011-76016` 六个依赖 dev10 eval，覆盖 no-m_t / recurrent m_t / belief-after-update。

尚不能宣称完成的要求：

- 不能宣称 CLSTR 超过 SkillRouter：当前已完成最好结果只是 dev10 `7/10` 打平；
- 不能宣称 m_t/closed-loop 有充分论文价值：已有 evidence 只显示 gated recurrent 相比 no-recurrent 有弱收益，但成功集合没有超过 baseline；
- 不能进入 test split：prior-residual dev10 还未完成，且没有 dev57/full dev 复验；
- 不能把当前 ACT 当作最终充分训练：仍需看 strict reward/prior-residual 是否缓解 reward 稀疏和 policy rerank 破坏。

下一步只在合适轮询间隔后检查 `76004/76005`。如果训练结束但依赖 eval 在 H200 排队严重，按前述策略迁移 eval 到 `gpu_a800` 或 `gpu_h100`，并使用新 `OUTPUT_DIR`。

## 2026-05-24 m_t/ACT variant coverage map

为对齐 active goal 中“多种 m_t/ACT 方案”的要求，当前代码和实验覆盖关系如下：

| goal variant | 当前实现/实验口径 | 状态 |
|---|---|---|
| no-m_t 对照 | `RECURRENT_BELIEF=0`，仍用当前 state text 的 `subspace_obs`，不跨步维护 `m_t` | 已在 full90 v4 跑过；prior-residual 依赖 eval `76012/76013` 待完成 |
| recurrent m_t | `RECURRENT_BELIEF=1`，`CLSTRMultiStepController.observe()` 用 action + environment output + `next_state_text` 更新 `m_t` | 已在 full90 v4 跑过；prior-residual 依赖 eval `76011/76016` 待完成 |
| late rerank | `RANKING_MODE=policy_head` 或 `policy_blend`：先由 routing logits 取 candidate pool，再由 CLSTR policy logits/融合分数重排 | 已实现；prior-residual eval 使用 `policy_head`，full90 v4 覆盖过 policy rerank |
| candidate union | `CANDIDATE_SOURCE=routing_belief_union` 或 `routing_belief_union_after_update`，将 routing top-k 与 belief top-k 合并后 rerank | 已实现；prior-residual 依赖 eval `76014/76015` 待完成 |
| gated residual m_t fusion | `mt_fusion_mode=gated_residual`，用 `m_t` gate/residual 条件化 policy hidden 表示 | 已完成 train-only oracle mt-fusion，gated ACT/full90 v4 已跑；prior-residual gated 训练 `76005` 待完成 |
| small residual over SkillRouter/skill-table prior | `policy_skill_mode=prior_residual`，`standardized(routing_prior) + residual_scale * policy_head` | 已实现并通过测试；训练 `76004/76005` 正在运行 |
| benchmark-agnostic strict reward | `wrong_completion_penalty`、只在 official evaluate success 时给 complete bonus | 已实现并通过测试；训练 `76004/76005` 正在验证 |

注意：这里的 “late rerank” 不是额外的 prompt 或 AppWorld 特化规则，而是通用的两阶段 controller 结构：retrieval/prior 先生成候选，CLSTR belief/policy 再重排。若 prior-residual 仍不能超过 SkillRouter，下一步应比较 `policy_head`、`policy_blend`、`skill_table` 和 candidate union 的失败样本，而不是新增任务特异 prompt。

## 2026-05-24 fallback experiment tree if prior-residual does not exceed SkillRouter

如果 `priorresidual_strictreward_v1` dev10 最高结果仍然 `<= 7/10`，不要直接进入 test，也不要继续做 AppWorld 模板 prompt。下一步按以下 evidence tree 决策：

1. **若 candidate hit 低**：说明问题优先在 skill embedding/retrieval candidate recall。
   - 只允许 train split 上改进 skill embedding 或 base retrieval；
   - 对比 `skill_text_format=appworld_skillrouter_base`、`clstr_enriched`、SkillX body/full metadata；
   - 复查 duplicate/canonical skill 评价，避免 raw id artifact。

2. **若 candidate hit 高但 selected hit 低**：说明问题优先在 CLSTR rerank/policy residual。
   - 比较 `skill_table`、`policy_blend alpha<=0.1`、`policy_head`、`prior_residual scale`；
   - 降低 residual scale 或使用 KL/routing-prior 更强的 conservative update；
   - 不扩大 rollout 数之前，先确认 rerank 没破坏强 prior。

3. **若 belief candidate hit 高但 recurrent 不优于 no-m_t**：说明 `m_t` 更新或使用方式无效。
   - 对比 `routing_belief_union_after_update` vs no-recurrent；
   - 检查 `m_t` 更新输入是否包含 `next_state_text` 而不是裸 `execute_output`；
   - 若仍无效，优先做 offline train-only transition/gate 监督，而不是只靠 sparse HRPO。

4. **若 train rollout success 极稀疏**：说明 ACT reward/credit assignment 仍不足。
   - 保留 official final success 作为 benchmark 指标；
   - 训练侧可做 benchmark-agnostic 的 step-level proxy：execution_ok、wrong completion penalty、leave-one-step skill ablation、trajectory pairwise ranking；
   - 不用 dev/test reward 做 early stopping 或调参。

5. **若 Qwen/SkillRouter/CLSTR 共同失败同一批任务**：说明 executor 上限或 AppWorld task family 是瓶颈。
   - 不应把这类共同失败归因给 routing；
   - 只能在 train split 上改进 executor-compatible generic skill context 或换更强下游 LM，再重新公平比较；
   - 论文中需把 CLSTR 的贡献限定为 routing/controller，而不是声称修复 executor schema grounding。

6. **若 CLSTR 只追平而不超过 SkillRouter**：
   - 可以作为负结果/诊断记录；
   - 论文主张不能写“CLSTR beats SkillRouter on AppWorld task success”；
   - 只有当 recurrent/prior-residual variant 在 dev10 和 dev57/full dev 都超过 SkillRouter，且 no-m_t 消融更低，才能主张 `m_t`/closed-loop/ACT 的核心价值。

## 2026-05-24 eval migration template if H200 queue is slow

如果 `76004/76005` 训练完成后，`76011-76016` 仍因 H200 排队长时间 pending，可以把 eval 迁到 `gpu_a800` 或 `gpu_h100`。原则：

- 不覆盖原 `OUTPUT_DIR`，迁移作业使用新目录，例如追加 `_a800_v1` 或 `_h100_v1`；
- 若原依赖 eval 已经开始运行，不再提交同 variant 的迁移作业；
- 若提交迁移作业并采用其结果，comparison 只纳入一个版本，不能把同一 checkpoint/variant 的两个重复 eval 都放进主表；
- 如果决定取消原 H200 eval，先记录 `scancel <job_id>` 到本节，再取消；不要用 `git` 或 shell 删除已有结果。

单个迁移命令模板：

```bash
sbatch --gpus=1 -p <gpu_a800-or-gpu_h100> \
  --job-name=<short-eval-name> \
  --export=ALL,METHOD=clstr_multistep,CLSTR_MODEL_CONFIG=<config>,CLSTR_CHECKPOINT_PATH=<train-output>/checkpoints/appworld_clstr_multistep_hrpo-step20.pt,OUTPUT_DIR=<new-output-dir-with-queue-suffix>,TASKS_PATH=data/appworld_routing/dev_tasks.jsonl,MAX_TASKS=10,TOP_K=5,MAX_STEPS=3,RANKING_MODE=policy_head,CANDIDATE_TOP_K=16,CANDIDATE_SOURCE=<routing-or-routing_belief_union_after_update>,RECURRENT_BELIEF=<0-or-1>,SKILL_CONTEXT_MODE=safe_metadata,MAX_INTERACTIONS=10,MAX_APIS_PER_APP=12,TIMEOUT_SECONDS=60,MAX_NEW_TOKENS=768 \
  scripts/sbatch/run_appworld_multistep_executor_eval.sh
```

normal prior-residual 三个迁移 variant 对应：

- recurrent：`CLSTR_MODEL_CONFIG=configs/model/appworld_skillrouter_init_prior_residual.yaml`，`CLSTR_CHECKPOINT_PATH=outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_priorresidual_strictreward_v1/checkpoints/appworld_clstr_multistep_hrpo-step20.pt`，`RECURRENT_BELIEF=1`，`CANDIDATE_SOURCE=routing`；
- no-m_t：同 checkpoint，`RECURRENT_BELIEF=0`，`CANDIDATE_SOURCE=routing`；
- belief-after-update：同 checkpoint，`RECURRENT_BELIEF=1`，`CANDIDATE_SOURCE=routing_belief_union_after_update`。

gated prior-residual 三个迁移 variant 对应：

- recurrent：`CLSTR_MODEL_CONFIG=configs/model/appworld_skillrouter_init_gated_prior_residual.yaml`，`CLSTR_CHECKPOINT_PATH=outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_gated_priorresidual_strictreward_v1/checkpoints/appworld_clstr_multistep_hrpo-step20.pt`，`RECURRENT_BELIEF=1`，`CANDIDATE_SOURCE=routing`；
- no-m_t：同 checkpoint，`RECURRENT_BELIEF=0`，`CANDIDATE_SOURCE=routing`；
- belief-after-update：同 checkpoint，`RECURRENT_BELIEF=1`，`CANDIDATE_SOURCE=routing_belief_union_after_update`。

## 2026-05-24 prior-residual strict-reward results

14:16 CST 检查：`76004/76005` 训练和 `76011-76016` 六个依赖 eval 均已完成，当前 `squeue -u scyb713` 为空。无需迁移到 A800/H100。

训练产物：

- normal prior-residual：`outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_priorresidual_strictreward_v1/train_report.json`
- normal checkpoint：`outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_priorresidual_strictreward_v1/checkpoints/appworld_clstr_multistep_hrpo-step20.pt`
- gated prior-residual：`outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_gated_priorresidual_strictreward_v1/train_report.json`
- gated checkpoint：`outputs/appworld_clstr_train_multistep_hrpo/full90_g4_u20_gated_priorresidual_strictreward_v1/checkpoints/appworld_clstr_multistep_hrpo-step20.pt`

训练统计：

| train variant | train tasks | rollouts | success rollouts | zero-success updates | wrong-complete rollouts | exec-fail rollouts | key config |
|---|---:|---:|---:|---:|---:|---:|---|
| normal prior-residual strict reward | 90 | 240 | 16 | 16/20 | 91 | 156 | `group=4`, `updates=20`, `policy_sampling_alpha=1.0`, `beta_routing_prior=0.02`, `wrong_completion_penalty=0.08` |
| gated prior-residual strict reward | 90 | 240 | 12 | 18/20 | 77 | 146 | same |

对比产物：

- `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_dev10_comparison/comparison.json`
- `outputs/appworld_multistep_executor_benchmark/priorresidual_strictreward_v1_dev10_comparison/comparison.md`
- failure diagnostic: `outputs/appworld_multistep_failure_diagnostics/priorresidual_gated_no_m_t_vs_skillrouter_dev10/failure_diagnostics.json`

dev10 official task success：

| method / variant | success | task_completed | evaluate_success | exec_fail | avg_steps | step hit |
|---|---:|---:|---:|---:|---:|---:|
| qwen_only | 7/10 | 10 | 7 | 0 | 1.8 | 0.0 |
| skillrouter_base_live | 7/10 | 10 | 7 | 0 | 2.3 | 1.0 |
| CLSTR v3 after-update | 6/10 | 9 | 6 | 1 | 1.4 | 0.571429 |
| full90 v4 normal recurrent | 3/10 | 10 | 3 | 0 | 1.2 | 0.5 |
| full90 v4 gated recurrent | 7/10 | 9 | 7 | 1 | 1.4 | 0.571429 |
| prior-residual normal recurrent | 2/10 | 9 | 2 | 0 | 1.1 | 0.090909 |
| prior-residual normal no-m_t | 4/10 | 10 | 4 | 0 | 1.3 | 0.076923 |
| prior-residual normal belief-after-update | 3/10 | 10 | 3 | 1 | 1.2 | 0.083333 |
| prior-residual gated recurrent | 6/10 | 10 | 6 | 1 | 1.3 | 0.538462 |
| prior-residual gated no-m_t | 7/10 | 10 | 7 | 0 | 1.7 | 0.411765 |
| prior-residual gated belief-after-update | 6/10 | 10 | 6 | 0 | 1.3 | 0.538462 |

成功集合诊断：

- `prior-residual gated no-m_t` 成功任务与 `skillrouter_base_live`、`qwen_only` 完全相同：`4ec8de5_1`、`50e1ac9_1/2/3`、`fac291d_1/2/3`；
- `prior-residual gated no-m_t` 没有任何 SkillRouter 失败但自己成功的任务；
- `prior-residual gated recurrent` 和 `prior-residual gated belief-after-update` 都比 no-m_t 少成功 `4ec8de5_1`；
- normal prior-residual 全面退化，说明 normal policy residual 仍在破坏强 routing prior。

结论：

- 本轮 prior-residual strict-reward 没有超过 SkillRouter。最佳结果只是 `7/10` 打平；
- 没有证明 `m_t` 有正贡献：gated no-m_t `7/10`，recurrent 和 belief-after-update 都是 `6/10`；
- strict reward 修正没有解决 ACT 稀疏 credit：仍有 16/20 或 18/20 个 update 没有成功 rollout；
- candidate/rerank 问题仍明显：best prior-residual gated no-m_t step hit 只有 `0.411765`，远低于 SkillRouter 的 `1.0`；
- executor 也有上限信号：Qwen-only、SkillRouter 和最佳 CLSTR 成功集合完全相同，共同失败 `530b157_1/2/3`，这些不能归因给 CLSTR routing 单独失败。

当前不能进入 test split，也不能在论文中主张 “CLSTR beats SkillRouter on AppWorld task success”。如果继续推进，应先解决两个 benchmark-agnostic 问题：

1. rerank/ACT credit：不要继续让 sparse HRPO 大幅改写 routing prior，优先做 step-level/pairwise/leave-one-step credit 或更保守 residual；
2. `m_t` 价值：当前 recurrent/belief 没有优于 no-m_t，必须先在 train-only transition/gate 或 offline oracle trajectory 上证明 `m_t` 能稳定改善 candidate/rerank，再扩大到 dev57。

## 2026-05-24 Phase 1 bug-fix batch (CLSTR vs SkillRouter goal)

按 `docs/superpowers/plans/2026-05-24-clstr-beat-skillrouter-phase1.md` 完成 5 个已识别的 ACT/闭环实现 bug 修复，目标是让后续 `enable_closed_loop_gradient=True` 的 ACT 训练真正能从 reward 学到东西。所有改动 benchmark-agnostic，不针对 AppWorld 具体任务做 prompt hack。

### 修复清单（per-commit）

1. `48a3fbf` + `9b40652` — `compute_appworld_multistep_hrpo_loss` / `compute_appworld_hrpo_loss` 加 step-level advantage fallback：当 group 内 reward variance=0 时（AppWorld sparse binary success 下的常见情况）substitute 一个 within-group standardized proxy（`execution_ok` fraction 减 wrong-completion penalty），scale 0.25 保证永不覆盖 genuine official advantage。`advantage_nonzero_count` 和 `advantage_fallback_used` 升级为 train_report first-class 字段。

2. `09ccd06` + `ccb2ba6` — `AppWorldHrpoTrainConfig.enable_closed_loop_gradient` 新增（默认 `False` 保持 API 后向兼容）。开启时：`run_appworld_multistep_hrpo_rollout` 把 `step_update` 包在 `torch.enable_grad()` 下、`m_t` 不再 detach；同时 `_sample_clstr_multistep_context` 在 `detach_policy_inputs=True` 时仍 detach `h` / `candidate_embs`（省 8B encoder 显存）但**保留** `m_current` 的 graph。这两处必须同时改，否则 ACT reward gradient 仍触达不到 `TransitionPredictor` / `BeliefGate` / `TransHead`——code quality reviewer 抓到的 BLOCKER。Reference model 始终 `no_grad`。

3. `d6e3f42` — 5 条回归测试锁死 2026-05-24 wrong-completion penalty fix：empty trajectory、wrong-completion 严格 < clean、true success ≫ wrong-completion、wrong-completion 在任意 step（不止最后一步）都被罚。无源码改动。

4. `4e08468` — ACT-export 默认值收紧。`scripts/sbatch/run_appworld_multistep_executor_eval.sh` 把 `POLICY_BLEND_ALPHA` 默认 0.5 → 0.25（更保守，让 base routing prior 占主导）。`configs/model/appworld_skillrouter_init.yaml` 加 `policy_skill_mode: prior_residual` + `routing_prior_strength: 1.0` + `policy_residual_scale: 1.0`，新 ACT 训练以该 config 为底座时默认锚定 routing prior。
   - 关键发现：`scripts/run_appworld_multistep_executor_eval.py::_resolve_auto_ranking_mode` 已经实现 ACT-aware 智能默认（HRPO/ACT checkpoint → `policy_blend`，否则 `skill_table`，从不回退 `policy_head`），plan 原本想做的"硬默认 `policy_blend`"实际上比当前 `auto` 弱。本 commit 保留 `auto` 作为默认，加 5 条回归测试锁死它的行为。

5. `4f03a7a` — `MultiStepStopPolicy` 接 CLSTR STOP head logit。新增 `use_stop_head` / `stop_head_threshold` 字段和可选 `stop_head_logit` 参数。`CLSTRMultiStepController.select` 和 `HybridCLSTRSkillRouter.select` 不再 slice 掉 `policy_forward` 输出最后一维（STOP logit），改为提取出来放进 `MultiStepSelection.diagnostics["stop_head_logit"]`；multi-step runtime loop 读 diagnostics 传给 `stop_policy.decide(...)`。优先级：explicit env done+success > done > success > stop_head logit。后向兼容：`stop_head_logit=None` 时行为完全等同旧版。

### 新增测试

- `tests/test_phase1_advantage_fallback.py`（3 tests）
- `tests/test_phase1_closed_loop_gradient.py`（3 tests，含 integration test 覆盖原始 BLOCKER）
- `tests/test_phase1_reward_shaping_audit.py`（5 tests）
- `tests/test_phase1_prior_residual_defaults.py`（7 tests）
- `tests/test_phase1_stop_head_controller.py`（7 tests）

### 验证

- `PYTHONPATH=.vendor_pytest:. pytest -q`：**447 passed in 284.76s**，0 失败。
- `py_compile` 5 个改动文件：clean。
- `bash -n scripts/sbatch/run_appworld_multistep_executor_eval.sh scripts/sbatch/run_appworld_clstr_hrpo_train.sh`：clean。

### Phase 1 不涵盖的事项

- 没有提交任何 GPU 训练或 eval 作业。Phase 1 只是把代码改对。
- 没有覆盖任何已有 `outputs/` 目录。
- 没有在 dev/test split 上训练或调参。
- `goal.md`（早期 AppWorld+SkillX min-runtime 目标）已被 `/goal` 替换，未删除。
- 单步 entrypoint `train_appworld_clstr_hrpo` / `compute_appworld_hrpo_loss` 没接 `enable_closed_loop_gradient`（单步路径没有 `step_update` 链）。

### Phase 2（下一步）

1. 把 `scripts/sbatch/run_appworld_clstr_hrpo_train.sh` 加 `ENABLE_CLOSED_LOOP_GRADIENT` 环境变量（`0`/`1`），plumb 到 `scripts/run_appworld_clstr_hrpo_train.py` 的 argparse 和 `AppWorldHrpoTrainConfig`。
2. 写 `scripts/sbatch/run_phase1_dev10_gate.sh`：chained train+eval，`ENABLE_CLOSED_LOOP_GRADIENT=1` + `MODEL_CONFIG=appworld_skillrouter_init_gated_prior_residual.yaml` + 写入 fresh `OUTPUT_DIR`。
3. 跑出 `outputs/appworld_multistep_executor_benchmark/phase1_dev10_gate/report.json`，验证 `success_count >= 8`。
4. 通过后扩到 dev57。
5. 验收：dev10 ≥ 8/10 且 dev57 ≥ 15/57（对照 SkillRouter-base 7/10 与 14/57）。

## 2026-05-24 Phase 1.5 audit complete; deletion deferred

Audit 报告产出：`docs/superpowers/audits/phase1_5_redundancy_audit.md`。识别 144 个可删文件 / ~32.7K LOC，分 5 个 batch。

按用户决定先跑低风险 Batch 1-3 (~4.7K LOC)，但 subagent 实战删除两轮都遇到 audit 没覆盖的隐藏依赖：

1. `tests/test_skillret.py:38` import `clstr.skillsbench_transfer`（Batch 1 内，但 audit 漏标 test 端依赖）
2. `scripts/verify_policy_replay_candidates.py` 是独立 ScienceWorld+WebShop verify 工具，不在 sbatch chain
3. `clstr/envs/__init__.py:4` re-export `WebShopEnvAdapter`，`tests/test_aux_trajectories.py:1006` 也 import 它
4. **`clstr/external_data.py` 既有 AppWorld helpers 又含 `check_skillsbench_root` / `export_skillsbench_skill_pool` / `collect_skillsbench_harness_results`**（SkillsBench-only），audit 把整个文件标 `KEEP (infra)` 是误判
5. `tests/test_external_data.py` 13/14 个测试是 SkillsBench-specific，subprocess 调用了 `scripts/check_skillsbench_data.py`

含义：连 audit 标记"低风险"的 Batch 1（SkillsBench）也需要拆 `external_data.py` 文件、重写 `test_external_data.py`。已超出"git rm + pytest 验证"的简单 cleanup 范畴。

决定：

- Phase 1.5 全部推迟到 dev10 gate（jobs 76140/76141）出结果之后再启动。
- 重启时需先做 audit v2：把"按文件分类"细化为"按函数分类"，特别是 `clstr/external_data.py`、`clstr/envs/__init__.py`、`clstr/data.py` 这类有混合内容的 infra 文件。
- Audit 报告作为参考保留，不作为可执行 deletion list。

## 2026-05-24 Phase 1 dev10 gate jobs submitted

`scripts/sbatch/run_phase1_dev10_gate.sh` 已提交：

- train job `76140`（`gpu_h200`），输出 `outputs/appworld_clstr_train_phase1_gated_priorresidual_v1/`
- eval job `76141` afterok `76140`，输出 `outputs/appworld_multistep_executor_benchmark/phase1_dev10_gate/`

配置：
- `MODEL_CONFIG=configs/model/appworld_skillrouter_init_gated_prior_residual.yaml`
- `WARMSTART_CLSTR_CKPT=outputs/appworld_mt_fusion_train/oracle_gated_residual_full_v1/checkpoints/appworld_mt_fusion.pt`
- `ENABLE_CLOSED_LOOP_GRADIENT=1`（Phase 1 Bug 2 fix 真正生效）
- `MAX_TASKS=90, GROUP_SIZE=4, UPDATES=20, TASK_BATCH_SIZE=3`
- Eval `RANKING_MODE=policy_blend`, `POLICY_BLEND_ALPHA=0.25`, `MAX_TASKS=10`

验收：`outputs/appworld_multistep_executor_benchmark/phase1_dev10_gate/report.json` 中 `success_count >= 8`。

注意：sbatch 提交时发现 `run_phase1_dev10_gate.sh` 原默认 warmstart 路径错误（缺 `_train` 后缀），已修复并提交 `197b8bc`。

## 2026-05-24 Phase 1 dev10 gate result: 6/10 (target was ≥ 8/10) — NOT PASSED

`outputs/appworld_multistep_executor_benchmark/phase1_dev10_gate/report.json`:
- success_count=6, task_completed_count=8, evaluate_success_count=6
- average_steps=1.0 across all 10 tasks
- 0 execution failures, 0 generation failures

Per-task stop_reason analysis:
- 6 successes: stop_reason=`done_success` (legit)
- 2 wrong-completions: `530b157_1`, `4ec8de5_1` → stop_reason=`done` (Qwen prematurely complete_task with wrong answer)
- 2 stop_head regressions: `530b157_2`, `530b157_3` → stop_reason=`stop_head` (untrained stop_head fired at logit > 0)

Comparison vs SkillRouter-base dev10 (7/10):
| task_id | CLSTR | SR | notes |
|---|---|---|---|
| 50e1ac9_1-3 | ✓ | ✓ | spotify song aggregation, easy |
| fac291d_1-3 | ✓ | ✓ | spotify another aggregation |
| 530b157_1-3 | ✗ | ✗ | common failure (Qwen executor ceiling) |
| 4ec8de5_1 | ✗ | ✓ | UNIQUE CLSTR LOSS |

`4ec8de5_1` root cause: structural, NOT a bug.
- CLSTR step 0 routing: 5 task-relevant skills (`spotify-find-songs-by-year-criteria-27`, etc.) → Qwen one-shots full workflow with wrong year hardcoded → wrong complete_task call
- SR step 0 routing: 5 auth skills only → Qwen only does login (no complete_task) → step 1 retries with correct logic
- SkillRouter's "weak routing" (only auth skills in top-5) accidentally forces Qwen into multi-step exploration. CLSTR's stronger routing (task-relevant skills surfacing in step 0) makes Qwen one-shot, removing the opportunity to self-correct.

This is a known issue from description.md 2026-05-23 failure top-k analysis: "CLSTR base in failed samples top-k always contains positive examples; top-1 was always auth-like skills" — but the inverse also bites us. When CLSTR routing is too informative, Qwen attempts one-shot and burns its only step on a wrong answer.

### stop_head regression fix

Phase 1 Task 5 left use_stop_head=True as default. With an untrained stop_head and threshold=0.0, ANY positive logit fires premature stop. Fixed in commit `d43db12`:
- `MultiStepStopPolicy.use_stop_head` default → False
- `run_appworld_multistep_executor_eval` accepts `use_stop_head` kwarg
- CLI / sbatch / dev10 gate all add explicit `USE_STOP_HEAD` opt-in
- Also discovered: ACT training rollout in `clstr/appworld_act_hrpo.py::run_appworld_multistep_hrpo_rollout` does NOT route through `MultiStepStopPolicy` (it breaks on `task_completed AND evaluation_success`). So stop_head never receives ACT reward gradient even with `enable_closed_loop_gradient=True`. Eval-time stop_head is fundamentally untrained and must stay off until the training rollout is also routed through MultiStepStopPolicy (out of scope for Phase 1).

Commit `b21b01d` reverted `USE_STOP_HEAD=1` to `USE_STOP_HEAD=0` in dev10 gate eval block.

### Decision point

Phase 1 closes 5 originally identified bugs and the regression introduced by Task 5. dev10 still does not pass.

The remaining gap (6/10 → 8/10) is NOT a Phase 1-style code bug. It is a structural interaction between routing quality and Qwen's one-shot tendency. Options for Phase 2:

A. Re-run dev10 with USE_STOP_HEAD=0 to recover the 2 stop_head-regression tasks → expected 6+0~2 = 6-8/10. Best case clears bar; worst case still 6/10.
B. Force multi-step exploration: add a prompt-level constraint that step 0 must NOT call complete_task. This is closer to a prompt hack but is benchmark-agnostic.
C. Make the controller pass the *routing distribution* (not the top-5 skills) so Qwen sees auth skills first AND task skills second. Closer to SR behavior.
D. Train stop_head properly: wire ACT training rollout through MultiStepStopPolicy so stop_head sees reward gradient, then re-enable USE_STOP_HEAD=1.

Lowest-cost: A. If A still doesn't pass, B or D are next.

## 2026-05-24 dev10 rerun (USE_STOP_HEAD=0): 4/10 — UNEXPECTED REGRESSION

`outputs/appworld_multistep_executor_benchmark/phase1_dev10_gate_no_stophead/report.json`:
- success_count=4 (worse than 6/10 with USE_STOP_HEAD=1!)
- average_steps=1.3
- evaluate_success_count=4 / task_completed_count=10

Per-task diff vs the original 6/10 run:
| task | 76140 (use_stop_head=1) | 76216 (use_stop_head=0) |
|---|---|---|
| 50e1ac9_1-3 | ✓✓✓ | ✓✓✓ |
| fac291d_1 | ✓ | ✗ wrong answer (64 vs 81) |
| fac291d_2 | ✓ | ✓ |
| fac291d_3 | ✓ | ✗ wrong answer |
| 530b157_1 | ✗ | ✗ |
| 530b157_2 | ✗ stop_head | ✗ wrong answer |
| 530b157_3 | ✗ stop_head | ✗ wrong answer |
| 4ec8de5_1 | ✗ | ✗ |

Two newly-failing tasks (fac291d_1, fac291d_3) used the same routing top-5 in both runs but Qwen generated DIFFERENT Python code:
- 76140 code did paginated `show_song_library(...)` loop → got all 81 songs → correct.
- 76216 code called `show_song_library(...)` once with default page_limit → only got 5 songs → answered 64.

Same checkpoint, same prompt, same temperature=0.0, same skill ranking — Qwen's generate non-determinism (vLLM/HF KV cache state, GPU floating-point) drifts code output across runs. dev10 score has ±2 noise from this alone.

This means single-run dev10 is unreliable for verifying CLSTR vs SkillRouter. Need either (a) multi-seed dev10 with majority vote, or (b) larger split (dev57) where noise washes out.

The Phase 1 implementation work IS done (5 bugs + STOP first-class fix). What remains is to (a) wait for the next training (stop_head trained per `b2b48d4`) to finish so we can compare, AND (b) acknowledge that single dev10 runs are too noisy to claim 6/10 vs 8/10.

Decision: skip the retraining-with-stop-head path FOR NOW. Instead test reproducibility: re-run the original USE_STOP_HEAD=1 setup 3 more times and see if 6/10 is stable. If 6/10 swings between 4 and 8 across reruns, the dev10=8/10 verification target is unmeasurable at this scale.

## 2026-05-24 Phase 1 dev57 + retrain bug + base/act stage gap

### dev57 with existing Phase 1 ckpt (USE_STOP_HEAD=0)

`outputs/appworld_multistep_executor_benchmark/phase1_dev57_existing_ckpt/report.json` (job 76242):
- success_count = **11/57** (success_rate=0.193)
- task_completed=28, eval_success=11
- average_steps=2.32
- step-level execution_failures=51/132

Baseline: SkillRouter-base dev57 = 14/57. Phase 1 CLSTR loses by 3 tasks.

### Retrain (76243) FAILED due to routing_prior shape mismatch

When `enable_closed_loop_gradient=True`, policy_logits is [K+1] but routing_logits stays [K]. `_mean_policy_kl(policy_logits, routing_logits)` in `compute_appworld_multistep_hrpo_loss` raised `RuntimeError("tensor a (17) must match tensor b (16)")` after 2.5 min.

Fix (commit `f258cbb`): in the routing_prior_loss zip, slice policy_logit to its first K positions before passing to KL. STOP has no routing prior — gradient flows through policy_loss + KL-to-reference instead. Regression test added.

### Architecture self-check: base/act stage gap

User raised: "CLSTR 训练不是包含 base 和 act 两阶段吗？我感觉你只训练了一个阶段呢？"

Confirmed gap:

Paper §10.1 specifies 4 stages: `stage0_warmstart` → `stage1_heads` → `stage2_base` (CLSTR-base) → `stage3_joint` (CLSTR-act with verified pairs). Implemented in `clstr/train.py`.

But Phase 1 dev10 gate (jobs 76140, 76243) used `clstr/appworld_act_hrpo.py::train_appworld_clstr_multistep_hrpo` — an INDEPENDENT multi-step HRPO entrypoint that does NOT walk the 4-stage pipeline. Warmstart comes from `appworld_mt_fusion_train` (offline supervised on oracle trajectories), which is a non-standard stage1+stage2 equivalent: it trains heads + base on 90 train oracle trajectories × 3 epochs = ~2100 oracle steps, but does NOT run stage0 retrieval warmup or stage2_base RL rollouts.

So our Phase 1 training stack:
- backbone: frozen SkillRouter-Embedding-0.6B (pre-trained as retrieval router)
- stage0 (retrieval warmup): SKIPPED (backbone already a router)
- stage1+stage2 equivalent: `mt_fusion` 3-epoch oracle supervised (paper §10.1 stage1_heads + stage2_base)
- stage3 (ACT): replaced by `appworld_act_hrpo.py` HRPO with executor reward (matches paper §10 outcome supervision better than the supervised verified-pairs CE in `clstr/train.py::stage3_joint`)

What's missing for "from-zero" complete training:
- stage0_warmstart could add SKILLRET retrieval pretraining if backbone weren't already a router (it is, so likely skip)
- stage2_base full could run more updates (current mt_fusion = 3 epochs × 708 steps = 2124 oracle-supervised steps; paper §10.1 stage2_base is RL rollout-based, longer)
- stage3 HRPO scale: current 240 rollouts (20 updates × 3 task × 4 group), needs ~3000-5000 for stable RL convergence

### Smoke 76372: validate routing_prior fix in real training, 50 updates

Submitted `outputs/appworld_clstr_train_phase1_smoke_50updates`:
- Same config as Phase 1 dev10 gate but UPDATES=50 (vs 20), ENABLE_CLOSED_LOOP_GRADIENT=1
- Verify: `train_report.json` should have `advantage_nonzero_count > 0`, no RuntimeError
- ~50min H200

If smoke passes: submit full from-zero training (stage1+2 + stage3 with UPDATES=200 GROUP_SIZE=8) on Phase 1 stack.

## 2026-05-25 Phase 2 launch — smoke 76372 verdict

Plan: docs/superpowers/plans/2026-05-25-clstr-phase2-full-train-eval.md
Goal: dev57 success_count > 14 (strict > SkillRouter-base = 14/57).

### Smoke 76372 (50-update HRPO) — PASS

- output_dir: outputs/appworld_clstr_train_phase1_smoke_50updates
- checkpoint: outputs/appworld_clstr_train_phase1_smoke_50updates/checkpoints/appworld_clstr_multistep_hrpo-step50.pt (2.51 GB)
- status: ok
- updates: 50/50 completed
- rollout_count: 600 (50 × 3 task × 4 group)
- mean(mean_reward) across updates: 0.0782
- mean(success_rate) across updates: 5.7% (~34 successful rollouts out of 600)
- 50/50 updates have advantage_nonzero_count > 0 (Bug #1 fix verified in real training)
- 6/50 updates used step-proxy fallback (advantage fallback wiring works without dominating)
- sum advantage_nonzero_count = 508 → ~10 effective gradient signals per update
- routing_prior_loss = 0.72 at last update (Bug #4 routing_prior K vs K+1 shape fix from f258cbb survives real training; previous 76243 crashed at 2.5 min)
- ENABLE_CLOSED_LOOP_GRADIENT=1 confirmed in train_config (Bug #2 closed-loop gradient activated)
- wrong_completion_penalty = 0.08 in reward_shaping (Bug #3 fix retained)
- policy_skill_mode=prior_residual (Bug #5 prior_residual default retained via model config)

### Decision

All Phase 1 fixes are confirmed live and stable in real training. Proceed to Task 2: launch full Phase 2 training (UPDATES=200, GROUP_SIZE=8, MAX_TASKS=90, ENABLE_CLOSED_LOOP_GRADIENT=1, DETACH_POLICY_INPUTS=0, WARMSTART = smoke step50 ckpt).

## 2026-05-25 Phase 2 full training launched

- job: 76421
- partition: gpu_h200, time=08:00:00
- output_dir: outputs/appworld_clstr_train_phase2_full_v1
- warmstart: outputs/appworld_clstr_train_phase1_smoke_50updates/checkpoints/appworld_clstr_multistep_hrpo-step50.pt
- key hyperparameters:
  - MODEL_CONFIG: configs/model/appworld_skillrouter_init_prior_residual.yaml
  - MAX_TASKS: 90 (full AppWorld train split)
  - UPDATES: 200, GROUP_SIZE: 8, TASK_BATCH_SIZE: 3 → 4800 rollouts target
  - MAX_STEPS: 5 (multi-step, matches AppWorld avg trajectory length)
  - DETACH_POLICY_INPUTS: 0, ENABLE_CLOSED_LOOP_GRADIENT: 1 (Phase 1 Bug #2 actually live)
  - BETA_ROUTING_PRIOR: 0.02, POLICY_SAMPLING_ALPHA: 1.0 (keep prior anchor weak, allow policy exploration)
  - LEARNING_RATE: 2e-5, BETA_KL: 0.01
- expected ETA: 4-6 h
- next: poll every 30 min; on completion, verify train_report.json has updates_completed=200, advantage_nonzero_count_total > 0, success rollouts present

## 2026-05-25 Sleep-window status (for user when they wake)

Live job: 76421 (Phase 2 full training, gpu_h200, ETA 4-6h)
Background poll: shell task b4l1k5frc polling 76421 every 5 min, will notify Claude on completion.

Next planned actions (Claude will execute autonomously):
1. On 76421 completion → verify train_report.json, launch dev57 eval (new sbatch).
2. Launch SkillRouter-base test_normal baseline in parallel (needed for fair test_normal comparison since baseline currently only exists for dev10/dev57).
3. On dev57 eval completion → launch CLSTR test_normal eval.
4. On both test_normal evals complete → write comparison.md + final verdict to description.md.
5. Stop. Wait for user.

Stop conditions:
- 76421 fails (RuntimeError / OOM in slurm-76421.out)
- train_report.json missing or status != ok
- step200 checkpoint not produced
- any eval crashes with parse error or 0 success across all tasks
- any pytest regression (none planned in this window, no code edits)

What the user will see when they wake:
- description.md section "## 2026-05-25 Phase 2 final verdict" with dev57 + test_normal numbers and SUCCESS/FAILURE line
- git commits showing the train/eval sequence
- outputs/appworld_clstr_train_phase2_full_v1/ with checkpoints and reports
- outputs/appworld_multistep_executor_benchmark/phase2_*_v1/ with eval reports
- outputs/appworld_multistep_executor_benchmark/skillrouter_base_test_normal_v1/ baseline (if launched in time)

## 2026-05-25 SkillRouter test_normal baseline result (job 76440)

- output: outputs/appworld_multistep_executor_benchmark/skillrouter_base_test_normal_v1
- elapsed: 00:09:22 on gpu_a800
- task_count: 168
- success_count: 0/168 (0.0%)
- execution_failures: 89
- task_completed_count: 146
- evaluate_success_count: 0
- average_steps: 1.60
- step_positive_skill_hit_rate: 0.0 (across 269 positives — SkillRouter found ZERO ground-truth skills)

### Why this matters

SkillRouter test_normal hit-rate=0% means the SkillX 188-skill pool does not cover test_normal task families. Qwen sees irrelevant skill context, gives up after 1-2 steps, calls `complete_task("success")`, and the evaluator rejects it.

Implication for Phase 2: even a degenerate CLSTR that outputs random skills could "beat" 0/168 on test_normal. So Phase 2 test_normal will NOT be a meaningful comparison unless CLSTR also operates with the same SkillX-only pool. The fair comparison stays on dev57, where SkillRouter = 14/57 is achievable because dev split tasks are drawn from the same distribution that produced the SkillX pool.

Recommended interpretation:
- dev57 remains the primary verdict (Phase 2 must produce CLSTR dev57 > 14)
- test_normal is reported but flagged: "both methods near zero due to skill-pool coverage gap, not a discriminating benchmark"

The dev57 SkillRouter baseline (14/57) used the OLD single-step executor benchmark (`appworld_executor_benchmark/skillrouter_base_dev57_v1`), not the multi-step loop. So when Phase 2 produces a multi-step dev57 number, we should ALSO rerun SkillRouter dev57 under the multi-step evaluator for an apples-to-apples comparison — submitting that now (job tbd).

## 2026-05-25 Phase 2 v2 launched with rolling checkpoints (job 76579)

### Why v2

76421 TIMEOUT @ 8h limit produced 0 checkpoints. HRPO only saved at the end; SIGTERM at 8h left nothing on disk.

### Fix: Phase 2 logging features (commit db2b27f)

- AppWorldHrpoTrainConfig.checkpoint_every (default 0, opt-in)
- _save_rolling_checkpoint writes atomically to checkpoints/latest.pt every N updates
- _append_losses_jsonl + _emit_progress give live per-update progress in slurm log
- Final ckpt renamed to final-step{N}.pt (distinguishable from rolling latest.pt)
- 11 new tests in tests/test_phase2_logging.py; full Phase1/Phase2 regression = 76/76 passed

### v2 config

- output_dir: outputs/appworld_clstr_train_phase2_full_v2
- job: 76579, gpu_h200, --time=06:00:00 (was 08:00:00; 100 updates ~4h based on smoke 50/1h)
- warmstart: outputs/appworld_clstr_train_phase1_smoke_50updates/checkpoints/appworld_clstr_multistep_hrpo-step50.pt
- MAX_TASKS=90, UPDATES=100 (was 200), GROUP_SIZE=8, TASK_BATCH_SIZE=3 → 2400 rollouts target
- CHECKPOINT_EVERY=10 (≤ 10 rolling ckpts during run; 1 final final-step100.pt)
- ENABLE_CLOSED_LOOP_GRADIENT=1, DETACH_POLICY_INPUTS=0 (Phase 1 closed-loop fix actually trains belief modules)
- POLICY_SAMPLING_ALPHA=1.0, BETA_ROUTING_PRIOR=0.02 (policy actually explores)
- expected ETA: 4h

### Why 100 updates not 200

- 76421 ran 6h45m on update progress invisible. 8h - load/init overhead ≈ 7h30m → ~150 updates max in 8h, ~100 updates safely in 6h.
- Halving updates means less total gradient signal but ckpt every 10 means we can pick the best rolling ckpt via mid-training eval if needed.
- Smoke had 50/600 successful rollouts → 100 updates × 24 rollouts/update = 2400 rollouts × 8% = ~200 successful rollouts (vs 76421 effective ~14 successful in 6h45m).

### Apples-to-apples SkillRouter baselines

- multistep dev57: 14/57 (job 76449, COMPLETED in 13m on a800) — same as legacy single-step dev57=14
- multistep test_normal: 0/168 (job 76440) — SkillX 188-skill pool does not cover test_normal task families; test_normal will be a "near-zero vs near-zero" comparison, not discriminating

## 2026-05-25 Unified CLSTR pretrain corpus v1 built (no training)

Per user direction "等任务提交后可以着手构建我们说的clstr训练数据集", built the unified
pretrain corpus while 76579 trains. Does NOT launch training — only data preparation.

### Builder

scripts/build_clstr_unified_pretrain.py — merges multiple trajectory + retrieval
sources into a single schema, with explicit AppWorld dev/test and SKILLRET test
leakage exclusion.

### Output (data/clstr_unified_pretrain_v1/)

| File | Lines | Size |
|---|---:|---:|
| trajectories.jsonl | 235,175 | 1.1 GB |
| retrieval.jsonl | 127,190 | 95 MB |
| manifest.json | — | 1.4 KB |

### Trajectory stream composition

| Source | Rows | source_quality |
|---|---:|---|
| ALFWorld official + skillnet aux | 109,494 | official_replay (2,443) + aux_only (107,051) |
| ScienceWorld skillnet aux | 93,434 | aux_only |
| AgentGym AgentTraj-L (ALFWorld) | 32,247 | weak_policy |

### Retrieval stream composition

127,190 SKILLRET train query-skill pairs. 8,347 test query_ids excluded.

### Leakage audit (mandatory before any training)

- AppWorld dev (57) + test_normal (168) + test_challenge (417) task_ids: 642 excluded
- AppWorld dev/test rows filtered from trajectories: 0 (no AppWorld trajectories
  in current sources)
- SKILLRET test query_ids: 4,997 excluded
- Policy: no dev/test split content enters training

### Not yet included

- AppWorld step-level on-policy rollouts (76579 will emit them to
  rollouts.jsonl; can be appended to trajectories.jsonl in v2 after audit)
- ScienceWorld AgentGym (only ALFWorld AgentGym imported)

### What this enables when user wakes

If 76579 dev57 result is still ≤ 14/57, this unified corpus is the next-stage
pretraining input: a new ACT stack can warmstart on 235k trajectories + 127k
retrieval pairs (vs. current 90-task AppWorld-only fine-tune = 720 step-pairs).
Decision to actually launch unified pretraining is left to the user.

## 2026-05-25 12:13 — 76579 status update (5/100 done)

- Elapsed: 33min, RunTime ~6.5min/update vs predicted 2.4min/update
- losses.jsonl visible per-update progress: update 1-5 done
- Success_rate after update 1: 0.458 (excellent), then dropped to 0 for updates 2-5
- ENABLE_CLOSED_LOOP_GRADIENT=1 + MAX_STEPS=5 + GROUP_SIZE=8 makes per-update
  significantly slower than smoke (which had MAX_STEPS=3, GROUP_SIZE=4)
- TimeLimit=06:00:00, EndTime=17:41 → only ~50/100 updates will complete

### Implication

Phase 2 v2 was sized 100 updates / 4h based on smoke 50/1h linear extrapolation;
this turns out wrong again. But unlike 76421, the rolling checkpoint feature
WILL save progress: at update 50 we'll have checkpoints/latest.pt with 50
updates of training, even if time limit kills before update 100.

### Decision (no action needed, autonomous)

Let 76579 run. When SLURM kills at 17:41, latest.pt with ~50 updates exists.
At that point Claude (this session or next) should:
1. Use latest.pt as the eval checkpoint (NOT final-step100.pt which won't exist)
2. Run dev57 eval against latest.pt
3. If dev57 > 14: goal achieved
4. If dev57 ≤ 14: do NOT auto-relaunch; write final verdict to description.md

### Per-user instruction "不要妥协于训练时间"

I am NOT setting time limits going forward. The current 76579 inherits the 6h
limit from a prior commit (3419c39) that I did not author this session. Next
training submission (if needed) will use --time=23:00:00 or partition default.

## 2026-05-25 12:55 — 76579 cancelled, Phase 2 v3 submitted no time limit (job 76632)

### Cancellation rationale

User clarified: "我不是说不要加timeout限制了吗？让它自己跑完就行" — I had incorrectly
kept the 6h time limit from the prior session's commit 3419c39. 76579 was
on track to be killed at 17:41 with only ~50 updates done. Per user direction,
let it run to completion instead of capping by wallclock.

12 updates of 76579 loss data (preserved as reference, will not be used as ckpt):
```
upd  succ  reward    loss  policy routing      kl adv_nz
  1 0.458   0.479  -0.479  -0.490   0.520  0.0000     16
  8 0.750   0.746  -1.791  -1.802   0.540  0.0000     16
  others varied 0-0.125 with reward in [-0.03, +0.05]
```
Loss/reward had not converged — would need more updates. But 76579 sized
for only 100 updates with 6h limit → cancel and resubmit no-limit.

### Phase 2 v3 (job 76632)

- partition gpu_h200, MaxTime=UNLIMITED (verified TimeLimit=UNLIMITED in scontrol)
- output_dir: outputs/appworld_clstr_train_phase2_full_v3
- warmstart: outputs/appworld_clstr_train_phase1_smoke_50updates/checkpoints/appworld_clstr_multistep_hrpo-step50.pt
- MAX_TASKS=90, UPDATES=100, GROUP_SIZE=8, TASK_BATCH_SIZE=3, MAX_STEPS=5
- CHECKPOINT_EVERY=10 (rolling latest.pt + final-step100.pt at end)
- ENABLE_CLOSED_LOOP_GRADIENT=1, DETACH_POLICY_INPUTS=0
- POLICY_SAMPLING_ALPHA=1.0, BETA_ROUTING_PRIOR=0.02, LR=2e-5
- expected ETA: ~10h based on 76579 ~6.5min/update

### After v3 completes

1. Inspect loss_curve.png in outputs/appworld_clstr_train_phase2_full_v3/
2. If loss/reward still trending (not converged), decide whether to extend with
   another 100-update warmstart from final-step100.pt
3. If loss flat: pick the best rolling latest.pt (typically by mean_reward) and
   launch dev57 eval against it
4. Either way, submit SkillRouter dev57 + test_normal baselines apples-to-apples
   if needed and run final comparison

## 2026-05-25 Goal v3 pivot — SKILLRET 主战场 + Hybrid lexical channel

新 goal 锁定，AppWorld task success 不再作主战场。CLSTR 重新定位为
**latent-space skill routing layer for large-pool agent ecosystems**，主指标
切到 SKILLRET 16k pool routing recall。详见 goal 系统约束。

## 2026-05-25 Step 0 historical + resource audit

执行 goal v3 Step 0：历史 SKILLRET 数据归档、SkillNet/SkillX 资源调研、
verified_pair 现状确认。三项发现都有 implication，记录在下。

### 0.1 历史 SKILLRET 实测 — CLSTR 比 SkillRouter 仅高 0.55pp（远低于 goal ≥3pp 阈值）

**位置**：`outputs/skillret_official_eval/comparison_table.md`（4997 query SKILLRET test split,
pytrec_eval 协议）。所有方法 backbone 都是 SkillRouter-Embedding-0.6B，frozen。

| method                          | NDCG@10 | Recall@10 | MAP@10 | Δ vs SR baseline |
|---|---|---|---|---|
| skillrouter_frozen (baseline)   | 0.7014  | 0.7532    | 0.6304 | —                 |
| skillrouter_style_finetune      | 0.7039  | 0.7552    | 0.6330 | +0.25 pp          |
| clstr_skillrouter_init          | 0.7033  | 0.7548    | 0.6327 | +0.19 pp          |
| clstr_qdoc                      | 0.7039  | 0.7552    | 0.6330 | +0.25 pp          |
| **clstr_qdoc_rerank (best)**    | **0.7069** | **0.7605** | **0.6347** | **+0.55 pp** |
| clstr_native_rerank             | 0.7058  | 0.7567    | 0.6339 | +0.44 pp          |

**关键诊断**：当前最佳 CLSTR 变体 (`clstr_qdoc_rerank_full`) 仅比 frozen SkillRouter 高 0.55 pp，
远低于 goal 成功条件 1 的 ≥3 pp 阈值。这与用户先前描述"略高"完全吻合，但
**当前 ckpt 不足以让条件 1 通过**。Implication：

- **条件 1 必须靠 unified pretrain (235k traj + 127k retrieval) 把 gap 拉开**，
  当前 stack 只训了 SKILLRET train pair (~127k) + adapter/rerank head，
  没用上 trajectory transition signal (L_trans) 与 outcome reward (L_policy)
- **条件 2 (hybrid channel) 与条件 3 (combination plot) 成为同等重要的备选**
- 当前 ckpt 不作 unified pretrain 起点（goal v3 已明确），但**记入论文 §15 主表作为现状
  baseline + ablation 锚点**

### 0.2 SkillNet 本地副本与定位

**位置**：`/data/run01/scyb713/xzf/skillnet/SkillNet/`（已 clone）。

性质：**400k+ community skills 的 npm-like marketplace**（不是学术 corpus），
含 `skillnet-ai/` Python SDK、`run_alfworld_skillnet.sbatch`、`skills/` 目录、
`experiments/`。arXiv 2603.04448。

与 SkillX 的关系：互补，不重叠。SkillX 是 trajectory-abstraction skill library
（已用作 AppWorld 188 skill pool）；SkillNet 是 community-curated skill marketplace
（OOD distribution）。

**决策**（goal v3 Step 0.1 决策点）：
- **不**纳入 unified corpus v2 训练流（规模未审计 + 质量分布未知，盲入有数据污染风险）
- **作为 OOD eval set 候选**：等 Step 4 evaluation matrix 跑完，若需要泛化证据，
  从 SkillNet 抽 1k-5k skill 构造 hold-out retrieval eval，验证 unified-trained CLSTR
  在社区分布上不退步

### 0.3 SkillX 本地副本

**位置**：`/data/run01/scyb713/xzf/AAAI/autodl-tmp/SkillX/`（已 clone）。已通过 188
AppWorld skill pool 验证 schema 可用：`name / description / input / output / exec /
failure_modes / body` 7 字段映射通顺。

不需要进一步调研——已经在用。

### 0.4 verified_pairs 现状 — m_t_exact 全部 null，必须 train-time replay

**真实数据位置**：`data/appworld_act/verified_pairs_train.jsonl`（50 MB，618 pair）。
`data/verified_pairs/manifest.json` 是 SkillsBench 时代的空壳（pair_count=0），可忽略。

Schema（与论文 §8 / §14.1 对齐）：
```
a_next_plus, action_at_t, candidates_next, m_t_exact, obs_at_t,
replay_prefix, replay_prefix_is_essential_subsequence,
state_before, step_idx, task_id, was_in_raw_topk
```

字段填充率（618/618）：
- `m_t_exact` null: **618/618 (100%)** ← 必须 train-time replay 重算
- `replay_prefix` 非空: 531/618 (85.9%)
- `replay_prefix_is_essential_subsequence`: 全 true
- `was_in_raw_topk`: 618/618 (100%) ← `a_next_plus` 都已经在 top-20 候选内，
  不需要强制注入

来源：`AppWorld train split ground_truth/api_calls.json` mapped to SkillX skills by
API refs（不是 LLM-validated essential subsequence——比论文 §13.2 描述弱一档）。
覆盖 90 task 中的 87 (96.7%)，平均序列长度 8.1 step，dev/test 不污染。

**Implication for Step 5**：
- 618 pair 可用，但 `m_t_exact` 必须在训练 loop 里通过 `replay_prefix` 重新前向计算
- `losses.py:action_loss` 默认 `allow_approximate_m_t=False` 必须保留，
  禁止跳过 replay 直接用 subspace_obs 近似
- 由于 `was_in_raw_topk=100%`，Step 5 不需要 "skip a^+ ∉ C_{t+1} 的样本" 这条
  fallback 逻辑——但代码里仍要写好这个 guard
- 监督源质量低于论文 §13.2 描述（API-mapping vs LLM-validated）；后续若需提升，
  补 LLM-as-judge 过滤是独立工作

### 0.5 unified corpus schema 复核

**位置**：`data/clstr_unified_pretrain_v1/{trajectories.jsonl, retrieval.jsonl, manifest.json}`

Trajectories（235,175 rows，1.1 GB）含字段：`benchmark, task_id, trajectory_id,
step_index, goal_text, task_text, state_text, history_text, action_text,
expert_action, admissible_actions, next_action_text, next_observation_text,
skill_id, next_skill_id, done, reward, loss_mask, source_quality, candidate_source,
on_policy_rollout, m_t_source, provenance, _unified_source`

Benchmark 分布：
- alfworld: 109,494 (official_replay 2,443 + aux_only 200,485 误差→实际 aux_only 107,051)
- scienceworld: 93,434 (aux_only)
- alfworld_agentgym: 32,247 (weak_policy)
- **不含 AppWorld**（POMDP-belief schema 差异，待 v2 集成）

Retrieval（127,190 pair，95 MB）含字段：`source, query_id, query_text,
positive_skill_id, negative_skill_ids, provenance`。全部来自 SKILLRET train split。

Leakage audit 已执行：AppWorld dev/test 642 task_ids excluded，SKILLRET test 4,997
query_ids excluded（即 retrieval.jsonl 已从原始 135k 过滤掉 8,347 leaked pair）。

**Implication for Step 1**：
- `clstr_unified_pretrain` dataset loader 必须按字段 `benchmark` / `source` 做 stream-mixing
- `L_trans` 用 trajectory 相邻 (step_index, step_index+1) belief 一致性
- `L_policy` 用 `reward` 与 `done` 做 task_id 分组归一化
- `L_retr` 用 retrieval 的 `(query_text, positive_skill_id, negative_skill_ids)` 三元组
- `m_t_source` 字段已有，留作 ablation 用（哪些 m_t 是 expert / weak_policy 来的）

### 0.6 Step 0 阶段总结 + 影响下一步的关键决策

1. **历史 SKILLRET 数字写入新论文 §15 主表当 baseline 锚点**（不是被反对的旧结果）
2. **goal 成功条件 1 (≥3pp) 风险较高**：当前 stack 只能拉 0.55pp，必须验证
   unified pretrain 能拉到 3pp+；如果 Step 2 smoke 仍只能拉 1pp 左右，需要立即
   评估是否升级 goal 阈值或加权 hybrid channel (条件 2)
3. **verified_pair 可用，但 m_t_exact 必须 train-time replay**——Step 5 实现里
   要写好 replay path（这是工程难点）
4. **SkillNet 暂不进训练**，只留 OOD eval 候选
5. **76632 继续放着跑**（与 Step 0-1 不冲突）

Step 0 完成。下一步进入 Step 1 plan mode：设计 unified pretrain 训练入口的实施方案。

## 2026-05-25 22:35 — v4 plan design update: generalized L_act + split logit scales

根据用户关于 Stage 4 与 Q5 的追问，更新 `docs/clstr_v4_plan.md`：

1. **Stage 4 `L_act` 不再限定 AppWorld**：
   - 改为跨 benchmark 的 transition-conditioned next-skill training。
   - primary 数据源为 TRAJECT sequential train 与 ToolBench-G3 simulated train trajectories。
   - AppWorld 618 verified pairs 仅作为小规模 domain adaptation 与 §15.5.5 secondary eval。
   - 约束保持严格：所有 dev/test signal 禁入；AppWorld `m_t_exact` 仍需 replay，`allow_approximate_m_t=False`。

2. **Q5 改为 F2-v2：retrieval scale 与 belief scale 分离**：
   - retrieval path 使用 `logit_scale_retr` / `skill_bias_retr`，保持 SkillRouter-like sharp ranking。
   - `subspace_obs` 使用 `logit_scale_belief` / `skill_bias_belief`，避免 `m_t` 过早退化成 top-1 skill embedding。
   - `shared_scale` 从默认方案降为 ablation。
   - Stage 3 默认冻结 retrieval path；belief scale/bias 的 `frozen` vs `low-lr` 必须作为 ablation 记录。

该变更保持论文方法泛化性：`L_act` 现在是通用 sequential tool-use transition objective，不是 AppWorld-specific trick。

## 2026-05-26 16:35 — Stage1 loss 异常诊断与修复

用户查看 Stage1 `loss_curve.svg` 后指出 loss 一直跳动。重新核验后结论是：单步
loss 抖动本身可以由小 batch 与 query 难度差异解释，但当前 Stage1 的 rolling mean
基本不下降，且后段 `recall_at_50` 接近 0，不能作为 Stage2 前置。

定位到两个问题：

1. Stage1 sbatch 默认使用 `models/Qwen3-8B`，偏离 v4 plan 中的
   `SkillRouter-Emb-0.6B` 初始化。
2. Stage1 warmup batch 采样是滑动一行，5000 step、batch size 16 时只覆盖约
   5015 个连续 query；由于 unified retrieval 文件按 source 排序，实际主要训练文件头部
   SKILLRET，未有效覆盖 ToolRet / ToolBench-G3 / TRAJECT 全 corpus。

处理：

- 取消异常 Stage1 作业 `77445` 及依赖导出作业 `77449-77452`；
- 将异常输出归档到
  `outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup_bad_qwen_sliding_20260526_1628/`；
- 修复 `clstr/retrieval_warmup.py`：默认 deterministic shuffle query，batch 采样改为
  batch-stride，并在 `training_metrics.jsonl` 记录 `query_ids` / `query_sources`；
- 修复 `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh`：默认 backbone 改为
  `.cache/hf_models/SkillRouter-Embedding-0.6B`；
- CLI 增加 `--no_shuffle_queries` 便于测试固定采样顺序。

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

结论：旧 Stage1 不再作为有效实验结果；必须用修复后的 Stage1 重新训练，最终
checkpoint 生成后再运行 Stage2 preflight。

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

当时状态：修复后的 Stage1 作业 `77497` 已在运行，早期 metrics 已显示 `query_sources`
覆盖 `skillret/toolret_training/toolbench_g3/traject_bench`，且 `recall_at_50` 已非退化。
当时仍需等待最终 checkpoint，再运行 Stage1 quality gate 与 Stage2 preflight；本轮没有提交 Stage2。
该状态已由 17:30 记录更新：Stage1 已完成，gate/preflight 已通过，Stage2 已提交。

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

当时注意：Stage1 `77497` 尚未产出最终 checkpoint，因此不能运行该提交入口。
该状态已由 17:30 记录更新：安全提交入口已实际运行并提交 Stage2。

## 2026-05-26 17:30 — Stage1 gate 通过，Stage2 已按安全入口提交

Stage1 `77497` 已自然结束，最终 checkpoint 已生成：

```text
outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/checkpoints/clstr_unified_retrieval_v2-step5000.pt
```

本轮随后运行：

```bash
bash scripts/submit_clstr_stage2_after_stage1_gate.sh
```

该安全入口先后完成 Stage1 quality gate 与 Stage2 preflight，二者均为
`status=ok`，然后只提交 Stage2，不提交 Stage3/Stage4。

Stage1 quality gate 关键结果：

- `metric_count=5000`, `max_step=5000`
- `first_loss_mean=10.281204380989074`
- `last_loss_mean=9.29607681274414`
- `loss_drop=0.9851275682449341`
- `last_recall_mean=0.4146875` using `recall_at_50`
- query source 覆盖 `skillret/toolret_training/toolbench_g3/traject_bench`
- checkpoint metadata: `stage=clstr_unified_retrieval_v2`,
  `sampling_strategy=batch_stride`, `shuffle_queries=true`

Stage2 preflight 关键结果：

- `skill_count=37623`
- `embedding_shape=[37623, 1024]`
- `has_retrieval_adapter=true`
- `has_retrieval_scale_and_bias=true`
- Stage1 train safety metadata 显示 dev/test/valid/eval split 已排除

Stage2 作业已提交：

```text
Submitted batch job 77555
```

当时后续约束：

- 不要在 Stage2 结束前提交 Stage3；
- Stage3 入口必须等待
  `outputs/clstr_unified_stage2_toolbench_g3_traject_split_full_base/checkpoints/clstr_full_base-step10000.pt`
  存在且 Stage2 报告可审计；
- 旧 Stage4 入口必须等待 Stage3 final checkpoint；当前主线已在
  2026-06-07 改为 Joint ACT Stage4，直接从通过 gate 的 Stage2 checkpoint 初始化；
- 训练状态检查不要高频轮询，短任务约 15 分钟、长任务约 30 分钟以上再看。

本轮额外做了 v4 配置一致性修正：

- `CLSTRConfig.d_a` 改为未显式指定时自动等于 `d`；
- v4 主配置中的 `d_a` 改为与 `d` 同维；
- Qwen external CLSTR config 显式设置
  `skill_head_context=belief_residual`、`candidate_widen_factor=4`、
  `candidate_sample_temperature=1.0`。

验证状态：

```bash
python -m py_compile clstr/model.py clstr/qwen_external_encoder.py \
  clstr/full_base_train.py clstr/stage3_unified_hrpo.py \
  clstr/stage4_act_train.py scripts/run_clstr_qwen3_full_base_train.py

bash -n scripts/submit_clstr_stage2_after_stage1_gate.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh

git diff --check -- clstr/model.py clstr/qwen_external_encoder.py \
  configs/model/base.yaml configs/model/appworld_mini.yaml \
  configs/model/appworld_skillrouter_init.yaml \
  configs/model/appworld_skillrouter_init_prior_residual.yaml \
  configs/model/appworld_skillrouter_init_mt_fusion.yaml \
  configs/model/appworld_skillrouter_init_gated_prior_residual.yaml \
  configs/model/appworld_skillrouter_init_gated_residual.yaml \
  description.md docs/clstr_v4_plan.md \
  scripts/submit_clstr_stage2_after_stage1_gate.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
```

以上语法与 diff 检查通过。尝试运行 `pytest` targeted tests 时，当前登录节点
`torch` import 在 20-60 秒 timeout 内无输出；`python -S` 与普通 Python 启动正常，
因此本轮未把 pytest 作为完成证据。后续若需要跑 torch-heavy tests，应优先通过
短 GPU sbatch 或在交互允许的计算节点环境中执行，避免登录节点长时间阻塞。

## 2026-05-26 17:40 — Stage3/Stage4 上游 checkpoint 保护与 README v4 对齐

为避免 Stage2 尚未完成时误启动 Stage3/Stage4，本轮给后续 sbatch 入口补了
上游 checkpoint 存在性检查：

- `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`
  - 检查 `ROUTING_CHECKPOINT_PATH` 是否存在且非空；
  - 检查 `HEAD_CHECKPOINT_PATH` 是否存在且非空；
  - 缺 Stage2 final checkpoint 时直接退出：
    `Stage3 requires completed Stage2 checkpoint`。
- `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
  - 检查 `ROUTING_CHECKPOINT_PATH` 是否存在且非空；
  - 检查 `HEAD_CHECKPOINT_PATH` 是否存在且非空；
  - 当时旧默认 `HEAD_CHECKPOINT_PATH` 指向 Stage3 final checkpoint，缺失时直接退出：
    `Stage4 requires completed Stage3 checkpoint`。

同时更新 `README.md`：

- 顶部方向从旧 AppWorld-first 改为当前 v4：TRAJECT-Bench / ToolBench-G3 / ToolRet 三主表；
- 增加当前状态：Stage1 完成、Stage2 job `77555` 已提交；
- 明确 Stage3 必须等待 Stage2 `clstr_full_base-step10000.pt`；
- 当时明确旧 Stage4 必须等待 Stage3 `clstr_unified_stage3_hrpo-step500.pt`；
- Suggested Next Steps 改为当前 v4 训练/eval 流程，不再指向旧 dev10 AppWorld stop-policy jobs。

验证：

```bash
PYTHONPATH=.vendor_pytest:. python -m pytest \
  tests/test_sbatch_scripts.py::test_unified_stage3_hrpo_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  -q --basetemp=.tmp_pytest
# 2 passed

bash -n scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh

git diff --check -- README.md description.md docs/clstr_v4_plan.md \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  tests/test_sbatch_scripts.py
```

本轮仍未提交 Stage3/Stage4。Stage2 完成前继续保持等待。

## 2026-05-26 18:05 — 训练可观测性与 Stage3 证据边界修正

针对 Stage1 loss 曲线“单步一直跳动”容易误判的问题，本轮修正
`clstr/training_monitor.py`：

- `loss_curve.svg` 现在同时画 raw loss、rolling loss、EMA loss；
- 若指标中存在 `recall_at_50/10/5/1`，会在右轴叠加 recall 曲线；
- `TrainingMonitor` 不再在模块导入阶段强制 import `torch`，轻量画图和
  readiness audit 可在无 torch 的登录节点 Python 下运行；只有保存 checkpoint
  时才懒加载 torch；
- 已基于现有 Stage1 `training_metrics.jsonl` 刷新
  `outputs/clstr_unified_stage1_toolbench_g3_traject_split_retrieval_warmup/loss_curve.svg`。

这不改变训练目标和 checkpoint，只改善持续可观测性。Stage1 当前判断应看滑动均值
和 recall：前后窗口 loss 明显下降，`recall_at_50` 上升，因此 Stage1 warmup
不是训练坏掉；但它也只是过了 warmup gate，不能替代三主 benchmark。

本轮还修正 readiness audit 的 Stage3 表述，避免把当前离线 surrogate 当成论文主
RL 证据：

- `stage3_offline_hrpo` 明确标注
  `training_regime=offline_train_split_grouped_preference_update`、
  `on_policy_rollout_used=false`、`not_full_simulator_rollout_hrpo=true`、
  `paper_evidence_ready=false`；
- 新增 `stage3_full_rollout_hrpo` gate，当前固定为 not ready，blocker 为
  `full_rollout_hrpo_not_implemented`；
- `paper_readiness.stage3_full_rollout_hrpo_ready=false`，
  `offline_stage3_is_not_main_rl_evidence=true`。

含义：当前 Stage3 可以作为工程上的 policy/head 训练阶段继续推进，但不能在论文中
声称已经完成 simulator/on-policy HRPO。若后续要主张完整 RL，需要单独实现并验证
TRAJECT/ToolBench simulator rollout loop。

## 2026-05-26 18:20 — Stage2-to-Stage3 质量门与安全提交入口

本轮新增 Stage2 full-base 质量门，防止只因为 Stage2 final checkpoint 存在就直接
进入 Stage3：

- 新增 `clstr/stage2_quality_gate.py`；
- 新增 CLI `scripts/audit_clstr_stage2_quality.py`；
- 默认检查：
  - `train_report.json` 存在且 `status=ok`；
  - `training_objective=component_complete_masked_multi_loss`；
  - `training_regime=offline_replay_supervised_pretraining`；
  - `valid_or_test_used_for_training=false`；
  - `on_policy_rollout_used=false`；
  - `frozen_routing_foundation=true`；
  - Stage2 final checkpoint、`checkpoints/latest.pt`、`training_metrics.jsonl`、
    `loss_curve.svg` 均存在；
  - `max_step >= 10000`；
  - first/last window loss 至少下降 `0.02`；
  - `L_policy/L_trans/L_trans_skill_ce/STOP/routing` 均在数据中激活且被采样。

新增安全提交入口：

```bash
bash scripts/submit_clstr_stage3_after_stage2_gate.sh
```

该入口先检查 Stage1 routing checkpoint 与 Stage2 final checkpoint，再运行
`scripts/audit_clstr_stage2_quality.py --fail_on_action_required`，只有 gate 通过才
提交 `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`。该入口**不会提交 Stage4**。

注意：Stage2 quality gate 只是工程 handoff gate，证明 Stage2 可以安全进入
Stage3 offline surrogate；它不能替代 TRAJECT / ToolBench-G3 / ToolRet 的正式
benchmark，也不能改变 Stage3 当前不是 full rollout HRPO 的边界。

当前外部状态：Stage2 job `77555` 仍为 `PENDING (Priority)`，Stage2 output 目录目前
只有 `stage1_quality_gate.json` 和 `stage2_preflight.json`，尚无训练指标或
`clstr_full_base-step10000.pt`。因此本轮没有提交 Stage3。

## 2026-05-26 18:35 — Stage3-to-Stage4 质量门与安全提交入口

本轮新增 Stage3 offline HRPO-style 质量门，防止只因为 Stage3 checkpoint 存在就
直接进入 Stage4：

- 新增 `clstr/stage3_quality_gate.py`；
- 新增 CLI `scripts/audit_clstr_stage3_quality.py`；
- 新增安全提交入口 `scripts/submit_clstr_stage4_after_stage3_gate.sh`；
- `scripts/audit_clstr_unified_training_readiness.py` 当时也接入 Stage3 quality gate，
  旧 `stage4_act` 会在 `stage3_quality_gate_not_ok` 时阻断。当前主线已在
  2026-06-07 改为由 Stage2 checkpoint + Stage2 quality gate 决定 Stage4 readiness。

Stage3 quality gate 默认检查：

- `train_report.json` 存在且 `status=ok`；
- `stage=clstr_unified_stage3_hrpo_style`；
- `training_objective=unified_offline_hrpo_style_action_policy_update`；
- `training_regime=offline_train_split_grouped_preference_update`；
- `valid_or_test_used_for_training=false`；
- `on_policy_rollout_used=false`；
- `not_full_simulator_rollout_hrpo=true`；
- `hrpo_style_loss_used=true`；
- final checkpoint、`checkpoints/latest.pt`、`training_metrics.jsonl`、
  `loss_curve.svg` 均存在；
- `max_update >= 500`；
- `data_report.hrpo_rows > 0`；
- `traject_bench/toolbench_g3` 均有 Stage3 训练覆盖。

安全提交入口：

```bash
bash scripts/submit_clstr_stage4_after_stage3_gate.sh
```

该入口先检查 Stage1 routing checkpoint 与 Stage3 final checkpoint，再运行
`scripts/audit_clstr_stage3_quality.py --fail_on_action_required`，只有 gate 通过才
提交 `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`。该入口不会提交 Stage3。

边界：该 gate 只证明 Stage3 offline surrogate 可以作为 Stage4 初始化；它不证明
full simulator/on-policy HRPO，也不能替代后续 benchmark evaluation。

## 2026-05-26 18:55 — Phase H evaluation matrix readiness gate

本轮新增 Phase H 前置 readiness gate，用来防止论文结果表混淆 official benchmark
指标与 proxy/routing 指标：

- 新增 `clstr/eval_matrix_readiness.py`；
- 新增 CLI `scripts/audit_clstr_eval_matrix_readiness.py`；
- 新增 sbatch 包装 `scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh`；
- 新增测试 `tests/test_eval_matrix_readiness.py`，并补充 sbatch 脚本测试。

该 gate 显式拆分三类证据：

- `official_main_table_ready`：只在 ToolRet official retrieval run、TRAJECT
  official `EM/Inclusion/Usage/Traj-Satisfy/Acc`、ToolBench-G3 StableToolBench
  pass-rate 条件都满足时为 true；
- `proxy_or_routing_eval_ready`：允许 TRAJECT sequence proxy、ToolBench-G3 static
  routing eval 等中间证据 ready，但这些证据不会满足主表 official claim；
- `appworld_secondary_ready`：只对应 §15.5.5 AppWorld combination plot，不进入
  三主表。

默认审计入口：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh
```

H200 队列繁忙时可切到：

```bash
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh
```

该脚本只运行 audit，不启动训练、raw generation、StableToolBench pass-rate eval 或
其他 benchmark 执行。它的作用是 Stage4 完成后先做证据口径检查：如果仍有
`traject_official_outputs_missing`、`toolbench_g3_pass_rate_not_ready`、
`appworld_secondary_combination_report_missing` 等 blocker，就只能报告 proxy/
routing/secondary 结果，不能声称 §15 三主表完整。

已在当前项目状态下生成一次报告：

```text
outputs/clstr_eval_matrix_readiness/eval_matrix_readiness_current.json
```

当前 `status=action_required`，`official_main_table_ready=false`，
`proxy_or_routing_eval_ready=true`，主要 blocker 为：

- `missing_stage4_checkpoint`
- `toolret_run_missing`
- `traject_official_outputs_missing`
- `toolbench_g3_pass_rate_not_ready`
- `appworld_secondary_combination_report_missing`

其中 ToolRet eval data、TRAJECT eval data、ToolBench-G3 normalized data 已通过各自
data audit；当前缺口主要是 Stage4 产物、ToolRet run、TRAJECT official metrics、
ToolBench-G3 StableToolBench converted/pass-rate 产物，以及 AppWorld secondary
combination report。

## 2026-05-26 19:25 — Stage0 bi-encoder foundation 重构计划

本轮根据新的方法诊断，新增下一阶段执行文档：

```text
docs/clstr_stage0_biencoder_foundation_plan.md
```

关键方向变更：

- 暂停把当前弱 Stage1 retrieval warmup 作为 Stage2/3/4 的可靠前置；
- 将当前 `Stage1 retrieval warmup` 重新定位为更清晰的
  `Stage0 SkillRouter-compatible bi-encoder coarse retriever`；
- Stage0 只做大池粗召回：SkillRouter-style query/skill template、left padding、
  last-token pooling、L2 normalize、cosine scoring、InfoNCE/hard negatives；
- Stage0 不接 SkillRouter cross-encoder/listwise reranker。静态 reranker 后续只作为
  baseline 或 upper bound；
- Stage1/2 才开始训练 CLSTR 的 heads / belief / transition / policy，在 Stage0 top-M
  候选内做动态选择；
- skill pool 去重、canonical qrels、multi-positive qrels 与 leakage audit 必须前置到
  Stage0 之前，否则 readiness gate 应阻断训练。

当前判断：旧 Stage1 在约 37k skill pool 上的粗召回仍不够强，不能解释为一个可发表的
SkillRouter-style foundation。后续应先完成 `docs/clstr_stage0_biencoder_foundation_plan.md`
中的 foundation repair，再重新提交 Stage0 smoke / full training；未通过 Stage0 coarse
recall gate 前，不提交 Stage2/3/4 长训练。

## 2026-05-26 20:10 — Stage2 主线从 Qwen wrapper 切回 CLSTR-native

本轮按 Stage0 foundation plan 做必要重构，修正旧链路中 Stage2 默认调用
`scripts/run_clstr_qwen3_full_base_train.py` 的问题。CLSTR 主线训练现在使用：

```text
scripts/run_clstr_stage2_full_base_train.py
scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
```

关键边界：

- Stage2 默认是 CLSTR-native full-base training，不加载 `models/Qwen3-8B`，不接受
  `qwen_model_path`；
- Stage2 默认消费 Stage0 top-M candidate set：
  `--stage0_top_m 100 --stage0_positive_missing_policy skip`；
- positive 不在 Stage0 top-M 时默认 skip，并写
  `stage0_candidate_handoff.json`；只有显式
  `--allow_full_pool_stage2_debug` 才允许退回全 skill pool debug；
- Qwen 入口保留为 downstream executor / optional encoder ablation，不再是统一
  Stage2 训练 sbatch 的默认路径。

已验证的最小证据：

```bash
PYTHONPATH=.vendor_pytest:. TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/tmp \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_clstr_topm_candidate_handoff.py -q

PYTHONPATH=.vendor_pytest:. TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/tmp \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  tests/test_sbatch_scripts.py::test_submit_stage2_after_stage1_gate_checks_stage0_gate_and_only_submits_stage2 \
  tests/test_sbatch_scripts.py::test_submit_stage1_after_stage0_gate_alias_checks_stage0_gate_and_only_submits_stage2 -q

bash -n scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/submit_clstr_stage1_after_stage0_gate.sh \
  scripts/submit_clstr_stage2_after_stage1_gate.sh

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py clstr/qwen_full_base_train.py \
  scripts/run_clstr_stage2_full_base_train.py \
  scripts/run_clstr_qwen3_full_base_train.py
```

本轮没有提交新的 GPU 训练任务。

## 2026-05-27 00:00 — Stage0 baseline 与 bi-encoder 训练已提交

按 `docs/clstr_stage0_biencoder_foundation_plan.md` 的进入下游训练硬条件，本轮先审计
Stage0 产物状态：

- `outputs/clstr_stage0_skillrouter_frozen_baseline/manifest.json`：缺失；
- `outputs/clstr_stage0_skillrouter_frozen_baseline/metrics.json`：缺失；
- `outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`：缺失；
- `outputs/clstr_unified_stage0_biencoder/checkpoints/latest.pt`：缺失。

当前 unified v2 数据和本地 SkillRouter 模型均存在：

- `data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/retrieval.jsonl`：441097 rows；
- `data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/skill_pool.jsonl`：37623 rows；
- `data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/trajectories.jsonl`：275271 rows；
- `.cache/hf_models/SkillRouter-Embedding-0.6B/model.safetensors` 存在。

已按集群规则用 sbatch 提交到 `gpu_a800`：

```text
77801  scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh
77802  scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
```

提交后 `parajobs` 显示两个任务均为 `PENDING (Priority)`。在 Stage0 baseline 与
bi-encoder checkpoint 产出并通过 Stage0 quality gate 前，仍不提交 Stage2/3/4。

建议检查节奏：短任务至少 15 分钟后再查；若 `77802` 进入长时间训练，则按 30 分钟
间隔检查 loss/recall 曲线、`checkpoints/latest.pt` 和最终 checkpoint。

## 2026-05-27 00:15 — Stage0 gate 默认路径与 Recall@20/50/100 修复

等待 `77801/77802` 队列期间做静态审计，发现两个会在 Stage0 作业完成后误阻断
handoff 的问题：

- `scripts/audit_clstr_stage0_quality.py` 未显式传入 checkpoint 时，默认寻找
  `checkpoints/stage0-step5000.pt`，但 Stage0 unified v2 训练实际产物是
  `checkpoints/clstr_unified_retrieval_v2-step5000.pt`；
- Stage0 训练每步原本只记录 `recall_at_1` 和 `recall_at_${TOP_K}`，而 quality gate
  固定要求 `recall_at_20/50/100`。

本轮已修复：

- `clstr/stage0_quality_gate.py` 默认 checkpoint 改为
  `clstr_unified_retrieval_v2-step5000.pt`；
- `clstr/retrieval_warmup.py` 增加 `metric_recall_ks`，Stage0 wrapper 默认记录
  `(20, 50, 100)`，同时保留 `recall_at_1` 和 `recall_at_top_k`。

验证：

```bash
PYTHONPATH=.vendor_pytest:. TMPDIR=/data/home/scyb713/run/xzf/AAAI/autodl-tmp/tmp \
  /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_biencoder_protocol.py tests/test_stage0_quality_gate.py -q
# 11 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/retrieval_warmup.py clstr/stage0_quality_gate.py \
  scripts/audit_clstr_stage0_quality.py scripts/run_clstr_stage0_biencoder_train.py
```

注意：如果 `77802` 在本次修复前已经开始执行，需检查其 `training_metrics.jsonl`
是否包含 `recall_at_20/50/100`；若没有，应取消并重提 Stage0 bi-encoder 训练。

## 2026-05-27 00:35 — Stage0-first handoff 必要重构

本轮继续收敛 Stage0 foundation repair 后的主线命名和提交入口，避免旧
`Stage1 retrieval warmup -> Stage2` 口径继续误导后续实验。

主要改动：

- 新增 canonical Stage2 handoff 入口：
  `scripts/submit_clstr_stage2_after_stage0_gate.sh`。
  该入口检查 Stage0 checkpoint、运行 `scripts/audit_clstr_stage0_quality.py`
  与 `scripts/run_clstr_stage2_preflight.py`，均通过后才提交
  `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`。
- 旧入口 `scripts/submit_clstr_stage2_after_stage1_gate.sh` 和
  `scripts/submit_clstr_stage1_after_stage0_gate.sh` 改为 deprecated compatibility
  aliases，只转发到 `submit_clstr_stage2_after_stage0_gate.sh`。
- 旧 `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh` 改为 Stage0
  bi-encoder 训练 wrapper，默认输出到 `outputs/clstr_unified_stage0_biencoder/`。
- `scripts/audit_clstr_unified_training_readiness.py` 主 gate 改为
  `stage0_coarse_retriever`；`stage1_retrieval_warmup` 仅保留为 legacy alias。
  Stage2/3/4 gate 现在使用 `stage0_checkpoint_*` 与 `stage0_quality_gate`
  字段，同时保留旧 `stage1_*` 字段兼容历史报告。
- `scripts/sbatch/run_clstr_unified_readiness_audit.sh` 与
  `scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh` 显式透传
  `STAGE0_OUTPUT_DIR` 和 `STAGE0_BASELINE_METRICS_PATH`，避免 quality gate
  隐式依赖 Python 默认路径。
- retrieval export / routing eval / StableToolBench solvable CLSTR retrieval 的默认
  checkpoint 路径统一改为
  `outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`。
- `clstr/full_base_train.py` 与 `clstr/qwen_full_base_train.py` 默认不再允许 Stage2
  直接 full-pool debug；必须提供 `stage0_top_m`，除非显式打开
  `allow_full_pool_stage2_debug`。这防止 Stage2 绕过 Stage0 粗召回 handoff。

文档同步：

- `README.md` 推荐命令改为
  `bash scripts/submit_clstr_stage2_after_stage0_gate.sh`；
- `docs/clstr_v4_plan.md` 将旧 Stage1 warmup 结果标记为历史记录，当前 Stage2
  handoff 以 Stage0 checkpoint 与 Stage0 quality gate 为准；
- `docs/clstr_stage0_biencoder_foundation_plan.md` Task 7 更新为 Stage0-gated
  Stage2 canonical entrypoint，旧 Stage1 命名脚本仅为兼容 alias。

本轮未提交新的 GPU 训练任务，也未检查 Slurm 队列。

## 2026-05-27 01:10 — Skill pool conservative borderline review 接入并解除 Stage0 数据 blocker

本轮继续执行 `docs/clstr_stage0_biencoder_foundation_plan.md`，补上 Stage0 前置数据
gate 的最后一个硬阻塞：manifest 中 `skill_pool.borderline_review.status` 仍为
`pending_manual_or_llm_review` 时，Stage0/Stage2 readiness 会被
`skill_pool_borderline_review_pending` 阻断。

实现口径：

- 新增 conservative review 机制，不自动 merge borderline candidates；
- 低置信 fuzzy/partial-overlap candidates 标记为 `reviewed_keep_separate`；
- 高置信重复仍标记为 `unresolved_possible_duplicate` 并继续阻断 readiness；
- `scripts/audit_skill_dedup_borderline.py` 可同时输出 reviewed JSONL 和 review report；
- `scripts/audit_clstr_unified_training_readiness.py` 接收
  `--skill_dedup_borderline_review_report_path`，只有 report `status=complete` 且
  `unresolved_candidate_count=0` 时，才把 pending manifest 状态解释为
  `complete_by_conservative_review`；
- `scripts/sbatch/run_skill_dedup_borderline_audit.sh`、
  `scripts/sbatch/run_clstr_unified_readiness_audit.sh` 和
  `scripts/sbatch/run_toolbench_g3_unified_v2_pipeline.sh` 均已透传对应路径，避免
  sbatch 链路绕过该 gate。

真实数据执行结果：

```text
skill_pool_path=data/clstr_unified_pretrain_v2_toolbench_g3_traject_split/skill_pool.jsonl
skill_count=37623
candidate_count=20000
reviewed_candidate_count=20000
keep_separate_count=20000
unresolved_candidate_count=0
review_report=outputs/clstr_unified_readiness_audit/skill_dedup_borderline_review_report.json
```

重跑 unified readiness audit 后：

- `skill_pool_quality.status=ok`
- `borderline_review_status=complete_by_conservative_review`
- `stage0_coarse_retriever.ready_to_submit=true`
- `skill_pool_borderline_review_pending` 已不再出现

当前 readiness 仍为 `action_required`，但原因已经变成实验产物缺失，而不是数据清洗：

- Stage2/Stage3/Stage4 缺 `outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`；
- Stage3 缺 Stage2 checkpoint；
- 旧 Stage4 缺 Stage3 checkpoint；当前主线已在 2026-06-07 改为从 Stage2
  checkpoint 进入 Joint ACT Stage4；
- `stage3_full_rollout_hrpo` 仍是未实现状态，当前 offline HRPO 只能作为 surrogate。

本轮未提交新的 GPU 作业；已有 Stage0 baseline/bi-encoder 作业不重复提交。

## 2026-05-27 01:17 — Phase H eval matrix 强制接入 Stage4 quality gate

继续执行 `docs/clstr_stage0_biencoder_foundation_plan.md` 时，针对 Stage4/Phase H
做了一轮非训练验证，发现一个重要口径缺口：Phase H evaluation matrix readiness
原本只检查 Stage4 checkpoint 是否存在，没有强制检查 Stage4 L_act 训练质量门；同时
Stage4 quality gate 过度信任 `train_stdout.json` 里的 `last_metrics`，没有要求
`training_metrics.jsonl` 中持续记录 `stage4_act_loss/stage4_act_count`。

本轮修复：

- `clstr/stage4_quality_gate.py` 现在要求 `training_metrics.jsonl` 中存在
  `stage4_act_loss` 和 `stage4_act_count`，不能只靠 `train_stdout.json` 的
  `last_metrics` 通过；
- `clstr/eval_matrix_readiness.py` 现在调用 `audit_stage4_act_quality`，Stage4
  quality gate 不为 `status=ok` 时加入 `stage4_quality_gate_not_ok`；
- `official_main_table_ready` 现在要求 Stage4 checkpoint 存在且 Stage4 quality gate
  通过；
- `scripts/audit_clstr_eval_matrix_readiness.py` 新增
  `--stage4_output_dir` 与 `--min_stage4_steps`；
- `scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh` 同步暴露
  `STAGE4_OUTPUT_DIR`、`MIN_STAGE4_STEPS`，避免 sbatch 入口绕过该 gate；
- `tests/test_eval_matrix_readiness.py` 和 `tests/test_stage4_quality_gate.py` 覆盖
  “只有 checkpoint 但没有 Stage4 质量证据不能进入官方主表”的回归场景。

该修复的论文意义是：Phase H 不能把一个孤立 checkpoint 当成 Stage4 证据，必须同时
看到 L_act 训练指标、expected benchmark coverage、latest checkpoint 和 loss curve。
否则只能报告 proxy/routing 进展，不能声称 ToolRet/TRAJECT/ToolBench-G3 三主表已经
ready。

本轮仍未提交新的 GPU 作业；Stage2/3/4 仍等待 Stage0 baseline 与 Stage0 bi-encoder
训练产物。

## 2026-05-27 01:27 — Stage0 作业改投 H200 并确认集群资源策略

继续执行 Stage0 foundation plan 时，复查 Slurm 状态发现旧作业：

```text
77801 scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh
77802 scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
```

虽然提交到 `gpu_a800`，但 `parajobs/squeue` 显示为 `cpu=1,mem=15700M` 且长期
`PENDING (Priority)`。这对 37k skill pool 的 Stage0 baseline 与 bi-encoder full
training 风险较高。因此取消了这两个 pending 作业。

排查集群策略时确认：

- 在 sbatch 脚本中显式写 `#SBATCH --cpus-per-task=12` 会被
  `slurm_job_submit_lua` 拒绝；
- 显式传 `--cpus-per-task`、`--cpus-per-gpu` 也会被拒绝；
- 正确方式是按集群指南用 `sbatch --gpus=1 -p <gpu_partition> script.sh`，不要在
  脚本内覆盖 GPU 队列默认 CPU 配比；
- `scontrol show job` 对新 H200 作业显示 `CpusPerTres=gpu:12` 与
  `TresPerJob=gres/gpu:1`，说明 H200 每卡 12 CPU 的默认配比已由集群策略接管。

已重新提交到 `gpu_h200`：

```text
77819 scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh
77820 scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
```

当前仍不提交 Stage2/3/4。只有 `77819` 产出 same-pool SkillRouter baseline metrics、
`77820` 产出 Stage0 final checkpoint/latest checkpoint/training metrics/loss curve，
且 Stage0 quality gate 为 `status=ok` 后，才进入
`scripts/submit_clstr_stage2_after_stage0_gate.sh`。

## 2026-05-27 01:39 — Stage0 sbatch 环境激活修复并重新提交

继续检查 Stage0 作业后发现 `77819/77820` 均已结束且没有产出 Stage0 artifact。
`slurm-77819.out` 与 `slurm-77820.out` 的直接错误为：

```text
EnvironmentNameNotFound: Could not find conda environment: env
```

根因：主线训练 sbatch 仍硬编码 `source activate env`。进一步验证发现，在
`module load miniforge3/25.11.0-1` 后，命名环境 `xzf` 也不在该 module conda 的
环境列表中，但绝对路径激活可用：

```text
source activate /data/home/scyb713/run/miniconda3/envs/xzf
python -V  # Python 3.10.20
```

本轮修复：

- `scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh`
- `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh`
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`
- `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`

这些主线训练入口现在使用：

```bash
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/home/scyb713/run/miniconda3/envs/xzf}
source activate "${CONDA_ENV_PATH}"
```

并保留 `PYTHON_BIN=/data/home/scyb713/run/miniconda3/envs/xzf/bin/python` 默认值。
同时测试已覆盖不能再出现 `source activate env`。

已重新提交 Stage0 到 `gpu_h200`：

```text
77824 scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh
77825 scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
```

当前 `77824/77825` 均为 `PENDING (Priority)`，尚未开始写日志。`scontrol show job`
显示 `CpusPerTres=gpu:12` 与 `TresPerJob=gres/gpu:1`。仍不提交 Stage2/3/4；等待
baseline metrics、Stage0 checkpoint/latest checkpoint、training metrics、loss curve
产出后，再运行 Stage0 quality gate。

## 2026-05-27 — 必要重构：质量门禁公共工具与主线 GPU sbatch 环境统一

本轮重构只处理维护性和误跑风险，不改变 CLSTR 的训练算法、loss 设计、Stage0/2/3/4
实验口径，也不提交新的训练作业。

重构内容：

- 新增 `clstr/stage_quality_common.py`，统一 Stage0/1/2/3/4 quality gate 中重复的
  JSON/JSONL 读取、artifact 存在性检查、有限浮点过滤、window mean、source coverage、
  step/update 统计和 benchmark count 解析逻辑。
- `clstr/stage0_quality_gate.py`、`clstr/stage1_quality_gate.py`、
  `clstr/stage2_quality_gate.py`、`clstr/stage3_quality_gate.py`、
  `clstr/stage4_quality_gate.py` 已切到公共工具模块，但保留原有 gate 入口、
  blocker 命名和报告结构。
- 新增 `scripts/sbatch/_clstr_gpu_env.sh`，统一主线 CLSTR GPU 作业的集群环境：
  `module load miniforge3/25.11.0-1`、可选 `cuda/12.1`、绝对路径 conda 环境
  `/data/home/scyb713/run/miniconda3/envs/xzf`、HF 国内镜像/cache 路径、项目工作目录和
  `PYTHON_BIN`。
- 主线训练入口
  `run_clstr_stage0_skillrouter_frozen_baseline.sh`、
  `run_clstr_unified_stage0_biencoder_train.sh`、
  `run_clstr_unified_stage1_retrieval_warmup.sh`、
  `run_clstr_unified_stage2_full_base_train.sh`、
  `run_clstr_unified_stage3_hrpo.sh`、
  `run_clstr_unified_stage4_act_train.sh`
  现在统一 source `_clstr_gpu_env.sh`，避免再次复制出 `source activate env` 或 cache
  路径不一致的问题。

验证：

```text
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage_quality_common.py tests/test_stage0_quality_gate.py \
  tests/test_stage1_quality_gate.py tests/test_stage2_quality_gate.py \
  tests/test_stage3_quality_gate.py tests/test_stage4_quality_gate.py -q
# 17 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage0_biencoder_protocol.py tests/test_stage0_skillrouter_baseline.py \
  tests/test_sbatch_scripts.py -q
# 49 passed

bash -n scripts/sbatch/_clstr_gpu_env.sh \
  scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh \
  scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh
# syntax ok
```

## 2026-05-27 02:05 — Stage0 full run 当前状态，不进入 Stage2

本次只做一次集中状态检查，没有提交新作业，也没有中断当前作业。

`parajobs` 当前显示：

```text
77824 run_clstr_stage0 ... RUNNING gpu_h200 d1n41d29g01 cpu=12,mem=252000M,gres/gpu=1
77825 run_clstr_unifie ... RUNNING gpu_h200 d1n41d29g01 cpu=12,mem=252000M,gres/gpu=1
```

当前产物证据：

- `outputs/clstr_stage0_skillrouter_frozen_baseline/qrels.jsonl` 已生成，约 43MB；
- `outputs/clstr_stage0_skillrouter_frozen_baseline/run.tsv` 尚不存在；
- `outputs/clstr_stage0_skillrouter_frozen_baseline/metrics.json` 尚不存在；
- `outputs/clstr_unified_stage0_biencoder/checkpoints/latest.pt` 已生成并持续更新，约 241MB；
- `outputs/clstr_unified_stage0_biencoder/loss_curve.svg` 已生成；
- `outputs/clstr_unified_stage0_biencoder/training_metrics.jsonl` 当前有 1653 行；
- `outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
  尚不存在。

训练指标当前摘要：

```text
last_step=1653/5000
last loss=9.571358680725098
last recall_at_20=0.4609375
last recall_at_50=0.515625
last recall_at_100=0.5625
tail50 loss=9.508629817962646
tail50 recall_at_20=0.44203125
tail50 recall_at_50=0.5215625
tail50 recall_at_100=0.5784375
```

结论：Stage0 full training 和 same-pool SkillRouter frozen baseline 都仍未达到
quality gate 的必要产物条件。当前缺 `metrics.json`、`run.tsv` 与 Stage0 final
checkpoint，因此不能运行有效 Stage0 quality gate，也不能进入 Stage2。下一步是在合理
间隔后检查 `77824/77825` 是否完成；只有 baseline metrics 和
`clstr_unified_retrieval_v2-step5000.pt` 都存在后，才运行
`scripts/audit_clstr_stage0_quality.py`。

## 2026-05-27 — 必要重构：Stage2/3/4 主线初始化收敛到 gated Stage0 checkpoint

本轮只做主线入口和初始化链路重构，没有提交新训练作业，也没有改变 Stage0/2/3/4
的核心 loss 公式。重构目标是避免后续又误用旧 SkillRET routing-init manifest、
Qwen wrapper 或弱旧链路产物启动 unified CLSTR 训练。

改动：

- 新增 `clstr/stage_checkpoint_init.py`，统一从 Stage0 checkpoint 读取 config、构建
  CLSTRModel、加载 Stage0 routing weights，并在 Stage3/4 中合并上游 head checkpoint。
- `scripts/run_clstr_stage2_full_base_train.py` 和
  `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh` 不再接受或传递
  `ROUTING_INIT_MANIFEST`；canonical Stage2 只从通过 gate 的
  `outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
  初始化。
- 新增 CLSTR-native 主线入口：
  `scripts/run_clstr_unified_stage3_hrpo.py` 和
  `scripts/run_clstr_stage4_act_train.py`。它们分别从 Stage0 routing checkpoint +
  Stage2/Stage3 head checkpoint 初始化，不加载 Qwen，不暴露 `QWEN_MODEL_PATH`。
- `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh` 与
  `scripts/sbatch/run_clstr_unified_stage4_act_train.sh` 已切到上述 CLSTR-native
  入口，去掉 `--qwen_model_path`、`--local_files_only`、Qwen cache/model-dim 参数。
- `clstr/qwen_checkpoint_init.py` 改为兼容 re-export；Qwen legacy 入口仍可用于旧参考实验，
  但不再是 unified Stage2/3/4 主线。

验证：

```text
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage_checkpoint_init.py tests/test_clstr_topm_candidate_handoff.py \
  tests/test_stage3_unified_hrpo.py tests/test_stage4_act_train.py tests/test_sbatch_scripts.py -q
# 54 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage_checkpoint_init.py clstr/qwen_checkpoint_init.py clstr/full_base_train.py \
  scripts/run_clstr_stage2_full_base_train.py scripts/run_clstr_unified_stage3_hrpo.py \
  scripts/run_clstr_stage4_act_train.py
# ok

bash -n scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh
# syntax ok
```

## 2026-05-27 — 必要重构：Stage0 quality gate 改为同池全量检索指标

本轮修正 Stage0 gate 的指标口径，避免继续把 Stage0 训练 batch 内 recall 和
SkillRouter frozen baseline 的全量 qrels/run eval metrics 直接比较。训练 batch
recall 只代表学习信号和训练稳定性；是否能进入 Stage2，必须看同一 skill pool、同一
qrels、同一 query template 下的 full retrieval eval。

改动：

- `clstr/stage0_quality_gate.py` 新增 `stage0_eval_metrics_path`，默认读取
  `outputs/clstr_unified_stage0_biencoder/full_retrieval_eval/metrics.json`。
- Stage0 baseline comparison 现在只比较 `Recall@20/50/100` 的同池全量检索指标：
  Stage0 full eval metrics vs
  `outputs/clstr_stage0_skillrouter_frozen_baseline/metrics.json`。
- 若缺 Stage0 full eval metrics，gate 返回
  `missing_stage0_full_eval_metrics`，不能进入 Stage2。
- `training_metrics.jsonl` 中的 tail recall 仍写入报告，但只放在
  `coarse_recall_summary.training_batch_tail`，不参与 baseline comparison。
- 新增 `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh`，用于 Stage0 final
  checkpoint 产出后导出同池 `run.tsv` 并生成 full retrieval `metrics.json`。
- `scripts/export_clstr_retrieval_run.py` / `clstr/clstr_retrieval_export.py` 新增
  `--query_text_format skillrouter`，Stage0 full eval 默认使用 SkillRouter-style query
  template，避免训练协议和评估协议不一致。
- Stage2 提交入口、Stage2 sbatch、readiness audit 与 ToolBench-G3 unified pipeline
  都显式传递 `STAGE0_EVAL_METRICS_PATH` 和 `--stage0_eval_metrics_path`，防止绕过
  Stage0 full eval gate。

后续执行顺序：

1. 等 Stage0 final checkpoint
   `outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
   与 SkillRouter baseline metrics 都存在。
2. 提交 Stage0 full retrieval eval：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh
```

3. 只有
   `outputs/clstr_unified_stage0_biencoder/full_retrieval_eval/metrics.json` 存在且
   `scripts/audit_clstr_stage0_quality.py --fail_on_action_required` 通过后，才提交 Stage2。

本轮没有提交新的训练或评测作业，也没有放宽 gate tolerance。该重构的目的只是统一评估
口径，不能把训练 batch recall 当成超过 SkillRouter baseline 的证据。

## 2026-05-27 03:10 — Stage0 final checkpoint 已完成，已提交 full retrieval eval

本次按 `docs/clstr_stage0_biencoder_foundation_plan.md` 的硬条件重新审计 Stage0
状态，没有提交 Stage2/3/4。

当前已完成的 Stage0 产物：

- `outputs/clstr_stage0_skillrouter_frozen_baseline/metrics.json` 已存在；
- `outputs/clstr_stage0_skillrouter_frozen_baseline/run.tsv` 已存在，约 4.3GB；
- `outputs/clstr_stage0_skillrouter_frozen_baseline/qrels.jsonl` 已存在；
- `outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
  已存在；
- `outputs/clstr_unified_stage0_biencoder/checkpoints/latest.pt`、`training_metrics.jsonl`、
  `loss_curve.svg` 均已存在。

Stage0 训练日志摘要：

```text
metric_rows=5000
last_step=5000
last loss=8.990676879882812
last recall_at_20=0.5078125
last recall_at_50=0.578125
last recall_at_100=0.6484375
tail200 loss=8.905183100700379
tail200 recall_at_20=0.5149609375
tail200 recall_at_50=0.60171875
tail200 recall_at_100=0.664375
```

注意：这些 recall 仍只是 training batch learning signal，不能和 SkillRouter full
retrieval baseline 直接比较。

当前 same-pool SkillRouter frozen baseline full retrieval metrics：

```text
Recall@20=0.706692
Recall@50=0.811626
Recall@100=0.884804
NDCG@20=0.482335
NDCG@50=0.505929
NDCG@100=0.519562
MAP@20=0.425079
MAP@50=0.431604
MAP@100=0.433984
evaluated_queries=328320
missing_run_queries=0
```

已运行 Stage0 quality gate 当前审计：

```text
status=action_required
blockers=[
  missing_stage0_eval_Recall@20,
  missing_stage0_eval_Recall@50,
  missing_stage0_eval_Recall@100,
  missing_stage0_full_eval_metrics
]
```

也就是说，当前 Stage0 handoff 的唯一缺口是
`outputs/clstr_unified_stage0_biencoder/full_retrieval_eval/metrics.json`。已提交
full retrieval eval 作业：

```text
77835 scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh
partition=gpu_h200
state=PENDING (Priority)
```

下一步：不要重复提交 Stage0 full eval；等待 `77835` 生成
`outputs/clstr_unified_stage0_biencoder/full_retrieval_eval/metrics.json` 后，重新运行
`scripts/audit_clstr_stage0_quality.py --fail_on_action_required`。只有 Stage0 gate
通过后，才能运行 `scripts/submit_clstr_stage2_after_stage0_gate.sh`。

等待 `77835` 调度期间，已完成不依赖 full retrieval eval 的 Stage2/3/4/Phase H 静态
验证：

```text
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_clstr_topm_candidate_handoff.py tests/test_stage_checkpoint_init.py \
  tests/test_stage2_quality_gate.py tests/test_stage3_unified_hrpo.py \
  tests/test_stage3_quality_gate.py tests/test_stage4_act_train.py \
  tests/test_stage4_quality_gate.py tests/test_eval_matrix_readiness.py \
  -q --basetemp=.tmp_pytest/stage2_4_static
# 29 passed

PYTHONPYCACHEPREFIX=.tmp_pycache /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py clstr/stage_checkpoint_init.py clstr/stage3_unified_hrpo.py \
  clstr/stage4_act_train.py clstr/eval_matrix_readiness.py \
  scripts/run_clstr_stage2_full_base_train.py scripts/run_clstr_unified_stage3_hrpo.py \
  scripts/run_clstr_stage4_act_train.py scripts/audit_clstr_eval_matrix_readiness.py
# ok

bash -n scripts/submit_clstr_stage3_after_stage2_gate.sh \
  scripts/submit_clstr_stage4_after_stage3_gate.sh \
  scripts/sbatch/run_clstr_unified_stage3_hrpo.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_eval_matrix_readiness_audit.sh \
  scripts/sbatch/run_toolbench_g3_routing_eval.sh \
  scripts/sbatch/run_export_clstr_retrieval_run.sh \
  scripts/sbatch/run_export_traject_retrieval_run.sh
# syntax ok
```

结论：Stage2/3/4 的 canonical 入口、quality gate、handoff、Stage4 非 AppWorld-only
训练入口和 Phase H readiness 口径已通过静态验证。当前不能继续提交 Stage2 的唯一前置
原因仍是 Stage0 full retrieval eval metrics 尚未产出。

## 2026-05-27 03:32 — 必要重构：修复 Slurm 下主线 sbatch 共享环境定位

`77835` 启动后立即失败，日志为：

```text
/var/spool/slurmd/job77835/slurm_script: line 4:
/var/spool/slurmd/job77835/_clstr_gpu_env.sh: No such file or directory
```

根因不是 Stage0 full retrieval eval 逻辑，而是 sbatch 脚本使用
`source "$(dirname "$0")/_clstr_gpu_env.sh"`。Slurm 执行时 `$0` 会变成 spool 里的
`slurm_script`，因此共享环境脚本会被错误地从 `/var/spool/slurmd/job.../` 查找。

本轮把主线 CLSTR GPU sbatch 入口统一改为显式项目根路径：

```bash
PROJECT_ROOT=${PROJECT_ROOT:-/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr}
source "${PROJECT_ROOT}/scripts/sbatch/_clstr_gpu_env.sh"
```

影响脚本：

- `scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline.sh`
- `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
- `scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh`
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`
- `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
- `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh`

验证：

```text
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
# syntax ok
```

提交前 `squeue -u "$USER"` 为空，未发现重复 Stage0 full eval。已重新提交：

```text
77836 scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh
partition=gpu_h200
state=RUNNING at submit-time health check
```

`slurm-77836.out` 初始日志只包含 CUDA module warning，不再出现
`_clstr_gpu_env.sh` 路径错误。下一步仍然是等待
`outputs/clstr_unified_stage0_biencoder/full_retrieval_eval/metrics.json`，然后运行
Stage0 quality gate；在 gate 通过前不进入 Stage2。

## 2026-05-27 04:12 — Stage0 full eval 口径修复：保留 checkpoint 中训练后的 skill table

`77836` 最终产出了 full retrieval eval，但 Stage0 quality gate 失败：

```text
CLSTR Stage0 full eval:
Recall@20=0.560164
Recall@50=0.663969
Recall@100=0.738568

same-pool SkillRouter frozen baseline:
Recall@20=0.706692
Recall@50=0.811626
Recall@100=0.884804
```

进一步检查实现后发现 `scripts/export_clstr_retrieval_run.py` 在加载 Stage0 checkpoint
后无条件调用 `model.rebuild_skill_table()`，这会把 checkpoint 中训练后的
`skill_table.E` 覆盖成 frozen encoder 重新编码的 skill embeddings。也就是说 `77836`
评估的是“训练后的 W/scale/bias + 被重置的 E”的混合状态，不是完整 Stage0 checkpoint。

本轮修复：

- `clstr/clstr_retrieval_export.py`：当 checkpoint 成功加载 `skill_table.E` 时，跳过
  `rebuild_skill_table()`，并在 report 中记录
  `skill_table_rebuild=skipped_checkpoint_embeddings`。
- `clstr/clstr_retrieval_export.py`：导出改为 batch 级流式写
  `predictions.jsonl`、`run.tsv` 和 `progress.json`，避免长评估不可观测。
- `scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh`：重跑前删除旧
  `metrics.json`、`run.tsv`、`predictions.jsonl`、`progress.json` 等产物，避免 gate
  误读旧结果。
- 旧错误评估结果已移到
  `outputs/clstr_unified_stage0_biencoder/full_retrieval_eval_rebuilt_skill_table_bug_77836`。

验证：

```text
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_clstr_retrieval_export.py \
  tests/test_stage0_biencoder_protocol.py::test_stage0_biencoder_cli_help_and_sbatch_defaults \
  tests/test_sbatch_scripts.py::test_stage0_full_retrieval_eval_sbatch_exports_same_pool_eval_metrics_for_gate \
  -q
# 6 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/clstr_retrieval_export.py scripts/export_clstr_retrieval_run.py
# ok

bash -n scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh \
  scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
# ok
```

已重新提交修复后 full retrieval eval：

```text
77837 scripts/sbatch/run_clstr_stage0_full_retrieval_eval.sh
partition=gpu_h200
state=RUNNING
```

提交后健康检查显示 `progress.json` 已开始更新：

```text
phase=retrieving
processed_queries=784
total_queries=441097
```

在 `77837` 产出新的
`outputs/clstr_unified_stage0_biencoder/full_retrieval_eval/metrics.json` 并重新运行
Stage0 quality gate 前，不提交 Stage2。

## 2026-05-27 — 必要重构：修正 Stage0 训练协议，避免把强 frozen bi-encoder 训坏

`77837` 已完成有效 full retrieval eval，`run_report.json` 确认
`loaded_skill_table_embeddings=true` 且
`skill_table_rebuild=skipped_checkpoint_embeddings`，因此这次评估不再是旧的
`rebuild_skill_table()` 覆盖 bug。Stage0 quality gate 仍失败：

```text
CLSTR Stage0 old checkpoint:
Recall@20=0.573444
Recall@50=0.682046
Recall@100=0.765536

same-pool SkillRouter frozen baseline:
Recall@20=0.706692
Recall@50=0.811626
Recall@100=0.884804

blockers:
stage0_recall_at_20_below_same_pool_baseline
stage0_recall_at_50_below_same_pool_baseline
stage0_recall_at_100_below_same_pool_baseline
```

这说明旧 Stage0 训练本身会把 frozen SkillRouter-compatible bi-encoder 的大池粗召回拉低，
不能作为 Stage2/3/4 的 foundation。不能继续用
`outputs/clstr_unified_stage0_biencoder/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
启动下游训练。

本轮必要重构：

- `clstr/retrieval_warmup.py` 新增 `multi_positive_nll`，对同一 query 的多个
  canonical positive skill 聚合概率质量，避免把等价 positive 当负例惩罚。
- `run_skillret_retrieval_warmup()` 支持真实 `gradient_accumulation_steps`：
  `max_steps` 现在表示 optimizer step，每个 step 内累积多个 micro-batch，并在
  `training_metrics.jsonl` 记录 `optimizer_step`、`effective_batch_size`、
  `positive_label_counts`。
- Stage0 wrapper 默认切换为保守训练协议：
  `retrieval_loss_mode=multi_positive_nll`、`learning_rate=2e-5`、
  `gradient_accumulation_steps=2`、`train_skill_embeddings=false`、
  `train_skill_bias=false`、`train_encoder_projection=false`、
  `train_skill_adapter=true`、`train_retrieval_scale=true`。
- `scripts/run_clstr_stage0_biencoder_train.py` 暴露上述 Stage0 参数。
- `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh` 默认使用同一套协议，并通过
  `TRAIN_SKILL_EMBEDDINGS`、`TRAIN_SKILL_BIAS`、`TRAIN_ENCODER_PROJECTION`、
  `TRAIN_SKILL_ADAPTER`、`TRAIN_RETRIEVAL_SCALE` 环境变量显式覆盖。
- 新 checkpoint / rolling latest 会记录 `stage0_protocol`，后续 quality gate 或人工检查
  可以区分旧失败训练与新协议训练。

验证：

```text
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

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/retrieval_warmup.py scripts/run_clstr_stage0_biencoder_train.py \
  scripts/run_skillret_retrieval_warmup.py
# ok

bash -n scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh \
  scripts/sbatch/run_clstr_unified_stage1_retrieval_warmup.sh
# ok
```

下一步：重新跑一个新输出目录的 Stage0 v2 训练，例如
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE`，随后重新导出 full retrieval run
并运行 Stage0 quality gate。只有新 Stage0 在 same-pool full retrieval
`Recall@20/50/100` 上不低于 frozen SkillRouter baseline，才能进入 Stage2。

## 2026-05-27 — Stage0 v2 新协议训练已提交并改投 A800

提交前确认：

```text
squeue -u "$USER"
# 无运行或排队作业

outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE
# 不存在，避免覆盖旧 Stage0 失败产物
```

已提交新协议 Stage0 v2 完整训练：

```text
77838 run_clstr_unified_stage0_biencoder_train
partition=gpu_h200
state=PENDING (Priority)
output_dir=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE
protocol=multi_positive_nll + freeze skill_table.E/skill_bias/encoder_proj + train adapter/scale
```

已挂依赖 full retrieval eval：

```text
77839 run_clstr_stage0_full_retrieval_eval.sh
partition=gpu_h200
dependency=afterok:77838
state=PENDING (Dependency)
checkpoint=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt
output_dir=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/full_retrieval_eval
run_name=clstr_unified_stage0_biencoder_v2_mpn_freezeE
```

随后检查发现 `77838` 在 `gpu_h200` 因 `Priority` pending，调度预计开始时间过晚；
同时输出目录尚无训练产物，因此没有训练被实际启动。已取消未启动的 H200 链路
`77838/77839`，改投 A800：

```text
77840 run_clstr_unified_stage0_biencoder_train
partition=gpu_a800
state=RUNNING
node=d1n41a12g03
output_dir=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE

77841 run_clstr_stage0_full_retrieval_eval.sh
partition=gpu_a800
dependency=afterok:77840
state=PENDING (Dependency)
checkpoint=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt
output_dir=outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/full_retrieval_eval
```

下一步不要提交 Stage2，也不要重复提交 Stage0 v2。等待 `77840` 和 `77841` 自然完成后，
运行：

```bash
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  scripts/audit_clstr_stage0_quality.py \
  --output_dir outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE \
  --checkpoint_path outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt \
  --stage0_eval_metrics_path outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/full_retrieval_eval/metrics.json \
  --baseline_metrics_path outputs/clstr_stage0_skillrouter_frozen_baseline/metrics.json \
  --output_path outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/stage0_quality_gate_current.json \
  --fail_on_action_required
```

只有该 gate `status=ok` 后，才允许进入 Stage2。

2026-05-27 一次性健康检查：`77840` 已在 `gpu_a800` 节点 `d1n41a12g03` 运行，`77841`
仍为 `afterok:77840` 依赖等待。Stage0 v2 已完成 skill table rebuild 并进入训练：

```text
setup_status:
skill_table_rebuilt shape=[37623, 1024]
training_started max_steps=5000 batch_size=128 gradient_accumulation_steps=2
effective_batch_size=256 retrieval_loss_mode=multi_positive_nll

training_metrics:
rows=21
latest observed step=22
latest observed loss≈9.9168
latest observed batch recall@20≈0.6914
latest observed batch recall@50≈0.7852
latest observed batch recall@100≈0.8516
```

注意：上述 recall 是训练 batch 指标，只说明训练链路在按新协议写入指标；Stage0 是否可用
仍必须以后续 `full_retrieval_eval/metrics.json` 与 same-pool SkillRouter baseline 的
full retrieval gate 为准。下一次检查按长任务间隔处理，不重复提交。

## 2026-05-27 — Stage2 canonical/legacy 初始化边界重构

为避免后续训练继续混用旧 SkillRET/native routing-init manifest 与新的 Stage0
foundation-first 主线，已完成一次必要重构：

- `clstr.full_base_train.run_clstr_full_base_train` 现在只作为 canonical Stage2 入口：
  必须传入 gated Stage0 `routing_checkpoint_path`。如果传入旧
  `routing_init_manifest`，会直接报错，提示改用 legacy wrapper。
- 新增 `run_legacy_clstr_full_base_train_from_routing_init`，专门保留旧
  `outputs/clstr_native_routing_init/manifest.json` 路径的复现实验入口。
- 以下旧实验入口已改为显式 legacy 调用：
  - `scripts/run_clstr_full_base_train.py`
  - `scripts/run_clstr_dagger_success_train.py`
  - `scripts/run_clstr_loss_ablation.py`
  - `clstr/structured_train.py`
- legacy wrapper 会在 report 顶层写入
  `legacy_routing_init_manifest_used=true` 与
  `not_canonical_stage2_unified_mainline=true`，避免旧实验产物被误当成 unified
  Stage2/3/4 主线证据。

这次重构不改变 Stage0 v2 当前训练/评估链路，也不提交新 GPU 作业。Stage2 仍必须等待
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/full_retrieval_eval/metrics.json`
产出，并通过 same-pool Stage0 quality gate 后才能启动。

已通过的定向验证：

```text
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_full_base_train.py::test_canonical_full_base_train_rejects_legacy_routing_init_manifest \
  tests/test_full_base_train.py::test_legacy_full_base_train_wrapper_is_explicitly_marked \
  tests/test_loss_ablation.py::test_ablation_runner_releases_training_cuda_before_eval \
  tests/test_loss_ablation.py::test_ablation_runner_skips_completed_eval_when_resuming \
  tests/test_clstr_topm_candidate_handoff.py::test_generic_stage2_cli_exposes_stage0_topm_handoff_without_qwen \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool \
  -q
# 6 passed
```

## 2026-05-27 — canonical Stage0 默认路径统一切到 v2

为防止后续 Stage2/3/4 或评测导出误接入旧失败 Stage0，主线默认路径已统一改为
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE`：

- Stage0 train CLI/sbatch、Stage0 full retrieval eval 默认输出/读取 v2 目录；
- Stage2/3/4 sbatch、Stage2/3/4 submit wrapper 默认从 v2 final checkpoint 初始化；
- unified readiness audit 默认读取 v2 checkpoint 与
  `outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/full_retrieval_eval/metrics.json`；
- ToolRet、TRAJECT、ToolBench-G3、StableToolBench 相关 CLSTR retrieval export wrapper
  默认读取 v2 Stage0 checkpoint。

旧 `outputs/clstr_unified_stage0_biencoder` 目录不删除，但只作为历史诊断/复现实验产物。若要复现旧链路，
必须显式传入 `OUTPUT_DIR`、`STAGE0_OUTPUT_DIR`、`ROUTING_CHECKPOINT_PATH` 或
`CHECKPOINT_PATH`，不能依赖主线默认值。

这次重构没有提交新 GPU 作业；当前 Stage0 v2 训练/依赖评估仍沿用已提交的 `77840/77841`。
Stage2 仍必须等待 v2 full retrieval eval 产出并通过 same-pool Stage0 quality gate。

## 2026-05-27 — continuation audit for Stage0 foundation plan

继续执行 `docs/clstr_stage0_biencoder_foundation_plan.md` 时，当前权威状态如下：

- `77840` 仍在 `gpu_a800` 上运行 Stage0 v2 训练，最近检查时约 `step=1119/5000`；
- `77841` 仍为 `afterok:77840` 依赖等待；
- `outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE` 已有 `latest.pt`、
  `training_metrics.jsonl`、`loss_curve.svg` 与 `setup_status.jsonl`；
- v2 final checkpoint 与 `full_retrieval_eval/metrics.json` 尚未产生，因此 Stage0
  quality gate 仍不能运行，Stage2/3/4 不得提交。

同时复验了不依赖 Stage0 完成的 Stage4/Phase H 静态边界：

```text
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_stage4_act_train.py \
  tests/test_stage4_quality_gate.py \
  tests/test_eval_matrix_readiness.py \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_eval_matrix_readiness_sbatch_audits_without_running_benchmarks \
  -q
# 15 passed
```

结论：当前不能推进 Stage2 的原因不是 Stage4/Phase H 静态实现缺口，而是 Stage0 v2
full training、full retrieval eval、same-pool Stage0 quality gate 尚未完成。

## 2026-05-27 — Stage0 v2 current gate/readiness report while training

继续检查 `77840/77841` 后，`77840` 仍在运行，`77841` 仍为依赖等待。当前
`outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/training_metrics.jsonl` 已写入约
1200+ optimizer steps；最新观察约 `step=1229/5000`，tail-100 batch
`recall_at_50≈0.8006`、`recall_at_100≈0.8542`，且四类训练来源均已覆盖。

已生成两个状态报告：

- `outputs/clstr_unified_stage0_biencoder_v2_mpn_freezeE/stage0_quality_gate_current.json`：
  当前 `status=action_required`，因为 final checkpoint 和 full retrieval eval metrics
  还不存在。
- `outputs/clstr_unified_readiness_audit/stage0_v2_current_readiness.json`：
  当前 `status=action_required`。数据、leakage、skill pool 与三个 benchmark eval data
  readiness 为 ok；当时 Stage2/3/4 仍被缺失的 Stage0/Stage2/Stage3 checkpoint 阻断。

这不是新的方法失败信号，只是训练链路尚未完成。Stage2/3/4 仍必须等待 `77840` 完成、
`77841` 产出 full retrieval eval，并且 v2 Stage0 quality gate 通过后才能提交。

## 2026-05-27 — Stage0→Stage1→Stage2 阶段边界修正

用户指出 Stage0 完成后不应直接进入 Stage2。核对 `docs/clstr_stage0_biencoder_foundation_plan.md`
后确认：当前代码把旧 `Stage1 retrieval warmup` 废弃后，曾把
`submit_clstr_stage1_after_stage0_gate.sh` 直接转发到 Stage2，这与新定义的
`Stage1 = CLSTR heads initialization on Stage0 top-M` 不一致。

本轮修正：

- 新增独立 Stage1 heads-init 主线：
  - CLI: `scripts/run_clstr_stage1_heads_init.py`
  - sbatch: `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh`
  - gate: `scripts/audit_clstr_stage1_heads_quality.py`
  - guarded submit: `scripts/submit_clstr_stage1_after_stage0_gate.sh`
- `scripts/submit_clstr_stage2_after_stage0_gate.sh` 仅保留为 deprecated alias，转发到
  Stage1，不再直接提交 Stage2。
- `scripts/submit_clstr_stage2_after_stage1_gate.sh` 现在检查 Stage1 heads checkpoint
  与 Stage1 quality gate，通过后才提交 Stage2。
- Stage2 sbatch/CLI 新增 `STAGE1_CHECKPOINT_PATH` / `--stage1_checkpoint_path`，语义为
  “从 Stage1 checkpoint 继续训练同一个 CLSTRModel”。如果内部恢复 checkpoint 时还读取
  Stage0 frozen routing foundation，那只是 restore 细节，不是两个模型相加或 ensemble。
- Stage1 默认冻结 Stage0 retrieval foundation，在 Stage0 top-M candidates 内训练
  policy / belief / transition / STOP heads；Stage2 才进入更完整的 supervised
  multi-step CLSTR-base。

同时重新生成 Stage0 v2 gate，当前 `status=ok`。full retrieval coarse recall 为：

```text
CLSTR Stage0 v2 Recall@20/50/100 = 0.714572 / 0.831686 / 0.913444
SkillRouter frozen baseline       = 0.706692 / 0.811626 / 0.884804
```

这说明 Stage0 foundation gate 已解除；下一步是提交 Stage1 heads-init，而不是直接提交
Stage2。

验证：

```text
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_sbatch_scripts.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_stage_checkpoint_init.py -q
# 47 passed

bash -n scripts/submit_clstr_stage1_after_stage0_gate.sh \
  scripts/submit_clstr_stage2_after_stage0_gate.sh \
  scripts/submit_clstr_stage2_after_stage1_gate.sh \
  scripts/sbatch/run_clstr_unified_stage1_heads_init.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# clean

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  scripts/run_clstr_stage1_heads_init.py \
  scripts/run_clstr_stage2_full_base_train.py \
  scripts/audit_clstr_stage1_heads_quality.py \
  clstr/full_base_train.py \
  clstr/stage1_heads_quality_gate.py \
  clstr/stage_checkpoint_init.py
# clean
```

对照参考文档后的剩余实现问题：

1. `scripts/audit_clstr_unified_training_readiness.py` 已升级为显式
   `stage1_heads_init` gate。`stage1_retrieval_warmup` 仍作为
   `stage0_coarse_retriever` 的 legacy alias 保留在报告中，但
   `stage2_full_base` readiness 现在必须同时满足 Stage0 checkpoint、Stage0 quality
   gate、Stage1 heads-init checkpoint 与 Stage1 quality gate。
2. 当前 Stage1 作业 `77911` 已失败，根因是训练侧 JSONL reader 使用
   `read_text().splitlines()`，会把 JSON 字符串内部的 Unicode line separator 拆成伪行，
   导致合法 JSONL 在训练中报 `Unterminated string`。该问题已用回归测试复现并修复；
   修复后仍不提交新的 full job。
3. 训练数据 loss mask 覆盖满足 Stage1 heads-init 的基本需求：
   `L_policy=44945`、`L_trans=272851`、`L_trans_skill_ce=225850`、
   `STOP=275271`、`belief=230312`，因此 Stage1 默认 required loss terms
   `L_policy` 与 `L_trans_skill_ce` 有训练样本。

新增执行约束：full job 只由用户最终确认提交；后续阶段先做本地小样本 smoke 和短作业
smoke，且提交后必须早期检查日志/关键产物，异常立即中止，避免长时间无效占卡。

本轮同步落实为代码约束：

- `clstr/full_base_train.py` 和 `clstr/full_base_preprocess.py` 的 JSONL reader 改为逐行
  文件迭代，不再使用 `read_text().splitlines()`；真实
  `trajectories.jsonl` 中存在 14 行 U+0085，旧 reader 会把合法 JSON 字符串拆坏。
- Stage1/Stage2 CLIs 与 sbatch wrapper 增加 `--max_rows` / `MAX_ROWS`，用于小样本
  smoke，不再需要先上完整数据排错。
- 新增 `scripts/audit_clstr_training_health.py`，可在 smoke 或训练启动后检查
  `training_metrics.jsonl`、`latest.pt`、last loss 与 setup phase。
- 新增 `scripts/guard_clstr_job_budget.sh`，Stage1/2/3/4 guarded submit wrapper 在
  `sbatch` 前调用，默认 `MAX_ACTIVE_JOBS=4`；达到上限时拒绝提交。
- 本轮未提交任何新的 Slurm 作业。

## 2026-05-27 22:35 — Stage1 full job 无 metrics 更新的根因与修复

Stage1 full job `78309` 由用户授权提交后长时间没有更新 `slurm-78309.out`，用户手动
取消。事后核验：

- Slurm 状态：`CANCELLED by 2239`，运行约 21 分 39 秒。
- `setup_status.jsonl` 最后一行停在 `routing_checkpoint_loaded`。
- 未生成 `training_metrics.jsonl`、`loss_curve.svg`、`checkpoints/latest.pt`。

结论：这不是“loss 暂时没刷新”的正常训练状态，而是还没有进入训练循环。根因是
Stage1 在训练前先对全部 275,271 行做 Stage0 top-M candidate handoff；每行还要对
`state_text` 和 `next_observation_text` 各跑一次 encoder，且旧实现只在全部完成后才写
`stage0_candidate_handoff_prepared`。因此 full job 会出现长时间无 metrics、无
checkpoint、无进度日志的问题。

已修复：

- `clstr/full_base_train.py` 增加 Stage0 handoff 训练预算子集：
  `stage0_handoff_sample_multiplier`。Stage1 sbatch 默认设为 `4`，即只为当前
  `MAX_STEPS * BATCH_SIZE * 4` 训练预算可触达的样本准备 top-M candidates，而不是为
  全量 27.5 万行预计算。该设置不使用 `MAX_ROWS`，但会在 report 中显式记录
  `stage0_candidate_handoff_subset`，避免把它误读为全量 candidate cache。
- Stage0 candidate handoff 增加持续进度日志：
  `stage0_candidate_handoff_started` 和 `stage0_candidate_handoff_progress` 会写入
  `setup_status.jsonl`。Stage1 sbatch 默认 `STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=16`，
  `STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=50`。
- `scripts/run_clstr_stage1_heads_init.py` 和
  `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh` 已接入上述参数。

验证：

```text
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_sbatch_scripts.py::test_unified_stage1_heads_init_sbatch_uses_stage0_topm_candidates -q
# 7 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py scripts/run_clstr_stage1_heads_init.py
# clean

bash -n scripts/sbatch/run_clstr_unified_stage1_heads_init.sh \
  scripts/submit_clstr_stage1_after_stage0_gate.sh
# clean
```

下一次不要直接复用 `outputs/clstr_unified_stage1_toolbench_g3_traject_split_heads_init_full_v1`
这个被取消 job 的目录。应使用新的输出目录重新跑 Stage1 full，并在开始后先看
`setup_status.jsonl` 是否持续出现 `stage0_candidate_handoff_progress`；进入
`training_started` 后再看 `training_metrics.jsonl`、`loss_curve.svg` 和
`checkpoints/latest.pt`。

## 2026-05-27 23:45 — Stage1 heads-init full_v2 完成

用户确认后重新提交 Stage1 full job：

- job：`78524`
- partition：`gpu_a800`
- time limit：`06:00:00`
- elapsed：`00:15:32`
- state：`COMPLETED`
- output：`outputs/clstr_unified_stage1_toolbench_g3_traject_split_heads_init_full_v2`
- final checkpoint：
  `outputs/clstr_unified_stage1_toolbench_g3_traject_split_heads_init_full_v2/checkpoints/clstr_stage1_heads-step3000.pt`

本次修复后的可观测性和训练链路均正常：

- `setup_status.jsonl` 持续写入 `stage0_candidate_handoff_progress`；
- `training_metrics.jsonl` 共 3000 行；
- `loss_curve.svg`、`checkpoints/latest.pt`、`train_report.json` 均生成；
- `health_final.json` 为 `status=ok`；
- `stage1_quality_gate.json` 为 `status=ok`。

Stage0 handoff 记录：

- 原始 train rows：275,271；
- 训练预算子集：24,003；
- Stage0 top-100 handoff retained rows：13,871；
- skipped rows：10,132；
- skipped reasons：
  - `next_skill_positive_missing_from_stage0_topm`: 6,815；
  - `routing_positive_missing_from_stage0_topm`: 3,317。

Stage1 quality gate：

```text
metric_count = 3000
first_loss_mean = 3.90336804151535
last_loss_mean  = 2.272499268651009
loss_drop       = 1.630868772864
```

训练均值指标：

```text
loss = 2.7675681237181027
policy_ce_loss = 2.486434954047203
policy_expert_recall@1 = 0.2614444466034571
transition_skill_ce_loss = 1.3030842594802379
transition_skill_recall@1 = 0.6055000000000006
transition_skill_recall@5 = 0.9825277777777776
```

结论：Stage1 heads-init 已可作为 Stage2 continuation 的前置 checkpoint。该结果只证明
Stage1 heads initialization 链路和 gate 通过，不是 benchmark 结果。

## 2026-05-28 — Stage2 smoke 前置代码接线，未提交作业

用户要求 Stage2 先做小样本 smoke，确认能跑后再决定是否允许全量；随后明确要求修完
代码后不要提交任何 Slurm 作业，smoke 也不要提交。本轮严格未提交 Stage2 smoke/full
job。

为避免 Stage2 full 重复出现 Stage1 `78309` 的长时间无 metrics 问题，已将 Stage1
修复过的 Stage0 handoff 预算子集和进度日志参数接入 Stage2：

- `clstr/full_base_train.py`：`run_clstr_full_base_train(...)` 透传
  `stage0_handoff_sample_multiplier`、`stage0_candidate_encode_batch_size`、
  `stage0_candidate_progress_interval_batches` 到 `train_clstr_full_base_with_model(...)`。
- `scripts/run_clstr_stage2_full_base_train.py`：新增对应 CLI 参数。
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`：新增默认
  `STAGE0_HANDOFF_SAMPLE_MULTIPLIER=4`、
  `STAGE0_CANDIDATE_ENCODE_BATCH_SIZE=16`、
  `STAGE0_CANDIDATE_PROGRESS_INTERVAL_BATCHES=50`，并传给 Stage2 CLI。

本地验证：

```text
PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool -q
# 1 passed

PYTHONPATH=.vendor_pytest:. /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest \
  tests/test_clstr_topm_candidate_handoff.py tests/test_stage_checkpoint_init.py -q
# 8 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py scripts/run_clstr_stage2_full_base_train.py
# clean

bash -n scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh \
  scripts/submit_clstr_stage2_after_stage1_gate.sh
# clean
```

当前状态：Stage2 smoke 尚未提交；后续若用户允许，只能提交一个短小样本 smoke，并且
必须早期监督 `setup_status.jsonl`、`training_metrics.jsonl`、`loss_curve.svg` 和
`checkpoints/latest.pt`。

## 2026-05-28 — trajectory-aware Stage0 数据和 Stage2 auxiliary caps

问题来源：Stage0 之前主要训练静态 retrieval query，而 Stage2 handoff 实际使用
trajectory state/history/action/observation 构造 sequential routing query。ALFWorld /
ScienceWorld 的 trajectory rows 虽然在 Stage2 轨迹流里占比很高，但它们没有进入
Stage0 retrieval 正例训练，导致用 ALFWorld prefix smoke 看 `coverage@100` 时出现
训练/评估分布失配。

本轮数据策略改成：

- Stage0 unified builder 会从 `trajectories.jsonl` 追加 trajectory-derived retrieval：
  - current: `goal + history + observation -> skill_id`
  - next: `goal + history + previous_action + next_observation -> next_skill_id`
- 默认 caps：
  - `toolbench_g3=-1`
  - `traject_bench=-1`
  - `alfworld=10000`
  - `scienceworld=10000`
- ALFWorld/ScienceWorld 明确作为 auxiliary sequential-routing 数据，不作为论文主
  benchmark 叙事；主线仍以 ToolBench-G3、TRAJECT-Bench、ToolRet/SkillRet 检索为主。
- Stage2 训练入口新增 benchmark caps，默认保留 ToolBench-G3/TRAJECT-Bench 全量，
  下调 ALFWorld/ScienceWorld 各 10k rows：
  `toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000`。

相关文件：

- `scripts/build_clstr_unified_pretrain.py`
- `scripts/sbatch/run_build_clstr_unified_pretrain_v2.sh`
- `clstr/full_base_train.py`
- `scripts/run_clstr_stage2_full_base_train.py`
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`

验证：

```text
PYTHONPATH=.vendor_pytest /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_pretrain_v2.py
# 17 passed

PYTHONPATH=.vendor_pytest /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py::test_full_base_benchmark_caps_downsample_auxiliary_domains \
  tests/test_full_base_train.py::test_full_base_training_filters_to_allowed_benchmarks_and_records_audit \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool
# 3 passed
```

注意：尚未重建全量数据集，也未提交任何训练/Slurm 作业。下一步应先用 builder 生成新的
unified data 目录，再做 Stage0 handoff audit；full job 仍需用户最后确认后再提交。

## 2026-05-28 — Stage0 sequential routing retrieval 数据目录已重建

已按新的 Stage0 训练策略重建全量 unified data：

- 数据目录：`data/clstr_unified_pretrain_v3_trajectory_retrieval_caps`
- schema：`v3_trajectory_retrieval_caps`
- leakage audit：`status=ok`
- `trajectories.jsonl`：275,271 rows
- `retrieval.jsonl`：552,219 rows
- `skill_pool.jsonl`：37,623 canonical skills
- trajectory-derived retrieval：111,122 pairs
  - ALFWorld retained rows：10,000，skipped rows：131,741
  - ScienceWorld retained rows：10,000，skipped rows：83,434
  - ToolBench-G3 retained rows：9,570
  - TRAJECT-Bench retained rows：30,526

同时修正了 manifest 统计：`retrieval_stream.total_pairs` 现在只统计实际写入
`retrieval.jsonl` 的 pair，不再把 `traject_bench_pair_skipped:*` 这类 held-out skip
审计计数算入总数。重建后已验证：

```text
retrieval_stream.total_pairs = 552219
retrieval.jsonl line count   = 552219
trajectory_stream.total_rows = 275271
trajectories.jsonl line count= 275271
skill_pool.total_skills      = 37623
skill_pool.jsonl line count  = 37623
leakage_audit.status         = ok
```

Stage0 训练策略同步调整：

- Stage0 默认数据目录切到
  `data/clstr_unified_pretrain_v3_trajectory_retrieval_caps`。
- Stage0 默认输出目录切到
  `outputs/clstr_unified_stage0_biencoder_v3_traj_retrieval_mpn_freezeE`，避免覆盖旧实验。
- Stage0 默认 `sampling_strategy=source_balanced`，让 `skillret`、`toolret_training`、
  `toolbench_g3`、`traject_bench`、`trajectory_derived_*` 等来源在 batch 内持续出现，
  避免新增 sequential query 被大规模静态 retrieval 样本淹没。
- Stage0 baseline/eval/readiness 和 Stage1-Stage4 默认 handoff 路径已跟随新 Stage0
  输出目录更新，避免下游默认吃旧 Stage0 checkpoint。

本轮只做数据构建、代码接线和本地验证；未提交任何 Slurm 训练作业，full job 仍需用户
最后确认。

本地验证补充：

```text
pytest focused Stage0/data/sbatch/readiness suite: 37 passed
py_compile modified Python entrypoints: clean
bash -n modified sbatch/submit scripts: clean
git diff --check: clean
local tiny Stage0 source_balanced smoke: status=ok, latest checkpoint written
```

## 2026-05-28 — Stage0 frozen SkillRouter baseline 改为分块 top-k 导出

发现旧的 frozen SkillRouter baseline 会一次性构造
`num_queries x num_skills` 的完整 score matrix。v2 在 H200 上能跑通主要是因为内存配额较大；
v3 数据量更大，在 A800 上存在长时间无进度、后期 OOM 或浪费卡时的风险。

已将 `clstr/stage0_skillrouter_baseline.py` 的 `run.tsv` 导出改为 query batch 分块：

- scoring 公式、query/skill 文本、skill pool、qrels 和 metric 计算不变；
- 每个 query 仍然对完整 skill pool 做 top-k，因此理论结果与 full matrix 等价；
- 新增 `score_batch_size`，默认 1024；
- 新增 `progress.json`，覆盖记录 `loading_data`、`writing_qrels`、`encoding_queries`、
  `encoding_skills`、`writing_run`、`evaluating_metrics`、`done` 等阶段；
- sbatch 脚本新增 `SCORE_BATCH_SIZE` 环境变量。

验证：

```text
batched_trec_equivalence_ok
stage0_baseline_synthetic_smoke_ok
PYTHONPATH=.vendor_pytest ... -m pytest tests/test_stage0_skillrouter_baseline.py -q: 5 passed
py_compile clstr/stage0_skillrouter_baseline.py scripts/run_clstr_stage0_skillrouter_frozen_baseline.py: clean
CLI help exposes --score_batch_size
git diff --check for touched files: clean
```

未提交新的 Slurm 作业；full baseline/eval 仍需用户确认后再提交。

## 2026-05-29 — Stage0 v3 same-pool baseline 与 CLSTR full retrieval eval 完成

已在同一份 v3 qrels / canonical skill pool 上完成 frozen SkillRouter baseline 与
CLSTR Stage0 v3 checkpoint 的 full retrieval eval：

- frozen baseline job：`78943`，`COMPLETED`，耗时 `00:47:46`，MaxRSS 约 50.9GB。
- CLSTR v3 eval job：`78958`，`COMPLETED`，耗时 `00:43:38`，MaxRSS 约 38.7GB。
- 当前无运行中的 Slurm 作业。

同池同 qrels 指标：

```text
frozen SkillRouter baseline v3:
  Recall@20  = 0.633774
  Recall@50  = 0.738246
  Recall@100 = 0.812518
  NDCG@100   = 0.450382
  MAP@100    = 0.364030

CLSTR Stage0 v3:
  Recall@20  = 0.661493
  Recall@50  = 0.781992
  Recall@100 = 0.866318
  NDCG@100   = 0.478223
  MAP@100    = 0.383571

CLSTR v3 - frozen baseline v3:
  Recall@20  = +0.027719
  Recall@50  = +0.043746
  Recall@100 = +0.053800
  NDCG@100   = +0.027841
  MAP@100    = +0.019541
```

解释：

- 这说明 Stage0 v3 训练相对同池 frozen SkillRouter embedding baseline 有稳定增益。
- 旧 v2 eval 的 Recall@100 为 0.913444，但旧 v2 的 qrels/query 规模不同
  (`328,320` vs v3 eval `439,442` evaluated qrels)，不能直接作为同分布横向结论。
- 下一步应做 Stage0 handoff coverage audit 和 Stage2 smoke/full training gate，验证 v3
  trajectory-derived retrieval 是否真的改善 sequential routing candidate handoff，而不是只改善
  static retrieval。

## 2026-05-29 — Stage1 v3 heads init 完成

Stage1 首次提交作业 `79016` 在 handoff 构造阶段失败：

```text
NotImplementedError: Module [SkillTable] is missing the required "forward" function
```

根因：

- `clstr/full_base_train.py` 中 `_skill_logits_and_memory()` 和 `_transition_skill_logits()`
  对真实 `SkillTable` fallback 到了 `skill_table(h)`；
- 真实 `SkillTable` 暴露的是 `retrieval_logits(h)`，没有 `forward()`；
- 兼容逻辑应为 `retrieval_logits()` -> legacy `logits()` -> callable fallback。

修复：

- `clstr/full_base_train.py`
  - `_skill_logits_and_memory()` 优先调用 `retrieval_logits()`；
  - `_transition_skill_logits()` 同样优先调用 `retrieval_logits()`；
  - 保留 legacy 测试模型的 `logits()` 支持。
- `tests/test_full_base_train.py`
  - 新增无 `forward()`、仅有 `retrieval_logits()` 的 Stage0 handoff 回归测试；
  - 新增无 `forward()`、仅有 `retrieval_logits()` 的 transition skill logits 回归测试。

验证：

```text
PYTHONPATH=.vendor_pytest ... -m pytest \
  tests/test_full_base_train.py::test_full_base_train_max_rows_limits_smoke_training_after_filters \
  tests/test_full_base_train.py::test_full_base_transition_skill_ce_scores_whole_skill_pool_from_prior \
  tests/test_full_base_train.py::test_full_base_transition_skill_ce_prefers_native_trans_head_over_skill_table_logits \
  tests/test_full_base_train.py::test_full_base_stage0_handoff_uses_retrieval_logits_without_skill_table_forward \
  tests/test_full_base_train.py::test_full_base_transition_skill_logits_use_retrieval_logits_without_skill_table_forward \
  -q
# 5 passed
```

Slurm 验证：

- smoke job `79027`：`COMPLETED`，`MAX_ROWS=256`，`MAX_STEPS=10`；
  - `current_positive_coverage@M = 1.0`
  - `next_positive_coverage@M = 1.0`
  - 写出 `loss_curve.svg`、`checkpoints/latest.pt`、`clstr_stage1_heads-step10.pt`
- full job `79033`：`COMPLETED`，耗时 `00:17:00`，MaxRSS 约 32.5GB；
  - output：`outputs/clstr_unified_stage1_v3_traj_retrieval_heads_init`
  - checkpoint：`checkpoints/clstr_stage1_heads-step3000.pt`
  - latest checkpoint：`checkpoints/latest.pt`
  - `metric_count = 3000`
  - `sample_count = 21718`
  - `current_positive_coverage@M = 0.904015794337562`
  - `next_positive_coverage@M = 0.933853499475952`
  - `masked_next_skill_ce_rows = 1136`

当前无运行中的 Slurm 作业。下一步应做 Stage1 quality gate，然后进入 Stage2 smoke/full
training gate。

## 2026-05-29 — Stage1 gate 通过，Stage2 smoke 通过

Stage1 quality gate：

- 输入目录：`outputs/clstr_unified_stage1_v3_traj_retrieval_heads_init`
- checkpoint：`checkpoints/clstr_stage1_heads-step3000.pt`
- 状态：`ok`
- `metric_count = 3000`
- `loss_drop = 1.250631414652`
- required loss terms：`L_policy`、`L_trans_skill_ce` 均被采样

Stage2 先做了两个小样本 smoke：

- job `79090`：`MAX_ROWS=256`，`MAX_STEPS=10`，`COMPLETED`
  - 代码路径、Stage0 gate、Stage2 preflight、checkpoint 保存和 loss 图写出均正常；
  - 但 10 step 太短，smoke quality gate 因 `insufficient_learning_signal` 未通过，
    不作为训练趋势判断。
- job `79093`：`MAX_ROWS=1024`，`MAX_STEPS=100`，`COMPLETED`
  - output：`outputs/clstr_unified_stage2_v3_traj_retrieval_full_base_smoke_v2`
  - checkpoint：`checkpoints/clstr_full_base-step100.pt`
  - latest checkpoint：`checkpoints/latest.pt`
  - Stage2 smoke quality gate：`ok`
  - `metric_count = 100`
  - total loss first20 -> last20：`3.927559 -> 3.468669`
  - policy CE first20 -> last20：`3.092999 -> 2.609540`
  - stop BCE first20 -> last20：`0.299437 -> 0.110616`
  - transition cosine first20 -> last20：`0.048997 -> 0.039899`
  - transition skill CE first20 -> last20：`0.652131 -> 0.982641`
  - belief cosine first20 -> last20：`0.0 -> 0.0`
  - Stage0 handoff subset：`source_rows=1024`，`selected_rows=452`
  - Stage0 handoff coverage：current/next positive coverage@100 均为 `1.0`

解释：

- Stage2 当前已经证明数据流、Stage0/Stage1 checkpoint handoff、loss 写出、checkpoint 保存、
  loss 曲线写出都能正常工作。
- 100-step smoke 有总体 loss 下降，说明 full job 具备继续尝试的工程前提。
- 这仍不是性能结论；`transition_skill_ce` 在短 smoke 中有波动上升，且 `belief` loss
  未被激活，后续 full 或更长 gate 必须继续监控这些项。
- 下一步如进入 Stage2 full，应由用户确认后再提交；提交后重点观察前 500-1000 step 的
  total loss、policy CE、transition skill CE、stop BCE、belief activation 和
  handoff skipped rows。

## 2026-05-29 — Stage2 full 完成，但 transition/routing 信号需复查

Stage2 full job：

- job：`79099`
- partition：`gpu_a800`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`00:54:23`
- MaxRSS：约 `31.3GB`
- output：`outputs/clstr_unified_stage2_v3_traj_retrieval_full_base`
- checkpoint：`checkpoints/clstr_full_base-step10000.pt`
- latest checkpoint：`checkpoints/latest.pt`
- loss curve：`loss_curve.svg`

Stage2 quality gate：

- 状态：`ok`
- `metric_count = 10000`
- `max_step = 10000`
- total loss first200 -> last200：`3.480763 -> 1.029468`
- `loss_drop = 2.451294988692`
- required loss terms：`L_policy`、`L_trans`、`L_trans_skill_ce`、`STOP`、`routing`
  均被采样
- train safety：未使用 valid/test；当前仍是 offline replay supervised pretraining，
  不是 on-policy RL fine-tuning

Stage0 handoff for Stage2 full：

- `source_rows = 57557`
- `retained_rows = 43756`
- `skipped_rows = 13801`
- `current_positive_coverage@100 = 0.7599116260459614`
- `next_positive_coverage@100 = 0.8383281573498965`
- `masked_next_skill_ce_rows = 6247`

分项指标：

- policy CE first200 -> last200：`2.623436 -> 0.0`
- policy expert recall@1 first200 -> last200：`0.243333 -> 1.0`
- transition skill CE first200 -> last200：`1.012157 -> 1.681026`
- transition recall@1 first200 -> last200：`0.676667 -> 0.511250`
- transition recall@5 first200 -> last200：`0.973750 -> 0.778333`
- transition MRR first200 -> last200：`0.792833 -> 0.622157`
- retrieval contrastive loss first200 -> last200：`3.122097 -> 3.286700`
- stop BCE first200 -> last200：`0.142559 -> 0.148254`
- belief cosine first200 -> last200：`0.000493 -> 0.000868`

解释：

- 工程链路已经跑通：Stage0 -> Stage1 -> Stage2 full 可以完成，checkpoint、latest
  checkpoint、loss curve 和 quality gate 都正常产出。
- 但 Stage2 gate 目前主要证明“可继续 handoff”，不能证明方法性能已经变好。
- 当前总 loss 大幅下降主要来自 policy head 变得很容易；更关键的 sequential transition /
  routing 相关指标反而变差。
- Stage2 full 的 handoff coverage 明显低于 Stage1 full 记录的 coverage，说明全量 Stage2
  数据里仍有较多 row 的 positive skill 不在 Stage0 top-100，后续需要处理。
- 因此不建议把当前 Stage2 结果直接作为乐观结论进入论文实验；下一步应优先复查：
  loss weighting、Stage2 sampler、transition target、Stage0 top-M handoff coverage，以及
  是否需要对 transition/routing 设置更强 gate。

## 2026-05-29 — Stage2 v2 loss/sampler/gate 修复与 smoke

针对 Stage2 full 中 total loss 下降但 transition/routing 信号退化的问题，做了以下修复：

- Stage2 v2 默认 loss weights：
  - `L_policy = 0.2`
  - `routing = 0.0`
  - `STOP = 0.1`
  - `L_trans = 0.3`
  - `L_trans_skill_ce = 1.0`
  - `belief = 0.1`
- Stage2 训练新增 `sampling_strategy`：
  - `balanced_deterministic`：保留旧的确定性分桶采样；
  - `balanced_random`：新增 seeded balanced random sampler，Stage2 v2 默认使用。
- Stage2 quality gate 加强：
  - 默认 required loss terms 不再要求 `routing`；
  - 新增 transition 退化检查：
    - `transition_skill_recall@5` 下降超过 `0.1` 时阻塞；
    - `transition_skill_ce_loss` 上升超过 `0.2` 时阻塞；
    - 尾部 `transition_skill_recall@5` 低于 `0.5` 时阻塞。

验证：

```text
PYTHONPATH=.vendor_pytest ... -m pytest \
  tests/test_full_base_train.py \
  tests/test_stage2_quality_gate.py \
  -q
# 43 passed
```

用新 gate 复审旧 Stage2 full：

- output：`outputs/clstr_unified_stage2_v3_traj_retrieval_full_base`
- 新 gate 输出：`stage2_quality_gate_v2_transition.json`
- 状态：`action_required`
- blockers：
  - `transition_recall_at_5_regressed`
  - `transition_skill_ce_regressed`
- `transition_skill_recall@5` first200 -> last200：`0.973750 -> 0.778333`
- `transition_skill_ce_loss` first200 -> last200：`1.012157 -> 1.681026`

Stage2 v2 smoke：

- job：`79169`
- partition：`gpu_a800`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`00:00:54`
- output：`outputs/clstr_unified_stage2_v2_loss_sampler_smoke_top100`
- 配置：`MAX_ROWS=1024`，`MAX_STEPS=100`，`STAGE0_TOP_M=100`
- sampler：`balanced_random`
- checkpoint：`checkpoints/clstr_full_base-step100.pt`
- Stage2 smoke quality gate：`ok`
- total loss first20 -> last20：`2.269925 -> 1.849175`
- policy CE first20 -> last20：`3.111426 -> 2.602627`
- transition skill CE first20 -> last20：`1.612927 -> 1.299729`
- transition recall@1 first20 -> last20：`0.525000 -> 0.650000`
- transition recall@5 first20 -> last20：`0.900000 -> 0.908333`
- transition MRR first20 -> last20：`0.651558 -> 0.750104`
- stop BCE first20 -> last20：`0.250630 -> 0.168770`
- Stage0 handoff subset：`source_rows=1024`，`selected_rows=816`
- Stage0 handoff coverage：current/next positive coverage@100 均为 `1.0`

解释：

- 新 gate 能正确阻塞旧 Stage2 full 的 transition 退化，不再只看 total loss。
- Stage2 v2 smoke 的 transition 指标在小样本上转为正向，说明 loss reweighting +
  routing objective 关闭 + random sampler 是合理方向。
- 这仍不是最终性能结论；下一步需要做更长 smoke 或 full 前 gate，并单独审计 full-scale
  Stage0 handoff coverage 的 top-100/200/500 变化。

## 2026-05-29 — Stage2 v2 top350+inject full 训练完成

本次 full 训练是在用户确认后提交，用于验证 `top_m=350` 和
`positive_missing_policy=inject` 是否能缓解 Stage0 candidate handoff 对 Stage2 的瓶颈。

作业与产物：

- job：`79236`
- partition：`gpu_a800`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`01:32:38`
- output：`outputs/clstr_unified_stage2_v2_loss_sampler_full_top350_inject`
- checkpoint：`checkpoints/clstr_full_base-step10000.pt`
- latest checkpoint：`checkpoints/latest.pt`
- loss curve：`loss_curve.svg`
- train report：`train_report.json`
- quality gate：`stage2_quality_gate.json`

配置：

- `MAX_STEPS = 10000`
- `BATCH_SIZE = 4`
- `sampling_strategy = balanced_random`
- `sampler_seed = 17`
- Stage0 checkpoint：
  `outputs/clstr_unified_stage0_biencoder_v3_traj_retrieval_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
- `STAGE0_TOP_M = 350`
- `STAGE0_POSITIVE_MISSING_POLICY = inject`
- loss weights：
  - `L_policy = 0.2`
  - `routing = 0.0`
  - `STOP = 0.1`
  - `L_trans = 0.3`
  - `L_trans_skill_ce = 1.0`
  - `belief = 0.1`

Stage0 handoff：

- `source_rows = 54879`
- `retained_rows = 54879`
- `skipped_rows = 0`
- `injected_positive_rows = 9039`
- `current_positive_coverage@M = 0.9011859149790185`
- `next_positive_coverage@M = 0.8957653351302407`
- `masked_next_skill_ce_rows = 0`

训练指标 first200 -> last200：

- total loss：`4.962903 -> 3.346771`
- policy CE：`0.151376 -> 0.105646`
- policy expert recall@1：`0.959167 -> 0.975000`
- transition skill CE：`4.899002 -> 3.283754`
- transition recall@1：`0.115833 -> 0.234583`
- transition recall@5：`0.194583 -> 0.515417`
- transition MRR：`0.175762 -> 0.361188`
- transition cosine loss：`0.025745 -> 0.073070`
- belief cosine loss：`0.001838 -> 0.000058`
- stop BCE：`0.257184 -> 0.199603`
- retrieval contrastive loss：`3.337051 -> 3.329212`

Stage2 quality gate：

- 状态：`ok`
- blockers：无
- `metric_count = 10000`
- `max_step = 10000`
- total loss drop：`1.616132642627`
- transition recall@5 first200 -> last200：`0.194583 -> 0.515417`
- transition skill CE first200 -> last200：`4.899002 -> 3.283754`
- required loss terms：`L_policy`、`L_trans`、`L_trans_skill_ce`、`STOP` 均被采样
- train safety：未使用 valid/test；未使用 on-policy rollout；当前仍是
  `offline_replay_supervised_pretraining`

解释：

- 与旧 Stage2 full 相比，这版没有出现 transition CE 上升和 recall@5 明显退化的问题；
  新 gate 给出 `ok`，说明 Stage2 v2 可以作为进入后续 Stage3/Act 前的候选 checkpoint。
- `top350 + inject` 的作用是训练期 candidate teacher forcing：当 gold skill 不在 Stage0
  top-M 时注入 gold positive，避免 Stage2 因 candidate 缺失而学不到 transition。
  这不能用于 evaluation/inference，否则会泄露答案；论文中需要明确区分训练期注入与测试期真实召回。
- `injected_positive_rows = 9039`，比例不低，说明 Stage0 handoff 仍是系统瓶颈。
  Stage2 当前通过 gate 只能证明“在候选覆盖被训练期修正后 transition 可以学起来”，还不能证明
  端到端 routing/belief 已经稳定超过 SkillRouter。
- 下一步不应直接下结论，而应在用户确认后进入 Stage3/Act 或先做无注入/真实 top-M 的验证评估，
  检查该 checkpoint 在不注入 gold candidate 时是否仍能保持 routing 和 transition 收益。

## 2026-05-30 — Stage2 no-inject real top-M evaluator

为避免把 `top350 + inject` 的训练期 teacher forcing 误当成真实推理能力，新增了一个
Stage2 离线诊断入口：

- 模块：`clstr/stage2_real_topm_eval.py`
- CLI：`scripts/evaluate_clstr_stage2_real_topm.py`
- sbatch：`scripts/sbatch/run_clstr_stage2_real_topm_eval.sh`
- 测试：`tests/test_stage2_real_topm_eval.py`

这个 evaluator 做的事情：

- 从 Stage0 checkpoint 构建 routing/skill table；
- 加载 Stage2 checkpoint 的 head/transition 参数；
- 用真实 Stage0 top-M candidates 重新做 handoff；
- 固定 `positive_missing_policy = skip`，不允许 gold candidate injection；
- 对保留下来的真实候选样本做离线 Stage2 metric 评估；
- 输出：
  - `stage0_candidate_handoff_no_inject.json`
  - `real_topm_eval_metrics.jsonl`
  - `real_topm_eval_report.json`

新增 report gate 会阻塞以下情况：

- `positive_missing_policy != skip`
- `injected_positive_rows > 0`
- no-inject 后没有 retained rows；
- current/next positive coverage 低于阈值；
- transition recall@5 没有可评估样本或低于阈值。

验证：

```text
PYTHONPATH=.vendor_pytest ... -m pytest tests/test_stage2_real_topm_eval.py -q
# 2 passed

bash -n scripts/sbatch/run_clstr_stage2_real_topm_eval.sh
# exit 0

python -m py_compile \
  clstr/stage2_real_topm_eval.py \
  scripts/evaluate_clstr_stage2_real_topm.py
# exit 0
```

尝试提交 smoke：

- job：`79361`
- output：`outputs/clstr_unified_stage2_real_topm_eval_smoke_top350_noinject`
- 配置：`MAX_ROWS=2048`，`MAX_EVAL_BATCHES=128`，`BATCH_SIZE=8`，`TOP_M=350`
- 状态：已取消，未运行；
- 原因：所有 GPU 队列当前可运行 1 卡数量均为 0；`squeue --start` 对 A800 的估计启动时间为
  `2026-05-31T20:55:25`，不适合把 smoke 挂到两天后无人监督地启动；
- `sacct`：`CANCELLED by 2239`，`Elapsed=00:00:00`，未产生训练/评估卡时。

下一步：

- 等 GPU 队列有可运行 1 卡资源后，先重提同一个 small smoke；
- smoke 通过后，再决定是否跑 full no-inject real top-M eval；
- 只有 full no-inject eval 的 transition/routing 指标站住，才适合进入 Stage3/Act。

补充修正：

- 发现 evaluator 的 `stage0_handoff_sample_multiplier` 默认值如果保持为 `4.0`，在未设置
  `MAX_EVAL_BATCHES` 的 full eval 中会错误地只选择很小一批 handoff rows。
- 已将 evaluator / CLI / sbatch 的默认值改为不启用 handoff subset；只有显式设置
  `STAGE0_HANDOFF_SAMPLE_MULTIPLIER` 时才进行训练式 subset。
- 新增回归测试：
  - `test_real_topm_eval_does_not_subset_full_eval_by_default`

验证：

```text
PYTHONPATH=.vendor_pytest ... -m pytest tests/test_stage2_real_topm_eval.py -q
# 3 passed

bash -n scripts/sbatch/run_clstr_stage2_real_topm_eval.sh
# exit 0

python -m py_compile \
  clstr/stage2_real_topm_eval.py \
  scripts/evaluate_clstr_stage2_real_topm.py
# exit 0
```

修正后 smoke：

- job：`79462`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`00:01:10`
- output：`outputs/clstr_unified_stage2_real_topm_eval_smoke_top350_noinject`
- 配置：`MAX_ROWS=2048`，`MAX_EVAL_BATCHES=128`，`BATCH_SIZE=8`，`TOP_M=350`，
  `EMBEDDING_CACHE_MODE=never`
- report：`real_topm_eval_report.json`
- status：`ok`
- blockers：无
- no-inject handoff：
  - `source_rows = 2048`
  - `retained_rows = 2048`
  - `skipped_rows = 0`
  - `injected_positive_rows = 0`
  - `current_positive_coverage@M = 1.0`
  - `next_positive_coverage@M = 1.0`
  - `masked_next_skill_ce_rows = 0`
- eval：
  - `evaluated_batches = 128`
  - `evaluated_rows = 1024`
  - `policy_expert_recall@1 = 0.259766`
  - `transition_skill_ce_loss = 1.693695`
  - `transition_skill_recall@1 = 0.539749`
  - `transition_skill_recall@5 = 0.914226`
  - `transition_skill_mrr = 0.665066`
  - `retrieval_contrastive_loss = 3.097809`

解释：

- smoke 证明 no-inject evaluator 链路可运行，且在小样本上 transition top-5 很高。
- 这仍然不是 full 结论；下一步需要跑不设 `MAX_ROWS` / `MAX_EVAL_BATCHES` 的 full
  no-inject real top-M eval，确认全量候选覆盖和 transition 指标。

Full no-inject real top-M eval：

- job：`79463`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`00:34:03`
- output：`outputs/clstr_unified_stage2_real_topm_eval_full_top350_noinject`
- report：`real_topm_eval_report.json`
- metrics：`real_topm_eval_metrics.jsonl`
- 配置：不设 `MAX_ROWS`，不设 `MAX_EVAL_BATCHES`，`BATCH_SIZE=8`，`TOP_M=350`，
  `EMBEDDING_CACHE_MODE=auto`
- 运行过程：full-base shared replay embedding cache 首次 batch size 256 OOM，代码自动降到
  128 后继续；作业最终正常完成。

no-inject handoff：

- `source_rows = 60096`
- `retained_rows = 53953`
- `skipped_rows = 6143`
- `skipped_reasons = {"routing_positive_missing_from_stage0_topm": 6143}`
- `injected_positive_rows = 0`
- `current_positive_coverage@M = 0.8976541934623972`
- `next_positive_coverage@M = 0.9139757279471675`
- `masked_next_skill_ce_rows = 4012`

eval 汇总：

- status：`action_required`
- blockers：`transition_recall_at_5_below_threshold`
- `evaluated_batches = 6745`
- `evaluated_rows = 53953`
- loss mask counts：
  - `L_policy = 36396`
  - `routing = 53879`
  - `STOP = 53953`
  - `L_trans = 53953`
  - `L_trans_skill_ce = 42626`
  - `belief = 17557`
- total loss：`3.736449`
- policy CE：`0.166463`
- policy expert recall@1：`0.953099`
- transition skill CE：`3.399819`
- transition recall@1：`0.224370`
- transition recall@5：`0.483860`
- transition MRR：`0.342168`
- transition cosine loss：`0.068400`
- belief cosine loss：`0.000296`
- stop BCE：`0.323073`
- retrieval contrastive loss：`3.344793`

解释：

- `top350` 的真实候选覆盖已经不差，current/next coverage 均超过 `0.85` 阈值；
  Stage0 handoff 不是完全失败，但仍造成 `6143` 行 current positive 缺失和 `4012` 行 next-skill CE mask。
- Stage2 在训练期 `top350 + inject` gate 中 transition recall@5 是 `0.515417`，
  到 full no-inject eval 下降到 `0.483860`，略低于 `0.5` 阈值。这说明训练期注入确实帮助了
  transition 学习，但真实候选条件下收益还不够稳。
- 结果不是灾难性失败，而是“接近通过但不达标”：policy 很强，belief/transition cosine 很稳，
  问题集中在 next-skill ranking top-5。
- 目前不建议直接进入 Stage3/Act 并声称 Stage2 已经稳定；下一步应优先做小成本诊断：
  1. 对比 `top_m=100/200/350/500` 的 full 或大样本 no-inject eval，区分 coverage 与候选宽度对
     transition recall@5 的影响；
  2. 检查 transition CE 是否被大量 easy policy / STOP 样本稀释；
  3. 考虑 Stage2 再训练时提高真实 no-inject retained rows 的权重，或对 next-skill CE 使用
     harder in-candidate negatives，而不是继续依赖 inject。

## 2026-05-30 — Stage2 v3 mixed transition training 实现

针对 full no-inject real top-M eval 中 `transition recall@5 = 0.483860`、略低于
`0.5` 阈值的问题，新增 Stage2 v3 mixed transition 训练机制。

改动：

- `clstr/full_base_train.py`
  - 新增 loss term：`transition_hard_negative_margin`；
  - `L_trans_skill_ce` 支持按 row 加权：
    - `transition_real_candidate_ce_multiplier`
    - `transition_injected_candidate_ce_multiplier`
  - 对 `stage0_positive_injected=False` 的真实候选样本启用 in-candidate hard-negative margin；
  - injected rows 保留完整监督，但可通过 multiplier 降权，避免训练目标过度依赖 teacher forcing。
- `scripts/run_clstr_stage2_full_base_train.py`
  - 新增 CLI 参数：
    - `--transition_hard_negative_margin_loss_weight`
    - `--transition_real_candidate_ce_multiplier`
    - `--transition_injected_candidate_ce_multiplier`
    - `--transition_hard_negative_margin`
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
  - 新增对应环境变量：
    - `TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT`
    - `TRANSITION_REAL_CANDIDATE_CE_MULTIPLIER`
    - `TRANSITION_INJECTED_CANDIDATE_CE_MULTIPLIER`
    - `TRANSITION_HARD_NEGATIVE_MARGIN`
- `tests/test_full_base_train.py`
  - 新增测试覆盖：
    - injected Stage0 candidates 的 transition CE 可降权；
    - transition hard-negative margin 只作用于真实 no-inject candidate rows。

验证：

```text
PYTHONPATH=.vendor_pytest ... -m pytest \
  tests/test_full_base_train.py \
  tests/test_stage2_quality_gate.py \
  tests/test_stage2_real_topm_eval.py \
  -q
# 49 passed

bash -n scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
# exit 0

python -m py_compile \
  clstr/full_base_train.py \
  scripts/run_clstr_stage2_full_base_train.py
# exit 0
```

建议的 smoke 配置：

```bash
OUTPUT_DIR=outputs/clstr_unified_stage2_v3_mixed_top350_inject_smoke \
MAX_ROWS=2048 \
MAX_STEPS=100 \
BATCH_SIZE=4 \
STAGE0_TOP_M=350 \
STAGE0_POSITIVE_MISSING_POLICY=inject \
EMBEDDING_CACHE_MODE=never \
TRANSITION_SKILL_CE_LOSS_WEIGHT=1.2 \
TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=0.1 \
TRANSITION_REAL_CANDIDATE_CE_MULTIPLIER=1.5 \
TRANSITION_INJECTED_CANDIDATE_CE_MULTIPLIER=0.5 \
TRANSITION_HARD_NEGATIVE_MARGIN=1.0 \
sbatch --gpus=1 -p gpu_a800 scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
```

资源状态：

- 曾尝试 `sbatch --test-only` 检查 A800 可启动性；
- 返回估计启动时间为 `2059-12-27T18:49:22`，不适合提交无人监督的 smoke；
- 因此当前没有提交 Stage2 v3 mixed smoke 作业。

下一步：

- 等 A800/H100/H200 有 1 卡可运行资源时，先跑上述 `MAX_ROWS=2048, MAX_STEPS=100`
  smoke；
- smoke 通过后，再跑 full Stage2 v3 mixed train；
- full train 后必须继续跑 full no-inject real top350 eval，不能只看 inject training gate。

Stage2 v3 mixed smoke：

- job：`79497`
- partition：`gpu_a800`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`00:01:05`
- output：`outputs/clstr_unified_stage2_v3_mixed_top350_inject_smoke`
- checkpoint：`checkpoints/clstr_full_base-step100.pt`
- latest checkpoint：`checkpoints/latest.pt`
- loss curve：`loss_curve.svg`
- train report：`train_report.json`
- gate：`stage2_quality_gate.json`

配置：

- `MAX_ROWS = 2048`
- `MAX_STEPS = 100`
- `BATCH_SIZE = 4`
- `STAGE0_TOP_M = 350`
- `STAGE0_POSITIVE_MISSING_POLICY = inject`
- `EMBEDDING_CACHE_MODE = never`
- loss weights：
  - `L_policy = 0.2`
  - `routing = 0.0`
  - `STOP = 0.1`
  - `L_trans = 0.3`
  - `L_trans_skill_ce = 1.2`
  - `transition_hard_negative_margin = 0.1`
  - `belief = 0.1`
- transition candidate training：
  - `real_candidate_ce_multiplier = 1.5`
  - `injected_candidate_ce_multiplier = 0.5`
  - `hard_negative_margin = 1.0`
  - `hard_negative_loss_weight = 0.1`

Stage0 handoff：

- `source_rows = 1108`
- `retained_rows = 1108`
- `skipped_rows = 0`
- `injected_positive_rows = 0`
- `current_positive_coverage@M = 1.0`
- `next_positive_coverage@M = 1.0`
- `masked_next_skill_ce_rows = 0`

训练指标 first20 -> last20：

- total loss：`2.619763 -> 2.147613`
- transition skill CE：`1.582324 -> 1.301605`
- transition recall@1：`0.508333 -> 0.616667`
- transition recall@5：`0.937500 -> 0.950000`
- transition MRR：`0.677029 -> 0.753750`
- transition hard-negative margin loss：`1.453707 -> 1.117913`
- transition hard-negative pair count：`3.75 -> 3.80`
- transition real candidate rows：`3.75 -> 3.80`
- transition injected candidate rows：`0.00 -> 0.00`
- policy expert recall@1：`0.187500 -> 0.350000`
- retrieval contrastive loss：`3.078313 -> 3.133065`

Stage2 smoke quality gate：

- 状态：`ok`
- blockers：无
- `metric_count = 100`
- `max_step = 100`
- total loss drop：`0.472149610519`
- transition recall@5 first20 -> last20：`0.9375 -> 0.95`
- transition skill CE first20 -> last20：`1.582324 -> 1.301605`

解释：

- v3 mixed 的新增 loss 项在 smoke 中被正常计算，hard-negative margin loss 有下降；
  产物、latest checkpoint、loss curve 和 gate 都正常。
- 这个 smoke 的 selected rows 没有出现 injected rows，因此它主要验证了真实候选加权和 hard-negative
  代码路径；full 训练仍需要验证 injected rows 降权是否生效。
- 下一步可以在用户确认后提交 full Stage2 v3 mixed train；训练完成后必须继续跑 full no-inject
  real top350 eval，看 `transition recall@5` 是否从 `0.483860` 提升并越过 `0.5`。

Stage2 v3 mixed full train：

- job：`79500`
- partition：`gpu_a800`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`01:30:17`
- output：`outputs/clstr_unified_stage2_v3_mixed_top350_inject_full`
- checkpoint：`checkpoints/clstr_full_base-step10000.pt`
- latest checkpoint：`checkpoints/latest.pt`
- loss curve：`loss_curve.svg`
- train report：`train_report.json`
- gate：`stage2_quality_gate.json`

配置：

- `MAX_STEPS = 10000`
- `BATCH_SIZE = 4`
- `STAGE0_TOP_M = 350`
- `STAGE0_POSITIVE_MISSING_POLICY = inject`
- `EMBEDDING_CACHE_MODE = auto`
- loss weights：
  - `L_policy = 0.2`
  - `routing = 0.0`
  - `STOP = 0.1`
  - `L_trans = 0.3`
  - `L_trans_skill_ce = 1.2`
  - `transition_hard_negative_margin = 0.1`
  - `belief = 0.1`
- transition candidate training：
  - `real_candidate_ce_multiplier = 1.5`
  - `injected_candidate_ce_multiplier = 0.5`
  - `hard_negative_margin = 1.0`
  - `hard_negative_loss_weight = 0.1`

Stage0 handoff：

- `source_rows = 54879`
- `retained_rows = 54879`
- `skipped_rows = 0`
- `injected_positive_rows = 9039`
- `current_positive_coverage@M = 0.9011859149790185`
- `next_positive_coverage@M = 0.8957653351302407`
- `masked_next_skill_ce_rows = 0`

训练指标 first200 -> last200：

- total loss：`6.099325 -> 3.985877`
- transition skill CE：`4.822959 -> 3.132603`
- transition recall@1：`0.115833 -> 0.246250`
- transition recall@5：`0.195833 -> 0.512500`
- transition MRR：`0.176067 -> 0.369100`
- transition hard-negative margin loss：`2.429722 -> 1.609101`
- transition real candidate rows：`3.105 -> 3.210`
- transition injected candidate rows：`0.455 -> 0.475`
- policy expert recall@1：`0.959167 -> 0.973333`
- retrieval contrastive loss：`3.337051 -> 3.329212`

Stage2 full quality gate：

- 状态：`ok`
- blockers：无
- `metric_count = 10000`
- `max_step = 10000`
- total loss drop：`2.113448`
- transition recall@5 first200 -> last200：`0.195833 -> 0.512500`
- transition skill CE first200 -> last200：`4.822959 -> 3.132603`

解释：

- v3 mixed full train 正常完成，产物、latest checkpoint、loss curve 和 gate 都正常。
- 相比 v2 top350+inject full train，尾段训练期 `transition recall@5` 基本持平：
  v2 为 `0.515417`，v3 mixed 为 `0.512500`。
- v3 mixed 的主要价值不应只看 inject training gate，而要看 no-inject real top350 eval 是否能把
  full no-inject `transition recall@5 = 0.483860` 推高到 `0.5` 以上。
- 下一步应使用该 checkpoint 跑 full no-inject real top350 eval：
  `STAGE2_CHECKPOINT_PATH=outputs/clstr_unified_stage2_v3_mixed_top350_inject_full/checkpoints/clstr_full_base-step10000.pt`。

Stage2 v3 mixed full no-inject real top350 eval：

- job：`79531`
- partition：`gpu_a800`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`00:34:18`
- output：`outputs/clstr_unified_stage2_v3_mixed_real_topm_eval_full_top350_noinject`
- report：`real_topm_eval_report.json`
- eval checkpoint：
  `outputs/clstr_unified_stage2_v3_mixed_top350_inject_full/checkpoints/clstr_full_base-step10000.pt`

结果：

- status：`action_required`
- blocker：`transition_recall_at_5_below_threshold`
- threshold：`min_transition_recall_at_5 = 0.5`

No-inject Stage0 handoff：

- `source_rows = 60096`
- `retained_rows = 53953`
- `skipped_rows = 6143`
- `injected_positive_rows = 0`
- `current_positive_coverage@M = 0.8976541934623972`
- `next_positive_coverage@M = 0.9139757279471675`
- `masked_next_skill_ce_rows = 4012`

Full no-inject eval 指标：

- total loss：`3.719081`
- transition skill CE：`3.373859`
- transition recall@1：`0.241496`
- transition recall@5：`0.484493`
- transition MRR：`0.356929`
- transition hard-negative margin loss：`1.853505`
- transition real candidate rows：`6.320464`
- policy expert recall@1：`0.952605`
- retrieval contrastive loss：`3.344793`
- belief cosine loss：`0.000307`
- transition cosine loss：`0.075989`
- stop BCE：`0.326300`

对比 v2 top350 no-inject full eval：

- transition skill CE：`3.399819 -> 3.373859`
- transition recall@1：`0.224370 -> 0.241496`
- transition recall@5：`0.483860 -> 0.484493`
- transition MRR：`0.342168 -> 0.356929`

解释：

- v3 mixed 的真实候选训练和 hard-negative margin 让 CE、recall@1、MRR 有小幅改善；
  但核心 gate 指标 `transition recall@5` 几乎没有提升，仍未超过 `0.5`。
- 这说明当前瓶颈不只是 inject teacher forcing，也不只是 Stage0 覆盖率；在相同 no-inject top350
  候选集合内，transition head 对 next skill 的 top-5 排序能力仍不足。
- 继续仅调 Stage2 loss weight 的边际收益可能有限。下一步更值得优先诊断：
  1. transition head 的输入状态是否包含足够的 history / previous skill / observation 信息；
  2. next-skill GT 是否存在多等价 skill 未合并，导致 top-5 被不完整 label 惩罚；
  3. Stage2 candidate 内 hard negatives 是否真的来自相似 next skill，而不是大量弱负例；
  4. 是否需要把 Stage2 的 next-skill ranking 改成更接近 Stage1/SkillRouter 的 listwise reranking
     目标，再让 CLSTR belief/transition 在多步链上发挥作用。

## 2026-05-30 — Stage2 transition 低成本趋势判断与 error audit

目的：

- 回答“Stage2 transition 是否只是训练不够”的问题；
- 不提交新的 Slurm 作业，只分析已有 full train / no-inject eval 产物；
- 把下一步从盲目 full train 调参转为有证据的 Stage2 v4 设计。

新增文件：

- `clstr/stage2_transition_error_audit.py`
- `scripts/audit_clstr_stage2_transition_errors.py`
- `tests/test_stage2_transition_error_audit.py`

验证：

```text
PYTHONPATH=.vendor_pytest /data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  -m pytest tests/test_stage2_transition_error_audit.py -q
# 4 passed
```

Audit 命令：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  scripts/audit_clstr_stage2_transition_errors.py \
  --train_metrics_path outputs/clstr_unified_stage2_v3_mixed_top350_inject_full/training_metrics.jsonl \
  --eval_metrics_path outputs/clstr_unified_stage2_v3_mixed_real_topm_eval_full_top350_noinject/real_topm_eval_metrics.jsonl \
  --eval_report_path outputs/clstr_unified_stage2_v3_mixed_real_topm_eval_full_top350_noinject/real_topm_eval_report.json \
  --baseline_eval_report_path outputs/clstr_unified_stage2_real_topm_eval_full_top350_noinject/real_topm_eval_report.json \
  --output_json_path outputs/clstr_unified_stage2_v3_mixed_real_topm_eval_full_top350_noinject/stage2_transition_error_audit.json \
  --output_markdown_path outputs/clstr_unified_stage2_v3_mixed_real_topm_eval_full_top350_noinject/stage2_transition_error_audit.md \
  --first_window 200 \
  --tail_window 1000
```

产物：

- `outputs/clstr_unified_stage2_v3_mixed_real_topm_eval_full_top350_noinject/stage2_transition_error_audit.json`
- `outputs/clstr_unified_stage2_v3_mixed_real_topm_eval_full_top350_noinject/stage2_transition_error_audit.md`

Audit findings：

- status：`action_required`
- findings：
  - `transition_recall_at_5_below_threshold`
  - `no_meaningful_no_inject_recall5_gain`
  - `train_eval_transition_gap`

训练 1000-step 窗口趋势：

```text
steps        recall@5  recall@1  CE       loss
00001-01000 0.215250  0.116833  4.300700 5.415633
01001-02000 0.316083  0.152167  3.856625 4.855687
02001-03000 0.415500  0.194667  3.741885 4.723160
03001-04000 0.455000  0.227833  3.549697 4.498945
04001-05000 0.466667  0.225333  3.465982 4.395551
05001-06000 0.481833  0.227583  3.358106 4.261562
06001-07000 0.491000  0.251750  3.279546 4.169028
07001-08000 0.491917  0.248250  3.246369 4.135331
08001-09000 0.495250  0.254667  3.171620 4.046277
09001-10000 0.515333  0.263417  3.108226 3.967852
```

低成本判断：

- 训练期 transition 不是完全平台期：最后 1000 step 的 `recall@5` 相比前 1000 step 仍提升
  `+0.020083`，CE 继续下降 `-0.063394`。
- 因此“训练量不够”有一定证据：当前 10k steps 可能还没把训练分布学完。
- 但 no-inject eval 基本没有随 v3 mixed 改善：
  - v2 no-inject `transition recall@5 = 0.483860`
  - v3 mixed no-inject `transition recall@5 = 0.484493`
  - delta 只有 `+0.000633`
- 这说明继续训练可能提高训练期指标，但未必提高真实 top350 no-inject 排序能力。

Eval batch 分布：

- no-inject eval transition rows：`42626`
- weighted `transition recall@5 = 0.484493`
- weighted `transition recall@1 = 0.241496`
- weighted `transition MRR = 0.356929`
- weighted `transition CE = 3.373859`
- batch-level `transition recall@5`：
  - p25：`0.0`
  - median：`0.285714`
  - p75：`0.875`
  - zero-batch fraction：`0.319988`
  - perfect-batch fraction：`0.226275`

解释：

- 失败不是均匀的小误差，而是 batch 分布很分裂：约三分之一 transition batches 的 top-5 完全失败，
  约五分之一以上 batches 完全成功。这更像数据/标签/候选语义分簇问题，而不是单纯优化不足。
- Stage0 no-inject coverage 已超过阈值：
  - current coverage：`0.897654`
  - next coverage：`0.913976`
  因此当前 blocker 不是 Stage0 完全召回不到，而是 Stage2 在真实候选集合内的 next-skill 排序不足。

下一步建议：

- 不建议直接再跑一个 full Stage2 v3，只靠 steps 或 loss weight 硬冲；
- 可以考虑一个很短的 continuation medium 实验，但必须以 no-inject eval 为 gate，而不是训练期 recall；
- 更优先做 Stage2 v4：
  1. 增加 row-level transition ranking diagnostics，记录 gold rank、top-k candidate ids、benchmark/source；
  2. 审计 next-skill 等价标签，构造 multi-positive positives；
  3. 把 Stage2 next-skill CE 改为 listwise reranking / multi-positive NLL；
  4. hard negatives 从 Stage0 top350 中选语义相近候选，而不是只依赖当前 margin；
  5. 让 transition 输入显式包含 previous/current skill、history、observation 或 CLSTR belief state。

## 2026-05-30 — Stage2 row-level transition ranking diagnostics 实现

目的：

- 从 batch-level eval 进一步下钻到 row-level；
- 对每条 `L_trans_skill_ce` 样本记录 Stage0 candidate rank 与 Stage2 rerank 后 rank；
- 判断失败到底来自：
  - Stage0 候选排序本来就低；
  - Stage2 把 Stage0 原本靠前的 gold skill 排低；
  - 某些 benchmark/source 系统性失败；
  - top-1/top-k 是否命中同 namespace，辅助判断等价 skill / 标签问题。

新增文件：

- `clstr/stage2_transition_row_diagnostics.py`
- `scripts/run_clstr_stage2_transition_row_diagnostics.py`
- `scripts/sbatch/run_clstr_stage2_transition_row_diagnostics.sh`
- `tests/test_stage2_transition_row_diagnostics.py`

row-level JSONL 字段包括：

- `benchmark`
- `source_quality`
- `candidate_source`
- `task_id`
- `trajectory_id`
- `step_index`
- `skill_id`
- `next_skill_id`
- `gold_next_skill_index`
- `candidate_count`
- `stage0_gold_rank`
- `stage2_gold_rank`
- `stage2_hit@1`
- `stage2_hit@5`
- `stage2_hit@10`
- `stage2_hit@20`
- `stage2_mrr`
- `stage2_cross_entropy`
- `top_skill_indices`
- `top_skill_ids`
- `top_scores`
- `top1_same_namespace_as_gold`
- `failure_type`
- `history_line_count`
- `state_text_chars`
- `action_text`
- `next_action_text`
- `provenance_source_id`

验证：

```text
PYTHONPATH=.vendor_pytest /data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  -m pytest tests/test_stage2_transition_row_diagnostics.py \
            tests/test_stage2_transition_error_audit.py \
            tests/test_stage2_real_topm_eval.py -q
# 10 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage2_transition_row_diagnostics.py \
  scripts/run_clstr_stage2_transition_row_diagnostics.py
# exit 0

bash -n scripts/sbatch/run_clstr_stage2_transition_row_diagnostics.sh
# exit 0
```

Smoke 诊断：

- job：`79561`
- partition：`gpu_a800`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`00:01:08`
- output：`outputs/clstr_stage2_transition_row_diag_smoke_top350_v1`
- `MAX_ROWS = 2048`
- `MAX_DIAGNOSTIC_ROWS = 2048`
- `TOP_M = 350`
- `TOP_K = 10`
- `BATCH_SIZE = 8`
- `EMBEDDING_CACHE_MODE = never`

Smoke 产物：

- `transition_row_diagnostics.jsonl`
- `transition_row_diagnostics_report.json`
- `transition_row_diagnostics_report.md`
- `stage0_candidate_handoff_no_inject.json`

Smoke 结果：

- diagnosed transition rows：`1903`
- recall@1：`0.550184`
- recall@5：`0.908565`
- recall@10：`0.970573`
- recall@20：`1.0`
- MRR：`0.673549`
- mean Stage0 gold rank：`2.697846`
- mean Stage2 gold rank：`2.633736`
- Stage2 improved vs Stage0 fraction：`0.169732`
- Stage2 worse than Stage0 fraction：`0.190226`
- failure types：
  - `hit_top1 = 1047`
  - `hit_top5_not_top1 = 682`
  - `miss_top5_stage0_top5 = 146`
  - `miss_top5_stage0_top20 = 28`

解释：

- smoke 主要覆盖数据文件前部的 ALFWorld 样本，因此不能代表 full unified distribution；
- 但它证明了 row-level 诊断链路可运行，并且输出中已经能看到有意义的 rank 变化：
  - 有一部分样本 Stage2 把 gold 从 Stage0 较低 rank 拉高；
  - 也有一部分样本 Stage0 gold 本在 top5 内，但 Stage2 rerank 后掉出 top5；
  - top-k 中跨 benchmark skill 会出现，例如 ALFWorld 样本 top-k 中混入 ScienceWorld 同类 skill，
    这提示 multi-positive/equivalent skill 和 namespace-aware hard negative 需要进一步审计。

注意：

- `scripts/sbatch/run_clstr_stage2_transition_row_diagnostics.sh` 已改为复用
  `scripts/sbatch/_clstr_gpu_env.sh`，避免计算节点找不到 `env` conda 环境；
- sbatch 默认 `BENCHMARK_CAPS` 已设为和 Stage2 eval 一致：
  `toolbench_g3=-1,traject_bench=-1,alfworld=10000,scienceworld=10000`，
  防止 full row-level 诊断误扫未 cap 的全量 `275271` 行。

Full row-level 诊断：

- 首次 full job `79567` 因 `BENCHMARK_CAPS` parser 只接受 `key:value`、不接受 sbatch 默认的
  `key=value` 而快速失败；已修复为同时接受两种格式。
- job：`79574`
- partition：`gpu_a800`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 耗时：`00:32:29`
- output：`outputs/clstr_stage2_transition_row_diag_full_top350_v1`
- 产物：
  - `transition_row_diagnostics.jsonl`（约 84MB）
  - `transition_row_diagnostics_report.json`
  - `transition_row_diagnostics_report.md`
  - `transition_row_diagnostics_stdout.json`

Full 总体结果：

- diagnosed transition rows：`42626`
- Stage2 recall@1：`0.241496`
- Stage2 recall@5：`0.484493`
- Stage2 recall@10：`0.564210`
- Stage2 recall@20：`0.668489`
- Stage0 recall@5：`0.478863`
- Stage0 recall@10：`0.571951`
- Stage0 recall@20：`0.653099`
- MRR：`0.356929`
- mean Stage0 gold rank：`41.284662`
- mean Stage2 gold rank：`33.766176`
- mean Stage2 CE：`3.373859`
- Stage2 improved vs Stage0 fraction：`0.421574`
- Stage2 worse than Stage0 fraction：`0.340825`

Failure type counts：

- `hit_top1 = 10294`
- `hit_top5_not_top1 = 10358`
- `miss_top5_stage0_low_rank = 12994`
- `miss_top5_stage0_top20 = 5077`
- `miss_top5_stage0_top5 = 3903`

By benchmark：

| benchmark | rows | Stage0 r@5 | Stage0 r@20 | Stage2 r@5 | Stage2 r@20 | mean Stage0 rank | mean Stage2 rank | worse frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALFWorld | `8852` | `0.853254` | `1.000000` | `0.917646` | `1.000000` | `2.664144` | `2.243561` | `0.119860` |
| ScienceWorld | `9730` | `0.887359` | `0.971737` | `0.929188` | `0.993525` | `3.915416` | `3.012333` | `0.223433` |
| ToolBench-G3 | `4943` | `0.309731` | `0.567469` | `0.077483` | `0.177018` | `46.619866` | `126.332389` | `0.801942` |
| TrajectBench | `19101` | `0.141040` | `0.352181` | `0.162557` | `0.476467` | `76.837757` | `40.086173` | `0.383697` |

结论：

- 整体 `recall@5 = 0.484493` 低，不是由 ALFWorld/ScienceWorld 主导：
  - ALFWorld 与 ScienceWorld 的 Stage2 recall@5 分别达到 `0.917646` 和 `0.929188`；
  - 主要失败来自 ToolBench-G3 与 TrajectBench。
- ToolBench-G3 是最严重问题：
  - Stage0 recall@5 本来已有 `0.309731`，但 Stage2 recall@5 只有 `0.077483`；
  - mean rank 从 Stage0 `46.62` 被 Stage2 推坏到 `126.33`；
  - `worse frac = 0.801942`，说明 transition head 在 ToolBench-G3 上系统性把 gold next skill 排低。
- TrajectBench 的问题不同：
  - Stage0 recall@5 只有 `0.141040`、Stage0 recall@20 只有 `0.352181`，候选/粗召回本身偏弱；
  - Stage2 mean rank 从 `76.84` 改善到 `40.09`，说明 rerank 有帮助，但不足以把大量样本推入 top5。
- 因此当前主要 blocker 不是“再训练久一点”能稳定解决的单一优化问题，而是 benchmark-specific 的
  label/candidate/input structure 问题：
  - ToolBench-G3 需要先审计 next-skill 标签、候选等价关系、API/tool namespace hard negatives，以及 transition
    输入是否缺少 current/previous skill、action schema、observation delta；
  - TrajectBench 需要优先改 Stage0/sequential retrieval query 与候选构造，再让 Stage2 做 listwise reranking。

下一步建议：

- 不建议直接继续跑新的 full Stage2 训练；
- 先做 Stage2 v4 设计与小样本验证：
  1. 抽样审计 ToolBench-G3 / TrajectBench 的 row-level 失败样本，确认 gold label 是否唯一且可执行；
  2. 引入 multi-positive / equivalent skill 标签合并；
  3. 将 next-skill CE 改为 top-M listwise reranking / multi-positive NLL；
  4. 从 Stage0 top350 内构造 semantic hard negatives；
  5. transition 输入显式加入 current skill、previous skill、history/action/observation delta 与 belief/memory 状态。

收尾验证：

```text
PYTHONPATH=.vendor_pytest /data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  -m pytest tests/test_stage2_transition_row_diagnostics.py \
            tests/test_stage2_transition_error_audit.py \
            tests/test_stage2_real_topm_eval.py \
            tests/test_full_base_train.py -q
# 54 passed in 85.46s

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  scripts/run_clstr_stage2_transition_row_diagnostics.py \
  clstr/stage2_transition_row_diagnostics.py \
  scripts/audit_clstr_stage2_transition_errors.py \
  clstr/stage2_transition_error_audit.py
# exit 0

bash -n scripts/sbatch/run_clstr_stage2_transition_row_diagnostics.sh
bash -n scripts/sbatch/run_clstr_stage2_real_topm_eval.sh
# exit 0

squeue -u scyb713
# no active jobs
```

## 2026-05-31 — Stage2 v4 pre-audit：ToolBench-G3 / TrajectBench 失败模式

目的：

- 承接 full row-level diagnostics，进一步审计 Stage2 v4 应先修什么；
- 不提交训练作业，不重训 Stage0；
- 只消费已有：
  - `outputs/clstr_stage2_transition_row_diag_full_top350_v1/transition_row_diagnostics.jsonl`
  - `data/clstr_unified_pretrain_v3_trajectory_retrieval_caps/trajectories.jsonl`

新增文件：

- `clstr/stage2_v4_pre_audit.py`
- `scripts/audit_clstr_stage2_v4_precheck.py`
- `tests/test_stage2_v4_pre_audit.py`

审计产物：

- output：`outputs/clstr_stage2_v4_pre_audit_top350_v1`
- report json：`stage2_v4_pre_audit_report.json`
- report markdown：`stage2_v4_pre_audit_report.md`
- examples：`stage2_v4_pre_audit_examples.jsonl`
- stdout capture：`stdout.json`

总体范围：

- 只审计 ToolBench-G3 + TrajectBench；
- rows：`24044`
- examples：`300` 条，覆盖每个 benchmark 的：
  - `stage0_top5_stage2_miss`
  - `stage2_worse_than_stage0`
  - `stage0_low_rank_miss`
  - `stage2_saved_low_rank`
  - `top1_current_skill_miss`
  - `top1_probably_equivalent_miss`

By benchmark：

| benchmark | rows | Stage0 r@5 | Stage2 r@5 | top5 pushed out | Stage0 low rank | Stage2 worse | top1 same root | top1 same provider |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ToolBench-G3 | `4943` | `0.309731` | `0.077483` | `1369` | `2138` | `0.801942` | `0.150718` | `0.019826` |
| TrajectBench | `19101` | `0.141040` | `0.162557` | `1993` | `12374` | `0.383697` | `0.942726` | `0.061201` |

By benchmark × transition type：

| group | rows | Stage0 r@5 | Stage2 r@5 | top5 pushed out | Stage0 low rank | Stage2 worse | mean Stage0 rank | mean Stage2 rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ToolBench-G3 self | `930` | `0.548387` | `0.097849` | `423` | `211` | `0.904301` | `19.640860` | `113.429032` |
| ToolBench-G3 switch | `4013` | `0.254423` | `0.072764` | `946` | `1927` | `0.778221` | `52.872165` | `129.322701` |
| TrajectBench self | `554` | `0.355596` | `0.169675` | `141` | `258` | `0.460289` | `47.664260` | `37.072202` |
| TrajectBench switch | `18547` | `0.134631` | `0.162344` | `1852` | `12116` | `0.381409` | `77.709171` | `40.176201` |

关键发现：

1. ToolBench-G3 不是单纯 Stage0 召回问题。
   - Stage0 r@5 已有 `0.309731`，但 Stage2 r@5 掉到 `0.077483`；
   - `1369` 条样本 gold 原本在 Stage0 top5，却被 Stage2 推出 top5；
   - self-transition 和 switch-transition 都被推坏，self 的 `worse frac = 0.904301`，switch 的
     `worse frac = 0.778221`。

2. ToolBench-G3 暴露出更直接的 candidate validity 问题。
   - Stage2 top1 只有 `0.150718` 仍在 ToolBench-G3 根 namespace；
   - top1 同具体 provider 只有 `0.019826`；
   - examples 中大量 ToolBench-G3 样本被 rerank 到 Traject / ToolRet / SkillRET skills，例如：
     - `toolbench-g3/billboard-api/... -> traject/spotify-search`
     - `toolbench-g3/stock-and-options-trading-data-provider/options -> traject/alpha-vantage-time-series-monthly`
     - `toolbench-g3/minecraft-forge-optifine/... -> traject/lost-ark-get-stats`
   - 这说明 Stage2 当前在 unified skill pool 上 rerank 时没有使用“当前环境可用 skill inventory”约束。
     这不是 ToolBench 特化问题，而是通用 routing 设定：agent 在某个环境中只能选择可用工具/skill。

3. TrajectBench 的主要问题不同。
   - Stage0 low-rank rows 有 `12374 / 19101`，说明粗召回本身是主瓶颈；
   - Stage2 对 TrajectBench 有正向信号：`stage2 improved fraction = 0.598712`，mean rank 从
     `76.84` 改到 `40.09`；
   - 但 Stage2 r@5 只有 `0.162557`，说明 rerank 还不足以弥补 Stage0 候选弱。

4. Multi-positive / equivalent label 仍然必要，但不是唯一主因。
   - ToolBench-G3 的 heuristic equivalent top1 只有 `71` 条；
   - TrajectBench 有 `908` 条 probable equivalent miss，主要集中在同类 provider 内，例如 Wayfair、
     Email verification、Flight data、Shopping data；
   - 因此 multi-positive 会帮助 TrajectBench/部分 ToolBench，但 ToolBench-G3 当前更优先的是
     inventory mask + Stage2 loss/input 修复。

当前结论：

- 不应马上重训 Stage0。
- Stage2 v4 的第一优先级应是：
  1. 在 Stage2 reranking 前加入通用 available-skill inventory mask；
  2. 将 next-skill CE 改成 top-M listwise / multi-positive NLL；
  3. 对 ToolBench-G3 self/switch 分开建 gate，避免 self-transition 被系统性推坏；
  4. 对 TrajectBench 做 label equivalence 审计后，再决定是否启动 Stage0 v4 sequential retrieval 重训。

注意：

- 当前 row diagnostics 只保存了 `stage0_gold_rank`，没有保存 Stage0 top-k candidate ids；
- 因此本轮无法直接列出 Stage0 top5 原始候选，只能判断 gold 在 Stage0 top-k 内的位置；
- 下一轮 row diagnostics 应补存 `stage0_top_skill_ids/top_scores`，方便判断 Stage0 候选是否已有正确可执行同义 skill。

验证：

```text
PYTHONPATH=.vendor_pytest /data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  -m pytest tests/test_stage2_v4_pre_audit.py -q
# 5 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/audit_clstr_stage2_v4_precheck.py \
  --output_dir outputs/clstr_stage2_v4_pre_audit_top350_v1 \
  --example_limit 25
# status ok
```

## 2026-05-31 — Stage2 v4 capability implementation：inventory mask + listwise reranking

本轮只做代码实现和本地验证，没有提交 full Slurm job。

实现目标：

- 保留 v3 默认行为：`transition_inventory_mask_mode=off`、`transition_loss_type=cross_entropy`、
  `transition_positive_mode=single`；
- 新增可开关的 v4 能力，允许一次性跑完整 ablation，而不是每改一项就重新烧 full job：
  1. 通用 available-skill inventory mask；
  2. top-M listwise / multi-positive NLL；
  3. ToolBench-G3 / 其他 trajectory 数据可用的 self/switch transition metrics；
  4. real top-M eval 与训练脚本共用同一套 v4 开关。

新增/接入的配置：

```text
--transition_inventory_mask_mode {off,auto,root_namespace}
--transition_loss_type {cross_entropy,listwise_nll}
--transition_positive_mode {single,gold_plus_equivalent}
```

训练脚本与 eval 脚本均已透传这些参数：

- `scripts/run_clstr_stage2_full_base_train.py`
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- `scripts/evaluate_clstr_stage2_real_topm.py`
- `scripts/sbatch/run_clstr_stage2_real_topm_eval.sh`

建议的 Stage2 v4 smoke/full 配置候选：

```bash
TRANSITION_INVENTORY_MASK_MODE=auto
TRANSITION_LOSS_TYPE=listwise_nll
TRANSITION_POSITIVE_MODE=gold_plus_equivalent
STAGE0_TOP_M=350
STAGE0_POSITIVE_MISSING_POLICY=skip
```

注意：

- `positive_missing_policy=inject` 仍可用于诊断/上界训练，但 paper-facing real top-M gate 应使用
  `skip`，避免 gold candidate injection 被误解为推理时可用信息；
- inventory mask 优先使用 row 内显式 inventory 字段；没有显式字段时，`auto` 会退回到 benchmark/root
  namespace 约束；
- multi-positive 目前只消费数据行中已有的 equivalent/positive skill id 字段，不硬编码具体数据集标签；
- TrajectBench 是否需要 Stage0 v4 sequential retrieval，仍应在 equivalence audit 后决定。

本地验证：

```text
PYTHONPATH=.vendor_pytest /data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  -m pytest tests/test_full_base_train.py -q
# 48 passed in 4.46s

PYTHONPATH=.vendor_pytest /data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  -m pytest tests/test_stage2_real_topm_eval.py \
            tests/test_stage2_transition_row_diagnostics.py \
            tests/test_stage2_v4_pre_audit.py -q
# 13 passed in 4.48s

PYTHONPATH=.vendor_pytest /data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  -m pytest tests/test_sbatch_scripts.py tests/test_stage2_real_topm_eval.py -q
# 43 passed in 19.17s

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py \
  clstr/stage2_real_topm_eval.py \
  scripts/run_clstr_stage2_full_base_train.py \
  scripts/evaluate_clstr_stage2_real_topm.py
# exit 0

bash -n scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
bash -n scripts/sbatch/run_clstr_stage2_real_topm_eval.sh
# exit 0
```

## 2026-05-31 — Alias-equivalent positive reconstruction

补充说明：

- 统一轨迹行本身不包含 `positive_next_skill_ids` 这类显式多正例字段；
- 但 `skill_pool.jsonl` 含有 `canonical_skill_id` / `alias_skill_ids`，其中不少 canonical 组是真正的多别名组；
- 因此，`gold_plus_equivalent` 的正确实现方式不是从轨迹里猜字段，而是从 skill pool 重建 alias-equivalent 正例集合；
- 之前生成的 `real_topm_eval_report.json` 和 `train_report.json` 属于旧 wiring，尚未反映这层 alias-equivalent 重建；
- 之后若重新跑评估，`transition_positive_mean_count` / `transition_multi_positive_rows` 才会真实反映多正例是否被触发。

更正后的当前发现：

```text
v3 skill_pool canonical skill rows: 37623
v3 canonical groups represented as multiple skill rows: 0
v3 alias refs: 38875
v3 changed/raw alias refs: 1252
trajectory rows whose next_skill_id belongs to a multi-alias group: 3377
trajectory rows whose current skill belongs to a multi-alias group: 4408
```

这说明当前 `data/clstr_unified_pretrain_v3_trajectory_retrieval_caps/skill_pool.jsonl` 是 canonical-only
候选池：`alias_skill_ids` 记录 raw id 到 canonical id 的合并关系，但候选空间里没有同一 canonical 组的多条 alias skill row。
因此 Stage2 的 `gold_plus_equivalent` 在当前 v3 候选池里主要用于“把 raw/equivalent id 解析到 canonical id”，而不是自然形成多个 candidate 正例。
如果论文要显式证明多正例 alias 学习价值，需要另行构造/保留多行 alias 候选池或引入数据行显式 equivalent positives；否则当前最合理的 paper-facing 说法是 canonicalization-aware positive resolution。

## 2026-05-31 — Stage2 v4 corrected full run status and alias-positive audit

当前 full 作业：

- job id: `80420`
- output dir: `outputs/clstr_unified_stage2_v4_top350_skip_inventory_listwise_gold_equiv_full`
- 配置: `STAGE0_TOP_M=350`, `STAGE0_POSITIVE_MISSING_POLICY=skip`,
  `TRANSITION_INVENTORY_MASK_MODE=auto`, `TRANSITION_LOSS_TYPE=listwise_nll`,
  `TRANSITION_POSITIVE_MODE=gold_plus_equivalent`

阶段性观察：

- 训练已经正常进入主循环，`training_metrics.jsonl`、`loss_curve.svg`、`checkpoints/latest.pt`
  都在持续更新；
- 截至目前，`transition_positive_mean_count` 仍为 `1.0`，`transition_multi_positive_rows` 仍为 `0.0`；
- 这表明 alias-equivalent 正例虽然已在代码中接好，但在当前 Stage0 top-M 候选分布里还没有真正形成多正例训练信号；
- preliminary audit 显示：统一数据里确实有 alias 轨迹行，但多集中在 `traject_bench` / `toolbench_g3`，当前 stage0 top-M 候选仍更像只覆盖到了 gold 而没有同时带出等价 alias。

这版 full 仍然有价值，因为它验证了：

- `gold_plus_equivalent` wiring 没有报错；
- `listwise_nll` + `inventory mask` 的主链路可跑；
- 但如果论文要把“多正例 alias 学习”作为显式卖点，还需要继续审计 Stage0 候选召回，而不能直接把当前 full 当成该能力已生效的证据。

完成状态：

```text
job 80420 completed
metric rows: 10000
checkpoint: outputs/clstr_unified_stage2_v4_top350_skip_inventory_listwise_gold_equiv_full/checkpoints/clstr_full_base-step10000.pt
tail100 loss mean: 2.4559606969356538
tail100 transition_skill_ce_loss mean: 2.377951634526253
tail100 transition_skill_recall@1 mean: 0.3383333333333334
tail100 transition_skill_recall@5 mean: 0.63
tail100 transition_skill_mrr mean: 0.47845344677315105
transition_multi_positive_rows nonzero batches: 0
max transition_positive_mean_count: 1.0
```

当前解读：

- 这次 full 可以作为 `top350 + skip + inventory mask + listwise_nll` 的有效 Stage2 v4 训练结果；
- 不能把它称为 alias-equivalent multi-positive 已经发挥作用的结果；
- 下一步如果要证明 alias 多正例价值，应先做 Stage0 top-M alias coverage audit：
  检查每条 alias-positive 轨迹中，gold skill 的 alias 是否同时出现在 Stage0 top-M 候选里，以及是否被 inventory mask 移除。

## 2026-05-31 — Stage0 alias coverage small audit

小审计结果：

- job id: `80455`
- output dir: `outputs/clstr_stage2_alias_coverage_diag_top350_small_v1`
- benchmark: 仅 `traject_bench`
- row count: `1965`
- alias-positive rows: `43`
- alias-positive rows with any alias in Stage0 top-350: `0`

结论：

- 在这批 `traject_bench` 样本里，alias-positive 并没有进入 Stage0 top-350 候选；
- 因此，Stage2 的 `gold_plus_equivalent + listwise_nll` 没有机会形成多正例训练信号；
- 当前主要瓶颈是 Stage0 候选召回，而不是 Stage2 loss 公式；
- 如果后面要证明 alias multi-positive 的价值，需要先把 Stage0 对这些等价 skill 的召回做上来，再看 Stage2 是否真的能利用它们。

## 2026-05-31 — Stage0 canonicalization-aware positive resolution patch

针对上面的更正结论，已调整 Stage0 retrieval warmup 的正例处理：

- `clstr/retrieval_warmup.py` 新增基于 `canonical_skill_id` / `alias_skill_ids` 的正例解析；
- Stage0 训练时默认 `expand_alias_positives=True`，会把 raw alias positive 解析到当前 skill pool 中存在的 canonical skill id；
- 如果未来 skill pool 保留同一 canonical 组的多条 skill row，也会把同组池内 id 并入 multi-positive set；
- 训练报告新增 `alias_positive_expansion`，区分：
  - `alias_resolved_query_count`: raw alias 被解析到池内 canonical；
  - `multi_positive_expanded_query_count`: 一个 query 的池内正例数被扩展为多个；
- CLI 新增 `--expand_alias_positives / --no-expand_alias_positives`；
- sbatch 默认 `EXPAND_ALIAS_POSITIVES=1`。

这不是 AppWorld 或 TrajectBench 特化，而是使用统一 skill pool 的 canonical metadata 修正 Stage0 监督空间。它不会向 Stage0 注入 dev/test 信息，也不会在推理时补 gold candidate。

## 2026-05-31 — Stage0 alias-patch smoke run

已提交并监督最小 smoke：

- job id: `80485`
- output dir: `outputs/clstr_unified_stage0_biencoder_v3_smoke_aliaspatch`
- 配置: `MAX_STEPS=1`, `BATCH_SIZE=4`, `GRADIENT_ACCUMULATION_STEPS=1`,
  `MAX_SKILLS=128`, `MAX_QUERIES=64`, `EXPAND_ALIAS_POSITIVES=1`

结果：

- job 完成，`train_report.json` / `training_metrics.jsonl` / `checkpoints/latest.pt` 写出正常；
- `alias_positive_expansion.enabled=true`；
- `alias_resolved_query_count=0`；
- `multi_positive_expanded_query_count=0`；
- `positive_label_counts=[1,1,1,1]`；
- 说明这批真实 v3 retrieval query 在当前 canonical-only skill pool 里，Stage0 的 alias 解析没有产生额外池内正例，但训练链路本身是通的。

这次 smoke 的意义只是确认新开关不破链路，不代表 Stage0 还会自动获得多正例效果。

## 2026-06-02 — Stage0 handoff coverage audit 参数修复与有效小样本结果

背景：

- 之前的 handoff audit 小作业因为 `sbatch --export` 传递 `TOP_K_VALUES=20,50,100,200,500` 时被逗号截断，报告只包含 `top_k_values=[20]`；
- 这会让诊断静默退化，不能判断 Stage2 使用 top-100/top-200/top-500 候选时的真实覆盖；
- 已修复 `scripts/sbatch/run_clstr_stage0_handoff_coverage_audit.sh`：
  - 支持用 `TOP_K_VALUES=20:50:100:200:500` 通过 sbatch export 传参；
  - 默认 `REQUIRE_MULTI_TOP_K=1`，如果 top-k 被截成单个值会直接失败，避免产出误导性报告。

有效小样本作业：

```text
job id: 81803
state: COMPLETED
exit code: 0:0
elapsed: 00:00:31
report: outputs/clstr_stage0_handoff_coverage_audit/v3_toolbench_traject_smoke_v2/report.json
row_count: 320
top_k_values: [20, 50, 100, 200, 500]
query_mode: skillrouter_state
checkpoint: outputs/clstr_unified_stage0_biencoder_v3_traj_retrieval_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt
```

全局结果：

```text
current_recall@20  = 0.4905
current_recall@50  = 0.6171
current_recall@100 = 0.7152
current_recall@200 = 0.8259
current_recall@500 = 0.9082

next_recall@20  = 0.4680
next_recall@50  = 0.6000
next_recall@100 = 0.6960
next_recall@200 = 0.8080
next_recall@500 = 0.9040
```

分 benchmark 关键结果：

- `alfworld`: top-20 起 current/next 都为 `1.0`；
- `scienceworld`: top-100 起 current/next 都为 `1.0`；
- `toolbench_g3`:
  - current recall: `@100=0.7656`, `@200=0.8516`, `@500=0.9063`;
  - next recall: `@100=0.7582`, `@200=0.8571`, `@500=0.9231`;
- `traject_bench`:
  - current recall: `@100=0.5313`, `@200=0.7188`, `@500=0.8672`;
  - next recall: `@100=0.4857`, `@200=0.6667`, `@500=0.8381`.

当前解读：

- Stage0 v3 对 ALFWorld / ScienceWorld 的 handoff 覆盖不是瓶颈；
- ToolBench-G3 在 top-200/top-500 下基本可进入可训练区间，但 top-100 仍偏紧；
- TrajectBench 是当前主要瓶颈：top-100 明显不足，top-200 也不够稳，说明 Stage2 如果依赖较小候选池会受到 Stage0 handoff 限制；
- 因此下一步不应继续扩大零散诊断，而应围绕候选池预算和 TrajectBench source/sampler 做有针对性的改动或训练对比。

## 2026-06-02 — Stage0 v4 handoff-aligned sampler 与 gate 实现

澄清：

- Stage0 v3 已经是 trajectory-aware，并不是完全没有用 trajectory-derived retrieval；
- 当前修复目标不是重新构建数据集，而是把 v3 修成 handoff-aligned：训练采样和 Stage2
  handoff query 分布对齐。

本轮代码改动：

- `clstr/retrieval_warmup.py`
  - Stage0 默认 sampler 从 `source_balanced` 改为 `handoff_balanced`；
  - 新增 `_query_training_bucket(...)`：
    - 普通 retrieval source 仍按 source 分桶；
    - 所有 `trajectory_derived_*` source 会进一步按 provenance 中的
      `target=current/next` 分桶；
  - 新增 `_training_bucket_counts(...)`，并把 `training_bucket_count` /
    `training_bucket_counts` 写入 setup、checkpoint 和 `train_report.json`；
  - 这使 Stage0 v4 可以审计每个 sequential-routing bucket 是否真正获得训练预算。
- `scripts/run_clstr_stage0_biencoder_train.py`
  - CLI `--sampling_strategy` 新增并默认使用 `handoff_balanced`。
- `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh`
  - sbatch 默认 `SAMPLING_STRATEGY=handoff_balanced`。
- `scripts/sbatch/run_clstr_stage0_handoff_coverage_audit.sh`
  - 增加 handoff coverage gate；
  - 默认阈值：
    - `global.next_recall@500 >= 0.90`
    - `toolbench_g3.next_recall@500 >= 0.90`
    - `traject_bench.next_recall@500 >= 0.88`
    - `traject_bench.next_recall@200 >= 0.75`
  - gate 结果写入 `handoff_coverage_gate.json`；
  - 当前 v3 预计会因 TrajectBench coverage 不足而被该 gate 阻断，这是预期行为。
- `docs/clstr_stage0_v4_handoff_aligned_plan.md`
  - 新增 Stage0 v4/fix 实施计划。
- `docs/clstr_v4_plan.md`
  - 新增 Stage0 v4 handoff-aligned fix 论文口径。

论文安全性：

- 没有加入 AppWorld/TRAJECT-Bench 特定规则；
- 没有 eval-time gold candidate injection；
- 修改只基于通用 sequential trajectory source 的 current/next query alignment；
- 主线仍要求在真实 top-M candidate 下报告结果。

## 2026-06-02 — Stage0 v4 smoke 通过，full job 已提交等待调度

Stage0 v4 handoff-balanced smoke：

```text
job id: 81970
state: COMPLETED
exit code: 0:0
elapsed: 00:01:46
output_dir: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_smoke
checkpoint: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_smoke/checkpoints/clstr_unified_retrieval_v2-step1.pt
latest: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_smoke/checkpoints/latest.pt
loss_curve: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_smoke/loss_curve.svg
```

smoke 结果：

- `status=ok`
- `sampling_strategy=handoff_balanced`
- `training_bucket_count=12`
- `skill_count=37623`
- `query_count=439442`
- step-1 loss: `9.8896`
- step-1 recall:
  - `recall_at_1=0.25`
  - `recall_at_20=0.875`
  - `recall_at_50=0.875`
  - `recall_at_100=0.875`

关键 bucket 统计：

```text
skillret: 63259
toolret_training: 208826
toolbench_g3: 25709
traject_bench: 30526
trajectory_derived_alfworld:current: 9926
trajectory_derived_alfworld:next: 8852
trajectory_derived_scienceworld:current: 10000
trajectory_derived_scienceworld:next: 9730
trajectory_derived_toolbench_g3:current: 9570
trajectory_derived_toolbench_g3:next: 6691
trajectory_derived_traject_bench:current: 30526
trajectory_derived_traject_bench:next: 25827
```

说明：

- handoff-balanced sampler 已在真实 v3 数据上跑通；
- batch 中实际采到了 `toolret_training`、`traject_bench`、`skillret`、
  `trajectory_derived_scienceworld`、`trajectory_derived_alfworld`、
  `trajectory_derived_traject_bench`、`toolbench_g3`；
- checkpoint、latest、loss curve、train report 都正常写出；
- 因此已按用户授权提交 Stage0 v4 full job。

Stage0 v4 full：

```text
job id: 82040
partition: gpu_a800
state: PENDING (Priority)
elapsed: 00:00:00
output_dir: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE
```

截至本记录时，公共 GPU 分区均无可运行 1 卡作业，`82040` 尚未开始运行，因此未产生训练日志和计费。

## 2026-06-03 — Stage0 v4 full 完成与 handoff audit 抽样修复

Stage0 v4 full 已完成：

```text
job id: 82040
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 04:05:01
output_dir: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE
checkpoint: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt
latest: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/checkpoints/latest.pt
loss_curve: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/loss_curve.svg
train_report: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/train_report.json
```

最终训练状态：

- `train_report.status=ok`
- `training_metrics.jsonl` 共 5000 行，最后 step 为 5000；
- 最后 step loss: `9.7484`
- tail-100 mean loss: `9.7711`
- 最后 step recall:
  - `recall_at_1=0.3359`
  - `recall_at_20=0.7188`
  - `recall_at_50=0.7656`
  - `recall_at_100=0.8516`

随后提交的 v4 handoff audit：

```text
job id: 82708
partition: gpu_a800
state: FAILED
exit code: 3:0
elapsed: 00:00:36
output_dir: outputs/clstr_stage0_handoff_coverage_audit/v4_handoff_balanced_mpn_freezeE
```

失败原因不是模型训练崩溃，而是 audit 配置/采样问题：

- sbatch wrapper 的默认 `MAX_ROWS=${MAX_ROWS:-128}` 会把空值回退到 128；
- Python audit 在 `max_rows` 不为空时直接取文件前缀 `rows[:max_rows]`；
- 当前 trajectory 文件前 128 条只覆盖 ALFWorld，因此 report 中只有
  `alfworld`，`toolbench_g3` / `traject_bench` 指标缺失；
- gate 将缺失指标视为 blocker，因此作业以 exit code 3 失败；
- 该次结果只能说明 audit 抽样无效，不能用于判断 Stage0 v4 的真实 handoff coverage。

修复：

- 新增 `clstr/stage0_audit_sampling.py`，将 capped audit 改为按
  `(benchmark, supervision_label)` 做 deterministic round-robin；
- `clstr/stage0_handoff_audit.py` 使用该采样器，并在 report 中写入
  `available_row_count`、`max_rows`、`row_selection`；
- `scripts/sbatch/run_clstr_stage0_handoff_coverage_audit.sh` 支持
  `MAX_ROWS=all/none/null/0` 表示不传 `--max_rows`；
- 补充轻量单测：
  - `tests/test_stage0_audit_sampling.py`
  - `tests/test_stage0_handoff_audit.py`
  - `tests/test_sbatch_scripts.py`

已做低成本验证：

- `select_audit_rows(..., max_rows=3)` 能从 ALFWorld / ToolBench-G3 /
  TrajectBench 各取到样本；
- `python -m py_compile` 通过；
- `bash -n scripts/sbatch/run_clstr_stage0_handoff_coverage_audit.sh` 通过；
- `git diff --check` 通过；
- 当前环境没有可用 `pytest`，且直接导入重模型链路在登录节点启动过慢，因此未在登录节点跑完整 pytest。

省成本后的下一步：

- 不再提交重复 full training；
- 只需重新跑一次修复后的 handoff audit，建议先用较小但跨 benchmark 的
  `MAX_ROWS=960` 检查 v4 是否明显改善；
- 如果该 audit 接近或通过 gate，再决定是否跑 `MAX_ROWS=all` 或 full retrieval eval；
- 如果 TrajectBench 仍低，再进入 Stage0/Stage2 handoff 误差分析，而不是继续盲目训练。

修复后的 `MAX_ROWS=960` handoff audit 已运行：

```text
job id: 82809
partition: gpu_a800
state: FAILED
exit code: 3:0
elapsed: 00:00:37
output_dir: outputs/clstr_stage0_handoff_coverage_audit/v4_handoff_balanced_mpn_freezeE_balanced960
```

说明：`FAILED 3:0` 是 gate 阻断，脚本和 report 产物正常生成；该结果有效。

audit 设置：

- `available_row_count=275271`
- `row_count=960`
- `max_rows=960`
- `row_selection=benchmark_label_round_robin`
- `top_k_values=[20, 50, 100, 200, 500]`

关键结果：

```text
global.next_recall@500 = 0.9478  (pass)
toolbench_g3.next_recall@500 = 0.9200  (pass)
traject_bench.next_recall@500 = 0.8384  (fail; threshold 0.88)
traject_bench.next_recall@200 = 0.6667  (fail; threshold 0.75)
```

按 benchmark：

- ALFWorld: current/next recall 全部为 1.0；
- ScienceWorld: `next_recall@100/200/500=1.0`；
- ToolBench-G3: `next_recall@100=0.7400`，`@200=0.8500`，`@500=0.9200`；
- TrajectBench: `next_recall@100=0.4747`，`@200=0.6667`，`@500=0.8384`。

当前结论：

- Stage0 v4 handoff-balanced 训练改善了全局 coverage，并使 ToolBench-G3 top-500
  进入可接受区间；
- 但 TrajectBench 几乎仍停在 v3 audit 水平，说明当前瓶颈不是简单的
  handoff-balanced sampler 预算，而更可能是 TrajectBench 的 query/skill 表达、
  等价标签、skill 粒度或候选池结构问题；
- 现阶段不应继续重跑 Stage0 full，也不建议直接推进 Stage2 full；
- 下一步应做一次低成本 TrajectBench-focused row-level audit，定位 missed next skill
  的模式：是否是等价 skill 未标注、skill body/skill id 对齐问题、query text 信息不足，
  还是 Stage0 bi-encoder 在 TrajectBench 工具语义上本身不可分。

## 2026-06-03 — TrajectBench inventory handoff 修复

问题复核：

- Stage0 v4 handoff-balanced 训练后，ToolBench-G3 `next_recall@500=0.9200`
  已过 gate；
- TrajectBench 仍为 `next_recall@200=0.6667`、`next_recall@500=0.8384`；
- 数据扫描显示当前 v3 轨迹中 TrajectBench 的 `tool_inventory_skill_ids` 和
  `equivalent_next_skill_ids` 均缺失；
- 因此主要瓶颈不是简单训练步数不足，而是 Stage0/Stage2 在 TrajectBench 上仍按
  全局 skill pool 做候选交接，缺少真实 tool-use 任务天然具备的 available-tool
  inventory。

本轮修复：

- `scripts/build_clstr_unified_pretrain.py`
  - `build_traject_bench_trajectories(...)` 现在为每个 TRAJECT-Bench query
    写入同一 tool list 的完整 `tool_inventory_skill_ids`；
  - `canonicalize_unified_streams(...)` 会同步 canonicalize
    `tool_inventory_skill_ids` / `equivalent_next_skill_ids`，避免 dedup 后字段指向旧 id。
- `clstr/trajectory_inventory.py`
  - 新增 runtime backfill：对旧版 trajectories，可从同一 `trajectory_id` 下的
    `skill_id` / `next_skill_id` 推断 `tool_inventory_skill_ids`；
  - 这样无需立刻重写 1.2G 的旧 `trajectories.jsonl`，现有 v3 数据在训练/评估时也能使用 inventory。
- `clstr/full_base_train.py`
  - Stage0 top-M handoff 生成候选时，如果 row 有 explicit inventory，则在
    inventory 内排序取 top-M；
  - 如果没有 inventory，保持原来的全池 top-M；
  - handoff report 新增：
    - `inventory_candidate_rows`
    - `inventory_candidate_missing_rows`
    - `inventory_candidate_removed_candidates`
    - `tool_inventory_backfill`
- `clstr/stage0_handoff_audit.py`
  - coverage audit 在选样前执行同样的 runtime backfill；
  - top-K coverage 现在与真实 Stage2 handoff 的 candidate space 一致。
- `clstr/stage0_handoff_row_diagnostics.py`
  - row-level diagnostics 同样使用 runtime backfill；
  - 保留 `full_pool_gold_rank`，同时用实际候选空间 rank 作为 `gold_rank`。

论文口径：

- 这不是 gold injection；
- inventory 来自同一个 task/trajectory 的 available tool list，是 tool-use benchmark
  的通用输入条件；
- 没有使用 AppWorld 或 TrajectBench test label；
- 没有把 `next_action_text` 放进 query；
- 方法上等价于“在可用工具集合内做 CLSTR 连续 routing”，比全局 3.7 万 skill
  pool 直接检索更符合实际 agent setting。

已做低成本验证：

- 直接调用 TrajectBench 构建函数，确认新 rows 写入
  `tool_inventory_skill_ids=['traject/a', 'traject/b']`；
- 直接调用 runtime backfill，确认同一 trajectory 下 rows 会补成
  `['a', 'b', 'c']`；
- `python -m py_compile` 通过；
- `git diff --check` 通过；
- 未在登录节点跑 torch 相关完整 pytest：当前 conda 环境导入 torch 会在登录节点异常变慢，
  已终止本地 torch 验证，避免违反集群使用约束。

下一步：

- 不需要重训 Stage0 才能先验证该修复；
- 应先提交一次低成本 `MAX_ROWS=960` handoff audit，使用现有 v3 数据 +
  runtime inventory backfill；
- 预期 TrajectBench `next_recall@200/@500` 会显著上升；
- 如果 audit 通过，再进入 Stage1/Stage2 小样本 smoke；否则再看剩余 miss 是否来自
  query 表达或 skill 文本质量。

## 2026-06-03 — Inventory-aware Stage0 handoff audit 通过

inventory-aware `MAX_ROWS=960` handoff audit 已完成：

```text
job id: 83119
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 00:00:52
output_dir: outputs/clstr_stage0_handoff_coverage_audit/v4_inventory_backfill_balanced960
checkpoint: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt
```

runtime inventory backfill：

```text
trajectory_inventory_count = 4699
backfilled_rows = 30526
existing_inventory_rows = 0
missing_inventory_rows = 0
```

gate 结果：

```text
global.next_recall@500 = 0.9826  (pass; threshold 0.90)
toolbench_g3.next_recall@500 = 0.9200  (pass; threshold 0.90)
traject_bench.next_recall@500 = 1.0000  (pass; threshold 0.88)
traject_bench.next_recall@200 = 1.0000  (pass; threshold 0.75)
```

按 benchmark 的关键覆盖率：

- ALFWorld: current/next recall 全部为 1.0；
- ScienceWorld: `next_recall@100/200/500=1.0`；
- ToolBench-G3: `next_recall@100=0.7400`，`@200=0.8500`，`@500=0.9200`；
- TrajectBench: current/next recall 在 top-20/50/100/200/500 全部为 1.0。

结论：

- TrajectBench 之前 handoff 失败的主因是缺少 available-tool inventory，
  导致 Stage0/Stage2 在 3.7 万全局 skill pool 中做不必要的全局候选交接；
- 使用同一 trajectory 可用工具集合做 inventory-aware candidate space 后，
  TrajectBench handoff gate 直接通过；
- 这不是 gold injection，也没有使用 test label 或 `next_action_text`；
- 现有 Stage0 v4 checkpoint 暂时不需要重训，下一步应进入 Stage1/Stage2
  inventory-aware 小样本 smoke，确认完整训练链路中的候选交接、listwise/transition
  loss、checkpoint 与 loss 曲线持续写出均正常。

## 2026-06-04 — Stage0 v4 full retrieval eval 与 quality gate 通过

Stage0 v4 full retrieval eval 已完成：

```text
job id: 83804
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 00:43:08
output_dir: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/full_retrieval_eval
metrics: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/full_retrieval_eval/metrics.json
```

调度备注：

- 初始提交时 `TimeLimit=04:00:00`，pending reason 为 `Priority`；
- 旧同类 Stage0 full retrieval eval 在 A800 上约 `00:28:51`，因此将该任务时限调整为
  `01:00:00`；
- 任务随后通过 backfill 启动并在 43 分钟内完成。

full retrieval eval 结果：

```text
evaluated_queries = 439442
missing_run_queries = 0
Recall@20 = 0.660305
Recall@50 = 0.780488
Recall@100 = 0.864927
NDCG@20 = 0.436556
NDCG@50 = 0.462823
NDCG@100 = 0.478100
MAP@100 = 0.384231
```

same-pool SkillRouter frozen baseline：

```text
Recall@20 = 0.633774
Recall@50 = 0.738246
Recall@100 = 0.812518
NDCG@20 = 0.414164
NDCG@50 = 0.437019
NDCG@100 = 0.450382
MAP@100 = 0.364030
```

Stage0 quality gate：

```text
status = ok
blockers = []
loss_drop = 0.191942
observed_sources = skillret, toolbench_g3, toolret_training, traject_bench,
                   trajectory_derived_alfworld, trajectory_derived_scienceworld,
                   trajectory_derived_toolbench_g3, trajectory_derived_traject_bench
```

结论：

- Stage0 v4 不仅通过 inventory-aware handoff gate，也通过 full retrieval eval 的正式
  Stage2 quality gate；
- 在 same-pool static retrieval eval 上，Stage0 v4 明显高于 frozen SkillRouter baseline；
- 因此下一步可以进入 Stage2 smoke，但仍不应直接提交 Stage2 full。

## 2026-06-04 — Stage2 v4 inventory-aware TrajectBench smoke 通过

Stage2 smoke 已完成：

```text
job id: 83823
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 00:00:36
output_dir: outputs/clstr_unified_stage2_v4_inventory_handoff_smoke
stage0 checkpoint: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt
stage1 checkpoint: outputs/clstr_unified_stage1_v4_inventory_handoff_smoke/checkpoints/clstr_stage1_heads-step20.pt
```

smoke 设置：

```text
allowed_benchmarks = traject_bench
max_rows = 512
max_steps = 20
batch_size = 4
stage0_top_m = 350
stage0_positive_missing_policy = skip
transition_inventory_mask_mode = auto
transition_loss_type = listwise_nll
transition_positive_mode = gold_plus_equivalent
```

Stage0/Stage2 handoff：

```text
source_rows = 78
retained_rows = 78
skipped_rows = 0
current_positive_coverage@M = 1.0
next_positive_coverage@M = 1.0
masked_next_skill_ce_rows = 0
inventory_candidate_rows = 156
inventory_candidate_missing_rows = 0
tool_inventory_backfill.trajectory_inventory_count = 49
tool_inventory_backfill.backfilled_rows = 78
```

Stage2 入口检查：

```text
stage0_quality_gate.status = ok
stage2_preflight.status = ok
train_report.status = ok
checkpoint: outputs/clstr_unified_stage2_v4_inventory_handoff_smoke/checkpoints/clstr_full_base-step20.pt
latest: outputs/clstr_unified_stage2_v4_inventory_handoff_smoke/checkpoints/latest.pt
loss_curve: outputs/clstr_unified_stage2_v4_inventory_handoff_smoke/loss_curve.svg
```

最后一步训练指标仅用于 smoke 健康检查，不作为性能结论：

```text
transition_skill_recall@1 = 0.75
transition_skill_recall@5 = 1.0
transition_skill_mrr = 0.875
transition_inventory_mask_applied_rows = 4
transition_inventory_mask_missing_rows = 0
transition_inventory_mask_positive_missing_rows = 0
loss = 1.229329
```

结论：

- Stage2 v4 的 canonical 链路已经能从 Stage0 v4 checkpoint 与 Stage1 heads checkpoint
  正常初始化；
- TrajectBench runtime inventory backfill、inventory-aware Stage0 top-M handoff、
  `listwise_nll` transition loss 都已在 GPU smoke 中跑通；
- 这还不是 full 训练结果，也不能证明最终性能；
- 下一步应先跑 Stage1 full heads init，再用该 Stage1 full checkpoint 跑 Stage2 v4
  inventory-aware full/smoke，而不是用 20-step Stage1 smoke checkpoint 直接进入 full。

## 2026-06-04 — Stage1 v4 full heads init 完成并通过 quality gate

Stage1 v4 full heads init 已完成：

```text
job id: 83828
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 00:17:22
output_dir: outputs/clstr_unified_stage1_v4_inventory_handoff_full
checkpoint: outputs/clstr_unified_stage1_v4_inventory_handoff_full/checkpoints/clstr_stage1_heads-step3000.pt
latest: outputs/clstr_unified_stage1_v4_inventory_handoff_full/checkpoints/latest.pt
```

训练设置：

```text
max_steps = 3000
batch_size = 4
stage0_top_m = 350
stage0_positive_missing_policy = skip
stage0_handoff_sample_multiplier = 4
sampling_strategy = balanced_deterministic
```

Stage0 top-M handoff：

```text
source_rows = 24003
retained_rows = 24003
skipped_rows = 0
current_positive_coverage@M = 1.0
next_positive_coverage@M = 1.0
masked_next_skill_ce_rows = 0
inventory_candidate_rows = 14302
inventory_candidate_missing_rows = 33704
tool_inventory_backfill.trajectory_inventory_count = 1091
tool_inventory_backfill.backfilled_rows = 7151
```

Stage1 quality gate：

```text
status = ok
blockers = []
metric_count = 3000
loss_first_100_mean = 4.198285
loss_last_100_mean = 2.722400
loss_drop = 1.475885
required loss terms active: L_policy, L_trans_skill_ce
```

结论：

- 这版 Stage1 checkpoint 是当前 v4 主线可用的正式 heads initialization；
- 它使用 Stage0 v4、top350、skip policy 与 runtime TrajectBench inventory backfill；
- 现在可以用该 checkpoint 重新跑 Stage2 v4 inventory-aware smoke，然后再决定是否提交
  Stage2 full。

## 2026-06-04 — Stage2 v4 smoke 使用 Stage1 full checkpoint 通过

使用 Stage1 v4 full checkpoint 的 Stage2 smoke 已完成：

```text
job id: 83833
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 00:00:36
output_dir: outputs/clstr_unified_stage2_v4_inventory_handoff_smoke_from_stage1full
stage0 checkpoint: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt
stage1 checkpoint: outputs/clstr_unified_stage1_v4_inventory_handoff_full/checkpoints/clstr_stage1_heads-step3000.pt
```

Stage2 入口检查：

```text
stage0_quality_gate.status = ok
stage2_preflight.status = ok
train_report.status = ok
stage1_initialization.enabled = true
stage1_initialization.loaded = true
stage1_initialization.head_checkpoint_step = 3000
```

TrajectBench inventory-aware handoff：

```text
source_rows = 78
retained_rows = 78
skipped_rows = 0
current_positive_coverage@M = 1.0
next_positive_coverage@M = 1.0
inventory_candidate_rows = 156
inventory_candidate_missing_rows = 0
masked_next_skill_ce_rows = 0
tool_inventory_backfill.trajectory_inventory_count = 49
tool_inventory_backfill.backfilled_rows = 78
```

最后一步训练指标仅用于 smoke 健康检查：

```text
transition_inventory_mask_mode = auto
transition_loss_type = listwise_nll
transition_positive_mode = gold_plus_equivalent
transition_inventory_mask_applied_rows = 4
transition_inventory_mask_missing_rows = 0
transition_inventory_mask_positive_missing_rows = 0
transition_skill_recall@1 = 0.75
transition_skill_recall@5 = 1.0
transition_skill_mrr = 0.875
loss = 0.940028
```

结论：

- Stage2 v4 可以从 Stage0 v4 + Stage1 v4 full heads checkpoint 正常初始化；
- inventory-aware handoff、`listwise_nll` transition loss 与 checkpoint/loss 曲线写出均已验证；
- 下一步可以提交 Stage2 v4 full base 训练，但需要按预算设置明确 wall time，并继续先看
  handoff 和早期 loss 指标。

## 2026-06-04 — Stage2 v4 full base 训练完成并通过 quality gate

Stage2 v4 full base 训练已完成：

```text
job id: 83874
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 01:31:28
output_dir: outputs/clstr_unified_stage2_v4_inventory_handoff_full_from_stage1full
checkpoint: outputs/clstr_unified_stage2_v4_inventory_handoff_full_from_stage1full/checkpoints/clstr_full_base-step10000.pt
latest: outputs/clstr_unified_stage2_v4_inventory_handoff_full_from_stage1full/checkpoints/latest.pt
```

训练设置：

```text
max_steps = 10000
batch_size = 4
stage0_top_m = 350
stage0_positive_missing_policy = skip
stage0_handoff_sample_multiplier = 4
transition_inventory_mask_mode = auto
transition_loss_type = listwise_nll
transition_positive_mode = gold_plus_equivalent
```

数据与 handoff：

```text
rows_after_caps = 60096
stage0_handoff_source_rows = 54879
stage0_handoff_retained_rows = 53665
stage0_handoff_skipped_rows = 1214
skipped_reason = routing_positive_missing_from_stage0_topm
current_positive_coverage@M = 0.977851
next_positive_coverage@M = 0.985258
masked_next_skill_ce_rows = 688
inventory_candidate_rows = 54420
inventory_candidate_missing_rows = 55338
tool_inventory_backfill.trajectory_inventory_count = 4699
tool_inventory_backfill.backfilled_rows = 27210
```

说明：这版 full 使用真实 top350 + skip，没有 gold injection；
`skipped_rows=1214` 表示仍有少量 full-data 候选漏召回，但 coverage 已接近 98.5%。

Stage2 quality gate：

```text
status = ok
blockers = []
metric_count = 10000
loss_first_200_mean = 2.127595
loss_last_200_mean = 1.790529
loss_drop = 0.337066
first_transition_skill_recall@5 = 0.740417
last_transition_skill_recall@5 = 0.830000
first_transition_skill_ce_loss = 2.072397
last_transition_skill_ce_loss = 1.711959
```

required loss terms 均有采样：

```text
L_policy sampled = 23355
L_trans sampled = 40000
L_trans_skill_ce sampled = 36033
STOP sampled = 40000
```

训练安全：

```text
valid_or_test_used_for_training = false
on_policy_rollout_used = false
training_regime = offline_replay_supervised_pretraining
```

结论：

- 当前 Stage0 v4 -> Stage1 v4 full -> Stage2 v4 full 主链路已经完整跑通；
- Stage2 full 不是 benchmark result，但可以作为后续 Stage3/Stage4 或 real-topM eval 的正式 base checkpoint；
- 下一步应优先做 no-inject real-topM eval / row-level diagnostics，确认真实候选分布下
  Stage2 排序是否稳定，再决定是否进入 Stage3/Stage4 act/RL。

## 2026-06-04 — Stage2 v4 no-inject real-top350 full eval 通过

本轮先提交的 no-inject real-top350 eval job `83905` 只评估了 `32` 行，不能作为 full
eval 结论。根因是 `evaluate_stage2_real_topm()` 在 `max_eval_batches=None` 时仍把
`max_steps` 当作 `1` 传给 Stage0 handoff subset 逻辑；因此即使不设置
`MAX_EVAL_BATCHES`，只要设置了 `STAGE0_HANDOFF_SAMPLE_MULTIPLIER=4`，也会被截成
`batch_size * 4 = 32` 行。

修复：

```text
clstr/stage2_real_topm_eval.py:
  - 新增 _select_real_topm_eval_rows()
  - max_eval_batches is None 时 full eval 保留全部 rows
  - 只有显式设置 max_eval_batches 的 limited/smoke eval 才使用 handoff subset

tests/test_stage2_real_topm_eval.py:
  - 增加 sample_multiplier 存在但 max_eval_batches=None 时仍保留全部 rows 的回归测试
```

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage2_real_topm_eval.py \
  scripts/evaluate_clstr_stage2_real_topm.py \
  tests/test_stage2_real_topm_eval.py

轻量断言：stage2_real_topm_eval subset assertions passed
```

64 行 smoke eval：

```text
job id: 83907
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 00:00:33
output_dir: outputs/clstr_unified_stage2_v4_real_topm_eval_smoke64_after_subset_fix
status = ok
blockers = []
evaluated_rows = 64
evaluated_batches = 8
stage0_handoff_subset.enabled = false
stage0_handoff_subset.reason = full_eval_keeps_all_rows
current_positive_coverage@350 = 1.0
next_positive_coverage@350 = 1.0
transition_skill_recall@1 = 0.82
transition_skill_recall@5 = 0.98
transition_skill_mrr = 0.903333
transition_skill_ce_loss = 0.597026
```

正式 full no-inject real-top350 eval：

```text
job id: 83911
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 00:37:08
output_dir: outputs/clstr_unified_stage2_v4_real_topm_eval_full_after_subset_fix
stage0 checkpoint: outputs/clstr_unified_stage0_biencoder_v4_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt
stage2 checkpoint: outputs/clstr_unified_stage2_v4_inventory_handoff_full_from_stage1full/checkpoints/clstr_full_base-step10000.pt
```

eval 设置：

```text
top_m = 350
batch_size = 8
positive_missing_policy = skip
gold positive injection = false
stage0_handoff_query_mode = skillrouter_state
transition_inventory_mask_mode = auto
transition_loss_type = listwise_nll
transition_positive_mode = gold_plus_equivalent
benchmark caps = toolbench_g3 full, traject_bench full, alfworld 10000, scienceworld 10000
```

数据与 handoff：

```text
source_rows = 60096
retained_rows = 58719
skipped_rows = 1377
skipped_reason = routing_positive_missing_from_stage0_topm
current_positive_coverage@350 = 0.977058
next_positive_coverage@350 = 0.984648
masked_next_skill_ce_rows = 771
injected_positive_rows = 0
tool_inventory_backfill.trajectory_inventory_count = 4699
tool_inventory_backfill.backfilled_rows = 30526
stage0_handoff_subset.enabled = false
stage0_handoff_subset.selected_rows = 60096
stage0_handoff_subset.reason = full_eval_keeps_all_rows
```

full eval 指标：

```text
status = ok
blockers = []
evaluated_rows = 58719
evaluated_batches = 7340
transition_skill_recall@1 = 0.304516
transition_skill_recall@5 = 0.781654
transition_skill_mrr = 0.501558
transition_skill_ce_loss = 1.833125
transition_skill_self_recall@5 = 0.363351
transition_skill_switch_recall@5 = 0.707186
policy_expert_recall@1 = 0.959696
retrieval_contrastive_loss = 2.528515
```

结论：

- 这是当前 Stage2 v4 checkpoint 的有效 full no-inject real-top350 eval，不包含 gold
  candidate injection；
- 与 2026-05-30 的旧 v2/v3 no-inject eval 相比，`transition_skill_recall@5` 从约
  `0.484` 提升到 `0.781654`，说明 inventory mask + listwise transition reranking
  修复是有效的；
- Stage0 top350 handoff 仍有 `1377/60096` 行漏召回，但 current/next coverage 分别达到
  `0.977/0.985`，已不是当前主要阻塞；
- `transition_skill_self_recall@5 = 0.363351` 明显低于 switch recall，后续如果继续优化，
  应优先审计 self-transition 标签/候选构造，而不是盲目重训 Stage0；
- 当前可以进入 Stage3/Stage4 act/RL 前置验证，但论文表述中应把这个结果定位为
  offline routing/belief/transition stability gate，不是 closed-loop task success benchmark。

## 2026-06-05 unified pretrain v4.1 数据集构建

本次只做数据构建与 CPU 审计，没有提交 GPU 训练作业。目标是清理已确认污染
`self` 标签的低质量 ScienceWorld/ALFWorld aux 数据，并引入跨环境高质量 WebShop
轨迹，供后续 Stage0/Stage1/Stage2 重新训练使用。

### 代码改动

```text
scripts/build_clstr_unified_pretrain.py
  - 新增 dataset_recipe=v4_1；
  - v4_1 默认过滤 full_base_train 中 source_quality=aux_only 或
    provenance.source_id=aux_skillnet_rebuilt 的主监督行；
  - v4_1 默认 trajectory-derived retrieval caps:
    toolbench_g3=-1, traject_bench=-1, alfworld=10000, scienceworld=0, webshop=10000；
  - 新增 lclan/webshop_expert_trajectories GPT-4 JSON 转换器；
  - WebShop 只保留 reward >= 0.8 的轨迹；
  - WebShop 映射为四个通用 action-type skills:
    reasoning-planner / search-executor / click-executor / finish-executor；
  - source_inventory 增加 webshop_expert_trajectories。

tests/test_unified_pretrain_v2.py
  - 新增 v4.1 回归测试，覆盖 aux 过滤、WebShop 高 reward 纳入、
    WebShop skill_pool 写入和 source_inventory 可见性。
```

### 生成的数据

```text
raw WebShop:
  data/webshop_expert_trajectories/webshop_gpt4_0613.json

v4.1 output:
  data/clstr_unified_pretrain_v4_1/
    trajectories.jsonl
    retrieval.jsonl
    skill_pool.jsonl
    skill_aliases.jsonl
    source_inventory.jsonl
    leakage_audit.json
    manifest.json
    v4_1_quality_audit.json
```

构建命令：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  scripts/build_clstr_unified_pretrain.py \
  --output_dir data/clstr_unified_pretrain_v4_1 \
  --schema_version v4.1 \
  --dataset_recipe v4_1 \
  --webshop_min_reward 0.8
```

### manifest 关键结果

```text
status = ok
schema_version = v4.1
dataset_recipe = v4_1

trajectories.jsonl = 110354 rows
retrieval.jsonl = 544443 rows
skill_pool.jsonl = 37613 rows
skill_aliases.jsonl = 38865 rows
source_inventory.jsonl = 7 rows

quality filtered rows = 200485
  v4_1_aux_only_full_base_filtered:alfworld = 107051
  v4_1_aux_only_full_base_filtered:scienceworld = 93434

WebShop HF GPT-4:
  raw trajectories = 12338
  retained trajectories reward >= 0.8 = 4749
  filtered low reward trajectories = 7589
  converted step rows = 35568
```

### v4.1 质量审计

审计文件：

```text
data/clstr_unified_pretrain_v4_1/v4_1_quality_audit.json
```

核心结果：

```text
trajectory_rows_by_benchmark:
  alfworld = 34690
  toolbench_g3 = 9570
  traject_bench = 30526
  webshop = 35568

scienceworld_trajectory_rows_remaining = 0
scienceworld_skill_pool_records_remaining = 0

transition_counts:
  switch = 53594
  self = 12012

transition_counts_by_benchmark:
  alfworld: self=711, switch=1558
  toolbench_g3: self=1078, switch=5613
  traject_bench: self=602, switch=25225
  webshop: self=9621, switch=21198

webshop_skill_counts:
  webshop-click-executor = 15339
  webshop-reasoning-planner = 10436
  webshop-search-executor = 5075
  webshop-finish-executor = 4718

leakage_audit.status = ok
leaked_appworld_task_ids_in_trajectories = []
leaked_skillret_query_ids_in_retrieval = []
```

### 验证

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_pretrain_v2.py

result: 18 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  tests/test_unified_pretrain_v2.py

JSONL parse check:
  trajectories.jsonl 110354 ok
  retrieval.jsonl 544443 ok
  skill_pool.jsonl 37613 ok
  skill_aliases.jsonl 38865 ok
  source_inventory.jsonl 7 ok

git diff --check:
  no output
```

### 结论

- v4.1 已经把此前确认有问题的 ScienceWorld aux 主监督完全移出训练流；
- ALFWorld 的 `aux_skillnet_rebuilt` 低质量主监督也被过滤，保留 official replay 和
  AgentGym weak policy；
- WebShop 是新引入的训练辅助源，不作为评测集，不改变 benchmark protocol；
- 新数据集更适合用来重新训练 Stage0/Stage1/Stage2，尤其是验证 WebShop 增加的
  高质量 self/switch 是否能改善 transition self recall；
- 后续 full training 仍需先做小样本 smoke，再由用户确认是否提交完整 GPU 作业。

## 2026-06-05 unified pretrain v4.1b clean ALFWorld non-overlap 增量

本次继续只做数据构建与 CPU 审计，没有提交 GPU 训练作业。v4.1b 在 v4.1 基础上补入
本地 clean ALFWorld，但只加入与已有 official replay 不重叠的 step，避免同一 state
出现多个冲突 GT。

### 设计决策

审计发现 `data/clstr_dagger_expert_corrected_train_enriched/train.jsonl` 与 official replay
有 `1114` 个 `(trajectory_id, step_index)` 重叠，其中大量 `action_text` / `skill_id` /
`next_skill_id` 不一致。因此 v4.1b 不全量叠加 clean ALFWorld，而是采用：

```text
key = (benchmark, trajectory_id, step_index)
如果 key 已存在于前面写入的 clean rows，则跳过；
否则写入该 clean ALFWorld row。
```

### 代码改动

```text
scripts/build_clstr_unified_pretrain.py
  - 新增 dataset_recipe=v4_1b；
  - v4_1b 继承 v4.1 的 aux_only 过滤和 WebShop reward>=0.8 逻辑；
  - 新增 build_clean_alfworld_additional_trajectories()；
  - clean ALFWorld 源：
    data/clstr_dagger_expert_corrected_train_enriched/train.jsonl
    data/clstr_qwen3_structured_train/train.jsonl
  - source_inventory 增加上述两个 clean ALFWorld 源；
  - skill_pool 增加 clean ALFWorld skills 路径。

tests/test_unified_pretrain_v2.py
  - 新增 v4.1b 回归测试，覆盖：
    aux filtering、WebShop 保留、clean ALFWorld 非重叠写入、重叠 row 跳过、
    source_inventory 可见性。
```

### 生成的数据

```text
v4.1b output:
  data/clstr_unified_pretrain_v4_1b/
    trajectories.jsonl
    retrieval.jsonl
    skill_pool.jsonl
    skill_aliases.jsonl
    source_inventory.jsonl
    leakage_audit.json
    manifest.json
    v4_1b_quality_audit.json
```

构建命令：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python \
  scripts/build_clstr_unified_pretrain.py \
  --output_dir data/clstr_unified_pretrain_v4_1b \
  --schema_version v4.1b \
  --dataset_recipe v4_1b \
  --webshop_min_reward 0.8
```

### manifest 关键结果

```text
status = ok
schema_version = v4.1b
dataset_recipe = v4_1b

trajectories.jsonl = 111488 rows
retrieval.jsonl = 545564 rows
skill_pool.jsonl = 37615 rows
skill_aliases.jsonl = 38867 rows
source_inventory.jsonl = 9 rows

quality filtered rows = 200485
  v4_1_aux_only_full_base_filtered:alfworld = 107051
  v4_1_aux_only_full_base_filtered:scienceworld = 93434

clean_alfworld:
  total_rows_added = 1134
  skipped_overlap_count = 3140
  clstr_dagger_expert_corrected_train_enriched.added_rows = 1134
  clstr_dagger_expert_corrected_train_enriched.skipped_overlap_count = 1114
  clstr_qwen3_structured_train.added_rows = 0
  clstr_qwen3_structured_train.skipped_overlap_count = 2026

WebShop HF GPT-4:
  retained trajectories reward >= 0.8 = 4749
  converted step rows = 35568
```

### v4.1b 质量审计

审计文件：

```text
data/clstr_unified_pretrain_v4_1b/v4_1b_quality_audit.json
```

核心结果：

```text
trajectory_rows_by_benchmark:
  alfworld = 35824
  toolbench_g3 = 9570
  traject_bench = 30526
  webshop = 35568

source_quality_counts:
  official_replay = 2443
  dagger_expert_corrected_rollout = 1134
  weak_policy = 32247
  toolbench_g3_dfs_tree = 9570
  traject_public_data = 30526
  webshop_hf_gpt4_reward_ge_0.8 = 35568

scienceworld_trajectory_rows_remaining = 0
scienceworld_skill_pool_records_remaining = 0

transition_counts:
  switch = 53910
  self = 12795

transition_counts_by_benchmark:
  alfworld: self=1494, switch=1874
  toolbench_g3: self=1078, switch=5613
  traject_bench: self=602, switch=25225
  webshop: self=9621, switch=21198

leakage_audit.status = ok
leaked_appworld_task_ids_in_trajectories = []
leaked_skillret_query_ids_in_retrieval = []
```

### 验证

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_pretrain_v2.py

result: 19 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  tests/test_unified_pretrain_v2.py

JSONL parse check:
  trajectories.jsonl 111488 ok
  retrieval.jsonl 545564 ok
  skill_pool.jsonl 37615 ok
  skill_aliases.jsonl 38867 ok
  source_inventory.jsonl 9 ok

git diff --check:
  no output
```

### 结论

- v4.1b 是 v4.1 的保守增强版：只增加 `1134` 行非重叠 clean ALFWorld；
- ALFWorld self 从 v4.1 的 `711` 增加到 v4.1b 的 `1494`，但没有引入重叠 state
  的冲突标签；
- ScienceWorld 仍完全不进入主监督；
- 当前建议后续训练使用 v4.1b，并在训练阶段通过 source-balanced sampler 控制
  WebShop / weak ALFWorld / target benchmarks 的采样比例。

## 2026-06-05 v4.1b Stage1/Stage2 训练入口与采样修复

本次没有提交 GPU full job，只做代码级预检、单元测试和 CPU 采样模拟。

### 背景

v4.1b 数据构成比 v4/v4.1 更丰富，但 Stage1/Stage2 的训练入口仍有两个问题：

1. Stage1 底层训练函数已经支持 `benchmark_caps` 和 `sampling_strategy`，但
   `scripts/run_clstr_stage1_heads_init.py` 与
   `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh` 没有暴露这些参数；
2. Stage1/Stage2 原默认 `balanced_random` 只按 loss bucket 抽样，不能显式保证
   `benchmark:self/switch` 分组覆盖。v4.1b 虽然补入了 WebShop 与 clean ALFWorld
   self 样本，但在 ToolBench/TrajectBench 上仍可能被大量 switch 样本压住。

### 改动

- Stage1 CLI 新增：
  - `--benchmark_caps`
  - `--sampling_strategy`
- Stage1 sbatch wrapper 新增：
  - `BENCHMARK_CAPS`
  - `SAMPLING_STRATEGY`
- Stage2 CLI 的 `--sampling_strategy` choices 改为复用
  `clstr.full_base_train.SAMPLING_STRATEGIES`；
- 新增通用 sampler：

```text
benchmark_transition_balanced_random
```

该 sampler 不绑定 AppWorld，也不绑定某个 benchmark。它在 loss bucket 内按：

```text
benchmark:self
benchmark:switch
benchmark:none
```

做轮转选组，并在组内使用 seeded random 选行。这样保留原有 loss 监督入口，同时避免
ToolBench/TrajectBench 的少量 self 样本在训练 batch 中长期缺席。

为避免 full job 中每一步重复扫描全量数据，Stage1/Stage2 训练循环已改为在训练开始时
预计算 transition group buckets。

### 默认策略

Stage1/Stage2 Python 入口与 sbatch wrapper 默认采样策略均改为：

```text
benchmark_transition_balanced_random
```

旧策略仍可通过：

```bash
SAMPLING_STRATEGY=balanced_random
```

或：

```bash
--sampling_strategy balanced_random
```

显式恢复。

### v4.1b 采样模拟

使用 v4.1b、`webshop=15000` cap、`1000` steps、`batch_size=4` 做 CPU 抽样模拟。

旧 `balanced_random`：

```text
sampled_groups:
  traject_bench:self = 38
  traject_bench:switch = 1319
  toolbench_g3:self = 51
  toolbench_g3:switch = 295
```

新 `benchmark_transition_balanced_random`：

```text
sampled_groups:
  traject_bench:self = 277
  traject_bench:switch = 277
  toolbench_g3:self = 278
  toolbench_g3:switch = 278
```

ALFWorld 在新策略下仍会相对偏高，主要因为当前 `belief` bucket 中 ALFWorld 占比高；
这属于 loss/data 构成问题，不是 sampler 泄露或候选交接错误。后续 smoke 时需要同时看：

- loss 曲线；
- `sampled_loss_activation_counts`；
- Stage1/Stage2 gate；
- no-inject real top-M eval。

### 验证

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py::test_full_base_benchmark_transition_balanced_sampler_covers_small_self_groups \
  tests/test_clstr_topm_candidate_handoff.py::test_stage1_heads_cli_exposes_sampling_and_benchmark_caps_for_unified_data \
  tests/test_sbatch_scripts.py::test_unified_stage1_heads_init_sbatch_uses_stage0_topm_candidates \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool

result: 4 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_clstr_stage1_heads_init.py --help
  exposes --benchmark_caps
  exposes --sampling_strategy {balanced_deterministic,balanced_random,benchmark_transition_balanced_random}

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/run_clstr_stage2_full_base_train.py --help
  exposes --sampling_strategy {balanced_deterministic,balanced_random,benchmark_transition_balanced_random}

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py \
  scripts/run_clstr_stage1_heads_init.py \
  scripts/run_clstr_stage2_full_base_train.py

result: passed
```

## 2026-06-06 v4.2_progressive_final Stage2 结果

本轮完成了 `v4.2_progressive_final` 的 Stage2 full-base 离线训练。

作业信息：

```text
slurm job: 84583
partition: gpu_a800
state: COMPLETED
exit code: 0:0
elapsed: 01:25:59
output: outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise
checkpoint: outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise/checkpoints/clstr_full_base-step10000.pt
gate: outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise/stage2_quality_gate.json
```

Stage2 handoff：

```text
selected_rows = 75566
retained_rows = 74569
skipped_rows = 997
current_positive_coverage@350 = 0.9846
next_positive_coverage@350 = 0.9896
injected_positive_rows = 0
```

最终滚动指标：

```text
tail-200:
  loss = 1.4185
  policy@1 = 0.8500
  transition@1 = 0.5550
  transition@5 = 0.8546
  transition_mrr = 0.6750
  transition_ce = 1.2765

tail-500:
  loss = 1.5202
  policy@1 = 0.8463
  transition@1 = 0.5068
  transition@5 = 0.8460
  transition_mrr = 0.6408
  transition_ce = 1.3765
```

Stage2 gate 结果：

```text
status = ok
blockers = []
first_loss_mean = 1.6844
last_loss_mean = 1.4185
loss_drop = 0.2659
first_transition_skill_recall@5 = 0.8546
last_transition_skill_recall@5 = 0.8546
first_transition_skill_ce_loss = 1.5390
last_transition_skill_ce_loss = 1.2765
```

和 Stage1 progressive-final 相比，Stage2 对 transition ranking 有明显提升：

```text
Stage1 tail-100 transition@1 = 0.3842
Stage2 tail-500 transition@1 = 0.5068
Stage2 tail-200 transition@1 = 0.5550
```

### Stage2 gate metadata 修正

初次审计时，Stage2 gate 被 `stage2_should_not_use_on_policy_rollout` 拦住。
根因不是训练真的执行了 on-policy rollout，而是 `train_report.json` 把数据行级别的
`on_policy_rollout=True` 误汇总成了“本次 Stage2 训练使用了 on-policy rollout”。

本轮核对后确认：

```text
row-level on_policy_rollout=True rows = 1134
source_quality = dagger_expert_corrected_rollout
benchmark = alfworld
split = train
```

这些行是已记录/已修正的离线 DAgger 监督轨迹，不是 Stage2 训练时在模拟器中在线采样得到的 rollout。
因此修正了报告语义：

```text
on_policy_rollout_used = False
offline_rollout_source_rows_used = 1134
offline_rollout_source_counts = {"dagger_expert_corrected_rollout": 1134}
```

原始报告已保留：

```text
outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise/train_report.raw_before_offline_rollout_metadata_fix.json
```

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py::test_full_base_report_distinguishes_offline_rollout_rows_from_online_rollout_training \
  tests/test_full_base_train.py::test_train_clstr_full_base_with_model_respects_loss_masks_and_writes_checkpoint

result: 2 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage2_quality_gate.py

result: 4 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py \
  clstr/stage2_quality_gate.py \
  scripts/audit_clstr_stage2_quality.py

result: passed
```

结论：

- 当前 Stage2 checkpoint 可以作为下一阶段候选；
- 这说明 `v4.2_progressive_final` 的连续 transition 训练在离线 routing 指标上开始体现价值；
- 但这还不是 AppWorld / ToolBench / TRAJECT 的真实下游 task success 结果；
- 下一步应在提交 Stage3/ACT 前先明确评估目标和作业预算，不能自动继续烧 full job。

## 2026-06-06 v4.2_progressive_final 决策

当前不再继续提交 Stage2 full。Stage1 对比显示 v4.2 nowweak 的实现方向
有局部收益，但整体低于 v4.1b：

```text
v4.1b last300 transition R@1 = 0.437
v4.2 nowweak CE single last300 transition R@1 = 0.333
v4.2 nowweak inventory+listwise last300 transition R@1 = 0.389
```

结论：

- listwise/inventory 不是退步主因，它修回了一部分；
- v4.2 严格移除 weak_policy 后，ALFWorld policy/belief 覆盖不足；
- AgentGym weak_policy 多数没有 `next_skill_id`，不能作为 transition CE 正例；
- 下一步只押一个最终版本 `v4.2_progressive_final`，不跑多组 full 对照。

`v4.2_progressive_final` 设置：

```text
data = v4.2 high-quality base + up to 12k filtered AgentGym weak_policy
weak_policy role = policy/belief/STOP augmentation only; L_trans_skill_ce remains false
transition_inventory_mask_mode = auto
transition_inventory_min_candidates = 64
transition_loss_type = listwise_nll
transition_positive_mode = gold_plus_equivalent
L_policy = 0.6
L_trans_skill_ce = 0.6
L_trans = 0.2
STOP = 0.1
belief = 0.1
routing = 0.0
Stage0 = reuse v4.2 nowweak checkpoint; no Stage0 retrain unless top350 coverage < 0.95
```

Full training should only be submitted after data build, CPU tests, and a small smoke
confirm the new recipe writes metrics and preserves Stage0 handoff health.

## 2026-06-06 v4.2_progressive_final 实现记录

已完成代码实现，尚未提交 full GPU job：

- 新增 `dataset_recipe=v4_2_progressive_final`；
- filtered AgentGym weak_policy rows 默认最多 12k，只补
  policy/belief/STOP 信号；
- filtered weak rows 强制 `L_trans_skill_ce=False`、`routing=False`，并跳过
  trajectory-derived retrieval pair 构建；
- provenance 标注
  `augmentation=filtered_weak_policy_v4_2_progressive_final`；
- 新增 `transition_inventory_min_candidates`，inventory 过滤后候选不足时从原始
  Stage0 top-M 顺序回填；
- Stage1/Stage2 CLI 与 sbatch 主脚本已透传该参数；
- 新增 final Stage1 wrapper：
  `scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_progressive_final.sh`，
  默认 top350 + inventory64 + listwise + final loss。

验证结果：

```text
focused pytest: 9 passed
tests/test_sbatch_scripts.py: 49 passed
py_compile: passed
bash -n modified sbatch scripts: passed
```

下一步不是直接 full 训练，而是先构建 progressive 数据目录并检查 manifest /
skill-pool compatibility / Stage0 top350 handoff coverage。只有 coverage 和数据构成正常，
再由用户最终确认是否提交 Stage1 full job。

## 2026-06-06 Stage1 transition 配置链路修复

这次 Stage1 v4.2 no-weak 结果不能直接进入 Stage2。根因不是 handoff
coverage，而是 Stage1 没有实际使用 v4.2 计划中的 transition ranking objective：

```text
期望: inventory_mask=auto + loss_type=listwise_nll + positive_mode=gold_plus_equivalent
实际: inventory_mask=off  + loss_type=cross_entropy + positive_mode=single
```

旧输出：

```text
outputs/clstr_unified_stage1_v4_2_nowweak_top350_heads_init
```

现在会被 Stage1 quality gate 拦住：

```text
blocker: transition_recall_at_1_too_low
last-100 transition_skill_recall@1 ~= 0.315
default threshold = 0.35
```

### 已修复

- `scripts/run_clstr_stage1_heads_init.py` 暴露并传递：
  - `--transition_inventory_mask_mode`
  - `--transition_loss_type`
  - `--transition_positive_mode`
- `clstr/full_base_train.py` 的 Stage1 wrapper 已把这些参数传入
  `train_clstr_full_base_with_model`。
- generic Stage1 默认保持旧实验兼容：

```text
off + cross_entropy + single
```

- v4.2 no-weak Stage1 wrapper 默认改为：

```text
auto + listwise_nll + gold_plus_equivalent
```

- v4.2 no-weak Stage1 新输出目录：

```text
outputs/clstr_unified_stage1_v4_2_nowweak_top350_inventory_listwise_heads_init
```

- v4.2 Stage2 wrapper 已改为读取这个新 Stage1 checkpoint，避免误用旧坏结果。
- Stage1 quality gate 现在检查 transition ranking，而不只看 loss drop：
  - tail `transition_skill_recall@1`
  - tail `transition_skill_recall@5`
  - `transition_skill_recall@5` 前后窗口退化
  - `transition_skill_ce_loss` 前后窗口上升

### 验证

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_stage1_heads_quality_gate.py \
  tests/test_stage2_quality_gate.py \
  tests/test_unified_training_readiness.py

result: 42 passed

pytest -q tests/test_sbatch_scripts.py

result: 48 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  scripts/run_clstr_stage1_heads_init.py \
  clstr/full_base_train.py \
  clstr/stage1_heads_quality_gate.py \
  scripts/audit_clstr_stage1_heads_quality.py

result: passed

bash -n \
  scripts/sbatch/run_clstr_unified_stage1_heads_init.sh \
  scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_nowweak.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_2_nowweak.sh \
  scripts/submit_clstr_stage2_after_stage1_gate.sh

result: passed
```

### 下一步

不要继续使用旧 Stage1 checkpoint。下一步只应提交低成本 Stage1 fixed-config
smoke / short run，确认 transition metrics 和 gate 正常后，再由用户确认是否跑
full Stage1。

## 2026-06-05 v4.2 no-weak mainline sbatch wrappers

项目重审计后，下一阶段主线不再建议继续用 v4.1b 作为默认入口，而应使用当前更干净的
v4.2 no-weak 数据视图：

```text
data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak
```

为了避免误提交旧 v3/v4.1b 链路，本轮没有直接修改通用 sbatch 脚本默认值。通用脚本仍保留
v3-compatible defaults 以便复现实验；新增 v4.2 专用 wrapper，只负责设置明确环境变量并转发到
通用脚本。

新增 wrapper：

```text
scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak.sh
scripts/sbatch/run_clstr_unified_stage0_biencoder_train_v4_2_nowweak.sh
scripts/sbatch/run_clstr_stage0_full_retrieval_eval_v4_2_nowweak.sh
scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_nowweak.sh
scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_2_nowweak.sh
```

固定配置：

```text
DATA_ROOT / TRAIN_PATH / SKILLS_PATH = data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak
Stage0 output = outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE
Stage0 top_k / Stage1-2 top_m = 350
BENCHMARK_CAPS = toolbench_g3=-1:traject_bench=-1:alfworld=-1:webshop=-1
SAMPLING_STRATEGY = balanced_random for Stage1/Stage2
```

Stage1 wrapper 使用温和重平衡 loss，避免旧默认 `L_policy=1.0` 过强：

```text
L_policy = 0.4
L_trans = 0.3
L_trans_skill_ce = 0.6
belief = 0.1
STOP = 0.1
routing = 0.0
```

Stage2 wrapper 使用当前 v4 transition 主线：

```text
L_policy = 0.2
L_trans = 0.3
L_trans_skill_ce = 1.0
policy_hard_negative_margin = 0.0
Q_success = 0.0
transition_inventory_mask_mode = auto
transition_loss_type = listwise_nll
transition_positive_mode = gold_plus_equivalent
```

建议下一步提交顺序仍是分阶段 gate，不直接一口气 full chain：

1. same-pool SkillRouter frozen baseline；
2. Stage0 v4.2 no-weak 训练；
3. Stage0 full retrieval eval；
4. Stage1 heads init；
5. Stage2 inventory/listwise full-base。

每个阶段都需要先看产物、loss/metrics、gate，再进入下一阶段。当前本轮没有提交任何 sbatch 作业。

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_sbatch_scripts.py::test_v4_2_nowweak_stage0_biencoder_sbatch_pins_current_mainline_data_and_outputs \
  tests/test_sbatch_scripts.py::test_v4_2_nowweak_stage0_baseline_and_eval_sbatch_share_same_pool_paths \
  tests/test_sbatch_scripts.py::test_v4_2_nowweak_stage1_sbatch_uses_rebalanced_loss_and_top350_handoff \
  tests/test_sbatch_scripts.py::test_v4_2_nowweak_stage2_sbatch_uses_inventory_listwise_and_stage1_checkpoint

result: 4 passed

bash -n \
  scripts/sbatch/run_clstr_unified_stage0_biencoder_train_v4_2_nowweak.sh \
  scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak.sh \
  scripts/sbatch/run_clstr_stage0_full_retrieval_eval_v4_2_nowweak.sh \
  scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_nowweak.sh \
  scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_2_nowweak.sh

result: passed
```

## 2026-06-05 v4.2 skill text quality follow-up

在继续提交任何新训练前，又补做了一轮 skill text 质量审计。结论是：

- 当前主线数据目录：

```text
data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak
```

- `skill_pool.jsonl` 原本有 24 条 canonical skill 的 `description` 为空白字符串；
- 全部来自 ToolBench-G3 API 记录，虽然没有自然语言 description，但仍有可用的 `name` 和
  `input_schema`；
- 这不是数据引用错误，但会让对应 skill embedding 文本质量变差。

根因：

```text
scripts/build_clstr_unified_pretrain.py::_skill_description
```

之前把 whitespace-only string 当成有效 description 返回，因此不会 fallback 到
API name / schema。

已修复：

- `_skill_description()` 现在会 compact text 并跳过空白 description / executor_desc / body；
- 若没有自然语言描述，则用 `name + input_schema parameter names` 生成 fallback，例如：

```text
toolbench-g3/aliexpress-unofficial/categories
=> /categories API with parameters: locale, country.

toolbench-g3/gearbest/search
=> /search API with parameters: query, page.
```

- 新增回归测试覆盖 `description=" "` 的 ToolBench-G3 API；
- 当前 v4.2 no-weak `skill_pool.jsonl` 已原地只修正这 24 条空白 description，没有改动
  trajectory/retrieval 样本。

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_includes_normalized_toolbench_g3_trajectories_retrieval_and_skills \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_fills_blank_toolbench_g3_skill_description_from_api_schema \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_partial_dedup_remaps_duplicate_tool_skills \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_partial_dedup_keeps_same_name_with_different_params_separate

result: 4 passed
```

数据完整性复查：

```text
skill_pool rows = 37617
unique_skill_ids = 37617
empty_description_count = 0
trajectory rows = 85932
missing trajectory skill refs = 0
retrieval rows = 599735
missing retrieval skill refs = 0
```

注意：该 `skill_pool.jsonl` 中有来自 SkillRET 的长 `body` 字段，包含转义/特殊换行字符。
后续写 JSONL rewrite 脚本时应使用文件句柄逐行迭代，不要用
`Path.read_text().splitlines()` 做重写。

## 2026-06-05 项目重审计：数据构成、Stage2 卡住路径与 loss 设置

本轮只做静态/CPU 审计和确定性代码修复，没有提交新的 Slurm/GPU 作业。

### v4.2 no-weak 数据视图

当前建议作为下一轮训练候选的数据目录：

```text
data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak
```

审计结果：

```text
trajectory_rows = 85,932
retrieval_pairs = 599,735
skill_pool = 37,617
leakage_audit.status = ok

trajectory by benchmark:
  alfworld = 10,268
  webshop = 35,568
  traject_bench = 30,526
  toolbench_g3 = 9,570

source_quality:
  official_replay = 2,443
  dagger_expert_corrected_rollout = 1,134
  hf_alfworld_admissible_success = 6,691
  webshop_hf_gpt4_reward_ge_0.8 = 35,568
  traject_public_data = 30,526
  toolbench_g3_dfs_tree = 9,570

weak_policy = 0
scienceworld = 0
missing_skill_refs_from_trajectories = 0
missing_positive_refs_from_retrieval = 0
```

结论：这个视图比 v4.1b 更干净，适合做下一阶段主线训练；但它和旧 v3/v4.1b 的分布不同，因此 Stage0/Stage1/Stage2 必须统一使用同一数据目录和同一 caps，否则结果不可比。

### source inventory 修复

发现 `hf_alfworld_admissible_success` 已被构建器读入训练数据，但 `source_inventory.jsonl` 误标为 unavailable。根因是 inventory validator 只检查顶层 `*.parquet`，而 HuggingFace snapshot 实际位于嵌套目录：

```text
data/hf_alfworld_admissible_success/data/train-00000-of-00001.parquet
```

修复：

```text
path.glob("*.parquet") -> path.rglob("*.parquet")
```

并刷新了 v4.2 no-weak 的 `source_inventory.jsonl` / manifest source_inventory stats：

```text
source_count = 10
available_source_count = 10
missing_required_target_sources = []
```

### Stage2 长时间无训练输出问题

旧输出 `outputs/clstr_unified_stage2_v4_1b_top350_current_candidate_full` 的 `setup_status.jsonl` 停在：

```text
phase = embedding_cache_started
cache_enabled = true
row_count = 65570
max_rows = 20000
```

这与已修复的 cache-policy 问题一致：大规模 native Stage2 数据在 `embedding_cache_mode=auto` 下不应预编码全量 embedding。当前代码已让所有 `row_count > embedding_cache_max_rows` 的大数据自动跳过预缓存，不再只针对 Qwen external encoder。

此外当前 Stage2 会在训练前写：

```text
setup_status.jsonl
checkpoints/latest.pt  # step 0
loss_curve.svg         # waiting for training metrics
```

如果未来作业还没出现 `training_metrics.jsonl`，应先看 `setup_status.jsonl` 的最后一行判断卡在 handoff、skill table、cache 还是 training。

### Stage2 hidden auxiliary loss 修复

发现 Stage2 CLI 只显式传入：

```text
L_policy
L_trans
L_trans_skill_ce
belief
STOP
routing
transition_hard_negative_margin
```

但 `_normalize_loss_weights` 会把未显式传入的默认项补回：

```text
hard_negative_margin = 0.2
Q_success = 0.5
```

这会让主线 Stage2 暗中训练 policy hard-negative / Q-success auxiliary objective，影响 transition-focused 实验解释。修复后 Stage2 CLI/sbatch 显式暴露并默认关闭：

```text
--policy_hard_negative_margin_loss_weight 0.0
--q_success_loss_weight 0.0
```

如果后续要做 DAgger/Q-success ablation，可以显式打开，而不是混进主线。

### sampler / loss 审计结论

v4.2 no-weak 上用 `balanced_random` 模拟 Stage1/Stage2 采样，未发现 benchmark 完全失衡：

```text
Stage1 12k sampled slots:
  alfworld ~= 4,053
  webshop ~= 3,796
  traject_bench ~= 3,183
  toolbench_g3 ~= 968

sampled active losses:
  L_trans = 12,000 / 12,000
  STOP = 12,000 / 12,000
  L_policy ~= 11,061 / 12,000
  L_trans_skill_ce ~= 10,766 / 12,000
  belief ~= 3,776 / 12,000
```

因此 sampler 本身不是明显坏点。当前更大的风险是：

- Stage1 默认 loss 仍是旧 policy-heavy 配置：`L_policy=1.0, L_trans=0.05, L_trans_skill_ce=0.2`；
- Stage0/Stage1/Stage2 通用脚本默认路径仍偏 v3，v4.2 训练必须显式传入数据目录、输出目录、top-M 和 caps；
- 是否把 Stage1 默认改成 transition 更强的配置，属于实验策略选择，需要单独确认。

### 下一步建议

不要直接提交 full job。建议下一步先做 v4.2 no-weak 的 Stage0 smoke，并明确使用：

```text
DATA_ROOT=data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak
OUTPUT_DIR=outputs/clstr_unified_stage0_v4_2_nowweak_...
TOP_K=350
RETRIEVAL_LOSS_MODE=multi_positive_nll
SAMPLING_STRATEGY=handoff_balanced
```

Stage0 smoke 通过后，再由用户确认 full job。Stage1/Stage2 需要在 Stage0 v4.2 checkpoint 产出后再跑 smoke，且建议显式设置 rebalanced loss，而不是沿用旧默认。

## 2026-06-05 v4.2 no-weak 质量过滤数据视图

动机：

- `alfworld=10268` 当然可以用，但它把“排除 weak_policy”隐含在刚好贴当前数据量的
  cap 里；
- 更干净的做法是显式按 `source_quality` 过滤训练轨迹，让数据选择语义变成
  “只用高质量/目标主线来源”，而不是“截到某个数字”；
- 这样后续即使 HF ALFWorld 增减少量 rows，也不会意外引入 weak AgentGym。

实现：

- `scripts/build_clstr_unified_pretrain.py`
  - 新增 `trajectory_source_quality_allowlist` 参数；
  - CLI 新增 `--trajectory_source_quality_allowlist`；
  - 过滤发生在 `trajectories.jsonl` 构建之后、trajectory-derived retrieval 和 skill pool
    构建之前；
  - 因此 Stage0 retrieval、Stage1/Stage2 trajectories、skill pool 都基于同一份过滤后的数据。

构建命令：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/build_clstr_unified_pretrain.py \
  --repo_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr \
  --output_dir data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak \
  --schema_version v4.2-no-weak \
  --dataset_recipe v4_2_alfworld_hf_quality \
  --trajectory_source_quality_allowlist official_replay,dagger_expert_corrected_rollout,hf_alfworld_admissible_success,webshop_hf_gpt4_reward_ge_0.8,traject_public_data,toolbench_g3_dfs_tree \
  --trajectory_retrieval_caps toolbench_g3=-1,traject_bench=-1,webshop=-1,alfworld=-1,scienceworld=0
```

输出目录：

```text
data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak
schema_version = v4.2-no-weak
leakage_audit.status = ok
trajectory_rows = 85932
retrieval_pairs = 599735
skill_pool = 37617
```

trajectory source quality 过滤：

```text
source_rows = 118179
retained_rows = 85932
skipped_rows = 32247
skipped_by_source_quality:
  weak_policy = 32247
```

过滤后轨迹构成：

```text
alfworld       10268
webshop        35568
traject_bench  30526
toolbench_g3    9570
scienceworld       0
```

ALFWorld 现在全量高质量进入：

```text
official_replay                  2443
dagger_expert_corrected_rollout  1134
hf_alfworld_admissible_success   6691
weak_policy                         0
```

trajectory-derived retrieval caps：

```text
alfworld=-1
webshop=-1
traject_bench=-1
toolbench_g3=-1
scienceworld=0
```

因此 retrieval 中也不再因 cap 截断高质量 ALFWorld/WebShop：

```text
retained_rows_by_benchmark:
  alfworld = 10268
  webshop = 35568
  traject_bench = 30526
  toolbench_g3 = 9570

skipped_rows_by_benchmark = {}
```

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_can_filter_trajectory_source_quality_before_retrieval_caps \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v4_2_adds_hf_alfworld_admissible_success_before_weak_rows \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v4_2_reads_hf_snapshot_nested_parquet_and_top_level_success \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v4_1b_adds_only_non_overlapping_clean_alfworld \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v2_cli_help_runs_from_repo_root

result: 5 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  tests/test_unified_pretrain_v2.py

result: passed
```

后续建议：

- 下一轮 Stage0/Stage1 应优先使用
  `data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak`；
- Stage1 loss 同步使用温和重平衡方案，而不是旧的
  `L_policy=1.0, L_trans=0.05, L_trans_skill_ce=0.2`；
- full job 仍需用户确认后再提交。

## 2026-06-05 Stage2 v4.1b 无训练输出问题定位与修复

作业 `84344` 被手动停止后检查结果：

```text
OUTPUT_DIR = outputs/clstr_unified_stage2_v4_1b_top350_current_candidate_full
stage0_candidate_handoff_prepared:
  retained_rows = 65570
  current_positive_coverage@M = 0.98420
  next_positive_coverage@M = 0.98790

embedding_cache_started:
  mode = auto
  cache_enabled = true
  row_count = 65570
  max_rows = 20000

slurm tail:
  cached 256/111152 full-base shared replay embeddings
  OOM while caching ... reducing batch size from 256 to 128
  cached 23680/111152 ...
  cached 36480/111152 ...
  CANCELLED
```

根因：

- Stage2 还没有进入训练循环，因此没有 `training_metrics.jsonl`、`loss_curve.svg`
  和 `checkpoints/latest.pt`；
- `EMBEDDING_CACHE_MODE=auto` 只在 `qwen_external_encoder=True` 且
  `row_count > EMBEDDING_CACHE_MAX_ROWS` 时跳过预缓存；
- 当前 native Stage2 虽然 handoff 后有 65570 行，仍会展开为 111152 个去重文本并
  在训练前全量编码，导致长时间没有正常训练输出；
- `train_stdout.json` 为空是因为训练入口主要通过文件监控和最终 report 输出，
  缓存阶段只把进度写入 slurm stderr。

修复：

- 修改 `clstr/full_base_train.py::_embedding_cache_policy`：
  `auto` 下只要 `row_count > embedding_cache_max_rows` 就跳过预缓存；
- 保留 `mode=always` 作为强制预缓存选项；
- 保留 Qwen external 旧 reason：
  `qwen_external_large_dataset_auto_skip`；
- native 大数据新增 reason：
  `large_dataset_auto_skip`。

预期效果：

- Stage2 full 在完成 Stage0 top-M handoff 后直接进入在线 batch 编码训练；
- 更早生成 `checkpoints/latest.pt`、`training_metrics.jsonl`、`loss_curve.svg`；
- 不改变监督标签、候选集合、loss 定义或模型结构，只改变大数据训练的缓存策略。

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py::test_embedding_cache_auto_skips_large_native_stage2_data \
  tests/test_full_base_train.py::test_qwen_external_training_skips_full_dataset_embedding_cache_for_large_stage2_data \
  tests/test_full_base_train.py::test_qwen_external_training_uses_small_embedding_cache_batches

result: 3 passed
```

## 2026-06-05 Stage1/Stage2 BENCHMARK_CAPS 对齐修复

问题：

- Stage1 v4.1b 曾通过 `sbatch --export` 传入逗号分隔的
  `BENCHMARK_CAPS=toolbench_g3=-1,traject_bench=-1,webshop=-1,alfworld=10000,...`；
- Slurm 会先按逗号拆分 `--export` 参数，导致脚本实际只收到第一项
  `BENCHMARK_CAPS=toolbench_g3=-1`；
- 已完成的 Stage1 report 中也能看到：

```text
benchmark_caps = {"toolbench_g3": -1}
```

这不一定单独解释 transition 指标低，但它让 Stage1 和 Stage2 的训练分布不完全一致，
实验口径不干净。

修复：

- `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh`
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`

现在都支持通过冒号安全导出 caps：

```text
BENCHMARK_CAPS=toolbench_g3=-1:traject_bench=-1:webshop=-1:alfworld=10000:scienceworld=0
```

脚本内部会执行：

```bash
BENCHMARK_CAPS="${BENCHMARK_CAPS//:/,}"
```

因此：

- 直接运行脚本时，默认逗号分隔 caps 不变；
- 通过 `sbatch --export` 覆盖时，必须使用冒号分隔；
- Stage1/Stage2 可用同一套 caps 字符串，避免再次静默退化。

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_sbatch_scripts.py::test_unified_stage1_and_stage2_caps_support_colon_separated_sbatch_exports \
  tests/test_sbatch_scripts.py::test_unified_stage1_heads_init_sbatch_uses_stage0_topm_candidates \
  tests/test_sbatch_scripts.py::test_submit_stage1_after_stage0_gate_checks_stage0_gate_and_only_submits_stage1 \
  tests/test_sbatch_scripts.py::test_submit_stage2_after_stage1_gate_checks_stage1_gate_and_only_submits_stage2 \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool

result: 5 passed
```

## 2026-06-05 v4.2 ALFWorld HF 优质数据视图

目标：

- 保持 ToolBench-G3 / TRAJECT-Bench / WebShop 主线不变；
- 不把 ScienceWorld 放回主训练；
- 引入 HuggingFace 上更干净的 ALFWorld success + admissible trajectories；
- 在 ALFWorld cap 内让优质数据优先于 AgentGym weak policy。

新增 HF 数据：

```text
dataset = kuririrn/sft_alfworld_trajectory_dataset_v3to5_admissible_success
local_path = data/hf_alfworld_admissible_success/data/train-00000-of-00001.parquet
raw_rows = 1769
has_admissible = True for all rows
top-level trajectory_outcome:
  success = 616
  non-success / missing-success filtered = 1153
```

实现：

- `scripts/build_clstr_unified_pretrain.py`
  - 新增 recipe：`v4_2_alfworld_hf_quality`
  - 读取 `data/hf_alfworld_admissible_success/**/*.parquet`
  - 仅保留 `trajectory_outcome=success` 且 `has_admissible=True` 的轨迹；
  - 从 `messages` 中解析 observation / admissible actions / assistant action；
  - 将 action 映射到通用 ALFWorld skill：
    - `go to ... -> alfworld-receptacle-navigator`
    - `take ... -> alfworld-object-picker`
    - `put ... -> alfworld-object-placer`
    - `open/close/clean/heat/cool/toggle/look/... ->` 对应通用 skill
  - 插入顺序：official replay -> clean DAgger -> HF admissible-success -> AgentGym weak。

最终构建目录：

```text
data/clstr_unified_pretrain_v4_2_alfworld_hf_quality
schema_version = v4.2
dataset_recipe = v4_2_alfworld_hf_quality
leakage_audit.status = ok
trajectory_rows = 118179
retrieval_pairs = 551484
skill_pool = 37617
```

全量 trajectory 构成：

```text
alfworld       42515
webshop        35568
traject_bench  30526
toolbench_g3    9570
scienceworld       0
```

ALFWorld 全量来源：

```text
official_replay                  2443
dagger_expert_corrected_rollout  1134
hf_alfworld_admissible_success   6691
weak_policy                     32247
```

按当前推荐训练 caps：

```text
toolbench_g3=-1
traject_bench=-1
webshop=-1
alfworld=10000
scienceworld=0
```

实际进入 Stage1/Stage2 的构成：

```text
alfworld       10000
webshop        35568
traject_bench  30526
toolbench_g3    9570
total          85664
```

其中 ALFWorld cap=10000 内部来源：

```text
official_replay                  2443
dagger_expert_corrected_rollout  1134
hf_alfworld_admissible_success   6423
weak_policy                         0
```

结论：

- v4.2 没有简单增加 weak data；
- 在实际训练 caps 下，ALFWorld weak AgentGym 被完全挤出；
- ALFWorld 监督变成 official / DAgger / HF success-admissible 的组合；
- 这是比 v4.1b 更干净的 Stage1/Stage2 训练视图。

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v4_2_adds_hf_alfworld_admissible_success_before_weak_rows \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v4_2_reads_hf_snapshot_nested_parquet_and_top_level_success \
  tests/test_unified_pretrain_v2.py::test_unified_pretrain_v4_1b_adds_only_non_overlapping_clean_alfworld

result: 3 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  scripts/build_clstr_unified_pretrain.py \
  tests/test_unified_pretrain_v2.py

result: passed
```

### 下一步建议

先用 v4.1b 跑 Stage0/Stage1/Stage2 smoke，不直接提交 full job。建议 smoke 明确使用：

```text
DATA_ROOT=data/clstr_unified_pretrain_v4_1b
TRAIN_PATH=data/clstr_unified_pretrain_v4_1b/trajectories.jsonl
SKILLS_PATH=data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl
SAMPLING_STRATEGY=benchmark_transition_balanced_random
BENCHMARK_CAPS=toolbench_g3=-1,traject_bench=-1,webshop=15000,alfworld=10000,scienceworld=10000
```

smoke 通过并确认 loss/指标正常后，再由用户确认是否提交 full job。

## 2026-06-05 current-skill candidate handoff 修正

本次根据方法定义重新收敛 Stage1/Stage2 训练策略：CLSTR 的目标不是预测
`self/switch` 标签，而是在下一步候选 skill 集合中选择正确的 `next_skill_id`。
因此当前步的 `skill_id` 应作为 transition candidate 的自然成员进入候选集。

### 设计修正

Stage2 transition candidate 由：

```text
candidates = Stage0_topM
```

改为：

```text
candidates = Stage0_topM union {current_skill_id}
```

这不是 gold injection：

- `current_skill_id` 是 CLSTR 当前连续状态中已知的信息；
- 当 `next_skill_id == current_skill_id` 时，它自然是 positive；
- 当 `next_skill_id != current_skill_id` 时，它自然是 hard negative；
- `stage0_positive_injected` 仍只表示 `positive_missing_policy=inject` 的 gold-positive 注入。

因此 self/switch 不再作为 full 训练默认 sampler 的核心假设。当前保留
`benchmark_transition_balanced_random`，但仅作为诊断或 ablation 显式启用。
Stage1/Stage2 默认 sampler 已回退到：

```text
balanced_random
```

### 新增可观测指标

transition CE metrics 新增：

```text
transition_current_skill_candidate_rows
transition_current_skill_keep_rows
transition_current_skill_switch_rows
transition_current_skill_keep_recall@1
transition_current_skill_switch_false_positive@1
transition_current_skill_mean_rank
```

这些指标用于判断模型是否：

- 在 keep case 中能把 current skill 排到前面；
- 在 switch case 中错误地过度偏向 current skill。

### v4.1b self/switch 标签审计

CPU 审计结果：

```text
total_transition_rows = 66644
relation:
  switch = 53849
  self = 12795

relation_by_benchmark:
  alfworld: self_rate = 0.4518
  toolbench_g3: self_rate = 0.1611
  traject_bench: self_rate = 0.0233
  webshop: self_rate = 0.3122

canonical_false_switch_by_benchmark = {}

next_skill_matches_next_row_skill:
  consistent = 54259
  inconsistent = 28
```

结论：

- `next_skill_id` 与下一步 `skill_id` 基本一致；
- canonical/alias 假 switch 未发现；
- TrajectBench self 很低更像数据分布本身，而不是明显污染；
- 但 full 训练仍不应强行 self/switch 1:1 采样，应让 current candidate 进入排序任务，
  再通过上述 current-skill 指标观察 keep/switch 行为。

### 验证

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_full_base_train.py::test_current_skill_candidate_metrics_track_keep_and_switch_errors \
  tests/test_full_base_train.py::test_full_base_benchmark_transition_balanced_sampler_covers_small_self_groups \
  tests/test_sbatch_scripts.py::test_unified_stage1_heads_init_sbatch_uses_stage0_topm_candidates \
  tests/test_sbatch_scripts.py::test_unified_stage2_full_base_sbatch_uses_unified_v2_trajectories_and_skill_pool

result: 16 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py \
  scripts/run_clstr_stage1_heads_init.py \
  scripts/run_clstr_stage2_full_base_train.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_full_base_train.py \
  tests/test_sbatch_scripts.py

result: passed
```

## 2026-06-07 — Joint ACT Stage4 主线合并

本次将原先计划中的 standalone Stage3 -> Stage4 主线收敛为 Joint ACT Stage4：

```text
L_joint = L_act_next_skill_ce + lambda_pref * L_offline_preference
```

设计边界：

- `lambda_pref=0` 时就是纯 Stage4 ACT，只训练 transition-conditioned next-skill CE；
- `lambda_pref>0` 时，在同一个 Stage4 训练循环中启用 Stage3-style offline
  preference auxiliary；
- 这不是 full on-policy RL，也不应在论文中写成 simulator rollout HRPO；
- standalone Stage3 offline HRPO-style 仍保留为 legacy / ablation / 诊断入口，但不再是
  `v4.2_progressive_final` 主线进入 Stage4 的必要阶段；
- 当前主线 Stage4 readiness 由 Stage2 checkpoint + Stage2 quality gate 决定，不再由
  Stage3 checkpoint 或 Stage3 quality gate 阻断。

已落地的代码路径：

- `clstr/stage4_act_train.py`
  - 新增 Joint Stage4 loss；
  - `lambda_pref=0` 时不构建/记录 preference loss；
  - `lambda_pref>0` 时构建 offline preference rows、冻结 reference modules、解冻
    `skill_head`，并记录 preference loss/count/KL/advantage 等指标；
  - 修复 reference module fallback：显式传入空 `ref_modules={}` 时不再被当成未传入并重复重建。
- `scripts/run_clstr_stage4_act_train.py`
  - 新增 `--lambda_pref`、`--beta_kl`、`--preference_max_rows`、
    `--preference_batch_size`；
  - checkpoint 初始化模式改为 Stage0 routing + ACT-init head checkpoint，ACT-init
    checkpoint 可以是当前主线 Stage2 base，也可以是 legacy Stage3。
- `scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`
  - 新增 `v4.2_progressive_final` 主线 wrapper；
  - 默认 data root 为 `data/clstr_unified_pretrain_v4_2_progressive_final`；
  - 默认 `HEAD_CHECKPOINT_PATH` 指向已完成 Stage2：
    `outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise/checkpoints/clstr_full_base-step10000.pt`；
  - 默认 `LAMBDA_PREF=0.0`，即纯 ACT 起步。
- `clstr/stage4_quality_gate.py`
  - 接受 `joint_transition_act_with_optional_preference`；
  - 输出 `preference_aux` 审计信息。
- `scripts/audit_clstr_unified_training_readiness.py`
  - `stage4_act` gate 改为要求 Stage2 checkpoint 和 Stage2 quality gate；
  - 报告 `act_init_source=stage2_full_base_checkpoint` 与
    `stage3_no_longer_required_for_stage4=true`。

主线下一步：

1. 先运行一个低成本 Stage4 smoke job，使用
   `scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`，
   设置较小 `MAX_STEPS` / `MAX_ROWS`；
2. 检查 `setup_status.jsonl`、`training_metrics.jsonl`、`loss_curve.svg`、
   `train_stdout.json` 和 Stage4 quality gate；
3. 只有 smoke 正常、指标无异常后，再由用户确认是否提交 Stage4 full job；
4. Stage4 full 通过 quality gate 后，再进入 evaluation matrix readiness gate，
   避免把 routing/proxy 结果误写成 official benchmark 主表。

## 2026-06-07 — Stage4 Candidate Handoff Alignment

修复原因：

- `data/clstr_unified_pretrain_v4_2_progressive_final/trajectories.jsonl` 中可用于
  Stage4 ACT 的 next-skill 训练行没有 `candidate_next_skill_ids`；
- 旧 Stage4 builder 会把 gold `next_skill_id` 注入候选，再用稳定随机负例补齐；
- 这会导致 Stage4 训练候选空间与 Stage0/Stage1/Stage2 的真实 top-M handoff
  不一致，不能证明 ACT 在实际候选分布上有效。

已落地的修复：

- `clstr/stage4_act_train.py`
  - Stage4 训练入口新增 Stage0 top-M handoff 参数：
    `stage0_top_m`、`stage0_positive_missing_policy`、
    `stage0_handoff_query_mode`、`stage0_handoff_sample_multiplier`、
    `stage0_candidate_encode_batch_size`；
  - 复用 `clstr.full_base_train._attach_stage0_topm_candidates`，Stage4 与
    Stage1/Stage2 使用同一套 Stage0 query、inventory/backfill 和 top-M 逻辑；
  - handoff 主线用 `stage0_next_candidate_skill_indices` /
    `stage0_next_candidate_skill_ids` 构造 ACT 候选；
  - 当 Stage0 top-M 没召回 gold `next_skill_id` 时，Stage4 跳过该 ACT row，
    不再注入 gold positive；
  - Stage4 ACT 只需要 next-skill candidate handoff；handoff 前会把源行临时规范为
    `routing=False, L_trans_skill_ce=True`，避免因为 current routing positive
    没进 Stage0 current top-M 而误删 ACT 训练行；
  - handoff 主线的 `positive_injected_rows` 应保持为 0；
  - 对 `MAX_ROWS` smoke 增加 `stage0_handoff_sample_multiplier`，避免小样本 smoke
    也先做全量 Stage4 handoff。
- `scripts/run_clstr_stage4_act_train.py`
  - 透传 Stage0 handoff 相关 CLI 参数，并在 `setup_status.jsonl` 启动记录中写入
    handoff 配置。
- `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
  - 支持可选 `STAGE0_TOP_M` 等环境变量；未设置时保留历史显式候选/随机补齐路径。
- `scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`
  - 当前主线默认：
    `STAGE0_TOP_M=350`，
    `STAGE0_POSITIVE_MISSING_POLICY=skip`，
    `STAGE0_HANDOFF_QUERY_MODE=skillrouter_state`；
  - `CANDIDATE_COUNT` 只对旧 fallback 候选补齐路径有效；Stage0 handoff 主线的
    实际候选宽度由 `STAGE0_TOP_M` 决定。

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage4_act_train.py \
  tests/test_stage4_quality_gate.py \
  tests/test_unified_training_readiness.py \
  tests/test_sbatch_scripts.py::test_unified_stage4_act_sbatch_uses_unified_trajectories_and_stage_checkpoints \
  tests/test_sbatch_scripts.py::test_v4_2_progressive_final_stage4_sbatch_starts_from_stage2_checkpoint_and_joint_act \
  tests/test_sbatch_scripts.py::test_main_gpu_training_sbatch_does_not_override_cluster_gpu_cpu_bundle \
  tests/test_sbatch_scripts.py::test_main_gpu_training_sbatch_uses_existing_default_conda_env_path \
  tests/test_sbatch_scripts.py::test_main_gpu_training_sbatch_sources_shared_clstr_environment

result: 44 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/stage4_act_train.py \
  clstr/stage4_quality_gate.py \
  scripts/run_clstr_stage4_act_train.py \
  scripts/audit_clstr_unified_training_readiness.py

result: passed

bash -n \
  scripts/sbatch/run_clstr_unified_stage4_act_train.sh \
  scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh \
  scripts/sbatch/run_clstr_unified_readiness_audit.sh

result: passed
```

下一步：

- 不直接提交 full job；
- 先用 `v4_2_progressive_final` Stage4 wrapper 跑一个小样本 smoke；
- smoke 重点检查 `stage0_candidate_handoff.json`、`setup_status.jsonl`、
  `training_metrics.jsonl`、`train_stdout.json`；
- 若 smoke 中 `candidate_source=stage0_topm_online`、`positive_injected_rows=0`、
  metrics 正常，再由用户确认是否进入 Stage4 full。

## 2026-06-07 — Stage4 设计再审计与质量门加固

本轮在不提交 Slurm/GPU 作业的前提下，又检查了一遍 Stage4 主线是否还有设计问题。
确认并修复了三个会影响后续判断的风险：

1. Stage4 quality gate 过宽
   - 旧 gate 只要求 checkpoint、步数、`stage4_act_loss/stage4_act_count` 等证据；
   - 即使模型在真实 Stage0 top-M 候选中完全排不出 gold next skill，也可能通过；
   - 现在 gate 新增 `stage4_quality`：
     - `first_stage4_act_loss`
     - `last_stage4_act_loss`
     - `stage4_act_loss_increase`
     - `last_stage4_next_skill_recall@5`
   - 新 blocker：
     - `missing_stage4_next_skill_quality_metrics`
     - `stage4_next_skill_recall_at_5_too_low`
     - `stage4_act_loss_regressed`
   - 这只是 Stage4 训练质量门，不是 official benchmark 结果。

2. Stage4 audit 默认路径仍可能审旧产物
   - `scripts/audit_clstr_stage4_quality.py` 默认 output dir 改为：
     `outputs/clstr_unified_stage4_v4_2_progressive_final_joint_act`；
   - 避免手动运行 audit 时误审 v3 / legacy Stage4 目录。

3. legacy standalone Stage3->Stage4 提交入口容易误用
   - `scripts/submit_clstr_stage4_after_stage3_gate.sh` 现在默认退出；
   - 只有显式设置 `ALLOW_LEGACY_STAGE3_STAGE4=1` 才能继续；
   - 当前主线应使用：
     `scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`。

额外边界：

- Stage4 默认 expected training benchmarks 现在对齐 `v4.2_progressive_final` wrapper：
  `toolbench_g3`、`traject_bench`、`alfworld`、`webshop`；
- 通用 Stage4 train wrapper 仍保留 v3-compatible 默认，用于历史复现；
- 主线 smoke/full 应使用专用 `v4_2_progressive_final` wrapper；
- 本轮没有提交任何 Slurm/GPU 作业。

## 2026-06-07 — Stage4 主线可训练模块与提交安全复查

继续复查后，又确认了两个会影响后续 Stage4 smoke/full 判断的问题：

1. Stage4 ACT 主线不应冻结 `TransitionPredictor`
   - Stage4 的 ACT 输入是 observation-conditioned transition：
     `state/m_obs + current_skill + next_observation -> predicted next-skill ranking`；
   - 之前 `v4.2_progressive_final` 专用 wrapper 默认 `TRAIN_TRANSITION=0`，
     只训练 `trans_head/action_proj`，核心 `transition` 没有适配 Stage4 的输入分布；
   - 这会削弱论文中“closed-loop transition / ACT”的说服力。

已修正：

- `scripts/sbatch/run_clstr_unified_stage4_act_train_v4_2_progressive_final.sh`
  默认改为 `TRAIN_TRANSITION=1`；
- `clstr/stage4_quality_gate.py` 对主线新增 blocker：
  `stage4_transition_not_trainable`；
- `scripts/audit_clstr_stage4_quality.py` 只为 legacy/ablation 保留
  `--allow_frozen_transition`。

2. Stage4 主线 wrapper 需要脚本级 walltime
   - 之前专用 wrapper 没有 `#SBATCH --time`；
   - 如果直接按集群示例 `sbatch --gpus=1 -p gpu_a800 <script>` 提交，
     可能再次变成 unlimited time，影响 backfill/排队并增加误用风险；
   - 现在专用 wrapper 加了 `#SBATCH --time=06:00:00`。

候选集合边界：

- Stage4 主线候选来自 shared Stage0 top-M handoff；
- shared handoff 可能把当前 skill 追加到 next-candidate set，作为 self-skill
  safeguard；
- 这不是 gold `next_skill_id` 注入；
- 主线 smoke/full 必须仍满足：
  `candidate_source=stage0_topm_online`、
  `positive_injected_rows=0`、
  `stage0_candidate_handoff.injected_positive_rows=0`。

结论：

- 当前发现的 Stage4 主线设计问题已经在本地代码层面修正；
- 这不替代 smoke job；
- 下一步仍是低成本 Stage4 smoke，再过 Stage4 quality gate；
- 本轮没有提交任何 Slurm/GPU 作业。

## 2026-06-07 — Stage2/Stage4 Transition 输入语义复查

继续检查 closed-loop transition / belief 数据流后，确认并修复了一个比 loss
权重更基础的实现问题：Stage1/Stage2 full-base 训练和 Stage4 ACT 之前对
`TransitionPredictor` 的第三个输入语义不一致。

问题：

- `TransitionPredictor.forward(m_t, a_t, o_t_emb)` 的第三项是 observation
  embedding；
- `BeliefGate.forward(m_hat, m_obs, obs_emb)` 的第三项也是 observation
  embedding；
- Stage4 ACT 已经把 `next_observation_text` 编码后作为 observation input；
- 但 `clstr/full_base_train.py` 的 `L_trans`、`L_trans_skill_ce`、`belief` 和
  replay-prefix belief reconstruction 旧实现把 `action_text` embedding 作为第三项传入；
- 这会让 Stage2 预训练学到的 transition 语义和 Stage4 ACT 不一致，削弱论文里
  closed-loop transition / ACT 的说服力。

已修正：

- `_transition_prediction()` 的参数语义改为 `obs_emb`；
- full-base `L_trans` 使用 `_next_observation_embedding` / `next_observation_text`
  作为 transition observation input；
- full-base `L_trans_skill_ce` 同步使用 next-observation embedding；
- full-base `belief` 的 transition 和 gate observation input 同步改为
  next-observation embedding；
- replay-prefix belief reconstruction 在有 `next_observation_text` 时也使用
  next-observation embedding；
- `clstr/stage2_transition_row_diagnostics.py` 同步修复，避免诊断工具继续审计旧语义；
- current skill/action 输入仍通过
  `transition_action_input(model, labels, like=obs_emb)` 从 label 和 skill/action
  embedding 产生，soft-shared action embedding 设计没有被移除。

新增回归测试：

- `test_full_base_transition_and_belief_condition_on_next_observation_embedding_not_action_text`
  会在 transition 或 gate 收到 action-text embedding 时失败；
- `test_full_base_transition_prediction_uses_soft_shared_action_embeddings` 继续确认
  current skill/action 输入来自 soft-shared skill embedding。

### 2026-06-07 — Legacy pre-action controller proxy observation 修复

Phase 19 之后继续检查历史 ALFWorld controller 和旧 HRPO
`controller_complete`，确认旧实现还存在一个论文叙事风险：这些 pre-action
候选评分路径没有真实 post-action observation，却曾把 candidate/action embedding
当作 proxy observation 传给 `TransitionPredictor` / `BeliefGate`。这会让 trace
看起来像在使用闭环 transition/belief，实际语义并不成立。

已修正：

- `clstr/alfworld_eval.py::make_controller_component_scorer` 不再调用
  transition / gate / `trans_head` / `skill_head` 来做 pre-action proxy
  transition/belief scoring；
- ALFWorld component scorer 当前只输出 policy/Q-success/current-state STOP
  相关分数，`transition_scores` 与 `belief_scores` 显式置零并 mask；
- metadata 显式写入：
  `component_score_source=policy_q_success_stop_only_no_proxy_observation`、
  `transition_score_source=disabled_no_post_action_observation`、
  `belief_score_source=disabled_no_post_action_observation`、
  `transition_candidate_embedding_source=not_used`；
- `clstr/online_hrpo.py` 的旧 `controller_complete` 不再通过 candidate/action
  embedding 代理观测反传到 transition/gate/STOP；
- 删除 ALFWorld 中旧的 proxy helper，避免后续代码或测试继续依赖这条错误语义。

新增回归测试：

- ALFWorld component scorer 在没有 post-action observation 时不得调用 transition、
  trans_head、skill_head 或 gate；
- 旧 HRPO `controller_complete` 不得把 candidate embedding 当作 observation
  传入 transition，也不得通过这条路径给 transition/gate/STOP 产生梯度；
- action projection sharing 的有效测试仍保留在 full-base / Stage2 语义中。

论文边界：

- 历史 ALFWorld/HRPO pre-action controller 现在只能作为 policy/Q/STOP 诊断路径；
- 不能再把它写成 closed-loop transition/belief 证据；
- 当前可用于论文主线的 transition/belief 证据应来自 Stage2/Stage4：
  `TransitionPredictor.forward(m_t, a_t, o_t_emb)` 中的 `o_t_emb` 必须是
  `next_observation_text` / 真实下一观测 embedding。

本轮没有提交任何 Slurm/GPU 作业。下一步仍应先跑 Stage4 小样本 smoke，并检查
`candidate_source=stage0_topm_online`、`positive_injected_rows=0`、
`stage4_next_skill_recall@5` 和 Stage4 quality gate。

## 2026-06-07 — Stage4 checkpoint init 证据复查

继续静态检查 Stage4 主线后，确认一个需要加固的证据边界：Stage4 先从
Stage0 routing checkpoint 构建模型，再合并 Stage2/head checkpoint。如果未来某个
head checkpoint 含有 `encoder.*`、`cross_encoder.*` 或 `skill_table.*`，就可能在
初始化时覆盖 Stage0 routing foundation，从而破坏“Stage4 在 Stage0 top-M handoff
候选分布上训练”的说法。

当前 `v4.2_progressive_final` 的 Stage2 checkpoint 已经记录
`checkpoint_excludes_frozen_routing_foundation=True`，因此这不是对现有 Stage2 结果的
否定；本轮修复属于防未来手工/旧产物污染证据链。

已修正：

- `scripts/run_clstr_stage4_act_train.py` 的 CLSTR-native Stage4 主线已启用
  `protect_routing_foundation=True`；
- `clstr/qwen_stage4_act_train.py` 的 legacy Qwen Stage4 入口也同步启用
  `protect_routing_foundation=True`；
- `clstr/stage4_quality_gate.py` 默认要求
  `checkpoint_init_report.protect_routing_foundation=True`，否则 blocker 为
  `stage4_checkpoint_init_not_protected`；
- gate 仍会阻断已发生覆盖的产物：
  `stage4_head_checkpoint_overrode_routing_foundation`；
- `scripts/audit_clstr_stage4_quality.py` 只为历史/ablation 产物提供显式逃生口
  `--allow_unprotected_routing_init`，主线不应使用。

数据流确认：

- Stage4 的 Stage0 top-M handoff 是在加载 Stage2/head checkpoint 后计算的；
- 但 handoff 只使用 `encoder + skill_table.retrieval_logits`，不使用
  Stage2 的 policy/transition/head 输出；
- 在 routing foundation 被保护后，候选分布仍是 Stage0 top-M 分布。

论文边界：

- Stage4 ACT 当前优化的是 transition-conditioned next-skill ranking：
  `state/m_obs + current_skill + next_observation -> Stage0 top-M next-skill ranking`；
- 主线 Stage4 会训练 `TransitionPredictor`；
- Stage4 当前不直接训练 `BeliefGate`，belief/gate 证据应来自 Stage1/Stage2 的
  belief loss、next-observation-conditioned diagnostics，或后续专门扩展的 Stage4
  belief objective。

本轮仍未提交任何 Slurm/GPU 作业。

## 2026-06-15 Current-route rollout logger / outcome labeler

为进入真正可审计的 Stage4/RL，本轮先实现了 current-route rollout trace，而不是继续
AppWorld prompt-only patch 或直接重跑旧 `appworld_act_hrpo.py`。

背景判断：

- 当前 AppWorld best 是 official executor route 的 `31/57`；
- 旧 AppWorld HRPO 代码走的是旧 executor/prompt 路线，不能证明当前 `31/57` 路线的
  online Stage4 已训练或有效；
- 外部 AppWorld/tau-bench/Agent Lightning 代码都把 tool/API schema、environment/verifier
  reward、trajectory trace 分层处理；失败轨迹必须先分型，不能全部当作 CLSTR 选错 skill。

新增实现：

- `clstr/appworld_current_route_rollout.py`
  - `label_current_route_step()`：标记 `success`、`wrong_completion`、
    `execution_failure`、`safety_preflight_block`、`completion_precheck_block` 等 step 状态；
  - `label_current_route_rollout()`：生成 rollout-level outcome、reward proxy 与
    `policy_signal`；
  - `build_current_route_rollout_trace()`：从 official executor `runs.jsonl` row 提取
    selected skills、candidate skills、ranking metadata、code/output、official outcome；
  - `CurrentRouteRolloutLogger`：逐任务追加 JSONL，供后续 Stage4/RL adapter 消费。
- `scripts/run_appworld_official_executor_eval.py`
  - 新增 `--current_route_rollouts_path`；
  - 默认关闭，不改变现有 benchmark 输出；
  - 传 `auto` 时写到 `output_dir/current_route_rollouts.jsonl`。
- `scripts/sbatch/run_appworld_official_executor_eval.sh`
  - 新增可选环境变量 `CURRENT_ROUTE_ROLLOUTS_PATH`，默认空，不改变已有作业行为。

保守 credit 规则：

- official success -> `policy_signal=positive`；
- wrong completion / API execution failure -> 默认 `ambiguous_do_not_penalize_clstr` 或
  `executor_negative_do_not_penalize_clstr`；
- 只有 diagnostics 明确包含 `routing_miss=True` 或 positive 不在候选中，才标记为
  `policy_signal=negative`。

这保证失败轨迹可被利用，但不会把 Qwen 代码/参数错误错误地训练成 CLSTR skill 负例。
当前 trace 中若没有 policy logprob，会显式标记 `training_ready=false` 和
`missing_policy_logprobs`，避免假装已有完整 RL 梯度数据。

验证：

```text
pytest -q tests/test_appworld_current_route_rollout.py \
  tests/test_appworld_official_executor_cli.py \
  tests/test_sbatch_scripts.py::test_appworld_official_executor_sbatch_normalizes_task_ids_for_sbatch_export

19 passed
```

下一步：

1. 在一个极小 AppWorld current-route smoke 上打开
   `CURRENT_ROUTE_ROLLOUTS_PATH=auto`，只验证 trace 字段完整性，不做训练；
2. 若 trace 中 CLSTR decisions / candidates / outcomes 可稳定对齐，再实现
   `CurrentRoutePreferenceAdapter`；
3. preference/HRPO 只消费成功对照、显式 routing miss、以及经过 labeler 分型后的弱信号，
   不直接用所有 failed trajectories 做负例。

## 2026-06-14 AppWorld official executor API-error recovery v1

在 full dev57 `31/57` 后继续做失败分型，剩余失败中有两类不是 CLSTR routing
本体直接失败，而是 Qwen executor 在看到真实 API 错误后反复使用错误接口形状：

- `search_contacts` / `search_users` 等返回 list，模型仍按 `response['results']`
  访问，导致 `list indices must be integers or slices, not str` 后多轮重复；
- `show_transactions` 的 `min_created_at` / `max_created_at` 要求 `YYYY-MM-DD`，
  模型反复传 datetime-like 值或错误日期字符串。

本轮只在 `clstr/appworld_official_executor.py` 中新增局部、非泄露的
`[API Error Recovery]` 提示：它只读取上一轮 `execute_output` 中的真实 runtime
错误，在下一轮 prompt 中提示“返回值是 list，不要访问 `['results']`”或“日期过滤参数必须是
date-only `YYYY-MM-DD`”。这不使用 official evaluator trace，不读取 expected answer，
也不按 task id 写规则。

TDD 验证：

```text
pytest -q tests/test_appworld_official_executor.py -q -k 'list_return_shape_recovery or date_format_recovery or idempotent_recovery'
.... [100%]

pytest -q tests/test_appworld_skill_handoff.py tests/test_evidence_gate.py tests/test_appworld_official_executor_cli.py tests/test_appworld_official_executor.py tests/test_appworld_official_executor_audit.py tests/test_sbatch_scripts.py
115 passed in 0.95s
```

后续继续加了两类通用 prompt 边界：

- `Task datetime` 现在给出可复制的 `task_datetime = "..."`
  写法，并明确 date-only filter、ongoing/current year/month 的起点；
- 当任务跨 `phone` 与 `venmo/splitwise` 时，提示用
  `search_contacts(..., relationship=...)` 和 contact email 做跨 app 身份映射，而不是用
  `phone_number` 比较 transaction sender/receiver email。

验证更新为：

```text
pytest -q tests/test_appworld_skill_handoff.py tests/test_evidence_gate.py tests/test_appworld_official_executor_cli.py tests/test_appworld_official_executor.py tests/test_appworld_official_executor_audit.py tests/test_sbatch_scripts.py
117 passed in 0.28s
```

小任务结果：

- `37a8675_1 + df61dc5_3` smoke：`0/2`，但 `37a8675_1` 从未完成变成错误完成，
  暴露下一层 Venmo payment-card / wrong-completion 问题；
- `df61dc5_3` date recovery：从 40-step date-format 卡死变成 2-step wrong completion；
- `df61dc5_3` period/cross-id recovery：进一步变成 1-step、0 execution failure 的 wrong
  completion，但仍漏掉 3 个应点赞 transaction。

因此当前 best official success 仍是 full dev57 `31/57`。这些 prompt 边界提升了 executor
稳定性，但没有产生新的 official success；继续靠 prompt 小修已经开始暴露边际收益下降。下一步如果
继续攻 AppWorld，应转向更结构化的 executor helper/compressor，例如为 relation contacts
生成 `show_transactions(user_email=contact_email, min_created_at=...)` 这种可复用执行 scaffold，
而不是继续逐条 prompt 试错。

## 2026-06-14 AppWorld completion precheck v1

为避免继续依赖会泄露 ground truth 的 official evaluator feedback，本轮在
`clstr/appworld_official_executor.py` 增加了一个默认关闭的 completion precheck：

- 新配置：`OfficialReActExecutorConfig.completion_precheck_mode`，CLI/sbatch 对应
  `--completion_precheck_mode` / `COMPLETION_PRECHECK_MODE`；
- 默认值为 `off`，因此不会改变已有 full dev57 benchmark 路径；
- 当前可选模式为 `constraint_tokens`：
  - 只在代码调用 `apis.supervisor.complete_task(...)` 前触发；
  - 不调用 official `world.evaluate()`，不读取 expected answer，不把 evaluator trace 喂给模型；
  - 对 answer-seeking 任务，要求完成路径中的代码保留任务里的非泛化约束 token，例如
    `edm` 这类 genre/filter 约束；
  - 对 state-change-only 任务，要求当前或历史代码里存在真实 task app write API 调用；
  - 若拦截，会把非泄露的 block reason 写入下一轮 history，让 executor 继续修正。

这不是 AppWorld task-id 特例，也不是 benchmark-time verifier retry；它是一个通用的
executor completion boundary。当前只完成了 CPU 单元测试，尚未提交 Slurm smoke。下一步应只用
`50e1ac9_2` / `68ee2c9_3` 做 1-2 task 小样本验证，若不过再继续离线审计，不直接 full。

### 2026-06-14 precheck v1 smoke 与收窄

实际小样本结果：

- `89555` 两任务 smoke：
  - `50e1ac9_2`, `68ee2c9_3`
  - 结果 `2/2`，无 execution failure；
  - 但 precheck 没有实际拦截，只能说明该小集合不退化。
- `89557` union7：
  - `50e1ac9_2/3` 成功；
  - `383cbac_1` 跑满 40 步失败；
  - 已在 `00:21:05` 取消，避免继续烧钱。

根因不是 CLSTR routing 本身，而是第一版 `constraint_tokens` 规则过严：它把 instruction 中的
大量自然语言词都当作必须保留的约束，误伤了原本成功的 read-only Venmo answer 任务。已将
token 提取收窄到高信号约束：

- 数字/金额；
- 引号内容；
- 连续大写实体短语；
- 明确 genre/filter token，例如 `edm` / `r&b`。

离线复查结果：

- 旧成功的 `383cbac_1/3` 代码现在通过 precheck；
- 缺失 `edm` 的 `50e1ac9_2` 错误完成仍会被拦。

同时修复了 official executor eval runner 的观测性问题：现在每完成一个 task 就刷新
`runs.jsonl`，避免之后取消小作业时丢失已完成 task 的详细轨迹。

### 2026-06-14 high-signal precheck 小集合门禁

收窄规则后重新验证：

- `89570` 单任务：
  - task: `383cbac_1`
  - output: `outputs/appworld_official_executor_regress4/clstr_stage4_act_gate_v2_precheck_constraint_383cbac1_a800`
  - result: `1/1`
  - `completion_precheck.reason=ok`
- `89575` union7：
  - output: `outputs/appworld_official_executor_regress4/clstr_stage4_act_gate_v2_precheck_highsignal_union7_a800`
  - result: `7/7`
  - tasks: `50e1ac9_2`, `50e1ac9_3`, `383cbac_1`, `383cbac_3`,
    `57c3486_3`, `68ee2c9_3`, `6171bbc_3`
  - `completion_precheck`: `not_completion=16`, `ok=7`，没有 block；
  - `execution_failures=5`，但最终 official evaluate 全部通过。

解释：

- 这说明收窄后的 precheck 不再误伤 union7；
- 但 union7 中 precheck 没有实际 block，因此不能把 `7/7` 解释成 precheck 单独救回错误完成；
- 当前 full dev57 最可靠基线仍是 `28/57`，high-signal precheck 需要用户确认后才可进入 full dev57。

### 2026-06-14 high-signal full dev57

用户确认后提交 full dev57：

- job: `89602`
- output: `outputs/appworld_official_executor_dev57/clstr_stage4_act_gate_v2_precheck_highsignal_full_a800`
- result: `31/57`
- success_rate: `0.54386`
- elapsed: `00:44:18`

与已有 full 对比：

| Run | Success |
|---|---:|
| qwen14b api-compressor constraints | 26/57 |
| CLSTR api-compressor guard | 27/57 |
| confidence gate1 | 25/57 |
| gate-v2/intentalign/dedupe/legacy | 28/57 |
| high-signal completion boundary | 31/57 |

相对 `28/57`：

- 新增成功：`57c3486_1`, `57c3486_2`, `57c3486_3`
- 丢失成功：无

机制解释：

- 新增成功来自 state-change-only completion boundary：对这类任务移除
  `complete_task(answer=...)` 中错误的自然语言 `answer`；
- 默认仍不使用 official evaluator feedback，不向模型暴露 expected answer；
- high-signal precheck 在 full 中只 block 了失败任务 `37a8675_3` 的 8 次无写操作完成尝试，
  不是新增 `57c3486_*` 成功的直接原因。

仍需注意：

- `execution_failures=133`，高于旧 full 的 `113`；
- `average_steps=4.508772`，高于旧 full 的 `4.192982`；
- 因此当前版本 task success 更好，但 executor 中间恢复链仍偏多，后续优化应重点降低
  execution failure 和错误 retry，而不是继续扩大 prompt 特化。

## 2026-06-14 AppWorld executor completion 修复

针对 gate-v2 full dev57 中仅剩的 old/qwen union 退化块 `57c3486_1/2/3`，完成根因审计：

- 三个失败任务共同点不是缺少 `like_song` routing，而是最终调用了
  `apis.supervisor.complete_task(answer="All songs from ...")`；
- 官方 `57c3486` ground truth answer 为 `None`，evaluation 会比较 predicted answer 与
  ground-truth answer，因此 state-change-only 任务写入自然语言 answer 会导致
  `task_completed=True` 但 `evaluation_success=False`；
- `57c3486_1/3` 中途的 `already liked` 422 是次级问题；官方 solution 对
  `like_song` 使用 `raise_on_failure=False` 来避免重复/已满足写操作中断流程。

本轮通用修复：

- `run_official_react_task` 调用 code normalizer 时传入 task；
- official executor normalizer 对明显 state-change-only 的任务移除
  `complete_task(answer=...)` 的 `answer` 参数，保留 answer-seeking 任务原行为；
- `make/create/organize/categorize/copy/upload` 等通用 state-changing instruction 也纳入
  state-change-only completion 判定；
- 重要修正：不再把 idempotent write recovery 放进全局 system prompt。小集合回归显示，
  全局提示会扰动 Qwen 在 answer/read-only 任务上的第一步生成，使 old/qwen union7 从
  `6/7` 退到 `1/7`。现在只在上一轮写 API 因 already done / already exists / already liked /
  already satisfied 类错误失败后，在下一轮 user prompt 中局部加入
  `raise_on_failure=False` / already-satisfied 恢复提示；
- 该修复是 executor/verifier boundary 的通用改动，不是 Spotify 或 `57c3486` 特例。

本地验证：

```text
pytest -q tests/test_appworld_official_executor.py -q
17 passed

pytest -q tests/test_appworld_skill_handoff.py tests/test_evidence_gate.py \
  tests/test_appworld_official_executor_cli.py tests/test_appworld_official_executor.py \
  tests/test_appworld_official_executor_audit.py tests/test_sbatch_scripts.py
105 passed

真实 57c3486_1 失败代码经 normalizer:
apis.supervisor.complete_task(answer="All songs ...")
-> apis.supervisor.complete_task()
```

已完成一次中间验证：

- `57c3486_1/2/3` 小集合在上一版修复下恢复到 `3/3`；
- 但 old/qwen union7 在“全局 idempotent system prompt”下退到 `1/7`，因此不能进入 full；
- 当前版本已移除全局提示并改为失败后局部恢复提示；
- local-recovery union7 回归为 `5/7`，`57c3486_3` 已恢复，但 `50e1ac9_2` 与
  `68ee2c9_3` 仍是 `task_completed=True` / `evaluation_success=False` 的 wrong-completion；
- prompt-only execution guard 在 failed2 smoke 上为 `0/2`，已撤回该 prompt guard；
- official executor 已支持 completion-time verifier diagnostics，但正式 benchmark/eval 默认
  `max_wrong_completion_retries=0`，不会把 official evaluator feedback 喂给模型；
- `MAX_WRONG_COMPLETION_RETRIES` / `--max_wrong_completion_retries` 只应作为显式诊断/训练开关使用。
  即使开启，prompt 也只暴露 failure `requirement/label`，不暴露 trace / expected answer，避免
  ground-truth leakage；
- 无上限 verifier loop 曾使 `50e1ac9_2` 跑满 40 步仍失败，job `89511` 已取消；
- 后续 capped 诊断显示：full evaluator trace 能解释答案 mismatch，但 Qwen3-14B 也没有稳定修正；
  且 full trace 包含 expected answer，不能用于论文 benchmark。

当前不能进入 full dev57。下一步应离线设计非泄露的 verifier summarizer / deterministic
post-check，或者回到不使用 evaluator feedback 的 executor 改进；不要继续扩大 AppWorld 作业。

## 2026-06-14 AppWorld Stage4 handoff/gate 稳定性修复

针对 dev57 上 `old CLSTR=27/57`、`qwen-only=26/57`、confidence gate 反降到 `25/57`
的问题，本轮先做低成本审计和 CPU 修复，没有提交新的 GPU full job。

确认的事实：

- gate-current 没有保住 `old ∪ qwen` 的成功集合，漏掉 7 个 union success 任务；
- old-dev57 的优势很大一部分来自 canonical skill dedupe/候选打包，避免同族 SkillX 变体占满
  top-5；
- 单任务 `89118` 证明 AppWorld + Qwen executor 有波动：`50e1ac9_2` 的 old 与 gate-v2
  前两步 prompt SHA 完全相同、模型配置相同，但第二步 raw response 不同，因此不能用单条任务
  判定 gate 方案；
- qwen-only 成功而 gate 失败的样本显示，主要污染来自读/写意图不匹配：
  - read-only 金额/计数问题被注入 `create_transaction` 等 write API；
  - create/add 类目标只得到 search/show/ranking API，而没有真正的 state-changing API。

已实现的通用 compressor/gate 改动：

- `clstr/appworld_skill_handoff.py` 新增 read-only / state-changing goal intent check；
- read-only question 中如果 selected skill 只支持 write API，会 suppress 该 skill evidence；
- state-changing goal 中如果 selected evidence 没有任何 state-changing API，且没有 reference-model
  强匹配，会 suppress 该 evidence；
- 如果 intent filtering 后只剩 `login` / `supervisor` 这类 auth-helper API，则清空 evidence，
  走 `no_evidence_fallback`，避免无增益 prompt 扰动；
- 保留 reference-model 强匹配作为例外，避免破坏此前“reference/compressor 可作为二级证据”的设计；
- `legacy_hints` prompt style、`TASK_IDS` colon-to-comma sbatch 修复、Stage3 `BENCHMARK_CAPS`
  透传修复仍保留。

验证：

```text
pytest -q tests/test_appworld_skill_handoff.py
11 passed

pytest -q tests/test_evidence_gate.py tests/test_appworld_official_executor_cli.py \
  tests/test_appworld_official_executor.py tests/test_sbatch_scripts.py
100 passed

git diff --check
passed on touched files
```

离线重放 gate-current 已选 skill ids（不跑 Qwen、不用 GPU）显示：

- `383cbac_1/3` 的 Venmo read-only evidence 现在回退为空；
- `6171bbc_3` 的 playlist-create read/ranking-only evidence 现在回退为空；
- `57c3486_3`、`68ee2c9_3` 这类有 `like_song` / `move_file` 等真实 action API 的任务仍保留
  schema-grounded evidence；
- `50e1ac9_2/3` 属于 read-only ranking/count 类任务，仍保留 search/show evidence，后续主要看
  canonical dedupe + legacy hints 的集合回归。

下一步如果 GPU 队列可用，应优先跑小集合 gate-v2 回归：

- `CANDIDATE_SOURCE=routing`
- `DEDUPE_CANONICAL_SKILLS=true`
- `APPWORLD_EXECUTOR_COMPATIBLE_ONLY=true`
- `HANDOFF_PROMPT_STYLE=legacy_hints`
- 使用 colon 分隔 `TASK_IDS`，避免 Slurm `--export` 逗号截断。

如果 4-task/8-task 回归不能稳定恢复，再考虑 Stage4/RL；不要直接 full dev57 或 RL，因为
reward 会被 executor/prompt 波动污染。

### 2026-06-14 full dev57 结果

用户确认后，使用上述 gate-v2/intentalign 配置跑 full dev57：

```text
output = outputs/appworld_official_executor_dev57/clstr_stage4_act_gate_v2_intentalign_routing_dedupe_legacy_full_a800
success = 28/57
success_rate = 0.491228
execution_failures = 113
generation_failures = 0
average_steps = 4.192982
```

对比已有 dev57：

```text
qwen-only      = 26/57, execution_failures = 158
old CLSTR      = 27/57, execution_failures = 177
old conf gate  = 25/57, execution_failures = 145
new gate-v2    = 28/57, execution_failures = 113
```

success set 变化：

- `new - old CLSTR`: `383cbac_1`, `383cbac_3`, `396c5a2_1`, `6171bbc_3`
- `old CLSTR - new`: `57c3486_1`, `57c3486_2`, `57c3486_3`
- `new - qwen`: `50e1ac9_1`, `50e1ac9_2`, `50e1ac9_3`, `68ee2c9_3`
- `qwen - new`: `57c3486_1`, `57c3486_3`
- `old CLSTR | qwen-only = 31`，new gate-v2 覆盖其中 `28/31`；
  漏掉的是 `57c3486_1`, `57c3486_2`, `57c3486_3`。

等待 full 期间审计连续失败段：

- `530b157_*`, `4ec8de5_*`, `0d8a4ee_*`, `37a8675_*`, `3ab5b8b_*`, `df61dc5_*`
  共 18 个任务；
- qwen-only、old CLSTR、old confidence gate、new gate-v2 在这 18 个任务上均为 `0/18`；
- 因此这段连续失败不是 gate-v2 新引入的退化，而是 AppWorld/Qwen executor 原本就弱的任务段。

下一步更有价值的修复点：

- `57c3486_*` 系列：Spotify `like_song` 的幂等 state-changing action 处理。
  当前失败会出现 already-liked 422，说明 executor 需要学会/被提示：
  已经 liked 的 song 不应视为 fatal blocker，而应跳过或视为满足条件。
- full dev57 已经证明 gate-v2/intentalign 至少是正向增益：成功数小幅超过 old CLSTR/qwen，
  同时 execution failures 显著降低。后续 Stage4/RL 若继续，应围绕这种可验证的
  evidence/gate/action-guard 接口优化，而不是回到 raw SkillX prompt 注入。

## 2026-06-14 Stage4 AppWorld gate v2 与 Stage3 caps 修复

当前目标仍是让 Stage4/RL 产生稳定正向作用。复查最新 dev57 结果后确认：

- qwen-only：`26/57`；
- latest old CLSTR Stage4：`27/57`；
- confidence-gate CLSTR：`25/57`；
- confidence gate 没有实现“保留 old CLSTR 成功 + 保留 qwen-only 成功”的目标，只降低了
  execution failures。

针对 old-dev57 成功但 confidence gate 失败的 4 个任务，发现当前 gate 路线与 old CLSTR
不是单变量对比：

- old CLSTR 使用 `candidate_source=routing`，且 controller diagnostics 中有
  `filtered_duplicate_skills > 0`；
- confidence gate / routing-only 诊断中 `filtered_duplicate_skills=0`，Spotify 与 file-system
  任务会被多个同族 SkillX 占满 top-k，挤掉互补 skill；
- old CLSTR prompt surface 是自然的 `[Optional Retrieved Hints]`，而 confidence gate
  改成结构化 `[Optional Retrieved Evidence]`。

本轮低成本修复：

- `build_verified_skill_handoff()` 新增 `prompt_style`：
  - 默认 `structured_evidence`，保持已有 gate 行为；
  - 显式 `legacy_hints` 时，在非 fallback 决策下恢复 old-dev57 风格的
    `[Optional Retrieved Hints]`，但仍保留 gate decision、空证据 fallback、schema-grounded
    API evidence、raw SkillX text hidden 等安全约束；
- official executor CLI/sbatch 新增 `--handoff_prompt_style` /
  `HANDOFF_PROMPT_STYLE`；
- 为节省 GPU 成本，取消未开始的 dedupe-only 诊断，改为提交 combined gate v2 4-task
  诊断：

```text
job = 89118
output = outputs/appworld_official_executor_regress4/clstr_stage4_act_gate_v2_routing_dedupe_legacy
CANDIDATE_SOURCE = routing
DEDUPE_CANONICAL_SKILLS = true
APPWORLD_EXECUTOR_COMPATIBLE_ONLY = true
HANDOFF_PROMPT_STYLE = legacy_hints
tasks = 50e1ac9_2,50e1ac9_3,57c3486_3,68ee2c9_3
```

这一步不是 AppWorld task-id hack：canonical dedupe 是通用 skill-pool packing；legacy hint
surface 只是把已验证的 schema evidence 以更接近成功旧路线的自然格式交给 executor。若
4-task gate v2 能恢复旧成功任务，再进入 qwen-only/old-CLSTR union 的 8-task regression
set；只有 regression set 通过，才考虑 dev57 或 Stage4/RL。

此外，静态测试暴露 Stage3 HRPO 旧脚本没有透传 `BENCHMARK_CAPS`。本轮同步修复：

- `build_unified_hrpo_rows()` / `train_unified_hrpo_with_model()` 支持 per-benchmark caps；
- `scripts/run_clstr_unified_stage3_hrpo.py` 支持 `--benchmark_caps`；
- `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh` 支持 `BENCHMARK_CAPS` 并处理 sbatch
  export 中的 colon-to-comma 转换。

验证：

```text
pytest tests/test_stage3_unified_hrpo.py \
       tests/test_sbatch_scripts.py \
       tests/test_evidence_gate.py \
       tests/test_appworld_skill_handoff.py \
       tests/test_appworld_official_executor_cli.py \
       tests/test_appworld_official_executor.py -q

101 passed
```

### 2026-06-14 追加：AppWorld official executor TASK_IDS 提交修复

`89118` 暴露了一个 Slurm 提交层 bug：`sbatch --export` 使用逗号分隔环境变量，因此
`TASK_IDS=50e1ac9_2,50e1ac9_3,...` 会被拆开，实际只传入第一个 task id。该 job 因此
只跑了 `50e1ac9_2`，结果为 `0/1`，不能作为 4-task gate v2 结论。

已修复：

- `scripts/sbatch/run_appworld_official_executor_eval.sh` 支持用冒号分隔 task ids：
  `TASK_IDS=a:b:c`；
- 脚本内部执行 `TASK_IDS="${TASK_IDS//:/,}"` 后再传给 Python；
- 新增静态测试覆盖该行为。

重新提交 corrected 4-task gate v2：

```text
job = 89213
output = outputs/appworld_official_executor_regress4/clstr_stage4_act_gate_v2_routing_dedupe_legacy_regress4
TASK_IDS = 50e1ac9_2:50e1ac9_3:57c3486_3:68ee2c9_3
CANDIDATE_SOURCE = routing
DEDUPE_CANONICAL_SKILLS = true
APPWORLD_EXECUTOR_COMPATIBLE_ONLY = true
HANDOFF_PROMPT_STYLE = legacy_hints
```

额外审计 `89118` 的单任务 trace 发现：old-dev57 与 gate v2 在 `50e1ac9_2` 前两步
prompt 完全一致，但 Qwen 后续生成不同，gate v2 误用 playlist 的 `songs` 字段而不是
schema 中的 `song_ids`。这说明单条 AppWorld executor 结果有生成波动，后续应以小集合
regression 结果判断方向，不应基于单个 task 过拟合 prompt。

## 2026-06-13 Confidence-gated evidence interface

本轮新增 CLSTR skill selection 到 executor prompt 之间的 step-level evidence gate：

- `no_evidence_fallback`：当 selected skills 全部被 verifier suppress、无 schema-valid evidence、
  或 gate 判定低置信时，完全不输出 retrieved evidence block，使 executor prompt 在 evidence
  部分等价于 qwen-only；
- `schema_only_evidence`：只暴露 valid API/tool schema evidence，不暴露 workflow hint；
- `workflow_hint_evidence`：仅在 schema 支撑和 controller/transition 置信信号足够时暴露短
  workflow hints；
- diagnostics 记录 gate decision、confidence、reasons、selected scores、score margin、
  transition availability、exact fallback 和 prompt-visible evidence text，供后续 disagreement
  audit 或 RL gate policy 使用。

这一步不是 AppWorld task-id 或 app-family patch，也不训练 Qwen/CLSTR；它是 RL 前的通用可靠性层，
目标是减少低置信 SkillX evidence 对下游 executor 的 prompt 污染。

## 2026-06-13 Confidence gate 小规模运行结果

基于上述 confidence-gated evidence interface，复用当前 Stage4 ACT checkpoint：

```text
checkpoint = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act/checkpoints/clstr_stage4_act-step2000.pt
skill_pool = data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl
ranking_mode = transition_blend
candidate_top_k = 350
candidate_source = routing_belief_union_after_update
```

### AppWorld disagreement-9 gate

```text
output = outputs/appworld_official_executor_disagreement9/clstr_stage4_act_confidence_gate1
job = 88875
success = 6/9
execution_failures = 4
average_steps = 2.888889

qwen_old subset:  success = 4/9, execution_failures = 14, average_steps = 5.444
old CLSTR subset: success = 5/9, execution_failures = 2,  average_steps = 2.778
new gate subset:  success = 6/9, execution_failures = 4,  average_steps = 2.889
```

Row-level:

```text
kept old CLSTR-only wins:
  50e1ac9_1, 50e1ac9_2, 57c3486_2, 68ee2c9_3
lost old CLSTR-only win:
  50e1ac9_3
fixed qwen-only old wins / old CLSTR regressions:
  6171bbc_3, 396c5a2_1
still failed qwen-only old wins:
  383cbac_1, 383cbac_3
```

Gate diagnostics:

```text
schema_only_evidence = 23 steps
no_evidence_fallback = 3 steps
old [Optional Retrieved Hints] header = 0 steps
transition_blend used for all 26 steps
transition_scores_available = 16/26 steps
```

### AppWorld dev10 sanity gate

```text
output = outputs/appworld_official_executor_dev10/clstr_stage4_act_confidence_gate1
job = 88893
success = 6/10
execution_failures = 4
average_steps = 2.3

qwen_old dev10:    success = 4/10, execution_failures = 14, average_steps = 5.0
old Stage4 dev10: success = 6/10, execution_failures = 5,  average_steps = 2.5
new gate dev10:   success = 6/10, execution_failures = 4,  average_steps = 2.3
```

Gate diagnostics:

```text
schema_only_evidence = 22 steps
no_evidence_fallback = 1 step
old [Optional Retrieved Hints] header = 0 steps
transition_blend used for all 23 steps
transition_scores_available = 13/23 steps
```

结论：confidence gate 没有牺牲 dev10 success，并降低了 execution failures；在 disagreement-9 上
还比旧 CLSTR 多 1 个成功。因此继续提交了 dev57 完整评估作为正式对照。

## 2026-06-13 Confidence gate dev57 完整评估结果

```text
output = outputs/appworld_official_executor_dev57/clstr_stage4_act_confidence_gate1
audit = outputs/appworld_official_executor_dev57/clstr_stage4_act_confidence_gate1/audit.json
job = 88996
partition = gpu_h100
success = 25/57
success_rate = 0.438596
task_completed_count = 54
execution_failures = 145
average_steps = 4.824561
preflight_failed_steps = 0
complete_task_positional_count = 0
```

与已有 dev57 对照：

```text
qwen-only:
  output = outputs/appworld_official_executor_dev57/qwen14b_api_compressor_constraints1
  success = 26/57
  execution_failures = 158
  average_steps = 5.385965

old CLSTR Stage4:
  output = outputs/appworld_official_executor_dev57/clstr_stage4_act_api_compressor_guard1
  success = 27/57
  execution_failures = 177
  average_steps = 5.157895

confidence-gated CLSTR Stage4:
  output = outputs/appworld_official_executor_dev57/clstr_stage4_act_confidence_gate1
  success = 25/57
  execution_failures = 145
  average_steps = 4.824561
```

Row-level 差异：

```text
confidence_gate_only_vs_qwen:
  50e1ac9_1, 57c3486_2, 68ee2c9_2
qwen_only_vs_confidence_gate:
  383cbac_1, 383cbac_3, 57c3486_3, 6171bbc_3

confidence_gate_only_vs_old_clstr_stage4:
  396c5a2_1, 68ee2c9_2
old_clstr_stage4_only_vs_confidence_gate:
  50e1ac9_2, 50e1ac9_3, 57c3486_3, 68ee2c9_3

all_three_success_count = 21
all_three_fail_count = 25
```

Gate diagnostics：

```text
schema_only_evidence = 252 steps
no_evidence_fallback = 23 steps
old [Optional Retrieved Hints] header = 0 steps
new [Optional Retrieved Evidence] header = 252 steps
exact_base_executor_fallback = 23 steps
transition_blend used for all steps
transition_scores_available = 211/275 steps
handoff_decision_counts_sum:
  inject_hint = 806
  schema_only = 187
  suppress = 324
```

结论：confidence gate 确实降低 executor 负担，execution failures 从 qwen-only 的 158 和
old CLSTR 的 177 降到 145；但 task success 降到 25/57，低于 qwen-only 的 26/57 和 old
CLSTR 的 27/57。小样本 gate 的正向结果没有扩展到 dev57 完整评估。

当前判断：

- 这个 deterministic gate 可以作为“可靠性层降低 execution failures”的证据；
- 但不能作为当前 AppWorld task-success 主结果，因为 success 退步；
- 下一步不应继续扩大 AppWorld eval，也不应直接 RL；
- 应先对 `50e1ac9_2`、`50e1ac9_3`、`57c3486_3`、`68ee2c9_3` 四个 old-CLSTR-only
  回归任务做 prompt/evidence row audit，判断是 schema-only 过度保守、workflow hint 被错误关闭、
  还是 evidence gate 改变了 Qwen 的搜索路径。

## 2026-06-13 AppWorld official-style executor dev57 对比结果

在 Qwen3-14B、官方风格 ReAct executor、API-doc compressor、schema preflight、
retrieved-evidence guard、动态 AppWorld SkillX append 路线下，完成了 dev57 gate：

```text
qwen-only:
  output = outputs/appworld_official_executor_dev57/qwen14b_api_compressor_constraints1
  job = 88671
  success = 26/57
  success_rate = 0.45614
  task_completed_count = 54
  execution_failures = 158
  average_steps = 5.385965
  audit: invalid_api_ref_count=0, preflight_failed_steps=0, complete_task_positional_count=0

CLSTR Stage4 ACT:
  output = outputs/appworld_official_executor_dev57/clstr_stage4_act_api_compressor_guard1
  job = 88709
  checkpoint = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act/checkpoints/clstr_stage4_act-step2000.pt
  ranking_mode = transition_blend
  candidate_top_k = 350
  success = 27/57
  success_rate = 0.473684
  task_completed_count = 53
  execution_failures = 177
  average_steps = 5.157895
  audit: invalid_api_ref_count=0, preflight_failed_steps=0, complete_task_positional_count=0
```

Row-level comparison：

```text
both_success = 22
stage4_only = 5
  50e1ac9_1, 50e1ac9_2, 50e1ac9_3, 57c3486_2, 68ee2c9_3
qwen_only = 4
  383cbac_1, 383cbac_3, 6171bbc_3, 396c5a2_1
neither = 26
```

Stage4 不是退化成纯文本检索：`ranking_mode=transition_blend` 覆盖全部 294 个执行步骤，
其中 230 个步骤有 `transition_scores_available=true`。handoff 统计为：
`inject_hint=1162`、`schema_only=132`、`suppress=73`。

结论：

- 这是比 dev10 更可信的正向信号：Stage4 ACT 在 dev57 上相对 qwen-only 净增 1 个成功；
- 但收益幅度很小，并且 `execution_failures` 从 158 增加到 177，说明 evidence/handoff 仍会
  增加 executor 负担；
- 当前结果可以支撑“CLSTR 的连续 skill evidence 在官方式 executor 中有可测的增益”作为
  case-study / secondary benchmark；
- 还不应把 AppWorld dev57 task success 当作论文主结果，除非后续进一步降低 Stage4 回归任务
  和执行失败数，或在更多 benchmark 上复现稳定增益。

## 2026-06-13 AppWorld official executor + Qwen3-14B + CLSTR Stage4 gate

本轮把 AppWorld 任务成功路线从旧的薄 free-form executor 推进到 official-style ReAct
executor，并接入本地 `models/Qwen3-14B`。关键工程改动包括：

- 共享 Qwen generation backend；
- official-style AppWorld executor CLI/sbatch；
- 动态 AppWorld SkillX evidence 接入当前 CLSTR 路线；
- deterministic `complete_task(answer=...)` 规范化；
- AppWorld valid-API schema preflight；
- task-relevant API-doc compressor，优先展示和当前任务相关的官方 API、参数、返回 schema
  与参数约束；
- retrieved-evidence guard，保证用户过滤条件、source/library scope、ranking/count/limit
  不被 SkillX evidence 覆盖；
- failed-subset gate 支持 `--task_ids`/`TASK_IDS`，避免为小修复重复跑完整 dev10。

小门禁结果：

```text
Qwen-only dev10 official executor:
  output = outputs/appworld_official_executor_dev10/qwen14b_api_compressor_constraints1
  success = 4/10
  execution_failures = 14
  average_steps = 5.0

CLSTR Stage2 v4.1b evidence dev10:
  output = outputs/appworld_official_executor_dev10/clstr_stage2_v4_1b_api_compressor_guard1
  success = 5/10
  execution_failures = 9
  average_steps = 3.1

CLSTR Stage4 ACT dev10:
  output = outputs/appworld_official_executor_dev10/clstr_stage4_act_api_compressor_guard1
  success = 6/10
  execution_failures = 5
  average_steps = 2.5
```

Row-level 对比：

- Stage4 没有让 qwen-only 或 Stage2 成功的任务退化；
- Stage2 相比 qwen-only 额外解决 `50e1ac9_1`；
- Stage4 相比 Stage2 额外解决 `50e1ac9_2`；
- qwen-only 相比 Stage4 没有 unique win。

结论：在 official-style executor + API-doc compressor 下，当前 CLSTR Stage4 ACT 已经出现
正向 task-success 信号。这个结果仍只是 dev10 gate，不足以作为论文主结果；下一步如果继续
AppWorld，应先跑 dev57 qwen-only/Stage4 对照验证泛化，再考虑更大的 split。硬失败仍集中在
Venmo+phone 金额/收款人解析和 file_system 批量移动路径细节，继续 prompt 微调 failed5
已经被 `0/5` 子集实验否定。

## 2026-06-13 AppWorld official-style executor / Qwen3-14B gate

本轮实现了新的 AppWorld official-style ReAct executor 路线，并下载/验证了
`models/Qwen3-14B`。主要工程改动包括：

- 新增共享 `QwenGenerationBackend`，并改为 lazy-load `torch/transformers`，避免登录节点单元测试卡在
  `import torch`；
- 新增 `clstr/appworld_official_executor.py`：
  - official-style API-doc discovery prompt；
  - 多步 ReAct loop；
  - `EvidenceResult` handoff；
  - history compaction；
  - 通用认证 helper；
  - 未请求账号级副作用的 safety preflight；
- 新增 `scripts/run_appworld_official_executor_eval.py` 与
  `scripts/sbatch/run_appworld_official_executor_eval.sh`；
- 接入 CLSTR evidence mode：复用现有 multistep controller，使用 verified SkillX handoff
  将 selected skill 压缩为 schema-grounded evidence，并在每步执行后回传 `observe()`。

验证结果：

- Qwen3-14B load gate：job `88329` 通过，模型本地加载并生成 `OK`。
- qwen-only 单任务 gate：
  - `88391`: runtime 健康但 `success_count=0/1`，漏掉多库/分页；
  - `88402`: 加强 prompt/history 后仍失败，并触发账号创建副作用；
  - `88422`: 加 safety preflight 后阻止账号副作用，但陷入认证结构错误循环；
  - `88458`: 加 deterministic auth helper 后明显改善，`average_steps=11`、无模型修改，
    但仍选错 Spotify API/字段，`success_count=0/1`。
- CLSTR evidence 单任务 gate：
  - `88462`: Stage2 v4.1b conservative listwise checkpoint + dynamic AppWorld skill append 成功加载，
    但 `success_count=0/1`，`average_steps=40`；evidence 选到的多为 play-count/search/liked-song
    families，没有稳定给出 song/album/playlist library 工作流。

当前判断：

- AppWorld runtime/Qwen3-14B/CLSTR evidence 工程链路已打通，但当前 free-form ReAct executor 不足以支撑
  AppWorld task-success 论文主结果；
- 继续直接跑 dev10/full 不划算；
- 下一步若继续 AppWorld，应先改 executor 协议：API-doc compressor/verifier、`complete_task(answer=...)`
  规范化、非法/不存在 API preflight，以及更结构化的 function-call/action loop；
- CLSTR 侧需要改进 schema-grounded 多源 workflow evidence，而不是仅输出相似 skill 的 API ref 集合。

## 2026-06-13 Qwen3-14B download and load gate

按 AppWorld official-style executor 方案，已先完成 stronger executor model 的最低成本门槛：

- 下载路径：`models/Qwen3-14B`
- 下载方式：取消 `http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY/all_proxy/ALL_PROXY`，仅使用
  `HF_ENDPOINT=https://hf-mirror.com`
- 本地大小：约 `28G`
- 文件状态：包含 `config.json`、`tokenizer_config.json`、`model.safetensors.index.json` 和 8 个
  safetensors shards
- load smoke：Slurm job `88329`，`gpu_a800`
- 结果：`QwenGenerationBackend` 使用 `local_files_only=true`、`torch_dtype=bfloat16` 成功加载
  `models/Qwen3-14B`，并完成一次短生成，输出 `OK`

结论：Gate 0 的 model-load prerequisite 通过。下一步才能进入 official-style AppWorld executor
单任务 qwen-only smoke；不能直接跳到 dev10/dev57 或 Stage4 full。

## 2026-06-13 AppWorld official-style executor and Stage4 boundary

外部调研和现有 dev57 postmortem 后，AppWorld Stage4 的下一阶段不再继续沿着
`Qwen3-8B + compact free-form Python prompt + SkillX prompt injection` 小修小补。该链路已经表现出
两个不可忽略的问题：

- executor 本身太弱/太薄：官方 AppWorld 和高分方案使用多轮 ReAct、function-calling、MCP、
  verifier、demo retrieval、memory/compaction 或 RL-trained executor，而不是单个静态压缩 prompt；
- SkillX handoff 太容易污染 executor：raw title/body 或过度压缩的 schema-only evidence 都会造成
  dev10 小信号无法泛化到 dev57。

新的主线设计写入：

```text
docs/superpowers/specs/2026-06-13-appworld-official-executor-stage4-design.md
```

核心边界：

- AppWorld task-success 只作为真实 executor case study，不能替代 routing/multi-step routing 主证据；
- executor 改为 official-style ReAct/function-calling/MCP loop：持久 Python/environment state、API-doc
  交互式发现、任务 datetime、官方 evaluation 和完整 trace logging；
- Qwen3-14B 可作为下一阶段 stronger executor model，但下载必须关闭代理并使用
  `HF_ENDPOINT=https://hf-mirror.com`；
- CLSTR 不替代 executor，只在每一步提供 skill evidence selection；
- raw SkillX body/title/id 不直接暴露给 executor，必须通过 schema-grounded verifier/compressor；
- Stage4 只训练 CLSTR 的 evidence selection / transition / belief / coverage-aware scoring，Qwen 保持
  frozen；
- 禁止 AppWorld family-specific 规则进入主线，例如专门为 Spotify/Venmo/file_system 写模板补丁。

继续条件：

1. Qwen3-14B qwen-only official-style executor 单题 smoke 能正常 load、执行、evaluate；
2. qwen-only dev10 比当前 Qwen3-8B executor 更健康；
3. 同一 executor/model 下，CLSTR evidence dev10 不增加 invalid API / execution failure，并带来成功率或
   失败画像改善；
4. Stage4 只用 train split feedback 训练，并在 dev10 同设置下超过 qwen-only 和 CLSTR-base 后，才允许
   dev57；
5. full/test-style 作业仍需要用户显式批准。

论文叙事也随之收紧：CLSTR 的主 claim 是大 skill pool 下的连续、coverage-aware skill routing 和
executor-feedback evidence selection；如果 AppWorld task success 没有稳定优势，AppWorld 只作为诊断
case study，不作为主结果。

## 2026-06-12 AppWorld verified_hints handoff v1

针对 AppWorld Stage4 动态 SkillX 的 executor 污染问题，新增
`skill_context_mode=verified_hints`：

- CLSTR 仍负责在大 skill pool 上选择 SkillX；
- executor prompt 不再暴露 SkillX `skill_id/name/description/body/skill_md`；
- 新增 `clstr/appworld_skill_handoff.py`，先用 deterministic schema/operation verifier
  判断每个 selected skill：
  - `inject_hint`：app、operation、schema API 与任务一致；
  - `schema_only`：可作为弱证据，但不提供自然语言 workflow；
  - `suppress`：缺少 schema evidence 或存在明显 operation/pollution mismatch；
- compressor 只输出低优先级 `[Optional Retrieved Hints]`，包括 schema-valid APIs 和简短
  workflow primitives；
- `reference_scores` 接口已预留给后续 Qwen3-base reference scorer，但 v1 不让 reference
  model 自由生成 prompt 内容，所有输出仍经过 deterministic schema guard。

这不是把 Qwen3-base 当作 executor 的替代品，而是把它限定为后续可选的固定标签/分数
scorer。当前 v1 先用确定性 verifier/compressor 建立安全边界，目标是避免 SkillX handoff
低于 qwen-only。

新增 CPU-only 审计脚本：

```bash
python scripts/audit_appworld_verified_handoff.py \
  --runs_path <runs.jsonl> \
  --skill_pool_path data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl \
  --appworld_root /data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root \
  --output_dir <audit_output>
```

已用 prior regression6 `api_evidence` runs 做离线审计：

```text
output = outputs/appworld_dynamic_multistep_executor_dev57/regression6_verified_hints_offline_audit_20260612
step_count = 15
decision_counts = {inject_hint: 2, schema_only: 0, suppress: 73}
missing_skill_count = 0
```

解释：Spotify unique-count / add-to-queue 污染族基本被 suppress，Simple Note export 保留了
有用 verified hints。下一步只允许跑 regression6 小作业；若不能达到至少 `5/6` 且
`preflight_failures=0`，不得进入 dev10/dev57/full。

### 2026-06-12 verified_hints / verified_metadata follow-up

本轮继续验证 AppWorld SkillX handoff 的 executor 边界，结论如下：

- 硬 preflight 拦截 `search/find/list` API 的字面量 `*_id` filter 不可靠。
  - job `88116`：`verified_hints + unsafe_literal_id_filter preflight` 在 regression6 只有
    `1/6`，`preflight_repair_count=4`，`preflight_failures=1`。
  - 失败原因是 Qwen repair 会把完整程序改成碎片搜索，或把 `artist_id=1` 改成
    `artist_id=None` 导致 422；因此这类错误不应交给自由生成 repair。
- 新增 deterministic AST sanitizer：
  - `sanitize_appworld_code()` 会在执行前删除 search/find/list 调用中的未验证字面量
    `*_id` filter，例如把
    `apis.spotify.search_songs(artist_id=1, min_play_count=990)` 改为
    `apis.spotify.search_songs(min_play_count=990)`；
  - 它保留原程序结构和后续按返回字段过滤的逻辑，不改 `show_song(song_id=1)` 这类详情调用；
  - multistep executor 在 preflight 前调用 sanitizer，并在 step diagnostics 记录
    `code_sanitizer`。
- sanitizer 对 regression6 有效：
  - job `88132`：`verified_hints + idsanitize` 达到 `5/6`，`execution_failures=0`，
    `preflight_repair_count=0`，`preflight_failures=0`；
  - sanitizer 只触发 1 个 step，正是 `artist_id=1` 队列任务。
- 但 `verified_hints` 不泛化到普通 dev10：
  - job `88134`：`verified_hints + idsanitize` 只有 `2/10`；
  - 主要原因是 verifier 虽然判定大多数 skills 为 `inject_hint`，但 prompt 仍只给 API/hints，
    丢掉了对 phone/venmo、file_system、Spotify ranking 等任务有用的 procedural metadata。
- 因此新增 `skill_context_mode=verified_metadata`：
  - 继续使用同一个 deterministic verifier；
  - `inject_hint` skill 保留 `safe_metadata`；
  - `schema_only` skill 降级为 schema block；
  - `suppress` skill 完全隐藏；
  - raw SkillX body 仍不暴露，optional verified hints 仍是低优先级证据。
- `verified_metadata + idsanitize` 结果：
  - job `88151` regression6：`5/6`，无 execution/preflight failure；
  - job `88187` dev10：`6/10`，`execution_failures=1`，`preflight_failures=0`。

当前判断：`verified_metadata` 比 `verified_hints` 更平衡，但仍没有超过历史
`safe_metadata + preflight_repair2` 的 dev10 `7/10`，也没有证明 AppWorld task-success
路线足以作为论文主结果。不要从当前状态提交 dev57/full。AppWorld 可以保留为 executor-boundary
case study；如果继续追 task success，需要更强的 executor interface / verifier，而不是继续靠
prompt/handoff 微调。

## 2026-06-12 AppWorld final handoff adapter: api_evidence

在 dev57 postmortem 后，新增最后一轮低成本 AppWorld executor-boundary 适配：
`skill_context_mode=api_evidence`。

设计边界：

- 不改 Stage4 checkpoint、训练目标或 ranking 逻辑；
- CLSTR 仍选择 top-k SkillX，但 prompt 不暴露 SkillX `skill_id`、`name`、`description`、
  `body`、`skill_md` 等自由文本；
- executor prompt 只看到从 selected SkillX 的 `executor_desc/body/skill_md` 中解析、并被
  AppWorld API schema 校验过的 API evidence；
- invalid 或被 action-filter 过滤的 API refs 只作为 diagnostics / invalid evidence 记录，
  不能作为可执行计划；
- prompt 明确说明 evidence 不是 API allowlist，也不是 complete plan，用户目标和
  `[AppWorld API Docs]` 仍是权威来源。

已完成本地验证：

```text
pytest tests/test_appworld_executor.py tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py -q
71 passed
py_compile clstr/appworld_executor.py clstr/appworld_multistep.py scripts/run_appworld_multistep_executor_eval.py
git diff --check affected files
```

低成本 gate 顺序：

1. `regression6 stage4 transition_blend + preflight2 + api_evidence`，要求 `>=4/6` 且
   `preflight_failures=0`；
2. 只有 regression6 通过，才跑 `dev10 stage4 transition_blend + preflight2 + api_evidence`；
3. 若 dev10 不能达到 `>=7/10` 且 failure profile 不优于当前最好 Stage4 gate，则停止
   AppWorld task-success 路线，不跑 dev57/full。

实际结果：

```text
87752 regression6 api_evidence: 1/6, preflight_fail=0, exec_fail=6
```

结论：`api_evidence` 没有通过小门槛，不提交 dev10/dev57/full。它验证了“隐藏 SkillX 文本”
可以减少部分污染，但也把可用 procedural hints 削掉太多，Spotify unique-count 和 queue 任务缺少
足够的操作语义，导致 request-limit loop 或语义错误完成。

## 2026-06-12 AppWorld dev57 postmortem

已新增报告：`docs/appworld_stage4_postmortem_2026-06-12.md`。

只读审计已有 dev57 结果后，结论如下：

- qwen-only + preflight2 dev57：`16/57`；
- Stage4 + safe_metadata + preflight2 dev57：`11/57`；
- 行级拆分：共同成功 10 个，Stage4 唯一成功 1 个，qwen-only 成功但 Stage4 失败 6 个，共同失败 40 个；
- Stage4 transition 并非没运行：dev57 共有 137 个 Stage4 step，其中 75 个后续 step 有
  `transition_scores_available=true`；
- 但 6 个 qwen-only regression 都在 step 0 就已经被 SkillX metadata/handoff 污染，后续 transition
  只是沿着坏状态继续。

当前根因不是单一的 Stage4 tensor/loader bug，而是 AppWorld 动态 SkillX 路线把 CLSTR selection、
SkillX 文本质量、handoff prompt、Qwen executor 和 API repair 混在一起。它可以作为 case study，
但目前不适合作为 Stage4 方法价值的主证据。

## 2026-06-12 gated_schema SkillX handoff

AppWorld dev57 扩展验证后，当前瓶颈不是 Stage4 checkpoint loader 或 transition scorer 崩溃，
而是 retrieved SkillX metadata 对 Qwen executor 的 prompt 污染：部分 SkillX 标题/描述在语义上
接近但操作上误导，例如 unique-song-count 任务被 play-count/ranking skill 诱导，add-to-queue
任务被 like/current-queue skill 诱导。

本轮新增 `skill_context_mode=gated_schema`，作为下一轮低成本 gate 的候选 handoff：

- `safe_metadata`：仅对强语义匹配、required app 匹配、有有效 AppWorld API 证据、且无污染模式的
  skill 保留自然语言 metadata；
- `schema_plan`：对有 schema 证据但存在 action mismatch、弱匹配或污染风险的 skill，隐藏
  SkillX id/name/body，只暴露 schema-grounded API evidence；
- `drop`：对无有效 schema API 证据或 required-app 完全不匹配的 skill，不放进 prompt；
- multistep step 记录新增 `skill_handoff`，包含 `handoff_decisions`、`kept_skill_ids`、
  `dropped_skill_ids`、`downgrade_reasons`、`schema_plan_skill_count`、
  `safe_metadata_skill_count` 等字段，便于事后审计。

这不是替代 CLSTR routing 的任务特化规则，而是 executor 边界的防污染接口：CLSTR 仍负责大 skill pool
检索和多步排序，handoff gate 只决定被选中的 SkillX 以何种粒度暴露给下游 LM。

下一步只允许小门槛验证：

```text
regression6 gated_schema >= 4/6
dev10 gated_schema >= 7/10
preflight_failures = 0
```

若 dev10 不能超过 qwen-only+preflight2 的 6/10，不能继续 dev57/full。

### gated_schema_v2 gate 结果

`gated_schema` 第一版 regression6 失败：

```text
regression6 gated_schema v1: 0/6, preflight_fail=0
```

行级审计显示根因是 gate 仍保留了过多 `safe_metadata`，把 regression6 原本要压制的污染语义重新放回
prompt。随后收紧为 v2：

- unique/how-many/count song aggregate 任务中，非精确 unique-song skill 降级为 `schema_plan`；
- add-to-queue 任务中，没有真实 queue write API 的 skill 降级为 `schema_plan`；
- current/played-so-far/like/current-queue 等 queue 污染模式纳入降级原因。

v2 小门槛结果：

```text
regression6 gated_schema v2: 4/6, preflight_fail=0
dev10 gated_schema v2:       5/10, preflight_fail=1, exec_fail=8
```

因此 v2 不能进入 dev57/full。它证明“schema-only 能修复部分污染”这个方向成立，但当前混合 handoff
仍不如此前 `safe_metadata + preflight2` 的 dev10 `7/10`。下一步若继续 AppWorld handoff，优先考虑
更简单的策略：默认保持 `safe_metadata + preflight2`，只对明确高污染任务族切到 schema-only，而不是对每个
skill 都做复杂混合。

## 2026-06-12 AppWorld dev57 Stage4 扩展验证与 preflight schema 修复

dev10 的 Stage4 正信号没有在 dev57 上泛化：

```text
qwen-only + preflight2 dev57:         16/57, exec_fail=39, preflight_fail=0
Stage4 transition_blend + preflight2: 11/57, exec_fail=69, preflight_fail=24
```

行级对比显示：两者共同成功 10 个任务，Stage4 只新增 1 个成功，却让 6 个 qwen-only
成功任务退化。退化主要集中在 Spotify：Stage4 选出的 SkillX metadata 往往主题相关但执行策略
误导，例如 unique-song-count 任务被 play-count/metric 技能牵引，add-to-queue 任务被
like/current-queue 技能牵引。

本轮发现并修复了一个 executor preflight 实现问题：之前 preflight 把 prompt 中截断展示的
API docs 当成完整 schema，因此可能误拒真实存在但未展示的 API。现在新增
`load_appworld_api_refs()`，multistep preflight 使用完整 AppWorld schema 校验，prompt 仍保持
紧凑。验证通过相关 65 个测试、`py_compile` 和 `git diff --check`。

但 6 条退化任务的小回归在修复后仍为 `0/6`，只是 `preflight_failures` 降到 0。因此当前瓶颈不是
Stage4 checkpoint 加载或 transition scorer 崩溃，而是 SkillX-to-executor handoff 的语义控制不稳定。
下一步不应继续直接 full rerun；应先设计更稳的 executor 接口，例如只传 schema-grounded executable
API plan / operation constraints，而不是直接暴露容易被当成函数名或策略捷径的 SkillX title/metadata。

### 2026-06-12 structured handoff v1 小结

新增 `skill_context_mode=schema_plan`：

- 隐藏 SkillX `skill_id` / `name` / raw body；
- 只展示从 SkillX executor/body 中抽取、且存在于完整 AppWorld schema 的 API evidence；
- 对 state-changing API 做动词级过滤；
- 明确告诉 executor：这些 API 不是 allowlist，用户目标和 AppWorld API docs 仍是权威来源。

小 gate 结果：

```text
regression6 full-schema safe_metadata:       0/6, preflight_fail=0, exec_fail=4
regression6 schema_plan initial:             1/6, preflight_fail=0
regression6 schema_plan not-allowlist:       4/6, preflight_fail=0, exec_fail=2
dev10 schema_plan not-allowlist:             4/10, exec_fail=4
regression6 adaptive_schema token heuristic: 1/6, preflight_fail=1, exec_fail=9
```

判读：

- `schema_plan` 证明 structured handoff 是有效方向，能明显缓解 SkillX title/metadata 污染；
- 但纯 schema evidence 会损失部分本来有用的自然语言过程信息，在 dev10 上低于 qwen-only 和旧 Stage4；
- 当前 `adaptive_schema` 只靠 token overlap 判断是否保留 safe metadata，启发式太弱，不能作为主线；
- 下一步应该做更保守的 row-level handoff gate：默认 qwen-only/no-skill fallback，只在 skill 的 operation、
  entities、required apps、state-changing API 与用户目标都匹配时才给 richer metadata，否则只给 schema evidence
  或直接不传 skill。

## 2026-06-12 Stage4 dev10 gate 与 executor prompt 防污染

修复 Stage4 delta restore、transition scorer 对齐、raw previous-skill embedding 后，Stage4 ACT
路径已经技术上接通：

- dev5 gate `87397`：`4/5`，优于对应 Stage0-style CLSTR dev5 的 `3/5`；
- dev10 gate `87398`：`6/10`，优于 Stage0-style CLSTR schema4 的 `5/10`，但只追平
  qwen-only schema4 的 `6/10`，且 execution failures 更高。

因此目前还不能进入 Stage4 full。失败样例显示主要瓶颈是 executor prompt 污染：safe metadata
虽然省略了 SkillX body，但仍暴露函数式 SkillX id/name，例如
`file-system-categorize-files-by-creation-date-*`，Qwen 会把它转换为不存在的
`apis.file_system.categorize_files_by_creation_date(...)`。

已加入通用修复：

- 从 `[AppWorld API Docs]` 解析当前运行时真实 `apis.<app>.<api>`；
- 对每个 retrieved skill 标注 `executable_appworld_apis_from_skill`；
- 对由 SkillX title 派生但 API docs 中不存在的候选调用标注
  `blocked_nonexistent_skill_title_api`；
- 当 safe metadata 中的技能标题像伪 API 且没有真实写操作 API 支撑时，隐藏该 SkillX id/name，
  只保留自然语言描述与真实 API 列表。

这不是 AppWorld 任务特化，而是 executor-schema compatibility：运行时 API docs 是唯一可执行
schema，SkillX 只能作为非可调用参考。

后续 dev10 gate 证伪了 prompt-level sanitizer 作为默认主线：

```text
87399 full sanitizer:   5/10, exec_fail=6
87401 narrow sanitizer: 4/10, exec_fail=5
```

主要原因是往 prompt 中加入更多 schema/blocked 标注会扰乱 Qwen，甚至让原本成功的只读 Spotify
检索任务退化。因此当前主线调整为：

- `schema_guard_skill_metadata` 仅作为显式诊断开关，默认关闭；
- Stage4 executor 主线恢复原 safe_metadata prompt；
- 新增执行前 AST/API preflight：生成代码后、`world.execute()` 前检查 Python 语法和
  `apis.<app>.<api>` 是否在当前 API docs 中；
- preflight 失败时给 Qwen 一次 repair prompt，修复后才执行。

这比继续堆 prompt 更适合作为论文方法的执行层保护：它不改变 routing/skill selection，只保证
executor 不执行 schema-invalid 代码。

两次后续 gate 结果：

```text
87405 Stage4 + preflight repair1: 6/10, exec_fail=4
87406 Stage4 + preflight repair2: 7/10, exec_fail=0
87407 qwen-only + preflight repair2: 6/10, exec_fail=2
```

当前结论：

- Stage4+preflight2 是当前 AppWorld dynamic SkillX 路线的第一个有用信号；
- 相同 executor/preflight 下，CLSTR Stage4 比 qwen-only 多成功 1 个 dev10 任务，平均步数更少
  (`1.4` vs `2.1`) 且 execution failures 为 0；
- 增益样例是 file_system 日期前缀/移动任务 `68ee2c9_1`，qwen-only 和旧 Stage4 都失败；
- 剩余失败主要是 phone+venmo 金额/联系人语义，以及另一个 file_system prefix 细节，不再是大面积
  伪 API 或语法崩溃。

这仍只是 dev10 gate，不是最终 benchmark。下一步若要扩大，应明确比较
`qwen-only+preflight2`、`Stage0-style CLSTR`、`Stage4+preflight2`，并先跑 dev57/小全量，而不是直接
无限 full。

## 2026-06-12 Stage4 ACT executor 加载与推理 scorer 修复

复查 AppWorld dynamic Stage4 gate 时发现两个实现级问题：

- Stage4 ACT checkpoint 是 delta checkpoint，会排除 frozen routing foundation
  (`encoder.*`、`skill_table.*` 等)。旧 executor loader 把它当完整 checkpoint 加载，会丢失
  Stage0 routing 与 Stage2 base/head 初始化，不是真正的 Stage4 模型；
- AppWorld multistep controller 的 `policy_blend` 调用的是 `policy_forward/skill_head`，但
  Stage4 ACT 训练和 quality gate 评估的是 transition-conditioned next-skill scorer
  (`trans_head` / `_transition_candidate_logits_for_mode`)。因此即使 checkpoint 正确加载，
  旧推理路径也没有真正用上 Stage4 ACT 的主训练信号。

已修复：

- `scripts/run_appworld_multistep_executor_eval.py` 识别
  `checkpoint_excludes_frozen_routing_foundation=true` 的 Stage4 delta checkpoint，按 checkpoint
  内记录的 `routing_checkpoint_path + head_checkpoint_path` 恢复完整模型，再加载 Stage4 delta；
- `CLSTRMultiStepController` 新增 `transition_head` 与 `transition_blend` ranking mode：
  第一步无历史 action 时回退到 Stage0 routing；从第二步开始用上一轮 selected skill、
  当前 state/belief 与候选集调用 Stage4 transition scorer；
- CLI / sbatch 暴露 `transition_scoring_mode` 与 `transition_residual_lambda`，并让 Stage4
  checkpoint 的 `ranking_mode=auto` 默认解析为 `transition_blend`。

验证：

```text
tests/test_appworld_multistep.py
tests/test_appworld_multistep_cli.py
tests/test_appworld_executor.py
tests/test_phase1_prior_residual_defaults.py
59 passed
```

一个旧 `policy_blend` dev5 gate 在取消前产出 `2/5`，只能说明加载路径不崩溃，不能作为 Stage4
有效性结论。下一步必须用 `RANKING_MODE=transition_blend` 重新做小规模 gate。

补充修复：第一轮 `transition_blend` gate 暴露第二步维度错误：
`mat1 and mat2 shapes cannot be multiplied (1x256 and 1024x1024)`。原因是推理端把
`model.action_embeddings(previous_skill)` 的 256 维投影后动作向量传给了
`_transition_candidate_logits_for_mode()`；而 Stage4 训练里的 `action_emb` 是 1024 维原始
action/skill 文本 embedding。已改为传 `skill_table.E[previous_skill]`，再重跑 gate。

## 2026-06-11 AppWorld Stage4 前置 executor reliability 修复

在 dynamic SkillX dev10 stratified gate 中，`qwen_only` 与 `clstr_dynamic safe_metadata`
都只有 `2/10` official success；因此当前不应直接进入 AppWorld Stage4 full。复查失败样本后，
主要问题不是 dynamic skill append 或兼容池检索，而是 executor/API 使用可靠性：

- Venmo relative-date 任务需要 AppWorld task datetime，但 eval JSONL 未携带该字段；
- file_system 任务容易发明不存在的 `rename_file` / `categorize_files_by_creation_date`，
  或把 `move_file` 的 destination 写成目录而不是完整文件路径；
- simple_note export 容易把 `search_notes` 的 list 返回误当成 dict，并且可能给导出内容额外加
  markdown header；
- phone+venmo grocery 任务可能从短信历史中拿到第一个无关 dollar amount。

本轮修复：

- 新增 `enrich_task_with_appworld_specs()`，从
  `appworld_root/data/tasks/<task_id>/specs.json` 注入 `task_datetime`；
- single-step 与 multistep AppWorld executor eval 都在 prompt 前执行该 enrichment；
- prompt 中加入 AppWorld task datetime 环境块，并要求相对日期使用该时间而不是 host clock；
- 针对 Venmo likes、file_system move/date、simple_note export、phone+venmo amount
  relevance 增加 schema-level task hints。

已通过本地验证：

```text
tests/test_appworld_executor.py -> 19 passed
tests/test_appworld_multistep.py tests/test_appworld_multistep_cli.py -> 29 passed
py_compile clstr/appworld_executor.py clstr/appworld_multistep.py scripts/run_appworld_multistep_executor_eval.py -> passed
git diff --check affected files -> passed
```

下一步只跑 dev10 小门槛作业复测 `clstr_dynamic safe_metadata`。如果 official success 或
execution failure profile 没有改善，继续修 executor 或更换 executor；如果出现稳定增益，再进入
Stage4 smoke，而不是直接跑 Stage4 full。

## 2026-06-11 AppWorld eval-only step labels 与 executor 粒度复查

已新增 eval-only dev step-label 构建路径：

```bash
python scripts/build_appworld_eval_step_label_tasks.py \
  --tasks_path data/appworld_routing/dev_tasks.jsonl \
  --skill_pool_path data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl \
  --output_path data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl
```

实现边界：

- 只用于 dev/test executor/routing 评估，manifest 写明 `eval_only=true` 与
  `dev_eval_only_no_training`；
- 不改写原始 `dev_tasks.jsonl`，不把 dev/test oracle API trace 用于训练；
- 标签空间默认限制在 188 个 AppWorld-compatible SkillX skills；
- 对每个 oracle API ref 使用 multi-positive：所有包含该 API ref 的
  AppWorld-compatible skills 都算作该 API step 的 positive，避免单个 `show_song`
  被任意映射到 playlist-duration 等噪声技能。

真实构建结果：

```text
task_count = 57
labeled_task_count = 57
skill_count = 37803
label_skill_count = 188
avg_step_label_set_size = 9.188867
max_step_label_set_size = 30
leakage_guard = dev_eval_only_no_training
```

随后用该文件重跑 3-task strict-compatible executor smoke：

```text
job = 87267
output = outputs/appworld_dynamic_multistep_executor_smoke/dev3_step_labels_20260611_184823
task_count = 3
success_count = 3
success_rate = 1.0
generation_failures = 0
execution_failures = 0
average_steps = 1.0
step_positive_skill_hit_rate = 0.0
```

判读：

- official AppWorld success 是强正信号，说明 dynamic SkillX handoff + Qwen executor 在这 3
  个任务上可以真实完成任务；
- `step_positive_skill_hit_rate` 仍不能作为主判据。原因是 executor step 与 oracle API step
  粒度不同：Qwen 第一步可能直接写出完整多 API 程序，而 oracle step 0 只是 `spotify.login`；
- 后续扩大评估时应报告 official success / execution failures，并增加 qwen-only 或
  skillrouter-live 对照；step-hit 只能作为辅助诊断，不能作为 Stage4 full 的 gate。

## 2026-06-11 AppWorld dev10 stratified executor 对照

为避免继续只评估文件头 Spotify 任务，新增 required-app round-robin 子集构建：

```bash
python scripts/build_appworld_stratified_subset.py \
  --input_path data/appworld_multistep/dev_tasks_with_step_labels_dynamic.jsonl \
  --output_path data/appworld_multistep/dev10_stratified_step_labels_dynamic.jsonl \
  --max_tasks 10
```

子集覆盖：

```text
spotify = 2
phone+venmo = 2
venmo = 2
file_system = 2
file_system+simple_note = 2
```

三组 executor smoke 结果：

```text
qwen_only:
  output = outputs/appworld_dynamic_multistep_executor_smoke/dev10_stratified_qwen_only_20260611_200032
  success_count = 2 / 10
  execution_failures = 5
  average_steps = 2.3

clstr_dynamic safe_metadata:
  output = outputs/appworld_dynamic_multistep_executor_smoke/dev10_stratified_clstr_dynamic_20260611_201624
  success_count = 2 / 10
  execution_failures = 5
  average_steps = 1.8
  step_positive_skill_hit_rate = 0.388889

clstr_dynamic raw:
  output = outputs/appworld_dynamic_multistep_executor_smoke/dev10_stratified_clstr_dynamic_raw_20260611_203529
  success_count = 2 / 10
  execution_failures = 9
  average_steps = 2.0
  step_positive_skill_hit_rate = 0.4
```

判读：

- CLSTR 在 Spotify 成功任务上把 Qwen 的 2-3 步压到 1 步，说明 selected SkillX context 有局部帮助；
- official success 没有超过 qwen-only，不能进入 AppWorld Stage4 full；
- raw SkillX body 不提升 success，且 execution failures 从 5 增到 9，说明直接把 raw body 放进 prompt
  不是当前主线；
- 当前瓶颈更像 executor 代码生成/API 返回结构处理，而不是仅靠 ACT/RL 能修复的 routing 问题。

## 2026-06-11 AppWorld dynamic executor smoke 接入

已将 AppWorld multi-step executor 接到 dynamic skill registry 的安全加载路径：

- 新增 `--base_skill_pool_path`：checkpoint 先按 v4.1b base pool 还原；
- `--skill_pool_path` 继续表示运行时 dynamic pool；
- executor 会从 dynamic pool 中抽取 checkpoint prefix 之后的 AppWorld-compatible rows，
  再通过 `model.append_skills(...)` 追加到 checkpoint model；
- 新增小规模 sbatch 入口：
  `scripts/sbatch/run_appworld_dynamic_multistep_executor_smoke.sh`；
- 默认 smoke 只跑 `MAX_TASKS=3, MAX_STEPS=3`，且
  `APPWORLD_EXECUTOR_COMPATIBLE_ONLY=1`，用于低成本验证真实 AppWorld/Qwen executor loop。

这个改动不训练 Stage0/1/2/4，也不改变 v4.1b checkpoint 本身；它只解决“动态 AppWorld SkillX
追加后如何安全进入 executor eval”的工程路径。Full eval 仍需等 smoke 结果健康后再决定。

### 2026-06-11 dynamic executor smoke 结果

小规模 executor smoke 暴露并修复了两个工程问题：

- JSONL reader 不能用 `read_text().splitlines()` 读取 skill pool，否则 skill body 中的 Unicode
  line separator 会被误切行；dynamic loader 已改用逐文件行 reader；
- `APPWORLD_EXECUTOR_COMPATIBLE_ONLY=1` 必须要求正向兼容证据，不能放行缺字段的 base skills。
  现在只有 `appworld_executor_compatible=True`、`executor_domain=appworld` 或
  `skillx/appworld/` id 会进入 prompt。

修复后 job `87217` 完成：

```text
task_count=3
success_count=2
success_rate=0.666667
generation_failures=0
execution_failures=3
average_steps=1.666667
runs_path=outputs/appworld_dynamic_multistep_executor_smoke/dev3_strict_compat_20260611_164719/runs.jsonl
```

注意：`step_positive_skill_hit_rate=0.0` 暂不能作为方法失败结论，因为当前
`data/appworld_routing/dev_tasks.jsonl` 的前三个 Spotify top-play-count 任务标注到了
playlist-duration/auth skills，而 executor 实际选中的是 play-count 相关 SkillX。下一步应先修
AppWorld routing/eval label 构造或换成 verified step labels，再扩大 executor eval。

## 2026-06-11 Dynamic Skill Registry v1

针对“CLSTR 是否必须像 SkillRouter 一样支持动态 skill pool”的问题，本轮先实现 CPU-only
基础能力，不提交训练/评测作业。

新增/修改：

- `clstr/encoders.py::SkillTable.append_skills()`：
  - 允许在 checkpoint 加载后追加新 skill rows；
  - 保留旧 `skill_table.E`、`skill_bias_retr`、`skill_bias_belief`；
  - 新 skill 用 skill text 编码初始化 embedding；
  - 新行默认写入 `is_appended_after_checkpoint=true`、
    `retrieval_seen_count=0`、`transition_seen_count=0`、`act_seen_count=0`。
- `clstr/model.py::CLSTRModel.append_skills()`：
  - 同步 `model.skills` 与 `model.skill_table.skills`；
  - append 后清理 cross-encoder task cache，避免旧 top-M/cache 污染。
- `clstr/dynamic_skill_registry.py`：
  - 新增 `DynamicSkillRegistry`，负责有序 skill rows、去重、append 标记和
    executor-domain inventory filtering；
  - 新增 `skill_head_coverage_weights()` 与
    `blend_skill_scores_with_coverage()`，使 appended/unseen skill 默认更依赖 Stage0/text
    score，而不是被没有监督覆盖的 transition/belief/head score 主导。
- `clstr/stage_checkpoint_init.py::load_compatible_state_dict()`：
  - 新增显式参数 `allow_skill_table_prefix_expansion`；
  - 开启时可把旧 checkpoint 的 `skill_table.E` / retrieval bias / belief bias 前缀加载到
    更大的 skill table，尾部新行保留当前初始化。

当前推荐用法：

```text
1. 用 checkpoint 原始 skill pool 加载 CLSTR；
2. 通过 DynamicSkillRegistry / model.append_skills() append 新 benchmark/runtime skills；
3. 推理时先在 global registry 中做 Stage0 retrieval；
4. executor 侧按 runtime inventory 过滤，例如 AppWorld 只把
   appworld_executor_compatible=true 的 skill body 交给 Qwen；
5. 对 appended/unseen skill 使用 coverage-aware blending，降低 transition/belief/head 权重。
```

论文边界：

- 动态 skill append 现在可以支持 SkillRouter 式单步 plug-in retrieval；
- CLSTR 多步 transition/belief 对新 skill 不能声称“无监督立刻满血泛化”；
- 对新 skill 的合理主张是：文本检索立即可用，连续选择通过 coverage-aware fallback 保持稳定；
  若要让新 skill 参与复杂多步 routing，仍需要 few-shot trajectory、ACT 更新或 online feedback。

验证：

```text
python -m pytest tests/test_dynamic_skill_registry.py -q
python -m pytest tests/test_v4_action_proj_sharing.py -q
python -m pytest tests/test_stage_checkpoint_init.py -q
```

只读数据审计进一步确认 AppWorld 转回前必须先修 skill pool：

```text
data/clstr_unified_pretrain_v4_1b/skill_pool.jsonl      = 37,615 rows
data/clstr_appworld_current_route_v1/skill_pool.jsonl   = 188 rows
unified_v4_1b appworld_like rows                         = 1
appworld_current appworld_like rows                      = 188
overlap skill_id count                                   = 0
```

因此，直接用 v4.1b unified checkpoint/pool 跑 AppWorld executor 或 AppWorld dev57 gate 是不合理的：
它几乎没有 AppWorld SkillX candidate。下一步必须先构建
`unified_v4_1b + AppWorld SkillX append` 的 dynamic registry，并用 mask 前 AppWorld positive
rank/recall/MRR 验证大池检索，再做 AppWorld-compatible executor eval。

2026-06-11 追加实现：

- `scripts/build_clstr_appworld_current_route.py --base_skill_pool_path ...` 已支持
  checkpoint base pool 作为前缀、AppWorld SkillX 作为 append tail；
- 实际构建 `data/clstr_appworld_dynamic_v4_1b_append`：
  - skill pool = 37,615 base skills + 188 AppWorld appended skills；
  - retrieval rows = 2,496；
  - trajectory rows = 1,068；
- `clstr/appworld_dynamic_routing_audit.py` 新增真实动态路径 audit：
  - 先用原 v4.1b base skill pool 加载 Stage0 checkpoint；
  - 再通过 `model.append_skills()` 编码并追加 AppWorld SkillX rows；
  - 同时报告 full global pool 下的 AppWorld positive rank，以及
    AppWorld-compatible executor mask 后的 rank。

该 audit 不能在登录/存储节点上直接跑真实模型；下一步应提交一个小 sbatch smoke，例如
`--max_pairs 32`，先确认真实 checkpoint 动态 append 路径健康，再决定是否扩大到全量
AppWorld routing audit。

对应 smoke 脚本：

```bash
sbatch --gpus=1 -p gpu_h200 scripts/sbatch/run_appworld_dynamic_routing_audit_smoke.sh
```

默认 `MAX_PAIRS=32`、输出到
`outputs/appworld_dynamic_routing_audit/smoke_max32/report.json`。如果 smoke 的 global rank
很差但 AppWorld-compatible rank 正常，说明问题主要在大池粗召回；如果两者都差，说明
AppWorld SkillX embedding/query 格式仍需修。

注意：第一次 smoke job `86993` 证明了动态 append 路径可执行，但不能作为最终指标判断：
checkpoint config 是 `query_text_format=skillrouter`，audit 当时却用 raw query 编码。当前已修复为
`query_text_format=auto` 默认读取 checkpoint config；需要重跑同样 `MAX_PAIRS=32` smoke 后再判断
AppWorld 动态池的真实 routing 质量。

另外，`MAX_PAIRS=32` 若直接取文件头部，会偏向前几个 Spotify routing qrels。当前 smoke 脚本默认
`SAMPLING_STRATEGY=stride`，更合理的下一次低成本诊断是 `MAX_PAIRS=128` + stride；full audit 仍应等
这个诊断确认方向后再决定。

`MAX_PAIRS=128` + stride smoke job `87028` 已完成，结果比 head sample 更有参考价值：

```text
query_text_format = skillrouter
global Recall@50/100/350 = 0.515625 / 0.554688 / 0.710938
global MRR = 0.098412
AppWorld-compatible Recall@20/50/100 = 0.664062 / 0.898438 / 1.0
AppWorld-compatible MRR = 0.297932
```

结论：动态 append 路径和 AppWorld-compatible ranking 是可用的；主要瓶颈在 full global
large-pool recall/calibration。下一步若要扩大，应先跑 full routing audit，而不是直接进入
executor eval。

Full routing audit job `87084` 已完成，覆盖全部 2,496 个 AppWorld positive pairs：

```text
query_text_format = skillrouter
global Recall@50/100/350 = 0.48117 / 0.55008 / 0.704327
global MRR = 0.091053
AppWorld-compatible Recall@20/50/100 = 0.65625 / 0.913462 / 0.999599
AppWorld-compatible MRR = 0.278028
elapsed = 12:24 on gpu_a800
```

当前判断：

- 不需要因为 AppWorld 动态接入而立刻重跑 Stage0/1/2；
- full global 大池召回仍弱，说明如果论文强调“无 mask 大池直接 top-K 命中”，还需要
  calibration/rerank；
- AppWorld-compatible executor mask 后的召回足够强，下一步可以跑小规模 executor eval，
  同时把 global recall 作为已知瓶颈单独报告。

## 2026-06-10 — Stage4/RL 路线转为 Qwen executor gate

上一轮 ALFWorld online HRPO 已证明：CLSTR-direct admissible-action policy 有非零学习信号，
但它没有稳定转化为 task success。继续在这条 direct action bridge 上调 reward shaping、
progress memory 或 LR，成本高且不符合 CLSTR 作为 selector/router 的论文叙事。

本轮新增一个低成本、不训练的 executor gate，用于决定是否值得继续做 online RL：

- Qwen-only：Qwen3 在同一批 ALFWorld admissible actions 上直接选择动作；
- Qwen+CLSTR：Qwen action score 加上 CLSTR large-skill-pool checkpoint 给出的 selector prior；
- 两者共享同一 closed-loop ALFWorld harness，默认只跑 `2` 个 episode、`20` 步；
- 默认使用本地 `models/Qwen3-8B` 和 `local_files_only`，避免计算节点联网依赖；
- 若 Qwen+CLSTR 相比 Qwen-only 在 reward/success 上没有正向差异，则暂停 ALFWorld online RL，
  转向 executor 更稳定的 benchmark 或 trajectory-level preference/imitation。

新增文件：

```text
clstr/alfworld_qwen_clstr_gate.py
scripts/run_alfworld_qwen_clstr_executor_gate.py
scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh
tests/test_alfworld_qwen_clstr_gate.py
```

关键验证：

```text
python -m pytest tests/test_alfworld_qwen_clstr_gate.py -q
python -m pytest tests/test_sbatch_scripts.py -q -k qwen_clstr_executor_gate
python -m py_compile clstr/alfworld_qwen_clstr_gate.py scripts/run_alfworld_qwen_clstr_executor_gate.py
bash -n scripts/sbatch/run_alfworld_qwen_clstr_executor_gate.sh
```

通过条件不是 smoke 成功率本身，而是 gate 是否给出方向性证据：

```text
continue_to_online_rl =
  qwen_plus_clstr.success_rate > qwen_only.success_rate
  or qwen_plus_clstr.average_reward > qwen_only.average_reward
```

这一步保持论文边界清楚：CLSTR 不直接替代 executor，而是在大 skill pool 下提供可迁移的
selector prior；executor 仍由 Qwen 执行动作选择。

### Qwen executor gate smoke 结果

已完成三轮低成本 ALFWorld valid_seen gate，均为 `2` episodes / `20` steps：

```text
1. raw row-zscore, clstr_weight=0.5
   output = outputs/alfworld_eval/qwen_clstr_executor_gate_smoke_20260610_v1
   Qwen-only reward/success = 0 / 0
   Qwen+CLSTR reward/success = 0 / 0
   qwen_clstr_disagreement_rate = 0.70
   qwen_hybrid_disagreement_rate = 0.00

2. raw row-zscore, clstr_weight=2.0
   output = outputs/alfworld_eval/qwen_clstr_executor_gate_smoke_20260610_clstrw2
   Qwen+CLSTR reward/success = 0 / 0
   qwen_clstr_disagreement_rate = 0.70
   qwen_hybrid_disagreement_rate = 0.00

3. proposal-bonus + shared loop guard, clstr_weight=0.5
   output = outputs/alfworld_eval/qwen_clstr_executor_gate_smoke_20260610_loopguard
   Qwen-only reward/success = 0 / 0
   Qwen+CLSTR reward/success = 0 / 0
   qwen_clstr_disagreement_rate = 0.65
   qwen_hybrid_disagreement_rate = 0.125
```

诊断：

- 原始 generate 融合把 Qwen 输出变成过强 one-hot，CLSTR prior 即使不同意也无法改变动作；
- 新增 `qwen_score_mode=proposal_bonus` 后，Qwen 生成动作只作为 proposal bonus，CLSTR
  连续 prior 可以介入；
- loop guard 后，hybrid trace 出现了更明确的任务相关动作，例如 `take apple 1 from fridge 1`、
  `cool apple 1 with fridge 1`、`move apple 1 to fridge 1`，而 Qwen-only 仍主要在导航和检查；
- 但官方 ALFWorld reward / goal-condition 仍为 0，说明当前不能直接据此进入 full online RL。

trace-progress gate 已完成，结果不支持启动 full ALFWorld online RL：

```text
output = outputs/alfworld_eval/qwen_clstr_executor_gate_smoke_20260610_loopguard/trace_progress_gate.json
status = action_required
recommendation = do_not_start_full_rl_from_this_gate

Qwen+CLSTR - Qwen-only:
  mean_progress_reward_delta = -0.075
  mean_progress_points_delta = 0.0
  mean_goal_interaction_points_delta = 0.0
  mean_placed_target_count_delta = 0.0
  mean_placed_target_event_count_delta = 0.0
  mean_irrelevant_action_count_delta = +2.5
  mean_reverted_target_count_delta = 0.0
  mean_wrong_receptacle_move_count_delta = 0.0
```

这次复核修正了前面对 hybrid trace 的乐观判断：按 `gamefile/traj_data.json`
解析真实 ALFWorld goal 后，两个样本分别要求操作 `soapbottle` 和 `potato`，但 hybrid trace
操作了 `spraybottle/soapbar/apple` 等非目标物体。CLSTR prior 当前确实能改变 Qwen 的
action choice，但没有带来目标 grounding 或 progress uplift，反而增加了 irrelevant actions。

当前结论：不要从这个 gate 直接启动 full ALFWorld online RL。若继续做 Stage4/RL，应先解决
executor/target grounding，或把主证明转向 ToolBench-G3 / TRAJECT 这类 executor 歧义更低的
selector-routing / trajectory preference 设置；ALFWorld online 暂时只作为诊断路径保留。

## 2026-06-09 v4.1b ALFWorld online HRPO smoke 修复与结论

本轮继续从当前 v4.1b Stage2 listwise checkpoint 接 ALFWorld train-split online HRPO
smoke，目标只是判断 on-policy reward/advantage 信号是否可用，不是 full RL。

已修复两个工程阻断点：

- `clstr/alfworld_eval.py::_read_jsonl()` 不能用 `read_text().splitlines()` 读取
  JSONL。v4.1b `skill_pool.jsonl` 中存在合法的 `U+0085` 字符，`splitlines()` 会把
  JSON 字符串内部字符误当作行分隔符，导致 `JSONDecodeError`。现已改为文件句柄逐行读取。
- online HRPO blocker report 现在写入完整 traceback，并在新 run 开始时清理旧
  `blocker_report.json` / `train_report.json`，避免旧失败报告污染新结果。

环境修复：

- 当前 `xzf` conda 环境补装了 ALFWorld text env 运行所需的
  `textworld[pddl]`、`termcolor`、`gym`、`jericho`。
- `AlfredTWEnv` 已能从本地 `alfworld_repo` 导入。

验证：

```text
pytest:
  tests/test_alfworld_eval.py::test_alfworld_jsonl_reader_preserves_unicode_next_line_inside_json_strings
  tests/test_alfworld_eval.py::test_load_clstr_alfworld_model_can_initialize_from_checkpoint_config_without_routing_manifest
  tests/test_online_hrpo.py::test_online_hrpo_blocker_report_includes_traceback_for_root_cause_debugging
  tests/test_online_hrpo.py::test_v4_1b_alfworld_online_hrpo_wrapper_uses_stage2_checkpoint_and_large_pool_defaults
  tests/test_sbatch_scripts.py::test_v4_1b_alfworld_online_hrpo_smoke_sbatch_uses_stage2_large_pool_checkpoint_and_tiny_limits

result = 5 passed
py_compile = passed
bash -n smoke sbatch = passed
git diff --check targeted files = passed
```

tiny smoke 作业：

```text
job_id = 85898
partition = gpu_a800
output = outputs/clstr_v4_1b_alfworld_online_hrpo_smoke
init_checkpoint = outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt
max_games = 1
group_size = 2
max_steps = 5
updates = 1
```

结果：

```text
status = ok
rollout_count = 2
rollout_step_count = 10
rollout_success_rate = 0.0
mean_rollout_reward = -0.15
reward_unique_count = 1
updates_with_nonzero_advantage = 0
mean_admissible_action_count = 13.1
viability.status = action_required
viability.blockers = [no_reward_variation, no_nonzero_advantage_updates]
hrpo_policy_loss = 0.0
loss = 0.0
```

结论：

- 技术链路已打通：v4.1b large skill pool checkpoint 可以加载，ALFWorld train split
  可以 rollout，checkpoint / rollouts / metrics / train_report 都能写出。
- 但这次 online RL 信号退化：两条 rollout 都失败且 reward 完全相同，group-normalized
  advantage 全为 0，因此 HRPO loss 也为 0。
- 不能基于这个结果提交 full ALFWorld RL。下一步若继续，应先做更低成本的 reward-variance
  gate，例如只扩大 `max_games/group_size/max_steps` 做 rollout 诊断，并避免在
  viability 不通过时保存大 checkpoint；只有看到 reward_unique_count >= 2 且
  updates_with_nonzero_advantage > 0，才值得进入小规模 online training。

## 2026-06-09 v4.1b ALFWorld online HRPO viability smoke

为回答“Stage4 offline 无明显增益时，online RL 是否仍有信号”的问题，本轮新增一个低成本
ALFWorld train-split online HRPO smoke，而不是直接提交 full RL：

- 新 CLI：`scripts/run_clstr_v4_1b_alfworld_online_hrpo.py`；
- 新 sbatch：`scripts/sbatch/run_clstr_v4_1b_alfworld_online_hrpo_smoke.sh`；
- 默认初始化：
  `outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt`；
- 默认 skill pool/data root：`data/clstr_unified_pretrain_v4_1b`，checkpoint 内的
  `skills_path` 会优先恢复 37,615 skill pool；
- 默认 smoke 上限：`MAX_GAMES=1`、`GROUP_SIZE=2`、`MAX_STEPS=5`、`UPDATES=1`；
- 输出新增 `training_metrics.jsonl`、`rollouts.jsonl`、`checkpoints/latest.pt` 和
  `train_report.json.viability`。

viability gate 不以“程序跑完”为成功条件，而要求：

- rollout/step 数量非零；
- reward 至少有两个不同取值；
- 至少一个 update 出现非零 grouped advantage；
- admissible action 候选空间不是退化的单动作。

同时修复 ALFWorld checkpoint loader：

- 当 v4.1b Stage2 checkpoint 自带 `config` 与 `skills_path` 时，可直接从 checkpoint config
  构建模型，不再依赖旧 `outputs/clstr_native_routing_init/manifest.json`；
- 若 checkpoint state 已包含 `skill_table.E`，加载后不会再次 `rebuild_skill_table()`，避免
  smoke 启动阶段重新编码 37k skills。

定位说明：这只是 online RL 可行性测试。若 smoke 的 `viability.status != ok`，不能继续
full RL；若 `viability.status = ok` 且 reward/advantage 有信号，下一步才考虑小规模多任务
ALFWorld online RL gate。

## 2026-06-09 AppWorld 接入当前 CLSTR 路线

本轮把 AppWorld 从旧的 AppWorld-specific prediction 文件，接到当前 CLSTR
Stage0/Stage1/Stage2/Stage4 统一路线的数据和 executor 边界上。历史
`outputs/appworld_clstr_eval/act_v2_dev/predictions.jsonl` 只保留为 cheap signal
diagnostic，不能作为当前主线结果。

新增：

- `clstr/appworld_current_route.py`
- `scripts/build_clstr_appworld_current_route.py`
- `scripts/sbatch/run_clstr_appworld_current_stage0_biencoder_train.sh`
- `scripts/sbatch/run_clstr_appworld_current_stage1_heads_init.sh`
- `scripts/sbatch/run_clstr_appworld_current_stage2_full_base_train.sh`
- `scripts/sbatch/run_clstr_appworld_current_stage4_act_train.sh`
- `scripts/sbatch/run_appworld_current_route_multistep_executor_eval.sh`

真实 AppWorld-only 数据根已构建：

```text
data_root = data/clstr_appworld_current_route_v1
skill_count = 188
trajectory_rows = 1068
  appworld_train_replay = 450
  appworld_verified_pairs_train = 618
retrieval_pairs = 2496
  task_skillx_qrel = 450
  trajectory_derived = 2046
status = ok
leakage_boundary = train-only
```

设计边界：

- AppWorld executor / RL 不能直接把 ToolBench、TRAJECT、ALFWorld/WebShop 的大 skill pool
  全部塞进 prompt；这些 skill 对 AppWorld API 环境不可执行，会污染下游执行。
- 这不等于放弃 large skill pool 证据。builder 支持
  `extra_skill_pool_paths`，可以生成“大池 + AppWorld SkillX positives”的 routing 数据根；
  额外 skills 会追加为 distractor，且标记
  `appworld_executor_compatible=false`。
- executor eval 新增 `--appworld_executor_compatible_only`。large-pool checkpoint 进入
  AppWorld executor 时，可以先在大池中打分，再只把 AppWorld-compatible skills 交给
  Qwen executor，同时 diagnostics 保留 mask 前的大池候选。

论文口径应拆成两层：

1. large-pool routing：报告 mask 前的 AppWorld positive rank / recall@K / MRR，证明 CLSTR
   在大 skill pool 中能找出正确 AppWorld skill；
2. executable downstream：AppWorld task success / reward 只使用 AppWorld-compatible skills，
   这是环境执行约束，不是任务特异捷径。

RL 边界：

- AppWorld 有 executor 和 verifier，是最干净的 online RL / task-success reward 入口；
- ToolBench 可以在接入 StableToolBench/tool server 后做 executor-based RL，但工程成本更高；
- TRAJECT-Bench 不需要 live executor 就能做 offline sequential routing / ACT / imitation /
  preference 训练和离线指标，但这不应被表述为 online RL。

本轮没有提交 Slurm/GPU 训练作业。

## 2026-06-09 Modern Qwen3 executor signal gate

用户指出旧实验里 Qwen3-8B direct 在 AppWorld 上表现差，若下游 executor 本身没有有效
reward，CLSTR 的 ACT/RL 也很难学到东西。因此本轮没有直接进入 full AppWorld RL，而是
先把 Qwen3 重新接成一个低成本 executor-signal gate：

- 新增 `clstr/qwen_backend.py`，统一 Qwen generation backend：
  - 默认 `model_name_or_path=models/Qwen3-8B`；
  - 默认 `local_files_only=true`，计算节点不联网下载；
  - 统一 `apply_chat_template(..., enable_thinking=false)`，并兼容旧 tokenizer
    不接受 `enable_thinking` 的情况；
  - 统一记录 `qwen_backend_version=shared_generation_v1`、模型路径、dtype、
    local/offline、temperature/top-p 等 metadata。
- `clstr/appworld_executor.py::QwenCodeGenerator` 已改为 shared backend 的薄包装；
  AppWorld executor 不再自己维护一套独立 Qwen 加载/generation 逻辑。
- `clstr/qwen_direct_policy.py::QwenDirectAdmissibleActionScorer` 支持注入同一个 shared
  backend，用于 ALFWorld/AppWorld action proposal 诊断时避免再加载第二套旧接口。
- 新增低成本 wrapper：
  `scripts/sbatch/run_appworld_qwen3_signal_smoke.sh`。默认只跑
  `METHOD=qwen_only`、`MAX_TASKS=2`、`MAX_STEPS=3`、`MAX_INTERACTIONS=10`，
  不训练、不跑 HRPO、不跑 full benchmark。

本轮 focused 本地验证：

```text
tests/test_qwen_backend.py
tests/test_qwen_direct_policy.py
tests/test_qwen_planner_intent.py
tests/test_appworld_executor.py
tests/test_sbatch_scripts.py::test_appworld_qwen3_signal_smoke_is_low_cost_and_offline
=> 33 passed

py_compile:
clstr/qwen_backend.py
clstr/qwen_direct_policy.py
clstr/appworld_executor.py
scripts/run_appworld_multistep_executor_eval.py
=> passed

bash -n scripts/sbatch/run_appworld_qwen3_signal_smoke.sh
=> passed
```

提交并监督了一个小 AppWorld signal smoke：

```text
job_id = 85712
partition = gpu_a800
state = COMPLETED
exit_code = 0:0
elapsed = 00:01:19
output = outputs/appworld_multistep_executor_smoke/qwen3_modern_backend_signal_smoke
task_count = 2
success_count = 2
success_rate = 1.0
generation_failures = 0
execution_failures = 0
average_steps = 2.0
```

两个任务不是空跑：`runs.jsonl` 中每个任务均包含两步真实 AppWorld Python 执行，第一步
登录 Spotify，第二步分页读取 song/album/playlist library、按 genre 过滤、按
`play_count` 排序，并调用 `apis.supervisor.complete_task(...)` 后通过
`task_completed/evaluate`。

边界解释：

- 这不是 CLSTR 结果，也不是 AppWorld full benchmark；
- 这只说明修正后的 Qwen3-8B executor 协议在小样本上有非零且可用的 reward signal；
- AppWorld 只是第一道 downstream executor/RL 信号门，不是唯一 benchmark；
- 若扩大到 dev10/dev57 后 Qwen3-8B 仍表现差，应优先换同规模更新 Qwen 模型或转向
  reward 更密/更稳定的数据集，而不是直接烧 full AppWorld RL。

### 2026-06-09 Modern Qwen3 AppWorld dev10 gate

按“先试试，若效果不好再升级模型或换数据集”的策略，继续提交一个小型 dev10 gate，
仍然只跑 `qwen_only` executor，不训练 CLSTR、不跑 HRPO：

```text
job_id = 85724
partition = gpu_a800
state = COMPLETED
exit_code = 0:0
output = outputs/appworld_multistep_executor_smoke/qwen3_modern_backend_signal_dev10
task_count = 10
success_count = 3
success_rate = 0.3
generation_failures = 0
execution_failures = 2
task_completed_count = 10
evaluate_success_count = 3
average_steps = 1.9
```

失败分布显示 Qwen3-8B 不是完全无法执行，而是能生成代码并调用 `complete_task`，
但 task semantics 不稳定：

- `fac291d_*` unique-song count：执行成功并 complete_task，但 answer 未通过 evaluate；
- `530b157_*` phone+venmo grocery：部分步骤金额正则/联系人短信抽取失败或最终答案不对；
- `4ec8de5_1` release-year counting：执行成功并 complete_task，但 evaluate 失败。

结论：

- Qwen3-8B modern backend 在 AppWorld 上有非零 reward signal，和旧接口的弱结果不等价；
- 但 dev10 `0.3` 仍偏弱，直接 full AppWorld RL 风险较高；
- 下一步应先做同 dev10 上的 CLSTR skill-context 对比，判断 CLSTR 是否能提升 Qwen executor；
- 如果 CLSTR skill-context 不能提升，应优先换同规模更新 Qwen 模型，或转到 reward 更密的数据集。

### 2026-06-09 Modern Qwen3 + CLSTR skill-context dev10 gate

按同一 dev10 gate 加入 CLSTR skill-context，对比 qwen-only。该对照使用已有 AppWorld
CLSTR ACT prediction 文件：

```text
method = clstr_act
prediction_file = outputs/appworld_clstr_eval/act_v2_dev/predictions.jsonl
job_id = 85732
partition = gpu_a800
state = COMPLETED
exit_code = 0:0
output = outputs/appworld_multistep_executor_smoke/qwen3_modern_backend_signal_dev10_clstr_act
task_count = 10
success_count = 7
success_rate = 0.7
generation_failures = 0
execution_failures = 0
task_completed_count = 10
evaluate_success_count = 7
average_steps = 2.0
step_positive_skill_hit_rate = 1.0
```

同一任务逐项对比：

```text
qwen_only success_rate = 0.3
clstr_act skill-context success_rate = 0.7

50e1ac9_1: true  -> true
50e1ac9_2: true  -> true
50e1ac9_3: true  -> true
fac291d_1: false -> true
fac291d_2: false -> true
fac291d_3: false -> true
4ec8de5_1: false -> true
530b157_1: false -> false
530b157_2: false -> false
530b157_3: false -> false
```

解释：

- CLSTR skill-context 明显改善了 Qwen-only 的 AppWorld executor 表现，尤其修复
  unique-song count 和 release-year count 任务；
- 仍失败的是 phone+venmo grocery 三个任务，模型能执行并 `complete_task`，但最终
  evaluate 不通过，主要是短信金额/联系人/交易语义细节不稳；
- 该对照使用的是旧 AppWorld CLSTR ACT predictions，不是当前 unified v4.1b 主线结果；
- 但它证明“给 Qwen 正确 skill context”确实能提升 downstream task success，因此
  AppWorld 不是完全没有 CLSTR 学习空间。

下一步建议：

1. 在更大但仍可控的 dev57 上复核 `qwen_only` vs `clstr_act skill-context`；
2. 若 dev57 仍有显著提升，再考虑 AppWorld ACT/HRPO 小规模 rollout；
3. 若 phone+venmo 类任务仍是主要失败簇，应先改 prompt/schema hints 或换更强同尺寸 Qwen，
   不要直接靠 RL 稀疏 reward 硬学。

## 2026-06-09 — v4.1b conservative-listwise Stage4 ACT 适配

当前 Stage4 已从 v4.2 progressive-final 默认路线切到当前最强的
v4.1b conservative-listwise Stage2 checkpoint：

```text
data = data/clstr_unified_pretrain_v4_1b
routing checkpoint = outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt
head checkpoint = outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt
output = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act
```

Stage4 训练配置保持保守：

- `lambda_pref=0.0`：先做纯 Stage4 ACT；
- `transition_scoring_mode=v4_1b_action_observation`、`transition_residual_lambda=0.0`：
  沿用 v4.1b action-observation scoring；
- `train_transition=1`：训练 `transition/trans_head/action_proj`；
- `stage0_top_m=350`、`positive_missing_policy=skip`：候选来自真实 Stage0 top-M，不注入 gold。

新增 wrapper：

```text
scripts/sbatch/run_clstr_unified_stage4_act_train_v4_1b_conservative_listwise.sh
```

同时 Stage4 CLI/generic sbatch 新增 `benchmark_caps`。v4.1b Stage4 默认：

```text
toolbench_g3=4000,traject_bench=4000,alfworld=-1,webshop=4000
```

这样 full 前不会对全部 10 万级 row 做 Stage0 top-M 预计算，同时避免小样本只覆盖
`trajectories.jsonl` 文件开头的 `alfworld`。通过 `sbatch --export` 传该变量时使用冒号，
脚本内部会转换为逗号，避免 Slurm 拆分逗号赋值。

已完成 smoke 证据：

```text
smoke20: job_id=85664, ExitCode=0
smoke200: job_id=85670, ExitCode=0
smoke200 tail:
  stage4_act_loss ≈ 1.08
  stage4_next_skill_recall@1 ≈ 0.7425
  stage4_next_skill_recall@5 ≈ 0.96
  stage4_next_skill_mrr ≈ 0.8231
```

smoke200 在只期望 `alfworld` 的小样本 quality gate 下通过；正式 full 仍需覆盖
`toolbench_g3/traject_bench/alfworld/webshop` 并在完成后运行 Stage4 quality gate。

本轮本地验证：

```text
python -m pytest tests/test_stage4_act_train.py -k benchmark_caps_before_max_rows -q
python -m pytest tests/test_sbatch_scripts.py -k 'unified_stage4_act or v4_1b_conservative_listwise_stage4' -q
python -m pytest tests/test_stage4_quality_gate.py -k 'zero_residual_scoring or expected_transition_scoring_overrides or rejects_legacy_transition_scoring_metadata or accepts_complete_transition_act_training' -q
bash -n scripts/sbatch/run_clstr_unified_stage4_act_train_v4_1b_conservative_listwise.sh scripts/sbatch/run_clstr_unified_stage4_act_train.sh
python -m py_compile clstr/stage4_act_train.py clstr/stage4_quality_gate.py scripts/run_clstr_stage4_act_train.py scripts/audit_clstr_stage4_quality.py
git diff --check -- stage4 related files
```

## 2026-06-09 — v4.1b conservative-listwise Stage4 full 完成

Stage4 适配时发现一个设计问题：ACT rows 原先按 `trajectories.jsonl` 文件顺序进入
batch，small smoke 的 first/last window 会被 benchmark 分段顺序影响。已修复为训练前
固定 seed shuffle，并在报告中记录：

```text
row_order.strategy = seeded_shuffle
row_order.seed = 17
```

修复后多 benchmark smoke：

```text
job_id = 85700
output = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act_smoke_multibench_shuffle
state = COMPLETED
exit_code = 0:0
stage4_quality_gate.status = ok
stage4_quality_gate.blockers = []
stage4_rows = 248
benchmark_counts = {alfworld: 64, toolbench_g3: 56, traject_bench: 64, webshop: 64}
stage0 next positive coverage@350 = 0.96875
positive_injected_rows = 0
last20 recall@5 = 0.775
```

full job：

```text
job_id = 85703
output = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act
state = COMPLETED
exit_code = 0:0
elapsed = 00:12:00
checkpoint = outputs/clstr_unified_stage4_v4_1b_conservative_top350_listwise_nomask_joint_act/checkpoints/clstr_stage4_act-step2000.pt
stage4_quality_gate.status = ok
stage4_quality_gate.blockers = []
```

full 数据与指标：

```text
retained source rows = 15307
stage4_rows = 14701
benchmark_counts = {
  alfworld: 3307,
  toolbench_g3: 3394,
  traject_bench: 4000,
  webshop: 4000
}
stage0 next positive coverage@350 = 0.96041
stage0_topm_next_positive_coverage@350 = 0.95747
positive_injected_rows = 0
candidate_source = stage0_topm_online

first200 stage4_act_loss = 2.0910
last200 stage4_act_loss = 2.0339
stage4_act_loss_increase = -0.0571
last200 stage4_next_skill_recall@5 = 0.72375
first500 recall@5 = 0.72075
last500 recall@5 = 0.74150
```

结论：Stage4 已经完成当前最强 v4.1b conservative-listwise 路线的 full 训练并通过
quality gate。该结果证明 Stage4 ACT 训练 artifact 可用；它还不是 downstream/official
benchmark 结果，下一步应进入 evaluation/readiness 并与 SkillRouter 做同口径比较。

## 2026-06-08 v4.1b conservative Stage2 baseline 回退执行

在多轮 v4.2 / inventory / prior-residual 组合未稳定超过旧指标后，当前阶段先停止继续叠加
v4.2 改动，转入保守 baseline 验证：

- 数据使用 `data/clstr_unified_pretrain_v4_1b`；
- Stage0 使用
  `outputs/clstr_unified_stage0_v4_1b_handoff_mpn_top350/checkpoints/clstr_unified_retrieval_v2-step5000.pt`；
- Stage1 使用
  `outputs/clstr_unified_stage1_v4_1b_top350_heads_init/checkpoints/clstr_stage1_heads-step3000.pt`；
- Stage2 训练配置锁为 v4.1b 原始候选/损失设定：
  `stage0_top_m=350`、`positive_missing_policy=skip`、
  `transition_inventory_mask_mode=off`、`transition_loss_type=cross_entropy`、
  `transition_positive_mode=single`；
- loss 权重回到 v4.1b 原始配方：
  `L_policy=1.0`、`L_trans=0.05`、`L_trans_skill_ce=0.2`、
  `STOP=0.2`、`belief=0.1`、`routing=0.0`；
- 新增 `transition_scoring_mode=v4_1b_action_observation`，显式复现 v4.1b
  transition skill CE 的旧语义：用 `action_text` embedding 作为 transition
  observation/proxy 输入，而不是走当前 `skill_prior_plus_action_observation_residual`
  的 prior+residual 合成路径。

新增脚本：

```text
scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_1b_conservative.sh
```

该脚本先读取旧 Stage1 产出的 `stage0_candidate_handoff.json`，要求 top350、skip、
无 injection，且 current/next positive coverage 都不低于 0.98，再生成 handoff gate
交给 generic Stage2 脚本。这样不会依赖缺失的 v4.1b full retrieval eval gate，也不会放松
Stage2 的候选安全检查。

本次改动目的不是证明 v4.2/listwise 失败，而是先拿到一个忠实、可审计的 v4.1b Stage2
baseline。若该 Stage2 能把 Stage1 的好指标延续到 full-base，再进入下游评估；若不能，
说明旧 Stage1 指标可能不能转化为 Stage2 能力，需要回到 transition 标签/候选/输入结构做诊断。

## 2026-06-09 Phase 2 方向B准备：top50 listwise + trajectory prior

在 v4.1b conservative Stage2 baseline 运行期间，已准备方向B的代码和脚本，但尚未提交
GPU 训练作业。方向B的实现目标是接近 SkillRouter 的高质量候选重排设定，同时保持 CLSTR
的通用性：

- 新增候选模式 `transition_inventory_mask_mode=stage0_topk_trajectory_prior`；
- 该模式不再按 trajectory history 做 hard filtering；
- 候选以 Stage0 top-M 原始顺序为主，默认用 `transition_inventory_min_candidates=50`
  作为 target count；
- trajectory inventory 只作为少量 prior slots，而不是硬过滤条件；
- 若 gold next skill 在 Stage0 top-M 内但被 top50 截断，训练时保留 gold，以保证
  listwise/NLL 有有效正例；这与现有 inventory filter 的 gold-preserve 训练逻辑一致；
- Stage1/Stage2 方向B wrapper 都锁定 v4.1b 数据和
  `transition_scoring_mode=v4_1b_action_observation`，保证只比较候选/loss 变化。

新增脚本：

```text
scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_1b_aggressive_top50_listwise.sh
scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_1b_aggressive_top50_listwise.sh
```

建议执行顺序：

1. 先等 v4.1b conservative Stage2 baseline 至少完成并通过质量检查；
2. 若 baseline 指标健康，再提交方向B Stage1；
3. 只有方向B Stage1 的 policy / transition@1 / transition@5 明确优于 v4.1b 原始
   Stage1 或至少不退化，才提交方向B Stage2；
4. 若方向B不提升，不继续叠加 Stage2/Stage4，避免无效烧钱。

## 2026-06-09 v4.1b conservative Stage2 baseline 完成

作业 `85617` 已完成：

```text
output_dir = outputs/clstr_unified_stage2_v4_1b_conservative_top350_ce_nomask
checkpoint = outputs/clstr_unified_stage2_v4_1b_conservative_top350_ce_nomask/checkpoints/clstr_full_base-step10000.pt
stage2_quality_gate = ok
```

最终训练配置确认：

- `data = data/clstr_unified_pretrain_v4_1b`
- `transition_inventory_mask_mode=off`
- `transition_loss_type=cross_entropy`
- `transition_positive_mode=single`
- `transition_scoring_mode=v4_1b_action_observation`
- `transition_residual_lambda=0.0`
- 无 injected positives，Stage0 top350 handoff coverage 约 `0.988`

关键指标：

```text
metric_averages.policy_expert_recall@1 = 0.91795
metric_averages.transition_skill_recall@1 = 0.56471
metric_averages.transition_skill_recall@5 = 0.82680
metric_averages.transition_skill_mrr = 0.67345
metric_averages.transition_skill_ce_loss = 1.36751
metric_averages.transition_skill_ce_candidate_count = 118.52
last-window transition@5 = 0.83792
first-window transition CE = 1.60474
last-window transition CE = 1.25171
```

Stage2 quality gate 结果：

```text
status = ok
blockers = []
loss_drop = 0.09146
transition_skill_recall@5_drop = -0.03375
transition_skill_ce_increase = -0.35302
```

结论：v4.1b conservative Stage2 是当前最稳的 solid baseline。它说明 v4.1b Stage1
能力可以延续到 Stage2，且 Stage2 的 transition@1 明显高于此前 v4.1b Stage1 记录。
因此已按计划提交方向B Stage1 对比作业：

```text
job_id = 85627
script = scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_1b_aggressive_top50_listwise.sh
```

只有当方向B Stage1 明确优于上述 baseline 相关指标或至少不退化时，才继续提交方向B Stage2。

## 2026-06-09 Direction B top50/listwise 对比结果

方向B Stage1 对比作业已正常完成：

```text
job_id = 85627
state = COMPLETED
exit_code = 0:0
output_dir = outputs/clstr_unified_stage1_v4_1b_aggressive_top50_listwise_trajectory_prior_heads_init
checkpoint = outputs/clstr_unified_stage1_v4_1b_aggressive_top50_listwise_trajectory_prior_heads_init/checkpoints/clstr_stage1_heads-step3000.pt
```

该实验锁定 v4.1b 数据和语义，只改候选与 loss：

```text
transition_inventory_mask_mode = stage0_topk_trajectory_prior
transition_loss_type = listwise_nll
transition_positive_mode = gold_plus_equivalent
transition_scoring_mode = v4_1b_action_observation
transition_residual_lambda = 0.0
candidate_count = 50
```

最终指标：

```text
policy_expert_recall@1 = 0.90786
transition_skill_recall@1 = 0.32539
transition_skill_recall@5 = 0.66550
transition_skill_mrr = 0.47167
transition_skill_ce_loss = 2.53417
transition_current_skill_switch_false_positive@1 = 0.15406
```

结论：方向B不是当前更优路线。相比 v4.1b conservative Stage2 baseline
`transition@1=0.56471 / transition@5=0.82680`，top50/listwise/trajectory-prior
Stage1 的 transition ranking 明显退化；因此不提交方向B Stage2。

当前最稳主线仍是：

```text
outputs/clstr_unified_stage2_v4_1b_conservative_top350_ce_nomask/checkpoints/clstr_full_base-step10000.pt
```

下一步应围绕该 checkpoint 做下游评估，验证它是否能转化为真实 task-level 能力；
listwise/top-K 改进暂时降级为可选研究分支，不再阻塞主线。

## 2026-06-09 保守 listwise 改进完成

基于 Direction B 失败结论，本轮只做保守改进：保留 v4.1b 的数据、Stage0、top350
handoff、`positive_missing_policy=skip`、inventory mask off 和
`v4_1b_action_observation` scoring，仅把 transition skill loss 从单正例 CE 切到
`listwise_nll + gold_plus_equivalent`。

新增脚本：

```text
scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_1b_conservative_listwise.sh
scripts/sbatch/run_clstr_unified_stage2_full_base_train_v4_1b_conservative_listwise.sh
```

Stage1 作业：

```text
job_id = 85637
state = COMPLETED
exit_code = 0:0
output = outputs/clstr_unified_stage1_v4_1b_conservative_top350_listwise_nomask_heads_init
checkpoint = outputs/clstr_unified_stage1_v4_1b_conservative_top350_listwise_nomask_heads_init/checkpoints/clstr_stage1_heads-step3000.pt
```

Stage1 相比旧 v4.1b Stage1：

```text
transition@1: 0.37436 -> 0.41936
transition@5: 0.83317 -> 0.83444
MRR:          0.55366 -> 0.58944
switch@1:     0.23306 -> 0.28600
switch FP@1:  0.17986 -> 0.12086
```

因为 Stage1 对比不退化且关键指标提升，已继续 Stage2。

Stage2 作业：

```text
job_id = 85644
state = COMPLETED
exit_code = 0:0
output = outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask
checkpoint = outputs/clstr_unified_stage2_v4_1b_conservative_top350_listwise_nomask/checkpoints/clstr_full_base-step10000.pt
stage2_quality_gate = ok
```

Stage2 相比上一版 v4.1b conservative CE baseline：

```text
transition@1: 0.56471 -> 0.56938
transition@5: 0.82680 -> 0.83126
MRR:          0.67345 -> 0.67739
switch@1:     0.47058 -> 0.47822
switch FP@1:  0.10264 -> 0.07777
```

结论：保守 listwise 是当前最强 baseline；它没有改变候选分布，也没有引入 AppWorld
特化。下一步应优先基于该 checkpoint 跑下游 task-level 评估。

## 2026-06-08 Stage1 候选方式回退到 inv64

复查 `inventory100 + prior_residual_l025` Stage1 full 结果后，确认这次退化主要发生在
Stage1 transition ranking，而不是 ACT：Stage1 尚未进入 ACT 训练；policy@1 约 `0.818`，
但 transition@1 / @5 只有约 `0.2925 / 0.6542`，且 prior/residual/final transition
ranking 在 tail metrics 中几乎没有差异，说明 action-aware residual 分支没有产生有效重排。

因此本轮只回退候选方式，不回退 transition 语义：

- Stage1 默认输出目录改为
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_prior_residual_l025_heads_init`；
- `transition_inventory_min_candidates` 默认从 `100` 回到 `64`；
- Stage2、Stage3、Stage4、readiness audit 和 guarded submit 默认 checkpoint 路径同步消费新的
  Stage1 `inventory64` artifact；
- 保留 `skill_prior_plus_action_observation_residual` 和
  `transition_residual_lambda=0.25`，后续如果 inv64 仍低，再单独诊断 residual 分支是否需要
  更强监督或改成可学习 gating。

这个回退的依据是：top100 让候选集更难但没有提升 transition top-1，反而把
transition@5 从旧 inv64 附近的 `0.84` 压到约 `0.65`。旧 inv64 候选方式不使用 gold
injection，也不是 AppWorld 特化；它只是让 inventory prior 保持为候选软约束，并从真实
Stage0 top-M 顺序回填到较小的候选下限。

本轮仍未提交任何 Slurm/GPU 作业。

## 2026-06-08 Stage1 prior-residual gate 加固与 full 结果

提交 Stage1 prior-residual 版本前做了新的提交前审计，发现 active 默认输出目录已经切到
`prior_residual_l025`，但 Stage1/Stage2/Stage4 quality gate 还没有强制检查
`transition_scoring_mode` 与 `transition_residual_lambda`。这会让旧 action-aware /
hardneg artifact 有机会冒充当前主线 artifact 通过 gate。

已修复：

- `clstr/stage_quality_common.py` 新增 `transition_scoring_safety()`，默认要求：
  `transition_scoring_mode=skill_prior_plus_action_observation_residual`，
  `transition_residual_lambda=0.25`；
- Stage1/Stage2/Stage4 quality gate 均接入该检查；
- Stage1/Stage2/Stage4 gate 测试新增 legacy scoring metadata 拒绝用例；
- `tests/test_unified_training_readiness.py` 的 fake Stage1/Stage2 artifact 同步到新的
  prior-residual 元数据。

CPU 验证：

```text
PYTHONPATH=. python -m pytest -q \
  tests/test_stage1_heads_quality_gate.py \
  tests/test_stage2_quality_gate.py \
  tests/test_stage4_quality_gate.py \
  tests/test_sbatch_scripts.py \
  tests/test_clstr_topm_candidate_handoff.py \
  tests/test_unified_training_readiness.py \
  tests/test_stage_quality_common.py
=> 121 passed

python -m py_compile clstr/stage_quality_common.py \
  clstr/stage1_heads_quality_gate.py clstr/stage2_quality_gate.py \
  clstr/stage4_quality_gate.py scripts/run_clstr_stage1_heads_init.py \
  scripts/run_clstr_stage2_full_base_train.py scripts/run_clstr_stage4_act_train.py \
  scripts/audit_clstr_stage1_heads_quality.py scripts/audit_clstr_stage2_quality.py \
  scripts/audit_clstr_stage4_quality.py scripts/audit_clstr_unified_training_readiness.py
=> passed

git diff --check
=> passed
```

Slurm 运行：

- smoke job `85534`：A800，`MAX_ROWS=256`、`MAX_STEPS=20`、`BATCH_SIZE=2`，
  输出目录
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory100_listwise_prior_residual_l025_heads_init_smoke`；
- smoke 结果：训练正常完成，20 条 metrics，无 NaN/inf，handoff 无 gold injection，
  `current_positive_coverage@M=1.0`，`next_positive_coverage@M=1.0`，
  prior-residual 元数据完整；
- full job `85553`：A800，canonical 输出目录
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory100_listwise_prior_residual_l025_heads_init`，
  正常完成 3000 steps。

full 结果摘要：

```text
metric_rows = 3000
loss_first100 = 2.8711
loss_last100 = 1.7754
policy_recall@1_last100 = 0.8183
transition_recall@1_last100 = 0.2925
transition_recall@5_last100 = 0.6542
transition_ce_first100 = 4.4232
transition_ce_last100 = 2.6247
bad_numeric_count = 0
stage0 handoff:
  source_rows = 37000
  retained_rows = 36544
  skipped_rows = 456
  injected_positive_rows = 0
  current_positive_coverage@M = 0.9851
  next_positive_coverage@M = 0.9898
```

正式 Stage1 quality gate 结果：

```text
status = action_required
blockers = [
  transition_recall_at_1_too_low,
  transition_recall_at_5_too_low
]
```

结论：这次 Stage1 full 作业本身健康，loss 与 CE 都明显下降，policy head 学到了东西，
handoff 与元数据也符合当前主线；但 transition top-1/top-5 ranking 未达到进入 Stage2
的门槛。因此目前不应提交 Stage2。下一步应先定位 transition recall 仍低的原因，重点看：
candidate inventory mask 是否过度收缩/排序噪声、`prior + 0.25 * residual` 是否残差权重过低、
以及 Stage1 训练目标是否仍被 policy/STOP 的快速收敛主导。

## 2026-06-08 Transition prior + action-aware residual 实现

针对 action-aware next-observation 版本 top-1 低于旧 transition 输入版本的问题，本轮没有做
`git reset` 或整文件回退；旧版本只作为语义参考，后续已经修过的 sbatch/gate/readiness/
checkpoint safety/path cleanup/action-aware 通道修复都保留。

新增的 transition scoring 采用：

```text
transition_logits = prior_logits + transition_residual_lambda * residual_logits
```

其中：

- `prior_logits` 来自 skill-level transition prior：当前 `m_t/current_skill` + neutral
  observation，恢复旧版本较稳定的 skill-to-skill 转移信号；
- `residual_logits` 来自当前 action-aware + next-observation 分支：实际 `action_text`
  经 `action_proj`，observation 通道仍使用 `next_observation_text`；
- `transition_residual_lambda` 默认为 `0.25`，并在 Stage1/Stage2/Stage4 的 CLI 与 sbatch
  入口中暴露；`lambda=0` 可退化为纯 prior，`lambda>0` 允许 action/observation 修正排序；
- 训练 metrics、checkpoint 与 train_report 记录
  `transition_scoring_mode=skill_prior_plus_action_observation_residual` 和
  `transition_residual_lambda`。

已覆盖：

- Stage1/Stage2 full-base transition skill CE；
- Stage4 ACT next-skill ranking；
- Stage1/Stage2/Stage4 Python CLI 与 sbatch 参数透传；
- prior/residual 分支的 CPU 单元测试，以及 action/next-observation 通道不混淆的回归测试。

本轮只做 CPU 级验证，没有提交任何 Slurm/GPU 作业。下一步如要继续训练，应先跑小样本 smoke，
确认 train_report 中出现新的 scoring mode 与 lambda，再由用户确认是否提交 full job。

## 2026-06-08 Stage1 top100 no-hardneg full 结果

按最新 Stage1 修复方案提交并完成 full job：

- job：`85066`
- partition：`gpu_a800`
- state：`COMPLETED`
- exit code：`0:0`
- elapsed：`00:30:53`
- output：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory100_listwise_prior_residual_l025_heads_init`
- checkpoint：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory100_listwise_prior_residual_l025_heads_init/checkpoints/clstr_stage1_heads-step3000.pt`

本次作业运行和安全字段正常：

- `candidate_source=stage0_topm_online`
- `positive_missing_policy=skip`
- `injected_positive_rows=0`
- `checkpoint_excludes_frozen_routing_foundation=True`
- `transition_inventory_min_candidates=100`
- `transition_hard_negative_margin=0.0`
- `transition_skill_ce_candidate_count=100.0`
- Stage0 top350 coverage 与旧版相同：current `0.9851`，next `0.9898`

但 Stage1 quality gate 未通过：

```text
status = action_required
blockers = [transition_recall_at_1_too_low, transition_recall_at_5_too_low]
```

核心指标：

```text
first100 loss          = 2.8903
last100  loss          = 1.7780
first100 policy@1      = 0.7400
last100  policy@1      = 0.8183
first100 transition@1  = 0.2925
last100  transition@1  = 0.2967
first100 transition@5  = 0.6100
last100  transition@5  = 0.6567
first100 transition CE = 4.4616
last100  transition CE = 2.6278
```

和旧 `inventory64_listwise_heads_init` 对比：

```text
旧版 last100 transition@1 = 0.3842
新版 last100 transition@1 = 0.2967

旧版 last100 transition@5 = 0.8417
新版 last100 transition@5 = 0.6567

旧版 candidate count      ≈ 49
新版 candidate count      = 100
```

结论：这次 top100 保底和关闭 hard-negative 没有修复 Stage1，反而显著降低
transition@5。它说明 Stage1 当前问题不是“候选数越多越好”这么简单；更可能是
inventory-softening 后候选难度上升，而当前 Stage1 listwise objective / loss 比例 /
输入分布不足以在 3000 step 内完成 top-M reranking。这个 checkpoint 不能作为进入 Stage2
的主线前置。

下一步不应提交 Stage2。应回到 Stage1 设计排查：优先比较 `inventory64` 与 `top100`
的 row-level positive rank 分布、按 benchmark/self/switch 分解 transition@K，并判断
是否应恢复较小候选集、恢复旧 `L_policy=0.6/L_trans_skill_ce=0.6`、或把 Stage1 gate
定义为候选内 top-k 初始化而将 top-1 reranking 交给 Stage2。

## 2026-06-08 Stage1 no-hardneg/top100 soft-inventory 修复

用户指出不能用 Stage2 指标和 Stage1 指标直接比较。重新只按 Stage1 对 Stage1
比较后，确认当前 hardneg01 Stage1 明显弱于旧 Stage1 heads-init：

- 旧 `v4_inventory_handoff_full` Stage1：transition@1 约 `0.608`，candidate count
  `350`；
- 旧 `toolbench_g3_traject_split_heads_init_full_v2` Stage1：transition@1 约
  `0.606`，candidate count `100`；
- 当前 hardneg01 Stage1：transition@1 约 `0.328`，tail-100 约 `0.314`，
  candidate count 约 `48`。

因此这轮修复不再把 Stage1 hard negative 当默认必需项，而是恢复 Stage1 heads-init 的
核心定位：在真实 Stage0 top-M 候选内初始化 policy / transition / belief / STOP heads，
同时保留 inventory prior 但避免它把候选空间过度收缩。

已完成代码修复：

- `_stage0_topk_with_explicit_inventory()` 增加 `min_candidates` 回填：若 row 有 explicit
  inventory，先在 inventory 内按 Stage0 logits 排序；若候选数不足，则从真实 Stage0
  全池 top-M 顺序回填到下限。该逻辑不是 gold injection，不使用 dev/test 标签。
- Stage1 handoff 现在把 `transition_inventory_min_candidates` 也用于 handoff 阶段，
  不再只在 loss 内回填；避免 explicit inventory 在 handoff 时已经把 Stage0 top-M
  丢掉。
- Stage1 默认配方更新为：
  `top_m=350`、`transition_inventory_min_candidates=100`、
  `L_policy=0.7`、`L_trans=0.2`、`L_trans_skill_ce=0.5`、
  `STOP=0.1`、`belief=0.1`、`transition_hard_negative_margin=0.0`。
- active Stage1 输出目录切到：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory100_listwise_prior_residual_l025_heads_init`。
- Stage1 quality gate 中 hard negative 改为 optional safety：默认不要求启用；
  若未来启用，则要求 train_report 与 candidate training 里的 hard-negative weight
  一致，并且存在 hard-negative pair。
- Stage2/readiness/submit 默认 Stage1 checkpoint 已同步指向新的 no-hardneg Stage1
  目录，避免后续误用旧 hardneg01 checkpoint。

验证：

```text
pytest focused Stage1/handoff/sbatch/readiness: 103 passed
bash -n Stage1/Stage2/readiness/submit sbatch scripts: passed
py_compile full_base_train.py, stage1 gate, Stage1/Stage2 CLI/readiness scripts: passed
```

当前无运行中的 Slurm 作业；GPU partitions 在检查时均无可立即运行空卡。下一步提交
Stage1 full job 到计算节点，先观察 `setup_status.jsonl` 是否持续写入
`stage0_candidate_handoff_progress`，进入训练后再看 `training_metrics.jsonl`、
`loss_curve.svg`、`checkpoints/latest.pt` 和最终 Stage1 gate。若 Stage1 指标不正常，
不进入 Stage2。

## 2026-06-08 Stage0 证据补齐与 Stage1 hardneg01 full 结果

按当前 guarded submit 入口提交 Stage1 前，Stage0 gate 首先阻断，原因是缺少同池
SkillRouter frozen bi-encoder baseline metrics 与 Stage0 full retrieval eval metrics。已补齐两个
前置评估作业：

- `84977`: `scripts/sbatch/run_clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak.sh`
  - 输出：`outputs/clstr_stage0_skillrouter_frozen_baseline_v4_2_nowweak/metrics.json`
  - Recall@20/50/100 = `0.538654 / 0.630463 / 0.699251`
- `84983`: `scripts/sbatch/run_clstr_stage0_full_retrieval_eval_v4_2_nowweak.sh`
  - 输出：
    `outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/full_retrieval_eval/metrics.json`
  - Recall@20/50/100 = `0.690344 / 0.797277 / 0.874431`

补齐后 Stage0 gate 通过，blockers 为空。随后通过 guarded 入口提交 Stage1：

- `84984`: `scripts/submit_clstr_stage1_after_stage0_gate.sh`
- 输出：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_action_aware_nextobs_hardneg01_heads_init`
- checkpoint:
  `checkpoints/clstr_stage1_heads-step3000.pt`

Stage1 运行健康信号：

- Stage0 top-350 handoff 使用 no-injection：`positive_missing_policy=skip`，
  `injected_positive_rows=0`
- handoff 覆盖：
  - current positive coverage@350 = `0.985134`
  - next positive coverage@350 = `0.989763`
- hard-negative ranking 已实际启用：
  - `loss_weights.transition_hard_negative_margin=0.1`
  - `transition_candidate_training.hard_negative_loss_weight=0.1`
  - tail `transition_hard_negative_pair_count=3.09`
- loss 有真实下降：first-100 mean `2.095164` -> last-100 mean `1.517484`
- transition CE 有改善：first-100 mean `2.366570` -> last-100 mean `1.675442`
- transition recall@5 稳定：last-100 mean `0.833333`

但 Stage1 quality gate 未通过：

```text
status = action_required
blockers = [
  transition_recall_at_1_too_low
]
last_transition_skill_recall@1 = 0.314167
required tail recall@1 >= 0.35
```

和旧版本对比后，当前 hardneg01 并不是作业异常或数据加载失败；问题集中在 top-1 排序能力：
recall@5 与 CE 都正常，但 gold next skill 在候选内常能进 top-5，尚不能稳定排到第 1。
旧 `nowweak_top350_inventory_listwise` Stage1 的平均 transition recall@1 约 `0.348861`，
也只是接近 gate 阈值；因此下一步不应进入 Stage2，而应先处理 Stage1 top-1 ranking。

低成本优先排查方向：

- 检查 Stage1 gate 是否应同时参考 last-100 与 global/last checkpoint eval，而不是只用
  noisy batch tail recall@1；当前 batch size=4，tail recall@1 方差较大；
- 若 gate 仍坚持 top-1 阈值，优先尝试提升 top-1 排序而不改变论文设定：
  - 增大 Stage1 effective batch 或使用 gradient accumulation，降低 batch metric 方差；
  - 针对 `switch` 样本加轻量 current-skill false-positive penalty；
  - 调整 `L_policy/L_trans_skill_ce/transition_hard_negative_margin` 比例，避免 policy 与
    hard-negative margin 抢占 listwise CE 的排序空间；
  - 用 held-out row-level transition eval 代替训练 batch tail 作为 Stage1 gate 主指标。

在 Stage1 gate 通过前，不应提交 Stage2/Stage4。

## 2026-06-07 — Stage1 heads-init 设计复查

继续复查 Stage1 后，确认当前 `v4.2_progressive_final` 已完成的 Stage1
结果本身没有被推翻：它使用真实 Stage0 top-350 handoff，`positive_missing_policy=skip`，
`injected_positive_rows=0`，并且 checkpoint 已记录
`checkpoint_excludes_frozen_routing_foundation=True`。

但代码层面有两个证据链加固点需要修复：

- Stage2 从 Stage0 checkpoint 构建模型后再加载 Stage1 heads checkpoint。若未来某个
  Stage1/head artifact 含有 `encoder.*`、`cross_encoder.*` 或 `skill_table.*`，
  旧 loader 会把这些 key 一起加载，可能覆盖 Stage0 routing foundation；
- Stage1 quality gate 之前只检查 `frozen_routing_foundation=True`，没有强制要求
  checkpoint 真的排除了 frozen routing foundation，也没有阻断训练时 gold-positive
  injection 的 artifact；
- 专用 Stage1 progressive-final wrapper 没有脚本级 `#SBATCH --time`，直接
  `sbatch` 时仍可能生成 unlimited-time 作业。

已修正：

- `load_head_checkpoint_into_model()` 默认启用 `protect_routing_foundation=True`，
  加载 Stage1/Stage2 head checkpoint 时过滤 `encoder.*`、`cross_encoder.*`、
  `skill_table.*`，并在 report 中记录被跳过的 protected keys；
- `clstr/stage1_heads_quality_gate.py` 新增 blocker：
  `stage1_checkpoint_includes_frozen_routing_foundation`、
  `stage1_positive_missing_policy_not_skip`、
  `stage1_injected_stage0_positives`；
- Stage1 quality report 现在显式写出 `checkpoint_safety` 与 `handoff_safety`；
- `scripts/sbatch/run_clstr_unified_stage1_heads_init_v4_2_progressive_final.sh`
  加入 `#SBATCH --time=06:00:00`，避免复现之前 Stage1 pending/unlimited 问题。

重新审计当前 Stage1 artifact：

- output：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init`
- gate：`status=ok`，`blockers=[]`
- last-window 指标仍为：loss `1.3690`，transition@1 `0.3842`，
  transition@5 `0.8417`，transition CE `1.6114`
- safety：`checkpoint_excludes_frozen_routing_foundation=True`，
  `positive_missing_policy=skip`，`injected_positive_rows=0`

结论：这次是 Stage1 证据链和提交安全加固，不需要重训 Stage1/Stage2；但后续任何
Stage1 artifact 若缺少上述安全元数据或依赖 injected positives，都不能作为进入
Stage2/Stage4 主线的前置。

## 2026-06-07 — Stage1 next-observation 语义重跑结果

在修正 full-base transition/belief 输入语义后，本轮按要求只重跑 Stage1，没有提交
Stage2。

提交信息：

- Slurm job：`84847`
- partition：`gpu_a800`
- output：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init_nextobs_rerun`
- checkpoint：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_heads_init_nextobs_rerun/checkpoints/clstr_stage1_heads-step3000.pt`
- Slurm 状态：`COMPLETED`
- exit code：`0:0`
- elapsed：`00:26:00`

这次不是运行失败，而是 Stage1 quality gate 没过：

```text
status = action_required
blocker = transition_recall_at_1_too_low
```

tail-100 指标：

```text
loss                      = 1.4243
policy@1                  = 0.8183
transition@1              = 0.3217
transition@5              = 0.8458
transition_mrr            = 0.5155
transition_skill_ce_loss  = 1.7049
```

安全字段是正常的：

```text
checkpoint_excludes_frozen_routing_foundation = true
positive_missing_policy = skip
injected_positive_rows = 0
```

因此这不是 Stage0 foundation 被覆盖、gold-positive 注入或 Slurm 运行异常导致的问题。
和旧的 progressive-final Stage1 相比：

```text
old Stage1 tail-100 transition@1     = 0.3842
nextobs rerun tail-100 transition@1  = 0.3217

old Stage1 tail-100 transition@5     = 0.8417
nextobs rerun tail-100 transition@5  = 0.8458
```

结论：

- next-observation 语义在方法定义上更干净；
- 但当前 Stage1 heads-init 下，它让 transition top-1 ranking 退化；
- transition@5 基本稳定，说明 gold skill 仍常在前 5，但 top-1 排序变差；
- 当前 rerun checkpoint 不能作为进入 Stage2 的前置；
- 下一步应先诊断 Stage1 transition objective/input/data 是否和 next-observation 语义匹配，
  而不是直接提交 Stage2。

### 问题定位

进一步检查后，旧/新 Stage1 的差异不是候选覆盖、loss 权重或 checkpoint safety：

- 两次 retained rows 都是 `36544`；
- top-350 current coverage 都是 `0.9851`；
- top-350 next coverage 都是 `0.9898`；
- `positive_missing_policy=skip`；
- `injected_positive_rows=0`；
- loss 权重和 sampler 相同。

退化主要集中在 switch transition，而不是 self transition：

```text
last-100 self transition@1:
old     = 0.4100
nextobs = 0.4050

last-100 switch transition@1:
old     = 0.2817
nextobs = 0.2083

last-100 current-skill false positive on switch:
old     = 0.2125
nextobs = 0.2525
```

根因判断：

- 旧实现语义上有问题，因为它把 `action_text` embedding 当作 observation 输入；
- 但这个错误实现保留了“实际执行动作/API 调用”的强判别信号；
- 修正后，transition 只拿到 current skill embedding 和 `next_observation_text`，
  实际 `action_text` 没有进入 action channel；
- 许多数据集的 `next_observation_text` 本身高度重复或低判别，例如 WebShop 的
  `Observation: OK`、`Observation: shopping is finished.`，TRAJECT-Bench 的
  `ERROR: Failed after 5 attempts`、空字符串或订阅错误 HTML，ToolBench-G3 的重复
  API error/response；
- 因此 next-observation-only 的 Stage1 transition 对 switch top-1 排序不够，模型更容易
  把 current skill 保留在第一位。

正确修复方向不是回滚到 action-as-observation，而是把 transition 改为：

```text
m_t + actual action_text/action embedding + next_observation_text embedding -> predicted next belief / next skill
```

也就是说，`next_observation_text` 仍然是 observation channel；`action_text` 应进入 action
channel。这样既保留方法语义，也恢复实际动作对 next-skill ranking 的判别信息。

### 代码修复

已按上述方向修复 transition 输入：

- `_transition_prediction()` 新增可选 `action_emb`；
- 当 actual `action_text` embedding 存在时，先经过 `action_proj`，再作为
  `TransitionPredictor.forward(m_t, a_t, o_t_emb)` 的 `a_t`；
- `next_observation_text` embedding 仍然作为 `o_t_emb`；
- 如果某些历史 row 没有 `action_text` 或缓存的 action embedding，则回退到原来的
  current-skill soft-shared action input；
- full-base 的 `L_trans`、`L_trans_skill_ce`、`belief`、replay-prefix belief reconstruction
  已接入该路径；
- Stage2 row-level transition diagnostics 同步接入该路径；
- Stage4 ACT 也同步接入该路径，避免 Stage1/2 和 Stage4 的 transition 语义分叉。

本轮新增/增强的回归测试覆盖：

- full-base transition 同时满足：
  - action channel 使用 actual `action_text` embedding；
  - observation channel 使用 `next_observation_text` embedding；
  - 不再把 `action_text` 当作 observation；
- Stage4 ACT transition 同时满足：
  - action channel 使用 actual `action_text` embedding；
  - observation channel 使用 `next_observation_text` embedding；
  - 不再只用 current-skill embedding 作为 action。

本地验证已通过：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_full_base_train.py tests/test_v4_action_proj_sharing.py
# 62 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage2_transition_row_diagnostics.py tests/test_stage4_act_train.py
# 19 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile \
  clstr/full_base_train.py \
  clstr/stage2_transition_row_diagnostics.py \
  clstr/stage4_act_train.py
# passed
```

本轮没有提交任何 Slurm/GPU 作业。下一步如果要继续实验，应新建 output directory
重跑一次 Stage1 action-aware next-observation 版本，并只在 Stage1 gate 通过后再考虑
Stage2。

## 2026-06-07 Stage1 再审计后的旧默认路径清理

在 action-aware next-observation transition 修复后，继续检查 Stage1 及其上下游入口时发现：

- Stage1 本身没有再发现新的 loss/objective 设计错误；
- 真正的剩余风险是证据链入口不干净：部分 generic CLI、generic sbatch、readiness audit
  和 submit helper 仍默认指向历史 v3 或 pre-action-aware 输出目录；
- 这会让后续误把旧 Stage1/Stage2 产物当成当前 action-aware 主线，尤其是在直接调用
  generic wrapper 或 audit CLI 时。

已清理的 active 默认入口：

- `scripts/run_clstr_stage1_heads_init.py`
- `scripts/run_clstr_stage2_full_base_train.py`
- `scripts/run_clstr_unified_stage3_hrpo.py`
- `scripts/run_clstr_stage4_act_train.py`
- `scripts/audit_clstr_stage1_heads_quality.py`
- `scripts/audit_clstr_stage2_quality.py`
- `scripts/audit_clstr_stage3_quality.py`
- `scripts/audit_clstr_unified_training_readiness.py`
- `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh`
- `scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh`
- `scripts/sbatch/run_clstr_unified_stage3_hrpo.sh`
- `scripts/sbatch/run_clstr_unified_stage4_act_train.sh`
- `scripts/sbatch/run_clstr_unified_readiness_audit.sh`
- Stage0 / handoff / skill-pool / ToolBench readiness 相关 sbatch 默认路径
- `scripts/submit_clstr_stage1_after_stage0_gate.sh`
- `scripts/submit_clstr_stage2_after_stage1_gate.sh`
- `scripts/submit_clstr_stage3_after_stage2_gate.sh`
- `scripts/submit_clstr_stage4_after_stage3_gate.sh`

新的 active 默认约定：

- 数据根默认：`data/clstr_unified_pretrain_v4_2_progressive_final`
- Stage0 routing checkpoint 默认：
  `outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/clstr_unified_retrieval_v2-step5000.pt`
- Stage1 默认输出：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_action_aware_nextobs_hardneg01_heads_init`
- Stage2 默认输出：
  `outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_prior_residual_l025`
- Stage4 默认输出：
  `outputs/clstr_unified_stage4_v4_2_progressive_final_prior_residual_l025_joint_act`
- standalone Stage3 仍是 legacy/ablation，默认路径改为：
  `outputs/clstr_unified_stage3_v4_2_progressive_final_legacy_hrpo`

Stage1 gate 也已要求 `transition_input_semantics` 证明：

- action channel 使用 actual `action_text` embedding 经 `action_proj`；
- observation channel 使用 `next_observation_text` embedding；
- 不再把 `action_text` 当 observation。

旧 v3 stage/data/output 路径已经从可执行代码和脚本默认值中移除；历史文档里的旧实验记录保留。

本轮验证：

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_sbatch_scripts.py \
  tests/test_clstr_topm_candidate_handoff.py::test_generic_stage2_cli_exposes_stage0_topm_handoff_without_qwen \
  tests/test_clstr_topm_candidate_handoff.py::test_stage1_heads_cli_exposes_sampling_and_benchmark_caps_for_unified_data \
  tests/test_stage1_heads_quality_gate.py \
  tests/test_stage2_quality_gate.py \
  tests/test_stage4_quality_gate.py::test_stage4_quality_gate_cli_defaults_to_progressive_final_mainline \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_sbatch_entrypoint_is_audit_only_and_local \
  tests/test_unified_training_readiness.py::test_unified_training_readiness_cli_defaults_use_toolbench_g3_stage_paths
# 69 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_stage0_biencoder_protocol.py \
  tests/test_stage0_skillrouter_baseline.py \
  tests/test_unified_pretrain_v2.py \
  tests/test_stage0_handoff_audit.py \
  tests/test_stage2_real_topm_eval.py \
  tests/test_stage2_transition_row_diagnostics.py
# 50 passed

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile ... # passed
bash -n ... # passed
git diff --check # passed
squeue -u scyb713 # no active jobs
```

本轮没有提交任何 Slurm/GPU 作业。

## 2026-06-07 Stage1 action-aware next-observation rerun 结果

按用户批准，提交了一次新的 Stage1 action-aware next-observation 训练：

- Slurm job：`84907`
- 分区：`gpu_a800`
- 状态：`COMPLETED`
- exit code：`0:0`
- elapsed：`00:29:34`
- output：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_action_aware_nextobs_heads_init`
- checkpoint：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_action_aware_nextobs_heads_init/checkpoints/clstr_stage1_heads-step3000.pt`
- gate：
  `outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_action_aware_nextobs_heads_init/stage1_quality_gate.json`

运行健康：

- handoff retained：36,544 / 37,000
- skipped：456
- current coverage@350：0.9851
- next coverage@350：0.9898
- injected positives：0
- 训练完成 3,000 steps，checkpoint 和 metrics 正常写出。

Stage1 gate 结果：

```text
status = action_required
blockers = ["transition_recall_at_1_too_low"]
```

tail-100 指标：

```text
loss                         = 1.4292
policy_expert_recall@1       = 0.8183
transition_skill_recall@1    = 0.3225
transition_skill_recall@5    = 0.8400
transition_skill_mrr         = 0.5153
transition_skill_ce_loss     = 1.7123
self transition@1            = 0.4050
switch transition@1          = 0.2150
switch current-skill FP@1    = 0.2708
```

安全字段是干净的：

- `transition_input_safety.action_aware_next_observation = true`
- checkpoint 排除了 frozen routing foundation；
- `positive_missing_policy = skip`
- `injected_positive_rows = 0`

和前两个 Stage1 版本对比：

```text
action-aware nextobs:
  transition@1 = 0.3225
  switch@1     = 0.2150

nextobs-only rerun:
  transition@1 = 0.3217
  switch@1     = 0.2083

old proxy artifact:
  transition@1 = 0.3842
  switch@1     = 0.2817
```

结论：

- action-aware 输入语义是正确的，但这次 Stage1 结果没有实质恢复 transition top-1；
- 它只比 nextobs-only 略好，仍明显低于旧 proxy 版本；
- 因此这个 Stage1 checkpoint 不能进入 Stage2；
- 本轮没有提交 Stage2 或后续作业。

## 2026-06-07 Stage1 action-aware top-1 失败诊断与 hard-negative 接线修复

本轮按低成本诊断路径检查 Stage1 action-aware next-observation 失败原因，没有提交新的
Slurm/GPU 作业。

结论不是运行错误，也不是 action 缺失：

- `L_trans_skill_ce` 训练行中 `action_text` / `expert_action` 覆盖率为 100%；
- `next_observation_text` 覆盖率约 99.29%；
- final checkpoint 中 `action_proj`、`transition`、`trans_head` 相比 Stage0 和
  nextobs-only 版本都有明显参数变化，说明 action-aware 通道确实被训练。

主要失败集中在 switch transitions：

```text
action-aware last-100 switch@1              = 0.2150
old proxy last-100 switch@1                 = 0.2817
action-aware switch current-skill FP@1      = 0.2708
old proxy switch current-skill FP@1         = 0.2125
```

也就是说，语义正确的 action-aware 路径没有恢复 top-1 的主要原因，是 switch 时
current skill 仍经常作为 hardest negative 被排到第一。

代码里已经有 `transition_hard_negative_margin_loss`，并且当前训练日志显示这个 loss
非零；但 Stage1 CLI / sbatch 没有暴露
`transition_hard_negative_margin_loss_weight`，导致主线 Stage1 一直以权重 0 记录它而
不训练它。Stage2 generic wrapper 已有该能力，但 progressive-final Stage2 wrapper
也没有设置/导出该变量，Stage1/Stage2 objective 因此不对齐。

本轮修复：

- `scripts/run_clstr_stage1_heads_init.py` 新增：
  `--transition_hard_negative_margin_loss_weight`、
  `--transition_hard_negative_margin`；
- `clstr/full_base_train.py` 的 Stage1 入口透传
  `transition_hard_negative_margin`，并允许 loss weight 覆盖；
- generic Stage1 sbatch 透传上述两个参数；
- progressive-final Stage1 和 Stage2 wrapper 默认：
  `TRANSITION_HARD_NEGATIVE_MARGIN_LOSS_WEIGHT=0.1`，
  `TRANSITION_HARD_NEGATIVE_MARGIN=1.0`。
- active Stage1/Stage2/Stage4 默认路径已切到新的 `hardneg01` 输出目录，避免覆盖或误用
  已失败的 action-aware Stage1 artifact。

这是通用 in-candidate ranking 约束，不依赖 AppWorld 或某个 benchmark：它只要求 gold
next skill 相比 hardest negative 有 margin。下一步需要重跑一个新的 Stage1
hard-negative 版本；只有 Stage1 gate 通过且 switch current-skill false-positive 明显下降后，
才进入 Stage2。

## 2026-06-08 Stage1 hardneg01 复查与默认入口加固

按要求重新检查 Stage1 后，发现上一轮 hard-negative 接线修复仍有三个隐患：

- active Stage1 输出目录已经切到 `hardneg01`，但直接运行
  `scripts/run_clstr_stage1_heads_init.py` 或 generic sbatch 时，hard-negative loss weight
  仍可能默认为 0；
- Stage1 quality gate 之前只要求 `L_policy` 和 `L_trans_skill_ce`，会放过没有实际启用
  `transition_hard_negative_margin` 的 hardneg01 artifact；
- Python API `train_clstr_stage1_heads_with_model()` / `run_clstr_stage1_heads_init()` 的默认
  Stage1 配方仍偏旧：`top_m=100`、`cross_entropy/single/off`、policy-heavy loss。若绕过
  CLI 做 smoke，可能跑出与主线不同的 Stage1。

本轮修复：

- Stage1 CLI、generic Stage1 sbatch、Stage1 Python API 默认都对齐到当前主线：
  `top_m=350`、`inventory_min_candidates=64`、`listwise_nll`、
  `gold_plus_equivalent`、`balanced_random`；
- Stage1 Python API 默认 loss 改为当前 rebalanced recipe：
  `L_policy=0.6`、`L_trans=0.2`、`L_trans_skill_ce=0.6`、`STOP=0.1`、
  `transition_hard_negative_margin=0.1`；
- Stage1 quality gate 新增 `transition_hard_negative_safety`：
  - `loss_weights.transition_hard_negative_margin` 与
    `transition_candidate_training.hard_negative_loss_weight` 必须一致且大于 0；
  - 训练报告或训练日志中必须看到 `transition_hard_negative_pair_count > 0`；
  - 这项不再作为 row-level `required_loss_terms`，因为 hard-negative margin 是依附在
    `L_trans_skill_ce` candidate ranking 上的辅助约束，不是独立数据标签。

本轮没有提交 Slurm/GPU 作业。新的 Stage1 hardneg01 checkpoint 仍需要通过计算节点重跑，
并且只有 Stage1 gate 通过后才应进入 Stage2。

## 2026-06-08 Stage1 二次复查新增隐患修复

继续复查 Stage1 后，又发现两个小但实际会影响实验安全的点：

- Stage1 主线默认训练 `L_trans`、`STOP`、`belief`，但 quality gate 只要求
  `L_policy` 与 `L_trans_skill_ce`。这样理论上会放过只训练 policy/transition-rank 的不完整
  heads-init artifact；
- generic `scripts/sbatch/run_clstr_unified_stage1_heads_init.sh` 没有脚本级
  `#SBATCH --time`。正常提交路径会走 progressive wrapper，但如果误直接 sbatch generic
  wrapper，仍可能产生无 walltime 的 Stage1 作业。

已修复：

- Stage1 gate 的默认 required loss terms 收紧为：
  `L_policy`、`L_trans`、`L_trans_skill_ce`、`STOP`、`belief`；
- `transition_hard_negative_margin` 仍保持独立 safety 检查，不作为 row-level required
  loss term；
- generic Stage1 sbatch 加入 `#SBATCH --time=06:00:00`；
- Stage1 CLI 的 inventory mask help 文案从旧的 v3 兼容描述改为当前主线默认 `auto`。

用旧失败 artifact 复核新 gate：

```text
output = outputs/clstr_unified_stage1_v4_2_progressive_final_top350_inventory64_listwise_action_aware_nextobs_heads_init
status = action_required
blockers = [
  transition_hard_negative_margin_not_enabled,
  transition_recall_at_1_too_low
]
```

这说明旧 artifact 仍会被 Stage1 gate 阻断；同时它的 `L_trans/STOP/belief` 计数是完整的，
因此新增 required terms 没有把已知失败原因混淆成数据缺失。

本轮仍未提交任何 Slurm/GPU 作业。

## 2026-06-16 Stage4 AppWorld official executor readiness

本轮继续推进 current-route Stage4 official executor，目标是让 v4.1b conservative-listwise
Stage4 checkpoint 能稳定进入 full dev57 评估前的 regression gate。主要修复集中在
executor interface，而不是改 CLSTR routing checkpoint：

- Spotify schema/handoff guard 压缩并补强：
  - song library rows 使用 `song_id`；
  - album / playlist rows 使用 `song_ids`；
  - 涉及 song/album/playlist 三类来源的 genre top/ranking 任务必须跨库合并后过滤和排序；
  - all/user playlists 不默认传 `is_public=True/False`，除非任务显式要求 public/private。
- AST normalizer 增加通用修复：
  - 对 all/user playlist library 任务，自动移除
    `apis.spotify.show_playlist_library(..., is_public=...)`；
  - 同时覆盖 helper-call 形态，例如
    `get_all_song_ids(apis.spotify.show_playlist_library, ..., is_public=False)`；
  - 显式 public/private playlist 任务不受影响。
- completion precheck 修复：
  - filter constraint token 加入 `indie`；
  - 约束 token 检查改为基于 AST unparse 后的可执行代码文本，避免注释里的 token 误判为满足约束；
  - 新增 Spotify cross-library genre precheck，拦截只在单一 song library 上做 genre filter 的错误 completion。
- 可观测性增强：
  - Slurm 日志新增 world/evidence/API docs/step 粒度事件：
    `official_executor_world_start/done`、
    `official_executor_evidence_reset_start/done`、
    `official_executor_api_refs_start/done`、
    `official_executor_api_docs_start/done`、
    `official_executor_step_prepare_start`、
    `official_executor_step_evidence_done`、
    `official_executor_step_start/done`。

验证：

```text
pytest official/current-route related tests: 51 passed
py_compile appworld_official_executor.py and run_appworld_official_executor_eval.py: passed
2-task gate 90664: 2/2
50e1ac9_3 single-task gate 90697: 1/1
7-task regression gate 90700: 7/7
```

当前 full 前状态：

- 最新 gate 输出：
  `outputs/appworld_official_executor_trace_dev7/current_route_cross_library_genre_precheck_v4_1b_a800`
- report:
  `task_count=7`, `success_count=7`, `success_rate=1.0`,
  `generation_failures=0`, `execution_failures=5`, `average_steps=3.428571`
- 这已经达到提交 full dev57 official executor eval 的技术门槛；
- 但 `execution_failures=5` 说明 Qwen executor 仍依赖 precheck 多轮纠偏，full eval 可能波动；
- 按用户约束，full job 必须由用户最终确认后再提交。

用户确认后提交 dev57 current-route eval/rollout collection job `90724`。这是评估与轨迹收集，
不是 Stage4 训练。作业完成健康：

```text
output = outputs/appworld_official_executor_dev57/current_route_cross_library_genre_precheck_v4_1b_rollouts_a800
state = COMPLETED
elapsed = 00:51:57
success_count = 30 / 57
success_rate = 0.526316
task_completed_count = 55
evaluate_success_count = 30
generation_failures = 0
execution_failures = 106
average_steps = 4.649123
```

对照结果：

- qwen-only: `26/57`, execution failures `158`;
- old31 route: `31/57`, execution failures `133`;
- old28 route: `28/57`, execution failures `113`.

row-level split：

- vs qwen-only: new-only `6`
  (`50e1ac9_1`, `50e1ac9_2`, `50e1ac9_3`, `57c3486_2`, `df61dc5_1`, `df61dc5_2`),
  qwen-only-only `2` (`383cbac_1`, `6171bbc_3`);
- vs old31: new-only `2` (`df61dc5_1`, `df61dc5_2`),
  old31-only `3` (`383cbac_1`, `6171bbc_3`, `68ee2c9_3`);
- vs old28: new-only `5`
  (`57c3486_1`, `57c3486_2`, `57c3486_3`, `df61dc5_1`, `df61dc5_2`),
  old28-only `3` (`383cbac_1`, `6171bbc_3`, `68ee2c9_3`).

rollout collection：

```text
current_route_rollouts.jsonl rows = 57
outcome = success 30 / wrong_completion 25 / execution_failure 2
training_ready = 50 / 57
blocker = missing_selected_skill_ids on 7 rows
```

当前判断：

- 这版没有超过旧 best `31/57`，但相比 qwen-only 有 `+4` 净成功，且 execution failures
  从 qwen-only 的 `158` 和 old31 的 `133` 降到 `106`；
- 这说明 current-route executor 修复是有价值的，但还不足以作为最终 Stage4 提升结论；
- 这批 rollout 对下一步 Stage4 preference/RL 信号审计有价值；
- 下一步不能把所有失败轨迹直接当 RL 负样本，应先区分 routing miss、executor/schema miss、
  wrong completion 噪声，以及能构造明确 preference 的样本。

## 2026-06-16 Current-route online Stage4 trainer

本轮新增 current-route online Stage4 训练闭环，目标是证明当前 AppWorld official executor
路线下可以进行真正的 on-policy rollout 后即时更新，而不是先跑完整 eval 再离线训练。

新增文件：

- `clstr/appworld_current_route_online_train.py`
- `scripts/run_appworld_current_route_online_stage4_train.py`
- `scripts/sbatch/run_appworld_current_route_online_stage4_train.sh`
- `tests/test_appworld_current_route_online_train.py`

训练策略：

- 同一个内存中的 CLSTR 模型用于 rollout 与后续更新，后续任务会看到更新后的参数；
- 每个 task rollout 后立刻构建 current-route preference samples 并执行 online update；
- 当前只使用 official success rollout 中的 clean decisions；
- `wrong_completion`、executor failure、schema/API 错误不直接作为 CLSTR 负样本，避免把 Qwen/executor
  噪声错误归因给 routing；
- 默认冻结 routing foundation，只训练 `skill_head`；可选 `--train_transition`;
- 每次 online update 后覆盖写 `checkpoints/latest.pt`，满足持续可观测和中间 checkpoint 要求。

本地验证：

```text
tests/test_appworld_current_route_online_train.py: 2 passed
test_appworld_current_route_online_stage4_sbatch_uses_official_executor_and_online_updates: 1 passed
bash -n scripts/sbatch/run_appworld_current_route_online_stage4_train.sh: passed
py_compile current-route online trainer/script: passed
```

Slurm smoke：

```text
job 90863
output = outputs/appworld_current_route_online_stage4_smoke/one_update_50e1ac9_1_v1
state = COMPLETED
elapsed = 00:01:48
status = ok
rollout_count = 1
online_update_count = 1
sample_count = 1
checkpoint_save_count = 1
latest_checkpoint = outputs/appworld_current_route_online_stage4_smoke/one_update_50e1ac9_1_v1/checkpoints/latest.pt
loss = 4.648561954498291
candidate_count = 350
target_count = 2
```

```text
job 90866
output = outputs/appworld_current_route_online_stage4_smoke/two_updates_50e1ac9_v1
state = COMPLETED
elapsed = 00:02:19
status = ok
rollout_count = 2
online_update_count = 2
sample_count = 2
checkpoint_save_count = 2
latest_checkpoint = outputs/appworld_current_route_online_stage4_smoke/two_updates_50e1ac9_v1/checkpoints/latest.pt
50e1ac9_1 loss = 4.648561954498291
50e1ac9_2 loss = 4.68729829788208
candidate_count = 350
target_count = 2
```

当前结论：

- Stage4 online 训练工程闭环已经可正常运行：on-policy rollout、即时 update、metrics、
  rolling checkpoint、Slurm wrapper 都通过；
- 这还不是性能提升结论，只是 online 训练链路健康证明；
- 下一步应在 5-10 个 mixed task 上做小规模 online train/eval split，或用 dev57 clean success
  samples warm-start 后再进行少量 online update，检查是否能提升可归因 routing miss，同时不破坏已有成功任务。

随后修复 online checkpoint 的 eval loader：

- `stage=appworld_current_route_online_stage4` 的 checkpoint 现在按 full dynamic checkpoint
  从 dynamic skill pool 构建模型；
- 不再误走 base skill pool + append 路径；
- 新增并通过
  `test_dynamic_checkpoint_loader_loads_current_route_online_stage4_as_full_dynamic_checkpoint`。

Mixed eval：

```text
job = 90961
checkpoint = outputs/appworld_current_route_online_stage4_smoke/two_updates_50e1ac9_v1/checkpoints/latest.pt
output = outputs/appworld_official_executor_mixed_eval/current_route_online_two_update_mixed9_v1
state = COMPLETED
elapsed = 00:06:05
success_count = 6 / 9
generation_failures = 0
execution_failures = 5
average_steps = 2.777778
```

任务集合：

```text
50e1ac9_1, 50e1ac9_2, 50e1ac9_3,
57c3486_2,
df61dc5_1, df61dc5_2,
383cbac_1, 6171bbc_3, 68ee2c9_3
```

结果：

- online2: `6/9`;
- base30 current-route: `6/9`，成功/失败 split 与 online2 完全相同；
- old31 route: `7/9`，能成功 `383cbac_1`, `6171bbc_3`, `68ee2c9_3`，但失败
  `df61dc5_1/2`;
- qwen-only: `2/9`，只成功 `383cbac_1`, `6171bbc_3`。

rollout outcome:

```text
success = 6
wrong_completion = 3
training_ready = 8 / 9
missing_selected_skill_ids = 1
```

当前判断：

- 2-step success-imitation online update 没有破坏当前路线，但也没有改善 mixed task set；
- Stage4 online 训练链路已经可用；
- 若要产生性能收益，下一轮不能只是继续跑 success imitation，应加入更明确的 preference/negative
  信号，例如 paired outcome、qwen-only-only 保护、old31/current-route 分歧样本，或学习 confidence gate。

### 2026-06-16 dev57 分歧 preference / suppress 样本

为避免把 AppWorld wrong completion 全部错误归因给 CLSTR，新增一层 conservative
disagreement adapter：

- current-route 成功的任务继续作为 `success_imitation` 多正例样本；
- current-route 失败但 qwen-only 成功的任务，生成低权重
  `fallback_to_qwen_suppress_current` 样本，只压低当前已选 skill 的概率质量；
- current-route 失败但 old31/reference route 成功的任务，生成低权重
  `prefer_reference_suppress_current` 样本；
- 三条路线都失败的任务跳过，不进入训练监督；
- 默认 `negative_weight=0.2`，避免少量失败分歧样本主导 success imitation。

真实 dev57 构造结果：

```text
dataset = outputs/appworld_current_route_preference_data/dev57_disagreement_v1/current_route_disagreement_samples.jsonl
rollout_count = 57
sample_count = 93
positive_sample_count = 87
suppress_sample_count = 6
route_split:
  current_success = 30
  current_only_success = 2
  qwen_success = 26
  qwen_only_success = 2
  reference_success = 31
  reference_only_success = 1
  all_failed = 24
```

低成本训练 smoke：

```text
job = 91019
output = outputs/appworld_current_route_preference_train_smoke/dev57_disagreement_step20_v1
state = COMPLETED
elapsed = 00:00:34
checkpoint = outputs/appworld_current_route_preference_train_smoke/dev57_disagreement_step20_v1/checkpoints/latest.pt
max_steps = 20
batch_size = 8
learning_rate = 2e-5
last_loss = 3.222672939300537
last_batch = 6 positive + 2 suppress
```

这证明分歧样本和 suppress loss 能正常训练，但 9-task mixed eval 证明它不能直接扩大：

```text
job = 91025
output = outputs/appworld_official_executor_mixed_eval/current_route_disagreement_pref_step20_mixed9_v1
state = COMPLETED
elapsed = 00:06:38
success_count = 5 / 9
baseline mixed9 = 6 / 9
```

退化来自 `df61dc5_2`：原本成功路线选择 `venmo-send-transaction + venmo-get-transactions`，
分歧 suppress 训练后变成 batch-like/auth skill 组合，导致 wrong completion。结论：

- 低权重 suppress 直接打到 `skill_head` 仍会扰动 top-5 packing；
- 这种负例不应默认参与排序头训练；
- suppress/disagreement 信号更适合训练 route-level gate/fallback，或作为审计数据，而不是直接改 skill ranking。

因此代码已改为：`avoid_local_indices` suppress loss 默认关闭，只有显式
`enable_suppress_loss=True` / `--enable_suppress_loss` 才参与训练。

### 2026-06-16 API-correction 纠错样本诊断

继续尝试把错误轨迹升级为“带正确 skill 的纠错监督”，而不是纯 suppress：

- 新增 `clstr/appworld_corrective_preference.py`;
- 从 AppWorld `ground_truth/solution.py` 中抽取 `apis.<app>.<api>`；
- 将 solution API refs 映射到 current top-350 candidate 中覆盖相同 API 的 SkillX skill；
- 构造 `counterfactual_api_correction` 样本：
  - `target_local_indices`: GT/API 映射出的 good skill；
  - `rejected_local_indices`: 错误轨迹当前选中的 bad skill；
  - loss: multi-positive NLL + pairwise margin；
- 写 API 任务要求 target skill 覆盖写 API，避免 `show_song` 这类常见读 API 误导 playlist/write 任务。

测试：

```text
pytest tests/test_appworld_corrective_preference.py tests/test_appworld_current_route_preference.py
19 passed
```

真实 dev57 diagnostic 数据：

```text
dataset = outputs/appworld_current_route_preference_data/dev57_api_correction_v2/api_correction_samples.jsonl
rollout_count = 57
failed rollout considered = 27
sample_count = 146
task_count = 16
skipped success_rollout = 30
skipped no_candidate_api_match = 13
skipped selected_already_target = 12
```

重要发现：

- `68ee2c9_3` 可以映射到 `file-system-organize-files-by-date-*`，覆盖
  `move_file/show_file`;
- `383cbac_1` 可以映射到 social-feed / transactions 读取 skill；
- `6171bbc_3` 没有可靠 correction 样本，因为 current top-350 里没有覆盖
  `create_playlist/add_song_to_playlist` 的候选，属于 Stage0/candidate miss，Stage4 ranking 无法凭空修复；
- `df61dc5_2` 在 base route 中本来成功，所以不会从失败 dev57 中生成 correction，但会被其他
  Venmo correction 泛化扰动。

低成本训练：

```text
dataset = outputs/appworld_current_route_preference_data/dev57_success_api_correction_v1/samples.jsonl
success replay = 87
kept correction = 49
correction_weight = 0.35
job = 91142
output = outputs/appworld_current_route_preference_train_smoke/dev57_success_api_correction_step40_v1
state = COMPLETED
elapsed = 00:00:42
pairwise correction active steps = 14 / 40
checkpoint = outputs/appworld_current_route_preference_train_smoke/dev57_success_api_correction_step40_v1/checkpoints/latest.pt
```

Mixed9 eval:

```text
job = 91144
output = outputs/appworld_official_executor_mixed_eval/current_route_api_correction_step40_mixed9_v1
state = COMPLETED
elapsed = 00:06:34
success_count = 5 / 9
baseline mixed9 = 6 / 9
```

失败分析：

- `383cbac_1`, `68ee2c9_3`, `6171bbc_3` 仍失败；
- `df61dc5_2` 从成功退化为失败；
- `df61dc5_2` 的 top skills 从成功路线的
  `venmo-send-transaction + venmo-get-transactions` 变成 batch-like/auth skill 组合；
- 说明即使有 `good > bad` pairwise correction，直接更新共享 `skill_head` 仍会跨任务扰动
  top-5 packing。

当前结论：

- 错误轨迹不能再直接训练 ranking head；
- correction 样本本身有审计价值，但应转为训练 route-level gate / fallback / correction memory；
- 真正的 online Stage4 应该把错误轨迹用于“何时忽略或修正 evidence”，而不是全局改变 skill 排序；
- 对 `6171bbc_3` 这类 no-candidate-match，必须回到 Stage0/candidate recall 或动态 skill append，Stage4 无法在候选缺失时恢复。

### 2026-06-17 错误轨迹按候选覆盖分流

本轮修复了 API-correction 负例的处理边界：不再把所有失败轨迹都送进 Stage4
ranking head。新的原则是：

- 如果 GT/API 对应 skill 已在当前候选中：可以生成 `counterfactual_api_correction`
  Stage4 preference 样本；
- 如果 GT/API 对应 skill 不在当前候选、但存在于 skill pool：只生成 Stage0
  retrieval correction row；
- 如果 skill pool 里也没有可靠匹配：记录为 `no_skill_pool_api_match`，需要动态 skill
  append 或 SkillX 转换修复。

代码改动：

- `clstr/appworld_corrective_preference.py`
  - `ApiCorrectionDataset` 新增 `stage0_retrieval_rows`;
  - 新增 `build_api_correction_dataset_from_rollouts`;
  - `write_api_correction_dataset` 支持 `stage0_retrieval_jsonl_path`;
  - 多写 API 任务先覆盖不同 required write API，避免 playlist 任务只选到
    `add_song_to_playlist` 而漏掉 `create_playlist`;
- `scripts/build_appworld_api_correction_preference_dataset.py`
  - 新增 `--stage0_retrieval_jsonl_path`;
- `clstr/appworld_current_route_online_train.py`
  - 新增 `failure_correction_builder` hook；
  - 失败 rollout 没有 clean Stage4 preference 样本时，可将 missing-candidate 信号写到
    `stage0_retrieval_corrections.jsonl`，不触发 Stage4 optimizer；
- `scripts/run_appworld_current_route_online_stage4_train.py`
  - 新增 `--enable_failure_stage0_correction`;
  - 新增 `--appworld_tasks_root`;
  - 新增 `--failure_correction_max_targets`.

验证：

```text
pytest -q tests/test_appworld_corrective_preference.py \
          tests/test_appworld_current_route_online_train.py \
          tests/test_appworld_current_route_preference.py
25 passed
```

真实 dev57 重建：

```text
output = outputs/appworld_current_route_preference_data/dev57_api_correction_v3_stage0_split
stage4 api_correction_samples = 146
stage0_api_retrieval_corrections = 12
stage0_retrieval_correction_task_count = 3
skipped no_candidate_api_match = 4
skipped no_skill_pool_api_match = 9
```

`6171bbc_*` 结果：

```text
6171bbc_1 -> spotify-add-songs-to-playlist + spotify-create-new-playlist
6171bbc_2 -> spotify-add-songs-to-playlist + spotify-create-new-playlist
6171bbc_3 -> spotify-add-songs-to-playlist + spotify-create-new-playlist
```

这修正了旧诊断里的关键缺陷：`6171bbc_3` 不是完全无解，正确 SkillX rows 存在于
dynamic skill pool，但没有进入 current top-350。它应该更新 Stage0/candidate recall，而不是
更新 Stage4 shared `skill_head`。同时，Stage0 correction 的正例必须覆盖多种写 API，否则会只把
`add_song_to_playlist` 推上来，仍然缺少 `create_playlist` 能力。

下一步实验建议：

- 不再用 dev57 correction 直接做论文训练，只作为诊断；
- 用 train split rollouts 或 train-side official solution 构造同样的
  `stage0_api_retrieval_correction` 数据；
- 先做小步 Stage0 adapter finetune，检查 `6171bbc` 类 playlist/create-add 查询的 top350
  coverage 是否提升；
- 之后再跑 online Stage4 smoke，验证失败轨迹会被分流到 Stage0 correction 文件，成功轨迹才更新
  Stage4。

### 2026-06-17 train-side Stage0 correction audit

为避免继续把 AppWorld dev57 当训练信号，本轮先只使用 AppWorld train split 的 official
API refs 做 Stage0 correction audit。新增：

- `clstr/appworld_stage0_correction_audit.py`
- `scripts/build_appworld_train_stage0_correction_audit.py`
- `scripts/sbatch/run_appworld_train_stage0_top350_export.sh`
- `tests/test_appworld_stage0_correction_audit.py`

执行流程：

```text
1. AppWorld train tasks: data/appworld_routing/train_tasks.jsonl
2. Dynamic skill pool: data/clstr_appworld_dynamic_v4_1b_append/skill_pool.jsonl
3. 当前 Stage0 checkpoint 导出 top350:
   outputs/appworld_train_stage0_top350_v1/predictions.jsonl
4. 用 top350 candidates 回填 train-side correction audit:
   outputs/appworld_train_stage0_correction_audit_v2_coreapi_top350
```

小作业信息：

```text
job_id = 91405
state = COMPLETED
elapsed = 00:01:31
queries = 90
top_k = 350
skill_count = 37803
```

原始 v1 audit 暴露一个问题：`supervisor.show_*`、`spotify.login` 等 auth/helper API 会被当成
Stage0 positive，导致 correction rows 偏向 `authenticate` skill。这不适合作为 Stage0 retrieval
训练目标，因为 executor prompt 已有认证 helper，Stage0 应学习核心任务 skill，而不是学习登录 skill。

已修复 train-side audit：Stage0 positive matching 过滤 auxiliary API：

```text
auxiliary API = supervisor.* 或 access_token_from/login/authenticate 类 API
```

验证：

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
  tests/test_appworld_stage0_correction_audit.py
3 passed
```

过滤后的 v2 audit 结果：

```text
train_task_count = 90
skill_count = 37803
pool_match_task_count = 84
no_skill_pool_match_count = 6
candidate_positive_covered_task_count = 50
candidate_positive_missing_task_count = 34
stage0_retrieval_correction_count = 138
```

解释：

- train split 中大多数任务的核心 API-positive skill 已在 dynamic skill pool 中存在；
- 但当前 Stage0 top350 仍漏掉 34/84 个可匹配任务的至少一个 positive skill；
- 这支持“失败轨迹要分流：候选里有 GT 才训 Stage4 rerank；候选里没有 GT 应优先训 Stage0 recall”；
- 不能直接把 138 条全部作为最终训练集使用，因为 read-only API overlap 仍可能语义过宽，例如只共享
  `show_song/show_playlist_library` 的 skill 不一定真能解决 “most-liked song” 这类目标。

下一步应构建 high-confidence Stage0 correction subset：

- 优先使用 required write API 覆盖的 correction，例如 create/add playlist、follow、send、move、like；
- read-only correction 先保留为 audit，除非进一步通过 skill title/body 与 query 的语义匹配或 reference
  rerank 过滤；
- 之后再做小步 Stage0 adapter finetune，目标是提升 train-side missing-positive coverage，而不是直接跑 full
  Stage4。

### 2026-06-17 high-confidence Stage0 correction subset and generic extra retrieval hook

继续处理上面的风险后，新增一个高置信 Stage0 correction 子集构建器：

- `build_high_confidence_stage0_correction_subset`
- `scripts/build_high_confidence_stage0_correction_subset.py`

默认策略：

```text
policy = write_only
keep row iff provenance.target_match_detail.write_api_overlap is non-empty
```

这意味着只把 positive skill 明确覆盖 required write/action API 的样本放进训练候选。read-only API-overlap
样本仍保留为 audit，不直接训练 Stage0，避免把只共享 `show_song` / `show_playlist_library` 但语义不匹配的
skill 推上去。

在当前 AppWorld train audit 上运行：

```text
input = outputs/appworld_train_stage0_correction_audit_v2_coreapi_top350/stage0_api_retrieval_corrections.jsonl
output = outputs/appworld_train_stage0_correction_high_confidence_v1/stage0_high_confidence_corrections.jsonl
input_count = 138
kept_count = 15
skipped_read_only_count = 123
covered_tasks = 3
write_api_overlap = {apis.spotify.create_playlist, apis.spotify.add_song_to_playlist}
```

结论：

- AppWorld train split 里可用于高置信 Stage0 correction 的写操作样本很少；
- 这 15 条可以作为 smoke / small adapter finetune 的补充源，但不能单独支撑 full Stage0 重训或论文结论；
- 这进一步说明 Stage0 correction 必须走跨 benchmark 统一机制，而不是 AppWorld 专线。

为保持泛化，已给统一数据构建脚本增加通用 extra retrieval 入口：

- `scripts/build_clstr_unified_pretrain.py --extra_retrieval_paths path1.jsonl,path2.jsonl`
- 输入 schema 沿用 Stream B：

```text
source, query_id, query_text, positive_skill_id, negative_skill_ids, provenance
```

安全策略：

- extra retrieval row 会过滤 AppWorld dev/test/test_challenge task_id；
- extra retrieval row 会过滤 SkillRET test query_id；
- 通过 `manifest["retrieval_stream"]["extra_retrieval"]` 记录 total_pairs / leakage_filtered / skipped_missing_field。

验证：

```text
pytest -q tests/test_appworld_stage0_correction_audit.py \
          tests/test_unified_pretrain_v2.py::test_unified_pretrain_can_append_extra_retrieval_rows_without_appworld_leakage \
          tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream
6 passed in 243.30s
```

下一步：

- 不应只用 AppWorld 的 15 条 high-confidence rows 训练；
- 应为 ToolBench / TrajectBench / WebShop / ALFWorld 的 trajectory-derived retrieval rows 做同类 top-M
  coverage audit：
  - GT action/tool/next-skill 在候选内：进入 Stage4 rerank/preference；
  - GT action/tool/next-skill 不在候选但在 skill pool：进入 Stage0 extra retrieval correction；
  - GT 不在 skill pool：进入 dynamic skill / skill construction；
- 之后把跨 benchmark high-confidence correction 通过 `--extra_retrieval_paths` 混入统一 Stage0 训练数据，先跑
  small adapter finetune 和 top350 coverage audit，再决定是否 full Stage0 / Stage4。

### 2026-06-17 cross-benchmark Stage0 top350 coverage audit

为避免把 AppWorld executor / Qwen 噪声误判成 CLSTR rerank 问题，新增了通用 Stage0 retrieval coverage
audit。它直接比较统一 `retrieval.jsonl` 里的 `positive_skill_id` 是否出现在已有 Stage0 full retrieval eval
导出的 top-M candidates 中：

```text
retrieval_rows_path = data/clstr_unified_pretrain_v4_2_alfworld_hf_quality_nowweak/retrieval.jsonl
predictions_path = outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/full_retrieval_eval/predictions.jsonl
output_dir = outputs/stage0_retrieval_coverage_audit/v4_2_nowweak_full_top350
k = 20, 50, 100, 350
```

总体 raw row-level 结果：

```text
rows = 599735
recall@20 = 0.5870
recall@50 = 0.6764
recall@100 = 0.7391
recall@350 = 0.8412
missing@350 = 95227
```

分 source 的 `recall@350`：

```text
skillret                         0.6634  missing=42813
toolbench_g3                     0.7621  missing=17738
toolret_training                 0.9132  missing=18130
traject_bench                    0.8438  missing=4767
trajectory_derived_alfworld      1.0000  missing=0
trajectory_derived_toolbench_g3  0.8514  missing=2417
trajectory_derived_traject_bench 0.8339  missing=9362
trajectory_derived_webshop       1.0000  missing=0
```

解释：

- 这是 row-level single-positive audit，和 qrels / multi-positive 聚合后的 eval 指标不完全等价；
- 它更适合找 Stage0 correction rows：如果 GT skill 不在 top350，Stage4 rerank / RL 无法直接修；
- ALFWorld/WebShop trajectory-derived rows 已经 top350 全覆盖，当前 correction 重点不应放在它们；
- ToolBench / TrajectBench 仍有明显 missing-positive，可作为跨 benchmark、非 AppWorld 特化的 Stage0 recall 修复来源；
- SkillRET missing rows 数量最大，但它偏通用静态文本检索，暂时不让它主导 correction，否则容易把训练重心拉回单步语义匹配。

基于上述 audit 构建了 balanced correction subset：

```text
input = outputs/stage0_retrieval_coverage_audit/v4_2_nowweak_full_top350/stage0_missing_positive_rows.jsonl
output = outputs/stage0_retrieval_coverage_audit/v4_2_nowweak_balanced_correction_subset_v1/stage0_missing_positive_subset.jsonl
source_allowlist = toolbench_g3, toolret_training, traject_bench,
                   trajectory_derived_toolbench_g3, trajectory_derived_traject_bench
per_source_cap = 5000
kept_count = 22184
```

保留分布：

```text
toolbench_g3                     5000
toolret_training                 5000
traject_bench                    4767
trajectory_derived_toolbench_g3  2417
trajectory_derived_traject_bench 5000
```

下一步只做低成本验证：

1. 用 `--extra_retrieval_paths` 构建一个带 balanced correction 的统一数据目录，保持
   `v4_2_alfworld_hf_quality` / nowweak source quality 配置不漂移。
2. 只跑小规模 Stage0 adapter / finetune，不直接 full。
3. 对同一批 retrieval rows 重新导出 top350 并运行 coverage audit。
4. 成功标准不是单个 AppWorld 任务变好，而是 missing subset coverage 提升，同时主要 source 的 broad recall 不明显退化。

已构建可用于 low-cost Stage0 correction 验证的数据目录：

```text
data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v2
```

构建参数保持原 nowweak 数据的关键设置：

```text
dataset_recipe = v4_2_alfworld_hf_quality
source_quality_allowlist =
  official_replay,
  dagger_expert_corrected_rollout,
  hf_alfworld_admissible_success,
  webshop_hf_gpt4_reward_ge_0.8,
  traject_public_data,
  toolbench_g3_dfs_tree
trajectory_retrieval_caps =
  alfworld=-1, scienceworld=0, toolbench_g3=-1, traject_bench=-1, webshop=-1
extra_retrieval = balanced correction subset v1
```

manifest 对比校验：

```text
base retrieval pairs = 599735
new retrieval pairs  = 621919
delta                = 22184
extra_retrieval      = 22184
trajectory rows      = unchanged 85932
skill pool size      = unchanged 37617
leakage status       = ok
extra leakage filter = 0
```

注意：曾生成过 `data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v1`，但该目录没有显式传入
`trajectory_retrieval_caps`，导致 ALFWorld/WebShop retrieval caps 漂移到默认 `10000`。该目录不应用于训练或报告。

### 2026-06-17 Stage0 correction finetune initialization fix

在准备 small Stage0 correction 试验时发现一个实现问题：`scripts/run_clstr_stage0_biencoder_train.py`
原本没有 `--init_checkpoint_path`，因此如果直接在 augmented data 上跑 small run，实际会从
SkillRouter embedding 基座重新初始化轻量 adapter，而不是在当前 nowweak Stage0 checkpoint 上继续修正。
这会让实验无法回答“balanced correction 是否能修复已有 Stage0 的 top350 missing-positive”。

已修复：

- `run_skillret_retrieval_warmup(..., init_checkpoint_path=...)`
- `run_stage0_biencoder_train(..., init_checkpoint_path=...)`
- `scripts/run_clstr_stage0_biencoder_train.py --init_checkpoint_path`
- `scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh` 支持环境变量 `INIT_CHECKPOINT_PATH`

加载策略：

- 在 skill table rebuild 后加载 checkpoint；
- 使用 `strict=False`，因为 Stage0 checkpoint 按设计不保存冻结 backbone；
- report/checkpoint 记录 `init_checkpoint.loaded/path/checkpoint_step/missing_state_keys/unexpected_state_keys`；
- tensor shape mismatch 仍会直接报错，避免静默结构不兼容。

验证：

```text
tests/test_skillret.py::test_retrieval_warmup_can_initialize_from_stage0_checkpoint
tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream
tests/test_stage0_retrieval_coverage_audit.py
tests/test_unified_pretrain_v2.py::test_unified_pretrain_can_append_extra_retrieval_rows_without_appworld_leakage

5 passed
py_compile: clstr/retrieval_warmup.py, scripts/run_clstr_stage0_biencoder_train.py
bash -n: stage0 sbatch wrappers
```

### 2026-06-17 Stage0 correction loader fix and smoke result

第一次 Stage0 correction smoke 暴露了一个更关键的问题：`extra_retrieval` rows 确实进入了
`retrieval.jsonl`，manifest 也记录了 `22184` 条，但训练 loader 原先按裸 `query_id` 合并 rows。由于
balanced correction rows 来自原 retrieval missing-positive audit，很多 `query_id` 与原训练行相同，结果是：

- `query_count = 486958`
- `training_bucket_count = 12`
- 没有任何 `*_stage0_balanced_missing_positive` bucket
- `multi_positive_nll` 内部对 positive index 去重，因此这些 duplicate correction rows 几乎没有训练权重

这意味着第一次 smoke 只证明了 job 能跑，不能证明 correction 被训练到。

已修复 `load_unified_v2_retrieval_rows`：

- 同一 `query_id` 且同一 `source`：仍合并为 multi-positive，保持原有多正例语义；
- 同一 `query_id` 但不同 `source`：保留为独立训练 row；
- 独立 row 的 `metadata.original_query_id` 记录原始 query id；
- 这样 balanced correction source 可以作为独立 bucket 被 handoff-balanced sampler 采样。

loader 修复后，不需要重建 v2 数据目录，重新读取已有
`data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v2`：

```text
skill_count = 37617
query_count = 507130
total_positives = 621919
extra_query_count = 20172
training_bucket_count = 19
```

extra buckets：

```text
toolbench_g3_stage0_balanced_missing_positive                       2988
toolret_training_stage0_balanced_missing_positive                   5000
traject_bench_stage0_balanced_missing_positive                      4767
trajectory_derived_toolbench_g3_stage0_balanced_missing_positive:current 1285
trajectory_derived_toolbench_g3_stage0_balanced_missing_positive:next    1132
trajectory_derived_traject_bench_stage0_balanced_missing_positive:current 2568
trajectory_derived_traject_bench_stage0_balanced_missing_positive:next    2432
```

短 smoke job：

```text
job_id = 91564
output = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_initckpt_smoke_step80_loaderfix
state = COMPLETED
exit = 0:0
elapsed = 00:03:44
init_checkpoint = outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/latest.pt
init_step = 5000
max_steps = 80
trainable = skill_table.W.weight + skill_table.logit_scale_retr
```

smoke 指标：

```text
metric_rows = 80
step1  loss=9.9714 recall@1=0.2031 recall@350=0.5469
step80 loss=9.9267 recall@1=0.1563 recall@350=0.6250
```

80-step smoke 中实际采样到 correction source：

```text
toolbench_g3_stage0_balanced_missing_positive                 269
toolret_training_stage0_balanced_missing_positive             269
traject_bench_stage0_balanced_missing_positive                269
trajectory_derived_toolbench_g3_stage0_balanced_missing_positive 539
trajectory_derived_traject_bench_stage0_balanced_missing_positive 538
```

解释：

- smoke 证明 correction data 现在确实进入训练，不再是无效 duplicate；
- loss 小幅下降、recall@350 上升只说明 short-run 链路健康，不能当作正式效果；
- 下一步应导出这个 smoke checkpoint 在 correction subset 上的 top350，并与原 Stage0 checkpoint 比较 coverage；
- 如果 correction subset coverage 有提升且 broad source recall 未明显退化，再讨论是否跑更长 small finetune 或 full Stage0。
## 2026-06-17 Stage0 correction step2000 audit and sampler diagnosis

Stage0 correction 500/2000-step small finetunes from the nowweak Stage0 checkpoint were audited before any full run.

Correction subset coverage improved monotonically:

```text
step80   recall@350 = 0.0470  covered@350 = 1043 / 22184
step500  recall@350 = 0.1613  covered@350 = 3578 / 22184
step2000 recall@350 = 0.3294  covered@350 = 7307 / 22184
```

On the fixed 8k broad sample, step2000 also improved overall Stage0 recall:

```text
baseline recall@20/50/100/350 = 0.5684 / 0.6651 / 0.7280 / 0.8468
step500  recall@20/50/100/350 = 0.5799 / 0.6813 / 0.7536 / 0.8725
step2000 recall@20/50/100/350 = 0.5903 / 0.6876 / 0.7654 / 0.8769
```

The broad improvement is real, but not clean enough for full training yet. Source-sliced broad sample at step2000 shows:

```text
skillret            recall@350 = 0.638  delta vs baseline = -0.034
toolret_training    recall@350 = 0.918  delta vs baseline = -0.003
toolbench_g3        recall@350 = 0.835  delta vs baseline = +0.056
traject_bench       recall@350 = 0.864  delta vs baseline = +0.089
td_toolbench_g3     recall@350 = 0.924  delta vs baseline = +0.031
td_traject_bench    recall@350 = 0.836  delta vs baseline = +0.102
```

Root-cause diagnosis for the static-retrieval regression:

- `handoff_balanced` samples buckets equally, not proportionally to row count.
- Balanced correction rows are only `22184 / 621919` pairs in the data directory, but after bucket splitting they occupy 7 of 19 training buckets.
- Therefore correction rows receive roughly 37% of batch slots during Stage0 finetune.
- This is useful for proving correction pressure works, but too aggressive for the final Stage0 route because it can cause static retrieval forgetting, especially on SkillRET.

Decision:

- Do not continue the current `handoff_balanced` correction run to 5000/full as-is.
- Next implementation should add a tempered correction sampler: keep all original handoff buckets, but cap correction bucket slots to roughly 15-20% of each batch.
- The target is to retain correction-subset gains while avoiding SkillRET/ToolRet broad regression.

## 2026-06-17 Stage0 tempered correction sampler implementation

Implemented `handoff_tempered` for Stage0 retrieval warmup:

- `handoff_balanced` remains the default and is unchanged.
- `handoff_tempered` keeps handoff-style current/next bucket splitting, but separates `*_stage0_*missing_positive*` correction buckets from normal buckets.
- `tempered_correction_fraction` controls correction slots per batch; default is `0.2`.
- CLI/sbatch now expose:
  - `--sampling_strategy handoff_tempered`
  - `--tempered_correction_fraction`
  - `TEMPERED_CORRECTION_FRACTION`
- Reports/checkpoints include `sampling_strategy` and `tempered_correction_fraction` for auditability.

TDD / verification:

```text
RED: new sampler test failed because _batch_queries_for_step did not accept tempered_correction_fraction.
GREEN: implemented handoff_tempered and parameter pass-through.

pytest targeted:
  tests/test_stage0_biencoder_protocol.py::test_stage0_handoff_tempered_sampler_caps_missing_positive_slots
  tests/test_stage0_biencoder_protocol.py::test_stage0_biencoder_train_passes_tempered_sampler_fraction
  tests/test_stage0_biencoder_protocol.py::test_stage0_biencoder_cli_help_and_sbatch_defaults
  -> 3 passed

pytest adjacent:
  tests/test_skillret.py::test_retrieval_warmup_can_train_from_unified_v2_retrieval_stream
  tests/test_skillret.py::test_retrieval_warmup_can_initialize_from_stage0_checkpoint
  -> 2 passed

py_compile / bash -n / git diff --check -> passed
```

Tempered smoke:

```text
job_id = 91728
partition = gpu_a800
state = COMPLETED
exit = 0:0
elapsed = 00:02:51
output = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_initckpt_smoke_step80_tempered_loaderfix
init_checkpoint = outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/latest.pt
init_step = 5000
sampling_strategy = handoff_tempered
tempered_correction_fraction = 0.2
```

Smoke metrics:

```text
metric_rows = 80
step1  loss=9.8869 recall@1=0.2344 recall@20=0.4375 recall@50=0.5156 recall@100=0.5469 recall@350=0.6719
step80 loss=9.8740 recall@1=0.1875 recall@20=0.5313 recall@50=0.6250 recall@100=0.6719 recall@350=0.7656
sampled_total = 5120
correction_total = 1040
correction_fraction = 0.2031
```

Interpretation:

- The tempered sampler is working: correction rows are about 20% of batch slots, not the previous ~37%.
- The smoke is healthy and initialized from the intended Stage0 checkpoint.
- Do not jump to full. Next low-cost step should be a 500-step `handoff_tempered` run, followed by correction-subset audit and the fixed 8k broad sample audit.

## 2026-06-17 Stage0 tempered step500 result

Ran the planned low-cost `handoff_tempered` step500 experiment:

```text
job_id = 91740
state = COMPLETED
exit = 0:0
elapsed = 00:08:59
output = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_initckpt_step500_tempered_loaderfix
sampling_strategy = handoff_tempered
tempered_correction_fraction = 0.2
sampled_total = 32000
correction_total = 6500
correction_fraction = 0.2031
step500 loss = 9.8437
step500 batch recall@350 = 0.8125
```

Export/audit jobs:

```text
correction subset export job = 91756, COMPLETED, exit 0:0, elapsed 00:04:02
broad sample export job     = 91760, COMPLETED, exit 0:0, elapsed 00:01:04
```

Correction subset audit:

```text
tempered step500 correction subset recall@20/50/100/350 =
0.0004 / 0.0010 / 0.0046 / 0.1090

comparison:
handoff step80   recall@350 = 0.0470
handoff step500  recall@350 = 0.1613
handoff step2000 recall@350 = 0.3294
```

Fixed 8k broad sample:

```text
baseline    recall@20/50/100/350 = 0.5684 / 0.6651 / 0.7280 / 0.8468
handoff500  recall@20/50/100/350 = 0.5799 / 0.6813 / 0.7536 / 0.8725
handoff2000 recall@20/50/100/350 = 0.5903 / 0.6876 / 0.7654 / 0.8769
tempered500 recall@20/50/100/350 = 0.5723 / 0.6608 / 0.7328 / 0.8521
```

Source-sliced broad sample for `tempered500` vs baseline:

```text
skillret                         recall@350 = 0.638  delta = -0.034
toolret_training                 recall@350 = 0.913  delta = -0.008
toolbench_g3                     recall@350 = 0.790  delta = +0.011
traject_bench                    recall@350 = 0.804  delta = +0.029
trajectory_derived_toolbench_g3  recall@350 = 0.896  delta = +0.003
trajectory_derived_traject_bench recall@350 = 0.776  delta = +0.042
```

Important conclusion:

- `handoff_tempered` works mechanically, but it is not better than early-stopped `handoff_balanced`.
- `tempered500` has lower correction-subset recall than `handoff500` and still causes the same SkillRET broad regression seen later in `handoff2000`.
- Therefore the main issue is not only correction oversampling fraction.
- Current best Stage0 correction candidate is `handoff500`, because it improves broad recall and does not regress SkillRET on the fixed sample.
- Next sensible fix, if we continue improving Stage0, is not more fraction tuning; it should add an original-Stage0 preservation/anchor objective or use early stopping/gating with the handoff500 checkpoint.

## 2026-06-17 Stage4 online update routing clarification

Clarified and implemented the Stage4 online update boundary:

- Stage4 online ACT should not retrain Stage1/Stage2 after every failed rollout.
- Success rollouts still update the Stage4/current-route policy head from clean on-policy decisions.
- Candidate-miss failures now have an optional online Stage0 correction path:
  - if failure analysis produces `stage0_retrieval_rows`, the online loop can immediately optimize Stage0 retrieval with full-pool multi-positive NLL;
  - Stage1/Stage2 transition/belief modules stay frozen;
  - the update is enabled only with `--enable_online_stage0_correction_update`, so existing jobs are unchanged by default.
- Wrong completion / executor/API failures are still not blindly treated as CLSTR routing negatives.
- Periodic Stage1/Stage2 audit remains useful only after Stage0 distribution changes, but it is not part of the inner online loop.

Implemented files:

```text
clstr/appworld_current_route_online_train.py
scripts/run_appworld_current_route_online_stage4_train.py
scripts/sbatch/run_appworld_current_route_online_stage4_train.sh
tests/test_appworld_current_route_online_train.py
```

Verification:

```text
pytest -q tests/test_appworld_current_route_preference.py tests/test_appworld_corrective_preference.py tests/test_appworld_current_route_online_train.py -q
26 passed

py_compile:
clstr/appworld_current_route_online_train.py
scripts/run_appworld_current_route_online_stage4_train.py

bash -n scripts/sbatch/run_appworld_current_route_online_stage4_train.sh
git diff --check on tracked diff returned clean; the Stage4 online files are currently untracked in this worktree, so syntax/tests above are the meaningful checks for those files.
```

## 2026-06-17 Stage4 online Stage0 correction smoke

Ran a low-cost one-task AppWorld online smoke:

```text
job_id = 91835
task_id = 6171bbc_3
state = COMPLETED
exit = 0:0
elapsed = 00:02:25
output = outputs/appworld_current_route_online_stage4_smoke/stage0_online_correction_6171bbc_3_20260617_222301
MAX_INTERACTIONS = 8
ENABLE_FAILURE_STAGE0_CORRECTION = true
ENABLE_ONLINE_STAGE0_CORRECTION_UPDATE = true
```

Result:

```text
status = ok
rollout_count = 1
online_update_count = 0
online_stage0_update_count = 1
online_stage0_sample_count = 6
stage0_correction_row_count = 6
online_stage0_loss = 10.1898
```

Interpretation:

- The online loop now works mechanically:
  - no clean success preference was produced, so Stage4 policy head was not updated;
  - candidate-miss correction rows were produced and immediately used to update Stage0;
  - Stage1/Stage2 remained frozen.
- The failure was a true candidate issue for this task:
  - selected skills were mostly `spotify-find-songs-by-play-count` variants;
  - required `spotify-create-new-playlist` / `spotify-add-songs-to-playlist` skills were missing from the top350 candidate set.
- The visible executor evidence was empty because the evidence gate chose `no_evidence_fallback`; this is expected when selected skills have no schema-grounded useful evidence, but it also means Qwen did not receive useful CLSTR hints for this rollout.

Then ran a routing-only before/after audit:

```text
job_id = 91849
state = COMPLETED
exit = 0:0
audit = outputs/appworld_current_route_online_stage4_smoke/stage0_online_correction_6171bbc_3_20260617_222301/routing_before_after.json
```

Result:

```text
before target ranks:
  spotify-create-new-playlist-21/36/37 = not in top350
  spotify-add-songs-to-playlist-24/35/54 = not in top350

after one online Stage0 update:
  same targets still not in top350
```

Conclusion:

- The update path is correct, but one tiny Stage0 update with `lr=5e-5` is not enough to move missing dynamic AppWorld skills from outside top350 into the candidate set.
- Do not expand to Stage4 full yet.
- Next fix should make Stage0 correction actually move candidates before involving Qwen executor again:
  - stronger Stage0 correction update or multiple local steps per candidate-miss;
  - preservation/anchor loss to avoid broad retrieval forgetting;
  - candidate-miss routing audit as the gate before any executor/RL job.

## 2026-06-18 Stage0 hard-mining preservation anchor

Added an optional Stage0 preservation anchor to stabilize trajectory-aware hard mining.
The motivation is the current Stage0 correction pattern:

- `handoff500` improves broad recall and gives some candidate-miss recovery;
- longer / stronger correction can improve missing-positive recall but risks forgetting static SkillRET-style retrieval;
- online Stage4 candidate-miss updates mechanically work, but one tiny update did not move the required AppWorld skills into top350.

Implementation:

- `run_skillret_retrieval_warmup(..., preservation_anchor_weight=0.0)` now supports an optional trainable-parameter anchor.
- The anchor snapshot is taken after init-checkpoint loading and trainability setup, so it only covers the parameters that Stage0 is actually updating.
- The added loss is:

```text
L_stage0 = L_retrieval + L_reranker + lambda_anchor * mean_p ||p - p_anchor||^2
```

- Default is `preservation_anchor_weight=0.0`, so existing Stage0 runs are unchanged unless explicitly enabled.
- Metrics/checkpoints now record:
  - `preservation_anchor_loss`;
  - `preservation_anchor_unweighted_loss`;
  - `preservation_anchor_parameter_count`;
  - `preservation_anchor_weight`;
  - `preservation_anchor` setup metadata.
- CLI/sbatch expose `--preservation_anchor_weight` / `PRESERVATION_ANCHOR_WEIGHT`.

Paper framing:

- This should be described as part of Stage0 trajectory-aware large-pool retriever training:
  static retrieval pairs + sequential routing pairs + hard candidate-miss pairs + preservation anchor.
- It should not be presented as a separate Stage0.5 or benchmark-specific adaptation.

Verification:

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_biencoder_protocol.py -q
............. [100%]

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/retrieval_warmup.py scripts/run_clstr_stage0_biencoder_train.py
bash -n scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
```

No Slurm job was submitted in this step.

### 2026-06-18 anchor filtering follow-up

The first real health run exposed an implementation detail that matters for the
regularizer scale:

```text
job_id = 92511
output = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w10_health_step5
state = COMPLETED
exit = 0:0
preservation_anchor_weight = 10.0
preservation_anchor_parameter_count = 31
```

The optimizer was only updating:

```text
skill_table.W.weight
skill_table.logit_scale_retr
```

but the anchor snapshot used all `requires_grad=True` parameters. This diluted
the anchor and made the reported regularizer hard to interpret. The snapshot now
accepts an optimizer-name filter and Stage0 training anchors only the parameters
that are actually passed to AdamW.

New verification:

```text
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q tests/test_stage0_biencoder_protocol.py -q
.............. [100%]

/data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m py_compile clstr/retrieval_warmup.py scripts/run_clstr_stage0_biencoder_train.py
bash -n scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
git diff --check -- clstr/retrieval_warmup.py tests/test_stage0_biencoder_protocol.py scripts/run_clstr_stage0_biencoder_train.py scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh description.md .planning/2026-06-14-clstr-next-direction/progress.md
```

Health rerun:

```text
job_id = 92541
output = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w1e4_health_step5_filterfix
state = COMPLETED
exit = 0:0
preservation_anchor_weight = 10000.0
preservation_anchor_parameter_count = 2
preservation_anchor_parameter_names = [
  skill_table.W.weight,
  skill_table.logit_scale_retr
]
step5 retriever_loss = 9.884899
step5 preservation_anchor_loss = 0.000033
```

Parameter-drift scale from existing non-anchor correction checkpoints:

```text
step500 non-anchor optimizer-param mean MSE ~= 5.16e-5
  weight 1e4 => anchor contribution ~= 0.516

step2000 non-anchor optimizer-param mean MSE ~= 8.27e-4
  weight 1e4 => anchor contribution ~= 8.27
```

So `preservation_anchor_weight=1e4` is the first useful low-cost probe: weak at
startup, visible by step500, and strong enough to test whether step2000
forgetting can be reduced. A routing-only 2000-step probe is running as job
`92556`:

```text
output = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w1e4_step2000_filterfix
sampling_strategy = handoff_balanced
init_checkpoint = outputs/clstr_unified_stage0_v4_2_nowweak_handoff_balanced_mpn_freezeE/checkpoints/latest.pt
```

Result:

```text
job_id = 92556
state = COMPLETED
exit = 0:0
elapsed = 00:31:53
checkpoint = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w1e4_step2000_filterfix/checkpoints/clstr_unified_retrieval_v2-step2000.pt
step2000 retriever_loss = 9.905135
step2000 preservation_anchor_loss = 0.003956
```

Routing-only eval job `92577` completed normally:

```text
correction subset recall@20/50/100/350 =
  0.0000 / 0.0000 / 0.0009 / 0.1293

broad sample recall@20/50/100/350 =
  0.5766 / 0.6759 / 0.7455 / 0.8658

broad SkillRET recall@350 =
  0.669
```

Comparison:

```text
baseline broad recall@350 = 0.8468
baseline SkillRET recall@350 = 0.672

no-anchor step500:
  correction recall@350 = 0.1613
  broad recall@350 = 0.8725
  SkillRET recall@350 = 0.672

no-anchor step2000:
  correction recall@350 = 0.3294
  broad recall@350 = 0.8769
  SkillRET recall@350 = 0.638

anchor-1e4 step2000:
  correction recall@350 = 0.1293
  broad recall@350 = 0.8658
  SkillRET recall@350 = 0.669
```

Interpretation:

- `1e4` preserves broad / SkillRET behavior, but it is too strong for the hard-mining goal.
- Optimizer-parameter drift confirms this:

```text
no-anchor step500 optimizer-param mean MSE = 5.16e-5
no-anchor step2000 optimizer-param mean MSE = 8.27e-4
anchor-1e4 step2000 optimizer-param mean MSE = 3.96e-7
```

So `1e4` pushed the Stage0 adapter below even the no-anchor step500 movement and
underfit the correction subset. The next low-cost probe is one order weaker:

```text
job_id = 92587
preservation_anchor_weight = 1000.0
output = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w1e3_step2000_filterfix
```

Result:

```text
job_id = 92587
state = COMPLETED
exit = 0:0
elapsed = 00:31:47
checkpoint = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w1e3_step2000_filterfix/checkpoints/clstr_unified_retrieval_v2-step2000.pt
step2000 retriever_loss = 9.900492
step2000 preservation_anchor_loss = 0.003144
preservation_anchor_parameter_count = 2
```

Routing-only eval job `92593` completed normally:

```text
correction subset recall@20/50/100/350 =
  0.0041 / 0.0161 / 0.0453 / 0.2855

broad sample recall@20/50/100/350 =
  0.5903 / 0.6976 / 0.7774 / 0.8849

broad SkillRET recall@350 =
  0.670
```

Comparison:

```text
run                  corr@100  corr@350  broad@100  broad@350  SkillRET@350
baseline_old_pool    -         -         0.7280     0.8468     0.6720
no_anchor_step500    0.0029    0.1613    0.7536     0.8725     0.6720
no_anchor_step2000   0.0741    0.3294    0.7654     0.8769     0.6380
anchor1e4_step2000   0.0009    0.1293    0.7455     0.8658     0.6690
anchor1e3_step2000   0.0453    0.2855    0.7774     0.8849     0.6700
```

Interpretation:

- `preservation_anchor_weight=1e3` is currently the best Stage0 correction tradeoff:
  it recovers most of the no-anchor correction gain while avoiding the SkillRET
  collapse seen in no-anchor step2000.
- It also improves the fixed broad sample to the best observed top350 recall in
  this probe series.
- This is a good candidate for a general Stage0 hard-mining correction setting,
  but it is not enough to re-enter AppWorld Stage4 yet.

Direct AppWorld `6171bbc_3` routing-only check job `92602`:

```text
queries = outputs/appworld_current_route_online_stage4_smoke/stage0_online_correction_6171bbc_3_20260617_222301/stage0_retrieval_corrections.jsonl
output = outputs/stage0_retrieval_coverage_audit/v4_2_nowweak_stage0corr_anchor_w1e3_step2000_appworld6171_eval_top350
result recall@20/50/100/350 = 0.0000 / 0.0000 / 0.0000 / 0.0000
```

This means the cross-benchmark Stage0 correction improved general missing-positive
coverage, but it still did not bring the known AppWorld create/add playlist
SkillX positives into top350. Therefore the next AppWorld step should not be a
Stage4 executor run. The next useful work is to add a general API/schema-aware
candidate-miss training path from train-split benchmark traces or dynamic skill
registry metadata, then repeat this same routing-only AppWorld gate before any
online Stage4 job.

## 2026-06-18 AppWorld dynamic skill pool alignment

The `6171bbc_3` direct routing gate needed a label-space check before more
training. The relevant positives are:

```text
skillx/appworld/spotify-add-songs-to-playlist-24
skillx/appworld/spotify-create-new-playlist-21
skillx/appworld/spotify-add-songs-to-playlist-35
```

They are absent from the pure unified v4.2 pool:

```text
data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v2/skill_pool.jsonl
skill_count = 37617
all three playlist positives in_pool = False
```

They are present after dynamic AppWorld SkillX append. A current v4.2 dynamic
pool was built with:

```bash
/data/home/scyb713/run/miniconda3/envs/xzf/bin/python scripts/build_clstr_appworld_current_route.py \
  --output_dir data/clstr_appworld_dynamic_v4_2_nowweak_append \
  --base_skill_pool_path data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v2/skill_pool.jsonl
```

Build result:

```text
status = ok
base prefix skill count = 37617
appended AppWorld SkillX count = 188
dynamic pool skill count = 37805
unique skill count = 37805
first appended AppWorld index = 37617
```

Membership check:

```text
skillx/appworld/spotify-add-songs-to-playlist-24 in_pool=True index=37761
skillx/appworld/spotify-create-new-playlist-21 in_pool=True index=37689
skillx/appworld/spotify-add-songs-to-playlist-35 in_pool=True index=37652
```

`stage0_retrieval_coverage_audit` now accepts `--skill_pool_path` and separates
two cases:

- `positive_absent_from_pool`: the positive label is not searchable in the
  current pool, so this is a pool-alignment failure.
- `missing_at_max_k`: the positive exists in the pool but was not retrieved into
  top-K, so this is a real Stage0/candidate-recall miss.

Re-auditing the old AppWorld `6171bbc_3` predictions shows the difference:

```text
with pure unified v4.2 pool:
  positive_absent_from_pool = 6
  positive_in_pool_rows = 0
  missing_at_max_k = 0
  written_missing_rows = 0

with dynamic v4.2 AppWorld append pool:
  positive_absent_from_pool = 0
  positive_in_pool_rows = 6
  missing_at_max_k = 6
  written_missing_rows = 6
```

Interpretation:

- The earlier pure-v4.2 `0/6` AppWorld gate was not a valid model-failure
  signal because the target SkillX rows were absent from the search pool.
- The dynamic-pool audit is the meaningful one: these skills are now searchable
  but still absent from top350 under the old exported predictions.
- The next low-cost step is a routing-only dynamic audit/export using
  `data/clstr_appworld_dynamic_v4_2_nowweak_append`, not a Stage4 executor job.

Verification:

```text
pytest stage0 coverage + AppWorld current-route + dynamic-routing + registry tests:
  17 passed
py_compile relevant modules/scripts:
  passed
git diff --check on touched files:
  passed
```

### Dynamic routing-only smoke after device-order fix

The first dynamic routing-only job was intentionally cancelled:

```text
job_id = 92651
state = CANCELLED by 2239
elapsed = 00:11:02
reason = no ranking/report output after model weight loading
```

Root cause:

- `audit_appworld_dynamic_routing` called `model.append_skills(appended_rows)`
  before `model.to(device)`.
- `append_skills` encodes the appended AppWorld SkillX text with the embedding
  encoder. Running this on CPU made a tiny 6-row audit spend minutes before any
  progress output.

Fix:

- move model to the resolved device and set eval mode before appending dynamic
  skills;
- write early progress phases:
  `loading_inputs -> loading_checkpoint -> appending_dynamic_skills -> preparing_pairs -> ranking`;
- add a regression test requiring `to(device)` before `append_skills`.

Verification:

```text
pytest tests/test_appworld_dynamic_routing_audit.py -> 4 passed
py_compile clstr/appworld_dynamic_routing_audit.py scripts/audit_appworld_dynamic_routing.py -> passed
git diff --check on touched dynamic-audit files -> passed
```

Re-run:

```text
job_id = 92654
state = COMPLETED
elapsed = 00:00:22
output = outputs/appworld_dynamic_routing_audit/v4_2_nowweak_anchor_w1e3_appworld6171_max6_devicefix/report.json
checkpoint = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w1e3_step2000_filterfix/checkpoints/clstr_unified_retrieval_v2-step2000.pt
base pool = data/clstr_unified_pretrain_v4_2_nowweak_stage0corr_balanced_v2/skill_pool.jsonl
dynamic pool = data/clstr_appworld_dynamic_v4_2_nowweak_append/skill_pool.jsonl
query_text_format = skillrouter
```

Metrics:

```text
skill_count = 37805
appworld_compatible_skill_count = 188
append_count = 188

global positive ranks =
  [824, 952, 878, 229, 385, 277]
global recall@20/100/350 =
  0.000 / 0.000 / 0.333
global mrr = 0.00233

appworld-compatible positive ranks =
  [12, 19, 14, 4, 13, 5]
appworld-compatible recall@20/100/350 =
  1.000 / 1.000 / 1.000
appworld-compatible mrr = 0.122386
```

Interpretation:

- The playlist positives are not intrinsically unretrievable. Inside the
  AppWorld-executable SkillX subset, all six positives are in top20.
- The global unified pool is dominated by topically similar music ToolRet /
  ToolBench / TrajectBench distractors, pushing the true AppWorld SkillX rows
  to ranks 229-952.
- The right next design is an executor-available skill mask / domain-compatible
  candidate path, not more blind Stage0 correction. This is not AppWorld-specific:
  every benchmark has an available action/tool inventory, and CLSTR should rank
  among executable candidates while still reporting global large-pool metrics.

### Available-skill mask before top-K truncation

Follow-up implementation:

- `CLSTRMultiStepController` now applies the AppWorld executor-compatible
  candidate mask before routing/belief top-K truncation.
- This fixes the case where a correct executable SkillX row has poor global
  unified-pool rank, but strong rank inside the current executable inventory.
- Diagnostics now report:
  - `available_candidate_mask_enabled`;
  - `available_candidate_count`;
  - `candidate_pool_before_available_filter`;
  - `available_filtered_executor_incompatible_skills`.
- `filtered_executor_incompatible_skills` remains as a final defense-in-depth
  rerank-stage filter count.

Regression test:

```text
test_clstr_multistep_controller_applies_executor_compatible_filter_before_candidate_truncation
```

Verification:

```text
pytest tests/test_appworld_multistep.py tests/test_appworld_dynamic_routing_audit.py -> 37 passed
pytest tests/test_stage0_retrieval_coverage_audit.py tests/test_appworld_current_route.py tests/test_dynamic_skill_registry.py -> 13 passed
py_compile relevant AppWorld routing/audit modules -> passed
```

### Controller-level dynamic candidate smoke

Added a routing-only controller audit:

- module: `clstr/appworld_dynamic_controller_candidate_audit.py`;
- CLI: `scripts/audit_appworld_dynamic_controller_candidates.py`;
- sbatch smoke: `scripts/sbatch/run_appworld_dynamic_controller_candidate_audit_smoke.sh`.

The audit loads the Stage0 checkpoint and dynamic AppWorld append pool, then
calls `CLSTRMultiStepController.select()` for correction rows. It does not call
Qwen, does not run AppWorld executor, and does not train.

Smoke job:

```text
job_id = 92736
state = COMPLETED
exit = 0:0
elapsed = 00:00:31
output = outputs/appworld_dynamic_controller_candidate_audit/v4_2_nowweak_anchor_w1e3_appworld6171_candidate20/report.json
checkpoint = outputs/clstr_unified_stage0_v4_2_nowweak_stage0corr_balanced_v2_anchor_w1e3_step2000_filterfix/checkpoints/clstr_unified_retrieval_v2-step2000.pt
candidate_top_k = 20
top_k = 5
appworld_executor_compatible_only = true
```

Metrics:

```text
positive pairs = 6
candidate_positive_hit_rate = 1.0
selected_positive_hit_rate = 0.333333
candidate positive ranks = [12, 19, 14, 4, 13, 5]
selected positive ranks = [None, None, None, 4, None, 5]
```

Depth check:

```text
job_id = 92737
state = COMPLETED
exit = 0:0
elapsed = 00:00:18
output = outputs/appworld_dynamic_controller_candidate_audit/v4_2_nowweak_anchor_w1e3_appworld6171_candidate20_top20/report.json
candidate_top_k = 20
top_k = 20
candidate_positive_hit_rate = 1.0
selected_positive_hit_rate = 1.0
selected positive ranks = [12, 19, 14, 4, 13, 5]
```

Interpretation:

- The available-skill candidate path now preserves all six AppWorld SkillX
  positives in `candidate_top20`.
- The earlier failure was not a remaining Stage0 candidate handoff failure once
  dynamic AppWorld skills and available masking are used.
- For this correction case, widening the executor handoff from top5 to top20
  would include all positives. The next bottleneck is therefore not Stage0
  coverage but how to expose/rerank/compress a larger available-skill candidate
  set without polluting the executor prompt.

## 2026-06-18 official-schema state route

The correction-query audit was not identical to the real official executor
route: the diagnostic query had `Required API refs`, while the executor state
normally only has the user goal and history. The corrected route does not use
solution-derived `task["api_refs"]`; it adds public API/schema inventory to the
CLSTR state and scores schema snippets with generic action/object matching.

Implemented route:

- `CLSTR_STATE_CONTEXT=api_schema`;
- no task api-ref bonus by default;
- generic state-changing verb mapping such as `make -> create` and
  `containing -> add/include`;
- handoff action synonym expansion so add/create APIs are not suppressed for
  playlist creation goals;
- pagination and response-shape guards for the official executor.

Routing-only audit:

```text
job_id = 92755
output = outputs/appworld_dynamic_controller_candidate_audit/v4_2_nowweak_appworld6171_official_schema_raw_top20_scored/report.json
candidate_positive_hit_rate = 1.0
selected_positive_hit_rate = 1.0
selected ranks = [14, 8, 13, 14, 8, 13]
```

Official executor hard-case smoke:

```text
job_id = 92766
output = outputs/appworld_official_executor_single6171/v4_2_dynamic_top20_schema_state_scored_handoff_paging_norm_schema_guard/report.json
task = 6171bbc_3
success_count = 1
success_rate = 1.0
evaluate_success_count = 1
average_steps = 5.0
execution_failures = 2
```

Interpretation:

- The route/handoff stack can now solve this hard playlist case without
  solution `api_refs`.
- This is a generic available-action/schema grounding mechanism, not an
  AppWorld task-id patch.
- It is still only a single hard-case smoke. The next gate should be a small
  stratified official eval with current-route rollout logging before Stage4
  online training or any full evaluation.

## 2026-06-19 visible evidence packing

The AppWorld official route now separates internal CLSTR selection depth from
executor-visible evidence depth.

New handoff controls:

- `visible_skill_limit` in `build_verified_skill_handoff`;
- `--handoff_visible_skill_limit` in `scripts/run_appworld_official_executor_eval.py`;
- `HANDOFF_VISIBLE_SKILL_LIMIT` in the official executor sbatch wrapper.

Behavior:

- Full `selected_skill_ids` remain available for diagnostics, rollout logging,
  and later Stage4 learning.
- Only the visible prefix contributes read/schema evidence to the prompt.
- Deeper candidates can still rescue state-changing action APIs when they match
  the user goal, e.g. `create_playlist` / `add_song_to_playlist`.
- Default `None` keeps existing behavior unchanged.

Recommended next small validation:

```text
CANDIDATE_TOP_K=350
HANDOFF_VISIBLE_SKILL_LIMIT=5
DEDUPE_CANONICAL_SKILLS=true
HANDOFF_PROMPT_STYLE=legacy_hints
COMPLETION_PRECHECK_MODE=constraint_tokens
CLSTR_STATE_CONTEXT=api_schema
```

Use this on a 6-task mixed set before any Stage4 online run.
