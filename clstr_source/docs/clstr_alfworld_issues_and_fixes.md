# CLSTR ALFWorld 0% 成功率 —— 问题清单与解决方法

> 对照对象：`/root/autodl-tmp/markdown_fold/` 三份方法/实验设计文档
> 实现对象：`/root/autodl-tmp/clstr/clstr/`
> 现状参考：`/root/autodl-tmp/clstr/description.md`（2026-05-20 / 2026-05-21 多轮实验记录）

清单按"对 ALFWorld 0% 的贡献度"从高到低排列。每条包含：**问题** / **位置** / **与 markdown 规范的差距** / **解决方法** / **验收信号**。

---

## P0 — 训练目标偏离：`L_policy` 退化为 supervised CE

- **问题**：ALFWorld 主训练通路 `full_base_train._compute_full_base_loss` 把 `L_policy` 实现为 `F.cross_entropy(scores, expert_action_label)`，不是 markdown §6.1 规定的 GRPO outcome supervision。直接后果是 raw policy top-1 = `look`（训练集 24% expert 取 look），与 description.md 2026-05-20 段记录的"1000/1000 步 raw policy top-1=look"完全吻合。
- **位置**：[full_base_train.py:842](../clstr/full_base_train.py#L842)；自报告 `grpo_policy_loss_used=false`（[full_base_train.py:1199, 1248](../clstr/full_base_train.py#L1199)）。
- **规范要求**：[Hybrid_Latent_Next_Skill_Router_Method_方法与实现指导版_…实验版.md] §6.1：`policy_loss(trajectories)` 按 `task_id` 分组对 reward 归一化后 `-adv * log_prob`，并配 KL-to-ref。`losses.policy_loss` 已经按此写好，但 ALFWorld 主训练没用它。
- **解决方法**：
  1. 在 `full_base_train` 中新增 outcome 训练阶段：调用 `losses.policy_loss(trajectories)` + `losses.reference_kl_loss(...)`；trajectories 用 `online_hrpo` 的 rollout 接口产出。
  2. supervised CE 改为"warm-start only"阶段（≤ 1k step），训练完立即切到 outcome supervision，避免 expert log-marginal 长期主导。
  3. 在 `train_report.json` 中 `grpo_policy_loss_used` 必须由代码实际开关决定，不能硬编码 false。
- **验收**：
  - `grpo_policy_loss_used=true` 出现在 report 中。
  - raw policy top-1 verb 分布不再被 `look/go` 完全占据（trace 中 `look` 占比 < 30%）。

---

## P1 — CLSTR-act 主链 `L_act` 未接入 ALFWorld 训练

- **问题**：`losses.action_loss`（基于 `VerifiedPair + m_t_exact / replay_prefix` 做 next-skill CE）只在 SkillsBench 风格的 `total_loss(..., variant='act')` 里使用。ALFWorld 主训练用的 `_compute_full_base_loss` 没有这一支，因此跑出来的实际是 CLSTR-base，不是 CLSTR-act —— 对应论文中的 "CLSTR 主增益"信号在 ALFWorld 上根本没启用。
- **位置**：[losses.py:126](../clstr/losses.py#L126) vs [full_base_train.py:796](../clstr/full_base_train.py#L796)（全文搜不到 `action_loss(` 的调用，只有 `L_trans_skill_ce` 替代，且后者作用在全 skill pool 而非 verified pair `candidates_next`）。
- **规范要求**：§6.3 + §6.5；`L_act` 必须在训练时实时 forward `m_hat`，用 `m_t_exact` 或 `replay_prefix`，禁止 `subspace_obs` 近似（陷阱 #10）。
- **解决方法**：
  1. 接入 `clstr.full_base_data` 中 verified-pair 构造路径；写一份 ALFWorld 专用 `build_verified_templates`（essential-subsequence prefix）。
  2. 在 `_compute_full_base_loss` 中新增 `L_act` 分支，调用 `losses.action_loss(model, vp_batch, allow_approximate_m_t=False)`；`vp_batch` 由 `online_hrpo` rollout 实时 snapshot `m_t_exact` 写入。
  3. 评估 `retrieval_recall_raw`，作为召回器对 ALFWorld 主动作的覆盖率指标。
- **验收**：
  - `train_report.json` 出现 `action_loss_used > 0`、`retrieval_recall_raw` 非零。
  - `transition_skill_recall@1` 在 verified-pair 子集上 ≥ 0.9（当前全 skill pool 上是 0.77）。

---

## P2 — 闭环推理 transition / belief score 与训练目标脱节

- **问题**：推理时 controller 的 `transition_scores` / `belief_scores` 用的是 `cos(pred, goal_emb) - cos(m_obs, goal_emb)` 这种"和 goal 文本余弦距离的增量"，与训练时 `TransHead / SkillHead` 学到的 ranking 函数完全不是同一个量。结果是训出来的 head 在推理时形同虚设。description.md 2026-05-21 中 "go=797, examine=445" 说明 controller 的真实有效信号只是 penalty。
- **位置**：[alfworld_eval.py:486-494](../clstr/alfworld_eval.py#L486-L494)；伪 skill_id 映射 [alfworld_eval.py:351-375](../clstr/alfworld_eval.py#L351-L375)。
- **规范要求**：§3.7 / §3.9 / §5（一步策略前向）。`transition_scores` 应该是 `trans_head(m_hat, action_emb(candidates))`，`belief_scores` 应该是 `skill_head(candidate_embs, m_next)`，与训练时的 logits 同一个函数。
- **解决方法**：
  1. 重写 `make_controller_component_scorer`：`transition_scores = model.trans_head(m_hat, action_emb_for_admissible_actions)`；`belief_scores = model.skill_head(candidate_embs, m_next)`。
  2. 删掉 `_skill_id_for_alfworld_action` 的 verb 前缀映射；要么放弃在 ALFWorld 上使用 skill_id 空间，要么改成"每条 admissible command 自己即一个临时 skill 表示"（cross-encoder 编码 + skill_table 重建）。
  3. 评估时打开 trace，要求 transition/belief 分量在 `policy_score` 不主导时仍能改变 top-1 选择。
- **验收**：
  - controller_diagnostic.json 中 `transition_score_calibrated` 的标准差 > 0.5，且和 `policy_score_calibrated` 的相关性 < 0.95（说明分量独立）。
  - 推理 trace 中存在"policy top-1 ≠ final top-1"的步骤，比例 ≥ 20%。

---

## P3 — frozen 8B Qwen pooled-embedding 在长程任务上表示力不足

- **问题**：CLSTR-Qwen pipeline 把 frozen Qwen3-8B last-token pool 出来当 state/action embedding，再走小 MLP scorer。Qwen direct（chat template + AVAILABLE ACTIONS）在 valid_seen 上 11.4% (description.md)，同样输入信息走 pooled embedding 路径就是 0%。这不是模型不够强，是表示路径丢掉了 Qwen 的 token-空间规划能力。
- **位置**：[qwen_external_encoder.py](../clstr/qwen_external_encoder.py) + [alfworld_eval.py:287-346](../clstr/alfworld_eval.py#L287-L346) + description.md "available-actions latent context" 段。
- **规范要求**：markdown §10.4 已经把 ALFWorld 列为辅助 benchmark，没强制要求用 frozen 8B encoder；§5（warm-start）建议先用 SkillRouter-style 较小 encoder。
- **解决方法**（按工作量从小到大）：
  1. **短期**：ALFWorld 实验改回较小（≤ 1B）的 encoder + 真正 fine-tunable 顶部 2-4 层，跑通后再考虑是否需要 8B。
  2. **中期**：保留 Qwen3-8B，但让它在 token 空间产出一个 "latent plan token sequence"，再把这段 sequence 投影到 skill_subspace；CLSTR 头作用在这个投影上而不是 raw last-token。
  3. **长期**：description.md 自己提到的 "verified Qwen teacher rollout"：让 Qwen direct 跑成功的 episode 作为 expert，蒸馏到 CLSTR controller（既保留 frozen Qwen 经济性，又拿回 Qwen 的 planning 信号）。
- **验收**：
  - 任一方案：valid_seen 20 episode success_rate ≥ Qwen direct 的 10%。
  - 不达标则记入 blocker，并停止把 Qwen external encoder 当 ALFWorld 主路径。

---

## P4 — 训练 vocabulary 与推理 vocabulary 不一致

- **问题**：训练目标（`L_trans_skill_ce`、`routing`、`L_act`）作用在 8 个 alfworld 伪 skill_id 上；推理动作选择作用在每步动态变化的 admissible_commands 实例字符串上。两边没有可学习的桥梁，导致 `transition_skill_recall@1=0.77` 这种"看上去训得很好"的指标在推理 0% 时完全无意义。
- **位置**：[alfworld_eval.py:_skill_id_for_alfworld_action](../clstr/alfworld_eval.py#L351)、[full_base_train.py:_transition_skill_logits](../clstr/full_base_train.py#L745) 上的 skill_count 来源。
- **规范要求**：markdown §10.1 强调 ALFWorld 要 "map admissible text actions or high-level templates to pseudo-skills"，但要求**训练和评估用同一映射**。当前评估侧用 verb 前缀 hack，训练侧用 dataset 中的 `skill_id` 字段（来自 build_clstr_full_base_data），二者并不严格一致。
- **解决方法**：
  1. 统一一份 `alfworld_action_to_skill_id` 函数，训练/评估/数据构造**全部 import 同一个实现**，禁止散落副本。
  2. 对每个 admissible command 在推理时也跑同样映射，确保 `transition_logits` 是在与训练时对齐的 skill 空间上做。
  3. 在 leakage audit 中加一条：训练集 `skill_id` 分布 vs 评估时 `_skill_id_for_alfworld_action` 输出分布的 KL；KL > 0.5 视为不一致，必须报错。
- **验收**：
  - 单元测试：随机抽 100 条 admissible command，训练侧 builder 和评估侧 scorer 返回相同 skill_id。

---

## P5 — STOP 分量在 controller 中实质失效

- **问题**：`stop_calibrated = _calibrate_scores(-sigmoid(stop_logits))`；`stop_logits` 在同一 state 下对所有候选是同一个标量，z-score 校准后 = 0。也就是说当前 controller 的 STOP 这条加权项是 0×weight，对决策完全没有贡献。
- **位置**：[closed_loop_controller.py:296, 303](../clstr/closed_loop_controller.py#L296)。
- **规范要求**：§3.8 STOP 是动作集合 `[K+1]` 的一员，应当与 skill 候选竞争，不是逐候选 broadcast。
- **解决方法**：
  1. 在 controller 里把 STOP 当成"虚拟动作"加入候选集合，logits 取 `stop_head(h, m)` 标量；最终 argmax 在 K+1 上做。argmax = STOP 时 episode 终止。
  2. 训练侧的 STOP BCE 标签必须按 episode 真实终止 step 而不是逐 step 1/0；当前 [full_base_train.py:926](../clstr/full_base_train.py#L926) 用 `row["done"]` 是 ok 的，但需检查 dataset 中 `done` 的覆盖率。
- **验收**：
  - 推理 trace 中存在主动 STOP 的 episode（非 50-step timeout），比例 ≥ 5%（ALFWorld 长 horizon 不应永远跑满）。

---

## P6 — Penalty 是 hardcoded，无状态条件

- **问题**：`help_action_penalty=5.0`、`reversible_action_penalty=4.0` 等是固定值。当前能压住 `look/help/inventory`，但把行为推到 `go/examine` 后没有任何信号告诉模型"现在该 pick/heat/cool"。description.md 2026-05-21 的 "go=79%" 导航循环就是这条 penalty 形成的稳态。
- **位置**：[closed_loop_controller.py:18-39, 146-187](../clstr/closed_loop_controller.py#L18-L39)。
- **规范要求**：markdown 没规定这套 penalty —— 它本来就是工程 calibration。但 calibration 不能取代信号。
- **解决方法**：
  1. 接 P0 / P1 修好后，penalty 权重整体减半甚至清零，让真实 head signal 主导。
  2. 如果一定要 penalty，让它来自 belief：例如"当 belief 距离 goal_subspace 没下降超过 N 步时，penalize 高频访问 action"。
- **验收**：
  - penalty 全部 = 0 时，模型仍能跑出非零 success_rate；否则说明模型本身还没学到东西，应该回到 P0/P1。


## 实施顺序建议

1. P5 (低成本，立即拿到诊断信号) → P4 (统一 vocabulary，是 P1/P2 的前置) → P2 (推理对齐) → P1 (CLSTR-act 真正接入) → P0 (outcome supervision) → P3 (encoder 路径选择) → P6 (penalty 退役) → P7 (叙事修正)。
2. P0 / P1 / P3 任一条不解决，ALFWorld success_rate 不会有数量级变化；P2 / P4 是让其他修复"看得见"的前置。
3. 在每条修完后，跑 `valid_seen 20 episode` 快速回归；超过 5% 才进入 50 / full eval。
