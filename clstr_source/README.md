# CLSTR

CLSTR is a research codebase for Closed-loop Latent Skill Transition Routing. The current paper method is a trajectory-aware skill router: it ranks skills or tools, while a downstream language model, API executor, or benchmark simulator executes the selected action.

## Current method

The active training path is:

Stage0 → Stage1 → Stage2 → Stage4

The unified-memory route uses one consistent causal state definition:

    h_t = EncodeState(s_t)
    m_0 = InitialBelief(h_0)
    score_t = UnifiedRoute(h_t, m_t)

After the current skill/action is executed:

    u_t = ActionProjection(a_t)
    o_{t+1} = EncodeObservation(x_{t+1})
    h_{t+1} = EncodeState(s_{t+1})

    m_hat_{t+1} = Transition(m_t, u_t, o_{t+1})
    b_{t+1} = SubspaceObservation(E_skill, h_{t+1})
    gamma_{t+1} = Gate(m_hat_{t+1}, b_{t+1}, o_{t+1})
    m_{t+1} = gamma_{t+1} * b_{t+1}
                + (1 - gamma_{t+1}) * m_hat_{t+1}

    score_{t+1} = UnifiedRoute(h_{t+1}, m_{t+1})

The static comparator at the same next state is:

    m_static_{t+1} = InitialBelief(h_{t+1})
    score_static_{t+1} = UnifiedRoute(h_{t+1}, m_static_{t+1})

This keeps initialization, replay, training, and inference aligned. Current action text, next observation text, next state text, and replay history all have explicit roles.

## Four training stages

1. Stage0 routing

   Train the SkillRouter-compatible bi-encoder coarse retriever and verify full-pool recall against the frozen baseline.

2. Stage1 heads

   Load the Stage0 routing foundation, preserve its retrieval behavior, and initialize the CLSTR belief, transition, gate, action-projection, and route heads on Stage0 top-M candidates.

3. Stage2 full-base supervised training

   Train the component-complete supervised objective. In unified-memory mode, next-skill cross-entropy is computed from the causally updated pair (h_{t+1}, m_{t+1}), not from the current pair (h_t, m_t).

4. Stage4 causal next-skill training

   Focus training on causal transition-conditioned next-skill ranking. The initializer, transition, gate, action projection, and unified retriever are trainable; the old policy-preference training surface is not part of the current method.

The readiness audit exposes exactly these gates:

- stage0_routing
- stage1_heads
- stage2_full_base
- stage4_causal_next_skill

## Training and evaluation contracts

Training and current-state evaluation intentionally use different time indices.

- Causal Stage2/Stage4 training predicts the next skill from h_{t+1} and the post-action memory m_{t+1}.
- Current-state route evaluation predicts the target skill at h_t. Static memory is InitialBelief(h_t); dynamic memory is reconstructed only from the replay prefix. The current row's action is not applied before that prediction.
- Teacher rollouts are supervised-data extraction only.
- Closed-loop benchmark rollouts are evaluation only.
- Logged-online trajectory files remain offline replay/evaluation infrastructure; they are not a separate policy-training stage.

### Static-preserving memory candidate recall

Applicable known-pool sequential evaluations use one declared global skill
table and the following fixed contract:

- keep every static Top-M skill;
- append dynamic-memory Top-D skills after masking the static shortlist;
- rank the resulting union with the same unified-memory scorer;
- compare against static Top-(M+D) with the same final scorer and final K;
- use all eligible source rows as the primary strict denominator, including
  unknown/out-of-pool targets as zero rather than filtering them.

The global and ToolBench launchers expose these values as `STATIC_K`,
`DYNAMIC_EXTRA_K`, and `FINAL_K` (CLI aliases `--static_k`,
`--dynamic_extra_k`, and `--final_k`). Local saturated pools are diagnostic
only. ALFWorld and WebShop are explicitly inapplicable because the environment
supplies admissible actions. Reliability backoff changes scores on this fixed
union; it never resets recurrent memory or changes the candidate set.

`scripts/audit_clstr_unified_training_readiness.py` rejects candidate-recall
claims that use retained-only denominators, inject gold positives, infer a legal
pool from targets/namespaces, omit the equal-budget comparator, or request
`FINAL_K > STATIC_K`.

### Reliability-aware memory score backoff

The fixed candidate union supports four source-complete reliability modes:
`static`, `dynamic`, `fixed_alpha`, and `heuristic`. They fuse branch scores as
`static + alpha * (dynamic - static)`. Rows with no causal history use the exact
static endpoint, and reliability never resets or interpolates recurrent memory.

The global-pool and ToolBench evaluators can persist detached static/dynamic
route records with `--route_records_path`. A companion manifest binds the
records to the candidate-union versions and budgets, declared pool, ordered
skill mapping, feature schema, source rows, and effective checkpoint chain.
Use `scripts/audit_clstr_memory_utility_oracle.py` to run the
trajectory-disjoint fixed-alpha/oracle/linear audit over one or more manifests.

`learned` is fail-closed. The unified readiness audit accepts it only when
`--memory_utility_audit_report_path` names a passing known-global audit that
recommends a learned gate and `--memory_utility_gate_checkpoint_path` names a
checkpoint cryptographically bound to that exact report. Until real post-fix
held-out records satisfy those conditions, use a non-learned mode; no learned
gate is part of the current verified method.

## Main modules

- clstr/model.py: CLSTR model and unified retriever.
- clstr/belief.py: transition predictor, subspace observation, and belief gate.
- clstr/full_base_train.py: Stage2 data preparation, replay, causal memory update, and supervised losses.
- clstr/stage4_act_train.py: focused causal next-skill training.
- clstr/current_state_route_eval.py: current-state unified-memory evaluation.
- clstr/stage0_quality_gate.py: Stage0 retrieval gate.
- clstr/stage1_heads_quality_gate.py: Stage1 heads gate.
- clstr/stage2_quality_gate.py: Stage2 handoff gate.
- clstr/stage4_quality_gate.py: causal Stage4 evidence gate.
- clstr/candidate_utils.py: candidate positive-injection utility.
- clstr/device_utils.py: device selection utility.

Qwen external encoders are optional ablations or downstream executors. They are not required by the main CLSTR training path.

## Main entry points

- scripts/run_clstr_stage0_biencoder_train.py
- scripts/run_clstr_stage1_heads_init.py
- scripts/run_clstr_stage2_full_base_train.py
- scripts/run_clstr_stage4_act_train.py
- scripts/audit_clstr_unified_training_readiness.py

Cluster launchers:

- scripts/sbatch/run_clstr_unified_stage0_biencoder_train.sh
- scripts/sbatch/run_clstr_unified_stage1_heads_init.sh
- scripts/sbatch/run_clstr_unified_stage2_full_base_train.sh
- scripts/sbatch/run_clstr_unified_stage4_act_train.sh
- scripts/sbatch/run_clstr_unified_readiness_audit.sh

Guarded handoff helpers currently exist for Stage0-to-Stage1 and Stage1-to-Stage2. Full training jobs should be launched only after a small smoke run and the preceding quality gate reports status=ok.

## Data and leakage

The unified dataset contains:

- skill_pool.jsonl: canonical skill/tool inventory.
- retrieval.jsonl: train-safe retrieval supervision.
- trajectories.jsonl: train-safe sequential supervision.
- leakage_audit.json: split and identifier leakage report.
- source_inventory.jsonl: source availability and training eligibility.

Benchmark dev, validation, test, and evaluation rows must not enter training. Candidate handoff must record whether positives were present, skipped, or injected.

## Verification

Focused checks:

    env CUDA_VISIBLE_DEVICES='' PYTHONPYCACHEPREFIX=/tmp/clstr-pycache \
      /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m pytest -q \
      tests/test_full_base_train.py \
      tests/test_stage4_act_train.py \
      tests/test_current_state_route_eval.py

Static checks:

    /data/home/scyb713/run/miniconda3/envs/xzf/bin/python -m compileall -q clstr scripts
    bash -n scripts/sbatch/run_clstr_unified_readiness_audit.sh
    git diff --check

## Shared-cluster rules

- Keep project changes under /data/home/scyb713/run/xzf/AAAI/autodl-tmp.
- Use login nodes only for editing, inspection, and light tests.
- Submit long training, evaluation, and data-generation jobs through Slurm.
- Compute nodes may not have internet access; prepare models and datasets first.
- Keep large datasets, checkpoints, caches, and benchmark checkouts out of Git.
