# CLSTR SkillRouter Bridge Design

Archived historical design, superseded by the SkillsBench + SkillRouter eval-only + ALFWorld experiment skeleton.

**Date:** 2026-05-13

**Goal:** Build CLSTR as an independent training and inference project in `/root/autodl-tmp/clstr`, while keeping `/root/autodl-tmp/skillrouter` as a read-only upstream mirror used only for reference, warm-start checkpoint sourcing, and evaluation-format alignment.

## 1. Scope

This design covers:

- cloning the public `SkillRouter` repository into `/root/autodl-tmp/skillrouter`
- initializing `/root/autodl-tmp/clstr` as an independent git repository
- implementing the CLSTR codebase entirely inside `clstr`
- creating a `bridges/skillrouter` adapter layer inside `clstr`
- preparing GPU-first training and inference entry points without running formal experiments yet
- adding focused tests for module contracts and bridge behavior

This design does not include:

- downloading large model weights
- running formal GPU training
- running full SkillsBench/ALFWorld experiments or SkillRouter eval-only baselines
- claiming paper-level metrics or convergence

## 2. Hard Repository Boundaries

### 2.1 `skillrouter` repository

Path: `/root/autodl-tmp/skillrouter`

Purpose:

- preserve the upstream public mirror
- inspect public code and evaluation protocol
- source warm-start checkpoint conventions
- align retrieval and rerank export formats

Constraints:

- no CLSTR training code lives here
- no CLSTR inference code lives here
- no bridge logic lives here
- no project-local extensions should be added unless absolutely necessary for environment bootstrap, and even then they should remain isolated from CLSTR logic

### 2.2 `clstr` repository

Path: `/root/autodl-tmp/clstr`

Purpose:

- the single source of truth for all CLSTR implementation
- the only repository that contains CLSTR training, inference, rollout, validator, metrics, and testing code

Constraints:

- all bridges to `SkillRouter` must remain inside `clstr`
- CLSTR must not depend on modifying upstream `SkillRouter`
- `clstr` may read files from `/root/autodl-tmp/skillrouter`, but must not require writing into it

## 3. Recommended Repository Layout

```text
clstr/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── data/
│   ├── eval/
│   ├── model/
│   └── train/
├── docs/
│   └── superpowers/
│       ├── plans/
│       └── specs/
├── scripts/
│   ├── bootstrap_skillrouter.sh
│   ├── prepare_skill_cache.py
│   ├── build_verified_pairs.py
│   ├── train_stage.sh
│   ├── infer_static.sh
│   └── infer_agentic.sh
├── tests/
├── clstr/
│   ├── __init__.py
│   ├── data.py
│   ├── encoders.py
│   ├── belief.py
│   ├── heads.py
│   ├── model.py
│   ├── losses.py
│   ├── rollout.py
│   ├── validator.py
│   ├── metrics.py
│   ├── train.py
│   ├── infer.py
│   └── bridges/
│       └── skillrouter/
│           ├── __init__.py
│           ├── checkpoints.py
│           ├── datasets.py
│           ├── evaluation.py
│           └── serialization.py
└── examples/
```

## 4. Implementation Strategy

### 4.1 Why independent training in `clstr`

The public `SkillRouter` repository is an evaluation-oriented release. Its public structure is suitable for:

- retrieval export
- reranking evaluation
- benchmark alignment

It is not a CLSTR training framework. Trying to embed CLSTR training into that repository would create the wrong dependency direction and would make later experimentation brittle.

Therefore:

- all CLSTR training must be written independently in `clstr`
- all CLSTR inference must be written independently in `clstr`
- `SkillRouter` is treated as an upstream artifact source, not as a host project for CLSTR development

### 4.2 Why a bridge layer still exists

Although CLSTR training is independent, the method still needs a clean compatibility layer for:

- warm-start checkpoint loading
- skill metadata serialization compatible with current skill pools
- static evaluation export alignment with the `SkillRouter` public protocol

The bridge keeps that compatibility code out of the core CLSTR modules.

## 5. Core Module Design

### 5.1 `clstr/data.py`

Owns all core dataclasses and data-domain contracts:

- `Skill`
- `Task`
- `ExecutionState`
- `TrajectoryStep`
- `Trajectory`
- `ReplayStep`
- `VerifiedPair`
- `RetrievalPositive`

Also owns loaders for:

- SkillRouter eval data
- generic replay trajectory data

Training code should consume these structured objects, not raw JSON blobs.

### 5.2 `clstr/encoders.py`

Owns:

- `StateEncoder`
- `CrossEncoder`
- `SkillTable`

Responsibilities:

- encode `[query; serialized execution state] -> h_t`
- encode candidate-specific cross features for reranking or policy heads
- build and maintain `SkillTable.E`

Pooling behavior must explicitly support:

- masked mean
- last non-pad token
- CLS token

### 5.3 `clstr/belief.py`

Owns:

- `subspace_obs`
- `TransitionPredictor`
- `BeliefGate`

Supported gate modes:

- learned element-wise gate
- Bayes scalar gate
- optional Bayes diagonal gate

This module owns closed-loop belief update logic and should not be spread across the model or rollout code.

### 5.4 `clstr/heads.py`

Owns:

- `SkillHead`
- `StopHead`
- `TransHead`

`TransHead` must expose a spectral normalization switch and default to enabled in the main configuration.

### 5.5 `clstr/model.py`

Owns:

- `CLSTRConfig`
- `CLSTRModel`
- `policy_forward`
- `rollout_step`
- `step_update`
- `encode_skills`
- `rebuild_skill_table`

Responsibilities:

- compose module outputs into explicit tensor contracts
- keep orchestration in one place
- avoid embedding all loss or rollout logic into one monolithic `forward`

### 5.6 `clstr/losses.py`

Owns:

- `transition_loss`
- `retrieval_loss_weak`
- `retrieval_loss_strong`
- `policy_loss`
- `action_loss`
- `total_loss`

Each loss function should return a scalar loss and a stats dictionary for training logs.

### 5.7 `clstr/rollout.py`

Owns:

- `topk_recall_fn`
- `sample_action`
- `rollout`
- `_full_replay_m_t`
- retrieval-positive construction helpers

This module must explicitly handle:

- global `STOP_IDX = N`
- local rollout stop index `K`
- replay-based reconstruction of belief state
- top-K positive injection for training only

### 5.8 `clstr/validator.py`

Owns:

- `render_validator_prompt`
- `parse_validator_output`
- `build_verified_templates`
- `attach_belief_sources`

All terminology must use `LLM-based validator`, not `oracle validator`.

### 5.9 `clstr/metrics.py`

Owns:

- task success metrics
- next-skill accuracy and recall
- stop precision, recall, and F1
- belief MSE
- transition ranking metrics
- honest raw retrieval recall derived from `was_in_raw_topk`

### 5.10 `clstr/train.py`

Owns staged CLSTR training:

- stage 0 warm-start
- stage 1 retrieval bootstrap
- stage 2 transition and policy learning
- stage 3 CLSTR-act verified supervision

Also owns:

- optimizer and scaler setup
- checkpoint save and load
- config-driven execution

### 5.11 `clstr/infer.py`

Owns:

- static evaluation export
- agentic rollout inference
- query wrapping into `Task(task_id="infer", query=...)`

## 6. SkillRouter Bridge Design

Bridge path:

`/root/autodl-tmp/clstr/clstr/bridges/skillrouter/`

### 6.1 `checkpoints.py`

Responsibilities:

- load a local checkpoint path or model identifier
- map compatible upstream weights into `StateEncoder`
- optionally map reranker-compatible weights into `CrossEncoder`
- leave CLSTR-specific heads randomly initialized
- report which keys loaded and which did not

### 6.2 `datasets.py`

Responsibilities:

- read SkillRouter evaluation data layouts
- normalize task and skill metadata into CLSTR dataclasses
- support sharded pools

### 6.3 `serialization.py`

Responsibilities:

- serialize skill metadata using the CLSTR method specification
- rebuild `SkillTable.E` from the current skill pool
- cache serialized text or precomputed embeddings when useful

### 6.4 `evaluation.py`

Responsibilities:

- export retrieval outputs aligned with the SkillRouter scoring contract
- expose helpers for static retrieval or rerank comparison
- avoid contaminating CLSTR trajectory metrics with static evaluation shortcuts

## 7. Training and Inference Flow

### 7.1 GPU-first assumption

The code should be written for the intended execution path:

- CUDA-first
- support for `bf16` or `fp16`
- gradient accumulation
- gradient checkpointing
- architecture that can later accept multi-GPU extension cleanly

The current no-GPU environment only affects execution timing, not architecture decisions.

### 7.2 Stage 0: warm-start

- load SkillRouter-compatible backbone weights
- rebuild the current `SkillTable.E`
- freeze or unfreeze `E` by configuration

### 7.3 Stage 1: retrieval bootstrap

- train weak retrieval positives from successful trajectories
- align shared text representations with the skill subspace

### 7.4 Stage 2: transition and policy

- enable closed-loop rollout
- optimize transition and policy losses jointly
- support replay or deterministic environment backends

### 7.5 Stage 3: CLSTR-act

- consume `VerifiedPair` supervision
- inject positives into candidate lists for training only
- separately report honest raw top-K retrieval coverage

### 7.6 Inference

Two supported modes:

- `static`: SkillRouter-aligned retrieval or rerank export
- `agentic`: CLSTR rollout-based execution

## 8. Testing Strategy

The first implementation pass should aim for engineering reliability, not paper-level experimental completeness.

### 8.1 Test coverage

- `tests/test_data.py`
- `tests/test_encoders.py`
- `tests/test_belief.py`
- `tests/test_heads.py`
- `tests/test_rollout.py`
- `tests/test_losses.py`
- `tests/test_bridge_skillrouter.py`
- `tests/test_train_smoke.py`

### 8.2 Minimum validation bar

Before claiming the first implementation pass is complete:

- package installs successfully
- modules import successfully
- bridge interfaces load with mocked or local test fixtures
- toy rollout passes
- stage entry points run through smoke tests
- checkpoint save and load round-trip works

## 9. Risks and Controls

### 9.1 Upstream repo mismatch

Risk:

- public `SkillRouter` checkpoints or module names may not line up perfectly with CLSTR wrappers

Control:

- explicit key mapping
- strict and non-strict load options
- visible load reports

### 9.2 Data mismatch

Risk:

- static SkillRouter eval data does not provide trajectory supervision

Control:

- keep static benchmark data, trajectory data, and verified supervision as separate data lanes

### 9.3 Metric contamination

Risk:

- candidate injection can fake recall metrics

Control:

- compute reported raw recall from `was_in_raw_topk`
- keep training candidate sets separate from evaluation recall accounting

### 9.4 Environment lock-in

Risk:

- overfitting the implementation to a mock environment would make later replay or live integration expensive

Control:

- define environment-facing contracts early
- keep replay, mock, and live backends behind consistent interfaces

## 10. Execution Boundaries for This Round

This round should complete:

- clone `SkillRouter` into `/root/autodl-tmp/skillrouter`
- initialize `/root/autodl-tmp/clstr` as a git repository
- write this design spec
- write an implementation plan
- scaffold and implement the CLSTR project structure
- add bridge code, CLI entry points, configs, and tests
- run local minimum verification

This round should not claim:

- final training convergence
- final benchmark numbers
- full SkillsBench + ALFWorld experiment reproduction, with a SkillRouter eval-only baseline
- validated experiment quality

## 11. Decision Summary

The final design decision is:

- `SkillRouter` remains a read-only upstream mirror
- all CLSTR training and inference are implemented independently inside `clstr`
- the only integration surface is the bridge layer inside `clstr`

That boundary is intentional and should not be relaxed during implementation.
