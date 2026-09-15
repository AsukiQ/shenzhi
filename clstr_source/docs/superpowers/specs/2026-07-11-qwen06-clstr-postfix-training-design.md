# Qwen3-Embedding-0.6B CLSTR Post-Fix Training Design

## Status

Approved in conversation on 2026-07-11.

## Goal

Train and evaluate repaired CLSTR with Qwen3-Embedding-0.6B as its frozen encoder backbone. Stage0 is quality-gated and Stage1, Stage2, and Stage4 start fresh only when candidate recall is sufficient.

## Scope

This design covers:

- Qwen3-Embedding-0.6B backbone initialization;
- a versioned, causal state-query prompt used consistently across training and inference;
- a frozen-backbone, mined-hard-negative Stage0;
- segmented Stage0 continuation up to 5000 optimizer steps;
- post-fix Stage1, Stage2, and Stage4 training;
- route-record generation and the trajectory-disjoint memory utility audit.

The exact prompt contract is defined in `docs/superpowers/specs/2026-07-11-clstr-qwen-prompt-consistency-design.md`.

This design does not cover:

- BGE-M3 experiments;
- SkillRouter or ToolRex baseline training in other worktrees;
- backbone unfreezing, LoRA, or full-parameter tuning;
- learned reliability-gate training before a positive post-fix audit;
- top-K attention, reranker distillation, or other architecture changes.

## Non-Negotiable Safety Constraints

1. The encoder and tokenizer initialize from:
   /data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/models/Qwen3-Embedding-0.6B
2. Every Qwen backbone parameter remains frozen.
3. No launcher may set train_encoder_backbone, unfreeze_backbone, LoRA, or an encoder-backbone learning rate.
4. Unfreezing even one layer requires separate explicit user approval before code or training changes.
5. SkillRouter, BGE, and legacy CLSTR Stage1/2/4 weights are invalid initialization sources.
6. Model loading, skill-table construction, data scans, audits, and training run only through Slurm compute jobs.
7. The storage node is limited to lightweight inspection, targeted tests, shell syntax checks, and Slurm submission.
8. The existing clstr-qwen06-full-readiness and clstr-qwen06-sr-toolrex worktrees remain untouched.

## Initialization

The Qwen tokenizer and encoder load from the same Qwen3-Embedding-0.6B checkpoint. The encoder uses last-token pooling, left padding, bfloat16 on GPU, and model dimension 1024.

CLSTR-specific trainable components start fresh:

- encoder projection;
- skill adapter and skill bias;
- retrieval scale;
- initial-belief and unified-routing heads in Stage0;
- transition, observation/correction, and Stage4-specific heads when their stages begin.

Stage0 rebuilds its skill table from all 67,557 skills with the Qwen encoder. The frozen base skill embeddings and frozen encoder backbone are never optimized. No old CLSTR initialization checkpoint is loaded.

## Data

Stage0 uses:

/data/home/scyb713/run/xzf/AAAI/autodl-tmp/clstr/data/clstr_unified_pretrain_v4_2_progressive_final_function_aug_v2

Its manifest records:

- 67,557 skills;
- 663,471 retrieval rows;
- leakage-filtered v4.2 base data;
- function-schema training augmentation.

Function-augmentation rows are training supervision and must not be presented as held-out evidence.

Stage1, Stage2, and Stage4 use trajectories.jsonl and skill_pool.jsonl from the same function-aug-v2 data root above. The trajectory file is inherited from the leakage-filtered v4.2 base manifest; function augmentation changes retrieval supervision and the skill pool, not the trajectory claims. Every run manifest records checksums for model identity, tokenizer identity, skill pool, data manifest, and parent checkpoint.

## Stage0 Objective

The objective is:

L_stage0 = L_full_pool_multi_positive_nll
         + 0.2 * L_online_mined_hard_negative

The mined-hard-negative loss uses:

- margin 0.1;
- top 32 non-positive skills per query;
- positives and alias-equivalent positives masked before mining.

The labeled explicit-negative auxiliary is disabled with weight 0. Previous experiments showed that labeled explicit negatives did not improve fixed handoff coverage, while online mined hard negatives improved global, ToolBench-G3, and TrajectBench coverage.

Those mined-hard-negative results came from the earlier Stage0 scorer. The Qwen run applies the same evidence-backed loss to the current unified-memory Stage0 and must revalidate its benefit through the step-0 and step-1200 audit rather than assuming the gain transfers.

Other fixed settings are:

- route scorer: unified_memory;
- static candidate logits: unified_route_logits(h, initial_belief(h));
- initial belief top-k: 64;
- state query prompt: clstr_causal_state_v1;
- state query max chars: 2000;
- state query truncation: head_tail_v1;
- retrieval loss: multi-positive NLL;
- sampling: handoff-balanced;
- alias-positive expansion: enabled;
- train skill embeddings: false;
- train skill bias: true;
- train encoder projection: true;
- train skill adapter: true;
- train retrieval scale: true;
- train encoder backbone: false;
- initialization checkpoint: none.

## Stable Step-0 Control

The trainer saves a stable post-rebuild checkpoint before the first optimizer step:

checkpoints/clstr_unified_retrieval_v2-step0.pt

It contains:

- the rebuilt Qwen skill table;
- fresh CLSTR Stage0 heads;
- optimizer state;
- model, tokenizer, and skill-pool identity metadata;
- frozen-backbone evidence;
- no trainable encoder-backbone parameters.

The step-0 checkpoint is audited on the same fixed rows as later checkpoints. It is an operational initialization control, not a paper test-set result.

## Stage0 Execution

### Mechanical Smoke

The first Slurm job uses:

- at most 8192 skills;
- at most 4096 queries;
- 80 optimizer steps;
- the approved objective and frozen-backbone settings.

The smoke passes only when:

- status is ok;
- loss values are finite;
- mined-hard-negative row and pair counts are positive;
- a checkpoint is written;
- the report says freeze_backbone=true and train_encoder_backbone=false;
- no optimizer parameter belongs to encoder.backbone;
- optimizer parameters include initial_belief_head and unified_retriever;
- the checkpoint excludes frozen backbone weights.

Failure stops the run before full-pool training.

### Segmented Full-Pool Training

The maximum budget is 5000 optimizer steps:

1. step 0 to 1200;
2. resume 1200 to 2400;
3. resume 2400 to 3600;
4. resume 3600 to 5000 only if the 2400-to-3600 audit still improves.

Each continuation restores model and optimizer state from the explicit preceding checkpoint. It uses the same output lineage, data checksums, model identity, seed, objective, sampling settings, and skill pool.

## Fixed Handoff Audit

Step 0, 1200, 2400, 3600, and optional step 5000 are evaluated on the same deterministic 2048-row operational audit set using:

- query mode: checkpoint_state_query;
- top-k values: 20, 50, 100, 200, and 500;
- the 67,557-skill Qwen Stage0 pool;
- unified-memory static routing through m_0;
- global, ToolBench-G3, and TrajectBench current and next recall.

These rows are only for continuation decisions. Final paper claims come from held-out benchmark protocols.

## Promotion Rules

### Step 1200

Continue to step 2400 only if:

- global next recall at 100 and 500 both improve over step 0;
- neither ToolBench-G3 nor TrajectBench next recall at 500 regresses by more than 0.01;
- at least one of ToolBench-G3 or TrajectBench improves at both next recall at 100 and 500;
- mined-hard-negative supervision has positive row and pair counts.

Otherwise stop and report. Do not unfreeze the backbone.

### Steps 2400 and 3600

The primary composite is the mean of global, ToolBench-G3, and TrajectBench next recall at 100.

Continue only if either:

- the composite improves by at least 0.005 over the preceding segment; or
- one hard domain improves next recall at 100 by at least 0.01;

and no domain's next recall at 500 regresses by more than 0.01.

Step 5000 is optional rather than forbidden. It runs when step 3600 is still improving.

### Stage1 Release Floor

The selected Stage0 checkpoint must satisfy:

- global next recall at 500 at least 0.90;
- ToolBench-G3 next recall at 500 at least 0.84;
- TrajectBench next recall at 500 at least 0.70;
- TrajectBench next recall at 200 at least 0.52;
- no promotion metric worse than step 0 beyond the 0.01 safety tolerance.

These floors derive from the previously successful 1200-step mined-hard-negative audit. They are a minimum viable candidate handoff, not a final paper target.

If a floor is missed by less than 0.02, the pipeline stops in a borderline state and reports evidence to the user. Stage1 then requires explicit user authorization. Larger misses fail automatically.

## Stage1, Stage2, and Stage4

After Stage0 passes:

1. Stage1 initializes fresh supervised transition and correction heads from the selected unified-memory Stage0 checkpoint.
2. Stage2 initializes from Stage1 and trains the repaired unified-memory full-base objective.
3. Stage4 initializes from Stage2 and trains causal next-skill refinement using the repaired post-action memory update.

All three stages explicitly use route_scorer=unified_memory, Stage0 top-M 500, and skillrouter_state handoff queries. Stage1 retains at least 64 inventory candidates. Stage2 uses full-pool causal next-skill supervision with explicit-only inventory masking. Stage4 consumes 64 candidates after the Stage0 top-500 handoff.

Each stage rejects:

- a parent checkpoint with a different backbone identity;
- a different skill-pool checksum;
- a legacy pre-fix checkpoint;
- a missing or failed upstream gate;
- a configuration that enables backbone training.

Stage1, Stage2, and Stage4 do not reuse old Qwen or SkillRouter CLSTR heads.

## Evaluation

The completed Qwen chain evaluates:

- static routing;
- dynamic routing;
- fixed-alpha reliability fusion;
- heuristic reliability fusion;
- static-preserving memory candidate union.

Evaluation emits:

- benchmark reports;
- route-record JSONL;
- companion manifests with checkpoint, data, and pool identity;
- static-versus-dynamic and memory-candidate-recall diagnostics.

The trajectory-disjoint memory utility oracle and learnability audit runs after route records exist. Learned-gate training remains disabled unless this post-fix audit recommends it.

## Failure Handling

- Mechanical smoke failure: stop before full-pool training.
- Step-1200 gate failure: stop Stage0 and report.
- Later plateau: select the best safe checkpoint instead of spending the remaining budget automatically.
- Backbone mismatch or trainable-backbone evidence: fail immediately.
- Stage0 floor failure: do not launch Stage1.
- Borderline floor result: stop and request user direction.
- Storage-node termination or resource pressure: do not retry locally; move the check to Slurm.
- Any proposal to unfreeze the backbone: stop and request explicit approval.

## Verification

Before Slurm submission:

- targeted tests cover stable step-0 checkpoint creation, frozen-backbone guards, identity checks, segmented resume, and promotion decisions;
- relevant shell launchers pass bash -n;
- Python files pass compileall;
- git diff --check passes.

On a compute node:

- the mechanical smoke verifies model loading and gradient boundaries;
- full-pool Stage0 verifies resume state and gate artifacts;
- downstream jobs are dependency-gated on successful upstream reports.

No full repository pytest, model load, or dataset scan runs on the storage node.
