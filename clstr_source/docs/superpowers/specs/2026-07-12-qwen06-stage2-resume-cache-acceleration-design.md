# Qwen06 Stage2 Resume Cache Acceleration Design

## Goal

Resume the interrupted Qwen06 CLSTR Stage2 run from its verified step-2000 checkpoint, preserve the original 10,000-step sampling and loss schedule, and eliminate repeated frozen-Qwen encoding by forcing the existing full-base embedding cache.

## Training Identity

- Resume checkpoint: `stage2_full/checkpoints/resume-step2000.pt`.
- Checkpoint SHA-256: `79a618adc2cbf83f105feab0e2bf2388b6a1ba62ef57abc1aafe50c19aa9a896`.
- The checkpoint contains `step=2000`, model state, and AdamW optimizer state.
- The original `max_steps=10000` remains the sampling, handoff-subset, and counterfactual-warmup budget.
- A new optional `target_total_steps=10000` makes the resumed run execute only steps 2001 through 10000.

## Cache Strategy

Use the existing `_attach_full_base_embedding_cache()` path with `embedding_cache_mode=always`. It encodes unique frozen state and transition texts once in large batches, then training stacks cached CPU embeddings instead of tokenizing and running Qwen every step.

This first recovery does not redesign the cache representation. The node has sufficient host memory, and the existing small-run cache path is already used by Stage2 smoke. If the full cache build fails or does not improve measured throughput, the verified step-2000 checkpoint remains intact and a packed tensor-bank cache becomes a separate change.

## Launcher Contract

- The canonical Stage2 Python CLI accepts `--target_total_steps`.
- The lower Stage2 launcher forwards optional `TARGET_TOTAL_STEPS`.
- The Qwen06 Stage2 wrapper keeps `MAX_STEPS=10000`, defaults `TARGET_TOTAL_STEPS=10000`, and permits inherited `EMBEDDING_CACHE_MODE`/`EMBEDDING_CACHE_MAX_ROWS` overrides.
- Full checkpoint and quality-gate expectations use the target total step, which remains 10000 for the resumed paper run.

## Failure Handling

- Reject a target total step that is not greater than the resume step.
- Require the resume checkpoint to exist before submission.
- Preserve the frozen checkpoint copy independently of `latest.pt`.
- Do not submit Stage4 until the replacement Stage2 job completes and its quality gate and lineage pass.
- If cache construction OOMs or fails, do not fall back silently to online encoding; inspect and choose an explicit alternative.

## Verification

- Unit test that a step-1 checkpoint resumed with `max_steps=3,target_total_steps=3` trains exactly steps 2 and 3 and restores optimizer state.
- Launcher tests verify target forwarding and cache override behavior.
- Focused regression tests cover full-base resume and Qwen06 launchers.
- Replacement Slurm job must report handoff cache reuse, `full_base_embedding_cache.used=true`, `resume.start_step=2000`, `resume.optimizer_loaded=true`, and `final_step=10000`.

