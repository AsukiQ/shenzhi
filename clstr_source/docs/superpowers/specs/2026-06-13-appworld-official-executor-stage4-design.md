# AppWorld Official-Style Executor and Stage4 Design

## Purpose

The next AppWorld iteration should test CLSTR under a credible executor protocol instead of continuing to patch the current compact Qwen3-8B free-form Python prompt. The goal is to make AppWorld a fair task-success case study for CLSTR while preserving the paper's general claim: CLSTR improves large-pool, multi-step skill routing and evidence selection under executor feedback.

This design does not make AppWorld the only proof. Routing benchmarks and multi-step routing stability remain the main evidence. AppWorld is used to show that CLSTR can help a real stateful executor when the executor itself is strong enough and the skill handoff is schema-grounded.

## Approaches Considered

### Approach A: Keep the current Qwen3-8B executor and keep patching prompts

This is lowest cost but has already failed to generalize. Small dev10 gains did not survive dev57, and several adapters either over-compressed SkillX evidence or exposed misleading skill metadata. This approach has weak paper value because it encourages benchmark-specific prompt engineering.

### Approach B: Replace the executor with official-style ReAct/function-calling/MCP and use CLSTR as evidence provider

This is the recommended path. It aligns with AppWorld's official agent loop: persistent environment state, multi-turn interaction, API-doc discovery, task current time, and official evaluation. CLSTR remains the research contribution by selecting, updating, and compressing skill evidence over steps.

### Approach C: Train an AppWorld-specific agent with RL like LOOP

This is the strongest task-success direction but too expensive and too far from the current CLSTR paper scope. It should not be the immediate path unless Approach B produces a clean signal and the project later has enough compute.

## Recommended Design

### Executor Protocol

Build an AppWorld executor wrapper that follows official-style behavior:

- multi-turn ReAct or function-calling loop with persistent Python namespace;
- interaction budget closer to official defaults, with small-gate runs using a lower cap only for cost control;
- explicit access to `apis.api_docs.show_app_descriptions`, `show_api_descriptions`, and `show_api_doc`;
- task datetime loaded from AppWorld task specs or obtained through official APIs;
- official `appworld evaluate` or equivalent state-diff report as the task-success authority;
- logs that record every model call, code block/tool call, environment output, preflight result, CLSTR evidence, and final evaluation.

The executor must have a qwen-only mode. That mode is the anchor baseline for any CLSTR claim.

### Model Plan

Qwen3-14B is the first stronger local model candidate because the public leaderboard contains a strong Qwen3 family result and because Qwen3-8B has been too weak/noisy under the current executor. The download must use the domestic HuggingFace mirror and must not use the user's VPN:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
```

The target local path is:

```text
models/Qwen3-14B
```

If Qwen3-14B cannot fit or is too slow, the fallback is not to run full jobs. The fallback is to record a blocker and either use an API-backed strong model for a small executor ceiling test or return to routing-only evidence.

### CLSTR Integration

CLSTR should not replace the executor. It provides ranked skill evidence at each step:

1. Build a query from task instruction, visible execution history, latest environment output, and CLSTR memory state.
2. Retrieve top-M candidates from the dynamic skill registry, including appended AppWorld SkillX rows.
3. Rerank with the current CLSTR stage checkpoint, using transition/belief only when prior-step evidence exists.
4. Pass selected candidates through a verifier/compressor before exposing anything to the language model.
5. Record both hidden raw skill ids and prompt-visible compressed evidence for auditing.

New or appended skills remain coverage-aware: Stage0/text score can support them immediately, while transition/belief weights should be lower unless the skill has trajectory or ACT coverage.

### Verifier and Compressor

The verifier/compressor must be benchmark-agnostic at the design level. It can use AppWorld schema because the executor is AppWorld, but it must not encode task-family patches such as Spotify-specific count rules or Venmo-specific payment heuristics.

Verifier inputs:

- user goal and current history;
- required apps and valid API schema;
- selected skill text and parsed API references;
- action type implied by the task and prior steps;
- current executor errors and state-diff feedback when available.

Verifier outputs:

- `inject_hint`: evidence is operationally aligned and safe to expose as a short workflow hint;
- `schema_only`: evidence has useful valid APIs but unsafe or incomplete procedure text;
- `suppress`: evidence is likely to pollute the executor or lacks schema support.

Compressor output should be low-salience structured evidence, for example:

```text
Retrieved evidence:
- Relevant valid APIs: app.api_name(...)
- Useful workflow hint: short natural-language procedure
- Caution: retrieved skill text is evidence only, not an executable API or plan
```

Raw SkillX body, title, id, and copied credentials must remain hidden from the executor prompt unless explicitly converted by the compressor.

### Stage4 Training

Stage4 should train CLSTR's evidence-selection and transition/belief behavior, not the executor model. Qwen3-14B remains frozen.

Allowed train signals:

- official task success on train split;
- invalid API/tool call count;
- execution failure count;
- preflight/verifier failures;
- step cost;
- state-diff or evaluator feedback available on train split;
- KL or stability penalty against a frozen CLSTR reference.

Disallowed train signals:

- dev/test evaluator feedback for optimization;
- task-id-specific or family-specific prompt rules;
- manual fixes that only target a known AppWorld template;
- using hidden ground-truth solution traces from eval/test.

The first Stage4 implementation should be a small-gate online ACT loop, not a full train run. It should prove that CLSTR changes evidence selection in a useful direction under the stronger executor.

## Gates

### Gate 0: Model and executor health

- Qwen3-14B loads on a compute node through sbatch.
- A single-task qwen-only AppWorld ReAct/function-calling smoke reaches code execution and official evaluation.
- Logs include model calls, API-doc discovery, environment outputs, and evaluation result.

### Gate 1: Qwen-only baseline

- Run a stratified dev10 qwen-only gate.
- Continue only if success and failure profile are clearly better than or comparable to the current Qwen3-8B qwen-only anchors.
- If the stronger model still cannot execute reliably, do not train Stage4.

### Gate 2: CLSTR evidence gate

- Same dev10 split, same executor, same model, only CLSTR evidence enabled.
- CLSTR must not increase invalid API calls or execution failures.
- Proceed only if CLSTR improves task success or produces a clean positive failure-profile shift.

### Gate 3: Stage4 smoke

- Train only on train split.
- Run a small number of train rollouts first to validate reward logging and checkpoint updates.
- Evaluate on dev10 after training.
- Proceed to dev57 only if dev10 beats qwen-only and CLSTR-base under the same executor.

### Gate 4: Larger evaluation

- Run dev57 only after Gate 3 passes.
- No full/test-style job unless dev57 shows a stable advantage and the user explicitly approves.

## Paper Positioning

The paper should not claim that CLSTR is an AppWorld-specific agent. The claim should be:

CLSTR learns continuous, coverage-aware skill routing and evidence selection over a large dynamic skill pool. Under a standard executor loop, CLSTR can improve downstream task execution by selecting and compressing useful skill evidence across steps.

The necessary comparisons are:

- qwen-only executor;
- SkillRouter-style single-step retrieval/evidence;
- CLSTR-base without Stage4;
- CLSTR-stage4 with executor feedback;
- ablations for dynamic skill append, verifier/compressor, transition/belief, and coverage-aware blending.

If AppWorld task success does not improve, AppWorld remains a diagnostic case study. The main paper should then lean on routing and multi-step routing benchmarks instead of claiming AppWorld task-success gains.

## Cost Controls

- No full jobs before small gates pass.
- Short gates are checked roughly every 15 minutes; long gates roughly every 30 minutes.
- No more than four simultaneous Slurm jobs.
- Full jobs require explicit user approval.
- Large model downloads use `HF_ENDPOINT=https://hf-mirror.com` and disabled proxy variables.
- Login/storage nodes must not run model inference.

## Success Criteria

The route is considered worth continuing if:

- Qwen3-14B qwen-only under official-style executor is clearly healthier than the current Qwen3-8B executor;
- CLSTR evidence improves dev10 over qwen-only without increasing execution/preflight failures;
- Stage4 improves over CLSTR-base on dev10 and then dev57 under the same executor;
- the same verifier/compressor and Stage4 objective can be described without AppWorld family-specific rules.

The route is blocked if:

- Qwen3-14B cannot be loaded or is too slow for small gates;
- qwen-only remains too weak to produce reliable AppWorld interactions;
- CLSTR evidence repeatedly hurts qwen-only on dev10;
- Stage4 gains only appear after adding task-family-specific rules.
