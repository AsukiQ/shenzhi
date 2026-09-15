# AppWorld Stage4 Postmortem 2026-06-12

## Scope

本报告只使用已有结果做无 GPU postmortem，没有提交新作业。目标是回答：

- Stage4 在 AppWorld dev57 上退化，主要是 CLSTR routing/transition 坏，还是下游 executor/handoff 坏？
- AppWorld 是否还能作为 Stage4 task-success 主证据？

对比路径：

- qwen-only + preflight2 dev57:
  `outputs/appworld_dynamic_multistep_executor_dev57/dev57_qwen_only_preflight_repair2_20260612_074250`
- Stage4 transition_blend + safe_metadata + preflight2 dev57:
  `outputs/appworld_dynamic_multistep_executor_dev57/dev57_stage4_transitionblend025_preflight_repair2_20260612_080201`
- regression/dev10 handoff gates:
  `schema_plan`, `adaptive_schema`, `gated_schema_v2`

## Aggregate Result

| Run | Success | Exec failures | Preflight failures | Avg steps |
|---|---:|---:|---:|---:|
| qwen-only + preflight2 dev57 | 16/57 | 39 | 0 | 2.333 |
| Stage4 + safe_metadata + preflight2 dev57 | 11/57 | 69 | 24 | 2.404 |
| Stage4 + schema_plan regression6 | 4/6 | 2 | 0 | 2.500 |
| Stage4 + gated_schema_v2 regression6 | 4/6 | 1 | 0 | 1.833 |
| Stage4 + gated_schema_v2 dev10 | 5/10 | 8 | 1 | 2.200 |

dev10 的 Stage4 正信号没有扩展到 dev57。`gated_schema_v2` 能修 regression6，但不能通过 dev10。

## Row-Level Outcome

dev57 qwen-only vs Stage4：

| Group | Count | Qwen avg steps | Stage4 avg steps | Qwen exec fails | Stage4 exec fails | Stage4 preflight fails |
|---|---:|---:|---:|---:|---:|---:|
| both_success | 10 | 1.7 | 1.3 | 0 | 0 | 0 |
| qwen_only_success | 6 | 2.0 | 3.0 | 0 | 15 | 9 |
| stage4_only_success | 1 | 3.0 | 1.0 | 1 | 0 | 0 |
| both_fail | 40 | 2.525 | 2.625 | 38 | 54 | 15 |

Stage4 不是完全无效：共同成功任务上平均步数更少，并且有 1 个唯一成功任务。但它在 dev57 中造成 6 个 qwen-only 成功任务退化，代价大于收益。

## Regression Cases

### Spotify unique-song count: `fac291d_1/2/3`

User goal: count unique songs across Spotify song, album, and playlist libraries.

Stage4 selected:

- `spotify-find-songs-based-on-play-count-*`
- `spotify-find-songs-by-metric-*`
- `spotify-extract-unique-artist-ids-from-songs-*`
- `spotify-extract-artist-ids-from-songs-in-playlists-*`

Observed failure:

- qwen-only solved all three.
- Stage4 safe_metadata failed all three.
- Stage4 code introduced `apis.spotify.show_song`, which was rejected in the original truncated-schema preflight run and was semantically unnecessary for qwen-only's simpler set-union solution.
- `schema_plan_not_allowlist` and `gated_schema_v2` recover these tasks on regression6.

Classification:

- Primary: handoff pollution from topically related but operationally misleading SkillX metadata.
- Secondary: retrieval/routing did not surface an exact unique-song-count skill; selected skills were plausible but harmful for executor generation.

### Spotify add-to-queue: `396c5a2_2/3`

User goal: add songs by an artist above a play-count threshold to the Spotify queue.

Stage4 selected:

- play-count skills, which are partly relevant;
- `spotify-like-songs-by-source-*`;
- `spotify-identify-songs-played-so-far-including-current-*`.

Observed failure:

- qwen-only solved both using direct search/add queue behavior.
- Stage4 generated invalid or misleading APIs such as skill-title-derived calls and queue-inspection logic.
- `schema_plan_not_allowlist` recovered one of the two in regression6; `gated_schema_v2` still failed both queue cases.

Classification:

- Primary: handoff pollution from SkillX title/name metadata.
- Secondary: selected context mixes relevant play-count evidence with wrong operation hints. This is not a clean proof that Stage4 transition is wrong, but it shows current dynamic SkillX handoff is unsafe.

### Simple Note export: `6c2c621_1`

User goal: export all Simple Note notes to a file-system directory as `.md`.

Stage4 selected:

- `simple-note-get-expense-shares-from-note-*`
- `simple-note-parse-expense-shares-from-note-*`
- `file-system-write-song-data-to-csv-*`
- one useful `simple-note-search-and-retrieve-notes-*`

Observed failure:

- qwen-only solved it.
- Stage4 safe_metadata failed by spending steps on login/directory setup and not completing the note export.
- `gated_schema_v2` recovered this exact regression6 row, but failed the dev10 sibling `6c2c621_1`, so the fix is not stable.

Classification:

- Mixed routing/handoff issue. The selected SkillX set contains a useful note retrieval skill but is dominated by unrelated expense-share and CSV skills.

## Stage4-Only Positive Case

### Spotify release-year count: `4ec8de5_3`

User goal: count songs from Spotify song and album libraries released before this year.

Stage4 selected play-count/song-inspection skills and solved in one step. qwen-only failed after three steps.

Classification:

- Real but narrow positive Stage4 signal. Skill context helped Qwen use song-detail APIs and finish efficiently.
- This supports keeping AppWorld as a case study, but one unique success is not enough to support AppWorld task-success as the Stage4 main result.

## Transition Usage

Stage4 transition was not silently disabled:

- Total Stage4 dev57 steps: 137
- `transition_scores_available=true`: 75 steps
- Transition scores appeared on step 1 and step 2.

However, all 6 qwen-only regression tasks were already contaminated at step 0, where transition has no previous action and falls back to Stage0/dynamic routing. Later transition reranking usually continues from a bad execution state.

Conclusion: the dev57 failure is not primarily "Stage4 transition tensor did not run". It is mostly a selected-context / handoff problem before Stage4 transition can help.

## Root Cause

The current AppWorld dynamic SkillX route has three interacting problems:

1. Stage4/dynamic retrieval often selects topically plausible but operationally misleading SkillX skills.
2. `safe_metadata` exposes SkillX names/descriptions strongly enough that Qwen treats them as plans or API hints.
3. Preflight can block nonexistent APIs, but it cannot reliably repair semantic drift after the prompt has already pushed Qwen toward the wrong strategy.

This means AppWorld dev57 is currently measuring a mixture of:

- CLSTR selection quality;
- SkillX text quality;
- handoff format;
- Qwen executor robustness;
- AppWorld API/schema repair behavior.

That mixture is too noisy for a clean Stage4 task-success claim.

## Decision

Do not run more AppWorld dev57/full jobs from `gated_schema_v2`.

For the paper:

- AppWorld can remain a dynamic SkillX / executor case study.
- AppWorld should not be the main proof of Stage4 value unless a later simple, non-task-specific route beats qwen-only and Stage2 on a larger gate.
- The main Stage4 claim should be moved to multi-step routing value on ToolBench/TrajectBench if the final AppWorld iteration does not produce a clean signal.

If one final AppWorld iteration is attempted, it should be:

- default to the previous best `safe_metadata + preflight2`;
- use schema-only only at a coarse high-pollution task-family level;
- no per-skill complex gate;
- gate only on regression6 and dev10 before any dev57.
