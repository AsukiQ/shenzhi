# ToolBench-G3 Official SkillRouter Comparison

## Scope

This comparison uses the official SkillRouter repository for the SkillRouter baseline:

- official repo: `/data/home/scyb713/run/xzf/AAAI/autodl-tmp/skillrouter`
- entrypoint: `python -m src.run_open_model_eval`
- models: cached official `SkillRouter-Embedding-0.6B` and `SkillRouter-Reranker-0.6B`

The fair CLSTR comparison is trajectory next-skill routing, not static query retrieval:

- task: `ToolBench-G3 state_text -> next_skill_id`
- skill pool: full CLSTR v4.2 skill pool, 37,617 skills
- eval split: 1362 held-out rows split by `trajectory_id`
- CLSTR checkpoint: `outputs/clstr_unified_stage2_v4_2_progressive_final_top350_inventory64_listwise_rankprior_l025/checkpoints/clstr_full_base-step10000.pt`

We also ran an official-compatible finetuned SkillRouter baseline because the frozen official model is no-shot to this trajectory next-skill task. The public SkillRouter repo provides model/eval code but no public training entrypoint, so this baseline keeps official formatting, official metrics, official embedding/reranker checkpoints, and trains a lightweight query-side bi-encoder adapter for 1024 steps on the fixed train split. The official reranker is kept frozen.

## Results

| Method | Rows | Hit/Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Official SkillRouter retrieval | 1362 | 0.0228 | n/a | 0.0656 |
| Official SkillRouter pipeline | 1362 | 0.0286 | n/a | 0.0848 |
| Official-compatible SkillRouter finetune retrieval | 1362 | 0.1344 | 0.3833 FullCoverage@5 | 0.2390 |
| Official-compatible SkillRouter finetune pipeline | 1362 | 0.0433 | 0.2584 FullCoverage@5 | 0.1316 |
| Full CLSTR Stage2, all rows missing-as-zero | 1362 | 0.1821 | 0.4750 | 0.3095 |
| Full CLSTR Stage2, retained diagnostic only | 1003 transition rows | 0.2473 | 0.6451 | 0.4202 |
| Full CLSTR route, Stage4 eval-prefix online memory, all rows missing-as-zero | 1362 | 0.1542 | 0.2959 | 0.2290 |
| Full CLSTR route, Stage4 eval-prefix online memory, retained diagnostic only | 1137 Stage4 rows | 0.1847 | 0.3544 | 0.2744 |
| CLSTR Stage2 scorer + Stage4 memory, all rows missing-as-zero | 1362 | 0.2276 | 0.5308 | 0.3623 |
| CLSTR Stage2 scorer + Stage4 memory, retained diagnostic only | 1139 Stage4 rows | 0.2722 | 0.6348 | 0.4332 |
| Qwen3-14B generative rerank over Stage0 top-20, all rows missing-as-zero | 1362 | 0.1050 | 0.2555 | 0.1788 |
| Qwen3-14B generative rerank over Stage0 top-20, retained diagnostic only | 611 rows | 0.2340 | 0.5696 | 0.3985 |
| Qwen3-Reranker-8B zero-shot rerank over Stage0 top-100, all rows missing-as-zero | 1362 | 0.0617 | 0.3047 | 0.1746 |
| Qwen3-Reranker-8B zero-shot rerank over Stage0 top-100, retained diagnostic only | 939 rows | 0.0895 | 0.4420 | 0.2533 |

SkillRouter official metrics are `Hit@1`, `MRR@10`, and `Recall@10/20`; CLSTR reports candidate-set `transition_skill_recall@1`, `transition_skill_recall@5`, and `transition_skill_mrr`. The primary CLSTR row above counts all missing-candidate cases as zero over the full 1362-row denominator. The retained diagnostic row is only for model-in-candidate analysis and should not be used as the main end-to-end number.

Full-pool SkillRouter finetune details:

- output: `outputs/toolbench_g3_official_skillrouter_comparison/skillrouter_finetune_full_qproj_step1024_20260622_155615`
- train/eval/skills: 5329 / 1362 / 37617
- finetuned parameters: query-side projection adapter only; document projection and official reranker frozen
- retrieval metrics: `Hit@1=0.1344`, `MRR@10=0.2390`, `Recall@10=0.5044`, `Recall@20=0.6175`
- pipeline metrics: `Hit@1=0.0433`, `MRR@10=0.1316`, `Recall@10=0.4023`, `Recall@20=0.6175`
- observation: finetuning greatly improves retrieval over frozen SkillRouter, but the frozen reranker hurts top-rank ordering on this trajectory next-skill task.

Full CLSTR route clean-split details:

- output: `outputs/toolbench_g3_official_skillrouter_comparison/clstr_full_route_eval_evalprefix_full_20260622_172626`
- route: Stage0 top-350 handoff + Stage2 prior-residual checkpoint + Stage4 logged-online memory
- feedback mode: `eval_prefix`; each held-out trajectory row can only use earlier logged feedback from the same held-out trajectory, not future rows
- source/eval rows: `1362`; retained Stage4 candidate rows: `1137`
- Stage4 memory fired on `179` retained rows and boosted the positive on `85` rows
- retained prior metrics before Stage4 memory: r@1 `0.1469`, r@5 `0.3166`, MRR `0.2378`
- retained Stage4 metrics: r@1 `0.1847`, r@5 `0.3544`, MRR `0.2744`
- strict missing-as-zero Stage4 metrics over all 1362 rows: r@1 `0.1542`, r@5 `0.2959`, MRR `0.2290`
- strict Stage4 delta vs pre-memory prior: r@1 `+0.0316`, r@5 `+0.0316`, MRR `+0.0305`

Important caveat: the earlier CLSTR Stage2 comparison uses the Stage2 real-top-M evaluator and its transition/inventory-mask path. The new full-route row uses the Stage4 logged-online memory path. These are both useful, but they are not identical scorer surfaces. The full-route result is the appropriate row for "complete CLSTR" claims; the Stage2 row should be treated as a strong ablation/diagnostic unless we explicitly wire the same inventory-masked Stage2 scorer into Stage4.

Same-scorer Stage2+Stage4 memory follow-up:

- output: `outputs/toolbench_g3_official_skillrouter_comparison/clstr_stage2_memory_rerank_full_20260623_024634`
- route: Stage0 top-350 handoff + Stage2 real-top-M scorer + eval-prefix Stage4 online memory residual
- source/eval rows: `1362`; retained Stage4 candidate rows: `1139`
- Stage0 next-positive coverage@350: `0.8363`
- Stage4 memory fired on `179` rows and boosted the positive on `85` rows
- strict Stage2-only metrics over all 1362 rows: r@1 `0.2012`, r@5 `0.5147`, MRR `0.3400`
- strict Stage2+memory metrics over all 1362 rows: r@1 `0.2276`, r@5 `0.5308`, MRR `0.3623`
- strict Stage4 memory delta on the same scorer: r@1 `+0.0264`, r@5 `+0.0162`, MRR `+0.0223`
- retained Stage2-only metrics: r@1 `0.2406`, r@5 `0.6155`, MRR `0.4065`
- retained Stage2+memory metrics: r@1 `0.2722`, r@5 `0.6348`, MRR `0.4332`

This follow-up resolves the scorer-surface caveat above: Stage4 memory is positive when attached to the strong Stage2 real-top-M scorer. The previous full-route row remains useful as a separate implementation-path diagnostic, but the same-scorer Stage2+memory row is the cleaner evidence for Stage4's incremental value.

Qwen3-14B generative rerank baseline:

- output: `outputs/toolbench_g3_official_skillrouter_comparison/qwen3_14b__skill_rerank_top20_full_20260623_141723`
- route: Stage0 top-20 candidate handoff + Qwen3-14B JSON/list ranking over candidate skill previews
- source/eval rows: `1362`; retained rows with next positive inside the evaluated candidate set: `611`
- Stage0 top-20 next-positive retained fraction: `0.4486`
- parse health: empty parse rows `0`; mean generated rank count `19.9067` out of `20`
- strict missing-as-zero metrics over all 1362 rows: r@1 `0.1050`, r@5 `0.2555`, MRR `0.1788`
- retained diagnostic metrics: r@1 `0.2340`, r@5 `0.5696`, MRR `0.3985`

This baseline is not a full-pool Qwen generator. It measures whether a strong generative Qwen3 model can rerank the Stage0 top-20 candidates from the same trajectory state. The all-row missing-as-zero row is the main number because top-20 candidate coverage is part of the end-to-end route.

Qwen3-Reranker-8B zero-shot rerank baseline:

- output: `outputs/toolbench_g3_official_skillrouter_comparison/qwen3_reranker_8b__top100_full_20260623_163827`
- route: Stage0 top-100 candidate handoff + Qwen3-Reranker-8B yes/no logit scoring for `(trajectory state, candidate skill text)` pairs
- source/eval rows: `1362`; retained rows with next positive inside top-100 candidates: `939`
- Stage0 top-100 retained fraction: `0.6894`
- strict missing-as-zero metrics over all 1362 rows: r@1 `0.0617`, r@5 `0.3047`, MRR `0.1746`
- retained diagnostic metrics: r@1 `0.0895`, r@5 `0.4420`, MRR `0.2533`
- runtime: full job `95464` completed in `00:24:18`; scoring time was `1411.2s`, about `1.50s` per retained row at top-100/batch-8 on H200
- sanity check: reversing score direction produced near-zero retained r@1/r@5, so the low result is not caused by an inverted score convention

This is a stronger off-the-shelf rank-model baseline than Qwen3-14B-Instruct prompting, but it is still zero-shot to trajectory next-skill routing and uses CLSTR Stage0 as its candidate generator. It improves strict r@5 over Qwen3-14B top-20 because top-100 has higher candidate coverage, but its in-candidate top-rank ordering is weak on this task.

CLSTR skipped 192 rows where the current positive was missing from Stage0 top-350. Among the 1170 retained rows, 167 rows had the next positive missing from candidates, leaving 1003 rows with transition supervision inside the candidate set. No gold candidate injection was used.

Additional CLSTR diagnostics:

- current positive coverage@350: 0.8590
- next positive coverage@350: 0.8573
- transition metric rows: 1003 / 1362
- Stage0 prior MRR: 0.4119
- Stage2 transition MRR: 0.4202
- Stage2 delta vs Stage0 prior MRR: +0.0099
- Stage2 delta vs Stage0 prior recall@5: +0.0306

## Key Files

- exported shared data: `outputs/toolbench_g3_official_skillrouter_comparison/trajectory_full_fullpool_data`
- official SkillRouter result: `outputs/toolbench_g3_official_skillrouter_comparison/official_skillrouter_trajectory_full_fullpool/summary.json`
- official-compatible finetuned SkillRouter result: `outputs/toolbench_g3_official_skillrouter_comparison/skillrouter_finetune_full_qproj_step1024_20260622_155615/summary.json`
- CLSTR result: `outputs/toolbench_g3_official_skillrouter_comparison/clstr_stage2_trajectory_full_fullpool/real_topm_eval_report.json`
- full CLSTR route clean-split result: `outputs/toolbench_g3_official_skillrouter_comparison/clstr_full_route_eval_evalprefix_full_20260622_172626/full_clstr_route_eval_report.json`
- same-scorer CLSTR Stage2+Stage4 memory result: `outputs/toolbench_g3_official_skillrouter_comparison/clstr_stage2_memory_rerank_full_20260623_024634/stage2_memory_rerank_eval_report.json`
- Qwen3-14B generative rerank result: `outputs/toolbench_g3_official_skillrouter_comparison/qwen3_14b__skill_rerank_top20_full_20260623_141723/qwen_skill_rerank_eval_report.json`
- Qwen3-Reranker-8B zero-shot rerank result: `outputs/toolbench_g3_official_skillrouter_comparison/qwen3_reranker_8b__top100_full_20260623_163827/qwen_reranker_eval_report.json`
- capped smoke data: `outputs/toolbench_g3_official_skillrouter_comparison/trajectory_capped256_fullpool_data`

## Interpretation

This result supports the claim that CLSTR is useful for state-conditioned next-skill routing over a large pool. The strongest comparison is against the official SkillRouter pipeline on exactly the same held-out ToolBench-G3 trajectory states and the same 37,617-skill pool.

The result should be presented carefully: SkillRouter is a single-step query-to-skill router, while this experiment evaluates a trajectory-conditioned next-skill routing task where CLSTR's sequential state modeling is designed to help.
