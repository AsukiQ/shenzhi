# Matched SR / ToolREx Retraining

This path retrains the BGE SkillRouter and ToolREx baselines on the same
approved vNext train/dev release used by matched-data CLSTR. It does not read
raw trajectory unions or create a new random split.

The matched union and vNext views must be rebuilt with the finalized CLSTR
data code before exporting this corpus. In particular, do not reuse the old
`outputs/matched_multibench_vnext_views_v1` artifact: it predates the successful
Tau2 rollout, ToolSandbox DAG, and explicit model-input contracts. The exporter
requires the finalized `clstr_vnext_semantic_v1` manifest and fails closed on
older releases.

## 1. Build the shared corpus

```bash
python scripts/build_matched_baseline_corpus.py \
  --vnext_manifest_path /path/to/vnext/manifest.json \
  --output_dir outputs/matched_baseline_corpus/v1
```

The exporter verifies every source SHA256 and writes:

- `selected_skills.jsonl`
- `train_queries.jsonl`
- `eval_queries.jsonl`
- `manifest.json`

Targets are taken only from `required_tool_set_skill_ids` and
`current_state_route_set_skill_ids`. The training query is the same compact
causal skill/action state used by vNext Stage0. Runtime-local catalogs are
preserved; the global catalog is represented implicitly to avoid repeating
67k IDs on every row.

## 2. Full embedding retraining

Set `PREPARED_CORPUS_DIR` on either existing launcher. The launchers now derive
`PROJECT_ROOT` from their own checkout; an explicit override remains supported.
For a full SR run, override every smoke-sensitive cap explicitly:

```bash
PROJECT_ROOT="$(pwd)" \
PREPARED_CORPUS_DIR=outputs/matched_baseline_corpus/v1 \
OUTPUT_DIR=outputs/bge_sr_embedding_full_train/matched_v1 \
MAX_ROWS=ALL MAX_STEPS=2000 BATCH_SIZE=64 \
NEGATIVES_PER_QUERY=7 HARD_NEGATIVE_TOP_K=32 \
MAX_HARD_NEGATIVE_QUERIES=20000 MAX_LENGTH=512 \
TORCH_DTYPE=float32 USE_BF16_AUTOCAST=1 MINING_USE_BF16_AUTOCAST=1 \
sbatch scripts/sbatch/run_bge_sr_embedding_full_train.sh
```

```bash
PROJECT_ROOT="$(pwd)" \
PREPARED_CORPUS_DIR=outputs/matched_baseline_corpus/v1 \
OUTPUT_DIR=outputs/bge_toolrex_embedding_full_train/matched_v1 \
MAX_ROWS=ALL MAX_STEPS=2000 BATCH_SIZE=64 \
NEGATIVES_PER_QUERY=5 MAX_LENGTH=512 \
TORCH_DTYPE=float32 USE_BF16_AUTOCAST=1 \
sbatch scripts/sbatch/run_bge_toolrex_embedding_full_train.sh
```

Prepared mode uses deterministic 50/50 retrieval/static-route sampling and
round-robins sources inside each capability. The in-batch objective masks
tools outside each query's runtime catalog and uses a multi-positive
log-sum-exp target. Legacy `DATA_ROOT` behavior remains unchanged when
`PREPARED_CORPUS_DIR` is absent. `MAX_ROWS` and `EVAL_ROWS` are deterministic,
capability/source-stratified caps in prepared mode; leave `MAX_ROWS=ALL` for
full training. `EVAL_ROWS` bounds the retained reranker dev artifact, and
embedding monitoring draws one deterministic in-batch slice from that retained
set rather than exhaustively scoring every row. Final benchmark evaluation
remains uncapped. Capped semantic hard-negative mining first preserves the
training schedule's retrieval/static-route split, then round-robins sources
inside each capability.

The SR reranker launcher intentionally retains smoke defaults (`20` steps and
capped query/group counts). A full run must explicitly override
`MAX_STEPS`, `MAX_TRAIN_QUERIES`, `MAX_EVAL_QUERIES`, `MAX_TRAIN_GROUPS`, and
`MAX_EVAL_GROUPS`; do not treat a default launcher invocation as the paper
reranker result. The ToolREx reranker launcher defaults to its full pairwise
schedule but still requires explicit embedding output/checkpoint paths.

## 3. Exact reranker candidate cache

Both reranker launchers enable a shared cache by default at:

```text
outputs/shared_exact_candidate_cache
```

Override it with `CANDIDATE_CACHE_DIR`. The cache stores exact dense top-k
rankings and is keyed by inference checkpoint contents, corpus contents,
pooling, query mode, padding side, dtype, max length, top-k, query caps, and
ranking device. Initial exact dense ranking runs on the allocated GPU with TF32
disabled; a valid cache hit skips encoder loading and ranking entirely.
Contract mismatch, illegal per-query candidates, truncated rankings, or cache
corruption rebuilds or rejects the entry. Encoder/reranker training embeddings
are never cached while their weights are changing. New checkpoints also bind a
training contract; resume fails closed if corpus contents, objective, sampling,
or optimization-critical settings change. Older checkpoints without that
contract remain loadable but are reported as `legacy_unverified`.
