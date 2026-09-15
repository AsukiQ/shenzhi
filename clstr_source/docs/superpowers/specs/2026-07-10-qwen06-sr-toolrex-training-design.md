# Qwen 0.6B SR and ToolREx Training Design

**Date:** 2026-07-10
**Status:** Approved in conversation for implementation planning

## Goal

Prepare two scale-matched retrieve-and-rerank baselines on this repository's own
train-safe data:

1. a SkillRouter-method baseline using Qwen3-Embedding-0.6B and
   Qwen3-Reranker-0.6B; and
2. a ToolREx-method baseline using Qwen3-Embedding-0.6B and
   Qwen3-Reranker-0.6B over expanded tool documents.

The preparation includes data builders, model-specific training code, tests,
restartable checkpoints, Slurm launchers, and small GPU smoke runs. It does not
include full training submissions.

## Scope Boundary

In scope:

- full-parameter tuning for all four 0.6B training stages;
- paper-aligned SR hard-negative mining, false-negative filtering, and listwise
  reranking;
- ToolREx document expansion over this project's own tools;
- Qwen3-14B as the initial offline expansion and consistency-judging backend;
- train/eval leakage protection before any training artifact is produced;
- one-epoch-capable and step-capped training modes;
- resumable optimizer, scheduler, scaler, RNG, and model state;
- monitored GPU smoke jobs and checkpoint reload tests; and
- compatibility with the existing generic route evaluator.

Out of scope:

- CLSTR model, Stage0/1/2/4, causal-memory, or benchmark-controller changes;
- full SR or ToolREx training jobs;
- full expansion of all tools during this preparation pass;
- benchmark table updates or claims of completed benchmark performance;
- modifying or reusing the currently dirty causal-mainline files; and
- presenting this work as official SkillRouter or official ToolREx training.

## Method Labels

Reports and tables must use labels equivalent to:

- `Qwen3-0.6B SR method on our train-safe data`; and
- `Qwen3-0.6B ToolREx method on our train-safe data`.

They must not use `official SkillRouter`, `official ToolREx`, or `exact paper
reproduction`. The implementation follows the methods while deliberately
controlling data and model scale for the paper comparison in this repository.

## Evidence Behind the Redesign

The current BGE SR/ToolREx code is useful infrastructure but is not a faithful
implementation of either paper:

- SR mining currently uses semantic neighbors followed by random fill. It omits
  the paper's BM25 and same-taxonomy components.
- SR mining currently excludes only exact positive IDs. It omits name, body
  overlap, and embedding-similarity false-negative filters.
- BGE SR inputs use CLS pooling, right padding, raw queries, and short field
  limits. Qwen SR requires last-token pooling, left padding, and an instruction
  query format.
- BGE SR-Rank uses a sequence-classification model. Qwen3-Reranker is a causal
  language model scored through two relevance-label token logits.
- The current ToolREx builder concatenates ToolRet and ToolBench records but
  does not construct `tool_profile` document expansions.
- The current Tool-Rank objective is pointwise BCE over a BGE classifier and
  has already shown negative-class collapse.
- The current ToolREx unified root consumes all normalized ToolBench-G3 rows,
  whose source split is `train_or_released_g3`; it is therefore not an
  acceptable main-table training root without a protected-eval exclusion pass.

No new Qwen training path may silently call a BGE-named loader, hard-code CLS
pooling, or load Qwen3-Reranker through
`AutoModelForSequenceClassification`.

## Data Ownership and Leakage Boundary

The source datasets remain immutable. New versioned data roots are built under:

- `data/qwen06_sr_method_v1/`; and
- `data/qwen06_toolrex_method_v1/`.

Every builder first creates a protected-evaluation manifest. At minimum it
contains:

- normalized benchmark name;
- query/task/trajectory IDs when available;
- normalized query-text SHA-256 hashes;
- source paths and split labels; and
- positive skill/tool IDs used by evaluation.

Training rows are grouped by their original query/task ID before splitting.
No group may appear in more than one of train, validation, or protected eval.
Exact normalized-text overlap with protected eval is a hard failure. Positive
supervision attached to a protected evaluation query is excluded.

Evaluation documents remain visible in the retrieval inventory because a
retrieval benchmark exposes its corpus. Query-independent serialization and
document expansion of those documents are allowed. Evaluation queries, labels,
and query-conditioned fields are never inputs to expansion or training.

The data report records source counts, retained counts, exclusion reasons,
query-hash overlap, protected ID overlap, skill coverage, and deterministic
split fingerprints. Training launchers refuse to run unless the leakage report
has `status=ok`.

## Canonical Tool Documents

The raw tool serializer preserves information already present in the project's
records:

- tool ID and name;
- description and executor description;
- environment/category and source;
- input schema and output schema; and
- body or skill markdown when present.

The serializer is deterministic and field-aware. It does not rely on the
current `name | description | body` helper because many tool records have an
empty body but informative schemas.

An expanded ToolREx document is:

```text
<canonical original tool document>

Tool profile:
Function: ...
Tags: ...
When to use: ...        # only when supported
Limitations: ...        # only when supported
```

Both the original and expanded views are retained so the no-expansion ablation
uses the same queries, labels, split, and training budget.

## ToolREx Document Expansion

Expansion is an offline, query-independent data step. The implementation uses a
pluggable generator/judge interface. The initial local configuration points to
`models/Qwen3-14B`; the 14B model never participates in 0.6B training or
benchmark inference.

The generator receives only the canonical original document and emits a JSON
object with:

- required `function`, at most 20 words;
- required `tags`, three to five normalized strings;
- optional `when_to_use`, only when grounded in the source; and
- optional `limitations`, only when grounded in the source.

The pipeline applies deterministic JSON-schema and field-length validation,
then a separate judging prompt. Because the initial smoke may use Qwen3-14B for
both roles, its report explicitly says `self_judged=true`; this is not described
as independent model validation. Failed rows receive one deterministic retry
and are then excluded with an auditable reason. Raw generations are retained.

Each accepted profile stores:

- tool ID;
- original-document SHA-256;
- prompt version;
- generator and judge model paths;
- raw generation;
- normalized profile;
- validation and judging decisions; and
- creation timestamp and random seed.

Resume is keyed by tool ID plus original-document hash. Changed documents are
regenerated; unchanged accepted rows are not recomputed.

The preparation-pass GPU smoke expands and judges only a tiny fixed subset. It
does not generate the full roughly 30K-profile corpus.

## SR Encoder Data and Mining

The primary comparison uses this project's own train-safe query-skill pairs,
not the SkillRouter paper's released model or unavailable private training
code. Synthetic query generation is not added to the primary comparison because
the repository already has supervised queries and the comparison must keep its
training source controlled. This deviation is recorded in every report.

Skills use full canonical text. For every query, the miner targets ten negative
skills with the paper's composition:

- four semantic negatives from the base Qwen embedding top-50;
- three BM25 lexical negatives;
- two same-taxonomy negatives; and
- one random negative from a different taxonomy.

Taxonomy resolves in order from explicit category, environment, then source.
If a bucket cannot fill its quota, deterministic fallback sampling fills the
shortfall and records the reason and replacement source.

Every candidate passes the following false-negative filters in order:

1. exact canonical ID and alias exclusion;
2. normalized-name equality with any positive;
3. body/profile trigram Jaccard greater than 0.6; and
4. base-embedding cosine similarity greater than 0.92.

The miner saves candidates, source labels, raw scores, filter decisions,
fallback decisions, and corpus/model fingerprints. It supports resume without
re-encoding completed shards.

## SR-Emb Training

SR-Emb uses `models/Qwen3-Embedding-0.6B` through `AutoModel` with:

- left padding and last-token pooling;
- L2-normalized 1024-dimensional embeddings;
- query instruction:
  `Given a task description, retrieve the most relevant skill document that
  would help an agent complete the task`;
- raw query cap of 1,500 characters;
- description cap of 300 characters;
- body/profile cap of 2,500 characters;
- tokenized maximum length 2,048;
- in-batch InfoNCE with temperature 0.05;
- explicit mined negatives in the denominator;
- full-parameter tuning, BF16 autocast, gradient checkpointing, and
  `use_cache=False`;
- AdamW with weight decay 0.01, cosine decay, and 5% warmup; and
- one-epoch mode plus a `max_steps` smoke/debug cap.

The paper configuration of micro-batch 8 and gradient accumulation 4 is the
default full-run recipe. Smoke launchers override it only to fit the small test.

## SR-Rank Training

The trained SR-Emb checkpoint retrieves top-20 candidates from the complete
training inventory. Groups without any positive in top-20 are counted as
retriever misses and excluded from reranker optimization; they remain visible
in coverage reports.

The same false-negative filters used by SR-Emb are applied to candidate groups.
The full-text reranker format uses:

- query cap 1,500 characters;
- description cap 500 characters;
- body/profile cap 2,000 characters;
- maximum token length 4,096; and
- the Qwen3 relevance-judging chat prefix and suffix.

`models/Qwen3-Reranker-0.6B` is loaded through
`AutoModelForCausalLM`. The relevant and irrelevant answer token IDs are read
from model metadata or explicitly configured and are validated as distinct
single tokens. Candidate scores are relevant-logit minus irrelevant-logit.

Training applies listwise cross-entropy across each top-20 group with
temperature 1.0. Multiple valid positives use negative log probability mass
over all positive candidates. Defaults match the paper recipe where specified:
one group per micro-batch, gradient accumulation 16, LR 1e-5, one epoch,
AdamW weight decay 0.01, cosine decay, 5% warmup, BF16, and gradient
checkpointing.

## Tool-Embed Training

Tool-Embed uses train-safe query-tool pairs drawn from this project's ToolRet
and ToolBench-G3 sources after protected-eval exclusion. A deterministic,
source/category-stratified sample supplies 50K training examples when at least
that many valid pairs are available.

For each positive query-tool pair, five tools belonging to other queries are
sampled as negatives. Canonical aliases and any known positives for the query
are excluded. Training uses the expanded document view for the primary run and
the original view for the matched ablation.

The encoder is `models/Qwen3-Embedding-0.6B`, with left padding, last-token
pooling, L2 normalization, InfoNCE, full-parameter tuning, one epoch, BF16,
gradient checkpointing, `use_cache=False`, cosine scheduling, and restartable
state. Query instruction use is explicit in the serialized configuration and
cannot silently differ between training and evaluation.

## Tool-Rank Training

Tool-Rank training data is built from the trained Tool-Embed top-100 candidates
and the protected train split. The target is 200K labeled query-document pairs
when enough valid data exists. Sampling is query-group aware and label balanced
so a batch cannot contain only negatives. Candidate construction and label
balance are persisted in a report.

The model is `models/Qwen3-Reranker-0.6B`, loaded through
`AutoModelForCausalLM`. It uses the ToolREx query/tool-document prompt and a
two-token relevance cross-entropy objective. The score used for ranking is the
normalized relevant probability or its monotonic logit difference.

The user's requested full-parameter 0.6B training is an intentional deviation
from the ToolREx paper's 4B LoRA configuration. Reports state that deviation.
Inference reranks top-100, matching the ToolREx evaluation protocol rather than
the SR top-20 protocol.

Held-out evaluation is query-group based and reports both classification and
ranking health:

- positive and negative accuracy;
- positive and negative mean score;
- score separation;
- Hit/Recall at 1, 5, and 10; and
- MRR and NDCG@10.

Overall accuracy alone is never a completion gate. A run with zero positive
accuracy, missing either class, non-finite gradients, or non-positive held-out
score separation is `action_required`, not `ok`.

## Shared Checkpoint and Monitoring Contract

All four trainers write:

- `progress.json` with phase, step/epoch, elapsed time, and latest metrics;
- `training_metrics.jsonl`;
- `loss_curve.svg` and `diagnostic_curves.svg`;
- `metrics.json` or `blocker_report.json`;
- periodic Hugging Face model/tokenizer directories;
- `training_state.pt` containing optimizer, scheduler, scaler when used, step,
  epoch, data cursor, RNG states, and last metrics;
- `checkpoints/latest.json`; and
- checkpoint-retention reports.

Resume validates the dataset fingerprint, model family, pooling, prompt
version, objective, and candidate protocol before loading optimizer state.
Mismatches fail loudly instead of partially restoring a checkpoint.

## CLI and Slurm Surfaces

Separate CLIs and Slurm launchers are provided for:

1. protected-eval manifest and train-safe data construction;
2. ToolREx profile generation/judging;
3. SR negative mining and SR-Emb training;
4. SR top-20 construction and SR-Rank training;
5. Tool-Embed training;
6. Tool-Rank top-100 construction and training; and
7. checkpoint-compatible route-eval smoke.

Launchers default to local model paths, create a unique output directory, tee
stdout, expose resume paths, and never submit a dependent full chain. GPU work
is performed through Slurm; login-node work is limited to unit tests and light
data inspection.

## Test and Smoke Strategy

Implementation follows test-first development. CPU/unit coverage includes:

- canonical original and expanded serialization;
- protected-eval exclusion and overlap failure;
- deterministic grouped splits and sampling;
- exact SR negative-source quotas and fallback accounting;
- all three false-negative filters;
- Qwen last-token pooling and instruction formatting;
- relevant/irrelevant token score extraction with retained gradients;
- multi-positive SR listwise loss;
- label-balanced Tool-Rank sampling and binary token CE;
- checkpoint metadata and incompatible-resume rejection;
- CLI help and Slurm argument wiring; and
- generic route-evaluator recognition of fine-tuned checkpoints.

GPU smoke coverage includes:

- Qwen3-14B expansion and self-judging for a tiny document subset;
- one or two optimizer steps for each 0.6B trainer;
- finite loss and gradient checks;
- proof that intended trainable parameters change;
- checkpoint save and reload;
- resume for at least one additional step; and
- a tiny retrieval/rerank evaluation using the reloaded checkpoints.

Smoke success establishes training readiness only. It is not benchmark
evidence.

## Acceptance Criteria

Preparation is complete when:

- all new targeted tests pass without touching CLSTR causal tests;
- all launchers pass shell syntax validation;
- the data builders produce `status=ok` leakage and data manifests on fixtures;
- all four Qwen trainers can save and reload a structurally valid checkpoint;
- the tiny GPU smokes complete with finite gradients and no class collapse;
- the generic route evaluator loads both fine-tuned model families; and
- no full training or full benchmark job has been submitted.

## Intentional Paper Deviations

The following are explicit experimental choices, not implementation mistakes:

- all supervision comes from this repository's train-safe data rather than the
  papers' released/private training corpora;
- SR uses existing supervised queries rather than generating the paper's 37,979
  synthetic requests;
- initial ToolREx expansion and judging use local Qwen3-14B rather than the
  paper's Qwen3-32B, Llama-3.1-70B, and GPT-4o sequence;
- Tool-Rank uses Qwen3-Reranker-0.6B full-parameter tuning rather than 4B LoRA;
  and
- benchmark evaluation remains this repository's global/local pool protocol.

Every report carries these deviations so the resulting comparison is described
as a controlled method comparison on our data, not an exact reproduction.
