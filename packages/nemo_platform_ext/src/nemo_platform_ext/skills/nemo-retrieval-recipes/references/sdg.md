<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Stage 0+1: generate and prepare retrieval data

Stage 0 (`retrieval-generate`) turns a document corpus into judged Q&A pairs.
Stage 1 (`retrieval-prepare`) converts those pairs into `training.jsonl` plus a
BEIR `eval_beir/` split, and optionally mines hard negatives on GPU.

## Prerequisites

- NeMo CLI access to a running platform and the target workspace.
- A corpus fileset or pinned `hf://` dataset URI.
- Inference Gateway providers for the chat and embedding model roles.
- For hard-negative mining, a model entity with an attached encoder fileset and
  a GPU-capable execution profile.

## Resolve the base URL first

```bash
BASE_URL="${NMP_BASE_URL:-http://localhost:8080}"
curl -fsS "${BASE_URL%/}/health/ready"
```

`NMP_BASE_URL` is only an override for a remote platform. Empty or non-JSON CLI
output means the platform is unreachable; recheck health and retry rather than
treating it as a missing entity.

## Point Stage 0 at the corpus

`corpus` takes a fileset ref (`workspace/fileset`, optionally `#subdir`) or an
`hf://` URI. Pin a revision for anything reproducible:

```bash
nemo files filesets create my-docs --workspace default --purpose dataset --exist-ok
nemo files upload /path/to/docs/ my-docs --workspace default
```

Corpus reach and chunking: `file_extensions`, `min_text_length`,
`sentences_per_chunk`, `num_sections`, and `max_artifacts_per_type`. Leave
`num_files` unset to use the whole corpus; setting it to 1 is a plumbing check
only, because a single document can place every query in the test split and
leave train empty.

## Control generation volume

`num_pairs` (default 7) is queries per artifact. `query_counts` and
`reasoning_counts` must use their exact key sets **and each sum to `num_pairs`**,
which is validated at submit:

| Field | Keys | Default |
|---|---|---|
| `query_counts` | `multi_hop`, `structural`, `contextual` | `3 / 2 / 2` |
| `reasoning_counts` | `factual`, `relational`, `inferential`, `temporal`, `procedural`, `causal`, `visual` | `1` each |

Raising `num_pairs` without rebalancing both dicts fails the spec. Other knobs:
`min_hops` / `max_hops`, `min_complexity`, `similarity_threshold`, `buffer_size`.

## Wire the model roles

All four roles go through Inference Gateway. Use `provider` plus optional
`chat_provider` / `embed_provider` overrides; never `NVIDIA_API_KEY` on the job.
Read `model_provider_id` from a READY deployment instead of assuming the
deployment name is the provider name.

```bash
nemo inference deployments get <deployment> --workspace default -f json
```

## Iterate with preview, then generate

`retrieval-preview` runs the same two providers without a fileset job. Use it to
sanity-check extraction and judging before paying for a full Stage 0 run.

```bash
nemo data-designer retrieval-preview --spec '{
  "generate": {
    "corpus": "default/my-docs",
    "provider": "default/nvidia-build",
    "artifact_extraction_model": "nvidia/nemotron-3-nano-30b-a3b",
    "qa_generation_model": "nvidia/nemotron-3-nano-30b-a3b",
    "quality_judge_model": "nvidia/nemotron-3-nano-30b-a3b",
    "embed_model": "nvidia/nemotron-3-embed-1b"
  },
  "num_records": 1
}'
```

Stage 0 writes `generation_result.json` plus Q&A JSONL to its fileset.

## Shape the Stage 1 split

Stage 1 defaults: `quality_threshold: 7.0`, `train_ratio: 0.8`, `val_ratio: 0.0`,
`seed: 42`, `max_pos_docs: 5`, `split_strategy: random`. Raise `train_ratio` on a
small corpus; lower `quality_threshold` when the judge rejects too much. Prefer
`retrieval-run` to chain both stages in one job.

```bash
nemo data-designer retrieval-run --workspace default --spec '{
  "generate": {
    "corpus": "default/my-docs",
    "provider": "default/nvidia-build",
    "artifact_extraction_model": "nvidia/nemotron-3-nano-30b-a3b",
    "qa_generation_model": "nvidia/nemotron-3-nano-30b-a3b",
    "quality_judge_model": "nvidia/nemotron-3-nano-30b-a3b",
    "embed_model": "nvidia/nemotron-3-embed-1b"
  },
  "prepare": {
    "enable_mining": true,
    "model": "default/nemotron-3-embed-1b"
  }
}'
```

Hard-negative mining (`enable_mining: true`) is a GPU step and needs `model` to
be a model entity with an attached encoder fileset. Never mine after conversion
produced an empty train split.

Mining embeds every train query and the whole corpus on one GPU, so budget
roughly an hour for a ~190k-row split at the default batch of 16. Raise
`mining.query_embedding_batch_size` / `mining.document_embedding_batch_size`
(64–128 fits a 48 GB card) before assuming a long run is hung.

Convert-only (`enable_mining: false`, the default) writes `training.jsonl` with
`neg_doc: []`. Automodel `bi_encoder` / `cross_encoder` still samples
`train_n_passages - 1` negatives (default **4**) and fails with
`neg_doc must contain at least 1 document to sample N negatives`. Mine before
any encoder fine-tune. To fill an existing convert-only split without
regenerating frozen `eval_beir`, point `train_input_file` at that fileset and
set `enable_mining: true`. `nemo files list` must still show `additional/corpus/`
(convert-only artifacts already do); missing it fails with
`Metadata File for Corpus does not exist`.

### Pre-submit: non-empty `neg_doc`

Do not submit Automodel until a sample of `training.jsonl` has enough negatives.
Download the file (or the first lines) and count `len(neg_doc)`:

```bash
nemo files download <artifacts> --workspace default \
  --remote-path training.jsonl -o /tmp/training.jsonl
python3 - <<'PY'
import json
from collections import Counter
lens, n = Counter(), 0
with open("/tmp/training.jsonl") as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        n += 1
        negs = obj.get("neg_doc") or []
        lens[len(negs) if isinstance(negs, list) else "bad"] += 1
        if i >= 199:
            break
need = 4  # default train_n_passages=5
print({"sampled": n, "neg_lens": dict(lens)})
if lens.get("bad", 0):
    raise SystemExit("malformed neg_doc: expected a JSON list on every row")
if lens.get(0, 0):
    raise SystemExit("empty neg_doc: every row must have negatives before Automodel")
if any(isinstance(k, int) and k < need for k in lens):
    print("warn: some rows have fewer than", need, "negatives")
PY
```

If any sampled row has malformed or empty `neg_doc`, stop and fix the dataset;
for convert-only output, mine before training. A single empty list can be
selected by the collator and fail the run. Do not lower
`train_n_passages` to paper over an unmined convert-only fileset.

## Reuse an existing Stage 0 dump

Point `sdg_input` at a fileset or `hf://` URI. Live generate writes
`generation_result.json`, which is the default, so most runs set nothing extra.
A published dump under another name needs the filename on the ref or in
`generation_file`:

```bash
nemo data-designer retrieval-prepare --workspace default --spec '{
  "sdg_input": "hf://nvidia/Retrieval-Synthetic-NVDocs-v1@<revision>/nv_pp_dd_sdg.json",
  "enable_mining": false
}'

nemo data-designer retrieval-prepare --workspace default --spec '{
  "sdg_input": "default/retrieval-synthetic-nvdocs-v1",
  "generation_file": "nv_pp_dd_sdg.json",
  "enable_mining": false
}'
```

## Stage 1 output

One `artifacts` result per job. At the result root: `training.jsonl` and
`eval_beir/` (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`). Wrapped
`train.json`, corpus parquet, mining caches, and miner intermediates go under
`additional/`. Pass that artifacts fileset to Automodel as `dataset.training`
and to `retrieve-eval` as `dataset`.

```bash
nemo jobs get-status <prepare-job> -f json
nemo files list <artifacts-fileset> --workspace default
```

## Next Steps

- Choose `embed.md` for a bi-encoder or `rerank.md` for a cross-encoder.
- Pass the unchanged Stage 1 `artifacts` fileset to Automodel training and
  `retrieve-eval` so base and tuned models use the same frozen `eval_beir`.
