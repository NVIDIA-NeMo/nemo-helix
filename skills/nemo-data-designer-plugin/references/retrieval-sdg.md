# Retrieval SDG

Use dedicated Data Designer jobs to replicate Nemotron embed/rerank Stage 0 (`sdg`) and Stage 1 (`prep`). Do not use `nemo data-designer create` for this pipeline.

## Prerequisites

- A corpus fileset in the job workspace, or an `hf://` dataset URI.
- An Inference Gateway provider for chat and, when different, embeddings.
- For mining, a platform model entity with an attached encoder/tokenizer fileset.

Stage 0:

```bash
nemo data-designer retrieval-generate --spec '{"corpus":"default/my-docs","provider":"default/nvidia-build","artifact_extraction_model":"nvidia/nemotron-3-nano-30b-a3b","qa_generation_model":"nvidia/nemotron-3-nano-30b-a3b","quality_judge_model":"nvidia/nemotron-3-nano-30b-a3b","embed_model":"nvidia/nemotron-3-embed-1b"}'
```

Stage 1 (mine immediately; convert-only `enable_mining: false` is inspect-only):

```bash
nemo data-designer retrieval-prepare --spec '{"sdg_input":"default/stage0-out","enable_mining":true,"model":"default/nemotron-3-embed-1b"}'
```

Skip SDG entirely by pointing `sdg_input` at a Stage 0 fileset or `hf://` URI. Live generate writes `generation_result.json` (the default `generation_file`). For Hub dumps, name the file on the ref or set `generation_file`:

```bash
nemo data-designer retrieval-prepare --spec '{"sdg_input":"hf://nvidia/Retrieval-Synthetic-NVDocs-v1@<rev>/nv_pp_dd_sdg.json","enable_mining":true,"model":"default/nemotron-3-embed-1b"}'
# or
nemo data-designer retrieval-prepare --spec '{"sdg_input":"default/retrieval-synthetic-nvdocs-v1","generation_file":"nv_pp_dd_sdg.json","enable_mining":true,"model":"default/nemotron-3-embed-1b"}'
```

Model roles resolve through Inference Gateway (`provider` + served model names). Do not set `NVIDIA_API_KEY` on the job.

The Nemotron embed recipe uses [`nvidia/nemotron-3-ultra-550b-a55b`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) for all three chat roles. Examples here use `nvidia/nemotron-3-nano-30b-a3b` only so they can run on a typical already-deployed IGW model. Outside of examples, always use a larger model — for example [`nvidia/nemotron-3-ultra-550b-a55b`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) when the provider serves it. Nano in an example is not a recommendation for a real run.

Stage 0 quality controls are fields on the generate spec:
`file_extensions` (default `.txt`, `.md`, `.text`, and no extension),
`sentences_per_chunk` (5), `max_artifacts_per_type` (2), `num_pairs` (7),
`multi_doc` (false), and `max_parallel_requests_for_gen` (provider default).
When changing `num_pairs`, make both `query_counts` and `reasoning_counts` sum
to it. `multi_doc` also exposes `bundle_size`, `bundle_strategy`, and
`max_docs_per_bundle`. Stage 1 `quality_threshold` defaults to 7.0 and filters
generated pairs by judge score.

Stage 1 mining (`enable_mining: true`) is a GPU step. Pass `--profile` (or set
`data_designer.job_executor_profile`) so generate, convert, and mine share job
storage. `model` must be a platform entity with an attached encoder fileset.

Mining knobs (`hard_negatives_to_mine`, `query_prefix`, nested `mining`, …) go on
the prepare spec:

```bash
nemo data-designer retrieval-prepare --spec '{"sdg_input":"default/stage0-out","enable_mining":true,"model":"default/nemotron-3-embed-1b","mining":{"corpus_chunk_size":10000,"hard_neg_margin_type":"abs"}}'
```

Prefer `retrieval-run` when the user wants generate then prepare. Do not use
Data Designer `create` workflow chaining for this path.

Tiny corpora (one source file / `num_files: 1`) can place every query in the test split and leave train empty. Conversion must fail before mining when `train.json` has no records. Generate with enough documents (50+ recommended) or raise `train_ratio`.

Convert-only prepare (`enable_mining: false`) writes `training.jsonl` with empty `neg_doc` lists. Automodel `bi_encoder` / `cross_encoder` still samples `train_n_passages - 1` negatives (default 4) and fails with `neg_doc must contain at least 1 document to sample N negatives`. Before handing off to Automodel, check a representative sample of `training.jsonl` and require `neg_doc` to be a non-empty list on every checked row. A single empty list can be selected by the collator and fail the run. Re-run prepare with `enable_mining: true` (or `train_input_file` on the existing convert-only fileset so frozen `eval_beir` is not regenerated) if any row is empty.

## Stage 1 output

`retrieval-prepare` saves one `artifacts` job result holding `training.jsonl`,
`eval_beir/` (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`), and the wrapped
`train.json` that mining consumes. Copy `training.jsonl` and `eval_beir/` to a
fileset root (no path fragment) before Automodel `dataset.training` and
`retrieve-eval` `dataset`. Automodel's dataset discovery selects `training.jsonl`
and ignores the non-JSONL siblings; Stage 1 unique-keys qrels. Drop duplicate
`(query-id, corpus-id)` rows from `eval_beir/qrels/test.tsv` if you are uploading
an older dump — the BEIR loader rejects them. Splitting the artifacts into
separate training and eval filesets is optional.

## Previous / Next / artifacts

| Direction | Skill or job | Artifact |
|---|---|---|
| Previous | User corpus fileset or `hf://` URI | Raw docs |
| This stage | `retrieval-generate` → `retrieval-prepare` (or `retrieval-run`) | `generation_result.json`; `eval_beir/corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`; `training.jsonl` |
| Next | `nemo-retrieval-recipes`, or `nemo customization automodel submit` with `training.recipe: bi_encoder` (embed) or `cross_encoder` (rerank) | Stage 1 `artifacts` fileset as `dataset.training` and as the `retrieve-eval` dataset |

Do not regenerate `eval_beir` for base vs fine-tuned comparisons.

Use the emitted `training.jsonl` in an embedding or reranking customization job.
See [platform execution guidance](platform-execution.md) for fileset and platform validation guidance.
