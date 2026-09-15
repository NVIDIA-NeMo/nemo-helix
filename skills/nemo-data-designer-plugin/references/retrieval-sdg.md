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

Stage 1 (conversion only; mining is off unless you set `enable_mining`):

```bash
nemo data-designer retrieval-prepare --spec '{"sdg_input":"default/stage0-out"}'
```

Skip SDG entirely by pointing `sdg_input` at `hf://nvidia/Retrieval-Synthetic-NVDocs-v1@...` or a fileset that already contains `generation_result.json`.

Model roles resolve through Inference Gateway (`provider` + served model names). Do not set `NVIDIA_API_KEY` on the job.

Stage 1 mining (`enable_mining: true`) compiles to **`nmp-automodel-training`** as `python -m nmp.automodel.tasks.retrieval_mine` (torchrun of the Nemotron miner plus unroll/JSONL). All retrieval steps use one container-backed profile (`data_designer.job_executor_profile`, or `--profile`) so generation, conversion, model staging, and mining share job storage. Convert stays on `nmp-cpu-tasks`; mining uses the GPU image.

`model` is a platform model entity with an attached fileset. Before the GPU step,
`nmp-customizer-tasks` downloads that fileset into the shared job storage. The miner and
tokenizer load only from this staged directory with Hugging Face networking disabled.

The miner's recipe config is generated from the prepare spec, not shipped in the image. Common knobs (`model`, `hard_negatives_to_mine`, `query_prefix`, `dist_backend`, ...) plus the nested `mining` object are written to `mining_config.yaml` in the job artifacts and passed to the miner as `--config`:

```bash
nemo data-designer retrieval-prepare --spec '{"sdg_input":"default/stage0-out","enable_mining":true,"model":"default/nemotron-3-embed-1b","mining":{"corpus_chunk_size":10000,"hard_neg_margin_type":"abs"}}'
```

Chaining generate then prepare is a jobs-service multi-step job (`retrieval-run`), not Data Designer workflow chaining. Prefer `retrieval-run` when the user wants both stages.

Tiny corpora (one source file / `num_files: 1`) can place every query in the test split and leave train empty. Conversion must fail before mining when `train.json` has no records. Generate with enough documents (50+ recommended) or raise `train_ratio`.

## Previous / Next / artifacts

| Direction | Skill or job | Artifact |
|---|---|---|
| Previous | User corpus fileset or `hf://` URI | Raw docs |
| This stage | `retrieval-generate` → `retrieval-prepare` (or `retrieval-run`) | `generation_result.json`; `eval_beir/corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`; `training.jsonl` |
| Next | `nemo-retrieval-recipes`, or `nemo customization automodel submit` with `training.recipe: bi_encoder` (embed) or `cross_encoder` (rerank) | Dataset fileset with `training.jsonl` at the root; freeze `eval_beir` for `retrieve-eval` |

Do not regenerate `eval_beir` for base vs fine-tuned comparisons.

See [platform validation guidance](nemo-platform-plugin-additions.md) for fileset and platform validation guidance.
