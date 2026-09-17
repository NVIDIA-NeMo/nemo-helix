---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-retrieval-recipes
description: >-
  End-to-end NeMo Platform recipe for domain embedding and reranking fine-tuning:
  Data Designer retrieval-generate/prepare, Automodel bi_encoder/cross_encoder,
  Evaluator retrieve-eval on frozen eval_beir (NIM providers via Deployment
  Manager). Use when the user has a document corpus, wants better retrieval
  nDCG/recall, or asks to fine-tune Nemotron embed or rerank models on Platform.
triggers:
  - fine-tune embedding
  - fine-tune rerank
  - embedding recipe
  - retrieval recipe
  - domain retrieval
  - nDCG embedding
  - retrieve-eval
  - Nemotron embed
  - Nemotron rerank
  - bi_encoder
  - cross_encoder
  - retrieval-sdg
  - Retriever NIM
  - Ranking NIM
not-for:
  - nemo-customizer (chat/SFT/LoRA/DPO/GRPO on language models)
  - nemo-data-designer-plugin Autopilot create (tabular synthetic columns)
  - public BEIR/MTEB leaderboard eval unrelated to a domain corpus
  - generic vector-database selection
preconditions:
  - nemo_cli_available
  - platform_running
  - workspace_exists
compatibility: NeMo Platform with data-designer [retrieval-sdg], customizer automodel, and evaluator retrieve-eval plugins.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Grep]
version: "0.1.0"
---

# NeMo Platform retrieval recipes

Conduct the Nemotron embed/rerank pipeline **on NeMo Platform plugin CLIs**.
Do not use `nemo data-designer create`.

Plugin skills (`nemo-data-designer-plugin`, `nemo-customizer`, `nemo-evaluator-plugin`)
are stage manuals. This skill owns family routing, artifact IDs, frozen eval, and
the 15% nDCG@10 / Recall@10 README bar.

## Resolve CLI

Use `nemo`, or `uv run nemo` from the nemo-platform root. If neither works, route
to `setup`. The CLI defaults to `http://localhost:8080`; `NMP_BASE_URL` overrides
it for a remote platform. Confirm `/health/ready` on the resolved base URL before
submitting jobs, and recheck it before rereading a command that returned empty or
non-JSON output — that is an unreachable platform, not a missing entity.

## Choose a family

Read only the matching recipe after this routing step:

- Low Recall@k, missing relevant documents, first-stage retrieval → `references/embed.md`
- Recall@100 OK but nDCG@10 poor, relevant docs buried in top-k → `references/rerank.md`
- User asks both or a two-stage stack → embed first (candidate coverage), then rerank.

Use one workspace. Stage 1 publishes one `artifacts` fileset holding `training.jsonl`
and `eval_beir/`; Automodel and `retrieve-eval` both read that one fileset directly.
Keep `query:` / `passage:` prefixes for embed and
`question:{query} \n \n passage:{passage}` for rerank.

## Safe workflow

1. Confirm corpus governance: Stage 0 sends chunk text to Inference Gateway chat
   models. Do not ask users to paste API keys. Use `provider` / `chat_provider` /
   `embed_provider`, not `NVIDIA_API_KEY` on the job.
2. Dry-run schemas first: `nemo data-designer retrieval-generate --help`,
   `nemo customization automodel explain`, `nemo evaluator retrieve-eval explain`.
3. Live SDG is the default for a user corpus; drive it through Stage 0 generation
   knobs (see `references/sdg.md`). It needs **50+ documents**, since one file can
   dump every query into the test split and leave train empty (mining then crashes).
   Reach for a published Stage 0 dump only when the user has no corpus or wants a
   fast first eval.
4. Run Stage 0+1 once (`retrieval-run`, or generate then prepare). **Freeze** the
   resulting `eval_beir`; never regenerate it for base vs tuned comparisons.
5. Confirm the training model entity has a non-null `fileset`. Auto-discovered
   Inference Gateway model entities are endpoints, not trainable checkpoints.
6. Before Automodel, confirm `training.jsonl` has non-empty `neg_doc` lists
   (`references/sdg.md`). Convert-only Stage 1 leaves `neg_doc: []`; the collator
   then crashes sampling `train_n_passages - 1` negatives (default 4). Mine first
   (`enable_mining: true`).
7. Fine-tune with explicit `training.recipe: bi_encoder` or `cross_encoder` and
   `finetuning_type` `all_weights` or `lora_merged` only.
8. `retrieve-eval` needs IGW providers on every named model (`k: [1,5,10,100]`,
   `baseline` + tuned). Fileset-backed Automodel entities have empty
   `model_providers` until Deployment Manager serves them — follow
   `references/deploy.md` before submit.
9. Serving after eval is the same deploy contract. One GPU per NIM; Helm
   `imagePullSecrets` must include the NGC dockerconfig or the pod cannot pull
   `nvcr.io`.
10. Poll jobs at 60–300s. Parse JSON from **stdout only** (never `2>&1` into `json.load`).

Report absolute and relative nDCG@10, Recall@10, and Recall@100 on the same frozen
eval set. Prefer at least 100 queries; warn below 50. Treat 15% relative nDCG@10
and Recall@10 uplift as an indicative embed target, not a hard small-corpus gate.

References: `references/sdg.md` (Stage 0+1 corpus and generation control),
`references/embed.md`, `references/rerank.md`, `references/deploy.md`
(Retriever / Ranking NIM via Deployment Manager).

For stage-specific failures, hand off to `nemo-data-designer-plugin`,
`nemo-customizer`, or `nemo-evaluator-plugin`. NIM serving for this recipe
stays in `references/deploy.md`, not the `inference` skill.
