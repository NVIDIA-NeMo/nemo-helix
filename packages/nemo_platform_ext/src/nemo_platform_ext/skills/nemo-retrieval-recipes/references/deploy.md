<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Deploy embed / rerank NIMs (Platform)

`retrieve-eval` scores Inference Gateway embedding (and optional ranking)
endpoints, not filesets. Every model named in the eval spec must show a
non-empty `model_providers` list on `nemo models get`. Automodel outputs are
fileset-backed and start with `model_providers: []` until Deployment Manager
serves them.

Do not create deployments until the output model entity exists and reports a
non-null `fileset`. Unmerged LoRA cannot be served.

## Cluster pull secret

NIM images live on `nvcr.io`. Helm `imagePullSecrets` must include the NGC
dockerconfig (commonly `nvcrimagepullsecret`). That is chart install, not a
per-job flag. A pod with empty `imagePullSecrets` fails `ErrImagePull` /
`DENIED` on the registry.

## Config contract

Create `deployment-configs` with `engine=nim` and one GPU per NIM. Base+tuned
side-by-side therefore needs two GPUs.

| Role | `--model-spec` | Weights |
|---|---|---|
| Baked base | `'{}'` | Checkpoint baked into the NIM image |
| Tuned output | `{"model_namespace":"<workspace>","model_name":"<entity>"}` | Mount the Automodel fileset |

Always pass `--model-entity-id <workspace>/<name>` so `/v1/models` discovery
attaches a provider to the entity `retrieve-eval` names.

Set `override_config: {"nimLegacy": false}` for NIM images that retired
`NIM_MODEL_NAME` / `NIM_MODEL_PATH` (Retriever NIM 2.2.0 among them). Those
images fail config validation and crash-loop when a retired name is set; the
flag switches the compiler to `NIM_ENGINE_MODEL_NAME` / `NIM_ENGINE_MODEL_PATH`.
Leave it at the `true` default for 1.x NIMs. Setting both generations of names
does not work — the validator rejects the retired ones regardless.

On NIM 2.x, `NIM_ENGINE_MODEL_NAME` is the `/v1/models` id and must match the
entity FQDN (`workspace/name`). `NIM_ENGINE_MODEL_PATH` is the weights dir.
Pass a fileset-relative override through `additional_envs` (honored on Docker
and Kubernetes); it must stay under `/model-store`. Automodel default export
keeps ONNX at the fileset root, so Retriever NIM 2.2.0 needs
`NIM_ENGINE_MODEL_PATH=alternates/hf` unless training used
`training.retrieval.export.primary: hf`.

First image pull can take many minutes; `--wait --timeout 1800` is appropriate.

## Embed

Image: `nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.0`. Serving path is
`/v1/embeddings`. Pass `input_type` query vs document at request time.

```bash
EXECUTOR='{"gpu":1,"image_name":"nvcr.io/nim/nvidia/nemotron-3-embed-1b","image_tag":"2.2.0","override_config":{"nimLegacy":false}}'
TUNED_EXECUTOR='{"gpu":1,"image_name":"nvcr.io/nim/nvidia/nemotron-3-embed-1b","image_tag":"2.2.0","override_config":{"nimLegacy":false},"additional_envs":{"NIM_ENGINE_MODEL_PATH":"alternates/hf"}}'

nemo inference deployment-configs create embed-base-cfg \
  --workspace default --engine nim --model-spec '{}' \
  --executor-config "$EXECUTOR" \
  --model-entity-id default/nemotron-3-embed-1b --exist-ok

nemo inference deployment-configs create embed-tuned-cfg \
  --workspace default --engine nim \
  --model-spec '{"model_namespace":"default","model_name":"nemotron-3-embed-1b-tuned"}' \
  --executor-config "$TUNED_EXECUTOR" \
  --model-entity-id default/nemotron-3-embed-1b-tuned --exist-ok

nemo inference deployments create embed-base \
  --workspace default --config embed-base-cfg --exist-ok --wait --timeout 1800
nemo inference deployments create embed-tuned \
  --workspace default --config embed-tuned-cfg --exist-ok --wait --timeout 1800
```

## Rerank

Image: `nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0`. Serving path is
`/v1/ranking`. Keep the first-stage embedder fixed; deploy only the reranker
entities named in `target.reranker` / `baseline.reranker`.

```bash
EXECUTOR='{"gpu":1,"image_name":"nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2","image_tag":"1.10.0"}'

nemo inference deployment-configs create rerank-base-cfg \
  --workspace default --engine nim --model-spec '{}' \
  --executor-config "$EXECUTOR" \
  --model-entity-id default/llama-nemotron-rerank-1b-v2 --exist-ok

nemo inference deployment-configs create rerank-tuned-cfg \
  --workspace default --engine nim \
  --model-spec '{"model_namespace":"default","model_name":"llama-nemotron-rerank-1b-v2-tuned"}' \
  --executor-config "$EXECUTOR" \
  --model-entity-id default/llama-nemotron-rerank-1b-v2-tuned --exist-ok

nemo inference deployments create rerank-base \
  --workspace default --config rerank-base-cfg --exist-ok --wait --timeout 1800
nemo inference deployments create rerank-tuned \
  --workspace default --config rerank-tuned-cfg --exist-ok --wait --timeout 1800
```

## After READY

```bash
nemo inference deployments list-models embed-base --workspace default
nemo models get nemotron-3-embed-1b --workspace default
nemo models get nemotron-3-embed-1b-tuned --workspace default
```

Do not submit `retrieve-eval` until every named model has `model_providers`.
Then return to `embed.md` / `rerank.md` Stage 3.

Delete deployments when eval is done. Configs may stay (`--exist-ok` on the
next create). Domain runbooks may wrap these CLIs; keep names and GPU counts
in the wrapper, not in this contract.
