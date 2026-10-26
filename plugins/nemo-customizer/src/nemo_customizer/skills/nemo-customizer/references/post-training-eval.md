<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Post-training evaluation (train/eval format parity)

Use after a customization job reaches **`completed`** when the user wants to compare **base vs LoRA** on the validation split.

## Format contract

Training and evaluation must use the **same CHAT JSONL row shape**:

```json
{
  "messages": [
    {"role": "user", "content": "<user turn or multi-turn prompt>"},
    {"role": "assistant", "content": "<label to predict>"}
  ]
}
```

Multi-turn rows use the same rule: the **final** `messages[-1]` turn is the assistant label; all preceding turns are context.

| Do | Don't |
|----|-------|
| Pass rows with `messages` unchanged from the training fileset | Flatten to `prompt` / `expected` or `prompt` / `completion` for eval |
| Send **`messages[:-1]`** at inference (exclude only the final assistant label) | Pass full `messages` including the label turn, or use `{"messages": "{{ item.messages }}"}` unfiltered |
| Score against **`messages[-1].content`** (final assistant turn) | Score against a renamed `expected` field unless you also keep `messages` |

Single-turn rows (one user prompt + one assistant label) are the degenerate case: `messages[:-1]` is just the user turn.

Automodel and unsloth both train on this shape when `has_chat` is true (see `hf-conversion.md`, `dataset-formats.md`).

## Evaluator templates (required)

```python
CHAT_USER_PROMPT_TEMPLATE = {
    "messages": "{{ item.messages[:-1] }}",
}
CHAT_REFERENCE_TEMPLATE = "{{ item.messages[-1].content }}"
```

Import from `references/eval_helpers.py` — do not re-type these in one-off scripts.

## Inference defaults (thinking models, e.g. Nemotron 3.5 Lightning, Qwen3)

| Setting | Recommended | Avoid |
|---------|-------------|-------|
| `enable_thinking` | `false` via `chat_template_kwargs` for short-answer SFT | Thinking on without enough tokens — model never closes thinking tag so the strip hook fails |
| `max_tokens` | `64` (short assistant labels) | `16` with thinking on; `1024` thinking on without strip (verbose prose) |
| System prompt | Omit unless user asks — matches training | Extra system prompt changes decode path vs SFT |

For thinking-enabled eval, set `reasoning=ReasoningParams(end_token="``")` **and** ensure `max_tokens` is large enough for the model to emit the end token before generating the answer.

## Inference after customization (wrap-up)

Include this in the completed-job report. Agents must discover `<provider>` from `nemo inference providers list --workspace default -f json` and fill concrete URLs — do not leave placeholders.

### LoRA adapters: no new deployment

Applies when the job used **`finetuning_type: lora`** and **`output.save_method: lora`** (adapter output).

After a customization job reaches **`completed`**, the platform registers the adapter on the base **model entity**. On a deployment with **`lora_enabled: true`**, enabled adapters are **hot-reloaded automatically** (adapter sidecar → vLLM). **Do not** create a new inference deployment, update the deployment, re-create providers, or add the adapter to a `served_models` list before post-training eval — run eval as soon as the job completes.

| Prerequisite (one-time) | Per-adapter step after training |
|-------------------------|----------------------------------|
| A **READY** inference deployment for the **base** model entity with `lora_enabled: true` | Confirm adapter appears under `nemo models get <model-entity>` → `adapters` |
| Gateway reachable at the provider URL below | Target the adapter by name in the eval request (see table) |
| The deployment's maximum LoRA rank ≥ the adapter's `rank` (see below) | — |

#### LoRA rank limit

A deployment loads adapters up to a maximum rank and **refuses higher-rank adapters**. Defaults:

| Engine / image | Default max rank | Raise it on the deployment config |
|---|---|---|
| vLLM (`engine: vllm`, the default) | **16** | `executor_config.additional_args: ["--max-lora-rank", "<rank>"]` |
| Model-free NIM (`vllm-model-free-nim`, `sglang-model-free-nim`; the default NIM image) | **16** | `executor_config.additional_envs: {"NIM_PASSTHROUGH_ARGS": "--max-lora-rank <rank>"}` |
| Multi-LLM NIM (`llm-nim`, NIM 1.x) | **32** | `executor_config.additional_envs: {"NIM_MAX_LORA_RANK": "<rank>"}` |

vLLM accepts max ranks of 1, 8, 16, 32, 64, 128, 256, 320, 512. An inline `deployment_config` on the job cannot set `additional_args`, so for a vLLM adapter above rank 16 create a deployment config template with the flag and reference it by name. Before training at rank > 16, confirm the user will deploy with a raised limit.

#### Base already deployed without LoRA: ask the user

The one exception to "do not update the deployment". A LoRA job's `deployment_config` deploys the base when no active deployment of it has `lora_enabled: true`. If the base is already served by a deployment whose config has `lora_enabled: false`, the job creates a LoRA-enabled deployment of the base alongside it when it finishes, so the base runs twice. With inline parameters the new deployment is named `sft-deploy-<base>`; if the existing deployment already has that name (an earlier job created it), the job updates it to the LoRA-enabled config instead, which restarts it.

Check **before submitting** a LoRA job:

```bash
nemo inference deployment-configs list --workspace default --filter.model-entity-id default/<model-entity> -f json  # configs that serve the base
nemo inference deployments list --workspace default --filter.config <config-name> -f json                         # status and config_version per deployment
nemo inference deployment-configs versions get <config_version> --config <config-name> --workspace default -f json  # the version the deployment runs
```

A deployment runs the config version it was created with (`config_version`), which is older than the listed config when the config was updated afterwards; read `model_spec.lora_enabled` from that version. If a deployment is `CREATED`, `PENDING` or `READY`, the version it runs has `lora_enabled: false`, and no active deployment runs a version with `lora_enabled: true`, **stop and ask the user** which they want. Never change the deployment without a yes.

- **Enable LoRA on the existing deployment:** this restarts the base model, so anything using that deployment loses it until it is READY again. Read the config (`nemo inference deployment-configs get <config-name> --workspace default -f json`), set `model_spec.lora_enabled` to `true`, write back `engine`, `executor_config` and `model_spec` with `nemo inference deployment-configs update <config-name> --workspace default --input-file <file>`, then repoint the deployment at the new version: `nemo inference deployments update <deployment-name> --workspace default --config <config-name>`. Wait for READY, then submit; the adapter hot-loads when the job completes.
- **Deploy alongside it:** submit as is. Tell the user the base model will run twice, on the GPUs of both deployments, until they delete one; or, if the existing deployment is named `sft-deploy-<base>` and the job uses inline parameters, that the job restarts it with LoRA instead.

### Full SFT / merged checkpoints: deploy the output model

Applies when the job used **`finetuning_type: all_weights`** (full-weight SFT) or **`save_method: merged_16bit` / `merged_4bit`** (merged LoRA checkpoint). Output `type` is **`model`**, not `adapter`.

The fine-tuned weights live on a **new model entity** at `output.name` (`default/<output.name>`). **You must deploy that entity for inference** — create a new inference deployment or add it to a provider's `served_models` before chat or eval. Full checkpoints are **not** hot-reloaded onto the base model's LoRA deployment.

| Step | Action |
|------|--------|
| Confirm registration | `nemo models get <output.name> --workspace default` — entity exists with fine-tuned weights |
| Deploy for inference | Create or update an inference deployment / provider that serves `default/<output.name>` |
| Inference / eval route | **Model-entity** URL on `<output.name>` with `model: default/<output.name>` (not the base entity) |

**Pick the engine.** Deploy with **`engine: vllm`** by default; it is also the default for an inline `deployment_config` on the job. Use `engine: nim` only when the user asks for NIM or has a model-specific NIM image for the model. When using NIM for a Hugging Face checkpoint:

- Prefer the **model-free NIM** (`nvcr.io/nim/nvidia/vllm-model-free-nim`, the platform default NIM image, tag `latest`). It is the NIM 2.x successor to the multi-LLM NIM, not a rename, and the platform sets the configuration it reads. Supported models: [model-free NIM support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html#model-free-nim).
- Use the NIM 1.x **multi-LLM NIM** (`nvcr.io/nim/nvidia/llm-nim`) only when the user asks for it. Supported architectures: [multi-LLM NIM list](https://docs.nvidia.com/nim/large-language-models/1.15.0/supported-architectures.html).
- **The multi-LLM NIM cannot serve automodel `lora_merged` outputs.** The merge step saves `config.json` and the tokenizer with transformers 5; the multi-LLM NIM 1.15 ships transformers 4.56. Deploy merged LoRA outputs with vLLM or the model-free NIM, and warn the user if they ask for the multi-LLM NIM.

The per-model deployment configurations in the docs still list `llm-nim:1.15.5`; the merged-LoRA limitation applies to them.

Post-training eval for full models: compare against the base entity on its deployment, or eval the fine-tuned entity directly via `eval_helpers.py` is LoRA-oriented (`--adapter`); for full SFT, run generation eval against the **output** model entity's gateway URL.

### Request routing (base vs LoRA)

The model-entity proxy path **always** resolves to the base VirtualModel. Setting `"model": "default--<adapter-name>"` on `/model/<base-entity>/-/v1` does **not** select the adapter — gateway logs will show only the base path.

| Target | Gateway route | URL pattern | Request `model` field |
|--------|---------------|-------------|------------------------|
| Base entity | **Model entity** | `$NHX_BASE_URL/apis/inference-gateway/v2/workspaces/default/model/<model-entity>/-/v1` | `default/<model-entity>` |
| LoRA adapter | **Provider** | `$NHX_BASE_URL/apis/inference-gateway/v2/workspaces/default/provider/<provider>/-/v1` | `default--<adapter-name>` |

`eval_helpers.py` auto-discovers a READY provider that serves the base entity (or pass `--provider <name>`). LoRA adapter weights hot-reload on that deployment — no provider update per adapter. (Full SFT / merged outputs need a separate deployment — see above.)

Optional sanity checks:

- `nemo models get <model-entity> --workspace default` — adapter listed with `enabled: true`
- `nemo inference providers list --workspace default` — provider status **READY**
- LoRA eval/inference logs should show `path=…/provider/<provider>/-/v1/chat/completions`, **not** `…/model/<model-entity>/-/v1`
- JSON output includes `warnings` when routing looks wrong or adapter scores match base within ~1.5 pp

### Why earlier evals looked wrong

If base and LoRA scores were identical (~99% same outputs), the adapter was almost certainly called through the **model-entity** path. That path always resolves to the base VirtualModel — the `"model": "default--<adapter>"` field in the body is ignored. Fix: route LoRA through the **provider** URL with the same `model` field. `eval_helpers.build_platform_model_target()` and the CLI implement this split automatically.

### Classification / short-answer metric interpretation

For multiple-choice or short-label SFT, treat **`normalized_accuracy`** as the primary metric when labels need normalization (`normalize_mcqa_answer` strips `A. foo`, markdown, etc.).

| Observation | Likely meaning |
|-------------|----------------|
| Base & LoRA normalized scores match within ~1–2 pp | LoRA likely hit **model-entity** path (base only) — check `warnings` and gateway logs |
| Base raw exact match low, normalized much higher | Normal when the base model emits formatted prose but normalized labels match |
| LoRA normalized clearly above base | Correct provider routing and real adapter lift |
| Train loss dropped sharply but eval flat | Wrong eval routing, mismatched inference settings, or need more epochs — val loss ≠ accuracy |

### Epoch / adapter ablations

Resolve adapter names from completed job specs instead of guessing:

```python
import os
from eval_helpers import list_completed_job_adapters, compare_adapters, build_eval_payload

base_url = os.environ.get("NHX_BASE_URL") or "http://127.0.0.1:8080"

jobs = list_completed_job_adapters(
    base_url=base_url,
    workspace="default",
    model_entity="<model-entity>",
    dataset_fileset="<dataset-fileset>",
)
# jobs[0].epochs, jobs[0].adapter_name, jobs[0].backend — sorted newest first

summaries = compare_adapters(
    base_url=base_url,
    workspace="default",
    model_entity="<model-entity>",
    adapter_names=[jobs[0].adapter_name, jobs[1].adapter_name],
    rows=rows,
)
payload = build_eval_payload(..., summaries=summaries, adapter_names=[...])
# payload["lift_vs_base"], payload.get("warnings")
```

When comparing adapters from **different backends** (automodel vs unsloth) or batch configs, note confounds — epoch count alone may not explain the gap.

### Production chat requests (same rules as eval)

| Piece | LoRA adapter | Base model |
|-------|--------------|------------|
| HTTP base URL | `…/provider/<provider>/-/v1` | `…/model/<model-entity>/-/v1` |
| `"model"` | `default--<adapter-name>` | `default/<model-entity>` |
| `messages` | `messages[:-1]` from the training row (exclude final assistant label) | Same |
| Short-answer SFT (e.g. Nemotron 3.5 Lightning, Qwen3) | `"chat_template_kwargs": {"enable_thinking": false}` | Same |
| `max_tokens` / `temperature` | `64` / `0` typical for short labels | Same |

CLI shortcuts (substitute names from the job):

```bash
# LoRA
nemo inference gateway provider post v1/chat/completions <provider> --workspace default \
  --body '{"model":"default--<adapter>","messages":[{"role":"user","content":"…"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'

# Base
nemo inference gateway model post v1/chat/completions <model-entity> --workspace default \
  --body '{"model":"default/<model-entity>","messages":[{"role":"user","content":"…"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Metrics

| Task | Metrics |
|------|---------|
| MCQA / exact label | `ExactMatchMetric` + `normalize_mcqa_answer()` when models return `A. foo` or markdown |
| Similarity | `ROUGEMetric`, `BLEUMetric` with `CHAT_REFERENCE_TEMPLATE` |

Val loss from training is **not** accuracy — always run a generation eval for user-facing quality.

## Helper script

From this skill's directory; the script needs only the Python standard library:

```bash
# eval_helpers.py reads $NHX_BASE_URL when --base-url is omitted and self-defaults to
# http://127.0.0.1:8080 when it is unset. Do NOT blanket-export localhost — that would
# override a remote you already configured. For a remote platform, export its URL
# explicitly (export NHX_BASE_URL=https://…) or pass --base-url; for a local platform
# with no config, leave it unset and the script defaults to localhost.

# Base vs one adapter (--base-url optional when NHX_BASE_URL is set)
python3 references/eval_helpers.py \
  --model-entity <model-entity> \
  --adapter <adapter-name> \
  --provider <provider> \
  --dataset-fileset <dataset-fileset> \
  --split validation.jsonl \
  --output /tmp/fine-tune-eval.json

# Base vs multiple adapters (epoch ablation)
python3 references/eval_helpers.py \
  --model-entity <model-entity> \
  --adapter <adapter-a> \
  --adapter <adapter-b> \
  --dataset-fileset <dataset-fileset> \
  --split validation.jsonl \
  --output /tmp/fine-tune-eval-multi.json
```

Programmatic use:

```python
from eval_helpers import (
    load_chat_jsonl_from_platform,
    compare_adapters,
    compare_base_vs_adapter,
    build_eval_payload,
    list_completed_job_adapters,
    routing_sanity_warnings,
    CHAT_USER_PROMPT_TEMPLATE,
)
```

(Add `references/` to `sys.path` or run via `uv run python` from repo root.)

## Report to user

After compare, report for **base and each adapter**:

- **Normalized accuracy** (primary for MCQA)
- Raw exact match (strict string — often 0% on base for formatted answers)
- Lift vs base (`lift_vs_base` in JSON output)
- ROUGE / BLEU aggregates if requested
- Any `warnings` from routing sanity checks
- Inference settings (`enable_thinking`, `max_tokens`) and dataset fileset ref

Uses the **nemo-evals SDK** (`Evaluator`, metrics, `RunConfigOnlineModel`) under the hood — no separate evaluator skill doc required. For general BYOB/rubric eval outside customization, use the **nemo-evals** skill.
