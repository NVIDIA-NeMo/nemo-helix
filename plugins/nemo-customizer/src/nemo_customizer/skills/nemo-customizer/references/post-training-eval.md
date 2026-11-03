<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluation: base vs tuned on a frozen test split

An SFT or LoRA run ends with one answer: is the tuned model better than the base model on data neither saw in training? The flow is fixed:

1. Freeze a test split before uploading anything.
2. Serve the base model and score it.
3. Train, with `deployment_config` so the job serves its output.
4. Score the tuned model on the same split, with the same settings.
5. Report base vs tuned first (`reporting.md`).

The **nemo-evals** skill does the scoring; this file says what to hand it. GRPO measures this inside the job instead: set `val_at_start` and `val_at_end`, and the validation reward before and after is the comparison (`reporting.md` § **RL (GRPO) example**).

## 1. Freeze a test split

- Hold out about 10% of the rows, at most 500, chosen at random with a fixed seed, and remove them from `train.jsonl`. Use a split the dataset ships only when it has labels: many Hugging Face `test` splits do not (`tau/commonsense_qa` is one).
- Write them to `test.jsonl` as CHAT rows (**Format contract** below), even when the training rows use another shape.
- Upload `test.jsonl` to its **own** fileset, `<dataset-fileset>-test`, never to the training fileset. Kept apart, it cannot be trained on or used to pick a checkpoint.
- Keep the file: base and tuned are scored on exactly these rows.

## 2. Serve the base model and score it

Do this before submitting the job, so the baseline exists however the run ends.

- **LoRA:** deploy the base model with `lora_enabled: true`. The job's `deployment_config` then reuses that deployment and loads the adapter onto it.
- **Full weights:** deploy the base model, score it, then delete that deployment unless the user wants to keep it. The job deploys its own output.

Before scoring, wait until the model answers: the deployment is `READY` (`nemo wait inference deployment <name>`), then one request through the gateway succeeds. Hand off to **nemo-evals** (below).

### Base already deployed without LoRA: ask the user

Check this **before deploying the base**. A LoRA job's `deployment_config` deploys the base when no active deployment of it has `lora_enabled: true`. If the base is already served by a deployment whose config has `lora_enabled: false`, a new LoRA-enabled deployment runs the base a second time. With inline parameters the job names its deployment `sft-deploy-<base>`; if the existing deployment already has that name (an earlier job created it), the job updates it to the LoRA-enabled config instead, which restarts it.

```bash
nemo inference deployment-configs list --workspace default --filter.model-entity-id default/<model-entity> -f json  # configs that serve the base
nemo inference deployments list --workspace default --filter.config <config-name> -f json                         # status and config_version per deployment
nemo inference deployment-configs versions get <config_version> --config <config-name> --workspace default -f json  # the version the deployment runs
```

A deployment runs the config version it was created with (`config_version`), which is older than the listed config when the config was updated afterwards; read `model_spec.lora_enabled` from that version. If a deployment is `CREATED`, `PENDING` or `READY`, the version it runs has `lora_enabled: false`, and no active deployment runs a version with `lora_enabled: true`, **stop and ask the user** which they want. Never change the deployment without a yes.

- **Enable LoRA on the existing deployment:** this restarts the base model, so anything using that deployment loses it until it is READY again. Read the config (`nemo inference deployment-configs get <config-name> --workspace default -f json`), set `model_spec.lora_enabled` to `true`, write back `engine`, `executor_config` and `model_spec` with `nemo inference deployment-configs update <config-name> --workspace default --input-file <file>`, then repoint the deployment at the new version: `nemo inference deployments update <deployment-name> --workspace default --config <config-name>`. Wait for READY, then score the base on it; the job reuses it.
- **Deploy alongside it:** deploy a LoRA-enabled base for the baseline as above; the job reuses it. Tell the user the base model runs twice, on the GPUs of both deployments, until they delete one.

## 3. Score the tuned model

When the job completes, `deployment_config` has either created a deployment for the output (full weights) or the adapter loads onto the LoRA-enabled base deployment. Either can take a few minutes after the job reports `completed`, so retry a one-token request until it answers. Then hand off again with the tuned model's ID and everything else unchanged.

**An eval failure never fails the run.** The job trained and registered its output; report that, report the eval error separately, and do not re-train because scoring failed.

## Hand-off to nemo-evals

Give the **nemo-evals** skill these inputs, the same for base and tuned except the model ID:

| Input | Value |
|---|---|
| Rows | The frozen `test.jsonl` |
| Target | `Model(url="<gateway>/apis/inference-gateway/v2/workspaces/<ws>/openai/-/v1", name="<model ID>")`, where `<gateway>` is the platform URL the agent uses (`nemo inference get-url` prints the full route) |
| Prompt template | `{"messages": "{{ item.messages[:-1] }}"}`; a model without a chat template uses **Models without a chat template** below |
| Reference | `{{ item.messages[-1].content }}`. The assistant `content` is a string. |
| Metric | Short answers: `F1Metric`. Exact labels: `ExactMatchMetric`. Tool calls: `ToolCallingMetric`. Open-ended answers: an LLM judge, chosen with nemo-evals' metric selection |
| Inference | `temperature` 0; `max_tokens` 64 for short labels; for thinking models (Qwen3, Nemotron 3.5 Lightning) turn thinking off with `extra_body={"chat_template_kwargs": {"enable_thinking": false}}` |
| Execution | A local SDK run, which reaches the gateway at the agent's platform URL. A platform eval job also works, but its container must reach that URL: true for a local Docker platform, not for `localhost` on Kubernetes |

**Models without a chat template.** vLLM rejects chat requests for a model with no chat template, so score it through the completions endpoint, sending the text the model trained on. The SDK sends a completions request when the prompt template has `prompt` instead of `messages`. The rows stay in CHAT shape, and the reference is unchanged.

| Trained on | Prompt template |
|---|---|
| Automodel `prompt` / `completion` rows | `{"prompt": "{{ item.messages[0].content }} "}`: Automodel joins prompt and completion with a space |
| Unsloth `text` rows (`to_text` in `hf-conversion.md`) | `{"prompt": "{{ item.messages[0].content }}\n"}` |

Leave out `chat_template_kwargs`; it only applies to chat requests.

**Model IDs.** Every model is named by its entity ID on that one route:

| Model | `name` / `"model"` |
|---|---|
| Base | `<ws>/<base-entity>` |
| LoRA adapter | `<ws>/<base-entity>&adapters/<ws>/<output.name>` |
| Full-weight output | `<ws>/<output.name>` |

An adapter has no model entity of its own, so `ModelRef("<ws>/<adapter>")` fails, and a composite ID is not a valid `ModelRef`. Use the inline `Model` above. The gateway routes the adapter through its base model's deployment and applies that model's middleware; if the base model's VirtualModel routes requests to other models (Switchyard), say so in the report, because the scores then include those models.

To spot-check one model by hand:

```bash
curl -s "$(nemo inference get-url)/chat/completions" -H 'content-type: application/json' -d '{
  "model": "<ws>/<base-entity>&adapters/<ws>/<output.name>",
  "messages": [{"role": "user", "content": "<a prompt from test.jsonl>"}],
  "max_tokens": 64, "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

## Format contract

Training and evaluation use the **same CHAT JSONL row shape**:

```json
{
  "messages": [
    {"role": "user", "content": "<user turn or multi-turn prompt>"},
    {"role": "assistant", "content": "<label to predict>"}
  ]
}
```

The **final** turn is the label; every earlier turn is context. Send `messages[:-1]` and score against `messages[-1].content`. Do not flatten the rows to `prompt` / `expected`, and do not send the label turn; a model without a chat template gets its prompt from the completions template in **Hand-off to nemo-evals**.

An image row keeps the part list on the user turn. Send `messages[:-1]` with those image parts intact. The assistant label is a string, so the reference stays `{{ item.messages[-1].content }}`. A relative `image_url` is the fileset path the trainer reads. The inference gateway does not have that fileset, so a prompt sent there needs a `data:` URL or a URL the gateway can fetch.

## Reading the result

| Observation | Likely meaning |
|---|---|
| Tuned and base scores match within 1–2 points | The tuned request named the base model; check the model ID. Or the run learned little: check training loss and the epoch count |
| Base exact match near 0, F1 much higher | Normal when the base model answers in prose around the label |
| Training loss fell sharply but the score did not move | Mismatched inference settings (template, thinking, `max_tokens`) or overfitting. Validation loss is not accuracy |

When comparing several adapters, for example different epoch counts, score each one on the same frozen split. Results from different backends or batch settings have confounds; say so.

## Serving the tuned model

**LoRA rank limit.** A deployment loads adapters up to a maximum rank and **refuses higher-rank adapters**:

| Engine / image | Default max rank | Raise it on the deployment config |
|---|---|---|
| vLLM (`engine: vllm`, the default) | **16** | `executor_config.additional_args: ["--max-lora-rank", "<rank>"]` |
| Model-free NIM, vLLM (`vllm-model-free-nim`; the default NIM image) | **16** | `executor_config.additional_envs: {"NIM_PASSTHROUGH_ARGS": "--max-lora-rank <rank>"}` |
| Model-free NIM, SGLang (`sglang-model-free-nim`) | **16** if the server starts with no adapters; otherwise the highest rank among adapters present at start | `executor_config.additional_envs: {"NIM_PASSTHROUGH_ARGS": "--max-lora-rank <rank>"}` |
| Multi-LLM NIM (`llm-nim`, NIM 1.x) | **32** | `executor_config.additional_envs: {"NIM_MAX_LORA_RANK": "<rank>"}` |

vLLM accepts max ranks of 1, 8, 16, 32, 64, 128, 256, 320, 512. An inline `deployment_config` on the job cannot set `additional_args`, so for a vLLM adapter above rank 16 create a deployment config template with the flag and reference it by name. Before training at rank > 16, confirm the user will deploy with a raised limit.

**Pick the engine.** Deploy with **`engine: vllm`** by default; it is also the default for an inline `deployment_config` on the job. Use `engine: nim` only when the user asks for NIM or has a model-specific NIM image for the model. When using NIM for a Hugging Face checkpoint:

- Prefer the **model-free NIM** (`nvcr.io/nim/nvidia/vllm-model-free-nim`, the platform default NIM image, tag `latest`). It is the NIM 2.x successor to the multi-LLM NIM, not a rename, and the platform sets the configuration it reads. Supported models: [model-free NIM support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html#model-free-nim).
- Use the NIM 1.x **multi-LLM NIM** (`nvcr.io/nim/nvidia/llm-nim`) only when the user asks for it. Supported architectures: [multi-LLM NIM list](https://docs.nvidia.com/nim/large-language-models/1.15.0/supported-architectures.html).
- **The multi-LLM NIM cannot serve automodel `lora_merged` outputs.** The merge step saves `config.json` and the tokenizer with transformers 5; the multi-LLM NIM 1.15 ships transformers 4.56. Deploy merged LoRA outputs with vLLM or the model-free NIM, and warn the user if they ask for the multi-LLM NIM.

The per-model deployment configurations in the docs still list `llm-nim:1.15.5`; the merged-LoRA limitation applies to them.
