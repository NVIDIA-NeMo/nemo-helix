<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Report to user

After polling reaches a **terminal** status (`completed`, `error`, or `cancelled`), report using this template for **all** backends (automodel, unsloth, rl). Fill fields from the job JSON and `nemo jobs get-status`.

## Result template

```markdown
## Fine-tune result

- **Job:** <automodel-|unsloth-|rl-><id>
- **Backend:** <automodel|unsloth|rl>
- **Model entity:** default/<model-entity>
- **Dataset fileset:** default/<dataset-fileset>
- **Output fileset:** <output.name from job JSON>   <!-- adapter for automodel/unsloth LoRA; full-weight model for rl (DPO) and full SFT -->
- **Status:** <completed|error|cancelled>
- **Final train loss:** <last value in metrics.train_loss, or "n/a">
- **Final validation loss:** <last value in metrics.val_loss, or "n/a (no validation run)">
- **Notes:** <see below>
```

For **rl (DPO)** the output is a **full-weight model entity** (no adapter): label the line **Output model entity**, **skip Using the adapter**, and use **Using the fine-tuned model** (below) — confirm with `nemo models get <output.name> --workspace default`. The DPO loss series lands in `status_details.metrics` like the other backends.

## Field guidance

| Field | Source |
|-------|--------|
| **Job** | Job id from submit or poll (`automodel-…` / `unsloth-…` / `rl-…`) |
| **Backend** | Plugin used for submit |
| **Model entity** | `model` in job JSON (automodel & rl: string ref; unsloth: `model.name`) |
| **Dataset fileset** | automodel: `dataset.training`; unsloth: `dataset.path`; rl: `dataset` (single preference-fileset string) |
| **Output adapter fileset** | `output.name` from job JSON. Label **Output adapter fileset (planned):** when status is `error` or `cancelled` and no output was registered |
| **Status** | Top-level `status` from `nemo jobs get-status` — not step-level status |
| **Final train loss** | Last entry in `status_details.metrics.train_loss` (or nested under a step's `status_details.metrics`). Use the **last** `value` in the list — not `status_details.train_loss` alone (that is the most recent logged step, which may differ from epoch-average loss on some backends). Round to 3 decimal places. |
| **Final validation loss** | Last entry in `status_details.metrics.val_loss`. If the list is empty, report `n/a (no validation run)` and note whether validation data was configured. Automodel validates once per epoch by default. Unsloth validates once per epoch when `dataset.validation_path` is set and `schedule.eval_steps` is omitted (platform default: `max(1, effective_steps - 1)`). |
| **Notes** | See **Notes by status** below |

**Metrics extraction** — after polling, always run `nemo jobs get-status <job-id>` and read `status_details.metrics` (every backend accumulates time series there). Include the final losses in the report even when status is `error` if training completed before the failure (e.g. entity registration failed after upload).

**GRPO does not report on loss** — `val_loss` is always empty and `train_loss` is a policy-gradient surrogate that oscillates near zero regardless of run quality. Use the GRPO section below instead of the two loss rows above.

## Notes by status

| Status | Notes |
|--------|-------|
| `completed` | Brief success summary. LoRA (`save_method: lora`): adapter registered on base model entity. Full SFT / merged checkpoint: new model entity at `output.name`. When `metrics.train_loss` has ≥2 entries, add a loss-drop sentence: *Loss dropped from \<first value, 1 dp\> at step 1 to \<last value, 3 dp\> at step \<N\>; validation loss was \<val or n/a\>.* Append **Using the adapter** (LoRA) or **Using the fine-tuned model** (full SFT / merged) with discovered provider name and concrete gateway URLs (see below). |
| `error` | Quote `error_details.message` or the failing step; note setup that succeeded before the failure (auth, dataset upload, submit). |
| `cancelled` | Cancellation reason if available. |

## Training configuration

Append a `### Training configuration` table after the header block (before **Using the output** when `completed`). Fill rows from the submitted job JSON; omit rows whose fields were not set. Use backend-specific labels:

| Setting | automodel source | unsloth source |
|---------|------------------|----------------|
| Training type | `training.training_type` | `training.training_type` |
| Finetuning type | `training.finetuning_type` | `training.finetuning_type` |
| LoRA rank / alpha | `training.lora.rank` / `training.lora.alpha` | same |
| Quantization | omit (full-precision / bf16 base weights) | `model.load_in_4bit` → `4-bit (load_in_4bit: true)` or omit when false |
| Max sequence length | `training.max_seq_length` | `model.max_seq_length` |
| Epochs | `schedule.epochs` | `schedule.epochs` |
| Batch | `micro_batch_size` / `global_batch_size` | `batch.per_device_train_batch_size` / `batch.gradient_accumulation_steps` |
| Effective batch size | `global_batch_size` | `per_device_train_batch_size × gradient_accumulation_steps` |
| Learning rate | `optimizer.learning_rate` | same |
| Optimizer | `optimizer` fields used (e.g. `weight_decay`, `warmup_steps`) | `optimizer.optim` (e.g. `adamw_8bit`) |
| Precision | `bf16` (default) | `hardware.precision` |
| GPU | `parallelism.num_gpus_per_node` (and `tensor_parallel_size` when >1) | `hardware.gpus` |
| Output save method | `output.type` (e.g. `adapter`) | `output.save_method` (e.g. `lora`) |

The three examples below show the filled-in table per backend.

## Automodel example

```markdown
### Training configuration

| Setting | Value |
|---------|-------|
| Training type | SFT |
| Finetuning type | LoRA |
| LoRA rank / alpha | 16 / 32 |
| Max sequence length | 2048 |
| Epochs | 1 |
| Micro batch size | 16 |
| Global batch size | 64 |
| Effective batch size | 64 |
| Learning rate | 1e-4 |
| Optimizer | weight_decay 0.01, warmup_steps 0 |
| Precision | bf16 |
| GPU | 1 (TP=1) |
| Output save method | adapter |
```

## Unsloth example

```markdown
### Training configuration

| Setting | Value |
|---------|-------|
| Training type | SFT |
| Finetuning type | LoRA |
| LoRA rank / alpha | 16 / 32 |
| Quantization | 4-bit (`load_in_4bit: true`) |
| Max sequence length | 2048 |
| Epochs | 1 |
| Per-device batch size | 8 |
| Gradient accumulation steps | 16 |
| Effective batch size | 128 |
| Learning rate | 1e-4 |
| Optimizer | adamw_8bit |
| Precision | bf16 |
| GPU | 0 |
| Output save method | lora |
```

## RL (DPO) example

Map rows from the `training` (DPOTraining) block; there is no LoRA/save-method. Add DPO-specific rows (`ref_policy_kl_penalty` = β, `sft_loss_weight`):

```markdown
### Training configuration

| Setting | Value |
|---------|-------|
| Training type | DPO (full-weight) |
| Reference KL penalty (β) | 0.05 |
| SFT loss weight | 0.0 |
| Max sequence length | 1024 |
| Epochs | 1 |
| Micro batch size | 1 |
| Global batch size | 32 |
| Learning rate | 5e-6 |
| Optimizer | AdamW + cosine annealing |
| Precision | bf16 |
| GPU | 1 node × 1 GPU |
| Output | full-weight model entity |
```

## RL (GRPO) example

GRPO has no labelled completions and no meaningful loss, so the loss rows in **Field guidance** do not apply. Report reward instead.

| Field | Source | Notes |
|-------|--------|-------|
| **Mean training reward** | last value of `metrics.train_reward` | The headline. Include the delta from the first point. |
| **Validation reward** | last value of `metrics.val_accuracy` | NeMo-RL names the validation pass's **mean reward** `accuracy`; it is not an accuracy in the classifier sense. There is no `val_reward`. |
| **Steps** | `status_details.step` / `max_steps` | |
| **Rollouts generated** | `status_details.step` × `status_details.rollouts_per_step` | `rollouts_per_step` is reported once at training start. |
| **Truncation rate** | last value of `metrics.train_truncation_rate` | Rising truncation is a common reason reward stops improving. |
| **Step time** | last value of `metrics.train_timing/total_step_time` | Seconds of wall clock per training step. The per-phase breakdown (`train_timing/generation`, `train_timing/policy_training`, …) reports as a current value, not a series — read it when a step suddenly slows. |
| **Algorithm** | `status_details.training_type` | `grpo` or `dpo`. `backend` is `nemo_rl` for both, so this is the only field that distinguishes them. |

### Training health

Report these only when reward is flat or falling — they explain *why*, and are noise otherwise.

| Series | Healthy | What a bad value means |
|--------|---------|------------------------|
| `train_baseline_reward/pct_mixed` | well above 0 | The share of prompt groups whose rollouts disagreed. GRPO scores each rollout against the others in its group, so a group where every rollout got the same reward produces no gradient. Falling toward 0 means the run has stopped learning, and it shows here before the reward curve flattens. |
| `train_approx_entropy` | stable | Falling steadily is entropy collapse: generations degenerate while reward can still look acceptable. |
| `train_token_mult_prob_error` | near 1 | Rollout and training logprobs disagreeing. |
| `train_sampling_importance_ratio` | near 1 | Same drift, measured as a ratio. |
| `train_gen_kl_error`, `train_policy_kl_error`, `train_js_divergence_error` | small | Distance from the reference policy. |
| `train_truncation_rate` | low | Responses hitting the generation cap. Consider `overlong_filtering` or a larger `max_seq_length`. |

### Reward spread — read the caveat before charting it

`train_total_reward/stddev`, `/p25` and `/p75` describe the spread of rewards within a step. **Both are close to useless when the environment returns a binary 0/1 reward**, which most verifier environments do:

- `stddev` on a Bernoulli variable is a deterministic function of the mean — √(p(1−p)). It carries no information beyond `train_reward`.
- `p25`/`p75` collapse to a three-state step function: `[0, 0]` while mean reward ≤ 0.25, `[0, 1]` between 0.25 and 0.75, `[1, 1]` above. A p25 pinned at 0 for a whole run is correct, not a bug.

For a binary-reward environment, report `baseline_reward/pct_0` / `pct_1` / `pct_mixed` as the spread instead — that is group composition, it moves meaningfully, and it is what the algorithm actually learns from. The dispersion series are worth charting when rewards are continuous or built from several components.

## Using the output (`completed` only)

After **Training configuration**, branch on output type:

| Output | When | Report section |
|--------|------|----------------|
| LoRA adapter | `save_method: lora` (default) | **Using the adapter** — below |
| Full model | `finetuning_type: all_weights`, `save_method: merged_16bit` / `merged_4bit`, or **rl (DPO)** (always full-weight) | **Using the fine-tuned model** — below |

### Using the adapter (LoRA / `save_method: lora`)

**Automodel / unsloth LoRA only** — DPO (rl) never produces an adapter; for rl output use **Using the fine-tuned model** below. Run these discovery commands (parse stdout only; do not pipe `2>&1` into JSON parsers):

1. `nemo models get <model-entity> --workspace default` — confirm `<output.name>` appears under `adapters` with `enabled: true`.
2. `nemo inference providers list --workspace default -f json` — pick a **READY** provider whose `served_models` includes `default/<model-entity>` (base entity). Record its `name` as `<provider>` (often matches the deployment name).

On a deployment with `lora_enabled: true`, the adapter is **hot-reloaded automatically** — no new deployment, deployment update, or provider reconfiguration before inference or post-training eval. Append this section with **concrete URLs and provider name** from discovery:

```markdown
### Using the adapter

The adapter `<output.name>` is registered on `default/<model-entity>`. Weights are hot-reloaded on LoRA-enabled deployments serving the **base** entity — no new deployment or provider update after training.

#### Request routing (base vs LoRA)

| Target | Gateway path | OpenAI base URL | Request `"model"` field |
|--------|--------------|-----------------|-------------------------|
| **Base** weights | model-entity | `$NHX_BASE_URL/apis/inference-gateway/v2/workspaces/default/model/<model-entity>/-/v1` | `default/<model-entity>` |
| **LoRA adapter** | **provider** | `$NHX_BASE_URL/apis/inference-gateway/v2/workspaces/default/provider/<provider>/-/v1` | `default--<output.name>` |

**Common mistake:** posting to the model-entity URL with `"model": "default--<output.name>"` still runs the **base** model. Base-vs-adapter eval will look identical until LoRA requests use the **provider** URL above. See `references/post-training-eval.md` § **Request routing (base vs LoRA)**.

#### Chat inference (CHAT-trained models)

Match training context at inference — send **`messages[:-1]`** (all turns except the final assistant label). Single-turn rows are just the user message; multi-turn rows keep prior user/assistant history.

| Setting | Value | Why |
|---------|-------|-----|
| `messages` | All turns except the final assistant label from the JSONL row | Same decode path as SFT |
| `max_tokens` | `64` for short assistant labels | Training targets are brief (e.g. MCQA choice text) |
| `temperature` | `0` | Reproducible eval / regression checks |
| `chat_template_kwargs.enable_thinking` | `false` for Qwen3 short-answer SFT | Thinking mode needs extra tokens and changes output shape vs training |

#### Example — LoRA adapter via provider

\`\`\`bash
export NHX_BASE_URL=<platform-url>   # omit when using default localhost
nemo inference gateway provider post v1/chat/completions <provider> --workspace default \\
  --body '{
    "model": "default--<output.name>",
    "messages": [<all turns except final assistant label from the eval row>],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
\`\`\`

#### Example — base model via model-entity (comparison)

\`\`\`bash
export NHX_BASE_URL=<platform-url>
nemo inference gateway model post v1/chat/completions <model-entity> --workspace default \\
  --body '{
    "model": "default/<model-entity>",
    "messages": [<same prompt turns as LoRA example — exclude final assistant label>],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
\`\`\`

#### Post-training eval (optional)

Validation loss from training is **not** accuracy. To compare base vs adapter on the validation split with correct routing:

\`\`\`bash
cd /path/to/nemo-helix
uv run python plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/references/eval_helpers.py \\
  --model-entity <model-entity> \\
  --adapter <output.name> \\
  --provider <provider> \\
  --dataset-fileset <dataset-fileset> \\
  --split validation.jsonl
\`\`\`

Uses CHAT `messages` rows unchanged from the training fileset (`messages[:-1]` at inference). Repeat `--adapter` for multi-adapter compare. `--provider` is optional when a READY provider is auto-discovered. Set `NHX_BASE_URL` (or pass `--base-url`) when the platform is not localhost. LoRA only — full SFT / merged outputs need a deployed model entity (see **Using the fine-tuned model**).
```

### Using the fine-tuned model (full SFT / merged checkpoint / DPO)

When `finetuning_type: all_weights`, `save_method` is `merged_16bit` / `merged_4bit`, or the backend is **rl (DPO)**, the job registers a **model** entity at `output.name` with full fine-tuned weights. **Deploy that entity before inference or eval** — full checkpoints are not hot-reloaded onto the base model's LoRA deployment.

1. `nemo models get <output.name> --workspace default` — confirm the fine-tuned model entity exists.
2. Create or update an inference deployment / provider that serves `default/<output.name>` (same workflow as deploying any model entity).
3. Append this section with the **READY** provider or deployment name and concrete gateway URL.

```markdown
### Using the fine-tuned model

Fine-tuned weights are on model entity `default/<output.name>`. Unlike LoRA adapters, full checkpoints **require a new inference deployment** (or provider update) before chat or eval.

| Target | Gateway path | OpenAI base URL | Request `"model"` field |
|--------|--------------|-----------------|-------------------------|
| Fine-tuned model | model-entity | `$NHX_BASE_URL/apis/inference-gateway/v2/workspaces/default/model/<output.name>/-/v1` | `default/<output.name>` |

Use the same chat settings as LoRA inference (`messages[:-1]`, `max_tokens`, `temperature`, `enable_thinking` as appropriate). Post-training eval: run generation eval against this model-entity URL (not `eval_helpers.py --adapter`, which is LoRA-specific).
```

Use the user's platform URL in `NHX_BASE_URL` when they overrode it; omit the export line for default `http://127.0.0.1:8080`. Substitute `<provider>`, concrete URLs, and entity names with values from discovery — do not leave generic placeholders in the user-facing report. For **LoRA**, do **not** tell the user to update the deployment before calling the adapter — registration on the base model entity is sufficient. For **full SFT / merged / DPO**, tell the user they must deploy `<output.name>` before inference.

**Save report to `/tmp`** — unless the user opts out, write the full Markdown report (header, **Training configuration**, **Using the adapter** or **Using the fine-tuned model** when `completed`, and **Resources created** when a slug or new filesets were used) to `/tmp/fine-tune-result-<slug-or-job-suffix>.md`. Use the random slug from the run when one was assigned; otherwise use the job id suffix (e.g. `a925b07ff678`).

**Error follow-ups** — when the failure has a known fix, append sections **below** the header block (do not replace the header). Examples:

| Error type | Append |
|------------|--------|
| Missing training image + user-overridden `NHX_BASE_URL` | `references/troubleshooting.md` § **Missing training images** — on-target build steps, env vars, re-submit commands. **Do not** `docker build` locally for a remote platform. |
| Download fails / `Failed to access upstream storage` / 502 on gated HF model | `references/troubleshooting.md` § **Gated HuggingFace models** — create/update `hf-token`, add `token_secret` to fileset, confirm HF license, re-submit. |
| W&B not syncing / no `[launcher]` secret lines / `WandbCallback requires wandb` / wandb 401 | `references/troubleshooting.md` § **W&B / integrations not working** (jobs-launcher build, secret update, unsloth image). Setup: `references/integrations-setup.md`. |

For other terminal errors, keep the same header template; put remediation detail in **Notes** or a short **Next steps** section as appropriate.
