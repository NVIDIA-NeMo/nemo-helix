<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- NeMo-RL (DPO + GRPO) job JSON reference. Index + source-of-truth: `hyperparameters.md`. Preference dataset formats: `dataset-formats.md` § NeMo-RL. -->

# NeMo-RL job JSON (DPO + GRPO)

The `rl` backend (`nemo customization rl submit`) runs on a Ray cluster — **Kubernetes runtime only**, full-weight (no LoRA). Schema: `RlJobInput` in `plugins/nemo-rl/src/nemo_rl_plugin/schema.py`. Run `nemo customization rl explain` for the live schema.

Discriminated by `training.type`: `"dpo"` or `"grpo"`. **`training.type` is required** — it is the union discriminator, so a payload that omits it is rejected with `union_tag_not_found` rather than defaulting to DPO.

## DPO job JSON layout

```json
{
  "model": "default/<model-entity>",
  "dataset": "default/<preference-fileset>",
  "training": {
    "type": "dpo",
    "epochs": 1,
    "learning_rate": 5e-6,
    "max_seq_length": 1024,
    "batch_size": 32,
    "micro_batch_size": 1,
    "ref_policy_kl_penalty": 0.05,
    "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1 }
  },
  "output": { "name": "<output-name>" }
}
```

- `model` is a **string** ref to a registered model entity (`"name"` or `"workspace/name"`) — not an object (unsloth) and not the HF id.
- `dataset` is a **single string** ref to a preference fileset containing `training.jsonl` + `validation.jsonl` (see `references/dataset-formats.md` § NeMo-RL). There is no separate validation ref.
- `output.name` is the full-weight model entity the job registers. Output type is always `model` (no adapter).

## GRPO job JSON layout (NeMo Gym)

GRPO needs **two** FileSets: an environment package (code + config that runs a rollout and returns a reward) and a dataset of prompt rows. Building or converting the environment is the bulk of the work and has its own reference — **`gym-environments.md`** covers all three formats (`native-v1`, `wheels-v1`, `adapter-wheels-v1`), how to convert a Prime Intellect / verifiers env, how to package one of Gym's own, validation, and upload.

The short version, for a Prime Intellect hub env, on an **internet-capable host** (training clusters have no hub egress):

```bash
uv run --package nmp-rl pi-to-gym-conversion \
  --hub-id primeintellect/ascii-tree --hub-version 0.1.5 \
  --out-dir ./ascii-tree-pkg --validation-fraction 0.1 --upload
```

That writes the package plus `training.jsonl` / `validation.jsonl` and, with `--upload`, creates both FileSets. Then submit:

```json
{
  "model": "default/<model-entity>",
  "dataset": "default/<gym-jsonl-fileset>",
  "environment": "default/<environment-fileset>",
  "training": {
    "type": "grpo",
    "epochs": 1,
    "batch_size": 32,
    "micro_batch_size": 1,
    "num_generations_per_prompt": 8,
    "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1 }
  },
  "output": { "name": "<output-name>" }
}
```

- `environment` is **required** for GRPO — a string ref to a FileSet with `purpose=environment` carrying a valid `nemo-environment.yaml`. Any of the three formats works. See `gym-environments.md`.
- `dataset` is Gym JSONL (`training.jsonl` required, `validation.jsonl` optional) — rollout rows with the prompt under `responses_create_params.input`, **not** DPO preference triples and **not** `messages[]` at the top level. Schema and conversion: `dataset-formats.md` § **NeMo-RL (GRPO)**.
- `sandboxed` is **not** a job field — platform config `NMP_RL_SANDBOXED_GYM_DEFAULT` (default `true`). Shared clusters fail closed until OpenSandbox is declared installed (`NMP_SANDBOX_CLUSTER_CAPABLE=true`, Helm `sandboxClusterCapable` — a **platform** setting, not `NMP_RL_*`) and `NMP_RL_JOB_STORAGE_PVC_CLAIM` names the job-storage PVC the Gym sandbox re-mounts for the environment and dataset. Both are checked **at submit** — the job compiler raises and the API returns 422, before any GPU is claimed. See `rl-kubernetes-runtime.md` § **Sandboxed Gym (GRPO)**.

## Field reference — shared training knobs

### General

| Field | Default | Notes |
|-------|---------|-------|
| `learning_rate` | `1e-4` | Peak LR. DPO typically uses a **low** LR (e.g. `5e-6`–`1e-5`). |
| `min_learning_rate` | `null` | Floor for cosine decay. |
| `weight_decay` | `0.01` | |
| `adam_beta1` / `adam_beta2` | `0.9` / `0.999` | Adam betas. |
| `adam_eps` | `1e-5` | Adam epsilon (numerical stability). |
| `warmup_steps` | `0` | Linear warmup steps. |
| `optimizer_type` | `null` → `adamw_with_cosine_annealing` | One of `adamw_with_cosine_annealing`, `adam_with_cosine_annealing`, `adamw_with_flat_lr`, `adam_with_flat_lr` (optimizer × LR-scheduler). |
| `epochs` | `1` | Passes over the dataset. |
| `max_steps` | `null` | Global step cap. Caps the run at `min(max_steps, epochs × steps_per_epoch)`, so it's safe to combine with `epochs` to stop smoke jobs mid-epoch — omit for real runs. |
| `val_check_interval` | `null` | Float ≤ 1.0 = fraction of epoch; > 1.0 = step count. |
| `val_at_end` | `true` | Run a final validation pass after the last step. Keep enabled: it makes the final checkpoint carry validation metrics so best-checkpoint selection (`metric_name`/`keep_top_k`) works — otherwise NeMo-RL warns and falls back to the latest checkpoint. Set `false` only to skip the extra eval. **GRPO:** ignored when the dataset ships no `validation.jsonl` — the compiler disables validation rather than fail in `setup()`. |
| `keep_top_k` | `1` | Number of best checkpoints to retain (DPO ranks by validation loss; GRPO by `val:total_reward/mean`). |
| `batch_size` | `32` | Global batch across all GPUs. **DPO:** preference **pairs**. **GRPO:** must divide `num_prompts_per_step × num_generations_per_prompt` — see the GRPO table. |
| `micro_batch_size` | `1` | Per-GPU micro batch. |
| `max_seq_length` | `2048` | Max token sequence length. |
| `activation_checkpointing` | `false` | Recompute activations in the backward pass to cut memory — the first knob to enable for OOM / larger models / longer sequences. |
| `seed` | `null` → `42` | |
| `execution_profile` | `null` | GPU execution profile; falls back to the service default. |

### DPO-specific (`type: "dpo"`)

| Field | Default | Notes |
|-------|---------|-------|
| `ref_policy_kl_penalty` | `0.05` | **β** in the DPO paper — strength of the KL penalty tying the policy to the reference model. Higher = stay closer to the reference. The main DPO knob. |
| `preference_loss_weight` | `1.0` | Weight on the preference (DPO) loss term. |
| `sft_loss_weight` | `0.0` | Weight on an auxiliary SFT regularization loss (`0` = pure DPO). Raise (e.g. `0.1`) to anchor the policy to the chosen responses. |
| `preference_average_log_probs` | `false` | Normalize preference log-probs by sequence length. |
| `sft_average_log_probs` | `false` | Normalize SFT-loss log-probs by sequence length. |
| `max_grad_norm` | `1.0` | Gradient clipping norm. |

### GRPO-specific (`type: "grpo"`)

| Field | Default | Notes |
|-------|---------|-------|
| `num_generations_per_prompt` | `8` | Group size for relative advantages. |
| `num_prompts_per_step` | `null` | Derived from `batch_size / num_generations_per_prompt` when omitted. `num_prompts_per_step × num_generations_per_prompt` must be a multiple of `batch_size` (enforced by `validate_for_training`), so prefer a `num_generations_per_prompt` that divides `batch_size`. |
| `automodel_kwargs` | `null` | Passed to `policy.dtensor_cfg.automodel_kwargs`. **Requires `training.policy_backend: "automodel"`** for either LoRA or full-weight GRPO; under `"dtensor"` it is rejected at submit rather than ignored. `{"force_hf": true}` loads stock HuggingFace modules when a model's custom Automodel backbone is not compatible with the parallelizer; a `{"backend": {...}}` block picks the Transformer-Engine / DeepEP MoE implementation. |
| `router_aux_loss_coef` | `null` | MoE router auxiliary-loss coefficient, applied as a **top-level** HuggingFace config override. Set `0.0` for RL on a MoE model — the aux load-balancing loss is a pretraining regularizer and adds a gradient term unrelated to the reward. Models that nest their config need `hf_config_overrides` instead; a top-level key the model does not read is absorbed silently, leaving the aux loss on. |
| `hf_config_overrides` | `null` | Passed to NeMo-RL's `policy.hf_config_overrides` verbatim, which forwards it to the training model as HuggingFace config kwargs and to vLLM as `hf_overrides`. Nesting is preserved, so this reaches models that namespace their config — Qwen3.5 reads the router coefficient under `text_config`, i.e. `{"text_config": {"router_aux_loss_coef": 0.0}}`. Setting `router_aux_loss_coef` here *and* as its own field is rejected at submit time. |
| `vllm_tensor_parallel_size` | `null` | Tensor parallelism for the rollout engine alone. Defaults to `min(parallelism.tensor_parallel_size, parallelism.num_gpus_per_node)`. Set it when the model needs several GPUs to hold inference weights but you want the policy trained at a different tensor-parallel size. |
| `vllm_gpu_memory_utilization` | `0.5` | Fraction of each GPU vLLM reserves for weights plus KV cache. Raise toward `0.7` for large models, which otherwise cannot load their weight shard. |
| `val_at_start` | `false` | Run validation **before** the first training step. Enable to measure uplift: baseline and result come from one job on the same data and generation settings, instead of a separate baseline run that must be kept in sync. Costs a full rollout pass, hence off by default. Ignored without a `validation.jsonl`. **GRPO only** — DPO always validates at step 0, since its validation needs no generation. |
| `max_new_tokens` | `null` → `max_seq_length` | Cap on tokens generated per rollout turn. **Known limitation:** NeMo Gym's `verifiers_agent` does not yet honour this — see below. |
| `normalize_rewards` | `true` | |
| `overlong_filtering` | `false` | Zero the loss contribution of rollouts truncated at the generation cap, so the policy is not penalised for responses it was cut off from finishing. Enable when many rollouts are truncated. |
| `max_rollout_turns` | `1` | Multi-turn rollouts; most math envs use `1`. |
| `ref_policy_kl_penalty` | `0.0` | KL coefficient in the GRPO loss. |
| `ratio_clip_min` / `ratio_clip_max` | `0.2` / `0.28` | PPO-style ratio clip bounds. |
| `max_grad_norm` | `1.0` | Gradient clipping norm. |
| `top_k` | `null` | Restrict rollout sampling to the k most likely tokens. Unset samples the full distribution, which is standard GRPO. Narrowing it carries the same risk as lowering `temperature`: rollouts within a prompt group become more alike, and GRPO learns from how much they differ. |
| `batching_strategy` | `"dynamic"` | How rollouts are grouped into training micro-batches. `dynamic` fills each micro-batch to a token budget, so short rollouts share a batch instead of each paying for a full-length pad; this is the DTensor/Automodel norm in NeMo-RL's recipes and the only mode with no architecture restrictions on this path. `sequence_packing` concatenates rollouts into packed sequences — rejected for `context_parallel_size > 1`. `static` is one rollout per slot. One field rather than two flags because NeMo-RL asserts `dynamic` and `sequence_packing` are mutually exclusive. |
| `train_mb_tokens` | `null` → `max_seq_length × micro_batch_size` | Token budget per training micro-batch, read by `dynamic` and `sequence_packing`. The default is the peak `static` already provisions for, so it does not raise peak memory. Must be at least `max_seq_length`, else the longest rollout fits in no micro-batch. Lower it if you OOM. |
| `sequence_length_round` | `64` | Round bucketed micro-batch sequence lengths up to a multiple of this. Only read by `dynamic`. |

### `policy_backend` — which worker trains the model (GRPO only)

Lives under `training`, a **sibling of `parallelism`** and not a field on it. The value picks the NeMo-RL policy worker, which picks the Ray actor's venv and kernels.

**Leave `policy_backend` unset to get the default for each `finetuning_type`.** LoRA requires `automodel`. Full-weight defaults to `dtensor` and may be set to `automodel` when you want a consolidated HuggingFace export or expert parallelism for a MoE model.

| `finetuning_type` | Default backend | Worker | Notes |
|---|---|---|---|
| `"lora"` | `"automodel"` | `DTensorPolicyWorkerV2` (`_v2: true`) | Required pairing. Also the only backend with `expert_parallel_size > 1` and `automodel_kwargs`. Needs Transformer Engine, so **Hopper or newer** |
| `"all_weights"` | `"dtensor"` | `DTensorPolicyWorker` | Stock HuggingFace + PyTorch FSDP2, no Transformer Engine — also the **pre-Hopper** option. Set `policy_backend: "automodel"` to train full weights on V2. MoE models using expert parallelism require `automodel`; `dtensor` rejects `expert_parallel_size > 1`. |

`dtensor` plus LoRA / expert parallelism / `automodel_kwargs` is **rejected at submit**: the worker does not implement them; left to NeMo-RL, LoRA dies in a Ray worker and the other two are ignored silently. Every conflict is listed at once.

Pre-Hopper GPUs must keep the default `dtensor` backend. On Hopper or newer, switch to `automodel` only when you need its consolidated export or expert parallelism. `megatron` is not selectable: the image builds the extra, but the compiler still emits an inert `megatron_cfg`.

`v4_compatible` (`GRPOTraining.v4_compatible`, default `true`) is the compatibility control for that Automodel full-weight export. Automodel otherwise writes a transformers-v5 `config.json` that the platform's vLLM cannot load, so the compiler keeps the base checkpoint's v4 `config.json` on the published model and writes the in-memory v5 config beside it as `config.v5.json`. Set it `false` to export the v5 file as `config.json` instead. If the base checkpoint already uses transformers v5, the job logs warn you to opt out. The field has no effect on `dtensor` full-weight jobs or LoRA adapters.

```json
{
  "training": {
    "type": "grpo",
    "finetuning_type": "all_weights",
    "policy_backend": "automodel",
    "v4_compatible": false
  }
}
```

### GRPO advanced (`type: "grpo"`)

Collapse these behind an "advanced" affordance — the defaults are correct for standard GRPO and most runs never touch them.

| Field | Default | Notes |
|-------|---------|-------|
| `ratio_clip_c` | `null` | Dual-clip bound: puts a floor under the loss for tokens with a negative advantage, so one badly-scored rollout cannot dominate the update. Must be **greater than 1**; `3` is the usual choice. `null` leaves dual clipping off. Rejected at submit if `<= 1`, because the loss function asserts it at startup. |
| `advantage_clip_low` / `advantage_clip_high` | `null` / `null` | Bounds applied to advantages **after** normalization. `null` on either side means unbounded there. Bounding both limits how much one unusual rollout can move the policy. Rejected at submit when `low >= high`: an inverted or empty range clips every advantage to a single value, which zeroes the gradient and looks like a run that simply does not learn. |
| `normalize_rewards` | `true` | Divide each group's advantages by their standard deviation. |
| `use_leave_one_out_baseline` | `true` | Compare each rollout against the mean of the **other** rollouts in its group rather than the group mean including itself, which removes the bias of comparing a rollout to itself. |
| `use_on_policy_kl_approximation` | `true` | Estimate the KL term against the policy that generated the rollouts rather than the one currently training. The two differ once a rollout batch is reused across several optimizer steps. |
| `use_importance_sampling_correction` | `true` | Reweight each token by how much more or less likely the training policy is to produce it than the rollout policy was. Corrects the drift that builds up when a rollout batch is reused, at the cost of higher variance. |

The last two default to `true` on this platform, where the underlying library defaults them to `false`. They were fixed on before they were settings, and the defaults preserve that so an unchanged job computes the same loss it did before.

### LoRA (GRPO only)

**DPO is full-weight only.** GRPO trains either way, selected by `finetuning_type` on the `training` block:

| `finetuning_type` | Trains | Output |
|---|---|---|
| `"all_weights"` (default) | Every weight | Full model entity |
| `"lora"` | Adapter only | HF PEFT **adapter** entity, registered against the base model |
| `"lora_merged"` | — | **Rejected.** The DTensor path does not merge at export. |

```json
{
  "model": "default/<model-entity>",
  "dataset": "default/<gym-dataset-fileset>",
  "environment": "default/<environment-fileset>",
  "training": {
    "type": "grpo",
    "finetuning_type": "lora",
    "lora": { "rank": 16, "alpha": 32, "dropout": 0.0 },
    "epochs": 1,
    "batch_size": 32,
    "num_generations_per_prompt": 8,
    "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1 }
  },
  "output": { "name": "<output-name>" }
}
```

**You do not set the output type.** It is inferred: `finetuning_type: "lora"` produces an adapter entity, anything else a model entity. `output` carries only `name`.

| `lora` field | Default | Notes |
|---|---|---|
| `rank` | `16` | LoRA rank, mapped to NeMo-RL's `lora_cfg.dim`. |
| `alpha` | `32` | Scaling factor. The effective learning-rate multiplier on the adapter is `alpha / rank`, so changing `rank` without `alpha` also changes the effective LR. |
| `dropout` | `0.0` | Applied after the adapter. |
| `target_modules` | `null` | Module names to adapt. |
| `exclude_modules` | `null` | Module name patterns to skip. |
| `use_triton` | `null` → compiler picks (`true` at TP 1, `false` above) | Automodel's Triton LoRA kernels. **Leave it unset.** An explicit `true` with `tensor_parallel_size > 1` is **rejected at submit**, not silently downgraded — the kernels take raw tensors and TP makes the adapter weights DTensors. Setting `false` explicitly is always allowed. |

**Module selection is either/or.** Leaving both `target_modules` and `exclude_modules` unset turns on `match_all_linear`, which adapts every linear layer. Setting **either** list turns it off, because Automodel's `ModuleMatcher` rejects `match_all_linear` alongside a list. So `exclude_modules` alone does not mean "all linear except these" — it means "only what the exclusion logic leaves", which is a different and usually narrower set. Prefer `target_modules` when you want a specific set, and neither field when you want everything.

**Rules the schema enforces:**

- `lora` must be **omitted** when `finetuning_type` is `all_weights` — supplying both fails with `lora must be omitted when finetuning_type is all_weights`.
- Omitting `lora` while asking for `finetuning_type: "lora"` is fine: defaults are filled in.
- `lora_merged` is rejected at the schema level, so a merged checkpoint is not reachable from a GRPO job. To serve merged weights, train full-weight instead.

**The backend follows `training.policy_backend`, and `_v2: true` follows that field and nothing else.** Left unset it resolves from `finetuning_type`: `lora` gets `automodel`, `all_weights` gets `dtensor`. `dtensor` plus LoRA is **rejected at submit**. `automodel` plus `all_weights` is allowed. Note the consequence for comparisons: a LoRA run and a default full-weight run use **different workers**, so their results are not directly comparable.

### Using a GRPO adapter

The job registers an HF PEFT adapter entity whose base model is the entity you trained from, with `rank` and `alpha` recorded on it. It behaves like any other platform LoRA adapter: it hot-reloads onto a **READY** deployment of the base model that has `lora_enabled: true`, so there is no deployment to create before evaluating it. Route inference through the **provider** gateway (`/provider/<name>/-/v1` with `model: default--<adapter>`) — the model-entity path always hits the base model. See `post-training-eval.md` § **Request routing (base vs LoRA)**.

Full-weight GRPO instead registers a new **model** entity, which does need its own deployment before inference.

### When to use LoRA for GRPO

| Situation | Pick |
|---|---|
| Limited GPU memory, or the full-weight run OOMs after `activation_checkpointing` | **LoRA** — the optimizer state is the adapter, not the model |
| Several reward environments over one base model | **LoRA** — adapters share the base deployment |
| Reward barely moves and the task needs broad behaviour change | **Full weights** — the adapter may be the bottleneck; raise `rank` first, then reconsider |
| You need merged weights to export or serve elsewhere | **Full weights** — GRPO does not merge adapters |

### `parallelism`

Same block as automodel (`num_nodes`, `num_gpus_per_node`, `tensor_parallel_size`, `pipeline_parallel_size`, `context_parallel_size`, `sequence_parallel`), plus `expert_parallel_size` for MoE policy training (GRPO only; **implemented only by `policy_backend: "automodel"`** — a value above `1` under `"dtensor"` is rejected at submit, not silently upgraded). Divisibility rule (enforced by `RlJobOutput.validate_for_training`): `total_gpus = num_nodes × num_gpus_per_node` must be divisible by `tensor_parallel_size × pipeline_parallel_size × context_parallel_size × expert_parallel_size`, and `batch_size` by `micro_batch_size × data_parallel_size`. **Multi-node (`num_nodes > 1`)** additionally requires the platform to set `NMP_RL_MULTINODE_SHARED_STORAGE_PATH` (shared filesystem for Ray's cross-node coordination); the compiler fails fast otherwise.

### Known limitation: `max_new_tokens` does not reach the agent

The compiler stamps `max_new_tokens` onto `policy.generation.max_new_tokens`, and NeMo-RL puts it on every rollout row as `responses_create_params.max_output_tokens`. NeMo Gym's `verifiers_agent` then **drops it** — it reads `max_tokens` from its own environment config instead. The effective per-turn cap is therefore `min(agent max_tokens, max_seq_length - prompt_len)`, the second term applied by vLLM's `_clamp_max_tokens`.

This is not fatal, because the cap is a minimum of the two and `max_seq_length` is under your control. But until the agent honours `max_output_tokens`, bound response length through `max_seq_length`, or through `max_tokens` in the environment package's own config — not through `max_new_tokens`.

### How validation sampling works

There is no validation counterpart to `num_generations_per_prompt`. NeMo-RL's `validate()` runs **exactly one rollout per row** of `validation.jsonl` — no per-prompt fan-out. To score a prompt *k* times (mean@k), repeat its **row** *k* times when building the dataset. Validation cost is therefore just the row count, and `val_at_start` / `val_at_end` score the same rows, which makes the before/after comparison paired.

## Integrations (W&B / MLflow)

rl supports **W&B and MLflow** through the top-level `integrations` object (`integrations.wandb` / `integrations.mlflow`) — the same object shape used across all backends; full field reference in `hyperparameters.md` § **Integrations (all backends)**. rl specifics: the run name defaults to the **job id**, tags are auto-prefixed (`service:rl`, `framework:…`), and because rl runs on Kubernetes / Ray the `tracking_uri` / self-hosted W&B `base_url` must be reachable **from the cluster** (the local `docker0` recipe in `integrations-setup.md` is Docker-runtime only).

## DPO tuning guide

| Symptom | Action |
|---------|--------|
| Policy degenerates / drifts too far | Raise `ref_policy_kl_penalty` (β), e.g. `0.05` → `0.1`–`0.5`. |
| Barely changes from the reference | Lower β, or raise `learning_rate` one step (still keep it low for DPO). |
| Forgets base capabilities (DPO) | Add SFT regularization: `sft_loss_weight` `0.1`–`0.5`. |
| CUDA OOM | Set `activation_checkpointing: true`; then lower `micro_batch_size`; then `max_seq_length`. **GRPO:** also lower `num_generations_per_prompt` (keeping `batch_size` divisible). |
| Instability | Lower LR; raise β. **DPO:** check preference data quality. **GRPO:** check reward scale and `ratio_clip_max`. |
