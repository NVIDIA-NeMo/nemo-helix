<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fast path — automodel and unsloth (SFT / LoRA)

Copy-paste commands and job JSON for the two Docker/Kubernetes GPU SFT backends.

Substitute `<hf-repo>`, `<hf-dataset>`, `<model-entity>`, `<weights-fileset>`, `<dataset-fileset>`, `<output-name>`.

## Fast path — automodel

### Setup

```bash
# `nemo` resolves its target from the active config context (~/.config/nhx/config.yaml)
# or an explicit NHX_BASE_URL. Do NOT export a localhost fallback — it would override a
# configured remote. Leave NHX_BASE_URL unset to use your saved context; set it
# explicitly (export NHX_BASE_URL=http://127.0.0.1:8080) only to force a local platform
# for this shell when you have no local config context.
cd /path/to/nemo-helix
nemo auth status   # skip login if auth disabled; if enabled + unsigned JWT allowed → login --unsigned-token --email admin@example.com
nemo jobs list-execution-profiles -f json   # platform GPU profiles → automodel; set training.execution_profile if needed
```

### 1. Dataset — convert per `references/hf-conversion.md`, then:

```bash
DATASET=<dataset-fileset>   # e.g. commonsense_qa; new name when the data changes (no --exist-ok — see references/gotchas.md)
nemo files filesets create "$DATASET" --workspace default --purpose dataset
nemo files upload /tmp/train-data/train.jsonl "$DATASET" --workspace default --remote-path train.jsonl
# validation.jsonl if present
nemo files list "$DATASET" --workspace default
```

**2. Model** — skip if entity exists (`nemo models list --workspace default`). For **gated** HF repos, complete **HuggingFace token (gated models)** first — see `references/troubleshooting.md` § **Gated HuggingFace models** for `token_secret` on the fileset.

```bash
WEIGHTS=<weights-fileset>   # e.g. nemotron-3-5-lightning-30b-a3b
MODEL_ENTITY=<model-entity>   # Models API entity (not dataset fileset, not HF id)
HF_REPO=<hf-repo>           # e.g. nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

nemo files filesets create "$WEIGHTS" --workspace default --purpose model --exist-ok \
  --storage '{"type":"huggingface","repo_id":"'"$HF_REPO"'","repo_type":"model","revision":"main"}'

nemo models create "$MODEL_ENTITY" --workspace default --exist-ok \
  --input-data '{"name":"'"$MODEL_ENTITY"'","fileset":"default/'"$WEIGHTS"'","custom_fields":{"hf_model_id":"'"$HF_REPO"'"}}'
```

For gated repos, add `"token_secret":"hf-token"` to the `--storage` JSON (after creating the secret). See troubleshooting § **Gated HuggingFace models**.

**3. Job JSON** — write `/tmp/job.json`. `model` is the **registered model entity** (`default/<model-entity>`), not an HF repo id or dataset fileset. The block below holds the 1-GPU defaults for a small dense model. For a named model, take the values from its recipe in `references/recipes.md`. Full hyperparameter reference: `references/hyperparameters-automodel.md`.

```json
{
  "model": "default/<model-entity>",
  "dataset": {
    "training": "default/<dataset-fileset>",
    "validation": "default/<dataset-fileset>"
  },
  "training": {
    "training_type": "sft",
    "finetuning_type": "lora",
    "lora": { "rank": 16, "alpha": 32 },
    "max_seq_length": 2048
  },
  "schedule": { "epochs": 1 },
  "batch": { "global_batch_size": 4, "micro_batch_size": 1 },
  "optimizer": { "learning_rate": 5e-5, "weight_decay": 0.01, "warmup_steps": 0 },
  "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1, "tensor_parallel_size": 1 },
  "output": { "name": "<output-name>" }
}
```

**4. Submit and poll**

```bash
nemo customization automodel --job-json /tmp/job.json --workspace default > /tmp/automodel-submit.json
JOB=$(python3 -c "import json;print(json.load(open('/tmp/automodel-submit.json'))['name'])")
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"
```

`JOB` is the `"name"` field of submit stdout (JSON). It already carries the `automodel-` prefix, so pass it verbatim; do not prepend `automodel-` again. **Do not use `2>&1`** before `json.load` — warnings on stderr break parsing; see `references/gotchas.md`. Optional interval override: append seconds (e.g. `… 30`). Or poll manually: `nemo jobs get-status "$JOB"` every 30–60s.

## Fast path — unsloth

Same substitutions as automodel. Steps 1 (dataset) and 2 (model entity) are identical — the differences are the job JSON shape (`UnslothJobInput`) and the `unsloth --job-json` command.

**1. Dataset** — same as automodel Fast path step 1.

**2. Model** — same as automodel Fast path step 2.

**3. Job JSON** — write `/tmp/job.json` using the **`UnslothJobInput`** shape (see `references/hyperparameters-unsloth.md`). `model` is an **object** (not a string), `dataset.path` is a single fileset ref, `hardware.gpus` replaces the `parallelism` block (single GPU in the training container). `nemo customization unsloth explain` prints the live schema.

```json
{
  "name": "<job-name>",
  "model": {
    "name": "default/<model-entity>",
    "max_seq_length": 2048,
    "load_in_4bit": true,
    "dtype": "auto"
  },
  "dataset": {
    "path": "default/<dataset-fileset>",
    "text_field": "text",
    "apply_chat_template": true
  },
  "training": {
    "training_type": "sft",
    "finetuning_type": "lora",
    "lora": { "rank": 16, "alpha": 32 }
  },
  "schedule": { "epochs": 1, "warmup_ratio": 0.1 },
  "batch": { "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4 },
  "optimizer": { "learning_rate": 5e-5, "optim": "adamw_8bit" },
  "hardware": { "gpus": "0", "precision": "bf16" },
  "output": { "name": "<output-name>", "save_method": "lora" }
}
```

If the model uses `messages` chat format (preferred when the tokenizer has a chat template), keep `dataset.apply_chat_template: true`. Otherwise emit a single `text` column from your converter and set `apply_chat_template: false`.

**4. Submit and poll**

```bash
nemo customization unsloth --job-json /tmp/job.json --workspace default > /tmp/unsloth-submit.json
JOB=$(python3 -c "import json;print(json.load(open('/tmp/unsloth-submit.json'))['name'])")
bash plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/poll_customization_job.sh "$JOB"
```

`JOB` is the `"name"` field of submit stdout (JSON). It already carries the `unsloth-` prefix, so pass it verbatim; do not prepend `unsloth-` again. **Do not use `2>&1`** before `json.load` — warnings on stderr break parsing; see `references/gotchas.md`. Optional interval override: append seconds (e.g. `… 30`). Or poll manually: `nemo jobs get-status "$JOB"` every 30–60s. If submit fails on an unknown profile, re-list execution profiles and pass `--profile <name>` on submit (default is `gpu`).

Use `nemo customization unsloth --job-json <job.json>` for Unsloth jobs; the contributor CLI exposes no local `run` verb.

