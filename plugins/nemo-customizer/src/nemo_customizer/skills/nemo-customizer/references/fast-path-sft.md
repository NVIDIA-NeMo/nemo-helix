<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fast path — automodel and unsloth (SFT / LoRA)

Copy-paste commands and job JSON for the two Docker/Kubernetes GPU SFT backends, after `SKILL.md` § **Pre-flight** and **Plugin pick**. The rules each step must follow are in `SKILL.md` § **Workflow**.

Substitute `<hf-repo>`, `<hf-dataset>`, `<model-entity>`, `<weights-fileset>`, `<dataset-fileset>`, `<output-name>`.

## Fast path — automodel

### 1. Dataset — convert per `references/hf-conversion.md`, then:

```bash
DATASET=<dataset-fileset>   # e.g. commonsense_qa
nemo files filesets create "$DATASET" --workspace default --purpose dataset
nemo files upload /tmp/train-data/train.jsonl "$DATASET" --workspace default --remote-path train.jsonl
# validation.jsonl if present
nemo files list "$DATASET" --workspace default

# The frozen test split, in its own fileset (post-training-eval.md)
nemo files filesets create "$DATASET-test" --workspace default --purpose dataset
nemo files upload /tmp/test-data/test.jsonl "$DATASET-test" --workspace default --remote-path test.jsonl
```

**2. Model** — skip if entity exists (`nemo models list --workspace default`).

```bash
WEIGHTS=<weights-fileset>   # e.g. nemotron-3-5-lightning-30b-a3b
MODEL_ENTITY=<model-entity>   # Models API entity (not dataset fileset, not HF id)
HF_REPO=<hf-repo>           # e.g. nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

nemo files filesets create "$WEIGHTS" --workspace default --purpose model --exist-ok \
  --storage '{"type":"huggingface","repo_id":"'"$HF_REPO"'","repo_type":"model","revision":"main"}'

nemo models create "$MODEL_ENTITY" --workspace default --exist-ok \
  --input-data '{"name":"'"$MODEL_ENTITY"'","fileset":"default/'"$WEIGHTS"'","custom_fields":{"hf_model_id":"'"$HF_REPO"'"}}'
```

For a gated repo, add `"token_secret":"hf-token"` to the `--storage` JSON (`SKILL.md` § **Pre-flight**).

### Create resources at submit

Instead of steps 1 and 2, `--upload-model <local path or HF repo id>` and `--upload-dataset <local path>` create the model and dataset; rl GRPO also has `--upload-environment <local dir>`. Any flag works on its own, on two surfaces:

- `nemo customization --upload-model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --upload-dataset ./sft-data` creates the resources and prints their refs as JSON on stdout, to paste into the job JSON.
- `nemo customization <plugin> --job-json job.json --upload-model … --upload-dataset …` does the same and fills the refs into the job it submits. The file on disk is **not** rewritten, and it may leave out `model`/`dataset` entirely.

- **A ref belongs in one place.** Submit **fails**, before creating anything, when the job JSON sets `model` or `dataset` and you also pass the matching flag: the two name different resources. Remove the field to create it at submit, or drop the flag to use the file's ref.
- **A source is one file or one directory**, as with `nemo files upload`; a directory uploads recursively and keeps local file names. `--upload-model` also takes an HF repo id. `--upload-dataset` takes local files only, so convert an HF dataset first.
- **One `--upload-dataset` fills both dataset fields:** automodel's `dataset.training` and `dataset.validation`, unsloth's `dataset.path` and `dataset.validation_path`, all pointing at the one fileset. Upload a directory holding both splits (`train.jsonl` + `val.jsonl`); automodel's `train*`/`val*` discovery separates them. `rl` has one `dataset` ref and needs `training.jsonl` + `validation.jsonl` in it.
- **`--upload-environment` uploads a Gym package, it does not build one**; the **nemo-gym-environments** skill builds and validates it. automodel and unsloth reject the flag.
- **Names come from the source** (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` → `nvidia-nemotron-3-5-lightning-30b-a3b-bf16`, `sft-data/` → `sft-data`). A name the platform would reject gets a `model-`/`dataset-` prefix (`2024-train.jsonl` → `dataset-2024-train`).
- **`--exist-ok` applies to whichever sources were passed**, with the same stale-files risk as SKILL.md step 2. The CLI prints `Reused <ref>, which already existed.` when it reuses one.

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

Or poll manually: `nemo jobs get-status "$JOB"` every 30–60s.

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

If submit fails on an unknown profile, re-list execution profiles and pass `--profile <name>` on submit (default is `gpu`).

