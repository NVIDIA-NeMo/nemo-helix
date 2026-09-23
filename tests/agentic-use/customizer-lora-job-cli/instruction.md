<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# LoRA Customization Job via NeMo Helix Customizer (GPU)

This task tests submitting and running a real LoRA fine-tuning job through the **nemo-customizer** plugin with the **nhx-automodel** backend. Training is dispatched through the NeMo Helix jobs pipeline to GPU containers built from the dev registry.

You have access to the `nemo` and `nhx` CLIs for NeMo Helix operations. Note: MCP tools are not available in this environment — you must use the CLI.

The CLIs are available at `/app/.venv/bin/nemo` and `/app/.venv/bin/nhx`. The platform API runs at http://localhost:8080. CLI auth is pre-configured.

## Context

- The NeMo Helix API server is running with the jobs controller and customization plugin enabled
- Platform image registry/tag are configured for `my-registry/nemo-helix-dev` (see environment setup)
- The Docker backend is configured for GPU job execution; the Docker socket is mounted
- A workspace `lora-training-workspace` has been pre-created
- A model entity `smollm-135m` (HF weights fileset `smollm-135m-weights`) has been registered in the workspace

## Task

### Step 1: Prepare and upload a training dataset

1. Create a JSONL file with at least 20 prompt/completion training examples for a simple task (e.g., customer service responses). Example format:

   ```jsonl
   {"prompt": "Customer complaint: My order arrived damaged.\nResponse:", "completion": "I apologize for the inconvenience. I will arrange a replacement immediately."}
   ```

2. Create a dataset fileset and upload the data:

   ```bash
   nemo files filesets create sft-training-data --workspace lora-training-workspace --purpose dataset --exist-ok
   nemo files upload /path/to/train.jsonl sft-training-data --workspace lora-training-workspace --remote-path train.jsonl
   ```

### Step 2: Submit a LoRA customization job (automodel backend)

Write `/tmp/job.json` using the **AutomodelJobInput** schema, then submit:

```bash
cat > /tmp/job.json <<'EOF'
{
  "model": "lora-training-workspace/smollm-135m",
  "dataset": {
    "training": "lora-training-workspace/sft-training-data"
  },
  "training": {
    "training_type": "sft",
    "finetuning_type": "lora",
    "lora": { "rank": 8, "alpha": 16 },
    "max_seq_length": 2048,
    "execution_profile": "default"
  },
  "schedule": { "epochs": 2 },
  "batch": { "global_batch_size": 4, "micro_batch_size": 1 },
  "optimizer": { "learning_rate": 0.0001 },
  "parallelism": { "num_nodes": 1, "num_gpus_per_node": 1, "tensor_parallel_size": 1 },
  "output": { "name": "lora-email-model" }
}
EOF

nemo customization automodel submit /tmp/job.json --workspace lora-training-workspace
```

Use `nemo customization automodel explain` if you need the live schema. List GPU execution profiles with `nemo jobs list-execution-profiles -f json` when choosing `training.execution_profile`.

### Step 3: Monitor the job

1. Poll job status with `nemo jobs get-status <job-name>` (job names are typically prefixed `automodel-`).
2. Check status periodically until it reaches a terminal state.
3. If it fails, investigate with `nemo jobs get-status <job-name>` and check platform logs for missing-image errors.

### Step 4: Verify results

Once the job completes (or if it fails), document:

- The job status
- Any error messages if it failed
- The output model/adapter if it completed

## Success Criteria

The task is complete when:

- A fileset `sft-training-data` exists with training data uploaded
- A customization job was submitted via `nemo customization automodel submit`
- The agent polled for job status and reported the outcome
- The job progressed beyond "created" status (indicating the jobs controller dispatched it)

## Notes

- LoRA/SFT training uses the **automodel** contributor (`nhx-automodel-training` / `nhx-customizer-tasks` images), not the legacy customizer automodel path
- Model reference format: `workspace/model-entity-name` (e.g., `lora-training-workspace/smollm-135m`)
- Dataset reference format in job JSON: `workspace/fileset-name` inside `dataset.training`
- Jobs may take a few minutes depending on dataset size and GPU availability
