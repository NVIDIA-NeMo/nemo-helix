<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Simple Custom Evaluation Job (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Available CLI Commands

### Workspace Commands

- `nemo workspaces create <name>` - Create a workspace

### Fileset Commands

All fileset commands accept `--workspace <name>` to specify the workspace.

- `nemo files filesets create <name> --workspace <workspace>` - Create a fileset
- `nemo files upload <local_path> <fileset_name> --workspace <workspace>` - Upload a file to a fileset
- `nemo files list <fileset_name> --workspace <workspace>` - List files in a fileset

### Evaluation Metric Commands

All evaluation commands accept `--workspace <name>` to specify the workspace.

- `nemo evaluation metrics create <name> --type string-check --params '<json>' --workspace <workspace>` - Create a metric
- `nemo evaluation metrics list --workspace <workspace>` - List metrics

The `string-check` metric type compares fields using an operation. Example params JSON:
```json
{"field": "output", "expected_field": "expected", "operation": "equals"}
```

### Evaluation Run Commands

- `nemo evaluation metrics run --name <metric_name> --data '<jsonl_rows>' --workspace <workspace>` - Run a synchronous evaluation with inline data

Each row in `--data` should be a JSON object with the fields referenced by the metric (e.g., `output` and `expected`).

### Metric Job Commands

- `nemo evaluation metric-jobs create <job_name> --metric-name <metric_name> --dataset-fileset <fileset_name> --workspace <workspace>` - Create an async metric evaluation job
- `nemo evaluation metric-jobs get <job_name> --workspace <workspace>` - Check job status
- `nemo evaluation metric-jobs list --workspace <workspace>` - List all metric jobs

**Note:** In some environments, jobs may stay in "created" status if the job controller is not running — that is expected.

## Task

Set up and run a custom evaluation using a string-check metric via the `nhx` CLI:

1. **Create a workspace** named `eval-test-workspace`

2. **Prepare and upload a dataset** - Create a JSONL dataset file with rows containing `output` and `expected` fields, for example:
   ```jsonl
   {"output": "hello", "expected": "hello"}
   {"output": "world", "expected": "world"}
   {"output": "foo", "expected": "bar"}
   ```
   Include at least one row where the output does NOT match the expected value. Upload it to the NeMo Helix Files service as a fileset named `eval-dataset` in the `eval-test-workspace` workspace.

3. **Create a string-check metric** in the workspace that compares the `output` field against the `expected` field using an equals operation

4. **Run a synchronous evaluation** using the metric you created against a small inline dataset. Verify that matching rows score 1.0 and mismatched rows score 0.0.

5. **Create an asynchronous metric evaluation job** that references the `eval-dataset` fileset and the metric you created. Note: in some environments, the job may stay in "created" status if the job controller is not running - that is expected.

6. **Check the job status** to confirm the job was created successfully

## Success Criteria

The task is complete when:
- A workspace `eval-test-workspace` exists
- A fileset `eval-dataset` exists with the dataset uploaded
- A string-check metric exists in the workspace
- A synchronous evaluation has returned results with scores
- A metric evaluation job has been created
