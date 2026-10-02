<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Academic Benchmark Evaluation (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Available CLI Commands

### Workspace Commands

- `nemo workspaces create <name>` - Create a workspace

### Benchmark Commands

All benchmark commands accept `--workspace <name>` to specify the workspace.

- `nemo evaluation benchmarks list --workspace <workspace>` - List available benchmarks
- `nemo evaluation benchmarks get <name> --workspace <workspace>` - Get benchmark details

System benchmarks (MMLU, GSM8K, HumanEval, etc.) are in the `system` workspace. They are seeded asynchronously at startup, so if the list is initially empty, wait a few seconds and retry.

### Benchmark Job Commands

- `nemo evaluation benchmark-jobs create --spec '<json>' --workspace <workspace>` - Create a benchmark evaluation job
- `nemo evaluation benchmark-jobs get <name> --workspace <workspace>` - Get job details
- `nemo evaluation benchmark-jobs get-status <name> --workspace <workspace>` - Get job status
- `nemo evaluation benchmark-jobs list --workspace <workspace>` - List all benchmark jobs

The `--spec` parameter is a JSON string describing the benchmark job. For a system benchmark online evaluation, use this format:
```json
{
  "benchmark": "system/<benchmark-name>",
  "model": {
    "url": "<model-endpoint-url>",
    "name": "<model-name>"
  }
}
```

For example, to evaluate MMLU against a model:
```json
{
  "benchmark": "system/mmlu-instruct",
  "model": {
    "url": "http://localhost:8000/v1",
    "name": "mock-model"
  }
}
```

**Note:** In some environments, jobs may stay in "created" status if the job controller is not running — that is expected.

## Task

Set up and launch an academic benchmark evaluation job using the `nemo` CLI:

1. **Create a workspace** named `benchmark-eval-workspace`

2. **Discover available system benchmarks** - List the benchmarks available in the `system` workspace to find academic benchmarks like MMLU. System benchmarks are seeded asynchronously at startup, so if the list is initially empty, wait a few seconds and try again.

3. **Create an academic benchmark evaluation job** in the `benchmark-eval-workspace` workspace that runs an MMLU benchmark against a model. Use the following model configuration:
   - Model URL: `http://localhost:8000/v1`
   - Model name: `mock-model`

   The benchmark job spec should reference an MMLU system benchmark you discovered (e.g. `system/mmlu-instruct`).

4. **Verify the job was created** - Retrieve the job details and check its status. Note: in this environment the job may remain in "created" status since the job execution worker is not running - that is expected.

5. **List all benchmark jobs** in the workspace to confirm the job appears in the list.

## Success Criteria

The task is complete when:
- A workspace `benchmark-eval-workspace` exists
- System benchmarks were listed from the `system` workspace
- A benchmark evaluation job was created referencing an MMLU-based system benchmark
- The job details have been retrieved showing the job spec with the benchmark and model configuration
- The job appears in the benchmark jobs list
