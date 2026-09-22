<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Execute GPU Jobs Through NeMo Helix Jobs Pipeline

This task tests your ability to create and run GPU jobs through the NeMo Helix jobs system. The jobs controller dispatches real Docker containers with GPU access.

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

- The NeMo Helix API server is running with the jobs controller enabled
- The Docker backend is configured for GPU job execution (provider `gpu`, profile `default`)
- The Docker socket is mounted for real container execution
- A workspace `gpu-job-workspace` has been pre-created
- The `nvidia/cuda:12.8.0-base-ubuntu22.04` image has been pre-pulled

## Task

### Part 1: Create a GPU verification job

1. Create a GPU job named `gpu-verify-job` in workspace `gpu-job-workspace` that runs `nvidia-smi` in the `nvidia/cuda:12.8.0-base-ubuntu22.04` container to verify GPU access.

2. Poll for completion using `nhx jobs get-status gpu-verify-job --workspace gpu-job-workspace` until it reaches a terminal status (`completed` or `error`).

3. If completed, retrieve the job logs with `nhx jobs get-logs gpu-verify-job --workspace gpu-job-workspace` to confirm nvidia-smi output.

### Part 2: Create a GPU compute job

1. Create a GPU job named `gpu-compute-job` in workspace `gpu-job-workspace` that runs a CUDA computation. Use the `nvidia/cuda:12.8.0-base-ubuntu22.04` image with command:
   ```
   ["sh", "-c", "nvidia-smi && echo GPU_COMPUTE_START && python3 -c \"import ctypes; libcuda = ctypes.CDLL('libcuda.so'); print('CUDA driver loaded'); print('GPU_COMPUTE_DONE')\" 2>/dev/null || echo GPU_COMPUTE_DONE_BASIC"]
   ```

2. Poll for completion and verify it reaches `completed` status.

### Part 3: Create a failing GPU job and diagnose

1. Create a GPU job named `gpu-fail-job` in workspace `gpu-job-workspace` that intentionally fails. Use `nvidia/cuda:12.8.0-base-ubuntu22.04` with command `["sh", "-c", "echo GPU_FAIL_START && exit 1"]`.

2. Poll for the job status until it reaches `error` status.

3. Investigate the failure by checking status details and logs.

## Job Creation Reference

Use `--input-data` to pass the full request body:

```bash
nhx jobs create --workspace gpu-job-workspace --input-data '{
  "name": "job-name",
  "source": "agent-eval",
  "spec": {},
  "platform_spec": {
    "steps": [
      {
        "name": "step-name",
        "executor": {
          "provider": "gpu",
          "profile": "default",
          "container": {
            "image": "nvidia/cuda:12.8.0-base-ubuntu22.04",
            "command": ["nvidia-smi"]
          },
          "resources": {
            "num_gpus": 1
          }
        }
      }
    ]
  }
}'
```

**Important:** For `get-status`, `get`, and `get-logs`, the job name is a **positional argument**:

```bash
nhx jobs get-status <name> --workspace <ws>
nhx jobs get <name> --workspace <ws>
nhx jobs get-logs <name> --workspace <ws>
```

## Success Criteria

The task is complete when:
- `gpu-verify-job` was created and reached `completed` status (nvidia-smi ran on GPU)
- `gpu-compute-job` was created and reached `completed` status
- `gpu-fail-job` was created, reached `error` status, and the agent investigated the failure
