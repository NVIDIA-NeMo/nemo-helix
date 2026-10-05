<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Execute and Troubleshoot CPU Jobs

This task tests your ability to create jobs, monitor their lifecycle, and diagnose issues. You will create multiple jobs including one that intentionally fails, and demonstrate understanding of the jobs system by investigating the failure.

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

- The NeMo Helix API server is running with the jobs controller enabled
- The Docker backend is configured for CPU job execution (provider `cpu`, profile `default`)
- The Docker socket is mounted for real container execution
- A workspace `job-test-workspace` has been pre-created
- The `busybox:latest` and `alpine:latest` images have been pre-pulled

## Task

### Part 1: Create a successful job

1. Create a CPU job named `success-job` in workspace `job-test-workspace` that runs `echo "STEP_ONE_COMPLETE"` in a `busybox:latest` container.

2. Poll for completion using `nemo jobs get-status success-job --workspace job-test-workspace` and verify it reaches `completed` status.

### Part 2: Create a failing job and diagnose it

1. Create a CPU job named `fail-job` in workspace `job-test-workspace` that runs a command that will fail. Use `busybox:latest` with command `["sh", "-c", "echo STARTING_FAIL_JOB && exit 1"]`.

2. Poll for the job status. It should reach `error` status.

3. Investigate the failure:
   - Check the job status details for error information via `nemo jobs get-status fail-job --workspace job-test-workspace`
   - Try `nemo jobs get fail-job --workspace job-test-workspace` for full details
   - Describe what you found about the failure

### Part 3: Create a second successful job

1. Create another CPU job named `recovery-job` in workspace `job-test-workspace` that runs `["sh", "-c", "echo RECOVERY_COMPLETE && echo FINAL_STATUS=ok"]` in an `alpine:latest` container.

2. Poll for completion and verify it reaches `completed` status.

## Job Creation Reference

Use `--input-data` to pass the full request body:

```bash
nemo jobs create --workspace job-test-workspace --input-data '{
  "name": "job-name",
  "source": "agent-eval",
  "spec": {},
  "platform_spec": {
    "steps": [
      {
        "name": "step-name",
        "executor": {
          "provider": "cpu",
          "profile": "default",
          "container": {
            "image": "busybox:latest",
            "command": ["echo", "hello"]
          }
        }
      }
    ]
  }
}'
```

**Important:** For `get-status`, `get`, and `get-logs`, the job name is a **positional argument**:

```bash
nemo jobs get-status <name> --workspace <ws>
nemo jobs get <name> --workspace <ws>
```

## Success Criteria

The task is complete when:
- `success-job` was created and reached `completed` status
- `fail-job` was created, reached `error` status, and the agent investigated the failure
- `recovery-job` was created and reached `completed` status
- The agent demonstrated understanding of job lifecycle by handling both success and failure cases
