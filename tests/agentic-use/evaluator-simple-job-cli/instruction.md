<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Simple Custom Evaluation Job (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

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
