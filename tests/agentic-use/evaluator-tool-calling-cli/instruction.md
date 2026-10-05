<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tool Calling Evaluation - BFCL (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Set up and run a BFCL-style tool calling evaluation using the `nhx` CLI. This evaluation tests whether a model's tool call predictions match expected ground truth.

1. **Create a workspace** named `tool-calling-eval-workspace`

2. **Prepare a BFCL-format dataset** - Create a JSONL file containing tool calling evaluation rows. Each row should have:
   - `messages`: an array with a user message describing a task
   - `tools`: an array of tool definitions in OpenAI function calling format
   - `expected_tool_calls`: the ground truth tool calls the model should produce (array of objects with `function.name` and `function.arguments`)
   - `response`: a simulated model response in OpenAI chat completion format containing `choices[0].message.tool_calls`

   Include at least 3 rows:
   - At least one row where the response **matches** the expected tool calls (correct function name and arguments)
   - At least one row where the response has a **wrong function name** (should score 0.0 on both metrics)
   - At least one row where the response has the **correct function name but wrong arguments** (should score 1.0 on function_name_accuracy but 0.0 on function_name_and_args_accuracy)

   Upload this dataset as a fileset named `tool-calling-dataset` in the `tool-calling-eval-workspace` workspace.

3. **Create a tool-calling metric** named `tool-calling-accuracy` in `tool-calling-eval-workspace` with type `tool-calling`. The metric's `reference` field should use a Jinja template to extract the ground truth tool calls from each dataset row (e.g., referencing the `expected_tool_calls` field).

4. **Run a synchronous evaluation** using `tool-calling-accuracy` against a small inline dataset (2-3 rows with the same format described above - including `expected_tool_calls` and `response` fields). Examine the results to verify:
   - `function_name_accuracy` scores are reported per row
   - `function_name_and_args_accuracy` scores are reported per row
   - Matching rows score 1.0 and mismatched rows score 0.0

5. **Create a metric evaluation job** referencing `tool-calling-accuracy` and the `tool-calling-dataset` fileset. The job may stay in "created" status - that is expected.

6. **Check the job status** to confirm it was created.

## Success Criteria

The task is complete when:
- A workspace `tool-calling-eval-workspace` exists
- A fileset `tool-calling-dataset` exists with tool calling evaluation data uploaded
- A `tool-calling` type metric named `tool-calling-accuracy` exists in the workspace
- A synchronous evaluation was run and returned `function_name_accuracy` and `function_name_and_args_accuracy` scores
- A metric evaluation job has been created referencing the metric and dataset
