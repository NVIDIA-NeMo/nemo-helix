<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-tool-calling-plugin

The `tool-calling` strategy for `nemo agents optimize`. It finds the tool calls an agent gets wrong, says why, and
tests two fixes before suggesting them:

1. **Baseline.** By default, scores the tool calls the agent already made, as traced in Intake: the
   `execute_tool <tool>` spans of its newest `max_sessions` sessions in the trace window, one session per run. With a
   prompt dataset, it instead runs the agent `runs_per_prompt` times per prompt, in-process through Fabric, and
   scores the calls from the Relay stream. A call is bad when the tool rejects it, when its arguments fail the tool's
   published input schema, or when the result reads like an error (heuristic). Runs that retry a tool right after a
   bad call, or go over `call_budget` calls, count as retry loops.
2. **Diagnose.** Lints the published tool schemas (open objects, long optional tails, required keys declared late,
   untyped items, enums in prose, `const`, discriminators without variants, and schemas too loose to catch the
   tool's own rejections), then asks `analyzer_model` for a system-prompt fix and a tool-schema patch.
3. **Verify.** Re-runs the prompts with the prompt fix (`config-fix`), then also with the schema patch
   (`schema-fix`). Without a dataset, the prompts are the user messages that started the traced sessions (the first
   `user` message of each session's earliest LLM span), deduplicated and capped at `max_replay_prompts`. The patch
   is served by a schema-overlay MCP proxy that advertises the new schema and forwards every call to the unchanged
   tool, so the tool still validates.
4. **Report.** Writes the files in [CONTRACT.md](CONTRACT.md) to the job's results.

The stored agent and its tools are never modified.

## Spec

| Field | Default | Meaning |
| --- | --- | --- |
| `agent` | required | Agent to score (`name` or `workspace/name`). |
| `trace_window_hours` | `24` | Without a dataset: how far back from `until` to read traces. |
| `since` / `until` | window / now | Without a dataset: explicit trace window bounds. |
| `max_sessions` | `50` | Without a dataset: newest traced sessions to score. |
| `max_replay_prompts` | `5` | Without a dataset: distinct traced prompts to replay. |
| `optimize_config` + `optimize_config_fileset` | | Prompt dataset: a JSONL file of `{"id", "prompt"}` rows in a fileset. Re-runs the baseline. |
| `prompts` | | Inline dataset instead of the fileset, as a list of `{"id", "prompt"}`. |
| `runs_per_prompt` | `3` | Runs per prompt and variant. |
| `max_prompts` | all | With a dataset: use only the first N prompts. |
| `max_turns` | `30` | `runtime.max_turns` for every variant. A run that hits it counts in `runs_with_errors`. |
| `call_budget` | `3` | Tool calls a run may make before it counts as a retry loop. |
| `timeout_seconds` | `600` | Limit for one agent run. |
| `concurrency` | `1` | Agent runs in flight at once. |
| `analyzer_model` | `default/nemotron3-5-super` | Inference Gateway model that proposes the fixes. |
| `verify` | `["config-fix", "schema-fix"]` | Variants to re-run. A suggestion that is not re-run is still reported. |

## Example: the `data-designer-toolcall` agent

Chat with the agent (with telemetry on), then run the strategy. In Studio, pick **Tool-call repair** on the agent's
Optimize tab and launch without a bundle. The CLI equivalent:

```bash
nemo agents optimize run-strategy --strategy tool-calling --agent default/data-designer-toolcall
```

To re-run the baseline from a fixed prompt set instead:

```bash
nemo files filesets create dd-toolcall-prompts
nemo files upload agents/data-designer-toolcall/eval/prompts.jsonl dd-toolcall-prompts
nemo agents optimize run-strategy \
  --strategy tool-calling \
  --agent default/data-designer-toolcall \
  --optimize-config prompts.jsonl \
  --optimize-config-fileset default/dd-toolcall-prompts \
  --spec '{"runs_per_prompt": 3, "max_turns": 30}'
```

## Results

`tool-calling-result.json`, `report.md`, `candidate-agent.yaml` and `candidate-agent.diff` (the prompt fix), and
`schema-patch/<server>/<tool>.json` and `schema-patch.diff` (the schema patch for the tool's owner). Shapes are in
[CONTRACT.md](CONTRACT.md).

## Limits

- Only stdio MCP servers can be put behind the schema-overlay proxy.
- A traced baseline reports `runs_with_errors: 0`: tool spans don't record whether the run around them failed.
- Agents must use the `nemo-agents-spec-v1` config format.
