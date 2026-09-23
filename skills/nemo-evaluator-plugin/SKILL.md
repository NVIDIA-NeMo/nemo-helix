---
name: nemo-evaluator-plugin
description: Evaluate models, datasets, and agents with the NeMo Evaluator plugin. Use for metric selection, SDK checks, platform jobs, and result retrieval.
license: Apache-2.0
metadata:
  owner: nemo-helix
  author: nemo-helix
  maturity: active
  tags: [evaluation, metrics, agent-eval, nemo-helix]
allowed-tools: Bash, Read
---

# Evaluator Plugin

The Plugin CLI entrypoint is `uv run nemo evaluator`.

## Purpose

Use this skill to choose an evaluation interface and metric, validate a minimal
example, submit a NeMo Helix evaluation job, and retrieve its results.

## Inputs

Establish these inputs before building an evaluation:

- Evaluation interface: [dataset-driven, task-driven, or retrieval-driven evaluation](references/evaluation-shapes.md#difference-summary)
- Execution interface: standalone SDK evaluation or a durable NeMo Helix job.
- Pass/fail dataset examples: the smallest representative pass and failure cases.
- Metrics: the behaviors to score and the template fields they consume.
- Target: no target for offline scoring, or the model, agent, runner, or precomputed trials that produce outputs.

## Instructions

1. Clarify whether the input is [dataset-driven rows](references/evaluation-shapes.md#dataset-driven-evaluation),
   [task-driven agent work](references/evaluation-shapes.md#task-driven-evaluation), or a
   [BEIR retrieval corpus](references/evaluation-shapes.md#retrieval-driven-evaluation).
   If the user wants the full embed/rerank fine-tune recipe, hand off to `nemo-retrieval-recipes`
   and use this skill only to run and debug retrieve-eval.
2. Choose the simplest metric that measures the requested behavior. Prefer deterministic metrics when possible.
3. Build a tiny smoke case with one expected pass and one expected failure.
4. Validate metric behavior with the standalone SDK. Inspect row-level output, aggregate `count` and
   `nan_count`, and coverage before interpreting a mean.
5. Fix field mappings, prompts, parsers, or task definitions before scaling.
6. Submit the platform job only after the input and scoring shape works.

Read [Metric Selection](references/metric-selection.md) before choosing a
metric for a rubric, RAG workflow, or tool-calling evaluation.

## Choose the execution interface

| Need | Interface |
| --- | --- |
| Fast metric iteration without NeMo Helix | `nemo_evaluator_sdk.Evaluator` |
| Dataset-driven platform job | `client.evaluator.submit(...)` or `nemo evaluator evaluate` |
| Multiple inline/stored metric refs in one job | `nemo evaluator evaluate` with an `EvaluateInputSpec` |
| Task-driven platform job | `client.evaluator.submit(tasks=..., target=<runner>)` or `nemo evaluator agent-evaluate` |
| Retrieval-driven platform job | `nemo evaluator retrieve-eval` |
| Reusable platform definitions and result indexes | `client.evaluator.metrics`, `.tasks`, `.tasksets`, `.eval_results`, `.agent_eval_results` |

Default to `submit` for every plugin evaluation. The plugin's local execution
path is being retired: the `nemo evaluator ... run` CLI verb still exists but
should not be built on, even though `--help` still lists it. For fast metric
iteration without the platform, use the standalone `nemo_evaluator_sdk.Evaluator`
instead.

- Read [SDK Execution](references/execution.md) for datasets, targets,
configuration, field mapping, job lifecycle, and custom metric packaging.
- Read [Stored Resources](references/resources.md) for persisted definitions and
result queries.

## Limitations

- `api_key_secret` is an environment-variable name standalone but a NeMo
  Platform secret name on `submit`. See [API Auth](references/api-auth.md).
- HTTP 409 from a submission often means a referenced platform secret is
  missing, not a duplicate job. Read the response body.
- `intent` is grader metadata and is never shown to the agent; only `inputs`
  reaches it.
- Metric templates use `item.*` for dataset rows but `reference.*`, `sample.*`,
  and `inputs.*` in agent evaluation.
- Metric progress can reach 100 percent before the platform job is terminal.
  Always call `job.wait_until_done()` before retrieving results or downloading
  artifacts.

## CLI Interface

### Prerequisites

All commands in this file assume that the shell's working directory is the root
of the NVIDIA-NeMo/nemo-helix repository.

In a NeMo Helix repository checkout, run commands through the workspace:

```bash
# confirms plugin readiness and lists the registered evaluator jobs.
uv run nemo evaluator info
# lists available metric names; add a metric name to print its schema.
uv run nemo evaluator metric-types
# next three commands print the dataset-driven, task-driven, and retrieval job input and
# output schemas - can be very large, use with caution to avoid filling up the context window.
uv run nemo evaluator evaluate explain
uv run nemo evaluator agent-evaluate explain
uv run nemo evaluator retrieve-eval explain
```

When the skill and plugin are installed, use the installed `nemo` command
without assuming a repository root or manually activating `.venv`.

Resolve bundled assets relative to this skill directory. In this repository the
canonical path is `skills/nemo-evaluator-plugin`; an installed skill may live
under a different skills root.

## Bundled assets

| Path | Use |
| --- | --- |
| `assets/specs/exact_match_metric.json` | Two-row offline smoke spec; submit as-is |
| `assets/specs/llm_as_judge.json` | Online generation + judge; local-first (`NVIDIA_API_KEY`) |
| `assets/specs/fabric_agent_eval.json` | Task-driven Fabric runner spec |
| `assets/specs/gym_agent_eval.json` | Task-driven sandboxed Gym FileSet spec |
| `assets/examples/plugin_sdk_examples.py` | Copyable SDK snippets for each plugin surface |

## Available Scripts

| Script | Purpose | Arguments |
| --- | --- | --- |
| `scripts/generate_example_specs.py` | Generate or drift-check bundled specs | `--check`, `--write` |

In this repository, NeMo uses the displayed workspace command:

```bash
uv run --frozen python skills/nemo-evaluator-plugin/scripts/generate_example_specs.py --check
```

Do not assume a client-specific `run_script()` helper; use the displayed
`uv run` command.

## Examples

### Dataset-driven evaluation examples

- Follow [Validate standalone, then submit to the platform](references/execution.md#validate-standalone-then-submit-to-the-platform).
  for the two-row pass/fail smoke test and its CLI submission.
- Follow [Map noncanonical fields](references/execution.md#map-noncanonical-fields)
  when dataset columns need `field_mapping`.
- Follow [Getting job results](references/execution.md#getting-job-results)
  for submission, terminal waiting, result retrieval, and artifact download.
- Follow [Store a metric, task, and taskset](references/resources.md#store-a-metric-task-and-taskset)
  for reusable definitions, and [Query persisted results](references/resources.md#query-persisted-results)
  for result lookup.

### Task-driven agent evaluation examples

**Standalone SDK evaluation**

Use `AgentEvaluator().run(...)` for standalone task-driven SDK evaluation. Its
`target` can be a `Model`, a `GenericAgent`, or a direct `AgentTaskRunner`.

**Platform job evaluation**

Use the plugin `agent-evaluate` job for platform task evaluation. Its
target is a `ModelTarget`, `AgentTarget`, `FabricRunnerTarget`,
`HarborRunnerTarget`, or `GymRunnerTarget`; alternatively provide precomputed
`trials`. Provide exactly one of `target` or `trials`.

Submission accepts inline tasks or a stored `TasksetRef`. Stored tasksets are
resolved in the target workspace.

Read [Agent Evaluation](references/agent-evaluation.md) for inline tasks,
`TasksetRef`, concurrency, fail-fast behavior, sparse-output denominators,
result artifacts, and runner configuration, including Gym FileSet and
sandbox requirements.

### Prepare Fabric in a repository checkout

Fabric runner examples and tests need the optional harness adapters and the
matching Relay gateway:

```bash
uv sync --frozen --package nemo-evaluator-sdk --extra fabric --inexact
script/dev-install-fabric.sh
```

The install script downloads the checksum-verified `nemo-relay` binary that
matches the locked Python bindings. Add its reported directory to `PATH`, then
use `uv run --frozen --no-sync ...` for Fabric checks so uv does not remove the
optional adapters.

## Output Format

Report a completed platform evaluation in this form:

```text
Job: <job-name>
Status: <terminal-status>
Metrics: <metric-names>
Mean: <aggregate-mean>
Artifacts: <downloaded result or artifact location>
Errors: <error messages>
```

## Read specialized references

- Read [Evaluator API Auth](references/api-auth.md) before using a model,
  agent, remote metric, or durable submission.
- Read [LLM Judge](references/llm-judge.md) before writing judge scores,
  prompts, or parsers.

## Troubleshooting

Read [Evaluator troubleshooting](references/troubleshooting.md) when schema,
authentication, job, result, or runner behavior fails.

## Follow security best practices

Never print, serialize, or commit secret values. Store only environment-variable
names or platform secret references in specs and examples.
