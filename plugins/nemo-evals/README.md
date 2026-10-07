<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix Evals Plugin

The Evals plugin connects the NeMo Helix Evals SDK to NeMo Helix. It
provides:

- **CLI** `nemo evals` commands for plugin status, job schema inspection, and
  durable job submissions.
- **Service** routes for evaluator job management: `plugins/nemo-evals/src/nemo_evals/service.py`.
- **SDK resource** built with `Evaluator.from_client(client)` for status checks, job
  submission, status polling, result retrieval, and artifact download.
- **Evaluator job** support for inline SDK metric specs, inline rows, and
  Fileset-backed datasets.
  - Dataset-driven `evals.evaluate` jobs.
  - Task-driven `evals.agent-evaluate` jobs.
- **Evaluator skill** published through the plugin entry point for
  evaluator-specific guidance and troubleshooting.

## Registered plugin interfaces

| Surface | Entry point | Behavior |
| --- | --- | --- |
| CLI | `nemo.cli:evals` | Plugin status, metric discovery, job schema inspection, and durable submissions |
| Service | `nemo.services:evals` | Health, job, stored-resource, and result routes |
| SDK | `nemo.sdk:evals` | `Evaluator.from_client(client)` execution, job lifecycle, stored resources, and result indexes |
| Dataset job | `nemo.jobs:evals.evaluate` | Scores inline or Fileset-backed datasets |
| Agent job | `nemo.jobs:evals.agent-evaluate` | Runs or rescores task-driven agent trials |
| Skill | `nemo.skills:evals` | Publishes the evaluator agent skill |

## Developer setup

This plugin is a `uv` workspace member. From the repository root:

```bash
# The `make bootstrap` target creates the Python environment, syncs Python dependencies, builds Studio assets, and installs local plugins.
make bootstrap
source .venv/bin/activate
```

Verify the installation:

```bash
nemo --help
```

Check the plugin status:

```bash
uv run nemo evals info
```

Follow the repository `SETUP.md` for detailed setup instructions and starting local NeMo Helix services.

## Dataset-Driven vs. Task-Driven evaluation
Review the [Evals documentation](https://docs.nvidia.com/nemo-helix/documentation/evaluate-models#two-shapes-of-evaluation) for a detailed explanation of the difference between dataset-driven and task-driven evaluation.

## Dataset-Driven evaluation

### Dataset evaluation CLI commands

Inspect the current schema:

```bash
uv run nemo evals evaluate explain
```

Submit the checked offline example as a durable job:

```bash
uv run nemo evals evaluate \
  --spec-file skills/nemo-evals-plugin/assets/specs/exact_match_metric.json
```

The submit response includes a generated job name, for example `nemo-evals-zlhn1ecd`. Wait for the job to complete, then list and download its results:

```bash
nemo jobs get <job-name>
nemo jobs results list <job-name>
nemo jobs results download aggregate-scores --job <job-name> --output-file aggregate-scores.json
nemo jobs results download row-scores --job <job-name> --output-file row-scores.jsonl
```

See also the checked LLM-judge spec example in `skills/nemo-evals-plugin/assets/specs/llm_as_judge.json`.

### Platform SDK Execution

Use the mounted SDK resource to submit durable evaluation jobs:

```python
from nhx_evals_sdk import ExactMatchMetric, RunConfig
from nemo_helix_plugin.client.client import NemoClient
from nemo_evals.sdk import Evaluator

client = NemoClient(base_url="http://localhost:8080", workspace="default")
evaluator = Evaluator.from_client(client)
metric = ExactMatchMetric(
    reference="{{item.expected}}",
    candidate="{{item.output}}",
)
dataset = [
    {"expected": "Paris", "output": "Paris"},
    {"expected": "Paris", "output": "London"},
]

job = evaluator.submit(
    metric=metric,
    dataset=dataset,
    config=RunConfig(parallelism=2),
)

job.wait_until_done()
remote_result = job.get_result()
artifact_dir = job.download_artifacts("evaluation-artifacts")
```

`submit` returns an `EvaluatorJobResource`. Always call
`wait_until_done()` before retrieving result artifacts.

## Task-Driven Agent evaluation

### Agent evaluation CLI commands

#### Durable job

Inspect the task-driven job schema:

```bash
uv run nemo evals agent-evaluate explain
```

The checked spec gives Fabric one task and scores the runner's final response
with exact match. Copy it, replace `target.model` in the copy with a real
provider/model identifier, then submit the copy as a durable platform job:

```bash
cp skills/nemo-evals-plugin/assets/specs/fabric_agent_eval.json \
  fabric_agent_eval.local.json
# Edit target.model in fabric_agent_eval.local.json before submitting.
uv run nemo evals agent-evaluate \
  --spec-file fabric_agent_eval.local.json
```

Ensure the job environment includes the Fabric Codex adapter, Codex CLI, and
its provider credentials. Set
`capture_trajectory` to `true` only when NeMo Relay is also available.
For repository setup, follow
[Prepare Fabric in a repository checkout](../../skills/nemo-evals-plugin/SKILL.md#prepare-fabric-in-a-repository-checkout).

### SDK Execution
Plugin SDK execution is not supported for task-driven evaluation. Use the standalone Python SDK instead, which is available for local execution.

#### Standalone SDK

For an in-process agent callable, pass a direct `AgentTaskRunner` to the
standalone SDK:

```python
from nhx_evals_sdk import ExactMatchMetric
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.runtimes.callable_runtime import CallableAgentTaskRunner
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask


async def answer(task: AgentEvalTask) -> str:
    return "Paris"


task = AgentEvalTask(
    id="capital-france",
    intent="Name the capital of France.",
    inputs={"instruction": "What is the capital of France?"},
    reference={"expected": "Paris"},
    metrics=[
        ExactMatchMetric(
            reference="{{reference.expected}}",
            candidate="{{sample.output_text}}",
        )
    ],
)
result = AgentEvaluator().run_sync(
    tasks=[task],
    target=CallableAgentTaskRunner(answer),
)
print(result.summary)
```

See the [agent-evaluation reference](../../skills/nemo-evals-plugin/references/agent-evaluation.md)
for tasksets, other durable targets, and precomputed trials.

## Stored resources

The SDK namespace includes:

- `evaluator.metrics`
- `evaluator.tasks`
- `evaluator.tasksets`
- `evaluator.eval_results`
- `evaluator.agent_eval_results`

Metrics, tasks, and tasksets support create, retrieve, list, and delete. Result
resources support retrieve, list, and delete.

## Authentication

- Local model-backed evaluation resolves `api_key_secret` as a local
  environment-variable name, such as `NVIDIA_API_KEY`..
- Durable platformjobs resolve it as a NeMo Helix secret in the target workspace.

Never place a credential value in a spec or log.

## References

- [Evals documentation](https://docs.nvidia.com/nemo-helix/documentation/evaluate-models)
- [Canonical evaluator skill](../../skills/nemo-evals-plugin/SKILL.md)
- [Evaluator API auth](../../skills/nemo-evals-plugin/references/api-auth.md)
- [Troubleshooting](../../skills/nemo-evals-plugin/references/troubleshooting.md)
