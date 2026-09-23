# Agent Evaluation

Read this file for agentic task-driven evaluation, direct SDK runners, platform
`agent-evaluate` jobs, tasksets, precomputed trials, or Harbor and custom runners.

## Choose standalone SDK or platform job

Use `AgentEvaluator` for lightweight in-process evaluation that does not require a running nemo-helix:

```python
from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator

result = await AgentEvaluator().run(tasks=tasks, target=target)
print(result.trials)
print(result.summary)
```

The standalone target union is:

- `Model`
- `GenericAgent`
- Any object implementing `nemo_evaluator_sdk.agent_eval.trials.AgentTaskRunner` protocol

For a minimal direct runner:

```python
from nemo_evaluator_sdk.agent_eval.runtimes.callable_runtime import (
    CallableAgentTaskRunner,
)

async def answer(task):
    return task.inputs["instruction"]

runner = CallableAgentTaskRunner(answer)
result = await AgentEvaluator().run(tasks=tasks, target=runner)
```

Submit the plugin job when platform execution is required:

```bash
nemo evaluator agent-evaluate explain
nemo evaluator agent-evaluate \
  --spec-file skills/nemo-evaluator-plugin/assets/specs/fabric_agent_eval.json
```

A `GymAgentTaskRunner` already working under `AgentEvaluator()` can be submitted directly, without
describing its configuration a second time as a spec. Pass the live runner as `target` and a stored taskset as
`tasks`:

```python
from nemo_evaluator.api.schemas import TasksetRef
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner, GymRuntimeConfig

runner = GymAgentTaskRunner(
    config=GymRuntimeConfig(
        agent="simple_agent",
        agent_config="c.yaml",
        resources_server="mcqa",
    )
)
job = client.evaluator.submit(tasks=TasksetRef("my-suite"), target=runner)
job.wait_until_done()
```

`submit` has two shapes discriminated by what is supplied: `tasks` + `target` evaluates a stored
taskset, and `metric` + `dataset` evaluates rows. Supplying both, or passing a runner to the row
path, raises `TypeError` rather than running the wrong job. The row-only options — `config`,
`field_mapping`, `prompt_template`, `metric_bundle_packager` — are refused on the taskset path,
because a taskset run is configured by its runner instead.

Only a Gym runner can be converted into a target spec today. `submit` does that conversion with
`runner_to_target` (`nemo_evaluator.jobs.runner_targets`), which raises `UnsubmittableRunnerError`
for any other runner — for those, write the job input by hand with the matching runner target and
submit it, through the SDK or the CLI.

A Gym runner carrying state with no JSON form is refused for a different reason, and has a different
remedy. `hydra_params` is `dict[str, Any]`, so a callable or live object survives construction and
is rejected at submit. Writing the target by hand does not help: a hand-built `GymRunnerTarget`
fails the same `model_dump(mode="json")`, and a CLI `--spec` payload cannot encode the value either.
Replace it with something JSON-representable, or keep the run in-process with
`AgentEvaluator().run(...)`.

`submit` returns an `AgentEvaluatorJobResource`, which is read differently from the dataset-driven
job handle — see [Read results](#read-results).

## Build the job input

`AgentEvalInputSpec.tasks` accepts an inline task list or a stored `TasksetRef`.
Provide exactly one trial source:

- `target` to generate trials.
- `trials` to rescore precomputed trials.

`AgentEvalTaskInput` is the job-spec twin of the standalone SDK's
`AgentEvalTask`. The fields match; use `AgentEvalTaskInput` when building a
spec for `submit`.

```python
from nemo_evaluator.api.schemas import TaskInputs
from nemo_evaluator.jobs.agent_spec import (
    AgentEvalInputSpec,
    AgentEvalTaskInput,
    FabricRunnerTarget,
)

spec = AgentEvalInputSpec(
    tasks=[
        AgentEvalTaskInput(
            id="capital-france",
            intent="Name the capital of France.",
            inputs=TaskInputs(instruction="What is the capital of France?"),
            metrics=[metric_bundle],
        )
    ],
    target=FabricRunnerTarget(
        config={
            "metadata": {"name": "geography-smoke"},
            "harness": {"adapter_id": "nvidia.fabric.codex"},
        }
    ),
    max_concurrent_tasks=2,
    fail_fast=False,
    labels={"benchmark": "geography-smoke"},
)
```

`intent` is grader metadata and is never shown to the agent; only `inputs`
reaches it. Put the instruction the agent must act on in `inputs`.

Task metrics score against the task-driven template context
(`inputs.*`, `reference.*`, `task.*`, `trial.*`, `sample.output_text`), not the
dataset-driven `item.*` context.

Use `TasksetRef("default/geography")` with `submit` for persisted tasks. Stored
tasks carry the grader-only `reference` field too, so held-out per-task data
survives into taskset-driven runs; inline tasks are for one-off submissions.

Set `views` on a task to roll two or more of its metric outputs into one named,
reported score. See
[Score by Component](https://docs.nvidia.com/nemo-helix/documentation/evaluate-models/agent-eval/score-by-component).

## Choose a platform target

| Target | Use when |
| --- | --- |
| `ModelTarget` | Generate trials through an OpenAI-compatible model endpoint |
| `AgentTarget` | Generate trials through a generic HTTP or NeMo Agent Toolkit agent |
| `FabricRunnerTarget` | Run a configured NeMo [Fabric](https://github.com/nvidia/nemo-fabric) runner |
| `HarborRunnerTarget` | Run a Harbor task suite in Docker |
| `GymRunnerTarget` | Run a Gym environment and agent |

`ModelTarget` owns its `prompt_template` and online model params.
`AgentTarget` owns its agent request configuration. Runner targets are resolved
to an `AgentTaskRunner` inside the job runtime.

For [Fabric](https://github.com/nvidia/nemo-fabric), pass one complete `agent.yaml` as a JSON-shaped `config`; the
`harness.adapter_id` selects the harness:

```python
from nemo_evaluator.jobs.agent_spec import FabricRunnerTarget

target = FabricRunnerTarget(
    config={
        "metadata": {"name": "regression-suite"},
        "harness": {"adapter_id": "nvidia.fabric.codex"},
    },
    model="<provider>/<model>",
)
```

Do not use profile overlays. Fold the complete configuration into `config`.

### Configure Gym as a task runner

Use `discover_gym_tasks` to turn Gym JSONL rows into task definitions and attach
`GymRewardMetric` to score each rollout's reward. A standalone
`GymAgentTaskRunner` requires `agent`, `agent_config`, and `resources_server`.

For a durable job that uses components already installed in `nhx-gym-tasks`,
submit the validated live runner as shown above or build a `GymRunnerTarget`:

```python
from nemo_evaluator.jobs.agent_spec import GymRunnerTarget

target = GymRunnerTarget(
    agent="simple_agent",
    agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
    resources_server="mcqa",
    num_repeats=1,
    concurrency=4,
)
```

The caller chooses between a local SDK run and a durable platform job. A local
run executes Evaluator and the `gym` subprocesses on the caller's machine. For
a platform job, sandbox placement is an operator decision: sandbox-enabled
deployments run Gym in a separate `nhx-gym-host`; deployments without
OpenSandbox can run trusted, built-in Gym components together with Evaluator in
`nhx-gym-tasks`. The latter is the colocated compatibility path, not a separate
submission interface.

A custom environment supplies Gym component configuration, code, and
dependencies that are not built into the platform's Gym runtime image. Package
those files in a FileSet with `purpose=environment`, place
`nemo-environment.yaml` at its root, and set `target.environment` to the whole
FileSet reference. Evaluator accepts `native-v1` and `wheels-v1` packages.

FileSet-backed environments require sandboxed platform execution. Evaluator
compiles them into two ordered Jobs steps:

1. `stage-environment` downloads the FileSet onto job-scoped shared storage.
2. `agent-evaluate` provisions `nhx-gym-host` with the environment mounted
   read-only, collects and scores rollouts, then destroys the host.

```python
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_spec import GymRunnerTarget

target = GymRunnerTarget(
    environment=FilesetRef(root="default/my-gym-environment"),
    agent="simple_agent",
    agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
    resources_server="custom_greeting",
    env_secrets={"MODEL_API_KEY": "default/my-model-api-key"},
)
```

`agent_config` can be omitted when the FileSet declares the selected agent.
Set `agent_ref_name` when the package registers that agent under a different
instance name. Use `env_secrets`, not `env_vars`, for credentials; sandboxed
jobs reject credential-shaped plaintext environment variables.

From a live runner, `client.evaluator.submit(tasks=..., target=runner,
placement=GymPlacement(...))` builds this target without rebuilding it by hand.
`env_secrets` lives on `GymRuntimeConfig` (it means the same locally, resolved
from your environment); `environment` and `agent_ref_name` live on the
`GymPlacement`, because only a deployment can honor them.

`max_concurrent_tasks` limits tasks evaluated concurrently. Target-specific
settings such as inference parallelism or Harbor
`n_concurrent_trials` control concurrency inside trial generation.

## Use precomputed trials

Pass `trials=[...]` and omit `target` to rescore stored outputs and/or trajectories
without invoking the original model, agent, or runner. Keep stable `task_id`
values so trials match task definitions.

Individual trials are stored in the run bundle, not as queryable result entities.
Retrieve the run index, download its bundle, and hydrate `trials.jsonl`:

```python
from nemo_evaluator_sdk.agent_eval.persistence import read_trials

stored = client.evaluator.agent_eval_results.retrieve("<result-name>")
client.files.download(remote_path=stored.bundle_ref, local_path="previous-run")
trials = read_trials("previous-run")
```

CLI equivalent for downloading the bundle:

```bash
nemo jobs results download agent-eval-results \
  --job <job-name> --output-file agent-eval-results.tar.gz
mkdir -p previous-run
tar -xzf agent-eval-results.tar.gz -C previous-run --strip-components=1
```

Pass the hydrated `trials` with the same task definitions and omit `target`.

## Read results

A standalone run returns an `AgentEvalResult`:

- `result.summary` contains aggregate values per metric output plus coverage
  counts for scored, failed, and missing-output trials.
- `result.scores` contains one entry per task, trial, and metric, including
  metric outputs, status, and diagnostics.
- `result.trials` contains each agent output, its evidence, and its
  `completed`, `partial`, or `failed` status.
- `result.run_id` identifies the run; `result.metadata` contains its run
  provenance — labels, target identity, timings, and SDK version.

Call `result.persist()` to write the same information as a run bundle. It
defaults to the run's `work_dir` (`AgentEvalRunConfig.work_dir`); pass
`output_dir=` to choose another location:

| File | Contents |
| --- | --- |
| `summary.json` | Aggregate mean, minimum, maximum, standard deviation, counts, and coverage |
| `scores.jsonl` | Per-task, trial, and metric outputs, status, and diagnostics |
| `trials.jsonl` | Trial outputs, evidence, metadata, and status |
| `tasks.jsonl` | Tasks included in the run |
| `run.json` | Run ID and artifact manifest |
| `metadata.json` | Run provenance — labels, target identity, timings, and SDK version |
| `report.html` | Browsable dashboard when dashboard generation is enabled |

Use the in-memory result for programmatic follow-up and the bundle for
inspection, sharing, or rescoring. Platform jobs persist the bundle and create
a queryable record under `client.evaluator.agent_eval_results`.

A platform job hands back an `AgentEvaluatorJobResource`, which is not the
dataset-driven job handle: it offers `name`, `job`, `get_job_status()`,
`check_if_complete()`, and `wait_until_done()`, but no `get_result()` or
`download_artifacts()`. Read the scores through
`client.evaluator.agent_eval_results`.

Inspect failed and partial trials and score diagnostics before interpreting
aggregate values; a high mean with low coverage can hide missing or failed
work.

### Interpret sparse metric outputs

`MetricOutputSpec.required` is the producer contract:

- `required=True` is the default; the metric must emit the output.
- `required=False` permits the metric to omit an unmeasured output.
- Prefer omitting an unmeasured optional output so it stays out of the mean (it will be in `nan_count` for mean and `missing` for coverage ).
Emitting a filler (`0.0` or NaN) is allowed but not generally recommended: it changes aggregates. Emitting `None` is rejected.

Agent-eval averages finite measured values and exposes the denominator. For task A with two attempts,
`a1={reward:1.0, format_ok:1.0}` and `a2={reward:0.0}`:

| Consumer | Expected result |
| --- | --- |
| `harbor_reward.reward` | `mean=0.5`, `count=2`, `nan_count=0` |
| `harbor_reward.format_ok` | `mean=1.0`, `count=1`, `nan_count=1` |
| `format_ok` coverage | `total=2`, `scored=1`, `missing=1`, `failed=0` |
| View requiring `format_ok` | a2 is unmeasured; the view does not partially reduce or zero-fill |
| `format_ok.pass@1` | `mean=1.0`; measured `n=1` |
| `format_ok.pass@2` | unestimable: `count=0`, `nan_count=1`, `mean=None` |
| Persistence | `format_ok` stays absent for a2; no null, NaN label, or zero is synthesized |

- For ordinary aggregates, `count + nan_count` equals applicable opportunities.
- For coverage, `scored + missing + failed == total`.
- Pass@k's measured `n` includes failed trials as non-passes. A failed metric (the trial completed, but scoring raised an exception) or an omitted
  optional output is left out of `n` (unmeasured, not unsuccessful).
- If task A declares `format_ok` and task B does not, `format_ok`'s `count`, `nan_count`, and
  coverage are only over A's trials. B is omitted from that denominator, not counted as missing.

## Configure Harbor as a task runner

Harbor requires its Python package, Docker access, and a Harbor dataset. Task
discovery records the source dataset in each task's
`harbor_dataset_path` metadata; the durable runtime recovers the dataset from
that metadata.

**Standalone SDK:**

```python
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import (
    HarborAgentTaskRunner,
    HarborRuntimeConfig,
    discover_harbor_tasks,
)

tasks = discover_harbor_tasks("path/to/harbor-suite")
runner = HarborAgentTaskRunner(
    config=HarborRuntimeConfig(
        jobs_dir=Path("harbor-jobs"),
        agent_name="oracle",
        n_attempts=1,
        n_concurrent_trials=2,
        reward_key="reward",
    ),
)
result = await AgentEvaluator().run(tasks=tasks, target=runner)
```

**Platform SDK:**

```python
from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget

target = HarborRunnerTarget(
    agent_name="oracle",
    n_attempts=1,
    n_concurrent_trials=2,
    max_retries=0,
    artifacts=["/workspace/output"],
    trace_dir="/app/traces",
    reward_key="reward",
)
```

- `reward_key` selects the required primary reward by name (default `reward`). Mapping order and
  alphabetical order do not select it.
- The SDK scores only Harbor-valid `result.json` files (Harbor's `TrialResult`). A `null`,
  nonnumeric string, or object in the reward mapping fails that check, so the whole attempt is
  skipped and sibling rewards are not scored. Harbor writes `NaN` and infinity as `null`, which
  hits this gate.
- On a Harbor-valid trial, a finite primary is emitted unchanged. A missing or unusable primary
  (including Boolean) emits `0.0` with a diagnostic; the trial is still scored.
- Other keys from that task's Harbor-valid results become optional secondaries. Finite numbers are
  emitted. Missing or Boolean values are omitted with a diagnostic; usable siblings are kept.
- A secondary reward discovered for one task does not apply to another task.

Use `agent_import_path` for a custom Harbor agent and `agent_model_name` when
the agent requires a model. Pass the agent's constructor arguments as
`agent_kwargs` (a JSON mapping, Harbor's `--ak key=value`). Do not put secrets
in `agent_kwargs`: Harbor persists them unredacted across the job directory and
needs the real value to run. Credential-shaped plaintext is rejected at submit
time, but that check recognizes common key names and token formats, not every
secret. Put them in `env_secrets` (`{ENV_NAME: secret-ref}`) instead; the service resolves
the reference into the job environment and Harbor hands the agent a `${ENV_NAME}`
template. The module must be importable in the execution environment. Durable execution additionally requires an execution image and
runtime that provide Harbor and Docker access.
