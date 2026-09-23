# Stored Resources

Read this file when definitions or results must be reusable and queryable
through `client.evaluator`.

## Resource map

| Resource | Create | Retrieve | List | Delete | Update |
| --- | --- | --- | --- | --- | --- |
| `metrics` | yes | yes | yes | yes | no |
| `tasks` | yes | yes | yes | yes | no |
| `tasksets` | yes | yes | yes | yes | no |
| `eval_results` | no | yes | yes | yes | no |
| `agent_eval_results` | no | yes | yes | yes | no |

Metrics, tasks, and tasksets are immutable. Delete and recreate them, or use a
new versioned name.

## Store a metric, task, and taskset

```python
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    MetricRef,
    TaskInput,
    TaskInputs,
    TaskRef,
    TasksetInput,
)
from nemo_evaluator_sdk import StringCheckMetric
from nemo_helix import NeMoHelix

client = NeMoHelix(base_url="<platform-url>", workspace="<workspace>")

client.evaluator.metrics.create(
    "answer-exact",
    metric=StringCheckMetric(
        operation="equals",
        left_template="{{sample.output_text | trim}}",
        right_template="Paris",
    ),
)

client.evaluator.tasks.create(
    "capital-france",
    task=TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Name the capital of France.",
            inputs=TaskInputs(instruction="What is the capital of France?"),
            metrics=[MetricRef("answer-exact")],
        ),
    ),
)

client.evaluator.tasksets.create(
    "geography",
    taskset=TasksetInput(
        description="Geography smoke tasks.",
        tasks=[TaskRef("capital-france")],
    ),
)
```

For a task that needs held-out ground truth invisible to the agent, put it in `reference` and
use a metric that reads it. This works on a stored task, so it survives into taskset-driven runs:

```python
from nemo_evaluator.api.schemas import EvaluatorTaskDefinition, MetricRef, TaskInput, TaskInputs
from nemo_evaluator_sdk import ExactMatchMetric

client.evaluator.metrics.create(
    "answer-from-reference",
    metric=ExactMatchMetric(
        reference="{{reference.expected}}",
        candidate="{{sample.output_text}}",
    ),
)

client.evaluator.tasks.create(
    "capital-france-graded",
    task=TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent="Name the capital of France.",
            inputs=TaskInputs(instruction="What is the capital of France?"),
            reference={"expected": "Paris"},
            metrics=[MetricRef("answer-from-reference")],
        ),
    ),
)
```

`reference` is surfaced to metrics but never seeded into the agent's workspace or shown to the
agent, so a metric can grade against artifacts the agent cannot edit. It is held out from the
*agent*, not from the API — anyone who can read the task can read it. It is covered by the revision
digest, so changing ground truth publishes a new revision.

Stored tasks keep metric references. Inline task metrics are normalized into
content-addressed derived metrics. The same `reference` field is available on an inline
`AgentEvalTaskInput` for one-off submissions.

## Retrieve, list, and delete

```python
metric = client.evaluator.metrics.retrieve("answer-exact")
metrics = client.evaluator.metrics.list(metric_type="string-check")
tasks = client.evaluator.tasks.list(page=1, page_size=100, sort="name")
tasksets = client.evaluator.tasksets.list(page=1, page_size=100)

client.evaluator.tasksets.delete("geography")
client.evaluator.tasks.delete("capital-france")
client.evaluator.metrics.delete("answer-exact")
```

Metric listing supports `metric_type` and `include_derived`. Task and taskset
listing support pagination and sorting. Every method accepts an optional
`workspace`; create operations also accept `project`.

## Query persisted results

Dataset-driven durable jobs create `eval_results`; agent-evaluation jobs create
`agent_eval_results`.

```python
row_eval = client.evaluator.eval_results.retrieve("<result-name>")
row_page = client.evaluator.eval_results.list(
    job_id="<job-name>",
    target_kind="model",
    target_name="<model-name>",
    dataset_ref="default/eval-data",
)

agent_eval = client.evaluator.agent_eval_results.retrieve("<result-name>")
agent_page = client.evaluator.agent_eval_results.list(
    job_id="<job-name>",
    target_kind="harbor",
    target_name="oracle",
)
```

Both result resources support pagination, sorting, workspace override, and
delete. Result indexing is best effort and separate from the authoritative job
artifacts. Retry a short-lived `404`; if the record remains absent, inspect the
job logs and use the artifact bundle. A persisted record is a queryable
summary/index; use the bundle for complete row scores, trials, evidence, and
reports.
