<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SDK Execution

Read this file before guessing a metric schema, execution parameter,
online/offline setup, parser, template, or SDK result shape.

## SDK Reference Points

Assume the skill is usually installed with the plugin rather than running from
a NeMo Helix repo checkout. Prefer installed SDK inspection before using
repo-relative source paths:

- Import SDK classes and inspect their signatures, docstrings, and exported
  field names.
- Use package metadata, installed examples, and CLI help when available.
- Prefer the public import surface, such as `nemo_evaluator_sdk.Evaluator`,
  metric classes, `RunConfig`, `RunConfigOnlineModel`, `Model`, and parser
  classes, over private paths.
- If the user provides a repo checkout or you are already working inside one,
  use the source paths below as developer fallbacks before guessing a metric
  schema or execution parameter.

- Metric config schemas: `packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk/values/metrics.py`
- Metric type names: `packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk/enums.py`
- Runtime metric behavior: `packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk/metrics/`
- RAGAS runtime metrics: `packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk/metrics/ragas/metrics.py`
- Execution API: `packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk/execution/evaluator.py`
- Run config types: `packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk/values/params.py`
- Request config normalization: `packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk/execution/config.py`
- Public examples: `packages/nemo_evaluator_sdk/examples/examples.py`
- Focused tests: `packages/nemo_evaluator_sdk/tests/metrics/` and
  `packages/nemo_evaluator_sdk/tests/execution/`

Prefer the SDK class field names. For example, `StringCheckMetric` uses
`operation`, `left_template`, and `right_template`; do not invent `field` or
`expected_field`.

## Online Vs Offline Evaluation

Use offline evaluation when dataset rows already contain the output or response
to score. This is the default for reproducing a benchmark, validating known
responses, checking judge agreement against labels, or debugging metric
templates.

Use online evaluation when the evaluator should call a live model or agent
before scoring. In online mode, pass `target=Model(...)` or `target=Agent(...)`
and a `prompt_template`; metrics can then score the generated output through
the SDK's online result fields. Use this for generation-quality workflows,
live model comparisons, or platform-managed jobs that need fresh generations.

Keep these workflows separate. Do not use labels for existing baseline
responses as labels for newly generated outputs unless the benchmark protocol
explicitly defines that mapping.

## Good Loop Examples

Offline judge-quality loop: start with rows that already contain `input`,
`output`, and `expected`; choose `ExactMatchMetric`, `StringCheckMetric`, or
`LLMJudgeMetric` depending on the rubric; smoke-test one expected pass and one
expected fail; inspect `row_scores` before trusting aggregate scores.

Online generation-quality loop: start with rows that contain prompts and
references; pass `target=Model(...)` or `target=Agent(...)` plus a
`prompt_template`; score fresh generations with fixed deterministic metrics or
a fixed judge; inspect generated outputs, judge predictions, and aggregate
scores separately.

## SDK Execution Patterns

Use `Evaluator` directly for local, completed-result evaluation. `run` and
`run_sync` take a sequence of metrics — pass `metrics=[metric]` even for a
single one — and always return a `BenchmarkEvaluationResult`.

```python
from nemo_evaluator_sdk import Evaluator, RunConfig, StringCheckMetric


metric = StringCheckMetric(
    operation="equals",
    left_template="{{item.output | trim}}",
    right_template="{{item.expected | trim}}",
)

result = Evaluator().run_sync(
    metrics=[metric],
    dataset=[
        {"output": "hello", "expected": "hello"},
        {"output": "foo", "expected": "bar"},
    ],
    config=RunConfig(parallelism=4),
)

result.print_summary()
print(result.aggregate_scores)
print(result.row_scores)
```

Run multiple metrics together when evaluating multiple dimensions of the same
dataset:

```python
from nemo_evaluator_sdk import Evaluator, ExactMatchMetric, StringCheckMetric


metrics = [
    ExactMatchMetric(reference="{{item.expected}}", candidate="{{item.output}}"),
    StringCheckMetric(
        operation="contains",
        left_template="{{item.output}}",
        right_template="{{item.required_phrase}}",
    ),
]

result = Evaluator().run_sync(metrics=metrics, dataset=rows)
result.print_summary()
print(result.per_metric)
```

Use online generation only when the evaluator should call a model or agent
before scoring. Pass `target=Model(...)` or `target=Agent(...)` and a
`prompt_template`; otherwise keep evaluation offline and put outputs in the
dataset rows.

```python
from nemo_evaluator_sdk import (
    Evaluator,
    ExactMatchMetric,
    InferenceParams,
    Model,
    RunConfigOnlineModel,
)


metric = ExactMatchMetric(reference="{{item.expected}}")
target = Model(
    url="https://provider.example/v1",
    name="<model-id>",
    api_key_secret="<secret-or-env-name>",
)

result = Evaluator().run_sync(
    metrics=[metric],
    target=target,
    dataset=[{"prompt": "What is 2+2?", "expected": "4"}],
    prompt_template={"messages": [{"role": "user", "content": "{{item.prompt}}"}]},
    config=RunConfigOnlineModel(
        parallelism=2,
        inference=InferenceParams(max_tokens=128),
    ),
)
```

In `Model(...)`, `format` is deprecated and ignored; do not set it. Structured
output support is detected from the endpoint during preflight rather than
inferred from a label. `api_key_secret` is a secret
reference name, not a literal secret value. For local SDK runs, ensure the
matching local environment variable exists; for remote platform jobs, create a
platform secret with the same name in the job workspace.

## LLM Judge Pattern

Use `LLMJudgeMetric` when deterministic metrics cannot capture the behavior.
Define score names, descriptions, ranges or rubrics, parser behavior, judge
model, and prompt template. Score names must use lowercase letters, numbers,
and underscores.

```python
from nemo_evaluator_sdk import Evaluator, JSONScoreParser, LLMJudgeMetric, Model, RangeScore


judge_model = Model(
    url="https://provider.example/v1",
    name="<judge-model-id>",
    api_key_secret="<secret-or-env-name>",
)

metric = LLMJudgeMetric(
    model=judge_model,
    scores=[
        RangeScore(
            name="quality",
            description="Overall response quality from 1 to 5",
            minimum=1,
            maximum=5,
            parser=JSONScoreParser(json_path="quality"),
        )
    ],
    prompt_template={
        "messages": [
            {
                "role": "system",
                "content": 'Rate response quality from 1 to 5. Return JSON: {"quality": <score>}',
            },
            {
                "role": "user",
                "content": "Input: {{item.input}}\nResponse: {{item.output}}",
            },
        ]
    },
)

result = Evaluator().run_sync(
    metrics=[metric],
    dataset=[
        {"input": "Explain photosynthesis.", "output": "Plants use sunlight to make sugars."},
        {"input": "Explain photosynthesis.", "output": "I cannot help."},
    ],
)
```

For zero-config judge metrics, provide `model` and rubric/range `scores`, then
omit `prompt_template` and explicit parsers. Run a tiny local evaluation before
using a full dataset.

## Tool Calling Pattern

Use `ToolCallingMetric` when rows contain OpenAI-style tool responses and
ground truth tool calls. It emits:

- `function_name_accuracy`
- `function_name_and_args_accuracy`

```python
from nemo_evaluator_sdk import Evaluator, ToolCallingMetric


metric = ToolCallingMetric(reference="{{item.expected_tool_calls}}")
result = Evaluator().run_sync(metrics=[metric], dataset=rows)
```

Each row should include a `response` object shaped like an OpenAI chat
completion with `choices[0].message.tool_calls`. Ground truth should be a list
of OpenAI-style tool call objects with `function.name` and
`function.arguments`. The runtime is case sensitive, order insensitive for
parallel calls, and expects function arguments to be valid JSON strings in the
predicted response.

## Dataset And Template Checks

- Templates read dataset columns through `item`, for example
  `{{item.expected}}`.
- Online evaluations can read generated output through `sample.output_text`.
- Offline evaluations need response/output fields already present in each row.
- Keep field names identical between dataset rows and templates.
- Include at least one expected pass and one expected fail in smoke data.
- If a row fails to render a template, fix the dataset shape before changing
  the metric.
