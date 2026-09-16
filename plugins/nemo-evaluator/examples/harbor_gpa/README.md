<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# harbor_gpa — score *how* an agent worked, not just whether it finished

A Harbor verifier tells you whether the agent reached the end state. It says nothing about how the
agent got there. This example keeps the verifier and adds nine **Agent GPA** scores (Goal, Plan,
Action, after [TruLens](https://www.trulens.org) and the
[Agent GPA paper](https://arxiv.org/abs/2510.08847)), each judged from the agent's trajectory
after the run, each with a written reason, and averaged into one `view.gpa`.

It is also a tour of the Evaluator pieces you need to write any trajectory metric of your own.

| File | What it shows |
| --- | --- |
| [`run_harbor_gpa.py`](run_harbor_gpa.py) | the script, top to bottom: tasks, the deepagents agent inside Harbor, run, report |
| [`gpa_metrics.py`](gpa_metrics.py) | the `Metric` protocol, reading the trajectory from evidence, an LLM judge, attaching metrics and a view to Harbor tasks |
| [`run_harbor_gpa_trulens.py`](run_harbor_gpa_trulens.py) | the same script with TruLens's own feedback functions as the judges, via [`trulens_metrics.py`](trulens_metrics.py) |
| [`gpa_dataset/`](gpa_dataset/) | three small Harbor tasks; each has a `gpa.json` stating the goal the grader checks |

## How a metric works

A metric is a plain object with three members. Here is the whole idea, from `gpa_metrics.py`:

```python
class GpaMetric:
    @property
    def type(self) -> str:                      # its name in the results, e.g. "plan_quality"
        return self.name

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("score", required=False),
                MetricOutputSpec.label("reason", required=False)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trajectory = await read_trajectory(input)          # from input.candidate.evidence
        if trajectory is None:
            return MetricResult(outputs=[])                 # unmeasured, not zero
        score, reason = await self.judge.score(...)         # one LLM call
        return MetricResult(outputs=[MetricOutput(name="score", value=score),
                                     MetricOutput(name="reason", value=reason)])
```

Three things worth noticing:

- **The trajectory comes from evidence.** `input.candidate.evidence.trace(format="atif")` gives
  the agent's steps, tool calls, and tool results. Any runner that records an ATIF trace feeds
  this metric: Harbor here, the Fabric runner, a deployed agent with `trajectory_path`.
- **Outputs are optional.** `required=False` means a missing output is "could not measure". A
  trial with no trajectory shows up as `unmeasured` in the aggregates instead of dragging the mean
  to zero. A judge that returns garbage becomes a diagnostic on that score, for the same reason.
- **Nine metrics, one class.** `RUBRICS` holds what a 3 and a 0 look like per dimension. The judge
  always sees the whole trial (instruction, grader's goal, tools, stated plan, trace, final answer)
  and the rubric tells it what to look at.

`attach_gpa` then gives each discovered Harbor task its goal from `gpa.json`, appends the nine
metrics after the verifier's `HarborRewardMetric`, and adds a `gpa` view: the mean of the nine
`score` outputs, reported per trial and aggregated as `view.gpa`.

## Running it

Prerequisites: Python 3.12 or newer with the Harbor extra
(`uv sync --frozen --package nemo-evaluator-sdk --extra harbor`), Docker running, and
`NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) exported. Both the agent's
model and the judge read it.

```bash
export NVIDIA_API_KEY=...
uv run plugins/nemo-evaluator/examples/harbor_gpa/run_harbor_gpa.py
```

Harbor builds a `python:3.12-slim` container per task, installs NeMo Fabric's deepagents harness
into it, and runs the task. `fabric_telemetry="relay"` makes the harness record its trajectory
into the trial, which is what the metrics read. The first trial takes a few minutes for the
install. Then the report:

```
Aggregates (mean over trials the metric could measure):
  harbor_reward.reward                  1.000  measured=3 unmeasured=0
  answer_correctness.score              ...
  ...
  view.gpa                              ...

Per trial:
  gpa/count-words (completed)
    harbor_reward: {'reward': 1.0}
    answer_correctness: 1.00  step 6 writes word_counts.txt with the three expected lines ...
    plan_quality: 0.67  the agent listed the directory before reading, but read b.txt twice ...
    ...
```

Flags: `--model` (the agent's model), `--judge-model`, `--jobs-dir`, `--work-dir`.

### With TruLens as the judge

TruLens's trace-level feedback functions accept the trace as a plain string and return a score in
0 to 1 with a reason, so wrapping one in the `Metric` protocol is a few lines. TruLens is **not** a
dependency of this repository; install it yourself, then run the second entrypoint:

```bash
uv pip install trulens-core trulens-feedback trulens-providers-openai
uv run plugins/nemo-evaluator/examples/harbor_gpa/run_harbor_gpa_trulens.py
```

Same nine metric names, same `score` and `reason` outputs, same `view.gpa`, so nothing downstream
cares which judge produced them. TruLens's OpenAI provider is pointed at build.nvidia.com. On a
Nemotron judge TruLens falls back from structured output to text parsing plus a reformat call, so
expect about two judge requests per score.

## Adapting it

- **Your own scorer.** Copy `GpaMetric`, replace the judge call with whatever scores your trace,
  return your outputs. Add its `score` to the view in `attach_gpa` and it joins `view.gpa`.
- **Your own Harbor agent.** If you drive deepagents from a hand-written Harbor `BaseAgent`
  instead of the Fabric bridge, write the ATIF trajectory to `agent/trajectory.json` in the trial
  directory, or to any `traces/*.atif.json` under it. The Harbor runner picks up either.
- **Publishing to Intake.** `publish_to_intake(result, ...)` writes each trial's trajectory and one
  row per metric output, reasons included, so Studio shows the trajectory next to its scores. The
  [`harbor_to_intake`](../harbor_to_intake/) example walks through that round trip.
- **Scoring inside the task instead.** Harbor mounts the agent's log directory into the container
  for the verifier phase, so a `tests/test.sh` can read `/logs/agent/trajectory.json`, run these
  same metric classes, and write each score as a named reward. The trade is that the task image
  must then install the platform SDK and the judge's key must reach the verifier. Scoring after
  the run, as here, keeps the task image lean and the metrics runner-agnostic.

## Related

- [`packages/nemo_evaluator_sdk/examples/harbor/`](../../../../packages/nemo_evaluator_sdk/examples/harbor/) —
  the Harbor runner on its own, including the deepagents-on-Nemotron bridge this builds on.
- [Writing Metrics](../../../../docs/evaluator/agent-eval/writing-metrics.mdx) and
  [Score by Component](../../../../docs/evaluator/agent-eval/score-by-component.mdx) — the metric
  protocol and views in full.
