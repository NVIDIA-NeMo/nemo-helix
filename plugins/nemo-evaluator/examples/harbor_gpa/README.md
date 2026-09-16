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
  this metric: Harbor here, the Fabric runner, a deployed agent with `trajectory_path`. See
  [Trace limitations](#trace-limitations-of-the-harbor-fabric-route) for what Harbor actually
  hands you today.
- **Outputs are optional.** `required=False` means a missing output is "could not measure". A
  trial with no trajectory shows up as `unmeasured` in the aggregates instead of dragging the mean
  to zero. A judge that returns garbage becomes a diagnostic on that score, for the same reason.
- **Nine metrics, one class.** `RUBRICS` holds what a 3 and a 0 look like per dimension. The judge
  always sees the whole trial (instruction, grader's goal, tools, stated plan, trace, final answer)
  and the rubric tells it what to look at.

`attach_gpa` then gives each discovered Harbor task its goal from `gpa.json`, appends the nine
metrics after the verifier's `HarborRewardMetric`, and adds a `gpa` view: the mean of the nine
`score` outputs, reported per trial and aggregated as `view.gpa`.

## Trace limitations of the Harbor + Fabric route

Read this before relying on the trajectory scores from this example.

- **Relay's ATIF for deepagents is one step.** With `fabric_telemetry="relay"`, Relay records
  every LLM turn and tool call in its ATOF event stream, but its ATIF projection for the LangGraph
  integration collapses the whole run into a single `LangGraph` step (nemo-relay 0.7). That is
  the trace the Harbor runner promotes to `trace:atif`, and on its own it is useless to a
  trajectory judge.
- **This example works around it.** When the ATIF trace has one step, `read_trajectory` rebuilds
  the steps from the message list in Fabric's `fabric-result-*.json`, which the Harbor runner also
  exposes as evidence. You get user, assistant, tool-call, and tool-result steps, which is enough
  for these judges. You do not get per-step token usage, timing, or tool definitions.
- **No OTLP from this route.** The Fabric bridge inside Harbor configures Relay for ATIF and ATOF
  only. Collecting OTLP would mean running a receiver inside the task container and pointing
  Relay at it, which is more plumbing than an example should carry.

**If you want richer traces, run the Fabric runner directly.** `FabricAgentRuntime` and
`FabricContainerRuntime` run the same deepagents harness without Harbor, start an OTLP receiver
for Relay, and register both `trace:atif` and `trace:otlp` on every trial. The OTLP spans carry
`gen_ai.*` attributes for each LLM call and tool call, which is the input TruLens's span-aware
`Trace` selectors were designed for. Trade: you lose Harbor's per-task Docker image and verifier
reward, so tasks are plain `AgentEvalTask` rows and outcome checks are SDK metrics over the final
workspace. See the
[Fabric runner](../../../../docs/evaluator/agent-eval/fabric-runner.mdx) docs.

Pick by what you are scoring. For verifier-backed Harbor suites where the GPA scores are a second
opinion, this route is fine. For trajectory-first evaluation, or to reuse TruLens's own trace
selectors rather than a rendered string, use the Fabric runner and read the OTLP trace.

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
Aggregates:
  answer_correctness.score              0.556  measured=3 unmeasured=0
  answer_relevance.score                0.667  measured=3 unmeasured=0
  execution_efficiency.score            0.889  measured=3 unmeasured=0
  groundedness.score                    0.667  measured=3 unmeasured=0
  harbor_reward.reward                  0.667  measured=3 unmeasured=0
  logical_consistency.score             0.889  measured=3 unmeasured=0
  plan_adherence.score                  0.500  measured=2 unmeasured=1
  plan_quality.score                    0.667  measured=3 unmeasured=0
  tool_calling.score                    1.000  measured=2 unmeasured=1
  tool_selection.score                  0.556  measured=3 unmeasured=0
  view.gpa                              1.000  measured=1 unmeasured=2

Per trial:
  gpa/count-words (completed)
    harbor_reward: {'reward': 0.0}
    answer_correctness: 0.67  The agent correctly counted words in each file (a.txt: 4, b.txt: 5, c.txt: 1) ... However, the output file was written to /notes/word_counts.txt instead of the required location ...
    groundedness: 0.00  The final answer asserts that the files in the notes directory were not modified, but the agent observed writing a new file /notes/word_counts.txt (step 7 and 9) ...
    plan_adherence: 0.00  The agent wrote the word counts to /notes/word_counts.txt (step 7) instead of the working directory ... no explanation for the deviation is provided.
    tool_calling: UNMEASURED  Error code: 503 - Service temporarily overloaded
    ...
  gpa/fix-bug (completed)
    harbor_reward: {'reward': 1.0}
    answer_correctness: 1.00  The agent identified the bug in stats.py where mean divided by len(values)-1, corrected it to divide by len(values), and verified that check.py now passes ...
    execution_efficiency: 1.00  The agent performed each necessary action exactly once: listed directory contents, read stats.py and check.py, identified the bug ... edited stats.py ... ran a verification task ...
    ...
```

That is a real run (Lightning agent, Super judge, September 2026), trimmed. Two things it shows that a
verifier alone would not: count-words *failed* its verifier because the agent wrote the file inside
`notes/`, and the judges say so in words while also catching the final answer's false claim that
`notes/` was untouched. And two judge calls hit a `503` from build.nvidia.com; those scores are
unmeasured with the error attached, which is why `view.gpa` is measured on only one trial here. A
view needs every one of its signals, so one unmeasured metric leaves the whole view unmeasured for
that trial.

Flags: `--model` (the agent's model, Lightning by default), `--judge-model` (Super by default: it
answers a judge prompt in about two seconds on build.nvidia.com, where Lightning reasons for 15 to
50 seconds), `--jobs-dir`, `--work-dir`. Judge calls run one at a time; build.nvidia.com allows
little concurrency and retries only slow things down.

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
