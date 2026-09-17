<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Experimentalist

The Experimentalist improves an agent (source code). Give it a baseline
agent and a pair of Harbor-compatible train and validation datasets, and it
runs an evolutionary loop: LLM agents read the failures, propose code changes,
implement them as candidate agents, and score each one. Only candidates that
improve the held-out validation split survive. The winner can be published as a
draft PR.

Use it when the fix belongs in the agent's harness — its workflow, tool use,
state handling, or prompts-in-code. To tune a *deployed* agent's model routing
or cost without changing its implementation, use `nemo agents optimize`
instead.

## How it works

This is an agent-system optimizer, not a prompt tuner. It reads your agent as a
graph of deterministic code, LLM calls, and nested subagents, and it can change
any of those optimization levers:

| Levers | Example change |
| --- | --- |
| LLM prompts | Rewrite a prompt |
| Deterministic code | Move repeatable parsing or validation out of the LLM and into code |
| Tools | Add, remove, or constrain a tool |
| Skills and context | Add or correct domain guidance |
| Structure | Split a method, add a subagent, change routing |
| Model and config | Swap the model, or raise a limit the agent keeps hitting |

Which one it reaches for is decided by the diagnosed root cause. Every candidate
makes **exactly one** targeted change, so a score movement can be attributed to
it.

The loop builds a baseline (`agent-0`) and scores it on validation, which is
what the first selection has to work from. Then it repeats each round:

1. **Select** survivors from the previous round's validation scores.
2. **Evaluate on train** — the diagnostic split.
3. **Analyze** the failures, comparing each trace against a generated reference
   solution path and separating mechanical breakage from bad reasoning.
4. **Propose** one-change improvements, alternating between exploring new
   directions and refining the current best.
5. **Implement** each in its own copy of the agent: a bounded sub-problem
   optimization loop refines the change until the behavior it targets actually
   moves, then an integration check gates the result; candidates that fail the
   integration check are dropped.
6. **Evaluate on validation** — the selection split.

The run stops on the round budget (`max_rounds`) or when a convergence check
decides the rounds have stopped producing improvement.

**The two splits do different jobs, and validation is hidden while candidates
are generated.** Train diagnoses; validation selects. At run start the
validation data is moved to a held-out directory and the bash tool used by the
coding and analysis agents blocks reads of that path, restoring it only while
validation scoring runs — so candidates cannot be tuned against the data that
picks them. A third held-out test split, if you keep one, is yours to run
manually; the profile and CLI take only train and validation.

Candidates are ranked on two signals. The **outcome reward** comes from the
Harbor verifier: metrics normalized to `[0.0, 1.0]`, averaged across trials,
with failed trials counted as zero. The optional **trajectory score** rates the
path the agent took against a goal tree — a weighted rubric of the capabilities
a task needs. Outcome is the primary signal; trajectory breaks ties and keeps
candidates whose process is better than their score, which matters because
long-horizon agent failures are sparse and a directionally right change can
still fail the final answer. Survivors are kept as a Pareto front rather than a
single leader, so complementary strengths stay alive across rounds.

## Where it fits

- [NeMo Insights](../nemo-insights/README.md) produces the Insight — a
  failure pattern inferred from traces — that the Experimentalist optimizes
  against. Run `nemo agents analyst run` first. The Experimentalist does not
  analyze traces or host an Insight API.
- [Eval Author](src/nemo_experimentalist_plugin/eval_author/README.md) builds the
  Insight-specific evaluation suite, and Insight mode invokes it automatically. It
  ships inside this plugin.
- **Harbor** runs the task containers that score every candidate.
- **NeMo Experiments** mirrors each run and its candidates as an experiment
  group, so the lineage is visible in Studio. Structure only — rewards and
  trials are not copied there.

Both the Analyst and the Experimentalist read the same `optimizer.yaml`
profile, discovered by walking up from the current directory. See
[Insight-driven optimization](../../docs/agents/insight-driven-optimization.mdx)
for the full concept guide.

## Run it

**Start with
[Get started with an example agent](../../docs/get-started/example-agent.mdx).**
It walks a checked-in example agent end to end — platform setup, dataset
preparation, `.env` contents, the run itself, and how to read the result. It is
the only complete worked path, and it is far easier to point this at your own
agent once you have watched a run finish. The rest of this section is reference
for when you get there.

This plugin lives in the `nemo-platform` monorepo and shares the root `.venv`.
From the root of the checkout:

```bash
uv sync
```

Budget for it before you start: a run is an LLM-driven loop that builds and
scores agents in Harbor containers, so even a deliberately shortened run takes
about an hour and a real one takes several. Start it in a persistent session
such as `tmux`, and redirect output to a log if you need to disconnect.

Validate the effective inputs before spending anything. Run `doctor` from the
agent directory containing `optimizer.yaml`, or pass `--profile` — without a
loaded profile it cannot check the datasets or task template, and reports
success having verified neither:

```bash
uv run nemo agents experimentalist doctor
```

### Smoke first

Prove that source checkout, Harbor, credentials, and artifact collection work
together before committing to a real run. Use
[`experimentalist-smoke.yaml`](examples/tau3-nooa-agent/experimentalist-smoke.yaml)
with small copied subsets of your train and validation splits — one round, one
candidate, trajectory scoring off. Add `storage.publish_winner: false` to it
when the source is a Git repository; the checked-in file does not set it, so a
winning smoke candidate would open a draft PR.

A smoke result proves the wiring, not an improvement. It is far too small to
judge whether an agent got better; decide that on the full validation split.

### Two modes

**Insight-driven** — optimize against a diagnosed failure pattern. Uses the
local `.nemo-optimizer/insights.yaml` beside the profile by default; `--insight`
names another local file or a platform Insight ID. Requires a `--task-template`
so Eval Author can build the evaluation suite.

```bash
uv run nemo agents experimentalist run
```

**Dataset-driven** — optimize directly against a benchmark, when it already
provides a trustworthy reward. `--no-insight` bypasses both an explicit Insight
and the profile default, and no task template is needed.

```bash
uv run nemo agents experimentalist run \
  --no-insight \
  --agent path/to/agent \
  --train-dataset path/to/train \
  --validation-dataset path/to/validation
```

### Parameters

Every input can come from the `optimizer.yaml` profile or from a flag; flags
win. "Required" below means required *when the profile does not supply it*.

| Flag | What it does | Required |
| --- | --- | --- |
| `--profile` | The `optimizer.yaml` holding agent, source, datasets, task template, and workspace. | No — discovered by walking up from the working directory. |
| `--agent` | Baseline agent: a local directory, or a Git URL that also enables PR/MR publication. | Unless the profile or the Insight names one. |
| `--ethos` | Markdown describing the agent under test. | No — falls back to the profile, then `ETHOS.md`. |
| `--insight` | The problem to work on: a local Insight file or a platform Insight ID. | No — defaults to `.nemo-optimizer/insights.yaml` beside the profile. |
| `--insight-id` | Picks one entry out of a multi-Insight file: exact ID or title, otherwise a zero-based index. | Only for a file holding several Insights. |
| `--no-insight` | Switches to dataset-driven mode. Cannot be combined with `--insight` or `--insight-id`. | Dataset-driven mode. |
| `--train-dataset` | Local Harbor dataset or registry ref — the split candidates are proposed against. | Yes. |
| `--validation-dataset` | The held-out split that selects the winner. | Yes. |
| `--task-template` | Directory holding one Harbor task template (`task.toml` with placeholders); Eval Author fills a copy per failing trace. | Insight-driven mode only. |
| `--config` | Run configuration: round and candidate limits plus `source`, `storage`, `goal_config`, `coder`, `analyzer`, `proposer`, `evaluator`, `eval_author`. Rejects a `models:` key. | No — defaults apply. |
| `--workspace` | NeMo workspace for traces and run metadata. | No — profile, else `default`. |
| `--base-url` | URL of the running platform. | No — `NMP_BASE_URL`, else `http://localhost:8080`. |
| `--experiment-dir` | Where `eval-and-optimize/` is written. Also `-o`, `--output`, `--experiments-output`. | No — see [Output](#output). |
| `--framework-skills` | Directory of framework skills to load into the optimizer agents; repeatable. Two ship with the plugin — see below. | No. |

`doctor` takes `--profile`, `--insight`, `--insight-id`, and `--base-url`.

Only a Git `--agent` can publish a winner. To pin a ref, the `.git@` marker is
required — `git@github.com:owner/repository.git@main`. A URL such as
`https://github.com/owner/repository@main` is read as a plain repository URL and
silently uses the default branch.

### Framework skills

The coding agent modifies your source, so it does better when it knows the
framework your agent is written in. Two skills ship in
[`framework-skills/`](framework-skills/) — pass the directory with
`--framework-skills`, or list it in the profile:

- [`nooa`](framework-skills/nooa/) — NVIDIA-labs OO Agents: `nooa.Agent`
  subclasses, generation methods, CodeAct and Predict strategies, tools, skills,
  context, and tracing.
- [`langchain-framework`](framework-skills/langchain-framework/) — LangChain,
  LangGraph, and Deep Agents: tool-calling agents, stateful graph workflows, and
  multi-step orchestration.

Without a matching skill the optimizer still runs; it just has less framework
guidance to draw on when writing changes.

### Output

Each run writes under `--experiment-dir`: the resolved source agent, every
candidate, per-round analysis, evaluator results, `eval-and-optimize/run.json`
(which names the winner), and an `OPTIMIZATION.md` summary.

Two defaults worth knowing before a first real run:

- `storage.publish_winner` defaults to `true` with `pr_draft: true`. Against a
  Git source, a successful run **opens a draft PR**. Set it to `false` in
  `--config` if you do not want one.
- `--experiment-dir` defaults to `<profile-dir>/.nemo-optimizer/experiments/<timestamp>-<uuid>`
  — but to `./tmp` when no profile governs the run.

An interrupted run resumes. There is no resume flag: point `--experiment-dir` at
the same directory and the optimizer detects the last completed round from the
artifacts already there, discards anything from that round onward, and continues.
Do not reuse a partial directory for a different source revision, dataset, or
configuration — start a new one.

## Configuration

Two kinds, deliberately separated.

**Deployment settings** — which models the optimizer agents run on. One per
install, not per experiment, and they come from the active Platform context
rather than from this plugin. Run `nemo setup` once: it registers an inference
provider, stores its credential as a Platform Secret, and asks for two
workspace-qualified Model Entities.

| Model | Used by |
|---|---|
| default | Top-level reasoning in the optimizer loop, Coder, Analyzer, Proposer, Rationalizer, TraceAnalyzer, and the architecture doc |
| fast | Trajectory scorer, Terminator, and goal tree, plus high-volume sub-steps and summarizers inside most other components |

Give them different models — the default writes the code, fast runs the
high-volume judging — but the fast prompt accepts Enter to reuse the default,
which is what a single-model context gets. The Experimentalist resolves both
entities through Platform and routes each completion by the entity's backend
format, so any registered provider exposed as OpenAI Chat Completions or
Anthropic Messages works. It reads no optimizer-specific endpoint, provider key,
or provider model name of its own. `nemo agents experimentalist doctor` reports
the effective pair.

For non-interactive or isolated environments, `NEMO_DEFAULT_MODEL` and
`NEMO_FAST_MODEL` override the stored selections; both take Platform Model
Entity IDs in `workspace/model-name` form. The sandbox example below passes them
because the host's `~/.config/nmp/config.yaml` is not part of the clone. The
[example agent's `.env.example`](examples/tau3-nooa-agent/.env.example) shows
the shape, and an adjacent `.env` is loaded automatically when a profile is
found.

**Run configuration** — `--config` holds what *one experiment* does
(`max_rounds`, `max_survivors`, `storage`, per-component tuning). It takes no
environment override, so the file stays an accurate record of the run. It
rejects a `models:` key outright: models are a deployment setting, chosen by
`nemo setup`.

### What to improve, and what not to break

Declare the two separately. `objective_function` is an ordered list of evaluator
metrics the run should improve; `regression_metrics` lists what must not get
worse while that happens. Each entry is a metric name plus a direction, and the
optimizer only ever sees reported metric values and this declared policy — it
does not evaluate expressions, invent weights, or encode its own selection rule.

`objective_function` defaults to maximizing `reward`, so a run without one still
works. Set it when the evaluator emits something better to steer on:

```yaml
objective_function:
  - name: tokens
    direction: minimize
  - name: cost
    direction: minimize
regression_metrics:
  - name: success_rate
    direction: maximize
```

Objectives are what candidates are Pareto-ranked on, with minimized metrics
sign-inverted; regression metrics are deliberately kept out of that ranking. In
an Insight-driven run, Eval Author's authored Insight metrics take over as the
objective and your configured targets move to `regression_metrics`, so the
Insight gets fixed without giving up what the run already cared about.

The agent under test is configured separately, and none of the variables above
reach it. What arrives in the evaluation container is whatever each Harbor task
declares in its `task.toml` `env` block — typically `OPENAI_API_KEY` and
`OPENAI_BASE_URL` expanded from your shell, plus any task-specific variables.
Both example agents additionally read `AUT_MODEL_NAME` to choose their model; if
you follow that convention in your own agent, the benchmark runner sets it for
you.

## Agent trace formats

The agent under test can emit traces as OTLP or ATIF. **OTLP is the default** —
skip this section unless your agent emits ATIF.

Have the agent write its trajectory under its trace directory (`/app/traces` in
the Harbor task container) with a `.atif.json` suffix, select the format on the
evaluator, and run against a platform, which ATIF requires:

```yaml
experiment_config:
  evaluator:
    trace_format: atif   # otlp (default) | atif
```

```bash
uv run nemo agents experimentalist run --base-url https://<platform-host> ...
```

Experiment grouping, run counts, and evaluator scores in Studio behave the same
as for OTLP. ATIF traces carry no per-step timing, so step durations show as
zero.

Two errors point back here: *"configured `trace_format='otlp'` matched no trace
artifact, but atif artifacts are present"* means set `trace_format: atif`;
*"Cannot read ATIF trajectory from disk … this trace was never uploaded"* means
the run had no reachable platform, so supply `--base-url` and a workspace.

## Recommended laptop isolation

The Experimentalist executes LLM-authored code and shell and Docker commands.
Use [Docker Sandboxes](https://docs.docker.com/ai/sandboxes/) rather than a
privileged Docker-in-Docker container or a host Docker socket mount: the
Experimentalist runs inside an isolated microVM and Harbor uses that sandbox's
private Docker daemon for task containers, while clone mode gives it a private
writable clone instead of write access to your checkout. Requires Docker Engine
29.6.2 or later; tested with Docker Sandboxes (`sbx`) 0.37.1.

```bash
repo="$(git rev-parse --show-toplevel)"
sbx create --clone --name nemo-experimentalist shell "$repo"
sbx exec --workdir "$repo" \
  --env UV_PROJECT_ENVIRONMENT=/home/agent/.venvs/nemo-platform \
  --env NEMO_DEFAULT_MODEL \
  --env NEMO_FAST_MODEL \
  nemo-experimentalist \
  uv run --frozen --python 3.13 --package nemo-experimentalist-plugin --with ./plugins/nemo-agents \
  nemo agents experimentalist run
```

`UV_PROJECT_ENVIRONMENT` keeps the sandbox's Linux environment separate from the
host checkout's `.venv`. Append the run options described above.

Know the limits of this boundary:

- Clone mode stops sandbox writes to your checkout, but it is **not a secret
  isolation boundary** — the complete host repository, including ignored `.env`
  files, stays readable at `/run/sandbox/source`.
- Any value passed with `sbx exec --env` is readable by the optimizer, the
  candidate agent, the verifier, and their subprocesses. The two model overrides
  above are entity IDs rather than secrets, but the agent under test still needs
  its own credentials in there — use dedicated, revocable, spending-limited keys.
- Optimizer and task code can use any outbound access granted to the sandbox,
  which must reach your package, model, registry, Harbor dataset, and NeMo
  Platform endpoints.

`sbx stop nemo-experimentalist` preserves the VM, output, packages, and private
Docker cache; `sbx rm nemo-experimentalist` deletes them. Copy artifacts you
want to keep to the host with `sbx cp` — the
[example walkthrough](../../docs/get-started/example-agent.mdx#5-optimize-performance-with-the-experimentalist)
has concrete inspection and copy commands.

On Apple silicon, Harbor tasks that publish only `linux/amd64` images do not run
in the `linux/arm64` sandbox. This currently includes the Terminal-Bench
`fix-git` task; use an x86_64 machine or VM for that suite.

## Examples and benchmarks

Two example agents are checked in, both used as agents under test by the
benchmark suites:

- [`examples/tau3-nooa-agent`](examples/tau3-nooa-agent/) — a NOOA CodeAct agent
  reaching domain tools over MCP. The one used by the
  [example-agent walkthrough](../../docs/get-started/example-agent.mdx).
- [`examples/terminal-bench-agent`](examples/terminal-bench-agent/) — a
  deliberately small LangChain agent with a single bounded shell tool, running
  inside each Harbor task container.

[`benchmarks/`](benchmarks/) holds the canonical suites used to measure
optimization quality: Terminal-Bench 2.1 (89 tasks) and tau3-bench scoped to
banking, airline, retail, and telecom. They store only task IDs and download
task definitions into a local cache. See
[`benchmarks/README.md`](benchmarks/README.md) for provenance and how to run
them.

## Develop

```bash
uv run pytest plugins/nemo-experimentalist/tests
uv run ruff check plugins/nemo-experimentalist
```

The plugin registers two entry points: `nemo.cli.agents`, which mounts the
`experimentalist` verb under `nemo agents`, and `nemo.skills`, which ships the
skills bundled with the plugin.

Requires `uv >=0.10.10`. Source dependencies are pinned to tagged or immutable
revisions in the workspace root `pyproject.toml` under `[tool.uv.sources]`,
where each pin carries a comment explaining why.

License: Apache-2.0.
