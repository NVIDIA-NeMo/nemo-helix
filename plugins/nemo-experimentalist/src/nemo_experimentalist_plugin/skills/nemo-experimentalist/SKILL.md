---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-experimentalist
description: Improve an existing NeMo agent's source or harness from an Insight or explicit Harbor-compatible evaluation datasets. Run the Experimentalist to propose and validate candidate code changes, then optionally publish a changed winner as a draft PR or MR.
triggers:
  - run experimentalist
  - experimentalist plugin
  - optimize from an insight
  - improve an agent with evaluation data
  - improve an agent on a benchmark dataset
  - create an experimentalist candidate
  - validate a candidate code change
  - improve an agent harness
  - optimize an agent from train and validation splits
  - draft PR for an agent improvement
not-for:
  - nemo-explore (use to design a new agent before it exists)
  - nemo-experiments-upload (use to upload traces or evaluation results)
  - nemo-evaluator (use to author a new evaluation or metric)
  - nemo-insights / agents analyst (use to generate the Insight before optimizing from it)
  - nemo-eval-author (use to author an evaluation outside an Experimentalist run)
  - agents-optimize (use to tune routing, cost, or latency for a deployed agent)
compatibility: requires the enabled `nemo-experimentalist-plugin`, Docker for Harbor evaluation, a running local platform, and the repository-root `.venv` for LLM-authored code.
maturity: beta
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read, Write
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Experimentalist

Use the Experimentalist to improve an existing agent. It evaluates a baseline,
proposes code changes, and keeps only candidates that improve validation.

## Choose the optimizer

Use the Experimentalist when the improvement belongs in the agent **harness**:
the source that owns its workflow, tool use, state, and runtime behavior. It
changes that source in candidate branches, evaluates each candidate against the
agent's own harness, and can publish a validated winner for review.

Hand off to `agents-optimize` only when the request is explicitly limited to a
deployed agent's routing, skills, prompts, or cost/latency, with no runtime
implementation change. That workflow owns the deployment-tuning commands; do
not substitute it for an Experimentalist source-change run.

## Prepare the agent

The Experimentalist improves an existing, evaluable agent; it does not design
or scaffold one. Before running it, make sure you have:

- agent source code and locked dependencies. For PR/MR publication, use a Git
  repository with a clean working tree and a pushed baseline revision;
- an `ETHOS.md` covering the agent's **Goal**, **Scope**, **Tools**, and
  **Evaluation** contract, including access configuration and constraints but
  never secret values;
- versioned, Harbor-compatible train and validation datasets. Keep a separate
  test split for a final manual baseline-versus-winner comparison. The
  Experimentalist profile and CLI accept only train and validation datasets, so
  run this third split manually with the agent's evaluator; never use it for
  candidate selection;
- a baseline smoke test that proves the agent runs, scores, and emits its
  expected artifacts and traces; and
- an `optimizer.yaml` profile, or the explicit inputs described below.

Record the agent revision, dataset versions, configuration, and result path
with each run so another operator can reproduce it. If the agent or its Ethos
does not exist yet, use `nemo-explore` to design it and `nemo-ethos` to create
the Ethos before returning here.

## Configure the environment

The repository-root `.venv` is the only supported Experimentalist environment.
From the repository root, synchronize it and use its CLI:

```bash
uv sync
export NEMO="$PWD/.venv/bin/nemo"
```

Before running `$NEMO` or relying on a local platform, complete the
repository-root `SETUP.md`. Then set the Platform URL and verify readiness:

```bash
export NMP_BASE_URL=http://localhost:8080
curl -sf "$NMP_BASE_URL/health/ready"
```

Stop if setup is incomplete or the readiness check fails.

The Experimentalist needs a running NeMo Platform, an LLM endpoint for its
optimizer agents, and a model for each tier. For the NVIDIA Inference Gateway,
set the inference key and model tiers:

```bash
export INFERENCE_API_KEY=<gateway-api-key>
export NEMO_EXPERIMENTALIST_MODELS_SMART=<model-name>
export NEMO_EXPERIMENTALIST_MODELS_MID=<model-name>
export NEMO_EXPERIMENTALIST_MODELS_FAST=<model-name>
```

Choose a strong frontier model for `SMART`: it performs the highest-stakes
reasoning and code changes. Use a capable, lower-cost model for `MID`, where
good judgment still matters but the work is less demanding. Use a fast,
low-latency model for `FAST`; it serves high-volume supporting work and does
not need the same reasoning depth.

The CLI uses the NVIDIA Inference Gateway by default and reuses
`INFERENCE_API_KEY` as the Experimentalist key. For another OpenAI-compatible
endpoint, set these instead of `INFERENCE_API_KEY`:

```bash
export NEMO_EXPERIMENTALIST_API_BASE=https://llm.example.com/v1
export NEMO_EXPERIMENTALIST_API_KEY=<endpoint-api-key>
```

Keep these values in your shell environment or an ignored environment file,
never in `optimizer.yaml`, the run configuration, or Git. When a profile is
loaded, the CLI automatically loads `<profile-dir>/.env`; existing shell values
take precedence. Configure any credentials required by the agent under test
according to its own setup. To publish a winning Git candidate, authenticate
the relevant CLI in the same execution environment: `gh auth login` for GitHub
or `glab auth login` for GitLab.

## Isolate the run

The optimizer asks an LLM to edit the agent source and execute commands; Harbor
also runs task containers. Do not run an unreviewed optimization with broad host
access or production credentials. Prefer a disposable Docker Sandbox in clone
mode, use dedicated, revocable, spending-limited keys, and retain only the
network access the agent, Harbor, and model endpoint need. Clone mode protects
the host checkout from writes but does **not** make ignored files secret: they
remain readable in the sandbox. Follow the plugin's [recommended laptop
isolation](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/plugins/nemo-experimentalist/README.md#recommended-laptop-isolation)
when running locally.

From the repository root, create a clone-mode sandbox and run the
Experimentalist inside it. Append the run-mode options from this skill to the
last line:

```bash
repo="$(git rev-parse --show-toplevel)"
sbx create --clone --name nemo-experimentalist shell "$repo"
sbx exec --workdir "$repo" \
  --env UV_PROJECT_ENVIRONMENT=/home/agent/.venvs/nemo-platform \
  --env INFERENCE_API_KEY \
  --env NEMO_EXPERIMENTALIST_API_BASE \
  --env NEMO_EXPERIMENTALIST_API_KEY \
  --env NEMO_EXPERIMENTALIST_MODELS_SMART \
  --env NEMO_EXPERIMENTALIST_MODELS_MID \
  --env NEMO_EXPERIMENTALIST_MODELS_FAST \
  nemo-experimentalist \
  uv run --frozen --python 3.13 --package nemo-experimentalist-plugin --with ./plugins/nemo-agents \
  nemo agents experimentalist run
```

The sandbox has its own writable clone and virtual environment; do not point it
at the host `.venv`. It is not a secret boundary: ignored files remain readable
inside the sandbox. Forward only dedicated, revocable credentials and ensure
the sandbox can reach the Platform, model endpoint, registry, and Harbor
datasets it needs.

## Pre-flight

Complete the `SETUP.md` and readiness check above before continuing. Run the
built-in checks from the agent directory that contains `optimizer.yaml`,
or pass the profile explicitly. Without a loaded profile, `doctor` cannot check
the dataset or task-template artifacts:

```bash
$NEMO agents experimentalist doctor --profile path/to/optimizer.yaml
```

Stop and resolve any missing platform, credential, Docker, or dataset check
before starting a run.

## Choose a run mode

- **Insight-driven:** use an Insight ID or the default Insight written by
  `nemo agents analyst run` in the agent profile. An Insight is a systematic
  symptom inferred from a set of traces. Use this mode when the agent has no
  reliable, verifiable reward for the behavior you need to improve.
- **Explicit evaluation:** pass `--no-insight`, an agent, and separate train
  and validation datasets. Use this mode when a benchmark or other evaluator
  provides a trustworthy, verifiable reward that is a good measure of agent
  quality.

Keep validation independent from training. Never change an evaluation dataset
just to make a candidate win.

## Inputs

Choose either a reusable `optimizer.yaml` profile or pass the required inputs
directly as CLI flags. A profile is optional; when used, flags override its
values. With no `--profile`, the CLI discovers `optimizer.yaml` by walking up
from the current directory.

| Input | Purpose | Required |
| --- | --- | --- |
| `--profile` | The `optimizer.yaml` profile containing the agent, datasets, workspace, and optional config. | No; discovered when present. |
| `--insight` / `--insight-id` | The problem to improve. `--insight` accepts a local Insight file or a platform Insight ID; `--insight-id` selects an entry in a local multi-Insight file. | Insight-driven mode. |
| `--no-insight` | Disables profile Insight discovery for an explicit evaluation run. | Explicit mode. |
| `--agent` | Local agent directory or Git URL. A Git URL enables candidate branch and PR/MR publication. | Explicit mode; optional when an Insight supplies the agent. |
| `--ethos` | Markdown description of the agent. | Optional; use the profile or conventional `ETHOS.md` when available. |
| `--train-dataset` / `--validation-dataset` | Separate local Harbor datasets or registry references used to measure improvement. | Yes, unless the profile supplies both. |
| `--task-template` | A directory containing one Harbor task template (`task.toml`, with placeholder values). In Insight-driven mode, Eval Author copies and fills it for representative failing traces to create the targeted evaluation suite. | Required in Insight-driven mode unless the profile supplies it. |
| `--config` | YAML or JSON **mapping** that validates as the Experimentalist run configuration: top-level run limits plus optional `source`, `storage`, `goal_config`, `coder`, `analyzer`, `proposer`, `evaluator`, and `eval_author` sections. It does not configure model endpoints or model tiers. | No; profile or defaults apply. |
| `--workspace` / `--base-url` | NeMo workspace and platform URL. | Workspace defaults to the profile or `default`; base URL uses `NMP_BASE_URL` or localhost. |
| `--experiment-dir` / `--output` / `--experiments-output` / `-o` | Experiment directory that receives `eval-and-optimize/`: the resolved source agent, generated candidates, per-trial results, analysis, `run.json` state, and `OPTIMIZATION.md` summary. | No; default is `<profile-dir>/.nemo-optimizer/experiments/<timestamp>-<uuid>` with a profile, otherwise `./tmp/<timestamp>-<uuid>`. |
| `--framework-skills` | Additional framework-skill directories for the optimization agents. | No; may be repeated. |

An `optimizer.yaml` profile must identify the agent and its source, one task
template, and independent train and validation datasets. It can also set the
workspace, Ethos, run configuration, and framework skills. The
Analyst and Experimentalist share this profile, allowing the latter to use the
default Insight created by the former.

## Explore configuration

Start with the deliberately small configuration below. The top-level options control rounds and
candidate counts; `source` controls checkout behavior; `storage` controls
candidate branches and PR/MR publication; `outcome_evaluator_config` controls trial
execution; and `eval_author` controls Insight-driven evaluation authoring. See
the [example-agent walkthrough](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/docs/get-started/example-agent.mdx)
for a complete worked optimizer configuration and run.

To inspect the complete run schema—including nested options and their defaults—run
this in the environment where the Experimentalist plugin is installed:

```bash
python -c \
  'import json; from nemo_experimentalist_plugin.config import EvolutionaryOptimizerConfig; print(json.dumps(EvolutionaryOptimizerConfig.model_json_schema(), indent=2))'
```

`outcome_evaluator_config` deliberately renders as an open mapping in that schema. For its
Harbor-specific settings, use the smoke configuration below and the plugin's
Harbor evaluator documentation; for example, `outcome_evaluator_config.n_attempts` defaults
to `1`.

Keep model endpoint and model-tier settings out of this file; configure those
as Experimentalist deployment settings instead.

### Tune the Evolutionary Optimizer

Use a smoke configuration to prove that source checkout, Harbor, credentials,
and artifact collection work together. It is intentionally too small to judge
whether an agent is better. For a real run, give the optimizer enough
candidates and train tasks to learn, then rely on the separate validation split
to select the winner.

| Setting | Smoke run | Real run starting point | Effect |
| --- | --- | --- | --- |
| `max_rounds` | `1` | `10`–`15` | Maximum optimization rounds. |
| `max_candidates` / `max_survivors` | `1` / `1` | `3` / `3` | Exploration per round and candidates retained as parents. |
| `max_train_batch_tasks` | Small fixed batch, e.g. `4` | `null` for all train tasks, or a representative fixed batch | Cost and signal available while proposing changes. |
| `max_trajectory_tasks` | `2` | `8` | Tasks used for trajectory/goal-tree scoring. |
| `trajectory_scorer` / `terminator` | `null` / `null` | Omit them (they default to `goal-tree` and `convergence`) | `null` means no implementation of that role, skipping costly diagnostic work for smoke runs; omitting restores quality and early stopping for real runs. |
| `builder_config.max_fix_attempts` | `1` | `2` (default) | Maximum repair iterations when a candidate fails its integration check. |
| `outcome_evaluator_config.n_attempts` | `1` | `1`; increase only when task results are noisy | Repeats each evaluation trial. |
| `eval_author.max_traces` | `3` | `10` | Representative Insight traces deeply analyzed in Insight-driven mode. |

A small explicit smoke configuration looks like this:

```yaml
max_rounds: 1
min_rounds_before_stopping: 1
max_survivors: 1
max_candidates: 1
max_trajectory_tasks: 2
max_train_batch_tasks: 4
trajectory_scorer: null
terminator: null
storage:
  # The real default is true. Keep this false until you intend to create a PR/MR.
  publish_winner: false
builder_config:
  max_fix_attempts: 1
outcome_evaluator_config:
  n_attempts: 1
eval_author:
  max_traces: 3
```

`builder_config.max_architecture_doc_iterations` is how many iterations the
builder can use to write `architecture.md`, and it defaults to `100`. It is
separate from the evaluation budget above, but each additional iteration adds
model usage and run time to that step. Increase it when a run stops with
`Generation failed after 100 iterations (max_iterations=100)`. An agent that has
many source files needs more iterations.

### Create a low-cost smoke dataset

When the full dataset is expensive, create small **copied** train and
validation subsets for the smoke run. Select a few representative task
directories from each original split—for example, different task types or
known failure modes—and copy each complete Harbor task directory, including
its `task.toml` and verifier files:

```bash
mkdir -p smoke/train smoke/validation
for task in task-a task-b task-c; do
  cp -R "full/train/$task" smoke/train/
done
for task in task-d task-e; do
  cp -R "full/validation/$task" smoke/validation/
done
```

Point the smoke command at `smoke/train` and `smoke/validation`. Record the
selected task IDs with the smoke configuration. Do not edit or replace the
canonical splits, and do not use a smoke subset to decide whether a candidate
is an improvement; run that decision against the full validation split.

For a real run, begin with the default values, raise only the settings that
match the available evaluation budget, and preserve the same validation split
throughout. Use `storage.archive_candidates: true` only when you need every
candidate branch for review; it increases remote repository noise.

## 1. Run a smoke check

Use copied, low-cost dataset subsets and the smoke configuration above. Keep
`storage.publish_winner: false`: its default is `true`, so a Git source can
otherwise create a draft PR/MR when a candidate wins.

```bash
$NEMO agents experimentalist run \
  --no-insight \
  --agent path/to/agent \
  --train-dataset smoke/train \
  --validation-dataset smoke/validation \
  --config path/to/experimentalist-smoke.yaml
```

Run this command in a persistent terminal session such as `tmux`, because even
the shortened walkthrough can take up to an hour. Redirect output to a log if
you need to disconnect, then follow it with `tail -f`.

## 2. Run from an Insight

Before this mode, install the Experimentalist plugin, record agent traces in
Intake, and run `nemo agents analyst run` to create the Insight. Use explicit
evaluation instead if you already have a reliable reward and do not need an
Insight.

Pass the inputs explicitly when running a one-off optimization:

```bash
$NEMO agents experimentalist run \
  --insight <platform-id-or-local-file> \
  --agent path-or-git-url \
  --task-template path/to/harbor-task-template \
  --train-dataset path-or-harbor-ref \
  --validation-dataset path-or-harbor-ref \
  --workspace <workspace> \
  --config path/to/experimentalist.yaml
```

`--agent` is optional when the Insight already identifies the agent source.

For a reusable setup, the equivalent information can live in an
`optimizer.yaml` profile. When the profile is complete and the Analyst has
created a default Insight for the agent, this shorter command is enough:

```bash
$NEMO agents experimentalist run --profile path/to/optimizer.yaml
```

Pass `--insight <platform-id-or-local-file>` to override the profile's default
Insight.

## 3. Run from explicit datasets

```bash
$NEMO agents experimentalist run \
  --no-insight \
  --agent path-or-git-url \
  --ethos path/to/ETHOS.md \
  --train-dataset path-or-harbor-ref \
  --validation-dataset path-or-harbor-ref \
  --workspace <workspace> \
  --config path/to/experimentalist.yaml
```

For a Git source, append a ref such as
`git@github.com:owner/repository.git@main`.
The `.git@` marker is required to select a ref; a URL such as
`https://github.com/owner/repository@main` is treated as a repository URL with
no ref and uses its default branch.
For a local source, candidates remain in the experiment artifacts and no PR or
MR can be opened.

## 4. Verify and review

Inspect `<experiment-dir>/eval-and-optimize/run.json` for the selected winner
and compare candidate directories under `<experiment-dir>/eval-and-optimize/agents/`.

Review the validation result and generated PR or MR before merging. Do not
overwrite the baseline agent directly.

## If verification fails

- Fix every required `doctor` failure before retrying; run it with the same
  `--profile` and credentials as the optimization.
- If a run is interrupted, rerun with the **same** `--experiment-dir`. The
  optimizer detects completed rounds and resumes from its saved
  `eval-and-optimize/run.json` and artifacts.
- Do not reuse a partial experiment directory for a materially different
  source revision, dataset, or configuration; start a new directory instead.

## Gotchas

- A smoke result proves the workflow, not an improvement. Select candidates on
  the full validation split and perform any third test-split comparison manually.
- Real runs are expensive and can run for hours. Bound rounds and task counts
  before starting, and inspect the generated winner before merging its PR/MR.
