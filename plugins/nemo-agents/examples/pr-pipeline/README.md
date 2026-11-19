<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PR agent examples

This directory contains standalone agents and a configured pipeline that connects
them through independent execute-agent jobs.

## Standalone agents

| Directory | Registered name | Harness | Purpose |
| --- | --- | --- | --- |
| [planner](planner/README.md) | `pr-compose-planner` | Deep Agents | Read a Linear ticket and produce an implementation plan |
| [writer](writer/README.md) | `pr-compose-writer` | Codex | Implement a plan and produce a patch or draft PR |
| [reviewer](reviewer/agent.yaml) | `pr-compose-reviewer` | Pi | Coordinate multi-model review of a GitHub PR |

Each directory contains its canonical `agent.yaml` and example input. The
planner and writer include execute-job examples and setup documentation.
Moving these files does not change their registered names or update existing
Helix registrations. The writer's `run.local.json` is an ignored, personal
snapshot; see its README for the local image and runtime prerequisites.

## Shared local image

The [local Dockerfile](writer/Dockerfile.local) builds `pr-compose:local` with
Deep Agents, Codex, Pi, Git, and GitHub CLI. It also tags the same image as
`pr-compose-writer:local` for existing writer specs. Its build checks runtime
imports, required binaries, and Python dependency compatibility. See the
[build instructions and remaining local configuration](writer/README.md#shared-local-job-image).

## Local planner and reviewer jobs

All three `agent.yaml` files directly target local Helix at
`http://localhost:8080`, workspace `mmogallapall`. The reviewer YAML includes the
Pi 0.6.1 model metadata and sets `profile: "gpu"` on child-job requests (the
Docker-backed CPU profile in this setup). Edit these files directly; no config
rewrite step is needed.

The prepared, Git-ignored `run.local.json` files contain exact inline copies of
the YAML, job inputs, `pr-compose:local`, and secret references. If the YAML
changes, refresh those copies or register the YAML locally and reference the
registered agent in the job spec. Image and secret selection remain job settings.

The planner trial reads AIRCORE-1634 as a retrospective plan. The reviewer trial
reads draft PR #1 in `mmogallapalli/nemo-helix-hackathon`. These are independent
trials; the draft PR does not implement that Linear ticket.

```bash
NHX_BASE_URL=http://localhost:8080 nemo agents execute \
  --workspace mmogallapall --profile gpu \
  --spec-file plugins/nemo-agents/examples/pr-pipeline/planner/run.local.json

NHX_BASE_URL=http://localhost:8080 nemo agents execute \
  --workspace mmogallapall --profile gpu \
  --spec-file plugins/nemo-agents/examples/pr-pipeline/reviewer/run.local.json
```

Both parent configs and all three inline child configs passed offline Fabric
plan/doctor checks. Local execute-job trials completed on 2026-11-19:

| Agent | Job | Result |
| --- | --- | --- |
| Planner | `nemo-agents-plugin-e-ba9fepiq` | `ready`; retrospective `plan.json` and `plan.md` for AIRCORE-1634 |
| Writer | `nemo-agents-plugin-e-2bbhmorg` | `patch_ready`; six-line README addition, no publication |
| Reviewer | `nemo-agents-plugin-e-jfaiggcd` | All three children completed; no blocking findings, two nits |

The first reviewer trial's children could not connect using the old
`gateway.local` placeholder. The reviewer YAML now supplies `http://localhost:8080`
for child model routes; the successful retry verified Docker child scheduling,
model access, GitHub read access, and result aggregation. The writer's full
workdir includes its repository checkout, so artifact upload takes additional
time after the agent returns success. These trials validated the agents
individually; the pipeline below adds automatic artifact handoffs.

## Run the pipeline locally

[pipeline.yaml](pipeline.yaml) defines the reusable sequence:

1. Deep Agents reads the Linear ticket and writes `plan.json`.
2. Codex receives that plan as a JSON object and writes `result.json`.
3. If a PR was created, Pi receives its repository, PR number, and head SHA,
   then coordinates the existing multi-model review. Its output is
   `workspace/review-result.json`.

The runner requires a `ready` plan before starting the writer. The writer may
return `patch_ready`, `pr_created`, or `no_changes`; any other status stops the
pipeline. Review runs only for `pr_created`. An incomplete review also fails the
pipeline. Each agent remains independently usable; all handoffs live here.

### Build the runner image

First build the shared child image and Linux launcher as described in the
[writer instructions](writer/README.md#shared-local-job-image). Then, from the
repository root, build a wheel containing this branch and layer it onto the
shared image:

```bash
uv build --package nemo-helix --wheel --out-dir /tmp/helix-pipeline-wheel

docker build \
  -f plugins/nemo-agents/examples/pr-pipeline/Dockerfile.pipeline \
  -t pr-compose-pipeline:local /tmp/helix-pipeline-wheel
```

Use a wheel directory containing only the wheel from the intended revision.
The Docker context contains the wheel, not credentials or your checkout. The
orchestrator uses `pr-compose-pipeline:local`; the three child jobs use the
verified `pr-compose:local` image with their existing runtime compatibility fixes.
Rebuild the wheel and orchestrator image after runner changes.

The local Helix server must also run this branch. If the tracked local instance
was started before switching/updating the branch, reload it without resetting
its data:

```bash
NHX_BASE_URL=http://localhost:8080 nemo services restart
```

### Register and validate

Use the local `mmogallapall` workspace with its configured Inference Hub models
and secrets `linear-planner-token`, `gh-writer-token`, and `gh-reviewer-token`.
The configs use this workspace's model routes; edit them directly for another
workspace. Docker Desktop host networking must let job containers reach
`http://localhost:8080`.

```bash
export NHX_BASE_URL=http://localhost:8080

for stage in planner writer reviewer; do
  nemo agents register --workspace mmogallapall \
    --name "pr-compose-$stage" \
    --agent-config "plugins/nemo-agents/examples/pr-pipeline/$stage/agent.yaml"
done

nemo agents pipeline-configs validate --workspace mmogallapall \
  -f plugins/nemo-agents/examples/pr-pipeline/pipeline.yaml
nemo agents pipeline-configs create --workspace mmogallapall \
  -f plugins/nemo-agents/examples/pr-pipeline/pipeline.yaml
```

If `pr-compose` is already stored, use `pipeline-configs update pr-compose -f ...`
instead of `create`. Registration and pipeline storage do not start persistent
agent deployments.

### Submit

Copy [run.pipeline.example.yaml](run.pipeline.example.yaml) to a personal file,
replace the ticket placeholder, and select the repository and base branch.
Then run:

```bash
NHX_BASE_URL=http://localhost:8080 nemo agents pipeline \
  --workspace mmogallapall --profile gpu \
  --spec-file /tmp/pr-compose-pipeline-run.yaml
```

Use `--profile gpu` for this setup's Docker-backed CPU profile. Child jobs also
select that profile through `step_defaults.profile`. Each step gets its own
secret references and timeout. The 30-minute wait budget includes scheduling
and artifact upload; it is separate from each agent's run timeout.

The example defaults to **patch mode**, which skips review and never publishes a
branch or PR. To exercise all three stages, explicitly select `mode: draft_pr`,
a new unused `branch`, and `commit_author: {name: ..., email: ...}`. The writer
still returns `no_changes` if it finds the work already implemented. Completed
Linear tickets require an explicit retrospective-planning constraint; a ticket's
Done status does not establish what exists in the target repository.

Inspect the parent job and its child IDs:

```bash
nemo jobs get PIPELINE_JOB_NAME --workspace mmogallapall
nemo jobs get-logs PIPELINE_JOB_NAME --workspace mmogallapall --tail 50
```

The parent saves `pipeline_result.json` as its `pipeline_result` result, with
per-step status, response, attempts, and child job ID. Child jobs retain the plan,
patch, and review artifacts. The runner downloads just the selected JSON file,
not the full checkout archive, and stops on invalid or missing handoff data.

## Legacy pipeline examples

The top-level `planner.yaml`, `writer.yaml`, and `reviewer.yaml` remain the basic
configs for `pr-planner`, `pr-writer`, and `pr-reviewer`. Legacy task-only runs
continue using those names and text responses. Configured runs use
`pipeline.yaml` and the standalone agents above.
