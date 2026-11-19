<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PR agent examples

This directory contains the basic examples used by the current pipeline stub
and the standalone agents exercised for the hackathon.

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

The planner and reviewer `agent.yaml` files directly target local Helix at
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
time after the agent returns success. These trials do not implement automatic
plan/PR artifact handoffs.

## Pipeline stub examples

The top-level `planner.yaml`, `writer.yaml`, and `reviewer.yaml` are the original
basic configurations for `pr-planner`, `pr-writer`, and `pr-reviewer`. The current
pipeline implementation still hardcodes those names and forwards text responses.
The standalone agents use richer input/output contracts and are not wired into
that stub by this directory reorganization. Connecting them requires explicit
plan/PR handoffs and per-step runtime image selection.
