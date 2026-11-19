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

## Pipeline stub examples

The top-level `planner.yaml`, `writer.yaml`, and `reviewer.yaml` are the original
basic configurations for `pr-planner`, `pr-writer`, and `pr-reviewer`. The current
pipeline implementation still hardcodes those names and forwards text responses.
The standalone agents use richer input/output contracts and are not wired into
that stub by this directory reorganization. Connecting them requires explicit
plan/PR handoffs and per-step runtime image selection.
