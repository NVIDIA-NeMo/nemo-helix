---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
schema_version: 1
name: <agent-name>
created_timestamp: <ISO-8601-creation-timestamp>
author: <actual-author>
---

# Ethos: <agent-name>

## Role

<One concrete sentence describing the agent's role for its users.>

## Purpose & Outcomes

<Why the agent exists and what users should reliably accomplish. Include agreed
success targets, if any; do not invent business metrics.>

## Scope

- Audience: <intended users>
- Categories: <task categories separated by semicolons>
- In scope: <supported workflows>
- Out of scope: <excluded workflows>

## Tools

<Actual tools and knowledge sources, relevant side effects, data freshness, and
expected limitations. Use Prompt-only. when no tools exist.>

## Harness

<How the existing agent runs, invokes tools, maintains state, and stops.>

## Behavior

<Expected behavior, refusals, escalation, accepted limitations, and non-goals.>

## Principles

<User-confirmed judgment calls where fixed rules do not settle a choice.>

## Success Criteria

<Observable outcomes that distinguish success from failure.>

## Trade-offs

<Agreed priorities between competing outcomes, or _(none)_.>

## Constraints

<Actual operational, data, safety, or change constraints, or _(none)_.>

## Change Scope

- <existing change lever>: <yes, no, or with-approval>

## Evaluation Setup

<Existing local evaluation setup, or explicitly state that evals are not yet
created. Do not claim proposed commands have been executed.>

## Metric Semantics

<What available metrics support and do not support, or _(none)_.>

## Vision

<User-confirmed future direction, or _(none)_.>

## Open Questions

- <Unresolved fact affecting behavior or evaluation; use _(none)_ if settled.>
