---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-user-sim
description: >
  Use when someone wants to test an agent or model on NeMo Helix with simulated users, see which
  kinds of users it fails, or generate multi-turn conversations to fine-tune on. Covers every
  `nemo user-sim` command.
triggers:
  - nemo-user-sim
  - user-sim
  - test my agent with simulated users
  - simulate customers talking to my agent
  - which kinds of users does my agent fail
  - generate training conversations from simulated users
  - synthetic conversations for fine-tuning
  - export user-sim conversations as training data
not-for:
  - nemo-agent-hardener (use to red-team and harden an agent against attacks)
  - nemo-evals-plugin (use to score a model on a benchmark or a fixed dataset)
  - nemo-data-designer-plugin (use to design a synthetic dataset from a schema, without an agent in the loop)
preconditions:
  - nemo_cli_available
compatibility: >
  nemo-helix >= 0.9.0 with the nemo-user-sim plugin. Needs the platform running with the jobs
  controller, a model gateway with the simulator models, and Nemotron-Personas for the chosen locale
  (about 1 GB per locale, downloaded once from NGC with the user's NGC CLI config). Makes network calls.
maturity: experimental
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo UserSim

UserSim runs conversations between an agent and simulated users picked from census-grounded personas, scores each conversation, and exports the good ones as chat training data. Every conversation also lands in Studio as an Intake session, where people can review it.

Always use the `nemo user-sim` commands below. Never write your own simulation loop, persona generator or scorer: the plugin's jobs handle persona sampling, the agent connection, scoring and Intake publishing.

## 1. Ask first

Ask the user, in one message, whatever their request didn't already answer:

- Which registered agent (`--agent`) or which model (`--model`) to test. Don't pick one for them. `nemo agents list` shows registered agents.
- Whether the goal is finding failures, training data or both.
- Which locale their users are in, if not `en_US`.

## 2. Pre-flight

```bash
nemo user-sim doctor --locale en_US
```

If personas are missing, tell the user the download is about 1 GB from NGC and needs their NGC CLI config, then ask before running:

```bash
nemo user-sim setup --locale en_US
```

`setup` writes to the machine it runs on, so it only helps when the platform's jobs run there too, as with a local Helix. Run `doctor` again until every check passes. `doctor` checks that the UserSim service answers, that the simulated user, judge, summary and evaluator models answer through the gateway, and that personas are installed. If a model doesn't answer, stop and tell the user: a run started now fails on every conversation. `doctor` doesn't check the jobs controller or the agent's own model, so a run can still stall or fail on those (see Gotchas).

## 3. Pick probes

For a registered agent, leave probes to the plugin: `nemo user-sim run --agent <name>` without `--probe` infers them from the agent's config and its ETHOS.md, when it has one, and prints why each one fits. Pass `--probe` only when the user asks for specific ones. For a bare model, match probes with [references/scenarios.md](references/scenarios.md) and confirm with the user.

## 4. Run

```bash
nemo user-sim run --agent bank-support --users 20
```

- Use `--model <name>` instead of `--agent` to test a bare model, never both.
- Start with `--users 20` for a first run. The maximum is 2000.
- The same `--seed` doesn't guarantee the same personas. To compare an agent before and after a fix on the same personas, rerun the earlier run's conversations with `nemo user-sim rerun <run> --conversation <id> --conversation <id>`. A rerun keeps each conversation's persona and topic and runs the agent as it's registered now.
- `--max-turns` caps the simulated user's turns per conversation (default 5).
- When the user wants simulated users asking about their own domain, or wants to test the agent's tool use on their own situations (their tools, the action it should take, the one it shouldn't), use `--scenarios <folder or fileset>`. See "Custom scenarios" in [references/scenarios.md](references/scenarios.md).
- The command waits and prints the report. It exits with code 1 if the job outlives `--timeout` (default 7200 seconds); that isn't a failure, so run `nemo user-sim report <job>` later. For long runs, pass `--no-wait` and do the same.

## 5. Report

`run` and `report` print the pass rate, the persona segment with the highest failure rate for age, education, interaction style and probe, and one line per failed conversation. Conversations that ran but didn't pass scoring get a Studio link. Conversations that errored (agent timeout, step limit) print the reason instead.

Summarize for the user in this order: overall pass rate, the weakest segments, then two or three failed conversations. The report gives each one's scoring reason (such as `helpfulness`); open its Studio link to describe what actually went wrong in plain words. Under about 20 scored conversations, say that segment rates point at individual users, not patterns.

## 6. Export training data

Only offer this after a run with passing conversations. Read [references/training-data.md](references/training-data.md) first, then ask whether to require Studio review and which quality gate to use.

```bash
nemo user-sim training-data <run> --fileset bank-support-sft
nemo user-sim training-data <run> --fileset bank-support-sft --require-review
```

`training-data` starts an export job and waits for it, so the fileset is complete when the command returns. Add `--generous` to keep borderline conversations. Don't use the generated `export` command: it has no flag for the run. Then hand the fileset to the Helix post-training plugin for the fine-tune.

## Gotchas

- Jobs stay in `created` forever if the platform runs without the jobs controller. If a run never starts, check that first.
- `tool_calling` is rejected. Use `safety_agentic` or `financial_services` for tool-using agents.
- The gateway must serve the simulator models and the embedding model (`embedding_model` in the plugin config, `nvidia-nemotron-3-embed-1b` by default). `financial_services` uses it to search the bank's documentation; if it's missing, conversations fail with gateway errors.
- Most probes draw on banks with some unreviewed entries. Conversations that used one carry a `used_placeholder_*` warning in the job's `trajectories` result; say so when reporting those findings.
- `--require-review` exports nothing until a reviewer gives sessions a thumbs up in Studio.
- `--no-publish-to-intake` skips Studio entirely, so there are no session links and no review step.
- Very long runs can outlive the gateway token. If the last conversations fail with auth errors, split the work into several smaller runs.
