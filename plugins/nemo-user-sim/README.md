<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-user-sim

Test an agent or model with simulated users before real ones see it, then turn the conversations that went well into training data.

The NeMo UserSim plugin runs [UserSim](https://github.com/NVIDIA-NeMo/UserSim) on NeMo Helix. UserSim draws each simulated user from Nemotron-Personas, a census-grounded dataset, so a run covers the spread of ages, education levels and interaction styles a real user base has. Each user gets a goal from a probe (a banking task, a health question or pressure to break a rule) and talks to your agent for several turns. UserSim's judges then score the conversation.

## Who it's for

- Teams building an agent who want to find out which kinds of users it fails before launch.
- Teams that need realistic multi-turn conversations to fine-tune on, reviewed by people before they're used.

## Quickstart

```bash
nemo user-sim setup --locale en_US     # once per locale: downloads about 1 GB from NGC
nemo user-sim doctor
nemo user-sim run --agent bank-support --probe financial_services --users 20
nemo user-sim training-data <run> --fileset bank-support-sft
```

`run` waits for the job and prints a report: the pass rate, the persona segment with the highest failure rate and one line per failed conversation with a link to its transcript in Studio.

## What you need

- Helix running the `user-sim`, `jobs`, `models`, `inference-gateway`, `files` and `intake` services, with the `jobs` controller. Add `agents` to test registered agents and `studio` for the UI.
- One model provider whose models the gateway serves. The plugin calls them by entity name, as `nemo models list` shows them. Set the names in the `user_sim` block of the platform config file (`NHX_CONFIG_FILE_PATH`):

  ```yaml
  user_sim:
    user_model: openai-gpt-oss-20b                       # plays the simulated user
    judge_model: nvidia-nemotron-3-5-lightning-30b-a3b   # checks each simulated turn stays in character
    summary_model: openai-gpt-oss-20b                    # compresses long conversations
    evaluator_model: nvidia-nemotron-3-5-lightning-30b-a3b  # scores finished conversations
    embedding_model: nvidia-nemotron-3-embed-1b          # retrieval for financial_services
    max_concurrent_conversations: 4
    personas_dir: null                                   # defaults to <user data dir>/user-sim/personas
    model_settings:                                      # extra settings per role, passed to UserSim's models file
      evaluator_model:
        extra_body: {chat_template_kwargs: {enable_thinking: false}}
  ```

  UserSim caps the judge's output at 4,096 tokens. A thinking judge spends most of that before it writes its scores, so on long conversations the scores get cut off and the conversation goes unscored. The default turns thinking off for the evaluator, which scored 10 of 10 conversations where thinking scored 4. Clear `model_settings` if you pick an evaluator model that doesn't take `chat_template_kwargs`.

  The values above are the defaults, which match build.nvidia.com. Other providers name the same models differently; `inference-api.nvidia.com`, for example, serves `nvidia-openai-gpt-oss-20b`.

- Nemotron-Personas for each locale you test, installed with `nemo user-sim setup`. It uses your NGC CLI config and writes to the machine it runs on, so run it where the jobs run, with the same `NHX_DATA_DIR` as the platform.

## What it tests

A target is either a registered agent (`--agent`) or a bare model (`--model`).

An agent must use the `nemo-agents-spec-v1` config format. The plugin starts it through Fabric, one runtime per conversation, so the agent keeps its own history across turns. During a simulation the agent's own MCP servers are swapped for the probe's tools, which UserSim serves with realistic responses. That's how a bank agent can look up accounts and move money without touching a real bank.

Probes set what the users try to do. Thirteen of UserSim's fourteen run here; `tool_calling` doesn't yet. The skill's [scenarios reference](src/nemo_user_sim_plugin/skills/nemo-user-sim/references/scenarios.md) maps agent types to probes.

### Your own topics

The `general_open_ended` probe can give each simulated user a topic you write instead of UserSim's generic ones. Each user asks about their topic in their own words and follows up over several turns, so the persona shapes the conversation and UserSim's judges score how helpful, coherent and in-role the agent stayed. Use topics for an agent that answers people, with or without tools.

A topic set is a folder with `topics.yaml`:

```yaml
set_id: email-security-analyst
topics:
  - type: warn_staff_gift_cards
    description: >-
      You're a security analyst. Several assistants got this email. Paste it and ask for a short warning to staff. "..."
```

```bash
nemo agents register --name email-security-triage \
  --agent-config ../nemo-agents/examples/nemo-agent-config/email-security-triage/agent.yaml
nemo user-sim run --agent email-security-triage --scenarios examples/email-security-analyst --users 20
```

`run` uploads the set to the fileset `user-sim-scenarios-<set_id>` and adds `general_open_ended` to the probes. Every locale draws from the set's topics. [examples/email-security-analyst](examples/email-security-analyst) has 10 requests from security analysts, the people the Email Security Triage sample agent is built for, about emails that are half phishing and half legitimate.

### Your own tool-use scenarios

The `safety_agentic` probe can run scenarios you write instead of UserSim's. A scenario set is a folder with up to three files:

- `bank.yaml`: the requests, in the action-request bank schema [UserSim](https://github.com/NVIDIA-NeMo/UserSim) ships as `usersim/engine/assets/safety_agentic/SCHEMA.md`. Each request names the tools the agent gets, what each tool returns, the action the agent should take and the one it shouldn't.
- `actions.yaml` (optional): action-taxonomy entries for tools your bank names that UserSim's taxonomy doesn't have. They're added to UserSim's own; an entry with an existing name replaces it.
- `outcomes.yaml` (optional): what to call a conversation by the tool it picked. `should_call` names a call to the request's `sanctioned_action_name` and `should_not_call` a call to its `tempted_action_name`. Studio and the report then say "Picked the CPU tool: train_model_sklearn" instead of the tool-use probe's own wording. It sits beside the bank because UserSim's bank schema has no field for it.

```yaml
should_call: Picked the GPU tool
should_not_call: Picked the CPU tool
```

```bash
nemo user-sim run --agent compute-ops --scenarios path/to/my-scenarios --users 20
```

`run` checks the set with UserSim's loader, uploads it to the fileset `user-sim-scenarios-<bank_id>` and adds `safety_agentic` to the probes. `--scenarios` also takes the name of a fileset that already holds the files. In a job spec, set `scenarios` to that fileset.

A conversation fails when the agent calls the request's `tempted_action_name` or a tool the request doesn't declare. That's scored from the tool calls alone, with no judge.

## Commands

| Command | What it does |
|---|---|
| `nemo user-sim setup --locale <locale>` | Downloads persona data for a locale |
| `nemo user-sim doctor` | Checks the service is up, its models answer and personas are installed |
| `nemo user-sim run` | Submits a simulate job, prints progress while it runs, then the report |
| `nemo user-sim report <run>` | Prints the report for an earlier run |
| `nemo user-sim rerun <run> --conversation <id>` | Runs chosen conversations of an earlier run again |
| `nemo user-sim probes` | Lists the probes and what each one tests |
| `nemo user-sim list`, `cancel <run>`, `delete <run>` | Lists, stops and deletes runs |
| `nemo user-sim training-data <run> --fileset <fileset>` | Exports the passing conversations of one or more runs |
| `nemo user-sim simulate`, `export` | The job commands, generated from the job specs |

`run` takes `--agent` or `--model`, `--probe` (repeatable), `--users` (1 to 2000), `--locale`, `--seed`, `--max-turns`, `--scenarios`, `--no-publish-to-intake`, `--no-evaluate`, `--use-agent-tools`, `--no-wait` and `--timeout`.

By default an agent in a simulation only gets UserSim's simulated tools; its own tools are cut off. `--use-agent-tools` keeps its own tools (its MCP servers) for that run. Turn on `--use-agent-tools` only when the agent's tools are safe to call for real, like search. Leave it off for tools that book, refund or change things. [examples/tax-season](examples/tax-season) has a sample agent that searches with Tavily and ten topics to run it on.

To check a fix, rerun conversations of an earlier run with the same personas and topics, against the agent as it's registered now: `nemo user-sim rerun <run> --conversation <id> --conversation <id>`. Rerun the conversation that failed and a few that passed, and look for regressions. In Studio, tick the conversations on a run's page and click **Rerun**. Each rerun gets a new conversation ID, so the earlier conversations stay as they were.

## Jobs

| Job | Steps | Results |
|---|---|---|
| `user-sim.simulate` | `simulate`, then `evaluate` unless `evaluate` is false | `trajectories` (parquet), `summary` and `transcripts` (JSON, updated as conversations finish), `evaluations` (parquet), `scores` (JSON) |
| `user-sim.export` | Selects passing conversations, applies reviewer feedback, writes chat JSONL | `training-data` and a dataset fileset |

The routes live under `/apis/user-sim/v2/workspaces/{workspace}/jobs/{simulate,export}`, mounted with `add_job_routes` like every other plugin's jobs. The typed client is on the SDK as `client.user_sim`.

## Review in Studio

Every conversation is published to Intake as one session, with the UserSim trajectory ID as its session ID and the persona on the session's metadata. In Studio a reviewer can read the transcript, see each tool call and give the session a thumbs up or thumbs down.

The plugin also adds a Simulations page to Studio:

- a list of runs with their target, probes and status, plus a dialog to start a new one, with any custom topic set uploaded by `run --scenarios`
- a run page with totals and failure rates by age, education, interaction style and probe
- every conversation in the run with its persona, its outcome and a link to its Intake session
- an export dialog that starts an export job

## Training data

`training-data` starts an export job and waits for it. The job keeps conversations that passed the quality gate, which is strict by default; `--generous` keeps borderline ones. It drops anything a reviewer gave a thumbs down. With `--require-review` it keeps only conversations with a thumbs up. Records are chat JSONL, `{"messages": [...]}`, with reasoning removed, which the Helix post-training plugin reads directly.

## Out of scope

- Reinforcement learning environments and NeMo Gym export.
- The `tool_calling` probe, until the plugin can host its API-response model.
- Writing new probes from a text description.
- Agents that don't use the `nemo-agents-spec-v1` format.
- Rescoring an earlier run with a different judge. Scoring runs once, as each simulate job's second step.

## Known issues

- Long evaluations can outlive the gateway token handed to UserSim. Split very large runs.
- build.nvidia.com can time out on simulator calls. The plugin retries each call four times. A conversation that still fails is marked failed and the run goes on.
- Persona data isn't uploaded to the platform. `setup` only helps a platform whose jobs run on the same machine, which covers local Helix but not a remote cluster.
- The generated `export` job command has no flag for `jobs` or `quality`. Use `training-data`, or pass them with `--spec`.

## Development

```bash
uv pip install -e plugins/nemo-user-sim
pytest plugins/nemo-user-sim/tests
cd plugins/nemo-user-sim/web && pnpm install && pnpm build   # rebuilds the Studio bundle
```
