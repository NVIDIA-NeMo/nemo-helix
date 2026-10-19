<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# How to adapt this example for your own agent

**Prerequisites:** you can deploy and invoke the example ([README.md](README.md)).

**Goal:** turn the Email Security Triage example into your own agent — a
single-turn router that picks one capability from the user's request and answers
as that capability.

The shape you're reusing:

```text
one request ──▶ router prompt (deepagents, one generation, no tools)
                    ├── capability A ── contract: line 1 is <token>
                    ├── capability B ── contract: line 1 is <token>
                    └── capability C ── contract: line 1 is <token>
```

## Parts & what to change

| Path                             | What it is                                                                                                | Swap for your own                                                                                                                                                  |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `agent.yaml`                     | The `nemo-agents-spec-v1` config: harness, router prompt, model, blocked tools, timeout, telemetry        | Rewrite `instructions.system` — the routing rules and one `##` section per capability; set `models.default` (+ `temperature`); rename `name` / `telemetry.project` |
| `dataset.jsonl`                  | 28 rows carrying the agent's input (`user_message` + `emails[]`) plus `expected_tool` / `expected_answer` | Drop in your rows; keep the field names the `prompt_template` and the two metrics reference                                                                        |
| `eval-config.dataset-driven.yml` | One metric set over every row (YAML; its task-driven sibling is JSON — both accepted)                     | Rewrite `prompt_template` and the judge prompt to read _your_ contract                                                                                             |
| `eval-config.task-driven.json`   | 8 tasks in 4 families , each with its own metrics                                                         | Swap the tasks; keep one routing assertion per family                                                                                                              |

## Writing the capabilities

Each capability is a `##` section in the one system prompt. Three rules the
existing ones follow:

- **Give it a crisp output contract**, and let it differ per capability. Where a
  deterministic metric has to read the answer, line one carries it: `triage_message`
  emits a single lowercase verdict (`phishing` or `benign`), `review_messages` opens
  with the fixed header `ANALYSIS` and then one block per message — with any reasoning
  or structured detail after. Capabilities whose output has no single correct form need
  no such token: `draft_warning` emits warning prose only, and is graded by a rubric
  judge instead.
- **Make the contracts mutually exclusive.** Routing is measured by output shape:
  if two capabilities can open with the same line, no check can tell them apart, and
  a misroute becomes invisible in your scores.
- **Say when to use it, not just what it is,** in the section heading. The heading
  is the only routing signal the model gets — it does the job a tool `description`
  does in a tool-calling agent.

State the untrusted-data guardrail **once**, near the top, and cover every
capability with it. Repeating it per section is dead tokens on every request.

## Why single-turn, and when to break that rule

This example was previously an orchestrator that fanned out to sub-agents. Two
things pushed it back to one turn, and both are worth knowing before you fan out:

- **deepagents has no `return_direct`.** The orchestrator runs a generation _over_
  the sub-agent's reply, so a first-line contract is enforced by prompt alone. When
  it slips, a misroute and a mangled relay look identical in a score table.
- **Sub-agents are stateless.** They see only the task text. When the orchestrator
  fails to paste the material in, the sub-agent goes looking for it — and with a
  filesystem in reach it can burn an unbounded number of tokens hunting for an
  email that was never on disk.

Fan out anyway when your steps are genuinely independent work rather than
alternative phrasings of one answer, and when you want a trace span per step. If
you do, keep `tools.blocked` and `runtime.timeout_seconds` (below) — they are what
bound the damage when a delegation goes wrong.

## Blocked tools and the timeout

```yaml
tools:
  blocked:
    [ls, read_file, write_file, edit_file, delete, glob, grep, execute, task]

runtime:
  timeout_seconds: 120
```

deepagents grants its whole filesystem tool surface **by default**, plus the `task`
sub-agent-spawn tool. Leave `blocked` empty and an ambiguous prompt can become an
unbounded filesystem search. Keep this list unless your agent genuinely needs a
tool on it, and re-check it when you bump deepagents — a new built-in is granted
silently.

Two behaviours worth knowing:

- `blocked` gates at **call time** (the call returns "Tool '<name>' is blocked by
  the configured tools policy."); it does not remove the tool from the schema. The
  model may still attempt one — a bounded round-trip, not a hunt.
- A non-empty `blocked` list turns on the adapter's sub-agent gating path, which
  force-inserts a gated `general-purpose` sub-agent reachable only through `task`.
  That is why `task` is on the list. Do not drop it.

`nemo-agents-spec-v1`'s `ToolsConfig` exposes `blocked` and nothing else, and is
`extra="forbid"` — an allow-list is not expressible. `runtime.timeout_seconds` is
the only other brake: `runtime.max_turns` is ignored by the deepagents adapter.

**One knob that validates but does nothing.** `models.default.settings` is a
free-form dict on `ModelConfig`, and it reaches the adapter — but `build_chat_model`
forwards only `model`, `api_key`, `base_url`, `temperature`, `top_p` and `max_tokens`,
so nothing in `settings` is applied. Put `top_p` and `max_tokens` directly on the model
block; set `max_thinking_tokens` under `settings` and the model will happily exceed it.
This is the one place the config lets you down quietly: every other unknown key is
rejected outright, because `ModelConfig` is `extra="forbid"`.

Budget for the harness, too: deepagents prepends its own base prompt plus the
filesystem and sub-agent tool documentation to `instructions.system.content`, and
that text is sent on **every** request — including for the tools you blocked, since
`blocked` gates calls rather than trimming the schema. For a short router prompt the
harness text can be several times the size of your own. Trimming your prompt does
not reach it; only a smaller harness surface would.

## Adding a tool

This example has no tools — a single-turn router calls none, so it ships as config
plus data and is not a Python package at all. The sibling
[calculator-agent](../calculator-agent) is the worked example to copy from: it has
`mcps/calculator.py` (a pure function behind a FastMCP stdio server), a
`pyproject.toml` exposing it as a console script, and both `agent.yaml` and
`agent-with-mcp.yaml` so you can diff the with/without configs.

To add one here:

1. Copy calculator-agent's `mcps/` and `pyproject.toml`; replace the function body,
   keeping the `@mcp.tool()` wrapper and `main()`.
2. Set `pyproject.toml` `name` and `[project.scripts] <console> = "mcps.<module>:main"`.
3. Add to `agent.yaml`, and drop the tool's name from `tools.blocked` if it collides:

   ```yaml
   mcp:
     servers:
       <name>:
         transport: stdio
         url: <console>
   ```

4. Add the directory to the **root** `pyproject.toml` `members`, then
   `uv sync --all-packages` — this is what puts the console on `PATH`.
5. Tell the prompt when to call it — a tool the prompt never mentions is a tool the
   model will not use.

**Know the failure mode before you do this.** For `transport: stdio`, `url` is not
an address — it is a command name, resolved through `PATH` by the OS. If that
command is not installed wherever the agent runs, the adapter cannot enumerate the
server's tools, and **the whole runtime fails to start**: every request returns 503,
including ones that never touch the tool.

It does not fail at deploy. `nemo agents deploy` reports `running` normally, because
the runtime starts lazily on the first request. So a missing console script looks
like a healthy deployment that answers nothing. That is why this agent, which is
also shipped as a NeMo Studio sample, carries no `mcp:` block: Studio cannot
guarantee the script is installed in the deploy environment.

## Keep in sync

Couplings that break silently if you change one side only:

- **Console name:** `pyproject.toml` `[project.scripts]` **must equal** `agent.yaml`
  → `mcp.servers.<name>.url`. Only applies once you add a tool.
- **Workspace member:** if you add a tool, add your directory to the **root**
  `pyproject.toml` `members`, then `uv sync --all-packages`. Without a tool the
  directory is not a package and does not belong in `members`.
- **No `tojson` in any template:** the SDK JSON-parses the rendered output of any template
  containing that substring, so a `prompt_template` stops producing a string and `{{ prompt }}`
  in the target body goes undefined. It bites judge prompts the same way. Assemble text with a
  `{% for %}` loop instead.
- **Contract and judge:** wherever a capability _has_ a first-line contract, that
  contract is duplicated in the eval's judge prompt and in its routing `string-check`.
  Change one, change the others — the judge silently scores 0 against a contract the
  agent no longer emits. A capability with no fixed shape is asserted by exclusion
  instead: `draft_warning`'s check only proves line one is _not_ another capability's
  token, so its rubric judge is what actually grades the answer.
- **Per-run fields:** the eval configs ship without `target`, `params`, or a
  resolved `dataset`. Studio injects them at submit time; a CLI run must add them
  by hand (see README Step 5). Do not commit them into the config, or Studio runs
  will carry a stale target.

Task `id`s must be unique within a config, and a task may not carry two metrics of
the same type — use one metric with several outputs instead.

## Steps

1. **Copy** this directory to `nemo-agent-config/<your-agent>/` — a working starting point.
2. **Rename** the identifiers (`agent.yaml` `name`, `telemetry.project`).
3. **Swap the brains:** rewrite `instructions.system` — routing rules plus one section per capability.
4. **Swap the data** in `dataset.jsonl` and rewrite the judge prompts to your contract.
5. **Swap the tasks** in `eval-config.task-driven.json` — its 8 tasks and their routing
   assertions are email-specific. Left alone they score your agent against capabilities it
   no longer has, which reads as a failing agent rather than a stale suite.

Re-run the [README.md](README.md) tutorial against your agent name to validate.

## Related

- **Config reference:** the `nemo-agent-config` skill (authoring + validation for `nemo-agents-spec-v1`).
- **Deploy options:** [docs/agents/deploy-agents.mdx](../../../../../docs/agents/deploy-agents.mdx).
