---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-ethos
description: Captures a NeMo Platform agent Ethos as a durable artifact at agents/<name>-ethos/ETHOS.md. Validates the front matter and required markdown sections, writes the file, and uploads it to a NeMo Filesets fileset (the canonical copy). The Ethos location is fully derivable from the agent's workspace and name — this skill does not return or persist a ref. Use over generic planning skills for any NeMo Platform agent Ethos.
triggers:
  - write the ethos
  - save the design
  - capture what we agreed
  - persist the agent design
  - nemo ethos
  - write agent ethos
not-for:
  - nemo-explore (use to gather the design before writing the Ethos)
  - nemo-build-agent (use to scaffold and deploy once the Ethos is signed off)
  - nemo-skill-selection (use for dispatch when intent is unclear)
preconditions:
  - nemo_setup_complete
  - workspace_exists
  - agent_design_complete
compatibility: nemo-platform >= 0.1.0; writes one markdown file under agents/; uploads it to a NeMo Filesets fileset (the canonical copy) — local file is a write-through cache; safe under any sandbox; idempotent if user confirms overwrite.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Read, Write, Edit, Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Platform agent Ethos

Turn the answers from `nemo-explore` into a durable artifact. The Ethos is
the contract `nemo-build-agent` reads before producing the Platform-managed
`agent.yaml` or preserving an existing NAT compatibility workflow, and
the `ETHOS.md` that downstream optimization agents read as
their primary context. Without it, downstream skills have to re-ask
everything and the optimization loop has no contract for what the agent is
supposed to do or what may be changed.

The Ethos records the **intended** state of the agent, which is not the same as
the implemented state. The codebase already shows what the agent does. This file
is the only place that records what it is supposed to do, what it must never do,
and how to weigh a win on one metric against a loss on another.

## Storage model

Two copies of the Ethos exist intentionally:

* **Canonical**: a NeMo Filesets fileset named `<agent-name>-ethos` in the
  active workspace. It holds `ETHOS.md` and may also hold `agent.yaml`
  plus relative artifacts used by the executable agent package. Downstream
  services derive the relevant file ref from workspace and agent name.
* **Local cache**: `agents/<name>-ethos/` in the developer's working directory.
  `ETHOS.md` is the human-readable contract; `agent.yaml` is the optional
  machine-readable Platform config created by `nemo-agent-config` during the
  build path.

The Fileset wins on conflict. If a developer edits the local file, this
skill re-uploads to refresh the Fileset. If the platform copy has drifted
ahead (e.g. the refinement-mode skill updated it server-side), pull it
down before editing.

**The Ethos location is by convention, not by reference.** Given an
agent's workspace and name, the remote file ref is always
`<workspace>/<agent-name>-ethos#ETHOS.md`, mirrored locally at
`agents/<agent-name>-ethos/ETHOS.md`. The `Agent` entity does
**not** carry an `ethos_file_ref` field — downstream consumers compute the
ref from `(workspace, agent_name)` via
`nemo_agents_plugin.entities.ethos_file_ref`.

## Schema version

Write `schema_version: 1`. Every canonical body section is required. The parser
rejects a file missing any of those headings. When you have nothing to say,
write `_(none)_` rather than dropping the section.

The schema is a floor, not a ceiling. Extra `##` headings and extra YAML
front-matter keys are allowed. The parser keeps unknown body sections and
does not fail on unknown front-matter keys. If the user already added custom
sections, preserve them on rewrite. Do not strip custom content to make the
file look strict.

Do not invent content to look complete: `_(none)_` in `Constraints` is honest,
while a fabricated bound actively misleads the optimizer. If the user has no
answer yet, write `_(none)_`, record the gap in `Open Questions`, and move on.

Ethos holds durable intent, so keep run-scoped configuration out of it. A spend
ceiling, an experiment count, or a wall-clock limit for a single optimization
run belongs to the tool that runs it. If a user offers one, record the standing
policy it implies — a production cost ceiling in `Constraints`, or who approves
an overrun — and leave the run limit itself to the optimizer's own config.

## Hard preconditions

Before writing anything, the answers carried over from `nemo-explore` must
satisfy one non-negotiable. If it is missing or ambiguous, **stop and route back
to `nemo-explore` for that field only** — do not invent a default.

1. **Role** — one concrete sentence describing the role this agent plays. Vague
   answers ("help with stuff", "answer questions") make the artifact useless
   downstream even though the parser will accept them; push back in
   conversation rather than writing a placeholder.

The parser cannot catch a vague `Role`, which is why this skill enforces it
upstream: the user sees a clear gap-question instead of a file that validates
and then helps nobody.

## What you do

1. **Confirm the agent name.** Lowercase, hyphens, short: `it-helpdesk`,
   `support-triage`, `code-reviewer`. If the user has not named it, propose
   two options based on the role. Must match `[a-z][a-z0-9-]*`.

2. **Pre-flight: check the local file.** If `agents/${NAME}-ethos/ETHOS.md` exists,
   ask the user whether to overwrite or pick a different name.

   ```bash
   ls "agents/${NAME}-ethos/ETHOS.md" 2>/dev/null && echo "ethos_exists" || echo "ethos_new"
   ```

3. **Pre-flight: check the Fileset.** If the canonical copy exists, surface
   it before overwriting (it may be ahead of the local file).

   ```bash
   nemo files filesets get "${NAME}-ethos" 2>/dev/null && echo "fileset_exists" || echo "fileset_new"
   ```

   If `fileset_exists` and `ethos_new`, pull the canonical copy down before
   editing:

   ```bash
   mkdir -p "agents/${NAME}-ethos"
   nemo files download "${NAME}-ethos" ETHOS.md \
     --local-path "agents/${NAME}-ethos/ETHOS.md"
   ```

4. **Run a focus check before rendering.** The carried-over answers should be
   mission-led and reviewable, not a raw inventory of implementation details:

   - `Purpose & Outcomes` and `Success Criteria` must explain mission, user
     value, the measurable result, and the success bar. If they only summarize the
     current code, route back to `nemo-explore` to ask whether the user has
     outside context that is not in the codebase. If no such context exists,
     say the section is inferred from implementation.
   - `Trade-offs` must be decidable. "Balance quality and cost" is not usable;
     a priority order with named hard gates is. If the user has not ranked
     anything, ask for the ranking rather than writing a platitude.
   - `Constraints` must be checkable. Prefer "models must come from the
     internal gateway" over "use approved models." This is also where the
     permitted model and provider set lives; there is no `Model` section,
     because the config already records the model in use and it changes without
     touching this file.
   - `Tools` and `Harness` should be concise. For `Harness`, describe how this
     agent actually runs. Do not pick a named platform harness, and do not
     treat a framework import as a requirement. Group related helpers in
     `Tools` by capability or source when they share credentials, side
     effects, freshness, and failure modes. Keep only details that change how
     downstream agents evaluate behavior.
   - Avoid public shorthand like `AUT` or "agent under test." Use "this agent"
     for the agent being specified. Use "target agent" only when this agent's
     job is explicitly to inspect or modify another agent.

5. **Render the Ethos.** Use the template at
   `references/templates/ethos.md` as the starting point. Substitute
   every section from the `nemo-explore` answers. Set front matter as:
   `schema_version` = `1`, `name` = the canonical agent name,
   `created_timestamp` = current UTC timestamp in ISO 8601 form, and `author` =
   the human or coding agent creating the file. Add `owner` when a human or
   team is accountable for the approvals named in `Constraints`.
   Set `updated_timestamp` on edits, not on first write. Evaluation commands
   live in `Evaluation Setup`, not in front matter. Keep the required section
   headers exactly so the file stays parseable. Extra `##` headings after
   (or among) the canonical fifteen are allowed — keep them. The file is
   lightly validated by `nemo_agents_plugin.ethos_parse.parse_ethos`, which
   checks front matter, schema version, required sections, and duplicate
   sections. It does not reject unknown headings. Section bodies stay markdown
   for agents and humans to read directly.

6. **Write the file.** Path: `agents/<name>-ethos/ETHOS.md`. Create the
   `agents/<name>-ethos/` directory if it does not exist.

7. **Validate before upload.** Load the file through the parser and surface any
   warnings to the user. A parse failure means the file is malformed; fix it
   before uploading, because downstream consumers will reject the same content
   server-side. Warnings are not failures — report them so the user can decide
   whether to fill the gap now.

   ```bash
   python -c "
   from pathlib import Path
   from nemo_agents_plugin.ethos_parse import parse_ethos
   ethos = parse_ethos(Path('agents/${NAME}-ethos/ETHOS.md').read_text())
   print(f'valid: name={ethos.name} version={ethos.schema_version} role={ethos.role[:60]!r}')
   for warning in ethos.warnings:
       print(f'warning: {warning}')
   " || { echo "ethos_parse_invalid"; exit 1; }
   ```

8. **Upload to Filesets (canonical copy).** Create the per-agent fileset if
   needed and upload `ETHOS.md`:

   ```bash
   nemo files filesets create "${NAME}-ethos" 2>/dev/null || true
   nemo files upload "agents/${NAME}-ethos/ETHOS.md" "${NAME}-ethos" \
     --remote-path ETHOS.md
   ```

   No ref to capture or pass downstream — the location is by convention.
   `nemo-build-agent` and downstream optimization consumers both call
   `ethos_file_ref(workspace, name)` to compute
   `<workspace>/<name>-ethos#ETHOS.md` when they need it.

9. **Show a gut-check, then the file.** Before asking the user to read fifteen
   sections, state your impression of this agent in a short paragraph that
   combines `Role`, `Purpose & Outcomes`, `Scope`, and (when they are not
   `_(none)_`) `Principles` and `Vision`. This is a thin slice so the user can
   tell quickly whether the write got the agent right. Do not use shorthand
   like `AUT` or "agent under test."

   Shape:

   > **Gut check.** This is a [role] that exists to [mission / outcome]. It
   > serves [audience] on [in-scope work] and stays out of [out of scope].
   > When the rules run out, it [principle or none]. It is heading toward
   > [vision or none].
   >
   > If that is the wrong agent, say so. Then we can edit before treating
   > this file as signed off.

   Then print the full file contents and ask: "Does this match what we
   agreed? Edit anything you want to change." If the user edits, repeat
   steps 6–9, including a fresh gut-check.

10. **Hand off.** Once confirmed, tell the user the next skill:

    - `nemo-build-agent` will read `agents/<name>-ethos/ETHOS.md`, use
      `nemo-agent-config` to produce `agent.yaml` by default, and call
      `nemo agents create`. Existing NAT workflow YAML may remain on the
      compatibility path. No `--ethos-file-ref` flag is needed because the
      Ethos location is derivable.
    - The `eval-setup` skill (M2) will fill in the `Evaluation Setup`
      section when ready.
    - The insights plugin reads the same canonical fileset server-side once
      traces exist.

## Verification

After writing and uploading, all three must hold:

```bash
# Local file present and non-empty.
test -s "agents/${NAME}-ethos/ETHOS.md" && echo "local_ok" || echo "local_missing"

# Loads through the lightweight Ethos parser.
python -c "
from pathlib import Path
from nemo_agents_plugin.ethos_parse import parse_ethos
parse_ethos(Path('agents/${NAME}-ethos/ETHOS.md').read_text())
" && echo "ethos_parse_ok" || echo "ethos_parse_invalid"

# Canonical Fileset copy is reachable.
nemo files list "${NAME}-ethos" 2>/dev/null | grep -q ETHOS.md \
  && echo "fileset_ok" || echo "fileset_missing"
```

Do not announce success until `local_ok`, `ethos_parse_ok`, **and** `fileset_ok`
all print, the gut-check has been shown, and the user has confirmed the
contents.

## If verification fails

| Symptom | Cause | Recovery |
|---|---|---|
| `local_missing` after write | Wrong working directory or permission denied | Run `pwd`; check the user is in the cloned repo |
| `ethos_parse_invalid` | Ethos malformed — missing front matter, missing required section, duplicate section, or bad schema version | Read the parser error; fix the named section in place; do not silently work around |
| `fileset_missing` after upload | Files service down or auth missing | Check `nemo workspaces list`; if that fails, the platform is unreachable — re-upload after `nemo-status` clears |
| User says "this is wrong" | Ethos captured the wrong answers | Edit the relevant section in place; re-validate; re-upload |
| Name validation keeps failing | User keeps proposing names with underscores or capitals | Pin the regex `[a-z][a-z0-9-]*` and show one example that passes |
| `nemo-explore` was skipped | User invoked `nemo-ethos` cold | Route back to `nemo-explore` and return here when the conversation is done |

## What this skill is not

This skill does not produce `agent.yaml`, migrate NAT workflow YAML, or create
the `Agent` entity. The Ethos is the human-readable design. Machine-readable
config authoring belongs to `nemo-agent-config`, while registration and
deployment belong to `nemo-build-agent`.

It also does not encode the optimizer's decision policy. `Trade-offs` records
the developer's intent — hard gates, priority order, unacceptable regressions —
in prose. Turning that into thresholds, weights, and selection strategy is the
optimizer's job, not this file's.

## Gotchas

- **The template is the source of truth for the canonical outline.** Keep the
  required section headings intact. Extra `##` headings are allowed and must
  be preserved. The parser in `nemo_agents_plugin.ethos_parse`
  rejects missing or duplicate required sections, but it does not reject
  custom headings. Section bodies remain markdown for humans and agents to
  read directly.
- **Ethos lives next to the implementation config.** Keep `ETHOS.md`,
  Platform `agent.yaml`, and their relative artifacts under
  `agents/<name>-ethos/` so local and Filesets consumers share one package root.
- **The Fileset is canonical, not the local file.** If the two disagree,
  the Fileset wins. Re-pull before editing if you suspect server-side
  drift.
- **The Ethos location is convention, not configuration.** Always
  `<workspace>/<agent-name>-ethos#ETHOS.md`. Do not introduce a flag,
  env var, or persisted field to override it — if the layout needs to
  change, update `ethos_file_ref` in
  `nemo_agents_plugin.entities` and every consumer follows.
- **Names with underscores or capitals break tools.** Validate against
  `[a-z][a-z0-9-]*`.
- **Role is a hard requirement.** Do not write the Ethos without a concrete
  one. Route back to `nemo-explore` for that field only.
- **Honest empty answers belong in the section.** Write `_(none)_` for
  `Constraints` or `Trade-offs` when the user has no answer. Do not invent a
  bound. Record the gap in `Open Questions` as well.
- **`Purpose & Outcomes` cannot be implementation-only by accident.** If goal
  context was not found in the codebase and the user did not provide outside
  context, make that provenance clear instead of letting implementation details
  masquerade as mission. A mission with no stated outcome cannot be optimized.
- **Keep public terminology clean.** The generated Ethos is user-facing. Avoid
  `AUT` and "agent under test"; reserve internal shorthand for test harnesses
  and code comments.
- **Do not duplicate Insights into the Ethos.** Known issues / recurring
  failure patterns live in the Insights plugin as first-class entities; the
  Ethos has no `Known Issues` section, and no `Signals` section either — how a
  given consumer reads evidence is that consumer's configuration, not durable
  intent. Record what a metric cannot support in `Metric Semantics`, and what
  should not count as a failure in `Behavior`.
- **This file is the `ETHOS.md`.** Downstream optimization agents should
  not edit it; only the developer and the developer's coding agent do. Treat it
  as a long-lived contract, not a scratch pad.
