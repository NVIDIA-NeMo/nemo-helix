---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: agents-secure
description: >-
  Audits a deployed NeMo agent for safety and security risks: missing
  content-safety guardrails on the model endpoint, PII exposure, and leaked
  API keys / secrets in recent telemetry. Persists snapshots and structured
  suggestions to the NeMo files service for diffing and UI rendering. Use when
  the user asks to secure, harden, audit, or review the safety of an agent,
  when the user mentions guardrails, content safety, jailbreaks, PII, leaked
  secrets, leaked API keys, sensitive data leaks, telemetry redaction, Safe
  Synthesizer, Anonymizer, GLiNER, or NemoGuard.
  Trigger keywords - secure agent, harden agent, agent safety, agent security,
  guardrails, content safety, PII, data safety, secrets scan, leaked api key,
  leaked token, telemetry scan, redact telemetry, Safe Synthesizer, Anonymizer,
  gliner, nemoguard.
allowed-tools: Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Agent Security

Audit a deployed agent for guardrails coverage and PII exposure in telemetry,
and produce actionable suggestions the user can apply.

## Workflow

1. **Pick an agent.** Run `nemo agents list` and prompt the user to choose one
   to audit. Use the rest of this skill on that single agent.
2. **Run the analysis steps** below — guardrails (config inspection),
   PII + secret regex scan (telemetry sample, via the bundled script), and
   optionally a deep-scan recommendation.
3. **Persist** snapshot + JSONL suggestions (see Persistence below).
4. **Surface suggestions** to the user and let them pick which to apply.

## Analysis steps

### 1. Guardrails on the model endpoint

Inspect the selected agent's config (from `nemo agents list`). Every configured
LLM should route through a `base_url` that contains `/guardrails/` (a
guardrails virtual-model endpoint). If any LLM does **not**, suggest creating
a guardrailed virtual model and pointing the agent at it.

Relevant catalog models (verify availability with `nemo models list`):

- `nvidia-llama-3-1-nemoguard-8b-content-safety`
- `nvidia-llama-3-1-nemoguard-8b-topic-control`
- `nvidia-llama-3-1-nemotron-safety-guard-8b-v3`

**Use the `inference` skill** for the exact
`nemo virtual-models create … --request-middleware …` syntax that wires
guardrail middleware in front of the chat model. Pure config inspection — no
telemetry download required.

### 2. PII + secret regex scan on recent telemetry

Sample recent telemetry files for the selected agent since the last run.
**Cap downloaded data at 1 GB total** — pick the most recent files until the
quota is hit; never download everything. Telemetry typically lives in the
`nemo-agent-telemetry` fileset (this is the same fileset the optimizer reads
for PII analysis).

Run the bundled scanner — **`resources/pii_scan.py` is the source of truth**
for patterns, false-positive guards, and masking. Don't re-implement the
regexes inline.

```bash
python3 <skill_dir>/resources/pii_scan.py <local-sample-dir-or-file>
# or stream:
nemo files download nemo-agent-telemetry --remote-path <file> -o - \
  | python3 <skill_dir>/resources/pii_scan.py -
```

Categories the script covers (anchored / prefix-led to keep FP rate low):

- **PII**: email, SSN, phone, credit card (Luhn-validated).
- **Cloud / cluster credentials**: AWS access key id, Google API key.
- **Source-control / CI tokens**: GitHub PATs (classic + fine-grained),
  GitHub OAuth tokens, GitLab PATs.
- **Model-provider API keys**: OpenAI (`sk-` / `sk-proj-`), Anthropic
  (`sk-ant-`), Hugging Face (`hf_`), NVIDIA (`nvapi-`).
- **Messaging / payments**: Slack tokens, Stripe secret keys.
- **Generic**: JWT, `-----BEGIN … PRIVATE KEY-----` blocks.

The script emits a single JSON object on stdout:

```json
{
  "scanned_files": ["…"],
  "scanned_bytes": 12345,
  "counts_by_type": {"email": 3, "openai_api_key": 1, …},
  "hits": [
    {"type": "email", "masked_preview": "j***@e****.com",
     "file": "…", "line": 42, "column": 17, "context": "…"}
  ]
}
```

False-positive guards are baked in (±80-char context check that drops
SSN-shaped digit runs near `request_id`/`build_id`, phone-shaped runs in
ISO timestamps or 10-digit Unix epochs, and credit cards that fail Luhn).
**Trust the script's verdict** — don't second-guess individual hits unless
you spot a category-level miss; if you do, fix the regex / guard in
`pii_scan.py` and re-run, don't paper over it in the agent message.

For each hit, emit one `data_safety` suggestion containing:

- The `masked_preview` (already masked by the script).
- The file + line/column it came from.
- Suggested follow-up: **Safe Synthesizer** (regenerate trace data without
  PII) and/or the Anonymizer service (redact in-place); for leaked secrets
  also flag **rotate the credential immediately** as the first action.

If `python3` is unavailable in the runtime environment, fall back to the
inline anchored regexes for email / SSN / phone / Luhn-validated credit
card and skip the secret scan rather than ship loose secret patterns.

### 3. Deep-scan recommendation (informational)

After the regex pass, inform the user that for higher recall they can run
**GLiNER** or **NemoGuard** content-safety models across the data — much
slower but catches PII and unsafe content the regexes miss. Suggest
selecting a subset of recent data first to keep wallclock reasonable.

Example invocation against a content-safety model (only emit if the model
exists in `nemo models list`):

```bash
nemo inference gateway model post v1/chat/completions <vm-name> \
  --workspace <ws> \
  --body '{"model":"<workspace>/nvidia-llama-3-1-nemoguard-8b-content-safety","messages":[{"role":"user","content":"<sample telemetry text>"}],"max_tokens":8,"temperature":0}'
```

This is informational — emit it as `suggested_actions` text rather than an
auto-apply step.

## Persistence (every run)

Uses fileset `nemo-agent-security`. This is **separate** from the optimizer
fileset (`nemo-agent-optimizer`) so the two skills can run independently
without one overwriting the other's snapshot or merging foreign suggestions.

| Path | Contents |
|------|----------|
| `security_snapshot.json` | Per-agent snapshot map; one entry per agent ever audited |
| `security_suggestions.jsonl` | One JSON object per line; survives applies |

### Snapshot shape (per-agent, single file)

The snapshot is **one JSON file at the fileset root**, keyed by agent name so
running the skill against agent A never overwrites agent B's snapshot:

```json
{
  "agents": {
    "support-bot": {
      "guardrailsState": {
        "guarded": false,
        "unguardedModels": ["nvidia-llama-3-3-70b-instruct"]
      },
      "piiCountsByType": {"email": 3, "openai_api_key": 1},
      "deepScanRunAt": null,
      "updatedAt": "2026-05-07T12:34:56Z"
    },
    "triage-bot": {
      "guardrailsState": {"guarded": true, "unguardedModels": []},
      "piiCountsByType": {},
      "deepScanRunAt": "2026-05-06T09:11:02Z",
      "updatedAt": "2026-05-06T09:11:02Z"
    }
  }
}
```

The legacy flat shape (the previous "single-agent" snapshot) is **read-only
backwards-compatible** — if you load a file without the top-level `agents`
key, treat the whole object as the snapshot for the *current* selected agent
and migrate to the keyed shape on the next save. Never write the legacy shape
back out.

### Load → update → save flow

1. Try to load the prior snapshot
   (`nemo files download nemo-agent-security --remote-path security_snapshot.json -o …`).
   - 404 → no snapshot exists yet for the workspace; treat as first run.
   - Snapshot exists but `agents[<selected>]` is missing → first run *for that
     agent*. Other agents' entries are preserved as-is on save.
   - Snapshot exists and `agents[<selected>]` is present → use that entry to
     decide whether to suppress already-actioned suggestions (e.g. drop the
     `deep_scan_recommended` tile if `deepScanRunAt` is set).
2. After analysis, **load → update only `agents[<selected>]` → save** the
   whole file. Never overwrite blindly with just the current agent's data —
   that would erase every other agent's snapshot.
3. **Merge** the suggestions JSONL (do not overwrite blindly):
   1. Load the existing file.
   2. Compute fresh suggestions for the selected agent.
   3. Preserve every record where `applied === true` as-is.
   4. Preserve every non-applied record whose `agent` field is **not** the
      currently selected agent — those belong to other agents' runs and must
      not be dropped here. (Workspace-scoped suggestions with no `agent`
      field are owned by the most recent run; replace them.)
   5. Drop any fresh suggestion whose identity tuple `(type, agent, model)`
      matches an applied record.
   6. Replace the selected agent's non-applied records with fresh output.
4. On first run the fileset may not exist — create it with
   `nemo files filesets create nemo-agent-security`.

### JSONL record shape

Same shape as `agents-optimize` so a downstream UI can ingest both
streams. One JSON object per line.

Required: `type`, `title`, `detail`. Optional: `agent`, `model`, `severity`,
`suggested_actions`, `apply`, `apply_description`, `applied`, `applied_at`.
Keep this backwards compatible — the object may be ingested downstream.

`detail` describes WHY the suggestion was raised. `apply_description` (if
`apply` is set) describes what pressing the Apply button will do.

| `type` value | Meaning |
|--------------|---------|
| `guardrails` | Agent's model endpoint lacks content-safety guardrails |
| `data_safety` | Real PII or leaked secret detected in a telemetry sample |
| `deep_scan_recommended` | Suggest running GLiNER / NemoGuard for higher recall |

`severity` (optional) — use `low` / `medium` / `high`. PII hits in raw trace
content default to `high`; leaked API keys / private keys / JWTs default to
`high` with a "rotate immediately" line in `suggested_actions`; deep-scan
suggestions default to `low`.

### Optional `apply` block (one-click action)

Most security suggestions today are **textual** — they point the user at the
`inference` skill (to wire guardrails) or at Safe Synthesizer
(to regenerate telemetry). Emit `suggested_actions` with the literal CLI
commands the user should run.

If a future apply allowlist supports virtual-model creation
(`POST /apis/virtual-models/v1/workspaces/<ws>/...`) or guardrails-config
creation, an `apply` block with the **same shape and validation rules** as
the optimizer can be emitted. The current allowlist (in
`web/packages/studio/src/routes/AgentOptimizationsRoute/api.ts`) only covers
agent / deployment / evaluate-job actions, so omit `apply` for guardrails and
PII suggestions until the allowlist is extended.

If you do emit an `apply` block, follow the optimizer rules verbatim — see
the `agents-optimize` skill for the full method/path/origin/workspace/
allowlist contract. Cross-skill drift in the apply contract WILL cause
silent rejections at apply time.

## CLI reference (verified)

```bash
# List agents and models
nemo agents list
nemo models list --all-pages
nemo models list --filter.name nemoguard

# Telemetry inspection (for PII sample)
nemo files list nemo-agent-telemetry
nemo files download nemo-agent-telemetry \
  --remote-path <largest-recent-file> -o /tmp/sample.jsonl

# Files service for security persistence
nemo files filesets create nemo-agent-security
nemo files upload <path> nemo-agent-security \
  --remote-path security_suggestions.jsonl

# Calling a content-safety model through the gateway (deep scan)
nemo inference gateway model post v1/chat/completions <vm-name> \
  --workspace <ws> --body '<json>'
```

## What requires execution vs. what can be reasoned

**Reason directly from already-fetched data — no extra commands needed:**

- **Guardrails coverage**: inspect the agent's `config.llms[*].base_url` and
  `model_name` from `nemo agents list`. If no `/guardrails/` endpoint is in
  use, emit a `guardrails` suggestion. No telemetry download required.
- **Available guardrail/safety models**: read off `nemo models list` and pick
  the appropriate NemoGuard variant for the suggestion.

**Requires execution:**

- **PII + secret regex scan**: download a sample telemetry file (the largest
  recent one, capped at the 1 GB budget) and run
  `python3 <skill_dir>/resources/pii_scan.py <path>`. Parse the resulting
  JSON and emit one `data_safety` suggestion per hit. The script handles
  ±80-char FP guards, Luhn validation, and masking — don't re-implement.
- **Deep scan** (only if the user opts in): call a content-safety / GLiNER
  model through the inference gateway on a subset of recent traces.

## Execution checklist

- Run `nemo agents list`; prompt the user to pick one agent.
- Try to load prior snapshot
  (`nemo files download nemo-agent-security --remote-path security_snapshot.json -o …`).
  404 (or `agents[<selected>]` absent) → first run for this agent. Other
  agents' entries in the file are preserved on save.
- Inspect the selected agent's config — emit a `guardrails` suggestion if no
  `/guardrails/` endpoint is in use.
- If a telemetry fileset exists, download a sample (cap total ≤ 1 GB) and run
  `python3 <skill_dir>/resources/pii_scan.py <sample-path>`. Emit one
  `data_safety` suggestion per hit in the script's JSON output (the script
  has already applied FP guards and masking). Tag leaked-secret hits as
  `severity: high` with "rotate immediately" in `suggested_actions`.
- Always emit a `deep_scan_recommended` suggestion (informational) so the
  user knows GLiNER / NemoGuard exists as a more thorough fallback. Drop it
  if `agents[<selected>].deepScanRunAt` shows the user already actioned a
  deep-scan run.
- Create `nemo-agent-security` fileset if it doesn't exist
  (`nemo files filesets create nemo-agent-security`).
- Update only `agents[<selected>]` in the snapshot map and upload the whole
  file (`nemo files upload <path> nemo-agent-security --remote-path security_snapshot.json`).
  Do not write a flat single-agent shape — that erases every other agent.
- Merge + upload suggestions JSONL
  (`nemo files upload <path> nemo-agent-security --remote-path security_suggestions.jsonl`).
- Summarize key suggestions for the user. Let them pick which to apply.

## Related skills

- `agents-optimize` — performance / cost / quality optimization. Use
  alongside this skill when the user wants both a perf-and-safety pass.
  Both skills emit JSONL with the same record shape.
- `inference` — exact `nemo virtual-models create … --request-middleware …`
  syntax for wiring guardrail middleware in front of a chat model.
- `nemo-guardrails` — CLI reference for guardrails configs and self-check
  rails.
- `nemo-auditor` — for jailbreak / red-team scanning of a candidate model
  before promotion.
