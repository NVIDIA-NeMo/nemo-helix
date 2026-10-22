---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-model-selection
description: Recommends an LLM for a NeMo Helix agent
triggers:
  - which model should I use
  - what model is best for this
  - help me pick a model
  - recommend a model
  - I don't know what model to use
  - model selection
  - which LLM
not-for:
  - nemo-build-agent (use to scaffold the YAML once requirements are confirmed)
compatibility: nemo-helix >= 0.1.0; read-only; loads references/benchmark_cache.json if present; works offline; safe under any sandbox.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Read, Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Helix model selection

Recommend a model for a new agent from NIM or another provider configured on
the running Platform. Explain the capability fit first and benchmark evidence
second. Return the model choice in a form suitable for the
Platform-managed `agent.yaml`.

## Pre-flight

### 1. Load the benchmark cache

```bash
test -f packages/nemo_helix_ext/src/nemo_helix_ext/skills/nemo-model-selection/references/benchmark_cache.json && echo "cache_present" || echo "cache_missing"
```

If `cache_missing`, proceed with the static table in this file. Tell the user once that benchmark data is stale and they can refresh it with:

```bash
python scripts/refresh-benchmark-cache.py
```

The cache (schema v6+) carries four things the rest of this skill reads:
- `models[]` — editorial entries for a curated set of NIMs with `strong_at`, `watch_out_for`, `intent_hints`, `derived_from` lineage, and direct/inferred scores.
- `upstream_index.bfcl_v4` and `upstream_index.arena_elo` — full BFCL and per-category Arena Elo tables for ~84 and ~360 models respectively. Use these to look up scores for ANY model name, not just the registered ones.
- `namespace_to_type[]` — namespace-prefix → NAT `_type` mapping used only for legacy NAT workflow output.
- `name_decomposition_rules[]` — pattern→hint rules for synthesizing `intent_hints` when an unknown model name lands.

### 2. Fetch the live model list for a Platform-routed harness

```bash
nemo inference models list --output-format json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(m['id'] for m in d.get('data', []) if m.get('id')))" 2>/dev/null || echo "PLATFORM_UNREACHABLE"
```

Interpretation:
- **Model IDs returned** → these are the candidates the user can actually pick from. Each `id` is the routable VirtualModel identifier in `workspace/name` form: it is what the inference gateway's OpenAI-compatible route accepts as the request `model` and what the compatibility probe sends in its request body. Carry it through to Step 1+ unchanged. The probe's URL argument and `models.default.model` in `agent.yaml` take only the part after the `/` (see `references/harness-compat-check.md`): a deploy binds the agent to its own workspace's gateway route, which accepts a bare name or only that workspace's prefix. List models in the workspace the agent will deploy into. Do not substitute an id from `nemo models list`, which returns `ModelEntity` records whose ids are Platform entity ids and are not routable.
- `PLATFORM_UNREACHABLE` → platform isn't up. Recommendations may continue
  from the curated cache, but a model for Platform `agent.yaml` cannot be
  finalized until the live Platform model list and harness-specific inference
  route can be checked.

Use the `nemo` CLI rather than constructing a Platform URL or calling
`/v1/models` directly. The CLI resolves `nemo --base-url`, `NHX_BASE_URL`, the
active CLI context, authentication, and workspace consistently with subsequent
agent commands. Do not hardcode `localhost`, `127.0.0.1`, or port `8080`.

This list is authoritative only for models routed through Platform. For a
native-provider harness such as `claude`, use the configured provider's native
model catalog and validation tooling instead.

## Step 0 — Pick the conversation direction

Before the profile questions, ask which path the user is on:

```txt
Quick check before I ask the design questions:

A. **You're choosing a model.** Walk me through what the agent does and I'll
   recommend from what's deployed.
B. **You already have a model in mind.** Tell me which one — I'll assess
   whether it fits your task and surface what to watch for.

Which are we doing? (If unsure, A is the default.)
```

If the user picks **A** → continue to Step 1 with the recommend flow.
If the user picks **B** → continue to Step 1 with the assess flow:
- Skip the "recommend a model" framing entirely in Step 2.
- Still ask the three profile questions in Step 1 (you need them to evaluate fit).
- In Step 2, produce a fit assessment for the user's named model rather than a recommendation — same evidence-quality flagging, but the output frames as "here's what we know about <their model> for <their task>" not "use this model."

## Step 1 — Profile the agent

Ask all three questions in a single message. Skip any that the conversation has already answered.

```txt
Before I recommend a model, three quick things about what the agent will do:

1. **Tool density.** How many tools, and how do they interact?
   - One tool (search, lookup, or similar)
   - 3–5 tools whose outputs chain into each other
   - Many tools, often called in parallel

2. **Primary capability.** What does the model spend most of its time doing?
   - Calling APIs or tools reliably (MCP, structured function calls)
   - Working with code (reading repos, editing, writing patches)
   - Reading and reasoning over long documents
   - General conversation and instruction following

3. **Deployment.**
   - Cloud (NVIDIA Build API, OpenAI, Anthropic, etc.)
   - Self-hosted on a GPU — if so, roughly how much VRAM is available?
   - Not decided yet
```

Do not propose a model before all three answers are in. Push back on "you decide" — commit to a default and announce it ("I'll assume cloud and a tool-heavy agent. Tell me if that's wrong.").

## Step 1.5 — Build the candidate set and pick a presentation pattern

Identify the selected harness and whether it uses a Platform-routed or native
provider path before building candidates. Read it from the source config or
conversation; ask if it is still unknown.

### Building candidates

The candidate set is what the user can actually pick from. It comes from three joins:

1. **Start with the correct live catalog.** Use the pre-flight Platform model
   list for Platform-routed models. Use the configured provider's native model
   catalog for native-provider harnesses. If the required catalog is
   unreachable, the cache may support a conversational recommendation, but do
   not finalize a Platform `agent.yaml` model block. Availability alone does
   not establish compatibility with an agent harness.
2. **For each available model name, look up evidence** in this order:
   - Token-match against the editorial `models[]` entries → if hit, use the full editorial record (lineage, intent_hints, direct + inferred scores)
   - If no editorial match, token-match against `upstream_index.bfcl_v4` keys → if hit, use that BFCL score with `source: "direct_external"`
   - Same for `upstream_index.arena_elo` for per-category Elo
   - If neither editorial nor upstream matches, synthesize `intent_hints` by walking `name_decomposition_rules[]` and collecting every hint whose `pattern` token appears in the decomposed model id. Mark evidence as `source: "name_only"`.
3. **Rank candidates by the user's profile** — primary capability axis determines which score field dominates.

When the live Platform model list was unreachable, also tell the user the rest
of this flow is operating on the curated NIM set, not their actual deployment.

### Verify harness compatibility before config handoff

When the output targets Platform `agent.yaml`, rank a short candidate list from
the appropriate live catalog and the evidence above, then verify candidates
against the selected harness's actual model contract:

| Harness | Required model contract | Compatibility check |
|---|---|---|
| `codex` | OpenAI Responses API | Valid `v1/responses` inference request |
| `hermes` | OpenAI-compatible chat completions | Valid `v1/chat/completions` inference request |
| `deepagents` | Provider-specific; `nvidia`, `openai`, and `openai-compatible` use chat completions | Valid request for the selected provider path; use `v1/chat/completions` for an OpenAI-compatible route |
| `claude` | Native Anthropic provider | Require `provider: anthropic` and validate the configured credentials/model with native Anthropic tooling; do not route it through Platform IGW |

Before making any inference requests, show the user the candidate names and
explain that the checks make real, potentially billable model calls. Ask for
explicit confirmation and wait. The original request to select a model or
write a config is not confirmation for these calls. For native-provider
harnesses, use the provider's native validation path instead of forcing the
request through Platform IGW.

For a Platform-routed candidate, use the context-aware CLI (`nemo inference gateway model post v1/responses` for Codex, `v1/chat/completions` for Hermes or an OpenAI-compatible DeepAgents setup) with the `id` returned by `nemo inference models list`, split into workspace and name as shown in `references/harness-compat-check.md` (copy-paste commands).

A successful model list lookup, schema validation, Fabric planning, deployment
readiness, empty request, or `GET` does not establish compatibility for a
Platform-routed model. Only candidates that complete a valid request through
the required Platform model path may be returned to `nemo-agent-config`. For a
native-provider harness, require its adapter provider contract and successful
native credential/model validation instead. Exclude failed combinations and
try the next ranked candidate. If no candidate passes, stop without emitting a
model block and ask the user to configure a compatible provider or explicitly
choose a different harness. Never switch the harness silently.

Record the selected harness, provider, exact model name, required model
contract, and successful check in the handoff. Then apply the normal benchmark
ranking only among compatible candidates.

### Picking the presentation pattern

The pattern depends on the *evidence quality* of the candidate that best matches the profile:

```txt
Identify the best-matching candidate.
Inspect its evidence for the primary axis from profile question 2:
  - "code"               → scores.arena_elo.coding  AND  scores.bfcl_v4
  - "tools"              → scores.bfcl_v4
  - "long documents"     → scores.arena_elo.hard_prompts
  - "general/instruction"→ scores.arena_elo.overall AND scores.arena_elo.instruction_following

If the relevant scores all have source ∈ {"direct", "direct_external"}:
  → Pattern A (single recommendation, current Step 2 template)

If one or more relevant scores have source == "inferred_from_ancestor",
or the candidate has scores == {} (name-only) but has intent_hints:
  → Pattern B (forced trade-off, withhold model name until user resolves it)

If 3+ candidates have similar profiles and similar evidence-quality:
  → Pattern C (shortlist with explicit "I'd want an eval before deciding")
```

### Pattern B — forced trade-off

**Hard rule: when the primary candidate's evidence is anything other than `direct`, the model name does NOT appear in your response until the user has resolved the trade-off.** This is the anchoring guard. Without it, users default to the first model named regardless of caveats.

Present exactly two unnamed candidates, **A (matched specialist, with its caveat)** and **B (measured generalist, strongest *direct* score on the same axis)**, recommend a 5–10 prompt eval on both, and name a model only after the user resolves the trade-off. Never fill a slot with an incompatible model: if compatibility filtering leaves one candidate, present it unnamed with its caveat and recommend the eval before committing; if it leaves none, follow the no-candidate rule in **Verify harness compatibility before config handoff**. Full step list and an example output (code-heavy agent whose natural pick has inferred BFCL data): `references/pattern-b-example.md`.

### Pattern C — shortlist (rare)

Use only when 3+ candidates have similar profiles and similar evidence-quality, and no honest trade-off binary exists. Present them in a numbered list with `plain` summaries and **explicit phrasing that an eval is the only way to actually decide.** Do not commit to one. Default to recommending an eval as the next step.

## Step 2 — Recommend in plain English

Lead with the *capability that matters most for this agent*, then name the model as the conclusion. Use this shape (adapt the content, do not copy verbatim):

> "For an agent that <restate what the user described>, the thing that matters
> most is <one capability in plain words — e.g. 'reliably picking the right
> tool without hallucinating its arguments'>. That's what <model shortname> is
> built for — <one sentence on what the model actually does well, not what it
> scored>. The tradeoff is <honest caveat>. Given you're deploying <cloud /
> self-hosted>, **`<model string>`** is the practical choice."

### Recommendation table

| Agent profile | Recommended model |
|---|---|
| Many tools, parallel or chained calls | `qwen/qwen3-235b-a22b` |
| Many tools, cost or speed is a constraint | `qwen/qwen3-30b-a3b` |
| Code-focused: reading repos, writing patches | `qwen/qwen3-coder-30b-a3b-instruct` |
| Reasoning over long documents or many chunks | `nvidia/llama-3.1-nemotron-ultra-253b` |
| General-purpose, balanced default (platform-curated) | `nvidia/llama-3.3-nemotron-super-49b-v1` |
| General-purpose, widely-deployed alternative | `meta/llama-3.3-70b-instruct` |
| Fast and cheap, simple tool loops | `microsoft/phi-4-mini-instruct` |
| Self-hosted, single consumer GPU (<24 GB) | `qwen/qwen3-8b` |

### What each model is strong at

Read each model's `strong_at` and `watch_out_for` from `benchmark_cache.json` when present. Fallback text when the cache is missing: `references/model-strengths.md`.

## Step 3 — Mention the evidence only after the plain-English case

After explaining the recommendation, briefly note *how we know* — framed as "the test that simulates this" not "the score it got".

Frame it as the test that simulates the agent's work, never as a score. Right and wrong examples: `references/evidence-framing.md`.

If the user asks what benchmark was used or wants the raw number, tell them. Do not lead with it.

## Step 4 — Output

Show the blocks that fit the user's stage. Default machine-readable output to
Platform `agent.yaml`. **When the chosen model's primary-axis score has
`source: "inferred_from_ancestor"` or the model relies on `intent_hints` only,
include an explicit evidence caveat in the human-readable recommendation.** Do
not encode benchmark commentary as unsupported config fields.

If they are authoring Platform `agent.yaml`, emit a default model block:

```yaml
models:
  default:
    provider: <configured-provider>
    model: <model-string>   # Platform-routed: the VirtualModel name, without the workspace/ prefix
    api_key_env: <credential-env-var-if-needed>
    base_url: <provider-base-url-if-needed>
```

Use the provider identity configured on the Platform. Omit `api_key_env` and
`base_url` when the selected provider does not require user-supplied values.
Keep `base_url` directly in the model block, not under `settings`.

Before emitting this block, complete the harness-specific compatibility check
above. Do not infer compatibility from provider or model-list metadata alone.

The default model applies to every harness that does not declare its own
model. Add a harness-local override only when that harness intentionally uses
a different model or provider:

```yaml
harnesses:
  <harness-name>:
    kind: <harness-kind>
    model:
      provider: <configured-provider>
      model: <model-string>
```

Do not emit raw Fabric SDK model objects. `nemo-agent-config` owns final YAML
placement and validation.

If they are explicitly maintaining a legacy NAT workflow YAML, emit the NAT compatibility block, pick the NAT `_type` from `namespace_to_type[]`, and pair the model with a workflow type. All of that is in `references/legacy-nat-output.md`. For Platform `agent.yaml`, represent credentials with `api_key_env` and put `base_url` directly in the model block.

## Step 5 — Staleness notice

End with:

End with a staleness notice: say the recommendations reflect the last cache refresh, give `python scripts/refresh-benchmark-cache.py` to refresh, and link the raw leaderboards. Exact text: `references/staleness-notice.md`.

## Verification

This skill writes nothing. Verification is conversational: summarize the recommendation in 3 lines (capability that mattered most, model chosen, one tradeoff) and ask "Does this match what you need?" Do not hand off until the user confirms.

Use `nemo-agent-config` when the user wants to apply the selected model to `agent.yaml`.

## If verification fails

Wrong recommendation: re-ask only the misread profile question, do not restart. Model not in the table: say so and describe the capability gap versus the closest recommended model. Compatibility probe fails: exclude that combination and test the next ranked candidate; if none pass, stop without emitting a model block. Full symptom/cause/recovery table: `references/recovery.md`.

## Hard rules

- Never name a model before all three profile questions are answered.
- Never lead with a model name, benchmark name, or score.
- Never recommend a cloud-only model when the user said self-hosted.
- Never return an untested model route to a config-writing skill.
- Never make model inference calls without explicit user confirmation.
- Never silently change the selected harness to accommodate an available model.
- Never emit a model identifier without showing the plain-English reason alongside it.
- **When the primary candidate's evidence is anything other than `direct`, the model name does not appear in your response until the user has resolved the trade-off in Pattern B.** Anchoring is the failure mode this guards against — users default to the first model named regardless of caveats. The withhold is non-negotiable.
- For `agent.yaml`, present the evidence next to the YAML rather than inventing a config field.

## Gotchas

- **"You decide" needs a committed default, not a silent fill-in.** Pick something, name it, tell the user.
- **Do not transform model IDs by punctuation convention.** Use the identifier
  returned by the selected live provider or Platform model listing and pair it
  with the correct `provider`. Legacy NAT components and Data Designer may use
  different provider-specific identifiers; preserve the identifier required by
  that consumer instead of assuming the build skill converts it.
- **Watch the deployment column.** A 235B cloud-API recommendation aimed at a self-hoster with a 24 GB GPU is the most common mismatch and the easiest to catch by re-reading Step 1.
