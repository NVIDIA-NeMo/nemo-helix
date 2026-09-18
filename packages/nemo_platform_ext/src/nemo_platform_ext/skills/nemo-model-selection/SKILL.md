---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-model-selection
description: Recommends an LLM for a NeMo Platform agent based on what the agent actually has to do, explained in plain English before any benchmark name appears. Use when the user is choosing a model for a new agent, assessing a model they already selected, or deciding what belongs in ETHOS.md or Platform agent.yaml. Invoked by nemo-explore at the model question; also runs standalone when the user starts mid-flow.
triggers:
  - which model should I use
  - what model is best for this
  - help me pick a model
  - recommend a model
  - I don't know what model to use
  - model selection
  - which LLM
not-for:
  - nemo-explore (use first to capture the agent's job, audience, and tools)
  - nemo-ethos (use to persist the design once model is chosen)
  - nemo-build-agent (use to scaffold the YAML once the Ethos is signed off)
compatibility: nemo-platform >= 0.1.0; read-only; loads references/benchmark_cache.json if present; works offline; safe under any sandbox.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Read, Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Platform model selection

Recommend a model for a new agent from NIM or another provider configured on
the running Platform. Explain the capability fit first and benchmark evidence
second. Return the model choice in a form suitable for `ETHOS.md` and the
Platform-managed `agent.yaml`. Preserve NAT model configuration only when the
user is explicitly maintaining a legacy NAT workflow.

## Pre-flight

### 1. Load the benchmark cache

```bash
test -f packages/nemo_platform_ext/src/nemo_platform_ext/skills/nemo-model-selection/references/benchmark_cache.json && echo "cache_present" || echo "cache_missing"
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
nemo models list --all-pages --output-format json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(m['name'] for m in d.get('data', []) if m.get('name')))" 2>/dev/null || echo "PLATFORM_UNREACHABLE"
```

Interpretation:
- **Model names returned** → these are the candidates the user can actually pick from. Carry their exact names through to Step 1+; the JSON `id` value is a Platform entity id and must not be written to `agent.yaml` as the inference model identifier.
- `PLATFORM_UNREACHABLE` → platform isn't up. Recommendations may continue
  from the curated cache, but a model for Platform `agent.yaml` cannot be
  finalized until the live Platform model list and harness-specific inference
  route can be checked.

Use the `nemo` CLI rather than constructing a Platform URL or calling
`/v1/models` directly. The CLI resolves `NEMO_BASE_URL`, `NMP_BASE_URL`, the
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

Ask all three questions in a single message. Skip any that the conversation has already answered (for example, `nemo-explore` already captured tools and deployment).

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

For a Platform-routed OpenAI-compatible candidate, use the context-aware CLI
rather than a hardcoded URL. Preserve the exact model name returned by
`nemo models list`:

```bash
MODEL_NAME="<exact-platform-model-name>"

# Codex
nemo inference gateway model post v1/responses "$MODEL_NAME" \
  --body "{\"model\":\"$MODEL_NAME\",\"input\":\"Reply with exactly: compatibility check\"}"

# Hermes or an OpenAI-compatible DeepAgents configuration
nemo inference gateway model post v1/chat/completions "$MODEL_NAME" \
  --body "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: compatibility check\"}]}"
```

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

Structure:

1. State that there's a trade-off to settle before naming a model.
2. Present exactly two candidates labeled A and B (NOT named):
   - **A. The matched specialist** — the candidate that best fits the profile by name/intent, with its inferred/missing-data caveat.
   - **B. The measured generalist** — the model with the strongest *direct* score on the same primary axis, even if less specialized.
3. For each, summarize the `plain` text from the cache (which already carries the editorial caveat for inferred entries) in your own words.
4. Recommend a 5–10 prompt eval from the user's spec on both as the "real answer."
5. If the user can't run an eval, ask: which framing of your priority is right — A's specialist intent or B's measured discipline?
6. **Only after the user picks** — name the chosen model and proceed to Step 2's recommendation template (skipping the model-selection paragraph since it's resolved).

Example output for a code-heavy agent where the natural pick has inferred BFCL data:

> "Before I name a model — one thing to settle. Your profile (code-focused, three tools, cloud) has two reasonable candidates with different evidence quality:
>
> **A. The specialist.** Built for this exact job. Arena's coding-category Elo (1418, strong) IS measured for this exact model. *But BFCL hasn't measured its tool-calling discipline — only its base model (Qwen3-30B-A3B at 37%) and we're inferring from there.*
>
> **B. The measured generalist.** Best directly-measured tool-calling in our NIM set (BFCL 52%, mid). Not code-specialized, but disciplined.
>
> If you can run a 5–10 prompt eval from your spec on both, that's the real answer. If you need to commit now: which matters more — specialist intent (A) or measured discipline (B)?"

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

If `benchmark_cache.json` is present, read each model's `strong_at` and `watch_out_for` from the cache — those are kept in sync with the script. The text below is the fallback when the cache is missing.

**qwen/qwen3-235b-a22b**
- Correctly calls multiple tools in a single turn without mixing up arguments
- Handles parallel tool invocation where call order matters
- Recovers gracefully when a tool returns an error instead of hallucinating a result
- Watch out for: heavy VRAM footprint for self-hosting; overkill for single-tool agents

**qwen/qwen3-30b-a3b**
- High-throughput tool calling at low inference cost (sparse MoE, ~3B active params)
- Good baseline before you know whether you need the larger model
- Watch out for: lower ceiling on complex nested tool chains

**qwen/qwen3-coder-30b-a3b-instruct**
- Navigates unfamiliar codebases and makes targeted edits
- Agents that interact with git, CI, or code-review workflows
- Watch out for: general reasoning suffers from the specialization

**nvidia/llama-3.1-nemotron-ultra-253b**
- Follows threads across very long documents without losing context
- Multi-step reasoning over dense technical material
- Watch out for: slower inference; cloud API is the practical deployment path

**nvidia/llama-3.3-nemotron-super-49b-v1**
- The platform's curated default for cloud agents — well-tested across the build path
- Balanced: solid tool-calling, solid prose, no glaring weakness
- Watch out for: a specialist will beat it on heavy tool chains or hard code tasks

**meta/llama-3.3-70b-instruct**
- Reliable across a wide range of tasks with well-characterized behavior
- Good starting point when you don't yet know where the bottleneck will be
- Watch out for: not a specialist — pick one if tool-calling or code quality is critical

**microsoft/phi-4-mini-instruct**
- Fast, cheap inference for latency-sensitive agents or high-volume loops
- Fits on small hardware for edge or resource-constrained deployments
- Watch out for: lower ceiling for complex multi-tool orchestration

**qwen/qwen3-8b**
- Tool calling on a single consumer GPU (fits in 12–16 GB VRAM)
- Best choice for local development and prototyping
- Watch out for: not competitive on complex reasoning; context reliability drops faster

## Step 3 — Mention the evidence only after the plain-English case

After explaining the recommendation, briefly note *how we know* — framed as "the test that simulates this" not "the score it got".

Right framing:
> "We know this because it consistently outperforms comparable models on tests
> that simulate exactly this kind of multi-tool coordination — not toy examples,
> but real scenarios where the model has to decide which tool to call, in what
> order, with what arguments, and when to call nothing at all."

Wrong framing (do not do this):
> "It scored 0.91 on BFCL v4, which is top tier, and 1342 Arena Elo."

If the user asks what benchmark was used or wants the raw number, tell them. Do not lead with it.

## Step 4 — Output

Show the blocks that fit the user's stage. Default machine-readable output to
Platform `agent.yaml`. **When the chosen model's primary-axis score has
`source: "inferred_from_ancestor"` or the model relies on `intent_hints` only,
include an explicit evidence caveat in the human-readable recommendation.** Do
not encode benchmark commentary as unsupported config fields.

If they're authoring an Ethos for `nemo-ethos`, do not emit a `## Model`
section. Ethos has no such heading. Return the choice to `nemo-explore` with
family/size, NIM model id, reason, evidence, and deployment mode so explore
can record permitted providers and model families in `Constraints`. The
selected model itself belongs in `agent.yaml`.

If they are authoring Platform `agent.yaml`, emit a default model block:

```yaml
models:
  default:
    provider: <configured-provider>
    model: <model-string>
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

If they are explicitly maintaining a legacy NAT workflow YAML, emit the NAT
compatibility block:

```yaml
llms:
  primary_llm:
    _type: <nat-type>
    model_name: <model-string>
    # Chosen because: <one plain-English sentence — same words you used above>
    # Evidence: <direct | inferred from <ancestor-id> | name-intent only — eval recommended>
    max_tokens: 4096

workflow:
  _type: <agent_type>
  llm_name: primary_llm
  tool_names: []
```

### Picking the right legacy NAT `_type`

Only for NAT workflow output, match the chosen model's namespace prefix against
`namespace_to_type[]` from the cache:

```txt
For each rule in cache.namespace_to_type:
  if model_string.startswith(rule.prefix):
    use rule.nat_type for the YAML _type field
    if rule.note is present, surface it once to the user
    stop
If no rule matches:
  Tell the user the namespace is unrecognized. Run this to confirm the right type:
    uv run nat info components -t llm_provider
  Until the user provides the type, leave _type as <TBD-verify> in the emitted YAML
  rather than guessing.
```

Common mappings the cache carries today: `nim/*`, `openai/*`, `anthropic/*`, `bedrock/*`, plus vendor-published NIM names (`qwen/*`, `meta/*`, `nvidia/*`, `microsoft/*`, `mistralai/*`) that route through the NIM provider when served by the platform. Ollama's local endpoint maps to `_type: openai` since it exposes an OpenAI-compatible API.

For Platform `agent.yaml`, represent credentials with `api_key_env` and put
`base_url` directly in the model block. For a legacy NAT workflow, use the
provider fields required by that NAT LLM component.

### Pair a legacy NAT model with the right workflow type

Use this table only when maintaining NAT workflow YAML. Harness selection for
`nemo-agents-spec-v1` belongs to `nemo-agent-config` and must not be inferred
from a NAT workflow type.

| What the agent needs to do | Use |
|---|---|
| Call tools in a loop, observe results, adjust | `react_agent` |
| Dispatch tools directly without intermediate reasoning | `tool_calling_agent` |
| Reason through a problem before acting | `reasoning_agent` |
| Follow a fixed plan with no dynamic re-planning | `rewoo_agent` |

## Step 5 — Staleness notice

End with:

```txt
These recommendations reflect benchmark data current as of this skill's last
cache refresh. LLM rankings shift often. To pull fresh scores:

    python scripts/refresh-benchmark-cache.py

Raw leaderboards:
  Tool calling:        https://gorilla.cs.berkeley.edu/leaderboard.html
  Human preference:    https://lmarena.ai
                       (live Elo data: huggingface.co/datasets/lmarena-ai/leaderboard-dataset)
```

## Verification

This skill writes nothing. Verification is conversational: summarize the recommendation in 3 lines (capability that mattered most, model chosen, one tradeoff) and ask "Does this match what you need?" Do not hand off until the user confirms.

If `nemo-explore` invoked this skill, return control to `nemo-explore` with the chosen model so it can continue to the constraints question. If the user invoked standalone, hand off to `nemo-ethos` if they want to persist the design.

## If verification fails

| Symptom | Cause | Recovery |
|---|---|---|
| User says "wrong recommendation" | One of the three profile answers was misread | Re-ask just that question; do not restart |
| User wants a model not in the table | The table is curated, not exhaustive | Tell them honestly; describe the capability gap their choice would have vs the closest recommended model |
| `cache_missing` and user wants fresh data | Cache has never been refreshed in this checkout | Tell them the refresh command and note that the static table is still usable |
| User picks self-hosted but the recommended model needs cloud | Hard constraint conflict | Drop the recommendation; pick the closest self-hosted-compatible model from the table |
| Available model's harness API support is unknown | Availability was mistaken for harness compatibility | Ask permission to run a valid request through the harness's required model path |
| Compatibility request fails | The exact harness/provider/model/endpoint combination is incompatible in the current environment | Exclude that combination, surface the error, and test the next ranked candidate after the user-approved probe set |
| No candidate passes | No live model satisfies the selected harness contract | Stop without emitting a model block; ask the user to configure a compatible provider or explicitly choose another harness |

## Hard rules

- Never name a model before all three profile questions are answered.
- Never lead with a model name, benchmark name, or score.
- Never recommend a cloud-only model when the user said self-hosted.
- Never return an untested model route to a config-writing skill.
- Never make model inference calls without explicit user confirmation.
- Never silently change the selected harness to accommodate an available model.
- Never emit a model identifier without showing the plain-English reason alongside it.
- **When the primary candidate's evidence is anything other than `direct`, the model name does not appear in your response until the user has resolved the trade-off in Pattern B.** Anchoring is the failure mode this guards against — users default to the first model named regardless of caveats. The withhold is non-negotiable.
- When returning a model choice for Ethos, include the evidence so
  `nemo-explore` can put it in `Constraints`. Never write a `## Model`
  heading. For `agent.yaml`, present the evidence next to the YAML rather
  than inventing a config field.

## Gotchas

- **"You decide" needs a committed default, not a silent fill-in.** Same rule as `nemo-explore`. Pick something, name it, tell the user.
- **Do not transform model IDs by punctuation convention.** Use the identifier
  returned by the selected live provider or Platform model listing and pair it
  with the correct `provider`. Legacy NAT components and Data Designer may use
  different provider-specific identifiers; preserve the identifier required by
  that consumer instead of assuming the build skill converts it.
- **Watch the deployment column.** A 235B cloud-API recommendation aimed at a self-hoster with a 24 GB GPU is the most common mismatch and the easiest to catch by re-reading Step 1.
