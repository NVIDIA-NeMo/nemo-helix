<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ASTD-526 — cross-stack coverage for shared models

Which parts of the stack can consume a model that lives in the global workspace, how each
one resolves it, what has actually been exercised, and what is blocked on GPU.

## How anything reaches a shared model

Every consumer resolves a model through one of three paths. The sharing behaviour lives in
the first two, so a consumer inherits it by using them — it does not need its own change.

| Path | Mechanism | Where the fallback lives | Verified |
| --- | --- | --- | --- |
| **1. Entity store** | `EntityClient.get(Model, workspace, name)`, directly or via the models service | `get_entity_by_name` tries the request workspace, then the global one | yes |
| **2. Inference gateway** | `POST /workspaces/{ws}/openai/-/v1/...` with model id `ws/name` | `ModelCache` / `VirtualModelCache` lookups fall back to the global workspace | yes |
| **3. Customization** | `fetch_model_entity(ref, workspace, platform)` at submission; `model_weights_ref(model)` for the compiled download | Path 1 plus a weights-fileset check against the model's **own** workspace; the download is pinned to that workspace too | submission: all three backends; download: unit tests only |

All three require the caller to be entitled to the global workspace. A caller scoped to
their own workspace resolves nothing from it.

## Status legend

- **Verified** — actually exercised against a running platform, result recorded.
- **Inherits (untested)** — the code path was read and confirmed to go through one of the
  three mechanisms above, but no run exercised it. The most likely failure mode is a
  wiring detail, exactly like the customization weights bug found in automodel.
- **Needs GPU** — cannot be exercised without GPU hardware or a Ray/Kubernetes cluster.
- **Needs provider** — no GPU required, but needs an inference provider credential
  (`NVIDIA_API_KEY` or equivalent) to make a real completion call.

## Core platform

| Component | Path | Status | Notes |
| --- | --- | --- | --- |
| Entity store (`services/core/entities`) | 1 | **Verified** | 21/21 on SQLite and PostgreSQL; resolution, shadowing, pure listings, cascade guard |
| Models service (`services/core/models`) | 1 | **Verified** | Shared model resolves via the caller's workspace (`R7`) |
| Inference gateway (`services/core/inference-gateway`) | 2 | **Verified** | Catalogue stays workspace-pure; shared VirtualModel routable by name (`G1`–`G4`) |
| Auth / OPA (`services/core/auth`) | — | **Verified** | Unchanged by this PR; 19/19 authorization scenarios, `opa test` at main's baseline |
| Files (`services/core/files`) | — | **Deferred** | Filesets are not shared — ASTD-640 |

## Customization / training backends

All three go through `fetch_model_entity`, so all three inherit path 3 and the weights fix.
A second instance of the same bug was in each compiler's download step: a bare weights
fileset was passed on unqualified and resolved against the job's workspace at runtime.
Fixed through `model_weights_ref`, with unit tests on the compiled download source; no real
download has run yet.

| Plugin | Resolution | Training | Notes |
| --- | --- | --- | --- |
| `nemo-automodel` | **Verified** | Needs GPU | Four combinations of qualified/unqualified model and fileset refs all submit; model in an unrelated workspace rejected. Found and fixed the weights-workspace bug |
| `nemo-unsloth` | **Verified** | Needs GPU | Three shared-model refs submit `201`; `private-llm` from an unrelated workspace rejected `422` |
| `nemo-rl` | **Verified** | Needs Kubernetes + Ray + GPU | Valid refs pass `transform` and stop at the runtime gate (*"requires platform.runtime: kubernetes"*), which is downstream of resolution. Invalid refs fail earlier in `transform` with *"Model entity not found: 'marcus/private-llm'"*. The differing stage is what makes this conclusive |

All three resolve through `fetch_model_entity`, and all three are now confirmed to resolve a
shared base model. What is untested for each is whether training then **loads the weights** —
that needs hardware.

Note for reading `nemo-rl` results: a `422` alone proves nothing there, because it cannot
compile a spec locally at all. The error text distinguishes a resolution failure
(`Failed to transform...`) from the runtime gate (`Failed to compile...`).

## Inference consumers

These reach a model through the gateway (path 2) or an entity reference (path 1). None has
been exercised against a shared model.

| Plugin | Path | Status | GPU | Notes |
| --- | --- | --- | --- | --- |
| `nemo-guardrails` | 2 | Inherits (untested) | Needs provider | Guardrail configs reference a model; rails run on the gateway |
| `nemo-evaluator` | 2 | Inherits (untested) | Needs provider | Target model for a run; LLM-as-judge model too |
| `nemo-data-designer` | 1, 2 | Inherits (untested) | Needs provider | Generation model reference |
| `nemo-agents` | 2 | Inherits (untested) | Needs provider | Agent model via `utils.py:240` gateway URL |
| `nemo-optimization` | 2 | Inherits (untested) | Needs provider | Compares models, so several refs per run |
| `nemo-switchyard` | 2 | Inherits (untested) | Needs provider | Routing middleware — routes *between* models, highest chance of a workspace assumption |
| `nemo-auditor` | 2 | Inherits (untested) | Needs provider | Audit target points at a model |
| `nemo-agent-hardener` | 2 | Inherits (untested) | Needs provider | 44 model/gateway references, the largest surface |
| `nemo-experimentalist` | 2 | Inherits (untested) | Needs provider | Candidate models per experiment |
| `nemo-insights` | — | Not applicable | — | Telemetry analysis; no model entity resolution |
| `nemo-anonymizer`, `nemo-safe-synthesizer` | 1 | Inherits (untested) | Varies | Consume datasets more than models; datasets are ASTD-640 |
| `nemo-deployments` | — | **Not applicable by design** | — | Manages `Deployment` / `DeploymentConfig` / `Volume`, not model entities. `model_deployment` is deliberately **not** shareable: the GPU saving comes from *not* deploying a second copy elsewhere. A workspace routes to the global deployment through the gateway instead |

## Studio

| Area | Status | Notes |
| --- | --- | --- |
| Model pickers, customization UI, agent config | **Not started** | Studio has no awareness of shared entities. Since listings stay workspace-pure, a shared model will not appear in any picker — a user can only reach one by typing a qualified reference. This is the main gap between "works" and "usable" |

## What to do on GPU, in order

**Before booking hardware — none of these need GPU:**

1. ~~Training submissions for `nemo-unsloth` and `nemo-rl`.~~ **Done** — both resolve a
   shared base model correctly; see the training-backends table.
2. **One real inference call per inference consumer** against a shared model, using an
   external provider credential rather than a local GPU deployment. Start with
   `nemo-switchyard` (routes *between* models, so most likely to carry a workspace
   assumption) and `nemo-agent-hardener` (largest surface, 44 references).

**On GPU:**

3. **The headline case: deploy a model once in the global workspace, then invoke it from a
   different workspace.** This is what the ticket is for. The gateway routing is verified
   against a VirtualModel, but never against one backed by a real GPU deployment.
4. **One real training run per backend** against a base model in the global workspace.
   Confirms weights actually load, not just that the reference resolves.

## Honest summary

The sharing mechanism is verified at the three chokepoints, and everything else inherits it
by construction. That is a real basis for confidence, but it is not the same as coverage:
the one consumer actually exercised end to end (automodel) revealed a bug that no amount of
reading the entity store would have found. Expect roughly one wiring defect per consumer
family, not per plugin.
