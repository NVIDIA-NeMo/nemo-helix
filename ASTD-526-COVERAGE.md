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
| **3. Customization** | `fetch_model_entity(ref, workspace, platform)` at submission; `model_weights_ref(model)` for the compiled download | Path 1 plus a weights-fileset check against the model's **own** workspace; the download is pinned to that workspace too | submission: all three backends; real training runs: automodel and unsloth |

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
| Inference gateway (`services/core/inference-gateway`) | 2 | **Verified** | Catalogue stays workspace-pure; shared VirtualModel routable by name (`G1`–`G4`). Also against a real vLLM deployment on GPU: a call from another workspace is served by the one `default` deployment and no second deployment appears (brief, Priority 1) |
| Auth / OPA (`services/core/auth`) | — | **Verified** | Unchanged by this PR; 19/19 authorization scenarios, `opa test` at main's baseline |
| Files (`services/core/files`) | — | **Deferred** | Filesets are not shared — ASTD-640 |

## Customization / training backends

All three go through `fetch_model_entity`, so all three inherit path 3 and the weights fix.
A second instance of the same bug was in each compiler's download step: a bare weights
fileset was passed on unqualified and resolved against the job's workspace at runtime.
Fixed through `model_weights_ref`, with unit tests on the compiled download source, and
confirmed by real training runs for automodel and unsloth (brief, Priority 2).

| Plugin | Resolution | Training | Notes |
| --- | --- | --- | --- |
| `nemo-automodel` | **Verified** | **Verified** (GPU) | Four combinations of qualified/unqualified model and fileset refs all submit; model in an unrelated workspace rejected. Found and fixed the weights-workspace bug. All four combinations then trained to completion from another workspace, weights read from `default` only |
| `nemo-unsloth` | **Verified** | **Verified** (GPU) | Three shared-model refs submit `201`; `private-llm` from an unrelated workspace rejected `422`. All four model-ref × fileset combinations trained to completion from another workspace |
| `nemo-rl` | **Verified** | Needs Kubernetes + Ray + GPU | Valid refs pass `transform` and stop at the runtime gate (*"requires platform.runtime: kubernetes"*), which is downstream of resolution. Invalid refs fail earlier in `transform` with *"Model entity not found: 'marcus/private-llm'"*. The differing stage is what makes this conclusive |

All three resolve through `fetch_model_entity`, and all three are confirmed to resolve a
shared base model. automodel and unsloth are also confirmed to **load the weights** and train;
`nemo-rl` still needs a Kubernetes + Ray cluster for that.

**Open design question:** a LoRA job registers its adapter on the base model entity, so a job
in `marcus` writes an adapter onto the shared model in `default`. See the brief's results
section.

Note for reading `nemo-rl` results: a `422` alone proves nothing there, because it cannot
compile a spec locally at all. The error text distinguishes a resolution failure
(`Failed to transform...`) from the runtime gate (`Failed to compile...`).

## Inference consumers

These reach a model through the gateway (path 2), an entity reference (path 1), or a shared
provider / VirtualModel. Exercised 2026-09-22 from a non-default workspace against
`default/openai-gpt-oss-20b` and the `default/nvidia-build` provider (auth off); `<m>` below is
that model. Every consumer that resolves by name then routes by the entity's own workspace.

| Plugin | Path | Status | Needs | Notes |
| --- | --- | --- | --- | --- |
| `nemo-guardrails` | 2 | **Verified** | Provider | `/checks` blocked/allowed with `marcus/<m>` and `default/<m>`; a guarded VirtualModel in `marcus` backed by `marcus/<m>` blocked and allowed real requests |
| `nemo-evaluator` | 1 | **Verified** | Provider | Real job in `marcus` with `ModelRef("marcus/<m>")`: generated via `default/<m>`, 2/2 exact-match; missing ref rejected at submit. `resolve_model_reference` routes by the entity's own workspace. Error text for a missing ref names only the caller's workspace |
| `nemo-data-designer` | provider | **Verified** | Provider | Resolves *providers*, not models. `validate`, `check-models` and a 3-record `preview` in `marcus` with bare `provider="nvidia-build"` (shared from `default`); a missing provider is rejected |
| `nemo-agents` | 2 | **Verified (local config)** | Provider | `agents invoke --agent-config` in `marcus` with a bare `model_name`: the injected `marcus` gateway URL resolved it; ReAct agent used its calculator tool correctly. Deployed (Fabric) agents not exercised |
| `nemo-optimization` | VM | **Verified (preflight only)** | Provider | `prepare-fileset --dry-run` model check from `marcus` found the shared VirtualModel for a bare name and warned on a missing one. No study run (needs a hand-installed Hermes harness) |
| `nemo-switchyard` | 2 | **Verified** | Provider | VirtualModels in `marcus` routing to `marcus/<m>` and to `default/<m>` both served; a missing target 404s after routing, so resolution happens post-middleware. The response `model` echoes the routed name, not where it resolved |
| `nemo-auditor` | provider | **Verified** | Provider | Real garak scan (`test.Test`) from `marcus` with target provider `{workspace: marcus, provider: nvidia-build}`; requests went to `default/provider/nvidia-build`. Locally needs `--profile gpu`, or the job runs as a subprocess and fails with *garak interpreter not found* |
| `nemo-agent-hardener` | 2 | **Not run** | Provider | Needs `agent-hardener setup`, the OpenShell CLI and a full war-game against a chosen agent. Its sandboxed agent is gateway-bound (`agent_resolver.py:131`), so it would use this path |
| `nemo-experimentalist` | 1 | **Verified (model path)** | Provider | Its only model lookup is the `default`/`fast` refs (`NEMO_DEFAULT_MODEL`/`NEMO_FAST_MODEL`); with `marcus/<m>` both resolved and completed via `default/model/<m>`. `--workspace` never resolves models; the bundled agent under test uses a raw URL. No full run. Its SKILL.md documents env vars nothing reads (`NEMO_EXPERIMENTALIST_MODELS_*`, `…_API_BASE`) |
| `nemo-insights` | — | Not applicable | — | Telemetry analysis; no model entity resolution |
| `nemo-anonymizer`, `nemo-safe-synthesizer` | 1 | Inherits (untested) | Varies | Consume datasets more than models; datasets are ASTD-640 |
| `nemo-deployments` | — | **Not applicable by design** | — | Manages `Deployment` / `DeploymentConfig` / `Volume`, not model entities. `model_deployment` is deliberately **not** shareable: the GPU saving comes from *not* deploying a second copy elsewhere. A workspace routes to the global deployment through the gateway instead |

## Studio — known gaps, out of scope for this PR

**No Studio changes are made in this PR.** Studio has no awareness of shared entities; these
are logged for a follow-up. Only the first row was reproduced in a running Studio; the rest
come from a code read on 2026-09-22 (paths under `web/packages/`). "Likely" marks a failure
inferred from the code but not confirmed against it. Examples are a `marcus` user working
with a model shared from `default`.

| # | Where | Problem | Failure |
| --- | --- | --- | --- |
| S1 | `studio/src/hooks/useModelChatAvailability.ts` | For any model with `base_model` it checks the **base's** deployment, not the model's own `model_providers`; and fetches the base with `useModelsGetModel(model.workspace, model.base_model)`, passing a qualified ref as a name | **Reproduced:** `marcus/p2-us-merged` with a `READY` deployment in `marcus` shows *Chat Unavailable*; the base lookup `404`s. Also affects same-workspace full-SFT models whose base is undeployed |
| S2 | `studio/src/components/dataViews/CustomModelsDataView/index.tsx:241` | Adapter delete uses the route workspace, never `adapter.workspace` | A `default` user can delete `marcus`'s adapter on the shared base; `marcus` never sees it to delete |
| S3 | `CustomModelsDataView/index.tsx:158,177-189` | Adapter rows built from `model.adapters` of the current workspace's models | Adapters `marcus` trained on a `default` base never appear in `marcus`; they appear under the base in `default` |
| S4 | `CustomModelsDataView/index.tsx:185`; `studio/src/components/ModelChatPanel/index.tsx:76`; `ModelSelectV2/ModelDropdownItem.tsx:88,178,182`; `studio/src/hooks/evaluation/useEvaluationModels.tsx:47` | Adapters keyed by name only, ignoring `adapter.workspace` | Two workspaces each with adapter `foo` on the shared base: the wrong one is chatted with, evaluated, or shown |
| S5 | `studio/src/routes/CustomizationJobDetailsRoute/index.tsx:84,179` | Output model looked up as `workspace/output_model` | A LoRA job in `marcus` wrote its adapter to the `default` base, so the lookup `404`s and "Chat with your Model" never works (likely) |
| S6 | `studio/src/util/evaluations.ts:75-85` | Provider ref reduced to its last segment; adapter sent as a bare name | Evaluating a `marcus` adapter sends the wrong served name (`{ws}--{name}` expected) — likely `404` |
| S7 | `studio/src/components/evaluation/SubmitEvaluationModal.tsx:597-626`, `useJudgeModels.tsx:40` | Judge validity checked against the current workspace's models only | A metric whose judge is a shared `default` model is flagged invalid and forced to an override |
| S8 | `studio/src/routes/WorkspaceBaseModelsRoute/index.tsx:215,227` | Opening a `default` model from `marcus` writes a bare name into a `marcus` URL | Reload or a shared link looks up `marcus/<name>` and `404`s (likely) |
| S9 | `studio/src/components/sidePanels/ModelPanels/ModelPanel/components.tsx:259,273` | `base_model` shown as a raw string; job link built with `model.workspace` | No link to a cross-workspace base; job link points at the wrong workspace (likely) |
| S10 | `studio/src/hooks/useCustomizationJobForModel/index.ts:30` | Jobs searched in the route workspace only | A `default` model/adapter produced by a `marcus` job does not link to its job |
| S11 | `CustomModelsDataView/index.tsx:67-69`; `studio/src/components/FilterFields/SearchBaseModels.tsx:32-37` | Base-model filter compares `name` against `workspace/name`; filter lists one workspace | Filtering by base model never matches; only current-workspace bases offered |
| S12 | `studio/src/routes/ModelCompareRoute/index.tsx:38,71`; `ModelColumnSelect.tsx:28` | `?model=` preselect ignored unless in the current workspace | `?model=default/<name>` is silently dropped |
| S13 | `studio/src/components/NewCustomizationForm/ModelSelectionSection.tsx:31` | Base-model picker lists the current workspace only | A shared base cannot be picked for customization (reachable only via the `?model=` link) |
| S14 | `JudgeModelSelect.tsx:62`, `useEvaluationModels.tsx:65`, `ModelChatPanel/index.tsx:159`, `ModelConfigPanel/index.tsx:136`, `AddModelPalette/index.tsx:40`, `InsightsModelPairFields.tsx:54,66`, `AnalysisConfigPanel.tsx:168,187`, `GuardrailConfigurationPanel.tsx:48`, `CreateGuardrailModal/index.tsx:51`, `CreateExampleAgentModal/index.tsx:68`, `CloneAgentModal/index.tsx:47`, `MetricRunSidePanel/index.tsx:165`, `DescribeWithAiPanel.tsx:30`, `NewDeploymentRoute/WorkspaceSourceFields.tsx:112` | Every other model picker lists the current workspace only | Shared models cannot be selected anywhere except by typing a reference. `common/src/api/models/useModelsFromDefaultAndWorkspace.ts` exists but nothing uses it; only the Base Models page queries `default` too. (The deployment picker is arguably correct, since deployments are not shared.) |

S1, S4, S5, S6 and S8 are the PR's bug class on the Studio side — a related entity resolved in
the wrong workspace. S2–S4 depend on the adapter-placement question raised on the PR; their
fix follows from whichever option is chosen there.

## What to do on GPU, in order

**Before booking hardware — none of these need GPU:**

1. ~~Training submissions for `nemo-unsloth` and `nemo-rl`.~~ **Done** — both resolve a
   shared base model correctly; see the training-backends table.
2. ~~One real inference call per inference consumer.~~ **Done** 2026-09-22 for eight of
   nine — see the inference consumers table. `nemo-agent-hardener` is not run (needs a
   full war-game setup); optimization and experimentalist were exercised on their model
   paths only, not with a full study/run.

**On GPU:**

3. ~~The headline case: deploy once in the global workspace, invoke from another.~~
   **Done** 2026-09-22 on a GB300 — see the brief's results section.
4. ~~One real training run per backend.~~ **Done for automodel and unsloth** (all four
   combinations each). `nemo-rl` still needs a Kubernetes + Ray cluster.

## Honest summary

The sharing mechanism is verified at the three chokepoints, and everything else inherits it
by construction. That is a real basis for confidence, but it is not the same as coverage:
the one consumer actually exercised end to end (automodel) revealed a bug that no amount of
reading the entity store would have found. Expect roughly one wiring defect per consumer
family, not per plugin.
