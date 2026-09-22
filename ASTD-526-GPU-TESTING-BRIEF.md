<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ASTD-526 GPU verification brief

For whoever picks up the hardware-dependent testing of PR #2243, on branch
`astd-526-global-workspace-shared-entities/mschwab`. Read `ASTD-526-COVERAGE.md` first —
it is the map of what is already verified and what you are picking up.

## What the change does

A model living in the `default` workspace can be **referenced by name from another
workspace**, so it is deployed once instead of per-workspace. Specifics that shape your
tests:

- Resolution is **gated on the caller being entitled to `default`**. A user scoped only to
  their own workspace resolves nothing from it — that is intended, not a bug.
- **Listings stay workspace-pure.** A shared model does NOT appear in
  `GET /workspaces/{ws}/models` or in the gateway catalogue. You must reference it by name.
  If you expect to see it in a list, you will wrongly conclude it is broken.
- A response can have `workspace: "default"` while the path said `marcus`. That is correct —
  it is how you tell a resolved shared entity from a local one.

## Priority 1 — the headline case (needs GPU)

Deploy a model **once** in `default`, then invoke it from a different workspace. Nothing has
tested gateway routing against a real GPU-backed deployment; only against a VirtualModel
entity.

1. Deploy a model in `default` and wait until its provider/deployment is READY.
2. Create workspace `marcus`; grant your principal access to **both** `default` and
   `marcus` (with auth off everything is unrestricted, so this is moot).
3. From `marcus`, send a real chat completion through the gateway:
   `POST /apis/inference-gateway/v2/workspaces/marcus/openai/-/v1/chat/completions`
   with model id `marcus/<model-name>` — the caller's workspace, not `default`.
4. Confirm a real completion comes back **and that no second deployment was created**
   (`nvidia-smi`, and the deployment list in `marcus` should be empty).

Step 4 is the actual point of the ticket. Routing working while a duplicate deployment
quietly appears would pass a naive test and fail the goal.

## Priority 2 — training actually loads weights (needs GPU)

Submission is already verified for all three backends. What is untested is whether training
then **loads the base model weights** from the global workspace.

Run one real job per backend against a base model in `default`:

- `nemo-automodel` — GPU
- `nemo-unsloth` — GPU
- `nemo-rl` — Kubernetes + Ray + GPU

Test both a **qualified** (`default/llama-3`) and a **bare** (`llama-3`) model reference.
The bare form is where the one bug so far appeared.

## Priority 3 — inference consumers (provider credential, NOT GPU)

Nine plugins reach models through the gateway and inherit the behaviour, but none has been
exercised: `nemo-switchyard`, `nemo-agent-hardener`, `nemo-guardrails`, `nemo-evaluator`,
`nemo-data-designer`, `nemo-agents`, `nemo-optimization`, `nemo-auditor`,
`nemo-experimentalist`.

One real call each, from a workspace that is **not** `default`, against a shared model.
Start with `nemo-switchyard` (routes *between* models, so most likely to carry a workspace
assumption) and `nemo-agent-hardener` (largest surface, 44 model references).

None of these needs the GPU box — they only need an inference provider credential, so they
can run anywhere.

## The bug class to watch for

The one defect found so far: `fetch_model_entity` resolved a model out of `default`, then
looked for its **weights fileset** in the *caller's* workspace. Fixed, but the pattern will
recur. Anywhere code resolves an entity and then uses a *separate* workspace variable to
fetch something belonging to it, check which workspace that second lookup uses. It must be
the entity's own (`model.workspace`), not the reference's.

Filesets are deliberately NOT shared (deferred to ASTD-640), so anything reading a fileset
cross-workspace is suspect.

## Gotchas that have already cost time

1. **Assert your setup.** Several runs reported passes while testing nothing, because
   `curl -o /dev/null` hid a failed fixture creation. Every "does not leak" assertion passes
   trivially when the fixture was never created. Check status codes on setup and abort
   loudly.
2. **`nemo-rl` returns 422 for everything locally.** It cannot compile a spec without
   Kubernetes, so the status code proves nothing — read the error text.
   `Failed to transform ... Model entity not found` means resolution failed.
   `Failed to compile ... requires platform.runtime: kubernetes` means resolution
   **succeeded** and it only hit the runtime gate.
3. **VirtualModels with `autoprovisioned: true` are deleted within seconds** by the provider
   reconciler when no provider backs them (`Deleted orphaned autoprovisioned VirtualModel`
   in the models-controller log). Use `autoprovisioned: false` for hand-made fixtures.
4. **Workspace deletion is staged, not immediate.** Reusing a fixed test workspace name
   across runs silently fails to recreate it, and everything in it 404s. Use a unique name
   per run.
5. **A killed platform can hold port 8080.** If startup reports the port in use, run
   `nemo services stop --force`, then confirm with `lsof -iTCP:8080 -sTCP:LISTEN`. Note the
   CLI also reports a sandboxed `bind()` EPERM as a port conflict, which sends you after a
   lock that is not the problem.
6. Fileset `purpose` must be one of `dataset`, `environment`, `generic`, `model`.

## Reporting

For each test: scenario, expected, actual, pass/fail. Distinguish **verified** (you ran it)
from **inherited** (you read the code path). If something fails, capture the error text and
not just the status code — in this area the text is what carries the meaning.
