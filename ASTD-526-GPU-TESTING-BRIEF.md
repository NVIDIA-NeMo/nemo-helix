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

## Results so far (2026-09-22, local platform on a GB300, auth disabled)

All verified by running them. Auth was off, so nothing here tests the entitlement gate.

**Priority 1 — passed.** Qwen3-1.7B deployed once on vLLM in `default`; a chat completion
from another workspace, model id `<that-workspace>/<model>`, returned `200` and was served by
the `default` deployment. Deployments stayed at 1 in `default` and 0 in the caller's
workspace, GPU processes and vLLM containers stayed at 1, and the vLLM log shows exactly the
two requests sent (a same-workspace control plus the cross-workspace call). A nonexistent
model from the same workspace returned `404`. The shared model stayed out of the caller's
model list and gateway catalogue.

**Priority 2 — passed for automodel and unsloth; `nemo-rl` not run** (it needs Kubernetes +
Ray; the local platform runs jobs on Docker). All eight jobs — both backends, all four cells
of the matrix below — ran from another workspace and completed. Each compiled download
pointed at `default`; the platform log shows every weight-file read (96) against `default`
and none against the caller's workspace; and every job on a backend ended at the same loss,
consistent with all of them training from the same real weights.

**Open design question found in Priority 2.** A LoRA job registers its adapter on the base
model entity (`tasks/model_entity/run.py`, `create_model_adapter(workspace=base_me.workspace)`),
so a job in `marcus` writes an adapter onto the shared model in `default`, with the adapter's
weights in `marcus`. Consequences, none yet tested: with auth on, a caller who can read but
not write `default` likely fails at the final step after training; adapter names from
different workspaces collide, and on conflict the task *updates* the existing adapter, so one
workspace can silently replace another's; adapter names and fileset refs are visible to every
workspace that sees the shared model; and a `lora_enabled` deployment in `default` hot-loads
weights from `marcus`. With auth off that **works**: the adapter sidecar read each adapter's
fileset from `marcus` (`200`) and vLLM served it. Whether the sidecar may do so with auth on
is untested; filesets are not shared (ASTD-640), so expect a denial there.

**Customized models actually work.** On prompts outside the training set, all four
adapters on the `default` LoRA deployment answered in the trained `a + b = c.` format,
correctly (including 3-digit sums not seen in training), where the base model worked
step by step; general answers were unchanged. The bare-fileset model's adapters were not
chatted with (the GPU fits one vLLM deployment).

**Customized model owned by `marcus`, end to end.** A LoRA adapter cannot live in `marcus`
today (see above), but a merged checkpoint can: unsloth with `save_method: merged_16bit`,
run from `marcus` against the shared base, registered `marcus/p2-us-merged` with its
weights in `marcus` and `base_model: default/qwen3-p1-180943`, and added nothing to
`default`. Deployed in `marcus` with vLLM, it is listed in `marcus`'s gateway catalogue and
serves chat through `marcus`'s gateway; nothing is deployed in `default`. Its fine-tuning
shows (`43 + 76` switches to the trained format) but less strongly than the adapters.

**Studio cannot chat with that model.** It shows *"Chat Unavailable — This model does not
have an active deployment"* although the deployment is `READY`.
`web/packages/studio/src/hooks/useModelChatAvailability.ts` has two faults: for any model
with a `base_model` it checks the **base** model's deployment, not the model's own — wrong
for merged and full-SFT models, and not specific to sharing; and it fetches the base with
`useModelsGetModel(model.workspace, model.base_model)`, passing the qualified ref
`default/qwen3-p1-180943` as a name in the model's own workspace, which returns `404`. The
second is this PR's bug class on the Studio side. Chat works through the gateway directly.

**Priority 3 — passed for eight of nine consumers; no sharing defect found.** All run from a
non-default workspace against `default/openai-gpt-oss-20b` and the `default/nvidia-build`
provider (NVIDIA Build credential, no GPU), each with a missing-name negative control. Real
calls for switchyard, guardrails, evaluator, data designer, auditor, and agents (local
config); optimization and experimentalist on their model-resolution paths only (a
`prepare-fileset` preflight, and one completion through the Experimentalist's own resolver),
because a full study/run adds nothing for sharing. `nemo-agent-hardener` was not run: it needs
`agent-hardener setup`, the OpenShell CLI and a war-game against a chosen agent. Note that
data designer and auditor resolve **providers**, and optimization **VirtualModels** — both
shareable types — not model entities. Every consumer that resolves by name routes by the
entity's own workspace. Per-plugin detail is in the coverage doc. Smaller findings:

- The response `model` field is not a reliable resolution signal: through switchyard it
  echoes the routed name (`marcus/<m>`), while a direct call returns `default/<m>`.
- The evaluator's missing-`ModelRef` error names only the caller's workspace, not `default`.
- The Experimentalist SKILL.md documents `NEMO_EXPERIMENTALIST_MODELS_SMART/MID/FAST`,
  `NEMO_EXPERIMENTALIST_API_BASE` and `INFERENCE_API_KEY`, which no code reads. It actually
  uses `NEMO_DEFAULT_MODEL` / `NEMO_FAST_MODEL` (two tiers, `workspace/name` form).

**Local setup that is not in SETUP.md** — each cost a failed run:

1. Build `my-registry/nmp-api:local` (the deployment weights puller runs in it) and, for
   training, `nmp-customizer-tasks`, `nmp-automodel-training-docker` and
   `nmp-unsloth-training` via `make docker-load TARGET=<target>
   DOCKER_PLATFORMS=linux/arm64`. The last three need `USE_LOCAL_WHEELS=1`, or they try to
   pull prebuilt CUDA wheel images from the placeholder `my-registry`. Budget ~150 GB for the
   images and roughly as much again in build cache.
2. Pull the engine image yourself (`docker pull vllm/vllm-openai:v0.22.1`, 38 GB): the local
   config sets `pull_images: false`.
3. Start the platform with `--host 0.0.0.0`. On the default `127.0.0.1`, containers get
   `Connection refused` from `host.docker.internal:8080`. With auth off, that exposes the
   platform to the network while it runs.
4. **Do not trust the deployment puller's exit code.** When it cannot reach the platform it
   prints `✓ Downloaded`, exits `0`, and leaves `/model-store` empty; vLLM then fails with
   `Invalid repository ID or local directory specified: '/model-store'`. Read its log.
5. **Auditor jobs need `--profile gpu` locally.** Local dev translates `cpu/default` steps
   to subprocesses, so `nemo auditor audit submit` runs on the host and fails with *garak
   interpreter not found*. `--profile gpu` selects the Docker-backed `cpu/gpu` profile and
   the `my-registry/auditor-tasks:local` image (build target `auditor-tasks-docker`).
6. **Keep a local `.python-version` at 3.12.** The Makefile and Flox use 3.12; a stray
   3.13 pin makes plain `uv run` (e.g. the copyright pre-commit hook) rebuild `.venv` for
   3.13, deleting files out from under a running platform — guardrails then failed with
   `FileNotFoundError: …/nemoguardrails/rails/llm/llm_flows.co`.

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

Run each backend from a workspace that is **not** `default`, and cover both dimensions:

| | model's fileset qualified (`default/llama-3-weights`) | model's fileset bare (`llama-3-weights`) |
| --- | --- | --- |
| job model ref qualified (`default/llama-3`) | | |
| job model ref bare (`llama-3`) | | |

The dimension that matters is the **fileset on the base model**, not the job's model
reference: both defects so far only appeared when the model entity carried a bare fileset
name. A matrix that only varies the model reference, with a qualified fileset throughout,
passes whether or not the bug is there.

A job that gets past submission can still fail in the `model-and-dataset-download` step
before training starts. If it fails there, the download source in the compiled job spec
shows which workspace it tried.

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

This defect has been found twice, both times with a model in `default` carrying a bare
fileset name:

1. **At submission.** `fetch_model_entity` resolved the model out of `default`, then checked
   its weights fileset in the *caller's* workspace.
2. **At runtime.** The job compilers copied the bare fileset name into the download step
   unchanged, and the file_io task fills in a missing workspace with the *job's* workspace.
   Submission passed, then the download failed before training started. This affected the
   automodel (student and teacher), unsloth and rl compilers, and data-designer retrieval
   mining.

Both are fixed. The runtime sites now go through `model_weights_ref(model)` in
`nmp.customization_common.service.platform_client`, which qualifies the fileset with the
model's own workspace. Unit tests cover the download source for every site, but none has
run a real download yet; Priority 2 is what exercises that.

The pattern will recur. Anywhere code resolves an entity and then fetches something
belonging to it, check which workspace that second lookup uses. It must be the entity's
own (`model.workspace`), not the reference's and not the job's. A bare name passed on to a
later step is the same bug in disguise, because the later step fills in its own workspace.

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
