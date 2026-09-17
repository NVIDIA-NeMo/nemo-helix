<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sandbox seam (AALGO-321)

A provider-neutral sandbox contract for running agent-eval harnesses **inside a container**,
injecting context and retrieving artifacts across the boundary. Built for
[`FabricAgentRuntime(..., sandbox=provider)`](../fabric/_sandbox_execution.py) but usable by any runtime.

## Why an owned seam (not `nemo_gym.sandbox` directly)

`nemo_gym.sandbox` ships the same *shape* (exec + programmatic file I/O + async/sync facades), and
this seam deliberately mirrors it so a Gym backend could be adapted later. We do **not** depend on
the package because: it requires Python ≥3.12 (nemo-platform is 3.11) and pulls `ray`/`wandb`/`mlflow`;
importing it monkeypatches builtin `print` and mutates `sys.path`/HF env; and neither shipped Gym
backend (Apptainer, OpenSandbox) matches nemo-platform's Docker-local / Kubernetes-scale target — so
we write the providers ourselves regardless. See AALGO-321 for the full analysis.

## The contract

- [`base.py`](base.py) — `SandboxSpec`, `SandboxResources`, `SandboxExecResult`, `SandboxHandle`,
  and the `SandboxProvider` Protocol. File transfer is programmatic (`upload_*`/`download_*`), not
  mount-based, so it crosses a **remote** API boundary (`docker cp` today → `kubectl cp` next), which
  bind mounts cannot.
- [`api.py`](api.py) — `AsyncSandbox` (what runtimes use) and a thin sync `Sandbox`. Drives
  `create → seed files → exec → transfer → close`; tears a half-created sandbox down on seed failure.
- [`providers/docker.py`](providers/docker.py) — `DockerSandboxProvider`: one persistent container
  per sandbox (`docker run -d` keep-alive), `docker exec`, `docker cp`, `docker rm -f`. Single `_run`
  chokepoint (mocked in unit tests).
- [`providers/compose.py`](providers/compose.py) — `DockerComposeSandboxProvider`: one exclusive,
  caller-described Docker Compose project. It accepts ordered Compose files, profiles, an exact service
  topology, and an optional teardown hook. It runs existing images by default (`build=False`);
  source builds must be requested explicitly by the worker that owns the provisioned workspace.
  Project exclusivity uses a nonblocking POSIX `fcntl` lock; unsupported platforms fail before
  startup rather than running without cross-process ownership protection.

## Compose transfer contract

Compose file and directory uploads deliberately have different ownership boundaries:

- `upload_file` requires an exact, non-root file destination. Missing parents are created as the
  target service user. If that user cannot create them, the upload fails; the provider never changes
  the ownership or permission bits of a file parent or ancestor. Copying can still change the
  parent's entries and timestamps. After copying, the provider verifies a regular non-symlink leaf,
  assigns only that leaf to the runtime identity, and adds its owner-write bit.
- `upload_dir` requires a dedicated, non-root target that is not itself a symlink. The provider
  merges the source contents into that target and recursively assigns the complete resulting tree to
  the runtime identity. Do not point it at a shared or externally owned tree.

Uploads are non-atomic. If copying succeeds and ownership or mode repair fails, the copied object may
remain; retrying the same source and destination is supported. For either operation, the caller must
keep the exact destination and its complete ancestor chain stable for the duration of the upload.
Stable caller-supplied symlink ancestors use normal POSIX path resolution. These APIs are not a
security boundary against concurrent path mutation by the in-container workload.

The Docker and Compose providers converge only on creating missing file parents without privilege.
Compose still normalizes container destinations and performs its leaf ownership and mode repair.

## Isolation note

The Docker provider does **not** default to `--network none`: the agent harness needs egress to reach
its model endpoint. `network` is a provider option. Endpoint-scoped egress control (allow the model
API, deny the rest) is future work for a policy-capable backend (e.g. NVIDIA OpenShell), not this
provider.

## Roadmap

Docker (local, here) → agent-sandbox / k8s-sigs (remote scale; `Sandbox` CRD + Python SDK) →
NVIDIA OpenShell (once it exposes a programmatic file-I/O API; CLI/SSH-only today).

## Tests

- `tests/agent_eval/test_sandbox_docker_provider.py` — hermetic; asserts the exact `docker` argv.
- `tests/agent_eval/test_sandbox_api.py` — facade lifecycle over a fake provider.
- `tests/agent_eval/test_sandbox_docker_provider_live.py` — real `docker`; skipped without a daemon.
- `tests/agent_eval/test_sandbox_compose_contracts.py` — public API and configuration contracts.
- `tests/agent_eval/test_sandbox_compose_cli.py` — image-first policy and Compose command execution.
- `tests/agent_eval/test_sandbox_compose_inspection.py` — rendered-config, readiness, and port checks.
- `tests/agent_eval/test_sandbox_compose_lifecycle.py` — session, lock, teardown, and cleanup behavior.
- `tests/agent_eval/test_sandbox_compose_transfer.py` — command execution and file-transfer behavior.
- `tests/agent_eval/test_sandbox_compose_vendored_import.py` — standalone vendored import compatibility.

- `tests/agent_eval/test_sandbox_compose_provider_live.py` — real image-first/build/profile Compose
  flows; skipped without a Compose-capable daemon.

- `tests/agent_eval/test_fabric_container_runtime.py` — `FabricAgentRuntime` sandbox mode: evidence-contract
  mapping over a fake provider.
