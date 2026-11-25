<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nhx-sandbox

> **Work in progress.** This package is the first slice of a shared sandbox library. It currently holds the egress-policy code and the OpenSandbox Harbor environment, and its API may change without notice as code is added. `sandboxed_gym` still carries its own copy of an OpenSandbox integration; eventually, we'll standardize on a common base implementation.

Shared sandbox semantics for NeMo Helix components that create sandboxes.

## The backend protocol

One interface for every sandbox backend, so Harbor environments and Gym brokers can run on any of them. So far only the local test backend implements it. OpenSandbox and OpenShell come next, and the interface also fits a managed sandbox service.

- `nhx_sandbox.core.backend`: `SandboxBackend` (check, create, get, list, delete, exec, files and directories), plus `EgressControl` for changing a running sandbox's egress. Callers describe what they want in a `SandboxSpec`, and each backend declares what it supports, including how strongly it isolates workloads, in `Capabilities`. `check` reports why a task won't fit before anything is created, and backends raise `UnsupportedOperation` rather than quietly doing less. A task that accepts the platform's own network rules says so with `PlatformEgress`.
- `nhx_sandbox.core.errors`: the errors backends raise in place of their SDKs' own, and how a runner should treat each one: retry the call (`TransientBackendError`), retry the trial (`SandboxLost`, `ExecInterrupted`) or refuse the task (`UnsupportedOperation`).
- `nhx_sandbox.core.archive`: the gzipped tar format directories move in, with safe extraction.
- `nhx_sandbox.conformance`: `run_suite` checks a live backend against the interface's promises, including that egress is denied by default. It runs against the local backend in CI.
- `nhx_sandbox.backends.local:LocalProcessBackend`: sandboxes as local temporary directories. It isolates nothing and exists for tests.

## Egress and OpenSandbox

- `nhx_sandbox.egress`: provider-neutral, deny-by-default egress policy built from an allow-only list (`EgressAllowlist`, `EgressPolicy`, `build_egress_policy`).
- `nhx_sandbox.opensandbox_policy`: renders that policy into OpenSandbox's create options and verifies the policy a live sandbox reports (`to_opensandbox_policy`, `create_options_with_policy`, `verify_applied_egress`).

- `nhx_sandbox.harbor_opensandbox_environment`: `NemoOpenSandboxEnvironment`, the Harbor environment that runs trials in OpenSandbox. Harbor loads it by import path.
- `nhx_sandbox.harbor_opensandbox_driver`: the Harbor-free OpenSandbox SDK wrapper the environment drives.
- `nhx_sandbox.harbor_opensandbox_cleanup`: sandbox cleanup, also run as `python -m` inside a Harbor runner venv.
- `nhx_sandbox.harbor_opensandbox_services`: the Docker Compose services model, shared with the OpenSandbox server extension.

The base install depends only on Pydantic, so the package can be installed into third-party runner environments such as Harbor's. Only `harbor_opensandbox_environment` imports Harbor (and `httpx`) at module level, and the OpenSandbox SDK is imported lazily, so import the submodules you need rather than expecting them all to load everywhere.
