<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Community plugin runtime (experimental foundation)

## Trust model

`local-trusted` executes explicitly approved third-party code with the local
Helix OS user's privileges. It inherits the process environment, filesystem,
network access, and available credentials. A virtualenv and a separate process
provide dependency and crash isolation, **not security isolation**.

`isolated` and `kubernetes` are unsupported runtime stubs. They fail explicitly;
neither silently falls back to trusted execution. Future implementations need
not share the local runtime's artifact or process representation.

## Implemented boundary

`runtime.py` provides a POSIX backend with:

- Explicit `trust_code=True` acknowledgement before preparation.
- A local wheel copied into a private revision directory and verified against
  its SHA-256 **before** running the installer.
- One `uv` virtualenv per revision, wheel-only dependency installation, and a
  record of resolved requirements. Dependencies and host dependencies are
  resolved at installation time; this is **not** a hash-locked dependency bundle.
- A subprocess running `host.py` with that virtualenv's Python in isolated Python
  import mode (`-I`), without loading plugin entry points into the controller.
- Startup readiness polling, timeout/failure cleanup, process-group shutdown,
  explicit removal, and context-manager cleanup.
- A Unix socket in a private temporary directory, with no public TCP listener.

`host.py` is deliberately standalone: the plugin environment does not install
`nhx-platform-runner` or `nhx-common`. It loads exactly one matching
`nemo.services` entry point, checks the `NemoService` interface/name, mounts its
routers under `/apis/<name>`, and manages service lifecycle/exception handlers.

Internal endpoints:

- `GET /_helix/ready`: available after service startup.
- `GET /_helix/descriptor`: protocol version, plugin name, and runtime OpenAPI.

The descriptor is a first transport contract, not yet a validated public
operation catalog or an authorization policy contribution.

## Not implemented yet

This is **not a complete plugin installation feature** and is not wired into
`nemo plugins`. In particular:

- No registration API, remote URL downloader, upload route, HelixAdmin check,
  workspace enablement, authenticated gateway, or generic CLI/SDK bindings.
- No durable registration/controller reconciliation, restart recovery, upgrade,
  rollback, log rotation, or retained installation-failure diagnostics. The
  owning process must serialize lifecycle calls and stop children before exit;
  an abrupt owner crash can leave children requiring operator cleanup.
- No core-service dependency readiness, request-scoped typed-client injection,
  or authorization integration. Existing plugins that use those dependencies
  (including Tavily) require gateway/host integration before they work here.
- No automatic function/job route mounting, custom CLI execution, controllers,
  inference middleware, Studio code, or arbitrary SDK extension loading.
- No sandboxing, capability enforcement, secret brokerage, or egress controls.

Do not expose the socket through a public proxy until gateway authorization and
workspace/caller propagation are implemented. A private Unix socket separates
transport, not mutually hostile processes belonging to the same OS user.

## Tests

From the repository root:

```bash
uv run --frozen pytest packages/nhx_platform_runner/tests/test_community_plugin_host.py packages/nhx_platform_runner/tests/test_community_plugin_runtime.py -q
```

Tests verify artifact validation and installer commands with mocked installation,
plus real subprocess startup, Unix-socket invocation, descriptor retrieval, and
cleanup using a child-only fixture service. They do not download/install external
packages or constitute end-to-end Tavily validation.
