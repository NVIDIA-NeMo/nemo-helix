<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- markdownlint-disable MD041 -->
## Summary

Upgrade Helix 0.10 to Fabric 0.6 and Relay 0.11, and expose HTTP OTLP
allowlisting through `telemetry.insecure_http_hosts`. Hermes remains isolated
on Relay 0.10, as required by Fabric's Hermes adapter.

## Changes

- Upgrade Fabric dependencies, lockfiles, wheel constraints, Studio Assistant,
  examples, and third-party metadata. Pin the main runtime to Relay 0.11.0.
- Forward `telemetry.insecure_http_hosts` to Relay observability. Omission preserves
  inherited allowances; explicit `[]` clears them. Relay enforces exact host matching;
  HTTPS requires no allowlist.
- Update Hermes packaging and developer installation to Fabric's tested source
  revision with Python 3.14 and Relay 0.10. Use upstream dependency overrides
  and reject unexpected dependency conflicts. Reject HTTP allowlisting for Hermes
  only when telemetry is enabled.
- Reuse Hardener's Dockerfile parser and binary-policy helper to pass Hermes's
  interpreter and Relay config paths into the sandbox, honoring environment overrides
  during initialization and refresh. Require explicit paths when they cannot be inferred.
  The Hermes victim example requires published Helix 0.10 so registered builds
  do not depend on local wheels.
- Registered Hermes agents copy only the three Hermes runtime settings from the image;
  arbitrary Dockerfile environment values are not imported into their manifests.
- Explicit project environment settings replace other image values while retaining
  the Hermes runtime paths used by the binary policy.
- Require Relay CLI 0.11 for Claude and Codex and update its lock entry and export.
- Update the Pi descriptor, adapter, SDK peers, Relay CLI, and extension.
- Align Harbor with Fabric 0.6's supported 0.24 range and remove the workspace override.
- Adjust Compass compatibility tests for Relay 0.11's preserved callback exceptions.
- Remove redundant Fabric version assertions and reuse the configured version where needed.
- Document the Helix 0.10 upgrade, HTTP collector configuration, and image rebuild requirement.

## Type of Change

- [x] Code change with documentation updates

## Quality Gates

- [x] Tests added or updated for changed behavior
- [x] Documentation updated for user-visible behavior

## Verification

- [ ] Pull request title follows the repository's Conventional Commit format
- [x] Commit includes an appropriate `Signed-off-by:` trailer
- [x] Full pre-commit checks pass with the required uv 0.13.4
- [x] Targeted tests pass
- [x] No secrets, API keys, or credentials are included

Upgrade validation:

- 3,906 agent, evaluation, Compass, and wheel-metadata regression tests passed;
  11 skipped (optional tooling and slow tests).
- Three real Hermes invocation, streaming, and session tests passed with isolated Relay 0.10.
- Two real Deep Agents HTTP/HTTPS OTLP export tests passed.
- A packaged Docker agent running Fabric 0.6.0 and Relay 0.11.0 exported three
  spans over allowlisted HTTP. Removing the allowlist rejected the endpoint.
- The Helix wheel built successfully, and its service dependencies resolved
  with the committed constraints. Both lockfiles passed checks with uv 0.13.4.
- Ruff, formatting, type checks, OpenAPI generation, MDX validation, and documentation
  link checks passed. Link checks reported two existing placeholder URL warnings.
- Full hooks passed via `uvx --from uv==0.13.4 uv run --no-sync pre-commit run -a`.

Hardener follow-up validation:

- All 859 Hardener, Fabric translator, agent packaging, and wrapper tests passed.
  Coverage includes Hermes paths, environment overrides, image credential omission,
  disabled telemetry, and Relay CLI packaging.
- Ruff, formatting, type checks, applicable hooks, and OpenAPI generation passed;
  no generated API changes were needed.
- Verified registration's text-only bundle and the derived Hermes environment and
  binary policy against the previously built image with image ENV cleared.
- Relay CLI 0.11 started with the HTTP allowlist configuration rejected by 0.9.3.
- TruffleHog 3.95.3 found no secrets in the current PR files after removing
  synthetic credential URLs from the test fixtures.
- The release-based Hermes example has not been rebuilt; it requires published
  Helix 0.10. A full OpenShell hardening run was not performed.
