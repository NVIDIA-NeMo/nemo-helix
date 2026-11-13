<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OpenShell credential flow E2E

`e2e/test_openshell_credentials.py` sends synthetic credentials through real Helix Secrets, workload controllers, the credential driver and OpenShell 0.1.6. The receiver checks the Authorization header it receives. The sandbox probe keeps its original placeholder in one process throughout renewal and rotation.

The scenarios exercise auth-disabled Jobs and Deployments: uninterrupted requests beyond two credential lifetimes, rotation, blocked direct Secrets access, preserved external-service rules, provider removal after secret deletion, and continued denial when the secret is recreated. A heartbeat over the allowed health route confirms the original probe remains active. Before recreating the secret, the test requires a completed denied request within the credential lifetime plus ten seconds for request completion.

The test requires an isolated installation with both controllers, the driver in auth-disabled mode, and an approved injection profile for `API_TOKEN` with `stable_placeholder: true`. OpenShell's profile client must confirm gateway support before enabling it; raw clients check `LintProviderProfilesResponse.supports_stable_placeholder`. This keeps the process's original reference valid across externally renewed values while retaining credential expiry and detach behavior.

The profile endpoint allows only `POST /probe`. An authored sandbox rule allows `GET /health` on the same service; the probe checks that rule survives the Helix policy merge. `/expect` and `/observations` belong to the test runner and must not be reachable through sandbox policy.

`receiver.py` hosts those endpoints outside the sandbox. It runs in Helix's Python environment, which includes FastAPI and Uvicorn; a separate receiver image needs those dependencies too. For the example configuration below, start it with `uv run python e2e/resources/openshell_credentials/receiver.py --port 18081`. The optional `--cert-file` and `--key-file` arguments enable TLS together.

`probe.py` runs at `/app/credential_probe.py` in an image containing Python 3.13, `/tools/jobs-launcher`, curl and the configured sandbox user. The deployment executor's `platform_egress.binaries` must include `/usr/local/bin/python3.13`, the probe interpreter. This lets its Helix request reach the HTTP policy so the test can distinguish a denied Secrets route from a binary that cannot contact Helix at all.

When the probe runs as a job it also saves a result from inside the sandbox: it uploads the artifact into the job's fileset and registers the result record, using the storage context the backend injects. The test requires that save to succeed and the platform to record the result, which keeps the result route of the restricted platform policy under test alongside the credential checks.

Point the existing E2E client at that installation and provide its test settings:

```sh
export NHX_BASE_URL=http://127.0.0.1:18080
export NHX_OPENSHELL_CREDENTIAL_TEST_CONFIG=/absolute/path/to/test-config.json
uv run --frozen pytest e2e/test_openshell_credentials.py --run-e2e -v
```

```json
{
  "workspace": "credential-e2e",
  "image": "local/helix-credential-e2e-probe:main",
  "receiver_url": "http://127.0.0.1:18081",
  "sandbox_receiver_url": "http://host.openshell.internal:18081/probe",
  "platform_sandbox_url": "http://host.docker.internal:18080",
  "gateway_endpoint": "127.0.0.1:17670",
  "gateway_workspace": "default",
  "owned_secrets_workspace": "openshell-internal",
  "job_profile": "credential-e2e",
  "deployment_executor": "credential-e2e",
  "provider_profile": "credential-e2e-service",
  "cache_lifetime_seconds": 30,
  "version": "0.1.6",
  "auth_enabled": false
}
```

The workspace must exist and be approved in both backends' credential settings. Both require `renewal_mode: helix` and `credential_refresh_verified: true` in the isolated test installation, with controller polling and renewal intervals shorter than the driver's configured cache lifetime.

Jobs are submitted through the public API. Deployment configs are seeded through the typed Entities client because provider bindings are an internal field; public deployment creation then exercises the real controller.
