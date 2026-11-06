<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OpenShell credential flow E2E

`e2e/test_openshell_credentials.py` sends synthetic credentials through real Helix
Secrets, workload controllers, the credential driver and OpenShell 0.1.6. The
receiver checks the Authorization header it actually receives. The sandbox probe
keeps its original placeholder in one process throughout renewal and rotation.

The current scenarios cover auth-disabled Jobs and Deployments: requests beyond
two credential lifetimes, rotation, direct Secrets access blocked by sandbox
policy, preserved external-service rules, provider removal after secret deletion,
and no revival when the secret is recreated. A separate heartbeat over the allowed
health route distinguishes denied credential requests from a stopped probe. After
gateway removal, the test waits for a completed denied request within the credential
lifetime plus ten seconds for request completion, then recreates the secret. These
scenarios do not yet cover authenticated permission revocation, authority outages
or restart recovery.

The test expects an isolated installation with both controllers, the driver in
auth-disabled mode, and an approved injection profile for `API_TOKEN` with
`stable_placeholder: true`. OpenShell's profile client must confirm gateway support
before enabling it; raw clients check `LintProviderProfilesResponse.supports_stable_placeholder`.
This keeps the process's original reference valid across externally renewed values
while retaining credential expiry and detach behavior. Its endpoint
allows only `POST /probe`. An authored sandbox rule allows `GET /health` on the same
service; the probe checks that rule survives the Helix policy merge. `/expect`
and `/observations` belong to the test runner and must not be reachable through
sandbox policy. `receiver.py` hosts those endpoints. `probe.py` runs at
`/app/credential_probe.py` in an image containing Python 3.13,
`/tools/jobs-launcher`, curl and the configured sandbox user.
The deployment executor's `platform_egress.binaries` must include
`/usr/local/bin/python3.13`, the probe interpreter. This lets its Helix request reach
the HTTP policy so the test can distinguish a denied Secrets route from a binary
that cannot contact Helix at all.

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
  "job_profile": "credential-e2e",
  "deployment_executor": "credential-e2e",
  "provider_profile": "credential-e2e-service",
  "cache_lifetime_seconds": 30,
  "version": "0.1.6",
  "auth_enabled": false
}
```

The workspace must exist and be approved in both backends' credential settings.
Both use `renewal_mode: helix`, with controller polling and renewal intervals
shorter than the driver's configured cache lifetime. Test-local settings enable
`credential_refresh_verified` to exercise the flow; passing a test does not change
that setting in another installation.

Jobs are submitted through the public API. Deployment configs are seeded through
the typed Entities client because provider bindings are an internal field; public
deployment creation then exercises the real controller. Agent configuration
compilation and release-image packaging are outside these scenarios.
