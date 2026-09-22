<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Authentik Reference Example

This directory contains a local Authentik-backed NeMo Helix example. Use it
to validate three user-visible flows:

- log in to NeMo Helix with Authentik
- call NeMo Helix APIs through the Authentik gateway
- run a NeMo Helix job whose workload exchanges a managed workload proof token for a
  delegated NeMo Helix access token

All credentials in this example are for local development only.

## Tutorial

Use the shared tutorial when you want to understand or manually debug the
reference deployment:

- [Authentik Reference Tutorial](tutorial.md)

At the start, choose Docker Compose or Kubernetes. The rest of the tutorial uses
the same CLI and workload commands for both runtimes. Generated local material
lives under `.generated/`, including the workload-token signing key and gateway
TLS material when those files are created locally.

Runtime details live separately:

- [Compose Implementation Details](compose/implementation-details.md)
- [Kubernetes Implementation Details](kubernetes/implementation-details.md)

## Test Harness

`run.sh` is the automation entrypoint for CI-style validation and repeatable
local test runs. It can start durable Compose or Kubernetes auth-idp
environments, run the Compose or Kubernetes auth-idp contract tests, and clean
up local resources.

```bash
contrib/auth/authentik/run.sh --help
```

Common commands:

```bash
contrib/auth/authentik/run.sh up compose
contrib/auth/authentik/run.sh up k8s
contrib/auth/authentik/run.sh test compose
contrib/auth/authentik/run.sh test k8s
contrib/auth/authentik/run.sh prepare-local
contrib/auth/authentik/run.sh down compose
contrib/auth/authentik/run.sh down k8s
contrib/auth/authentik/run.sh clean
```

`up compose` and `up k8s` also create user NeMo CLI contexts named
`authentik-compose` and `authentik-k8s`. The contexts include the
local gateway URL, default workspace, and gateway certificate authority so users
can switch to them with `nemo config use-context`.

Use `--key KEY` with `up` to run a second durable instance. The key derives
managed names such as `authentik-compose-KEY`, `authentik-k8s-KEY`,
`authentik-e2e-KEY`, and `nhx-authentik-KEY`, and keyed `up` commands choose
an available local gateway port by default. Use the same key with `down` to
remove that instance:

```bash
contrib/auth/authentik/run.sh up compose --key dev
contrib/auth/authentik/run.sh down compose --key dev
```

`up k8s` uses `https://127.0.0.1:18082` by default for a stable manual URL.
`test k8s` chooses an available local port by default so it can run while other
local k8s auth-idp workflows are using that stable port. Set
`NHX_AUTHENTIK_K8S_GATEWAY_PORT` to force a specific test port. The `test`
actions do not add user NeMo CLI contexts.

The harness keeps generated local inputs in `contrib/auth/authentik/.generated`
so Compose and Kubernetes test runs can reuse the same workload-token signing
key. Durable `up` actions record lifecycle state under
`contrib/auth/authentik/.generated/instances` by default so `down` and `clean`
can remove recorded instances later. Override that path with
`NEMO_AUTHENTIK_STATE_DIR`. Diagnostics are written under
`docker/logs/authentik-*` by default, or under `E2E_SERVICES_LOG_DIR` when that
environment variable is set.

For manual startup and walkthroughs, prefer the shared tutorial above.

## Demo Identities

Both runtimes seed these local-only identities:

- Human user: `nemo-user`
- Human password: `nemo-user-password-dev`
- Human email: `nemo-user@example.com`
- CLI OIDC client: `nemo-helix-cli`
- Workload identity: `svc-nemo`
- Workload group: `nemo-workloads`

The shared blueprint also seeds `nemo-setup` with an app-password for the
automated auth-idp E2E test harness. That identity is not used by the
browser/device login flow or by managed workloads. Do not copy `nemo-setup` or
`e2e_setup_password_grant` into production deployments; production users should
authenticate interactively through OIDC, and production workloads should use a
workload identity/token-exchange mechanism instead of an app-password.

## Token Lifetimes

The Authentik OAuth provider lifetimes are fixed in the shared checked-in
blueprint at `helm/files/blueprints/nemo.yaml`: CLI access tokens use
`minutes=2`, and workload access tokens use `minutes=5`. Kubernetes projected
service account token expiration defaults to `600` seconds in the jobs backend.
Override it through the deployment's normal NeMo Helix configuration if you
need a longer projected token lifetime.

The 2-minute CLI access-token lifetime is a local demo/testing setting so token
refresh is easy to observe. Do not use it as a production default; use a longer
value such as `hours=1` outside the refresh demonstration.

In the Docker Compose runtime, Authentik idP authenticates users and controller
service principals, but managed Docker job OBO uses a NeMo Helix-owned
opaque workload proof token. The Docker backend writes that proof token into the
job token file, the SDK posts it to the NeMo Helix auth service, and the
gateway trusts the NeMo Helix auth service JWKS for exchanged workload access
tokens. Docker OBO does not depend on IdP `jti` claims or IdP-issued workload
subject tokens.
