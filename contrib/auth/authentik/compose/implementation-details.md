<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Authentik Docker Compose Implementation Details

This page explains how the Authentik Docker Compose reference deployment is
wired. For the step-by-step test flow, see the
[shared tutorial](../tutorial.md).

## Compose Structure

The Compose tutorial starts exactly one Compose file:
`contrib/auth/authentik/compose/docker-compose.yml`.

That file declares the Docker Compose project name
`nemo-helix-authentik`, while keeping Compose-specific orchestration in
`compose/`. Set `COMPOSE_PROJECT_NAME` before running `docker compose` to use a
different local namespace. The stack mounts shared Auth IdP example assets from
the parent directory:

- `../config/platform-compose-authentik.yaml` as the NeMo Helix config.
- `../gateway/envoy.yaml` as the local gateway config.
- `../helm/files/blueprints` as the NeMo Helix blueprint source.
- `../.generated` for local generated keys and certificates.

The shared tutorial does not build NeMo Helix images for Compose. It runs
`${IMAGE_REGISTRY:-my-registry}/nhx-api:${BAKE_TAG:-local}` for both the
NeMo Helix API service and workload jobs submitted by the tutorial.

## Services

The stack contains:

- `nemo`: the NeMo Helix API service configured for Authentik and the Docker
  jobs backend.
- `gateway`: Envoy terminating local HTTPS on host port `18080`.
- `gateway-tls-init`: a small init container that copies local TLS material into
  the named `gateway-tls` volume with permissions suitable for Envoy.
- `authentik-blueprint-init`: a one-shot init container that applies the shared
  NeMo Helix blueprint before the gateway starts.
- `authentik-postgres`: PostgreSQL for Authentik.
- `authentik-redis`: Redis for Authentik.
- `authentik-server` and `authentik-worker`: Authentik itself.

`nemo` is only on the internal network. Host and workload traffic reaches NeMo Helix
through the `gateway` service, which also joins the workload network as
`nemo-gateway`.

## Generated Local Inputs

The shared tutorial writes generated material under
`contrib/auth/authentik/.generated` so Compose and Kubernetes debugging can
share local keys:

- `.generated/workload-token-private-key.pem`
- `.generated/gateway-tls/tls.crt`
- `.generated/gateway-tls/tls.key`

The workload-token private key is mounted into `nemo` at
`/var/run/secrets/nemo-helix/workload-token-signing/private-key.pem`.
`platform-compose-authentik.yaml` points
`auth.token_signing.private_key_file` at that mounted path. The NeMo Helix
auth service uses the private key to sign workload-exchange access tokens and
Scoped Access Key JWTs, and Envoy validates those tokens through the
NeMo Helix auth service JWKS endpoints.

The gateway TLS files are copied into the `gateway-tls` named volume by
`gateway-tls-init`. The `gateway` service uses that volume to serve HTTPS, and
the `nemo` service mounts the same volume read-only so Python HTTP clients
inside NeMo Helix trust the demo gateway certificate.

All generated keys and certificates in this example are for local development
only.

## Authentik Blueprint

Compose applies the shared `../helm/files/blueprints/nemo.yaml` file with the
one-shot `authentik-blueprint-init` service. The gateway depends on that init
service, so `/health/gateway/ready` is checked only after the demo OIDC
providers, demo user, demo groups, workload identity, and E2E setup identity
have been seeded.

The init service retries blueprint application for up to about a minute because
Authentik can report healthy before its built-in default blueprints have created
the provider flows and OAuth scope mappings referenced by the shared blueprint.

The blueprint reads `AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD` when creating the
`svc-nemo` app-password token. Compose provides a local-development default,
`svc-nemo-token-secret-dev`. Override it only when you also keep the same value
for the lifetime of the Authentik database volume, or recreate the stack with
`docker compose down -v`.

The `nemo-setup` service account and app-password in the blueprint exist solely
for automated auth-idp contract tests. They are not part of the browser login
flow or the workload identity pattern.

## NeMo Helix Compose Configuration

`platform-compose-authentik.yaml` configures NeMo Helix for this topology:

- `platform.base_url` is `https://nemo-gateway:8080`, which is the gateway name
  visible to Docker workload containers.
- The embedded policy decision point stays on `http://127.0.0.1:8080` inside
  the `nemo` container.
- Host-side CLI login uses the port-forward-like public gateway URL
  `https://127.0.0.1:18080`.
- Authentik provides user and controller service-principal authentication. The
  Docker managed-job OBO binding is stored in NeMo Helix auth delegation
  state, not in Authentik.
- Exchanged workload access tokens come from NeMo Helix's `/apis/auth/token`
  endpoint.

The Docker jobs executor mounts the `gateway-tls` volume into workload
containers and sets `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` so workload code
trusts the local gateway certificate.

## Gateway And Auth Headers

Envoy is the public entrypoint for the Compose example. It routes:

- NeMo Helix paths such as `/.well-known/nemo-helix/`, `/apis/`,
  `/health/`, `/status`, and `/studio/` to `nemo`.
- `/health/gateway/ready` to an Envoy-owned readiness check that verifies both
  NeMo Helix and Authentik through their upstream clusters.
- Authentik paths to `authentik-server`.

Before authentication, Envoy removes incoming `X-NHX-Principal-*` and
`X-NHX-Scopes` headers so a client cannot spoof identity or scopes. For
protected `/apis/` requests, Envoy calls NeMo Helix's `/apis/auth/ext-authz`
endpoint with the presented bearer token. The auth service validates Authentik
OIDC tokens, NeMo Helix workload-exchange access tokens, and NeMo Helix
Scoped Access Keys, then returns trusted `X-NHX-Principal-*` and `X-NHX-Scopes`
headers for Envoy to forward upstream. The public `/apis/auth/authenticate`
endpoint remains a JSON diagnostic/API endpoint for direct callers and is not
the configured Envoy callout.

The gateway callout is required for dynamic or revocable Scoped Access Keys
because Envoy JWKS validation can only prove token signature, issuer, audience,
and time claims. It cannot check NeMo Helix's access-key lifecycle state. Compose
keeps `auth.access_keys.enabled=true` so Scoped Access Keys can be created and
validated; Envoy performs the bearer-to-header mapping before the request
reaches service middleware.
Scoped Access Keys are enabled in the checked-in Compose config because the Compose
test runtime advertises the `platform_access_keys` capability.

## Workload Token Exchange

For Docker jobs, the managed jobs backend owns workload identity injection. The
job request should not include `NHX_WORKLOAD_IDENTITY_TOKEN_FILE`,
`NEMO_WORKLOAD_TOKEN`, or `NEMO_WORKLOAD_TOKEN_FILE`.

When a managed Docker workload starts, the backend creates a dedicated workload
identity volume and writes a NeMo Helix-owned Docker workload proof token to:

```text
/var/run/secrets/nemo-helix/workload/token
```

It injects:

```text
NHX_WORKLOAD_IDENTITY_TOKEN_FILE=/var/run/secrets/nemo-helix/workload/token
```

The SDK reads that file and sends an RFC 8693 token exchange request to the
NeMo Helix auth service through the gateway. The Docker backend registered
an internal workload delegation row before the container started. The
NeMo Helix auth service validates the proof token, checks the matching row,
mints a NeMo Helix-signed delegated access token, and returns it to the
workload. The access token uses the captured job submitter as the top-level
subject and the Docker workload as the RFC 8693 `act.sub` actor.

Docker supports one proof-token mechanism in this flow. The file contains a
private opaque proof token whose secret is stored only as a hash in the
delegation row.

Docker job OBO therefore does not require Authentik to issue a workload token
and does not depend on an IdP `jti` claim. The Authentik workload-provider
configuration in the manifest is retained for direct provider-token contract
tests, not for the managed Docker job exchange loop.

The useful end-to-end validation is the workload job in the shared tutorial:
the job uses the exchanged token to call the NeMo Helix API and read the
workspace.

## Cleanup Behavior

`docker compose down -v` removes the Compose containers and volumes, including
the Authentik database and the named `gateway-tls` volume. It does not remove
files under `contrib/auth/authentik/.generated`. Keeping those files lets later
Compose and Kubernetes debugging reuse the same local workload-token signing
key.
