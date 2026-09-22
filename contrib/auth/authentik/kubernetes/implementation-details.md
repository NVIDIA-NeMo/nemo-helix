<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Authentik Kubernetes Implementation Details

This page explains how the Authentik Kubernetes reference deployment is wired.
For the step-by-step test flow, see the [shared tutorial](../tutorial.md).

## Chart Structure

Install exactly one chart for this example:
`contrib/auth/authentik/helm`. That umbrella chart depends on the official
Authentik chart and the repository's `k8s/helm` NeMo Helix chart. The
umbrella chart owns the shared PostgreSQL StatefulSet used by both Authentik
and NeMo Helix, plus the demo-specific glue resources needed for Authentik,
workload token exchange, and the NeMo Helix Envoy configuration.

The shared PostgreSQL image is pinned to
`docker.io/library/postgres:18.4`, which reports PostgreSQL `18.4` at runtime.

The chart blueprint includes a `nemo-setup` service account and app-password
solely for the automated auth-idp E2E test harness. It is not part of the
browser/device login path and it is not a workload identity pattern. Do not copy
`nemo-setup` or `e2e_setup_password_grant` into production deployments; use
normal OIDC login for users and workload token exchange for workloads.

## Demo Secrets

The local Authentik smoke test does not require an NGC key. The umbrella chart
leaves `nemo-helix.existingSecret` unset, so the NeMo Helix chart creates
its demo `ngc-api` Secret from chart values. If the NeMo Helix image comes
from a private registry instead of `kind load docker-image`, create an
image-pull Secret and add
`--set-string nemo-helix.imagePullSecrets[0].name=<secret>` to the Helm
upgrade command.

The chart creates or reuses these additional local-demo Secrets during Helm
rendering:

- `shared-postgresql` for the shared PostgreSQL superuser, Authentik, and NeMo Helix
  database passwords.
- `shared-postgresql-nemo` for the NeMo Helix external database password.
- `nemo-helix-envoy-tls` for the demo Envoy TLS certificate and CA.
- `nemo-workload-token-signing-key` for the NeMo Helix-issued workload access token
  signing key.

The chart generates `Secret/nemo-helix-envoy-tls` during Helm rendering and
reuses an existing Secret on upgrade. The Envoy TLS private key is never checked
into the repository and is not supplied through `values.yaml`.

All credentials and generated keys in this example are for local development
only.

## Authentik Blueprint

The umbrella chart packages the shared blueprint from
`helm/files/blueprints/nemo.yaml` into `ConfigMap/authentik-nemo-blueprint` and
configures the official Authentik chart to mount it. A Helm
`post-install,post-upgrade` hook Job runs `ak apply_blueprint` against that
mounted file.

Use `--wait --wait-for-jobs` when installing the chart so Helm only returns
after the blueprint has been applied.

## NeMo Helix Kubernetes Override

The umbrella chart passes the Kubernetes-specific NeMo Helix configuration
through `nemo-helix.platformConfig` values. It also configures
`nemo-helix.envoyProxy.configOverride` so the NeMo Helix chart's Envoy
deployment keeps the Authentik path split and validates both Authentik-issued
tokens, NeMo Helix workload-exchange tokens, and NeMo Helix Scoped
Access Key JWTs.

Kubernetes projected service account token expiration defaults to `600` seconds
in the jobs backend. Override it through the NeMo Helix chart values if you
need a longer projected token lifetime.

## Workload Token Exchange

The Kubernetes jobs backend projects a Kubernetes service account token into
workload pods and injects:

```text
NHX_WORKLOAD_IDENTITY_TOKEN_FILE=/var/run/secrets/nemo-helix/workload/token
```

The SDK reads that file and sends an RFC 8693 token exchange request to the
NeMo Helix auth service over HTTPS. The chart mounts `ca.crt` from
`Secret/nemo-helix-envoy-tls` into Kubernetes workload pods and sets
`SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` so in-pod Python HTTP clients verify
the demo Envoy certificate. Host-side `nemo` commands should use
`NHX_CLIENT_SSL_CERT_FILE` instead so unrelated tools keep their normal trust
store.

The NeMo Helix auth service validates projected service account tokens with
the TokenReview API and returns a NeMo Helix-signed JWT trusted by the
NeMo Helix Envoy. The TokenReview response must include exactly one
`authentication.kubernetes.io/pod-uid` value. The jobs controller observes the
created Pod, registers an internal delegation row keyed by that Pod UID,
service account subject, and audience, and auth later looks up that row from
the verified TokenReview metadata. The useful end-to-end validation is the
workload job in the tutorial: the job pod uses the exchanged token to call the
NeMo Helix API and read the workspace.

Auth only needs RBAC to create TokenReview requests. It does not need to read
Pods or Jobs for token exchange, and the chart grants no Pod or Job read
permissions to the auth/API service account for this path. Kubernetes labels,
annotations, and owner references help the jobs controller reconcile and clean
up backend resources, but they are not token-exchange authorization inputs.

The workload job request should not include workload auth environment
variables. The Kubernetes jobs backend owns `NHX_WORKLOAD_IDENTITY_TOKEN_FILE`;
users must not set `NHX_PRINCIPAL`, `NEMO_WORKLOAD_TOKEN`, or
`NEMO_WORKLOAD_TOKEN_FILE`. When workload OBO is enabled, the workload receives
only the subject token file path and obtains its delegated NeMo Helix access
token by calling `/apis/auth/token`.

To inspect the projected token mount for a submitted job:

```bash
kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" get pods \
  -l "nhx.nvidia.com/job_id=${JOB_NAME}"

kubectl --context "${KUBE_CONTEXT}" -n "${NAMESPACE}" describe pod \
  -l "nhx.nvidia.com/job_id=${JOB_NAME}"
```

## Shared Token Signing Key

Workload identity token exchange requires the NeMo Helix auth service to
sign the access token it mints from a Kubernetes projected service account subject
token. The chart creates `Secret/nemo-workload-token-signing-key` by default,
mounts `private-key.pem` into the NeMo Helix API pod at
`/etc/nhx/workload-token/private-key.pem`, and sets
`auth.token_signing.private_key_file` to that path. The same shared signing
configuration is used for workload-exchange access tokens and Scoped Access Key
JWTs. The matching public keys are served from `/apis/auth/jwks`.

The Helm-rendered Envoy config authenticates protected `/apis/` requests by
calling `/apis/auth/ext-authz` on the NeMo Helix API service. That endpoint
validates the presented bearer token and returns trusted `X-NHX-Principal-*`
response headers for Envoy to copy upstream. The public
`/apis/auth/authenticate` endpoint remains a JSON diagnostic/API endpoint for
direct callers and is not the configured Envoy callout. The gateway does not use
Envoy `claim_to_headers` for Scoped Access Keys, keeping revocation and dynamic
key checks inside the auth service before Envoy forwards trusted principal
headers.

Scoped Access Keys remain disabled in the checked-in chart values by default.
Runtime tests and local experiments can enable them with
`nemo-helix.platformConfig.auth.access_keys.enabled=true`.

The manual walkthrough can rely on the Helm chart to create and preserve this
Secret.

If you need the same deterministic key for manual debugging, generate one and
add the `--set-file` line to the Helm upgrade command:

```bash
mkdir -p contrib/auth/authentik/.generated
openssl genrsa -out contrib/auth/authentik/.generated/workload-token-private-key.pem 2048
chmod 600 contrib/auth/authentik/.generated/workload-token-private-key.pem
--set-file workloadTokenSigningKey.privateKeyPem=contrib/auth/authentik/.generated/workload-token-private-key.pem
```

For production-style deployments, provide an externally managed RSA private-key
Secret instead of relying on the demo-generated key. Set
`workloadTokenSigningKey.create=false`, keep
`workloadTokenSigningKey.secretName` and
`nemo-helix.api.extraVolumes[].secret.secretName` aligned, and keep
`auth.token_signing.private_key_file` pointed at the mounted file path.
