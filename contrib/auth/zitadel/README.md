<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ZITADEL Kubernetes Reference Example

This directory contains a Kubernetes-only ZITADEL-backed NeMo Platform auth
reference. It is intentionally parallel to the Authentik Kubernetes runtime, but
does not include a Docker Compose mode.

The local demo is designed around two ZITADEL traits:

- ZITADEL can issue opaque access tokens, so NeMo validates bearer tokens with
  RFC 7662 introspection. This requires the opaque-token support from PR #2008
  or later.
- ZITADEL does not support resource-owner password credentials. The test
  harness uses demo service applications for setup and workload-provider token
  acquisition, and device flow for the human login path.

The example keeps NeMo Platform runtime changes minimal by configuring ZITADEL
to emit a standard `groups` custom claim. NeMo can then use its existing
`auth.oidc.groups_claim: groups` behavior instead of adding ZITADEL-specific
role-claim parsing.

The Helm chart includes a post-install seed job that uses the ZITADEL
FirstInstance admin PAT to create the demo project, OIDC client, machine users,
project grants, and custom `groups` action. ZITADEL-generated client IDs and
secrets are stored in the `nemo-zitadel-seed-state` Kubernetes Secret and are
used to patch the NeMo Platform ConfigMap and Secret-backed API environment
before API/controller pods are restarted. Client secrets stay in Kubernetes
Secrets rather than being written into the ConfigMap.

All credentials and generated secrets in this example are for local development
only. The chart generates local Secrets for the ZITADEL master key, demo user
password, embedded PostgreSQL passwords, and NeMo Platform placeholder NGC key.
Do not copy those Secrets into non-local deployments.

## Prerequisites

- Docker with `buildx`
- `kind` or `k3d`
- `kubectl`
- `helm`
- `uv`

## Kubernetes

The Kubernetes deployment is installed through the umbrella Helm chart at
`contrib/auth/zitadel/helm`.

`run.sh` is the automation entrypoint for CI-style validation and repeatable
local Kubernetes instances:

```bash
contrib/auth/zitadel/run.sh --help
contrib/auth/zitadel/run.sh up k8s
contrib/auth/zitadel/run.sh test k8s
contrib/auth/zitadel/run.sh down k8s
contrib/auth/zitadel/run.sh clean
```

ZITADEL is Kubernetes-only in this reference, so the runner requires the `k8s`
target. It keeps provider-specific defaults under the `NMP_ZITADEL_K8S_*`
environment prefix.

Runtime details live in:

- [Kubernetes Implementation Details](kubernetes/implementation-details.md)

## Demo Identities

The runtime seeds these local-only identities:

- Human user: `nemo-user`
- Human password: stored in the `interactive_user_password` key of the
  `nemo-zitadel-seed-state` Kubernetes Secret
- Human email: `nemo-user@example.com`
- CLI OIDC client name: `nemo-platform-cli`
- Setup service user: `nemo-setup`
- Workload identity: `svc-nemo`
- Workload group claim: `nemo-workloads`

To inspect the generated human password in a running local demo:

```bash
kubectl -n nemo-zitadel get secret nemo-zitadel-seed-state \
  -o jsonpath='{.data.interactive_user_password}' | base64 -d
```

Do not copy these identities or generated client secrets into production.

## Next Steps

- Run the k8s contracts with `contrib/auth/zitadel/run.sh test k8s`.
- Review [Kubernetes Implementation Details](kubernetes/implementation-details.md)
  before adapting the example for another cluster or IdP.
