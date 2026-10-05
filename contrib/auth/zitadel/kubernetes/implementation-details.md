<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ZITADEL Kubernetes Implementation Details

## Prerequisites

- A Kubernetes cluster supported by `contrib/auth/zitadel/run.sh`
- `kubectl` and `helm` access to the target namespace
- A NeMo Helix API image available to the cluster

The ZITADEL runtime keeps the same trust boundary as the Authentik Kubernetes
example: a single Envoy gateway is the public edge for NeMo Helix and the
IdP. Envoy strips inbound NeMo trusted identity headers, proxies public OIDC
paths to ZITADEL, and sends protected NeMo API calls through NeMo's auth
service ext_authz endpoint.

NeMo validates ZITADEL opaque access tokens with RFC 7662 introspection. The
chart sets `auth.oidc.introspect_opaque_tokens=true` and intentionally leaves
UserInfo fallback disabled for bearer-token validation so audience and scope
semantics stay with the introspection response.

For group-based RBAC, the demo configures ZITADEL to emit a standard `groups`
custom claim from project roles. That keeps NeMo Helix configuration
provider-neutral:

```yaml
auth:
  oidc:
    groups_claim: groups
```

The chart generates Kubernetes Secrets for the ZITADEL master key, demo user
password, embedded PostgreSQL passwords, and NeMo Helix placeholder NGC key
instead of embedding those values in `values.yaml`.

The Helm seed job creates the ZITADEL project, role, OIDC app, setup machine
user, workload machine user, user grants, and complement-token action. It stores
the generated client credentials and demo user password in
`nemo-zitadel-seed-state`, patches non-secret rendered `nemo-helix-config`
ConfigMap placeholders, and restarts the NeMo API and core-controller
deployments. The API reads the introspection client secret from the
Secret-backed `NHX_AUTH_OIDC_INTROSPECTION_CLIENT_SECRET` environment variable,
whose name is referenced by `auth.oidc.introspection_client_secret_env_var`.

Managed Kubernetes jobs and deployments use NeMo Helix workload token
exchange with Kubernetes TokenReview, not ZITADEL-issued workload subject
tokens.

## Next Steps

- Run `contrib/auth/zitadel/run.sh test k8s` after changing the chart or seed
  job.
- Replace generated local demo Secrets with externally managed Secrets before
  adapting this reference for a shared cluster.
