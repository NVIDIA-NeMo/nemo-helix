<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Identity Provider References

This directory contains NeMo Platform identity-provider reference bundles.

Available references:

- `authentik`: Docker Compose and Kubernetes.
- `zitadel`: Kubernetes only.

Each provider bundle defines one contract for local validation and production
adaptation:

- expose OIDC discovery metadata
- include a gateway layer that strips inbound `X-NMP-Principal-*` headers
- define one human identity and one machine identity for shared auth testing
- treat external machine identities as ordinary OIDC principals authorized by
  group binding, not as internal `service:*` principals
- document provider-specific setup in a local `README.md`

Kubernetes-only references can seed generated IdP clients at install time. When
that is required, the provider chart should store generated test credentials in
a namespace-local Secret and patch only the demo ConfigMap values that cannot be
known before the IdP is initialized.

Managed job OBO tests use the provider for user and controller authentication,
but the workload-to-submitter binding is NeMo Platform auth state. Jobs receive
`NMP_WORKLOAD_IDENTITY_TOKEN_FILE`, exchange that subject token through
`/apis/auth/token`, and receive a NeMo Platform token whose top-level subject
is the job submitter and whose RFC 8693 `act.sub` is the workload actor. Provider
manifests may include workload-provider token grants for contract tests, but
managed Docker job OBO must not depend on provider-specific fields such as
`jti`.

Open-source providers with `mode: compose-ci` are intended for the shared auth
matrix. Reference-only providers stay documented and manifest-driven but are
excluded from the local Compose-backed matrix.
