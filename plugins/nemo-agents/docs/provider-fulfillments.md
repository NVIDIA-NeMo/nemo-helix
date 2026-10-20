<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Provider fulfillments for agent environments

Status: contract, compiler, and an experimental registration-ticket credential
driver are implemented. Environment specs can store bindings, but agent execution
and deployment reject them until provisioning and identity restrictions are wired.
See the [driver setup and tests](../../nemo-deployments/src/nemo_deployments_plugin/backends/openshell/credential_driver/README.md).
This is not yet a working agent-to-service gateway deployment.

This follows the [gateway-injected secrets discussion](https://nvidia.slack.com/archives/C0BN91FQ8MU/p1791911827914079):
reuse EnvironmentSpec's secret references, explicitly bind an injection profile,
run a credential driver per gateway, and provision providers per sandbox.

## User-facing configuration

Agent behavior stays in `agent.yaml`. An environment spec supplies service
configuration and credential bindings:

```yaml
name: service-access
env:
  SERVICE_URL: https://service.example
secrets:
  SERVICE_SECRET: team/service-token
  LOCAL_CONFIG_SECRET: team/local-config-token
providers:
  service:
    profile: service-read
    credentials:
      API_TOKEN: SERVICE_SECRET
```

`service` is a binding name. `service-read` identifies an operator-approved
injection profile. `API_TOKEN` is the profile's credential slot;
`SERVICE_SECRET` names an entry in this spec's `secrets` map. No values are
supplied in this configuration.

The compiler returns the provider credential reference separately.
`SERVICE_SECRET` is excluded from process-env injection, while the unconsumed
`LOCAL_CONFIG_SECRET` retains existing env-injection behavior. The agent receives
neither the provider reference nor a real credential in its compiled config.
A future provisioning layer must supply OpenShell's supported placeholder.

Validation rejects missing secret aliases, empty profiles or bindings, duplicate
credential slots, conflicting plaintext env assignments, and exposure of a
provider credential through another env alias or a declared MCP fulfillment.
Bare and workspace-qualified references to the same secret are treated alike
using the launch workspace, consistent with current job secret resolution.
Profile existence, allowed destinations, and authorization require the future
trusted provisioning layer; schema validation does not grant access.

## Ownership and runtime work

1. **Contract and compiler (this change).** Preserve bindings through environment
   APIs, partition secret references, and reject unsupported execution and
   deployment before creating runnable resources. Existing environments without
   providers keep their current behavior.
2. **Registration contract (driver implemented).** Establish a trusted association between a provider
   credential slot and an existing NeMo secret. OpenShell v0.1.2's public
   CreateProvider/UpdateProvider API rejects caller-supplied credential handles.
   Normal creation calls the active driver's StoreCredential operation with a
   value. A resolve-only adapter cannot establish our bindings using that API.
   The NeMo driver interprets an authenticated, short-lived registration ticket
   in that value field. StoreCredential validates and binds it to the provider
   UUID, then returns an opaque handle. It stores only binding metadata, not
   secret bytes. The ticket protocol is private to Helix's controller and driver;
   a bare secret reference or arbitrary credential value is rejected.
3. **Credential driver and authorization.** Run the implemented long-lived driver outside the
   sandbox, with scoped access to NeMo Secrets. Resolve only bindings authorized
   for the owning provider and active job binding. OpenShell must independently
   enforce the sandbox attachment. Deleting a provider releases its
   binding, never the user's NeMo secret. Agent job credentials must not allow
   the sandbox to read these secrets or create bindings through Jobs authority.
4. **Provisioning and cleanup.** Select approved profiles, create and attach
   providers before agent start, and reconcile cleanup after failures, retries,
   restarts, and job deletion. Translate Helix binding names to generated
   OpenShell provider names in `SandboxSpec.providers`. Keep driver sockets,
   endpoint policies, and OpenShell configuration out of `agent.yaml`.
5. **Lifecycle and enforcement.** Define and test rotation/revocation behavior
   before enabling launches. OpenShell caches static credentials in the runtime;
   a NeMo storage adapter does not make resolution happen on every HTTP request.
   Require invalidation or an enforced bounded lifetime, with fail-closed
   behavior on expiry or loss of authorization. Revoking the credential at its
   issuing service remains separate from disabling its NeMo binding.

NeMo owns persistent storage and binding authorization; OpenShell owns network
policy and credential injection. The driver still delivers plaintext temporarily
to trusted OpenShell components, so this design does not claim zero retention.

## Compatibility and open decisions

- `providers` is a proposed public contract, replacing the earlier
  `gateway_secrets` draft. Whether Helix should present a generic service-access
  profile registry or expose OpenShell profiles directly remains open.
- `EnvironmentSpec.provider` (singular) still selects the Fabric environment;
  it is unrelated to these credential bindings.
- Existing servers that ignore `providers` may inject every entry of `secrets`
  into the agent. Upgrade all environment consumers before storing these specs;
  do not use them in a mixed-version deployment or with old SDKs that discard
  the field. Runtime enablement needs a capability/version check.
- The v0.1.2 gateway selects a credential driver at gateway scope. The adapter
  must define how other provider credentials on that gateway are handled.
- Broad authorization inherited by today's job identity is a separate blocker
  to the promise that the agent cannot recover the credential through an API.

Upstream contracts:
[credential driver protocol](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/credential_driver.proto),
[provider registration](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-server/src/grpc/provider.rs).
