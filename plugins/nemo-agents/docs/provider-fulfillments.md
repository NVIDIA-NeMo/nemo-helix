<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Provider fulfillments for agent environments

Status: environment compilation, controller provisioning, sandbox attachment, cleanup, and lifecycle-aware credential resolution are implemented for OpenShell jobs and deployments. Operator enablement remains off by default: the selected gateway build must pass the expiry/refresh acceptance checks below. The pinned OpenShell 0.1.2 supervisor does not refresh an unchanged provider environment solely because a driver-returned credential expires. See the [driver setup and tests](../../nemo-deployments/src/nemo_deployments_plugin/backends/openshell/credential_driver/README.md).

This follows the [gateway-injected secrets discussion](https://nvidia.slack.com/archives/C0BN91FQ8MU/p1791911827914079): reuse EnvironmentSpec's secret references, explicitly bind an injection profile, run a credential driver per gateway, and provision providers per sandbox.

## User-facing configuration

Agent behavior stays in `agent.yaml`. An environment spec supplies service configuration and credential bindings:

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

`service` is a binding name. `service-read` identifies an operator-approved injection profile. `API_TOKEN` is the profile's credential slot; `SERVICE_SECRET` names an entry in this spec's `secrets` map. No values are supplied in this configuration.

The compiler returns the provider credential reference separately. `SERVICE_SECRET` is excluded from process-env injection, while the unconsumed `LOCAL_CONFIG_SECRET` retains existing env-injection behavior. OpenShell supplies its supported placeholder as `API_TOKEN`. The job specification contains reference metadata, but never registration tickets or provider credential values.

Validation rejects missing secret aliases, empty profiles or bindings, duplicate credential slots, conflicting plaintext env assignments, and exposure of a provider credential through another env alias or a declared MCP fulfillment. Bare and workspace-qualified references to the same secret are treated alike using the launch workspace. The controller verifies the operator's approved profile, workspace and slot list before registration; schema validation alone does not grant access.

## Ownership and runtime work

1. **Contract and compiler.** Preserve bindings through environment APIs and partition secret references. Local execution and other substrates reject providers. Jobs require a configured OpenShell execution profile; deployments require a configured OpenShell executor. Existing environments without providers keep their current behavior.
2. **Registration contract (driver implemented).** Establish a trusted association between a provider credential slot and an existing NeMo secret. OpenShell v0.1.2's public CreateProvider/UpdateProvider API rejects caller-supplied credential handles. Normal creation calls the active driver's StoreCredential operation with a value. A resolve-only adapter cannot establish our bindings using that API. The NeMo driver interprets an authenticated, short-lived registration ticket in that value field. StoreCredential validates and binds it to the provider UUID, then returns an opaque handle. It stores only binding metadata, not secret bytes. The ticket protocol is private to Helix's controller and driver; a bare secret reference or arbitrary credential value is rejected.
3. **Credential driver and authorization.** Run the driver outside the sandbox. Each resolution checks provider ownership and the current job attempt/step or deployment entity identity. Completed, cancelled, paused, deleted or replaced workloads are denied. Secret reads delegate to the signed creator identity so the Secrets service rechecks permission. Lifecycle and local revocation are checked again after the read. Deleting a provider releases its binding, never the user's NeMo secret.
4. **Provisioning and cleanup.** Controllers create a dedicated provider for each binding and workload generation, then attach its generated name in `SandboxSpec.providers`. Labels allow retries to reuse only matching providers and allow cleanup after controller restarts. Sandbox deletion precedes provider deletion; a periodic sweep reclaims unattached providers after a ten-minute grace period. OpenShell refuses deletion of attached providers.
5. **Sandbox platform access.** Provider-bearing sandboxes get an enforced REST allowlist instead of broad platform egress. Jobs can fetch their own step, report status/results, access their output fileset and declared input filesets, and use inference/telemetry. Only explicitly declared ordinary env secrets may be fetched. Bound-secret reads, entity access, token minting and new job submission are excluded. This restriction applies outside the sandbox even if the agent bypasses its loopback auth proxy. OpenShell admission, network isolation, TLS inspection and attachment authorization remain part of the trusted boundary.
6. **Cache lifetime and renewal.** The driver returns at most 30 seconds of validity and keeps no value cache. Live workload bindings survive beyond the standalone ticket's one-hour lease because lifecycle authorization is rechecked. With `renewal_mode: helix`, the existing Jobs and Deployments controllers periodically send fresh registration tickets through `UpdateProvider` for active, attached sandboxes. The driver rechecks lifecycle and permission when resolving each replacement. Failed updates retain the original expiry and retry on the next reconciliation. Revocation affects subsequent authenticated requests; it cannot undo requests already accepted by the external service.

NeMo owns persistent storage and binding authorization; OpenShell owns network policy and credential injection. The driver still delivers plaintext temporarily to trusted OpenShell components, so this design does not claim zero retention.

## Compatibility and open decisions

This feature covers credential binding, injection, refresh and cleanup. It adds no inference authorization lookup, watchdog, authorization controller, or whole-workload shutdown on permission loss. Permission denial or deletion prevents fresh credential resolution; already cached credentials remain usable until their existing expiry. Rotation takes effect on the next successful gateway resolution. Ordinary Jobs and Deployments controllers retain responsibility for workload lifecycle and provider cleanup.

With platform auth enabled, provisioning checks the creator's secret-access permission without fetching a value, and driver reads delegate to that signed creator identity. With auth disabled, provisioning checks that the configured secret exists and signs a binding without a principal. Start the driver explicitly with `--auth-disabled` in that mode; a missing token never selects it automatically. The driver rejects tickets and persisted bindings from the other mode. Profile approval, provider ownership, workload identity, signed tickets and the sandbox REST allowlist apply in both modes. Auth-disabled installations do not provide per-user access controls; their reachable platform APIs remain a trusted administrative surface.

- `providers` is a proposed public contract, replacing the earlier `gateway_secrets` draft. Whether Helix should present a generic service-access profile registry or expose OpenShell profiles directly remains open.
- `EnvironmentSpec.provider` (singular) still selects the Fabric environment; it is unrelated to these credential bindings.
- Existing servers that ignore `providers` may inject every entry of `secrets` into the agent. Upgrade all environment consumers before storing these specs; do not use them in a mixed-version deployment or with old SDKs that discard the field.
- The v0.1.2 gateway selects a credential driver at gateway scope. The adapter must define how other provider credentials on that gateway are handled.
- The REST allowlist narrows the sandbox's access despite existing broad job/service identities. It is not a replacement for platform-wide workload identity work; the restricted network path must remain mandatory.

## Operator enablement

Add this block inside the OpenShell executor `config` (deployments) or OpenShell execution profile `config` (jobs), alongside the gateway endpoint, workspace and platform egress:

```yaml
gateway_credentials:
  gateway_id: production-gateway
  registration_key_file: /run/nemo-driver-credentials/registration.key
  renewal_mode: helix
  renewal_interval_seconds: 10
  credential_refresh_verified: false
  profiles:
    service-read:
      profile_workspace: openshell-team
      workspaces: [team]
      credential_keys: [API_TOKEN]
```

The key and gateway ID must match the driver. `profile_workspace` is either the configured gateway workspace or empty for a global profile. Profiles are installed through OpenShell by the operator and must target explicit external hosts; profiles targeting the platform or requiring gateway-minted refresh material are rejected. Use a dedicated gateway for this ticket-only driver. Neither the signing key nor gateway administrator credentials enter a sandbox.

`renewal_mode: helix` is a temporary compatibility mode that renews active sandboxes through the existing Jobs and Deployments controllers, including idle sandboxes. `renewal_interval_seconds` defaults to 10 and accepts 1–15 seconds. It is a minimum interval between successful updates, not a scheduling guarantee. The driver advertises a configurable cache lifetime (300 seconds by default); controller polling and gateway propagation must fit within that lifetime. OpenShell is trusted to hold the resolved credentials. A temporary outage leaves existing credentials usable until expiry and retries renewal without extending expiry locally.

A confirmed creator permission denial or missing secret records cleanup intent in the provider's config, then detaches and deletes that provider. Failed cleanup resumes on the next reconciliation or orphan sweep, including after a controller restart. Restoring permission does not cancel recorded cleanup or recreate a missing provider during renewal; start a new workload to restore access. Other providers remain independently renewable. This removes the provider's credential and network attachment, without promising immediate process termination or cancellation of already accepted requests.

OpenShell 0.1.2 retains old placeholder generations. A 30-second cache lifetime can expire a retained generation before it falls out of that history. A five-minute lifetime passed sustained requests and rotation using one original placeholder in the isolated gateway test. Rotation may continue using the old value during propagation; retain overlapping source tokens when possible. Each renewal creates binding metadata and a tombstone for the previous handle; replay-safe registry compaction remains future work.

The default `renewal_mode: gateway` sends no periodic Helix updates and requires a gateway build that handles renewal itself. Neither mode requires changes to `agent.yaml` or the environment binding.

`credential_refresh_verified` is an operator attestation, not automatic capability detection. Leave it false until the selected mode passes acceptance on your gateway: repeated requests using the original startup placeholder continue beyond the configured cache lifetime; rotating the NeMo value updates those requests; deleting the secret or revoking creator permission detaches the provider through controller cleanup; outages do not extend the advertised credential lifetime; direct and encoded-path requests to Secrets, Jobs creation and token endpoints are denied. Preserve the original placeholder across the entire test: starting a fresh exec process and rereading its environment can hide failures in an already-running agent. A synthetic Secrets fixture can test caching and renewal, but does not establish real Helix authorization or sandbox isolation.

Upstream contracts: [credential driver protocol](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/credential_driver.proto), [provider registration](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-server/src/grpc/provider.rs).
