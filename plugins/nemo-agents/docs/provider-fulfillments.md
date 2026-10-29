<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Service credentials in OpenShell sandboxes

Helix binds existing Secrets references to operator-approved OpenShell injection profiles. The agent receives a placeholder; the trusted OpenShell gateway resolves the credential through the NeMo credential driver and injects it into matching outbound requests. Jobs support this feature when their OpenShell backend is configured and enabled by the operator.

Deployment and in-memory execution reject provider bindings.

## Environment configuration

Agent behavior stays in `agent.yaml`. Add the service URL and credential binding to its environment spec:

```yaml
name: service-access
env:
  SERVICE_URL: https://service.example
secrets:
  SERVICE_SECRET: team/service-token
settings:
  openshell:
    providers:
      service:
        profile: service-read
        credentials:
          API_TOKEN: SERVICE_SECRET
```

`service` names the binding. `service-read` selects an approved injection profile. `API_TOKEN` is the profile's credential slot, and `SERVICE_SECRET` names an entry in this environment's `secrets` map. The agent uses the `API_TOKEN` placeholder supplied by OpenShell; the spec contains no credential values.

OpenShell requires external injection bindings for all secrets. Unbound top-level or MCP process-env secrets reject launch, including when no providers are declared. Non-secret `env` entries pass through normally. Other backends retain their existing env-secret behavior and reject OpenShell provider bindings.

Compilation rejects missing aliases, duplicate slots, plaintext env conflicts, and exposure of a bound secret through another env alias or MCP fulfillment. It strips `settings.openshell` from the agent config and keeps the references separately for provisioning. `EnvironmentSpec.provider` (singular) selects the Fabric environment and is unrelated to credential bindings.

## Runtime behavior

1. The Helix controller checks the profile approval and secret access, then creates a dedicated OpenShell provider for each workload binding using a signed registration ticket.
2. The credential driver validates the ticket and stores reference and ownership metadata in a persistent registry. Registration does not fetch the secret value.
3. On credential resolution, the driver checks the owning job attempt/step and reads the secret. With platform auth enabled, it checks the signed creator's current permission through Auth, then reads using the driver's service identity. OpenShell receives the value; the sandbox receives a placeholder.
4. Controllers attach providers before launch and remove them after sandbox deletion. Ownership labels support retries and cleanup after controller restarts; an orphan sweep reclaims unattached providers after a ten-minute grace period.

Provider-bearing sandboxes have a restricted platform REST allowlist enforced by OpenShell. It permits required workload operations, declared filesets, inference and telemetry, while excluding secret reads, entity access, token minting and new job submission. The gateway's network policy, TLS inspection and attachment authorization must remain enforced. Gateway administrators are trusted.

## Operator configuration

Run the [credential-driver sidecar](../../nemo-deployments/examples/openshell/credential-driver.md) beside a dedicated gateway. Add this block inside the OpenShell job execution profile `config`:

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

The gateway ID and signing key must match the driver. Install profiles through OpenShell; `profile_workspace` must match the gateway workspace or be empty for a global profile. Profiles must target explicit external hosts and declare the approved credential slots. Platform destinations and profiles requiring gateway-minted refresh material are rejected. Signing keys and gateway administrator credentials stay outside sandboxes.

| Setting | Behavior |
| --- | --- |
| `renewal_mode: helix` | Jobs controllers update active, attached providers with fresh tickets, including idle sandboxes. |
| `renewal_interval_seconds` | Minimum interval between successful updates; default 10 seconds, range 1–15. Controller polling and gateway propagation add delay. |
| `renewal_mode: gateway` | Default; Helix sends no periodic updates. Requires renewal support in the selected gateway build. |
| `credential_refresh_verified` | Defaults to false and blocks provider workloads until the operator verifies refresh behavior. It is an attestation, not automatic capability detection. |

Before enabling, verify that requests using the original startup placeholder continue beyond the configured credential lifetime; rotation reaches those requests; permission denial or secret deletion removes the provider; outages do not extend cached credentials; and sandbox requests to disallowed platform endpoints are blocked. Keep the original placeholder throughout validation: a fresh process environment can mask a failure affecting a running agent.

With platform auth enabled, the creator's permission is required at both provisioning and credential resolution. With auth disabled, provisioning checks secret existence; start the driver explicitly with `--auth-disabled`. Tickets from the other auth mode are rejected. Signed scopes, profile approval, workload checks and sandbox network restrictions apply in both modes. Auth-disabled installations provide no per-user access control, so reachable platform APIs remain a trusted administrative surface.

## Rotation, revocation and limits

The driver has no value cache. OpenShell is trusted to hold resolved credentials for the advertised lifetime: 300 seconds by default, configurable from 1–3600 seconds with `--cache-lifetime-seconds`. Controller polling and propagation must fit within that lifetime. Rotation takes effect on a successful resolution and propagation; old values may remain usable during that interval. Retain overlapping source tokens when possible.

In Helix renewal mode, confirmed permission denial or a missing secret records cleanup intent, then detaches and deletes the affected provider. Cleanup retries after failure or controller restart. Restoring permission does not undo recorded cleanup or recreate a missing provider during renewal; start a new workload to restore access. Other providers remain independently renewable. Temporary outages retry without extending existing expiry.

Provider removal does not promise immediate process termination or cancellation of requests already accepted by the external service. Removing a provider leaves the underlying Helix secret intact. Source-service token revocation is separate. The Jobs controller retains responsibility for job lifecycle.

Upgrade all environment consumers before storing these specs. Older consumers that discard `settings.openshell` may inject the referenced secrets into process environments; mixed-version consumers are unsupported. The credential driver accepts registration tickets only, so its gateway cannot also serve unrelated raw provider credentials.
