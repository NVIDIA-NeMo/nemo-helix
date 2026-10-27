<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo credential registration driver

Experimental gateway-side implementation of OpenShell's credential-driver protocol. It registers existing NeMo secret references through StoreCredential. It never creates, updates, or deletes the underlying NeMo secret.

Jobs and deployments now provision and attach providers through their OpenShell backends. Enablement requires operator configuration and verified gateway expiry/refresh support. This implementation has protocol, controller, lifecycle and local proxy tests; it has not passed a live OpenShell sandbox security acceptance test.

## Lifecycle

1. Trusted Helix provisioning selects an approved profile and authorizes the job. `RegistrationTickets.issue` runs the OpenShell binding preflight: check the signed creator's `secrets.read` permission through the existing platform authorizer when auth is enabled, then fetch existing Secrets metadata. With auth disabled it only checks metadata. It returns a short-lived, signed `SecretStr` ticket without fetching a value or adding a Secrets endpoint.
2. Provisioning supplies the ticket as the credential value in the normal `CreateProvider` request. Never put it in an agent's environment or a persisted user-visible job specification.
3. OpenShell assigns its provider UUID and calls `StoreCredential`. The driver validates the ticket, saves the binding metadata in SQLite, and returns an opaque handle. No secret value is fetched or copied during registration.
4. `ResolveCredentials` checks the provider UUID, name, workspace, slot and revocation. For workload bindings it reads the owning job attempt/step or deployment and checks that its immutable identity remains active. It reads the secret with `X-NHX-Principal-On-Behalf-Of` set to the signed creator, so the Secrets service rechecks delegated access. It checks lifecycle and revocation again after the read and returns at most 300 seconds of validity by default (configurable with `--cache-lifetime-seconds`, from 1 to 3600; standalone grants are also capped by binding expiry). The driver has no value cache.
5. `DeleteCredential` tombstones the binding. It does not call NeMo delete.

Tickets expire within five minutes. Standalone grants expire within one hour; controller-created workload grants remain usable only while the owning workload and delegated access remain valid. Identical Store retries return the same handle while the ticket remains valid. A ticket cannot be replayed with another provider UUID, slot, workspace or write identity. A fresh ticket can stage a replacement handle; the prior handle stays valid until OpenShell commits its update and deletes it. Failed gateway creation must also delete the staged handles.

The registry persists retry identity and revocation across restarts. It contains references and scope metadata, never raw tickets or secret values. It is bound to one configured gateway ID. Preserve it across restarts; do not share it across unrelated gateways or discard it as a cache. Tombstones are currently retained without automatic compaction. SQLite is intended for one host with a persistent local volume, not a multi-replica deployment over a network filesystem.

The registry uses SQLAlchemy 2 typed ORM models for bindings, revoked jobs and gateway identity. It uses an immediate write transaction for atomic registration retries. A remote Helix entity store is possible; running beside the gateway does not prevent using it. Local storage keeps these metadata checks independent of another remote service, at the cost of a persistent volume and single-host operation. Moving to the entity store would need equivalent atomic retry/revocation semantics, restricted credentials and explicit handling of outages.

## Provisioning-side API

Use a dedicated, random signing key shared only by the trusted provisioning controller and this driver. This is not the service credential.

```python
from nemo_deployments_plugin.backends.openshell.credential_driver.tickets import (
    BindingScope,
    RegistrationTickets,
)

issuer = RegistrationTickets(registration_key_bytes, gateway_id="production-gateway")
ticket = await issuer.issue(
    BindingScope(
        gateway_id="production-gateway",
        gateway_workspace="openshell-team",
        provider="unique-provider-for-this-job",
        credential_key="API_TOKEN",
        secret_workspace="helix-team",
        secret_name="service-token",
        job_id="immutable-helix-job-id",
        principal_id="creator-principal-id",
    ),
    requester_secrets=request_scoped_secrets_client,
)
# In the trusted controller only:
# CreateProvider.provider.credentials["API_TOKEN"] = ticket.get_secret_value()
```

With auth enabled, use a controller service client delegated to the creator. The OpenShell preflight checks the signed creator directly using the platform's existing `AuthClient`; a privileged metadata lookup alone is insufficient. Profile and job authorization belong to the controller. With auth disabled, the signed scope has `principal_id=None` and checks secret existence. The driver must explicitly run with `--auth-disabled` instead of `--helix-token-file`; mismatched bindings are rejected before secret access. Ticket minting remains private to the trusted controller in both modes. Secrets retains its existing value-access enforcement when the driver resolves a credential.

## Gateway setup

For a local Docker gateway, the [Compose sidecar guide](../../../../../examples/openshell/credential-driver.md) reuses the Helix API image and supplies initialization, readiness checks, restart handling and persistent volumes. The overlay is opt-in and leaves the existing gateway example unchanged.

Run this module in an environment with `nemo-deployments-plugin` and its `openshell` extra installed:

```sh
python -m nemo_deployments_plugin.backends.openshell.credential_driver.server \
  --socket-path /run/nemo-credential-driver/driver.sock \
  --state-path /var/lib/nemo-credential-driver/bindings.db \
  --gateway-id production-gateway \
  --registration-key-file /run/nemo-driver-credentials/registration.key \
  --helix-url https://helix.example \
  --helix-token-file /run/nemo-driver-credentials/helix.token
```

The key must contain at least 32 random bytes. Key/token files must be regular, non-symlink files owned by the driver user with no group/other permissions. State and socket directories must be private (0700); the socket and database use 0600. Run the gateway and driver with the same trusted UID, or adapt the deployment to provide equivalent local isolation. Mounts must not be visible inside agent sandboxes. The bearer-token file is reread for each NeMo call. The driver identity needs Secrets access with delegation and read access to the relevant Jobs and deployment entities; it does not need permission to modify those resources. Only HTTPS NeMo endpoints are accepted by default; development HTTP requires `--allow-insecure-localhost` and a loopback URL.

`MAX_CREDENTIAL_FILE_BYTES` is a defensive 64 KiB input limit, not a protocol requirement. It leaves room for a service-account JWT without reading an arbitrarily large mounted file. Reading one extra byte distinguishes a file at the limit from one exceeding it.

For a prestarted driver, configure OpenShell with:

```toml
[openshell.gateway]
credential_drivers = ["nemo"]

[openshell.credential_drivers.nemo]
transport = "uds"
socket_path = "/run/nemo-credential-driver/driver.sock"
```

OpenShell supports one active credential driver per gateway. This adapter rejects raw credentials and refresh material: use a dedicated gateway for this experiment rather than switching a gateway containing unrelated provider credentials.

## Runtime enablement

Configure `gateway_credentials` on each Helix OpenShell executor/profile as described in the [environment integration guide](../../../../../../nemo-agents/docs/provider-fulfillments.md). Controllers check approved profiles and credential slots, create per-workload providers, attach them before launch, and retry cleanup using gateway ownership labels. They never persist tickets into user-visible job or deployment configuration.

Provider-bearing sandboxes use a restricted platform REST policy. They cannot fetch bound secrets or submit another job with broader authority. Ordinary env-secret references and input filesets remain explicit allowances; the gateway must enforce these rules and attachment authorization. Driver RPCs identify a provider, not a sandbox, so a unique name alone is insufficient. Users with gateway administrator privileges are outside this sandbox threat model.

`credential_refresh_verified` defaults to false. The optional `renewal_mode: helix` makes the existing Jobs and Deployments controllers update active, attached providers with fresh registration tickets, by default every 10 seconds. A five-minute driver cache lifetime passed sustained requests and rotation on stock OpenShell 0.1.2 using the original process placeholder; 30 seconds did not reliably outlast its retained placeholder history. A confirmed permission denial or secret deletion records cleanup intent in provider config, then detaches and deletes the provider. Controllers and orphan sweeps retry unfinished cleanup after restart; a revoked provider is not renewed even if permission returns. Outages alone do not trigger removal or extend existing expiry. This renews active but idle sandboxes too. The default `renewal_mode: gateway` requires renewal support in the selected gateway build. Validate the configured lifetime, rotation, cleanup, outages and platform isolation before enabling your installation. Synthetic gateway tests do not establish real Helix authorization.

Helix lifecycle changes and Secrets permission checks are evaluated on every resolution. `revoke_job` also remains available as a trusted local registry operation. NeMo secret deletion/rotation is observed on the next resolution; already accepted service requests cannot be revoked retroactively. The issuing service's own token revocation remains separate.

## Protocol and tests

`protocol/` contains only the generated descriptor needed by `wire.py` and its license notice. The `.proto` sources are not vendored. The descriptor was generated with `grpcio-tools==1.78.0` from OpenShell v0.1.2, source commit `6648bd0c290efbc41ba131ee9831ee45cd431f94`. Sources: [credential driver](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/credential_driver.proto), [datamodel](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/datamodel.proto), [extension](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/extension.proto), [options](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/options.proto).

The SDK omits driver stubs, so `wire.py` uses these official descriptors in a private pool. This avoids collisions with the SDK's generated datamodel. No field numbers or service paths are reimplemented.

To regenerate, check out that OpenShell commit separately and run from this module directory, setting `OPENSHELL_SOURCE` to the absolute path of that checkout:

```sh
uv run --no-project --with grpcio-tools==1.78.0 python -m grpc_tools.protoc \
  -I "$OPENSHELL_SOURCE/proto" --include_imports \
  --descriptor_set_out=protocol/credential_driver.pb \
  "$OPENSHELL_SOURCE/proto/credential_driver.proto"
```

Tests: `test_credential_driver.py`, `test_credential_provisioning.py` and `test_openshell_launch.py` under `plugins/nemo-deployments/tests/unit/`, plus the jobs/deployments backend tests. Coverage includes ticket integrity, replay, restart, concurrent retries, staged updates, active workload checks, replacement/deletion, expiry, permission/backend failures, atomic batch validation, real Unix-socket gRPC exchanges, provider attachment, orphan cleanup and an HTTP proxy subprocess exchange. The provisioning tests use real CreateProvider protobufs and the real StoreCredential handler behind a fake gateway; they do not run an OpenShell gateway or sandbox.

## Upstream SDK simplification

The SDK bindings are being added upstream in [OpenShell PR #4760](https://github.com/NVIDIA/OpenShell/pull/4760). Keep the generated descriptor and private descriptor loader until those bindings land and are available in the SDK version supported by Helix.

With a released SDK containing those bindings, this adapter could implement `CredentialDriverServicer` and use `add_CredentialDriverServicer_to_server`, replacing the private descriptor loader and generic gRPC registration. No gateway protocol change is required. The current `wire.py` remains compatible with the pinned 0.1.2 SDK.
