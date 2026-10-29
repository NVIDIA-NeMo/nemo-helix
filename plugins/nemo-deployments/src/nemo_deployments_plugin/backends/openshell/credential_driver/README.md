<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo credential driver

The gateway-side driver implements OpenShell's credential-driver protocol using the OpenShell 0.1.6 SDK. It binds existing NeMo secret references to providers, resolves their values for the trusted gateway, and never creates, updates or deletes the underlying secrets.


## Credential lifecycle

1. The owning Jobs or Deployments backend authorizes the workload and injection profile, checks the creator's secret-access permission and secret metadata without fetching a value, then signs a registration ticket. With auth disabled, it checks existence only.
2. The controller sends the signed ticket as the credential value in `CreateProvider` or `UpdateProvider`. Tickets never enter the sandbox environment or persisted user-visible workload specification.
3. OpenShell calls `StoreCredential` with its provider identity. The driver verifies the ticket, persists the binding metadata and returns an opaque handle.
4. `ResolveCredentials` validates provider ownership, auth mode, revocation and workload identity. It explicitly checks the signed creator's current `secrets.read` permission through Auth, reads the secret using the driver's identity, then rechecks workload state and local revocation before returning the value and expiry. The driver retains no value cache.
5. `DeleteCredential` tombstones the binding without deleting the NeMo secret.

Registration tickets expire within five minutes. Standalone bindings expire within one hour; workload bindings instead require the owning job attempt/step or deployment to remain active. Resolution rejects completed, cancelled, paused, deleted or replaced workloads. Returned credentials have a configurable lifetime of 300 seconds by default, capped by binding expiry for standalone bindings.

Identical `StoreCredential` retries return the same handle while the ticket remains valid. Tickets cannot be reused for another provider UUID, workspace, slot or write identity. Replacement registration stages a new handle; the previous handle remains valid until OpenShell commits the update and deletes it. The gateway is responsible for deleting abandoned staged handles.

## Persistent state

The SQLAlchemy registry stores references, ownership, retry identity and revocation metadata in SQLite, never raw tickets or credential values. It is bound to one gateway ID. Preserve the signing key and database across restarts: the registry is not a disposable cache.

Use a persistent local volume on one host. Shared network filesystems and multi-replica operation are unsupported. Tombstones have no automatic compaction.

## Run the driver

Install `nemo-deployments-plugin` with its `openshell` extra, then run:

```sh
uv run python -m nemo_deployments_plugin.backends.openshell.credential_driver.server \
  --socket-path /run/nemo-credential-driver/driver.sock \
  --state-path /var/lib/nemo-credential-driver/bindings.db \
  --gateway-id production-gateway \
  --registration-key-file /run/nemo-driver-credentials/registration.key \
  --helix-url https://helix.example \
  --helix-token-file /run/nemo-driver-credentials/helix.token \
  --cache-lifetime-seconds 300
```

For an auth-disabled Helix installation, replace `--helix-token-file` with `--auth-disabled`. A missing token never selects that mode automatically; tickets and stored bindings must match the driver's auth mode.

The registration key must contain at least 32 random bytes and be shared only with trusted provisioning. Key/token files must be regular, non-symlink files owned by the driver user with no group/other permissions. The driver limits credential files to 64 KiB. State and socket directories require mode 0700; socket and database use 0600. Run the gateway and driver with the same trusted UID. These mounts must not be visible inside sandboxes.

The driver rereads its bearer-token file for each NeMo call. Use a platform-issued service workload access token accepted by the Auth policy-evaluation endpoint. Its identity needs Secrets value access and read access to the relevant Jobs and deployment entities. Ordinary user/access-key tokens cannot perform the policy check and are rejected; the driver never impersonates a creator using identity headers. HTTPS is required by default; development HTTP requires `--allow-insecure-localhost` and a loopback URL.

Configure OpenShell to connect to the running driver:

```toml
[openshell.gateway]
credential_drivers = ["nemo"]

[openshell.credential_drivers.nemo]
transport = "uds"
socket_path = "/run/nemo-credential-driver/driver.sock"
```

Use a dedicated gateway: OpenShell selects one active credential driver, and this driver rejects raw credentials and refresh material. The driver socket exposes provider-scoped operations, not sandbox identities; OpenShell must enforce sandbox attachment authorization and network isolation.

## Protocol

The service subclasses the SDK's `CredentialDriverServicer`; `wire.py` converts SDK messages to and from validated request dictionaries. Credential enumeration is unsupported. Protocol negotiation requires the `openshell.credentials.contract` capability and protocol major version 1.

Upstream contracts: [credential driver protocol](https://github.com/NVIDIA/OpenShell/blob/v0.1.6/proto/credential_driver.proto), [provider registration](https://github.com/NVIDIA/OpenShell/blob/v0.1.6/crates/openshell-server/src/grpc/provider.rs).
