<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo credential registration driver

Experimental gateway-side implementation of OpenShell's credential-driver
protocol. It registers existing NeMo secret references through StoreCredential.
It never creates, updates, or deletes the underlying NeMo secret.

Agent execution and deployment remain gated. This implementation has protocol
tests, not a live OpenShell sandbox security acceptance test.

## Lifecycle

1. Trusted Helix provisioning selects an approved profile and authorizes the job.
   `RegistrationTickets.issue` checks secret metadata using the requesting user's
   `AsyncSecretsClient`. It returns a short-lived, signed `SecretStr` ticket.
2. Provisioning supplies the ticket as the credential value in the normal
   `CreateProvider` request. Never put it in an agent's environment or a persisted
   user-visible job specification.
3. OpenShell assigns its provider UUID and calls `StoreCredential`. The driver
   validates the ticket, saves the binding metadata in SQLite, and returns an
   opaque handle. No secret value is fetched or copied during registration.
4. `ResolveCredentials` checks the provider UUID, name, workspace, slot, binding
   expiry and job revocation. It reads the existing secret with the typed NeMo
   client, checks revocation again after the read, and returns the value with a
   lifetime of at most 30 seconds by default. The driver has no value cache.
5. `DeleteCredential` tombstones the binding. It does not call NeMo delete.

Tickets expire within five minutes and grants within one hour. Identical
Store retries return the same handle while the ticket remains valid. A ticket
cannot be replayed with another provider UUID, slot, workspace or write identity.
A fresh ticket can stage a replacement handle; the prior handle stays valid
until OpenShell commits its update and deletes it. Failed gateway creation must
also delete the staged handles.

The registry persists retry identity and revocation across restarts. It contains
references and scope metadata, never raw tickets or secret values. It is bound
to one configured gateway ID. Preserve it across restarts; do not share it across
unrelated gateways or discard it as a cache. Tombstones are currently retained
without automatic compaction. SQLite is intended for one host with a persistent
local volume, not a multi-replica deployment over a network filesystem.

The registry uses SQLAlchemy 2 typed ORM models for bindings, revoked jobs and
gateway identity. It uses an immediate write transaction for atomic
registration retries. A remote Helix entity store
is possible; running beside the gateway does not prevent using it. Local storage
keeps these metadata checks independent of another remote service, at the cost
of a persistent volume and single-host operation. Moving to the entity store
would need equivalent atomic retry/revocation semantics, restricted credentials
and explicit handling of outages.

## Provisioning-side API

Use a dedicated, random signing key shared only by the trusted provisioning
controller and this driver. This is not the service credential.

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
    ),
    requester_secrets=request_scoped_secrets_client,
)
# In the trusted controller only:
# CreateProvider.provider.credentials["API_TOKEN"] = ticket.get_secret_value()
```

Do not use the driver's service client to issue tickets: that would replace
the user's permission check with the driver's broader authority. The profile
and job authorization checks belong to the controller, before issuance.
With delegated authentication, the controller must also explicitly authorize
the effective user's `secrets.read` permission. The metadata endpoint does not
perform the value endpoint's additional on-behalf-of permission check.
The API deliberately does not expose an unauthenticated ticket-minting endpoint.

## Gateway setup

Run this module in an environment with `nemo-deployments-plugin` and its
`openshell` extra installed:

```sh
python -m nemo_deployments_plugin.backends.openshell.credential_driver.server \
  --socket-path /run/nemo-credential-driver/driver.sock \
  --state-path /var/lib/nemo-credential-driver/bindings.db \
  --gateway-id production-gateway \
  --registration-key-file /run/nemo-driver-credentials/registration.key \
  --helix-url https://helix.example \
  --helix-token-file /run/nemo-driver-credentials/helix.token
```

The key must contain at least 32 random bytes. Key/token files must be regular,
non-symlink files owned by the driver user with no group/other permissions.
State and socket directories must be private (0700); the socket and database
use 0600. Run the gateway and driver with the same trusted UID, or adapt the
deployment to provide equivalent local isolation. Mounts must not be visible
inside agent sandboxes. The bearer-token file is reread for each NeMo call.
Only HTTPS NeMo endpoints are accepted by default; development HTTP requires
`--allow-insecure-localhost` and a loopback URL.

`MAX_CREDENTIAL_FILE_BYTES` is a defensive 64 KiB input limit, not a protocol
requirement. It leaves room for a service-account JWT without reading an
arbitrarily large mounted file. Reading one extra byte distinguishes a file at
the limit from one exceeding it.

For a prestarted driver, configure OpenShell with:

```toml
[openshell.gateway]
credential_drivers = ["nemo"]

[openshell.credential_drivers.nemo]
transport = "uds"
socket_path = "/run/nemo-credential-driver/driver.sock"
```

OpenShell supports one active credential driver per gateway. This adapter rejects
raw credentials and refresh material: use a dedicated gateway for this experiment
rather than switching a gateway containing unrelated provider credentials.

## Remaining launch requirements

- Wire trusted provisioning to mint tickets, create and attach a dedicated
  provider per sandbox, and reconcile cleanup after retries and failures.
- Prevent sandbox identities from reading NeMo secrets or managing providers
  through their current Jobs/service authority.
- Enforce attachment authorization in OpenShell. The driver protocol carries
  provider identity, not sandbox identity; a unique provider name is not itself
  an attachment restriction.
- Integrate permission changes with binding revocation. `revoke_job` is a trusted
  local registry operation, not yet a controller or public API. Resolution uses
  the driver's service identity; user permission changes are not automatically
  rechecked until a new ticket is issued.
- Verify that the deployed gateway/supervisor honors returned expiry, refuses
  stale values and refreshes successfully. A driver TTL does not prove those
  behaviors. NeMo secret deletion/rotation is observed on the next resolution,
  but does not erase values already cached by OpenShell.

## Protocol and tests

`protocol/` contains the unmodified OpenShell v0.1.2 credential protocol and its
non-standard dependencies, plus a descriptor generated with
`grpcio-tools==1.78.0`. Source tree:
`6648bd0c290efbc41ba131ee9831ee45cd431f94`.
Sources: [credential driver](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/credential_driver.proto),
[datamodel](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/datamodel.proto),
[extension](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/extension.proto),
[options](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/options.proto).

The SDK omits driver stubs, so `wire.py` uses these official descriptors in a
private pool. This avoids collisions with the SDK's generated datamodel.
No field numbers or service paths are reimplemented.

Regenerate from this module directory:

```sh
uv run --no-project --with grpcio-tools==1.78.0 python -m grpc_tools.protoc \
  -I protocol --include_imports \
  --descriptor_set_out=protocol/credential_driver.pb \
  protocol/credential_driver.proto
```

Tests: `plugins/nemo-deployments/tests/unit/test_credential_driver.py`.
Coverage includes ticket integrity and authorization, replay, restart,
concurrent retries, staged updates, job revocation, expiry, permission/backend
failures, all-or-nothing batch validation, and real Unix-socket gRPC exchanges.
The SDK compatibility test constructs a real CreateProvider protobuf and
simulates the gateway's StoreCredential handoff. It does not run an OpenShell
gateway or sandbox.

## Upstream SDK simplification

The SDK bindings are being added upstream in
[OpenShell PR #4760](https://github.com/NVIDIA/OpenShell/pull/4760).
Keep the vendored protocol sources and private descriptor loader until those
bindings land and are available in the SDK version supported by Helix.

With a released SDK containing those bindings, this adapter could implement
`CredentialDriverServicer` and use
`add_CredentialDriverServicer_to_server`, replacing the private descriptor loader
and generic gRPC registration. No gateway protocol change is required.
The current `wire.py` remains compatible with the pinned 0.1.2 SDK.
