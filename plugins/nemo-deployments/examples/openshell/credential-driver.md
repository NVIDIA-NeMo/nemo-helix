<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo credential driver beside a local OpenShell gateway

This opt-in Compose overlay runs the driver in the Helix API image with a different entrypoint. Use a dedicated gateway: the driver expects registration tickets and cannot serve unrelated raw provider credentials. The gateway, supervisor and sandbox runtime use OpenShell 0.1.6.

The startup order is `credential-init` → healthy `credential-driver` → `gateway`. Initialization copies the operator's mounted files into private, container-owned files so Docker host UID differences do not break the driver's ownership checks. The driver and the local-development gateway run as UID 0, matching the existing example. The driver has no Docker socket, published port, or gateway state mount. Its capabilities are dropped and its root filesystem is read-only.

## Build the existing API image

Use a dedicated Docker host with sufficient memory for the full API build, including Studio and native dependencies. The container smoke test below is isolated by Compose project, but image-build resource exhaustion affects the whole Docker host.

From the repository root, build and load the current source with the existing image target:

```sh
make docker-load DOCKER_TARGET=nhx-api-docker IMAGE_REGISTRY=local BAKE_TAG=credential-driver DOCKER_PLATFORMS=linux/arm64
```

Use `linux/amd64` on an x86 host. This builds the full API image, including the driver and OpenShell SDK; no separate driver image is required.

Before supplying real credentials, run the isolated image smoke test from the repository root:

```sh
RUN_CREDENTIAL_DRIVER_COMPOSE_TEST=1 NEMO_HELIX_IMAGE=local/nhx-api:credential-driver \
  uv run --frozen pytest \
  plugins/nemo-deployments/tests/integration/backends/openshell/test_credential_driver_sidecar.py \
  --basetemp="$PWD/.pytest_cache/credential-sidecar" -v
```

The test uses synthetic credentials to check registration, restart recovery, token rotation and persistence in an isolated Compose project. It starts no gateway and accesses no real Helix endpoint. The temporary directory must be shared with Docker; the command above uses the checkout's ignored pytest cache. The test removes its own containers and volumes.

## Supply driver credentials

Use a private directory outside the checkout. It must contain two regular, non-symlink files:

- `registration.key`: at least 32 random bytes. Share this exact key only with the trusted Helix controller. It signs registration tickets; it is not a service credential. Preserve it with the binding database.
- `helix.token`: a platform-issued service workload access token accepted by the Auth policy-evaluation endpoint. Its identity needs Secrets value access and read access to the relevant Jobs and deployment entities. Ordinary user/access-key tokens are not sufficient. The driver checks the signed creator’s permission explicitly before reading a value; it does not impersonate the creator through request headers.

Keep the source directory mode 0700 and both files mode 0600. Each file must be no larger than 64 KiB. The initializer creates private destination files and never prints their contents. It refuses to overwrite an existing signing key with different bytes. Token replacement is allowed and atomic.

Set these variables, replacing the credential directory and Helix URL with your actual values:

```sh
export NEMO_HELIX_IMAGE=local/nhx-api:credential-driver
export NEMO_DRIVER_CREDENTIALS_DIR=/absolute/private/path/to/driver-credentials
export NEMO_DRIVER_HELIX_URL=https://helix.example
export NEMO_DRIVER_GATEWAY_ID=docker-dev
export NEMO_DRIVER_CACHE_LIFETIME_SECONDS=300
```

`NEMO_DRIVER_GATEWAY_ID` is the registration-ticket audience shared by the Helix controller, driver and binding database. It does not configure OpenShell’s own gateway identity. The example’s separate JWT issuer setting, `openshell.gateway.gateway_jwt.gateway_id`, remains `docker-dev` in `gateway-credentials.toml`.

The URL must be reachable from the driver container. `localhost` inside that container is not your host. The overlay adds `host.docker.internal` for a host endpoint; HTTPS with a trusted certificate is required. For a private CA, mount its PEM bundle read-only through an additional Compose overlay and set `SSL_CERT_FILE` in `credential-driver.environment`.

## Start the sidecar, then the gateway

From this directory, define a helper so every operation uses the same project and both files:

```sh
dc() {
  docker compose -p nemo-secrets-dev \
    -f docker-compose.yml -f docker-compose.credentials.yml "$@"
}
dc config --quiet
dc up -d --wait credential-driver
dc ps -a
```

A healthy driver means its local `GetCapabilities` RPC succeeds. It does not prove that the Helix token is authorized or that remote secret access works. No real secret is read by this probe.

Before starting the gateway, complete the base example's JWT-key generation and ensure its `/var/lib/openshell` host directory and port 17670 are not used by another gateway. A different Compose project name isolates the new named volumes, but does not isolate the base example's bind-mounted gateway state or published port. Do not run this against an existing gateway's data directory. The `gateway-credentials.toml` file preserves the base example's pinned runtime settings and adds the driver socket configuration.

```sh
dc up -d gateway
dc logs --tail=50 credential-driver gateway
```

Check that OpenShell completes credential-driver protocol negotiation. Gateway/driver startup is separate from sandbox credential expiry and refresh acceptance.

## Connect Helix provisioning

For a Helix installation with authentication disabled, add `-f docker-compose.credentials-noauth.yml` after the credentials overlay in the helper above. Supply `registration.key`; `helix.token` is unnecessary. The override explicitly selects `--auth-disabled` for initialization and the driver. Changing auth modes requires recreating the workload's providers; bindings from the other mode are rejected. TLS and sandbox network restrictions still apply.

Mount the same signing key into the trusted Helix controller, then configure its OpenShell executor or job execution profile with the matching gateway ID and approved injection profiles. Follow the [environment integration guide](../../../nemo-agents/docs/provider-fulfillments.md#operator-configuration). These controller settings are not set by the sidecar overlay.

`NEMO_DRIVER_CACHE_LIFETIME_SECONDS` configures the driver’s advertised credential lifetime (default 300; range 1–3600 seconds). This applies in both auth modes. Standalone grants are also capped by binding expiry; workload grants check live workload state on resolution. Change it by recreating the driver container. Previously resolved credentials retain their original deadlines.

Starting the sidecar alone does not enable controller renewal. Keep `credential_refresh_verified: false` until the installation passes the lifetime, rotation, cleanup and isolation checks in the environment integration guide. Driver readiness and protocol negotiation do not establish those guarantees.

## Restart, rotate and stop

The driver has `restart: unless-stopped`. On process startup it takes an exclusive socket lock. It can remove its own stale socket after a crash, but refuses to replace a live listener, a symlink, a regular file or another user's socket. The binding database survives container recreation.

The SQLite registry stores binding metadata and revocation tombstones, never raw tickets or credential values. Keep it on a persistent local volume on one host; shared network filesystems and multiple driver replicas are unsupported. Tombstones have no automatic compaction.

After replacing the source `helix.token`, restage it:

```sh
dc run --rm --no-deps credential-init
```

The running driver rereads the staged token on each NeMo call. The source bind mount is not read continuously, so updating only the host file is insufficient. Signing-key rotation needs an explicit migration and is deliberately not handled by this command.

```sh
dc restart credential-driver
dc ps
dc down
```

`down` retains the binding and credential volumes. Do not use `down -v` as routine cleanup: removing the registry also removes replay/revocation history. Socket, state and credential volumes are separate; only the socket volume is shared with the gateway. None is passed into agent sandboxes.
