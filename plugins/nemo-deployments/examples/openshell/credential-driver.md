<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Connect a local OpenShell gateway to Helix secrets

The NeMo credential driver lets OpenShell retrieve Helix secrets for agents and store its own SSH keys and launch material. This example runs it alongside an OpenShell 0.1.6 gateway whose providers are registered by Helix.

Before starting, follow the JWT-key generation instructions in [docker-compose.yml](docker-compose.yml) and check that another gateway is not using `/var/lib/openshell` or host port 17670. Both are fixed in the base configuration, regardless of the Compose project name.

## Prepare the credentials

The driver needs a signing key shared with the Helix controller and, when authentication is enabled, a token for calling Helix. Create a private directory outside the checkout with mode `0700` for these files:

- `registration.key` contains at least 32 random bytes. The controller uses this key to sign credential registrations, and the driver uses it to verify them.
- `helix.token` contains a platform-issued internal service workload access token. Its identity needs Secrets value access, read access to relevant Jobs and deployment entities, and permission to initialize the internal workspace and inspect its access grants. An ordinary user or service-account token is not sufficient.

At startup, the driver creates or validates `openshell-internal` for encrypted gateway-owned values. It rejects workspace grants to ordinary users and ensures its service identity can store, read and delete these values. Deleting an internal handle removes its owned Secret; deleting a provider binding leaves the user's source Secret intact.

The driver does not increase service permissions in an existing workspace. If initial setup stops between workspace creation and granting service access, an operator must grant the driver `ServiceSystem` in `openshell-internal` before restarting it.

The initializer copies these files into the driver's private credential volume. Each must be a regular file rather than a symlink, no larger than 64 KiB, with mode `0600`.

For Helix installations with authentication disabled, only `registration.key` is needed. Add `-f docker-compose.credentials-noauth.yml` after `-f docker-compose.credentials.yml` in the Compose commands below. If you are switching an existing installation between authentication modes, its workload providers must be recreated because bindings cannot be reused across modes.

## Configure and start the driver

Set the image tag to match your Helix installation, then fill in the credential directory and Helix URL. The gateway ID must match the Helix controller's credential configuration; it is separate from OpenShell's gateway JWT settings in `gateway-credentials.toml`.

```sh
export NEMO_HELIX_IMAGE=ghcr.io/nvidia-nemo/nemo-helix/nhx-api:YOUR_TAG
export NEMO_DRIVER_CREDENTIALS_DIR=/absolute/private/path/to/driver-credentials
export NEMO_DRIVER_HELIX_URL=https://helix.example
export NEMO_DRIVER_GATEWAY_ID=docker-dev
export NEMO_DRIVER_CACHE_LIFETIME_SECONDS=300
```

The Helix URL must be reachable from the driver container over HTTPS with a trusted certificate. The overlay provides `host.docker.internal` for reaching Helix on the Docker host. For certificates issued by a private CA, mount the CA's PEM bundle read-only through an additional Compose overlay and point `SSL_CERT_FILE` in `credential-driver.environment` to that path.

OpenShell may cache resolved provider credentials for `NEMO_DRIVER_CACHE_LIFETIME_SECONDS`, which defaults to 300 seconds and accepts values from 1–3600. Changing this setting requires recreating the driver container and affects newly resolved credentials; those already cached retain their existing deadlines. Internal SSH keys and launch material remain valid until OpenShell deletes their handles.

From this directory, start the driver and wait for its readiness check:

```sh
docker compose -p nemo-secrets-dev -f docker-compose.yml -f docker-compose.credentials.yml up -d --wait credential-driver
```

Once it is healthy, start the gateway and check the logs for credential-driver connection errors:

```sh
docker compose -p nemo-secrets-dev -f docker-compose.yml -f docker-compose.credentials.yml up -d gateway
docker compose -p nemo-secrets-dev -f docker-compose.yml -f docker-compose.credentials.yml logs --tail=50 credential-driver gateway
```

## Connect the Helix controller

The Helix controller registers workload providers with the driver using the shared signing key and gateway ID. Mount `registration.key` into the controller and configure its OpenShell executor or job execution profile with that ID and the approved injection profiles, as described in the [environment integration guide](../../../nemo-agents/docs/provider-fulfillments.md#operator-configuration).

The readiness check confirms that the driver is listening, but does not exercise Helix authorization or credential injection. Leave `credential_refresh_verified: false` in the controller's `gateway_credentials` configuration until the installation passes the lifetime, rotation, cleanup and isolation checks in that guide.

## Rotate the service token

When the service token changes, replace the source `helix.token` and run the initializer again to copy it into the credential volume:

```sh
docker compose -p nemo-secrets-dev -f docker-compose.yml -f docker-compose.credentials.yml run --rm --no-deps credential-init
```

The running driver reads that copy on subsequent Helix requests, so it does not need a restart. The initializer refuses changes to `registration.key`; this command rotates only the service token.

Keep the signing key and the entire driver state directory across restarts, including the binding database and its `.owned-key` file. Recovery also depends on the Helix Secrets database and its encryption keys. This setup supports a single driver instance with persistent storage on one host.

The driver can also run as a foreground systemd service in a dedicated venv using `python -m nemo_deployments_plugin.backends.openshell.credential_driver.server`. The gateway and driver need the same trusted Unix user to share the private socket. Keep persistent state outside the venv and start the driver after Helix is available.
