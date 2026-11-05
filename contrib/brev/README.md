<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix on Brev (published wheel)

This directory hosts the VM setup script for a [Brev Launchable](https://docs.nvidia.com/brev/latest/concepts/launchables).
The Launchable record and shareable deployment link live in the Brev console, not
in Git. The script installs a released wheel; it does not build this checkout.

The goal is a **published, reusable Launchable in Brev Explore** (a prefab), not
an install recipe every deployer must assemble. A maintainer creates and verifies
it once; deployers then supply their own API key and click Deploy. No Launchable
has been created or verified yet.

## Maintainer: create the candidate Launchable

1. In the Brev console, create **NeMo Helix — hosted inference** using **VM
   Mode**, with description: "Launch NeMo Helix Studio with your own OpenRouter
   API key. No local GPU or source build required." Start with 8 vCPUs, 32 GiB
   RAM and 100 GiB disk as an initial sizing estimate, not a measured minimum.
2. Use an Ubuntu VM with systemd, Docker running and accessible **without sudo**
   to the VM user, passwordless sudo, curl, and CA certificates. The script installs
   native build prerequisites using apt; deployers should not need an SSH session.
   Configure Docker group membership in the image and re-login before setup;
   changing membership inside a running setup shell does not update its groups.
3. Set the source to this repository at a reviewed commit, or choose no source
   files and fetch the script in the setup field as shown below.
4. Configure the launch parameters below.
5. Add a **Secure Link** named **Studio**, port **8080**, as the deployment CTA.
   Do **not** add a public TCP rule for 8080. The supervised server binds to all
   VM interfaces so the proxy can reach it; use Brev network controls to prevent
   bypassing Secure Link authentication. This is a single-user development VM,
   not a hardened multi-tenant installation.
6. Initially choose **Only my organization** and create the candidate. Verify it
   on a fresh VM before publishing. The image must run setup as the non-root VM
   user with a working systemd user session; treat this as an acceptance check,
   not an assumption about every Brev image.

Use this setup field, replacing the commit placeholder with a full reviewed SHA
from the repository containing this file:

```bash
#!/usr/bin/env bash
set +x
set -euo pipefail
script=$(mktemp)
trap 'rm -f "$script"' EXIT
curl -fsSL 'https://raw.githubusercontent.com/NVIDIA-NeMo/nemo-helix/<COMMIT_SHA>/contrib/brev/setup.sh' -o "$script"
bash "$script"
```

Run as the VM user, not root. If the configured Brev image runs setup as root,
configure a user-level startup wrapper that deliberately passes the selected
launch parameters; do not silently install into root's home. For a private repo,
use Brev's supported source authentication rather than embedding a token in the
fetch URL. Pinning a commit prevents later repository edits changing a launch.

### Launch parameters

| Name | Required | Default / guidance |
| --- | --- | --- |
| `NEMO_HELIX_VERSION` | Yes | Choice with one maintainer-tested, exact published release; set it as the default so deployers need not select a version |
| `OPENROUTER_API_KEY` | Yes | Text with **no default**; enter a key or select a Brev organization secret |
| `NEMO_DEFAULT_MODEL` | No | Model **entity ID**, not upstream served-model name |
| `NEMO_FAST_MODEL` | No | Model entity ID; otherwise reuses the default |

Before publication, replace the version Choice with the exact release tested
below. Do not use `latest` or assume that a source release branch already has a
matching published wheel. Similarly, set tested model entity IDs as defaults
for the two model parameters so a prefab does not rely on discovery order.

Keep the first published prefab OpenRouter-only to avoid a form with several
optional keys and no way to express "at least one required". For a separate,
advanced generic-provider launchable, make `OPENROUTER_API_KEY` optional and add
optional Text parameters `NEMO_DEFAULT_INFERENCE_KEY`,
`NEMO_DEFAULT_INFERENCE_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and
`GEMINI_API_KEY`. At least one supported key must be supplied; the script checks
this before installation. Use no credential defaults.

The CLI selects the first match:

```text
NEMO_DEFAULT_INFERENCE_KEY > OPENROUTER_API_KEY > OPENAI_API_KEY > ANTHROPIC_API_KEY > GEMINI_API_KEY
```

For NVIDIA Build, use `NEMO_DEFAULT_INFERENCE_KEY` and
`NEMO_DEFAULT_INFERENCE_BASE_URL=https://integrate.api.nvidia.com`.
For an alternative NVIDIA endpoint, use
`NEMO_DEFAULT_INFERENCE_BASE_URL=https://inference-api.nvidia.com/v1`.
`NVIDIA_API_KEY` alone does not create an auto provider. A base URL must be paired
with the generic key. With no explicit model IDs, setup selects the first
model discovered; that may not be the model you want.

## What setup does

- Checks the key, Docker access, and systemd user session.
- Installs native build prerequisites with apt and enables user lingering (via
  sudo) so services survive logout and reboot.
- Installs uv 0.13.4 if missing, and installs `nemo-helix[all]` at the selected
  release using Python 3.12 in a dedicated uv tool environment.
- Runs `nemo setup --auto --start-services --ready-timeout 360`.
- Stops the initial background process and transfers ownership to an enabled
  systemd user service. It does not rerun provider setup on reboot.
- Waits for `/health/ready` and lists workspaces, providers, and models.

Brev launch parameters are available only during initial setup. Provider keys
are registered in NeMo's secret store; the script does not write them to shell
profiles, the service unit, or an environment file. Keep the complete data
folder, including its encryption key, private. Privileged VM users can access
runtime secrets. Never enable shell tracing around credential handling.

State: `~/.local/share/nemo`. CLI configuration: `~/.config/nhx`.
Wheel environment: `~/.local/share/nemo-brev`. Unit:
`~/.config/systemd/user/nemo-helix.service`.
Existing state is never deleted. Setup refuses to proceed if the platform is
already running; stop it intentionally before retrying, and supply launch
parameters again. Installation/setup failures may leave partial state; inspect
logs rather than deleting the data directory.

## Verify on a fresh VM before publishing

```bash
curl -fsS http://localhost:8080/health/ready
~/.local/share/nemo-brev/bin/nemo workspaces list
~/.local/share/nemo-brev/bin/nemo inference providers list --workspace default
~/.local/share/nemo-brev/bin/nemo models list --workspace default
systemctl --user status nemo-helix.service --no-pager
```

Open Studio through the Secure Link and test an inference request with the
selected model. Reboot the VM, confirm readiness and Studio access again, and
verify provider configuration survived without resupplying a key. Verify port
8080 is not directly accessible from the Internet. Repeat with the generic
key/base-URL variant before advertising it.

Logs and lifecycle:

```bash
journalctl --user -u nemo-helix.service -n 100 --no-pager
systemctl --user restart nemo-helix.service
systemctl --user disable --now nemo-helix.service
```

Disabling the unit does not delete data or necessarily remove managed Docker
containers. Stop/delete the Brev VM through its console when finished to avoid
ongoing compute charges.

## Maintainer: publish the prefab

After the fresh-VM checks pass:

1. Record the reviewed script commit, exact wheel version, VM image, compute
   size, model entity IDs, and verification date in the Launchable description.
   Record startup time so deployers know how long to wait for Studio.
2. Deploy the candidate again as a second user with their own key and no SSH
   setup. Confirm it creates independent platform state and does not reuse the
   maintainer's credentials. Verify optional model parameters have safe defaults.
3. In **My Launchables**, enable **Publish to community** / **Everyone
   (published)**. This makes the Launchable discoverable in Explore; being
   selected as a Featured pick is separate and is not guaranteed.
4. Copy the real deployment URL and **Copy Markdown Badge** from Brev. Add them
   here only after verification; do not publish a placeholder deployment link.
5. For updates, validate a private candidate before changing the published
   configuration. Keep the script commit and wheel release pinned, and rerun
   launch, inference, isolation, and reboot checks when either changes.

Once published, the deployer flow should be: open the badge or Explore card,
enter an API key, choose compute, Deploy, and open Studio. If SSH or manual
package installation is needed, the candidate is not ready to publish.

## Local checks

```bash
bash -n contrib/brev/setup.sh
uv run --no-project --with shellcheck-py shellcheck contrib/brev/setup.sh
```

There is intentionally no fully mocked installation test: it cannot establish
wheel compatibility, systemd behavior, or Brev connectivity. The fresh-VM checks
above are the acceptance test for publication.
