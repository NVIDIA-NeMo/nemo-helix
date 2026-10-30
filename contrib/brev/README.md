<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix on Brev (published wheel)

This directory hosts the VM setup script for a [Brev Launchable](https://docs.nvidia.com/brev/latest/concepts/launchables).
The Launchable record and shareable deployment link live in the Brev console, not
in Git. The script installs a released wheel; it does not build this checkout.

## Create the Launchable

1. In the Brev console, create a Launchable using **VM Mode**. Hosted inference
   does not require a GPU. Start with 8 vCPUs, 32 GiB RAM and 100 GiB disk as an
   initial sizing estimate, not a measured minimum.
2. Use an Ubuntu VM with systemd, Docker running and accessible **without sudo**
   to the VM user, passwordless sudo, curl, CA certificates, a C compiler, and
   Python development headers (`build-essential python3-dev` on Ubuntu).
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
6. Choose view/deploy access and create the Launchable. Share its generated link.

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
| `NEMO_HELIX_VERSION` | Yes | Exact published version, e.g. `0.8.1`; validate the selected release before sharing |
| `OPENROUTER_API_KEY` | Yes, for the simplest OpenRouter launchable | No default; enter a key or select a Brev organization secret |
| `NEMO_DEFAULT_MODEL` | No | Model **entity ID**, not upstream served-model name |
| `NEMO_FAST_MODEL` | No | Model entity ID; otherwise reuses the default |

For a generic-provider launchable, make `OPENROUTER_API_KEY` optional and add
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
- Enables user lingering (via sudo) so services survive logout and reboot.
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

Local script contract tests (mocked commands, no VM or credentials needed):

```bash
uv run --frozen pytest contrib/brev/tests/test_setup.py -v
bash -n contrib/brev/setup.sh
```

These checks are not a substitute for the fresh-VM verification above.
