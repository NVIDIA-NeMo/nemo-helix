<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Help the user get Harbor ready

Use this when the [runtime prerequisite checks](../SKILL.md#runtime-prerequisite-checks) cannot
use Harbor. This reference owns installation remedies; the shared
[Milestone check-ins](../../eval-author/references/milestone-checkins.md) procedure owns
explaining the prerequisite and deciding what work follows.

## Resolve the existing installation first

Follow discovery's probes before concluding Harbor is missing. If a launcher
exists but fails, investigate its actual error before suggesting another
installation. A failed import in one environment is not a reason to upgrade or
reinstall. Follow the repository's documented environment and version requirements
when present rather than introducing a competing installation.

## Install when needed

If the repository does not manage Harbor, the stable installation in
[Harbor's getting-started guide](https://www.harborframework.com/docs/getting-started)
uses uv, a Python package and tool manager:

```bash
uv tool install harbor
```

Explain that this installs Harbor as a command-line tool. Provide it as a user-run
instruction by default; a broad request to get evals working does not authorize
installation or upgrades. With explicit installation authorization, use the
host's normal permission mechanism and proceed to verification.

If uv is missing, link the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
for the user's operating system. Offer guided setup rather than multiple
package-manager alternatives. If network, permissions, or a sandbox blocks
installation, record the actual error and explain how to run the documented setup
in an appropriate terminal. Do not change repository dependencies to work around
a blocked tool installation.

## Return to verification

After installing or repairing Harbor, return to discovery's **Runtime prerequisite
checks** using the resulting installation. Discovery owns interpreter and CLI
verification, the optional assistant-skills check, and saved setup evidence.
During prerequisite-only work, return that setup evidence to the calling stage;
do not start suite inventory or source selection. If the caller is already in
full discovery and an earlier report was blocked, rerun its pass with the verified
interpreter and replace the report. Docker, other task backends, credentials, and
application access belong to the selected task's environment and connection work.
