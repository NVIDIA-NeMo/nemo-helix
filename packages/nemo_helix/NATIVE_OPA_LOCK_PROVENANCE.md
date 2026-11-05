<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native OPA cutover: dependency lock provenance

The canonical DataDesignerPlugins index returned TLS connection resets during
this cutover. Its official release assets were reachable. A temporary local
mirror, generated with upstream's own documentation-index generator, allowed
**uv 0.13.4** to resolve the real dependency graph without fabricating cache
metadata or editing package versions, dependencies, or hashes.

## Authoritative inputs

- Generator repository: `NVIDIA-NeMo/DataDesignerPlugins`, commit
  `cd745a56c9ac99f7b8a5068be873474e9d75f82a` (identified from the downloaded
  GitHub archive's commit header).
- Commit archive URL:
  `https://codeload.github.com/NVIDIA-NeMo/DataDesignerPlugins/tar.gz/cd745a56c9ac99f7b8a5068be873474e9d75f82a`
- Measured commit archive SHA256:
  `b72ab08ebdee112bf2d5676a4e889fb5384c6d28f77ca4121ff5dd580573f266`.
- Official release base:
  `https://github.com/NVIDIA-NeMo/DataDesignerPlugins/releases/download/ddp-package-assets`.
- Downloaded `packages.json` SHA256:
  `ef692401defab6e64f75e86040878cc96191572f2737774a45d53ac494b12db4`.
- Downloaded `data_designer_retrieval_sdg-0.2.1-py3-none-any.whl` SHA256:
  `5c2b5d7da3340ed730d03bc0163318fae96656e4e414e8907b295ab72ea6fffd`.
  This matches **both** the publisher's manifest and the preexisting `uv.lock`.

The archive/manifest measurements describe these exact fetched inputs, not a
publisher signature. None of these are OPA binary hashes. The upstream
`.github/workflows/docs.yml` builds its Pages index from that release manifest;
there is no `gh-pages` source branch containing the generated index HTML.

## Reproduce the mirror and resolution

Run from the NeMo repository root. Use a scratch interpreter/environment for
the upstream generator; do not synchronize another worktree's environment.
The example uses the absolute path to an existing Python interpreter as
`$PYTHON`, with uv providing generator dependencies in an ephemeral environment.

```bash
repo=$PWD
scratch=$(mktemp -d)
commit=cd745a56c9ac99f7b8a5068be873474e9d75f82a
release=https://github.com/NVIDIA-NeMo/DataDesignerPlugins/releases/download/ddp-package-assets
curl -fLsS --connect-timeout 5 --max-time 30 \
  "https://codeload.github.com/NVIDIA-NeMo/DataDesignerPlugins/tar.gz/$commit" -o "$scratch/generator.tar.gz"
curl -fLsS --connect-timeout 5 --max-time 30 "$release/packages.json" -o "$scratch/packages.json"
curl -fLsS --connect-timeout 5 --max-time 30 \
  "$release/data_designer_retrieval_sdg-0.2.1-py3-none-any.whl" -o "$scratch/plugin.whl"
printf '%s  %s\n' \
  b72ab08ebdee112bf2d5676a4e889fb5384c6d28f77ca4121ff5dd580573f266 "$scratch/generator.tar.gz" \
  ef692401defab6e64f75e86040878cc96191572f2737774a45d53ac494b12db4 "$scratch/packages.json" \
  5c2b5d7da3340ed730d03bc0163318fae96656e4e414e8907b295ab72ea6fffd "$scratch/plugin.whl" | sha256sum -c -
tar -xzf "$scratch/generator.tar.gz" -C "$scratch"
upstream="$scratch/DataDesignerPlugins-$commit"
(
  cd "$upstream"
  PYTHONPATH="$upstream/devtools/ddp/src" uv run --no-project \
    --with 'dumb-pypi>=1.15.0' --with packaging --python "$PYTHON" \
    python -m ddp.package_index build --package-list "$scratch/packages.json" \
    --packages-url "$release" --site-dir "$scratch/mirror"
)
cd "$repo"
uv --system-certs lock --index "data-designer-plugins=file://$scratch/mirror/simple"
```

The generator writes PEP 503 pages with authentic official release URLs and
publisher hashes. uv fetches those artifacts or reuses their authentic cached
metadata. The named-index override is command-local: **do not change project
index configuration**. If the upstream mutable manifest changes, do not ignore
a checksum failure; inspect and record the new authoritative inputs.

Restore only the generated mirror source identifiers to the canonical index:

```bash
MIRROR="$scratch/mirror/simple" uv run --no-project python - <<'PY'
import os
from pathlib import Path

path = Path("uv.lock")
text = path.read_text()
mirror = os.environ["MIRROR"]
canonical = "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
for old, new in (
    (f'registry = "{mirror}"', f'registry = "{canonical}"'),
    (f'index = "file://{mirror}"', f'index = "{canonical}"'),
):
    assert text.count(old) == 1, (old, "unexpected lock shape; stop and inspect")
    text = text.replace(old, new)
path.write_text(text)
PY
uv lock --offline --check
git diff -- uv.lock
```

For this cutover, the final lock diff is exactly **16 deleted lines** removing
the obsolete Regorus package and references: no other versions, graph edges,
URLs, artifact hashes, or project index configuration changed. The canonical
`uv lock --offline --check` succeeded after restoration.

The normal license generator was then run without environment synchronization:
`nemo-helix-tools license generate --sequential`, using current source through
`PYTHONPATH`. This regenerated requirements and license outputs. Its scanner
reported remote RPC TLS resets and reused preexisting raw output. Re-merging
configured jobs-launcher/callouts Go scans duplicated those existing sections;
that raw-output delta is not a valid fresh scan and must be discarded. The
UV-exported requirements and final formatted license delta (only removing the
obsolete binding) remain useful. This does not claim a fresh successful remote
vulnerability scan.

## Remaining verification limits

The universal SDK wheel was rebuilt from the native runtime and inspected for
absence of binding code/imports/dependencies and platform OPA executables.
Frozen installs no longer select the removed binding. The official OPA image's
multi-platform index digest and Linux amd64/arm64 entries were subsequently
verified with Docker manifest/imagetools inspection and pinned in both
Dockerfiles; see [BUNDLING.md](BUNDLING.md). The pinned image subsequently ran
successfully without network, and its extracted Linux amd64 binary passed
runtime tests. ARM64 execution, full NeMo Docker image builds, and live official
release binary/checksum downloads remain separate verification gates; a passing
lock check or SDK build does not establish those results.
