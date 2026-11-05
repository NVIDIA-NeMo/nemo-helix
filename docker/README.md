<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Docker Builds

Dockerfiles and Docker build support files live under this directory. The
canonical Buildx Bake file is `docker-bake.hcl`.

Examples:

```bash
docker buildx bake --print docker-cpu
docker buildx bake --print nhx-automodel
docker buildx bake --print nhx-unsloth
```

## Release image CVE floors

`python-cve-overrides.txt` records the urllib3 2.8.0 and fsspec 2026.6.0
security floors. `scripts/patch-python-cve-deps.py` applies them after dependency
provisioning to the shared task venv (inherited by OpenShell and Agent Hardener),
Agent Hardener's isolated venvs, every prefetched Gym-host component venv, and
the Unsloth/Automodel training venvs. Only already-installed packages are
requested. Older Datasets releases must move to the reviewed workspace floor
of 5.1 to admit patched fsspec; fsspec retains that release's 2026.7 upper bound.

The release task workspace remains frozen: a full workspace relock was blocked
locally by connectivity to the DataDesigner plugin index. The explicit
post-sync upgrade runs before source collection instead. The standalone
Gym-host lock is regenerated with uv. Gym's stale package cache entries are
removed after patching, and the GPU images remove the shadowed NGC fsspec copy
only after installing a patched venv-local copy.

Local regression tests (no platform instance or credentials required):

```bash
uv run --no-project --with pytest --with packaging python -m pytest \
  -c /dev/null --noconftest docker/scripts/tests/test_patch_python_cve_deps.py \
  tests/unit/test_docker_workspace_slices.py
```

Rebuild the five affected images, run their image smoke tests, and rescan the
resulting linux/amd64 digests before treating the reported findings as closed.
