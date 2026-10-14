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

## Patched W&B core

`wandb-core-artifacts` builds W&B **0.30.0** from release commit
`e5ec9dc9236d1a4fbb9fe7c0795bbccdff1e0e77`, using **Go 1.27.2** and
**golang.org/x/net v0.60.0**. PyPI 0.30.0 still bundles vulnerable Go 1.27.1 /
x/net v0.59.0, so a Python package upgrade alone is insufficient.

`docker/base/Dockerfile.wandb-core` cross-compiles once per architecture on the
build platform. It replaces only the vendored x/net module, preserving all
other vendored files (including upstream's custom Arrow patches) and the
upstream `disable_grpc_modules,parquet_read_only` tags. Do **not** regenerate
the entire vendor tree. The artifact includes binary SHA-256, source commit,
module checksum, architecture, and verified `go version -m` output.

Consumers inherit `_with-wandb-core` in Bake and mount the artifact into a final
build step. `docker/scripts/install-wandb-core.py` updates every discovered
installed binary and its wheel RECORD, including extracted uv caches shared by
RL/Gym environments. It breaks hardlinks before writing, so build-only caches
are not modified. Retained compressed W&B wheels are removed. Version or
architecture mismatches fail the build rather than mixing SDK/core versions;
upgrade the Python SDK or remove an **unused** stale cache before retrying.
Never delete a cache that prefetched environments symlink into.

The final image contains no Go toolchain or Go module cache from this builder.
BuildKit caches and original binaries in ancestor image layers are not erased;
verification covers the merged runtime filesystem. Future runtime dependency
installs can reintroduce an upstream binary and must run the installer again.
Image rebuilds and container scans remain required for release validation.

Validate the shared build and a real offline logging run:

```bash
docker buildx bake wandb-core-smoke-test --set '*.platform=linux/amd64'
docker buildx bake wandb-core-artifacts --set '*.platform=linux/amd64,linux/arm64'
```

When changing W&B versions, update the release commit/archive checksum, builder
and installer expectations, Docker package pins/overrides and relevant uv
locks together. Re-run metadata verification, offline logging, and downstream
image builds for both supported architectures before shipping.
