<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# W&B core container remediation

## Scope and evidence

Branch: `centralized-wandb-core-patch/mkornfield`, in an isolated worktree.
Prepared as a separate draft PR.
Python runtime and jobs-launcher changes are intentionally excluded.

Reviewed Pulse run **38554823128**, image revision
`84402a4698c9e8163dd0b204e57937ab1d19b393`:

- Extracted reports/SBOMs under `/tmp/pulse-38554823128/` for nhx-api,
  nhx-automodel-training, nhx-customizer-tasks, nhx-garak-tasks, nhx-gym-host,
  nhx-gym-tasks, nhx-rl-training, nhx-safe-synthesizer-tasks, and nhx-unsloth-training.
- Matching ten ZIPs under `/tmp/pulse-zips-38554823128/`.
- Supplied normalized `/tmp/pulse-summary.txt` and raw `/tmp/pulse-high.tsv`.
- Upstream release/source inspection, including `core/hatch.py`, `hatch_build.py`,
  `core/go.mod`, and `pyproject.toml` (inspection copies under `/tmp/wandb-*`).

The address-cves summarizer was rerun against all ten ZIPs with `--limit 0`.
Counts are scanner-effective severity, per package/image/location finding:

| Category before remediation | Critical | High |
| --- | ---: | ---: |
| Project dependencies | 0 | 0 |
| Container-only, all reports | 61 | 253 |
| W&B core subset (included above), seven images | 22 | 110 |

W&B remains at PyPI **0.30.0**: the scanned wheel embeds Go **1.27.1** and
x/net **v0.59.0**. There is no newer reviewed upstream wheel supplying this fix.

## Changes

- Shared `wandb-core-artifacts` Bake target, cross-compiled once per architecture
  using Go **1.27.2**. Only binary/provenance/helper artifacts are exported;
  no Go toolchain or Go module cache is copied into consumers.
- W&B source pinned to release commit
  `e5ec9dc9236d1a4fbb9fe7c0795bbccdff1e0e77` and archive SHA-256
  `0be0f4f840f0df49d558bc63987a90a2c2bcf812cee7d9db746591855eebaeb5`.
- Replace only vendored **golang.org/x/net v0.60.0**, verified through Go's
  checksum database and pinned checksum
  `h1:79p50tfZlm0J9YfoDsSi639qSXNGVwEzOPLCxM2FsYU=`.
  Preserve/hash-check **5,642 other vendor files**; do not regenerate vendor or
  discard W&B's custom Arrow patches. Preserve upstream build tags
  `disable_grpc_modules,parquet_read_only`, vendor mode, and commit ldflag.
- Shared install/verify helper runs after final dependency installs in API,
  CPU task, Garak, Gym task/host, Customizer, Safe Synthesizer, Automodel
  base/training, RL base/training, and Unsloth build paths. Bake wiring includes
  their smoke/E2E/OpenShell variants (22 inherited consumers including the
  dedicated shared smoke test).
- Installer checks SDK version, architecture, binary hash, and security baseline;
  updates installed/cache binaries and RECORD hashes. Atomic replacement breaks
  hardlinks to build-only caches. Retained extracted caches are patched, not
  deleted; original compressed W&B wheels are removed. Missing cores, broken or
  escaping symlinks, and mismatched retained SDK versions fail the build.
- Container W&B pins/overrides are **==0.30.0** to match the rebuilt core. This is
  an interim downstream rebuild, not a claim that the upstream wheel is fixed.
  Update pins/source/checks together when a fixed upstream release is available.
- Updated Docker documentation, third-party source provenance in NOTICE, and
  regression tests. No scanner suppressions or VEX exceptions added.

## Findings targeted by these changes

| Component | Old → rebuilt | Finding IDs |
| --- | --- | --- |
| Go stdlib | go1.27.1 → go1.27.2 | GO-2026-6603, GO-2026-6605, GO-2026-6607, GO-2026-6608, GO-2026-6610, GO-2026-6611, GO-2026-6612, GO-2026-6613 |
| golang.org/x/net | v0.59.0 → v0.60.0 | GO-2026-6603, GO-2026-6610, GO-2026-6611, GO-2026-6612 |

Fixed dependency metadata and runtime behavior were verified below. A new Pulse
scan of all rebuilt final images has **not** been performed; these are not
post-scan closure counts.

## Verification

Successful:

```bash
uv run --no-project python /home/mkornfield/.codex/skills/address-cves/scripts/summarize_findings.py \
  --limit 0 /tmp/pulse-zips-38554823128/*.zip

uv run --no-project --with ruff ruff check \
  docker/scripts/{build,install}-wandb-core.py \
  tests/unit/test_{wandb_core_patch,docker_workspace_slices}.py
uv run --no-project --with ruff ruff format --check \
  docker/scripts/{build,install}-wandb-core.py \
  tests/unit/test_{wandb_core_patch,docker_workspace_slices}.py

uv run --no-project --with pytest --with pytest-asyncio --with pytest-env --with pytest-timeout \
  pytest --confcutdir=tests/unit -o addopts='' \
  tests/unit/test_wandb_core_patch.py tests/unit/test_docker_workspace_slices.py -q
# 40 passed. Standalone harness avoids unrelated root conftest/platform dependencies.

# Also passed after bootstrapping the isolated workspace environment (40 tests):
flox -q activate -- uv run --frozen pytest \
  tests/unit/test_wandb_core_patch.py tests/unit/test_docker_workspace_slices.py -v --no-cov
flox -q activate -- uv run pre-commit run -a

# All seven CI Bake graph groups also passed make docker-print TARGET=<group>:
# docker-cpu, docker-gpu, docker-garak, docker-python-base,
# nhx-automodel-gpu-wheels, nhx-automodel, nhx-unsloth.
docker buildx bake --print '*'
docker buildx bake wandb-core-artifacts \
  --set '*.platform=linux/amd64,linux/arm64' \
  --set 'wandb-core-artifacts.output=type=local,dest=.wandb-validation/multiarch'
docker buildx bake wandb-core-smoke-test --set '*.platform=linux/amd64'
git diff --check
```

- Both exported ELF binaries independently inspected with host `go version -m`:
  **go1.27.2**, **golang.org/x/net v0.60.0**, correct GOARCH, **CGO_ENABLED=0**.
- amd64 SHA-256:
  `b048c29a712c02a4123b0afe65b5fbcb06169d29756bd31b7cee79f5793a9e4a`.
- arm64 SHA-256:
  `bf8631b59c68b164e4d5a44bbb86e7fa998939026ddf0dbfd1ad9130e6fec4b8`.
- Dedicated Bake smoke test exercises the real named artifact context and shared
  installer, then performs W&B init/log/finish under `RUN --network=none`,
  checking that an offline `.wandb` run file was produced.
- Additional disposable `docker run --rm --network none --user 0` on existing
  `nvcr.io/nvidia/nemo-helix/nhx-gym-host:0.7.0`: shared installer and verify-only
  pass across **five copies** (base venv, three prefetched server venvs, retained
  uv archive); offline init/log/finish also passes. This checks patching a real
  image, not a full source rebuild of that image.

## Remaining work / limitations

- **Outside this change set:** 39 Critical and 143 High report rows concern other
  components. Python/jobs-launcher are handled separately; OS/base-image,
  unrelated Python/Java/native packages, and any no-fixed-version findings are
  not addressed here. No blanket scan-clean claim is made.
- **Release validation outstanding:** full downstream CPU/GPU image source builds
  and Pulse rescans, including caches. GPU training/inference was not exercised.
- **Arm64 runtime outstanding:** cross-compilation and ELF/Go metadata passed;
  this host has no arm64 runtime/emulator, so offline arm64 execution was not run.
- GHCR image metadata requests returned `denied: denied`. The dedicated smoke
  image now uses the same official uv release from Docker Hub and passed; other
  existing Dockerfiles still require their normal registry access.
- An initial standalone test invocation failed on root conftest's missing `nhx`
  dependencies; the standalone targeted suite above passed. After bootstrapping
  this isolated worktree, `flox -q activate -- uv run pre-commit run -a` passed,
  including repository-wide ty checks. Initial bootstrap hit a transient Git
  connection reset; retry succeeded. The UI lint-staged hook initially lacked
  web dependencies; frozen pnpm installation resolved that prerequisite.
  License generation was not run.
- Retained extracted caches in the merged image are patched. Original binaries
  can remain in ancestor image layers and build-only download caches; this work
  does not claim to erase history. Future runtime dependency installs can fetch
  the original wheel and must reapply the helper. Unexpected SDK versions fail
  closed rather than being silently paired with this core.
