<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Container CVE investigation: orphaned RL W&B cache binary

## Scan artifacts reviewed

- Platform-Deploy run: https://github.com/NVIDIA-NeMo/Platform-Deploy/actions/runs/43009282129
- All 14 `pulse-container-scan-*-43009282129.zip` artifacts were normalized with the `address-cves` summarizer.
- Focused review: `pulse-container-scan-nhx-rl-training-43009282129.zip`, `reports/vulns.json` and `reports/sbom.json`.
- Scanned image: `nvcr.io/0921617854601259/nemo-helix-dev/nhx-rl-training:5649d84559c50edfb8daf6dac3533fa76ca87631`.
- Reported image digest: `sha256:188c76c68a25f4113a3d2ec783fd99e4706705b762a856cec40f90fb1e573f79`.

## Findings before remediation

The summarizer reports finding rows across artifacts, not globally unique advisories:

| Category | Critical | High |
| --- | ---: | ---: |
| Project dependencies | 0 | 0 |
| Container-only, all downloaded artifacts | 85 | 709 |
| Focused RL Go findings (subset of container-only) | 3 | 13 |

All 16 focused findings refer to the same file:

`/opt/uv_cache/files-v0/95/9577435c3fe4be4b0470ca503c2e161e50593ec730742114a88774549bd43707`

The SBOM identifies its main Go module as `github.com/wandb/wandb/core`, compiled with `go1.27.1`. The findings report `golang.org/x/net v0.59.0`. These are not the rebuilt W&B executables in the runtime environments or `archive-v0`.

## Changes made

- Extend `docker/scripts/update-python-trust.py` to remove unreferenced regular files from uv's `files-v0` content-addressed store after binary replacement and obsolete certificate archive cleanup.
- Match uv 0.13.4's `Cache::prune_archive_files` hardlink-count rule: remove only regular files whose link count is one. Preserve files still hardlinked into archives or environments and do not follow symlinks.
- Avoid full `uv cache prune`, which could affect the archives shipped for prefetched RL environments.
- Add regression tests reproducing the original hardlinked W&B binary, its patched archive replacement, orphan cleanup, preservation of other hardlink consumers, symlink safety, and absent-cache handling.
- No additional version overrides, scanner exclusions, or suppressions added. The previous W&B upgrade and patched Go rebuild remain in place.

## Findings addressed by the change, pending rebuilt-image verification

- Critical: GO-2026-6604 (stdlib), GO-2026-6612 (stdlib and x/net).
- High: GO-2026-6603, GO-2026-6610, GO-2026-6611, GO-2026-6617 (stdlib and x/net), plus GO-2026-6605, GO-2026-6607, GO-2026-6608, GO-2026-6609, GO-2026-6613 (stdlib).

The fix removes the obsolete cache executable, rather than changing the scanner's view of a retained vulnerable binary. If another consumer still hardlinks the file, cleanup preserves it; such a consumer must be upgraded rather than deleted silently.

## Findings not addressed

- The other container findings in this newer scan (including Python/OS packages, BusyBox, and additional Go dependencies) are outside this focused follow-up and remain unverified/unremediated here.
- No conclusion about the earlier Kubernetes inference deployment crashes follows from this cache finding.
- The artifact reviewed identifies amd64. A rebuilt-image scan is required for both amd64 and arm64; no clean rescan is claimed.

## Verification

- Ran the skill summarizer against the provided Actions URL and inspected the raw vulnerability report and streamed SBOM: confirmed all 16 focused findings share the cache path and W&B module identity.
- Compared cleanup semantics against uv 0.13.4 `crates/uv-cache/src/lib.rs`, `Cache::prune_archive_files`.
- `UV_PROJECT_ENVIRONMENT=/home/mkornfield/nemo-platform/.venv uv run --no-sync pytest docker/scripts/tests/test_update_python_trust.py -q`: 19 passed (4 unrelated pytest temporary-directory cleanup warnings). Existing test fakes were isolated from shared standard-library modules to satisfy the current test patch guard.
- `uv run --no-sync ruff check docker/scripts/update-python-trust.py docker/scripts/tests/test_update_python_trust.py` and `uv run --no-sync ruff format --check` for the same files, using the existing development environment: passed.
- `git diff --check`: passed.
- Full pre-commit attempted with `UV_PROJECT_ENVIRONMENT=/home/mkornfield/nemo-platform/.venv uv run --no-sync pre-commit run -a`: not green. Config-reference, root-lock, and copyright hooks could not fetch OpenShell metadata (certificate error); UI lint-staged is not installed. Ruff, formatting, ty, Helm docs, lock-drift, toolchain checks, and the remaining hooks passed.
- Container rebuild and Pulse rescan remain required to verify removal from the shipped image.
