<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vendored Switchyard (lib + telemetry)

Snapshot of [NVIDIA-dev/switchyard](https://github.com/NVIDIA-dev/switchyard) at commit
[`94079222829d67fa278ac5b50799c8162e4c0409`](https://github.com/NVIDIA-dev/switchyard/commit/94079222829d67fa278ac5b50799c8162e4c0409)
("ci: enforce conventional commits", 2026-05-11).

Only `switchyard.lib` and `switchyard.telemetry` are vendored. The CLI, server,
and experimental subpackages from upstream are intentionally omitted — Platform
only depends on `switchyard.lib.*` and `switchyard.telemetry`.

This vendor directory is installed in editable mode by `nemo-switchyard-plugin`
via a local-path `tool.uv.sources` entry. The plan is to replace this
snapshot with a git submodule pinned to the same commit once the upstream repo
is reachable from CI without per-developer SSH credentials.

To refresh the snapshot to a newer upstream commit, re-run the vendor extraction
against the new commit hash, update the version suffix in `pyproject.toml`, and
update this README.

Upstream license: Apache-2.0 (see `LICENSE` and `NOTICE`).
