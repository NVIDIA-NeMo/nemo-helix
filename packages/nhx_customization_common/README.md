<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nhx-customization-common

Shared library for the NeMo Helix customization training backends (`unsloth`, `automodel`).

It hosts code that both backends previously duplicated, under the `nhx.customization_common`
namespace:

- **Plugin side** (imported by the flat `nemo_<svc>_plugin` modules): a parametrized jobs-client SDK
  factory, a `BaseContributor`, the CLI `submit`/`run` override machinery, a base plugin config, and a
  `BaseSubmitJob`.
- **Service side** (imported by `nhx.<svc>`): the job context, platform client helpers, progress
  reporters, file_io / model_entity task runners, training callbacks, shared step schemas, value enums,
  container-path constants, image resolution, and the compiler scaffold.

Each backend keeps thin shim modules at the paths its entry points and the customizer SDK hub import by
string (`nemo_<svc>_plugin.contributor:<Svc>Contributor`, `nemo_<svc>_plugin.jobs.jobs:<Svc>Job`,
`nemo_<svc>_plugin.sdk.resources:{<Svc>Customization, Async<Svc>Customization}`) — those symbols must
not move.

This package ships into the shared `nhx.*` namespace (no `src/nhx/__init__.py`) and carries no entry
points; discovery stays on the concrete plugins.
