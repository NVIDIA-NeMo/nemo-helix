<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fabric Config Boundary

This package contains the NeMo Helix-side config and translation helpers for
Fabric-backed NeMo Agents.

NeMo Helix owns the persisted agent contract. A Fabric-backed agent is stored
using the NeMo Helix-managed `nemo-agents-spec-v1` config shape, authored as
`agent.yaml` in the Ethos fileset and represented in code as `AgentConfig`.

Fabric is an execution dependency, not the persisted NeMo Helix contract. Before
calling Fabric SDK APIs, NeMo Agents translates the NeMo Helix-managed config into a
typed in-memory `FabricConfig`.

```text
NeMo Helix agent.yaml -> AgentConfig -> FabricConfig
```

`agent.yaml` is not treated as a Fabric SDK file-backed config or profile. The
NeMo Helix config may keep product concepts, defaults, and artifact references in
the shape NeMo Helix needs, while the Fabric translator owns the mapping into
Fabric's runtime fields such as harness adapter, model, environment, and
telemetry config.
