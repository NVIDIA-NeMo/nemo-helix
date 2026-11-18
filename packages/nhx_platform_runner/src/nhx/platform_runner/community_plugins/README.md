<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Manifest-based community plugins

A community plugin is a **YAML manifest**, not an installed Python distribution.
The registry declares gateway endpoint mappings, optional workspace credential
bindings, and optional pinned Studio UI assets. JSON remains accepted as a wire
format. Both formats validate to the same snapshot and consent digest.

The API service runs elsewhere. Its deployment can be composed independently
using any language/runtime/host; registration neither deploys it nor invokes its
Python entry points. The earlier wheel/virtualenv/subprocess prototype has been
removed from this implementation. Deployment, containers, and sandbox runtimes
remain out of scope.

- [GATEWAY.md](GATEWAY.md): API contract, trust model, and runnable demo flow.
- [`../../../../examples/community_plugins/tavily.yaml`](../../../../examples/community_plugins/tavily.yaml):
  deployment-independent example targeting Tavily's hosted MCP API.
- `gateway.py`: declared external API proxy and workspace-secret binding.
- `installation.py`: URL preview, explicit consent, registration, and lifecycle.
- `demo_provider.py`: separately operator-started local stub; never launched by
  Helix. It serves example YAML manifests and a fake Tavily-compatible API.

Shared manifest validation, atomic state, and pinned asset helpers live in
`nemo_helix_plugin.community_plugins` so Studio and the gateway use the same
contract without depending on each other's service packages.

Legacy Python plugin/Studio discovery remains a compatibility surface, separate
from this community registry. A legacy API-only Tavily package can coexist with
the new `/apis/plugins/tavily/...` mapping; it is not needed to register the YAML.
