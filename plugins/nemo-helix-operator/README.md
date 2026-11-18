<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-helix-operator-plugin

Builds an nhx-api image from a local NeMo Helix checkout, using the builder. Only Python source changes take effect:
the image overlays every `src/` tree on a published nhx-api base.

```bash
uv sync --group helix-operator
export NEMO_HELIX_OPERATOR_OPERATOR_URL=http://<helix-operator>:8080
nemo helix-operator upload .                          # Files: helix-src/source.tar.gz + source.json
nemo helix-operator build --fileset helix-src --wait  # {"name": "helix-<n>.<commit>", "status": "ready", "api_image": ...}
nemo helix-operator status <name>
```

The service runs in its own pod (`Dockerfile`, `chart/`), with `NHX_BASE_URL` pointing at the Helix API. It is not in
`enabled-plugins`, so no platform image loads it. Configuration is `HelixOperatorConfig` (`NEMO_HELIX_OPERATOR_*`).
