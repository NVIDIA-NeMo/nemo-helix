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

Deploy is independent of build: it runs `helm upgrade` on the Helix release with the uploaded checkout's `k8s/helm`
chart, changing only the api and core image values. Set `deploy.enabled=true` in the chart to grant the pod `admin` in
the release and builds namespaces.

```bash
nemo helix-operator deploy --api-image <repo>:<tag> [--core-image <repo>:<tag>] [--rollback-on-failure] --wait
nemo helix-operator deployment <revision>             # status comes from `helm history`
```

Test runs execute the uploaded checkout's `e2e/test_hello_world.py` inside the operator pod, against the Helix API
Service (`api_url`), with the checkout's own packages first on `PYTHONPATH`. One run at a time, kept in memory.

```bash
nemo helix-operator test --wait                       # {"id": ..., "status": "passed", "passed": 1, "helm_revision": ...}
nemo helix-operator test-run <id>
```

The service runs in its own pod (`Dockerfile`, `chart/`), with `NHX_BASE_URL` pointing at the Helix API. It is not in
`enabled-plugins`, so no platform image loads it. Configuration is `HelixOperatorConfig` (`NEMO_HELIX_OPERATOR_*`).
