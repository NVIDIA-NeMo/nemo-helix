<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Garak Plugin Reference

The garak-plugin plugin is a first-party scaffold for garak-plugin functionality. It keeps the plugin identity separate from the legacy garak-plugin service while providing the basic surfaces needed for SDK-backed jobs.

## Registered Surfaces

| Surface | Entry point | Current behavior |
|---|---|---|
| CLI | `nemo.cli:garak-plugin` | Adds `nemo garak-plugin info` and hosts garak-plugin job commands. |
| Service | `nemo.services:garak-plugin` | Mounts health status at `/apis/garak-plugin/v1/healthz`. |
| SDK | `nemo.sdk:garak-plugin` | Adds `client.garak_plugin.plugin_status()`. |
| Job | `nemo.jobs:garak-plugin.scan` | Runs an garak-plugin scan against a configured target. |
| Docs | `nemo.docs:garak-plugin` | Publishes this reference page. |
| Skills | `nemo.skills:garak-plugin` | Publishes the garak-plugin plugin development skill. |

## Current Job

`garak-plugin.scan` is a `NemoJob` stub that accepts a target identifier and an optional list of probe names. The current implementation returns an empty findings list; integration with the garak-plugin SDK is intentionally left for the next design pass.

## CLI Examples

Check that the plugin is installed and reports the registered job key:

```bash
nemo garak-plugin info
```

Inspect the generated job metadata:

```bash
nemo garak-plugin scan explain
```

Submit a scan job:

```bash
nemo garak-plugin scan --spec '{"config": "default/<config-name>", "target": "default/<target-name>"}'
```

## Python Examples

Read the plugin service status through the platform SDK namespace:

```python
from nemo_helix import NeMoHelix

client = NeMoHelix(base_url="http://localhost:8000")
status = client.garak_plugin.plugin_status()
```
