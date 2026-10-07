<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Garak Plugin Reference

The garak plugin is a first-party scaffold for garak functionality. It keeps the plugin identity separate from the legacy garak service while providing the basic surfaces needed for SDK-backed jobs.

## Registered Surfaces

| Surface | Entry point | Current behavior |
|---|---|---|
| CLI | `nemo.cli:garak` | Adds `nemo garak info` and hosts garak job commands. |
| Service | `nemo.services:garak` | Mounts health status at `/apis/garak/v1/healthz`. |
| SDK | `nemo.sdk:garak` | Adds `client.garak.plugin_status()`. |
| Job | `nemo.jobs:garak.audit` | Runs a garak scan against a configured target. |
| Docs | `nemo.docs:garak` | Publishes this reference page. |
| Skills | `nemo.skills:garak` | Publishes the garak plugin development skill. |

## Current Job

`garak.audit` is a `NemoJob` stub that accepts a target identifier and an optional list of probe names. The current implementation returns an empty findings list; integration with the garak SDK is intentionally left for the next design pass.

## CLI Examples

Check that the plugin is installed and reports the registered job key:

```bash
nemo garak info
```

Inspect the generated job metadata:

```bash
nemo garak audit explain
```

Submit an audit job:

```bash
nemo garak audit --spec '{"config": "default/<config-name>", "target": "default/<target-name>"}'
```

## Python Examples

Read the plugin service status through `GarakPluginResource`:

```python
from nemo_garak.sdk import GarakPluginResource
from nemo_helix_plugin.client.client import NemoClient

garak = GarakPluginResource(NemoClient(base_url="http://localhost:8000"))
status = garak.plugin_status()
```
