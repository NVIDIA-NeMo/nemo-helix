<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Auditor Plugin Reference

The auditor plugin is a first-party scaffold for auditor functionality. It keeps the plugin identity separate from the legacy auditor service while providing the basic surfaces needed for SDK-backed jobs.

## Registered Surfaces

| Surface | Entry point | Current behavior |
|---|---|---|
| CLI | `nemo.cli:auditor` | Adds `nemo auditor info` and hosts auditor job commands. |
| Service | `nemo.services:auditor` | Mounts health status at `/apis/auditor/v1/healthz`. |
| SDK | `nemo.sdk:auditor` | Adds `client.auditor.plugin_status()`. |
| Job | `nemo.jobs:auditor.audit` | Runs an auditor scan against a configured target. |
| Docs | `nemo.docs:auditor` | Publishes this reference page. |
| Skills | `nemo.skills:auditor` | Publishes the auditor plugin development skill. |

## Current Job

`auditor.audit` is a `NemoJob` stub that accepts a target identifier and an optional list of probe names. The current implementation returns an empty findings list; integration with the auditor SDK is intentionally left for the next design pass.

## CLI Examples

Check that the plugin is installed and reports the registered job key:

```bash
nemo auditor info
```

Inspect the generated job metadata:

```bash
nemo auditor audit explain
```

Submit an audit job:

```bash
nemo auditor audit --spec '{"config": "default/<config-name>", "target": "default/<target-name>"}'
```

## Python Examples

Read the plugin service status through `AuditorPluginResource`:

```python
from nemo_auditor.sdk import AuditorPluginResource
from nemo_helix_plugin.client.client import NemoClient

auditor = AuditorPluginResource(NemoClient(base_url="http://localhost:8000"))
status = auditor.plugin_status()
```
