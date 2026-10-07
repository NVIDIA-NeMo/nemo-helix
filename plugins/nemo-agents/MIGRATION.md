<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-agents-plugin migration notes

## AgentsResource client constructor

`AgentsResource` and `AsyncAgentsResource` now take the typed platform clients
from `nemo_helix_plugin.client.client`. Do not pass the legacy generated
`NeMoHelix` or `AsyncNeMoHelix` SDK objects directly.

Before:

```python
from nemo_helix import NeMoHelix
from nemo_agents_plugin.sdk import AgentsResource

sdk = NeMoHelix(base_url="http://localhost:8080", workspace="default")
agents = AgentsResource(sdk)
```

After:

```python
from nemo_agents_plugin.sdk import AgentsResource
from nemo_helix_plugin.client.client import NemoClient

client = NemoClient(base_url="http://localhost:8080", workspace="default")
agents = AgentsResource(client)
```

Async jobs use `AsyncNemoClient` with `AsyncAgentsResource`:

```python
from nemo_agents_plugin.sdk import AsyncAgentsResource
from nemo_helix_plugin.client.client import AsyncNemoClient

async with AsyncNemoClient(base_url="http://localhost:8080", workspace="default") as client:
    job = await AsyncAgentsResource(client).jobs.execute.create(
        spec={"agent": "agent-name", "input": "hello"},
    )
```
