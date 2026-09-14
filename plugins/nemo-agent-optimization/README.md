<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-agent-optimization

Entry point for `nemo agents optimize`. Discovers installed
`nemo.optimization-strategy` plugins, validates the chosen strategy's
config, and dispatches to it. See `nemo_agent_optimization_plugin.strategies`
for the plugin contract.

## Discover the installed strategies

`nemo agents optimize --strategy` takes any name this command prints, one per
line:

```bash
$ nemo agents optimization-strategies list
experimentalist
nat
prompt-master
switchyard
```

The list is whatever is installed in the current environment: each name comes
from a `nemo.optimization-strategy` entry point, so installing a plugin that
declares one adds it here without any change to this package.
