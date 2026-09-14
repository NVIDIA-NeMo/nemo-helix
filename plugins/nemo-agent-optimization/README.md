<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-agent-optimization

Entry point for `nemo agents optimize`. Discovers installed
`nemo.optimization-strategy` plugins, validates the chosen strategy's
config, and dispatches to it. See `nemo_agent_optimization_plugin.strategies`
for the plugin contract.
