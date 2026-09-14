<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-optimization-plugin

Shared library for Fabric-backed numeric hyperparameter optimization (Optuna)
and the Agents ``optimize`` job implementation.

## Prerequisites

Install the agents CLI (``uv sync --package nemo-agents-plugin``) and a running
NeMo Platform instance you have access to (workspace + inference credentials).
See [examples/hermes-optimize/README.md](examples/hermes-optimize/README.md#one-time-setup-platform)
for the full one-time setup, including the Hermes harness install.

Primary user surface (Alt 5):

```bash
nemo agents optimize prepare-fileset \
  --source examples/hermes-optimize \
  --optimize-config optimize-chatonly.yaml \
  --fileset hermes-optimize-chatonly \
  --workspace default

nemo agents optimize \
  --strategy nat \
  --optimize-config-fileset default/hermes-optimize-chatonly \
  --optimize-config optimize-chatonly.yaml \
  --workspace default
```

``prepare-fileset`` validates a bundle directory
(:mod:`nemo_optimization.bundle`) and uploads it. ``optimize`` hands the study
to the platform, which cannot read the client's filesystem, so CLI submissions
require ``--optimize-config-fileset``: a fileset holding the whole bundle
(config, Agent under Test, dataset, ``eval.fabric.base_dir`` tree, hooks and MCP
configs) with ``--optimize-config`` given relative to its root.

Golden-path agent shape: Fabric Hermes (``nvidia.fabric.hermes``). See
``examples/hermes-optimize/`` — runnable ``optimize-*.yaml`` packages:

* ``optimize-chatonly.yaml`` — chat-only Hermes smoke
* ``optimize-chatonly-via-agent.yaml`` — same study with a platform ``--agent``
* ``optimize-mcp.yaml`` — phishing analyzer via MCP (separate agent checkout)

Install and QA steps live in that directory's README.

Per-task Fabric lifecycle hooks are author-supplied via string references
(``eval.run_hook.ref``, ``path``+``attr``, or ``nemo.fabric.task_hooks``
entry points). The platform does not vendor example-agent packages such as
email phishing analyzer.

Job registration: ``agents.optimize`` (mounted by the agents plugin, which also
owns the ``prepare-fileset`` CLI command).  ``compile`` selects the ``subprocess``
execution profile when the platform registers one and otherwise the ``cpu``
profile with the ``nmp-cpu-tasks`` image.
Backend registry: ``nemo.optimization.backends`` (``optuna``, ``ga`` stub).

Trials execute the Agent under Test in the study's own process tree; see
[docs/trial-sandboxing.md](docs/trial-sandboxing.md) for the proposed per-trial
container isolation (not implemented).

This package is intentionally not a Customizer contributor. A future
Experimentalist / Customizer agent may call the same library.

## Next Steps

* Try the runnable bundle example: [examples/hermes-optimize](examples/hermes-optimize).
* Stage and launch a study on the platform with ``prepare-fileset`` + ``optimize``;
  see the "Primary user surface" commands above and that example's README for
  the full ``prepare-fileset`` → ``optimize`` workflow.
