<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-sandboxed-gym

Trusted **episode broker** + **job-level Gym host** orchestrator for running NeMo-Gym
inside an OpenSandbox (or compatible) isolation boundary.

This package has **no NeMo-RL / GRPO dependency**. Callers get raw Gym host
`/health` and `/rollouts/run` results. RL-specific postprocessing belongs in the
training client (e.g. NeMo-RL's thin `SandboxedGymActor` adapter).

## Provenance

Vendored from [`soluwalana/RL`](https://github.com/soluwalana/RL) branch `nmp/customizer` at
**`67821a9`** (2026-08-03), the fork this code still shares an ancestor with. Upstream changes have
been applied selectively since; the divergences are deliberate and documented where they occur.

Recorded because the original vendoring landed as one squashed commit with no provenance, and
recovering this took a per-file comparison against every revision of the upstream tree. Diffing
from any earlier commit shows `host/entrypoint.py` and `host/gym_host.sh` as drift when they are
already here. Update this line whenever upstream is merged again; it becomes moot once the shared
wheel exists.

## Architecture

![Architecture](docs/architecture.svg)

The trusted side holds the OpenSandbox credential and the episode broker; the sandbox gets a
per-run token and nothing else. The episode tier is only used by environments that ask for nested
sandboxes -- an ordinary evaluation never creates one.

### Two job-host providers

![Host provider paths](docs/host-provider-paths.svg)

`get_host_provider` selects by name. **`docker` is for local execution only**: it runs the host as
a container, records the egress allowlist rather than applying it, and bind-mounts host
directories. It is not an isolation boundary. **`opensandbox`** is the real one -- Kubernetes pod,
PVC mounts, enforced egress, kata-qemu -- and is where the isolation properties actually live.

## Install

```bash
pip install -e packages/sandboxed_gym
# optional Ray wrappers
pip install -e "packages/sandboxed_gym[ray]"
```

## Serve (always starts the episode broker)

```bash
# Mode A — local orchestrator proxy (recommended for cross-job clients)
sandboxed-gym serve --config serve.yaml --mode orchestrator --bind 0.0.0.0:8090 \
  --session-file /tmp/session.json

# Mode B — print Gym host URLs and wait
sandboxed-gym serve --config serve.yaml --mode host-urls --session-file /tmp/session.json
```

### Cross-job pattern

1. **Job A** runs `sandboxed-gym serve --mode orchestrator` with environment/dataset
   PVC mounts (same `SandboxConfig` shape as sandboxed GRPO).
2. Job A writes a `SandboxedGymSessionDescriptor` (`--session-file`).
3. **Job B** POSTs to Job A's orchestrator Service: `POST /rollouts/run`.

A wildcard `--bind` advertises the node address, which is routable from another pod but not from
outside the cluster. Pass `--advertise-url` when the reachable address is not the one Job A binds
-- a published container port, or an ingress in front of the proxy.

Set `episode_broker.advertise_url` to a stable Service DNS name so the Gym host
egress allowlist does not depend on a pod IP.

## Library

```python
from sandboxed_gym import SandboxedGymOrchestrator, SandboxedGymServeConfig

session = SandboxedGymOrchestrator().start(SandboxedGymServeConfig.model_validate(...))
results = session.run_rollouts([{"agent_ref": {...}, ...}])
session.shutdown()
```

## Config sketch

See `examples/serve.yaml`. Key fields:

- `sandbox.environment_pvc_claim` / `dataset_pvc_claim` / `workspace_pvc_claim`
- `episode_broker` (allowlists, TTL, `advertise_url`)
- `gym_global_config` — opaque Gym JSON injected as `NMP_GYM_GLOBAL_CONFIG`
- `rollout_auth_token` — optional bearer for the orchestrator proxy
