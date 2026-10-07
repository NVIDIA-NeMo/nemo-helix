<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Garak Plugin

A NeMo Helix plugin which provides Garak, an LLM
vulnerability scanner service powered by [Garak](https://github.com/NVIDIA/garak)

## CLI quickstart

### Prerequisites

Before running the `nemo garak configs create` and `nemo garak targets
create` commands below, make sure you have:

- **NeMo CLI installed and platform running** — follow
  [SETUP.md](../../SETUP.md) (`make bootstrap` + `nemo setup`).
- **CLI pointed at the platform** — `nemo setup` configures
  `http://localhost:8080` automatically; otherwise set it explicitly with
  `nemo config set --base-url <url>`.
- **A workspace to operate in** — the examples use `-w default`, which
  `nemo setup` creates. Substitute another workspace name as needed.

For detailed guides and reference material, see the full garak
documentation at [`docs/garak/`](../../docs/garak/index.md).

### Managing configs and targets

Two persistent entity types — `AuditConfig` (probe / detector / reporting
settings) and `AuditTarget` (the model under test) — are managed through the
NeMo CLI:

```bash
# Create a config from a JSON file
nemo garak configs create quick-scan -w default --data-file ./quick-scan.json

# Create a target inline
nemo garak targets create nemotron-3.5-lightning-30b -w default -d '{
  "type": "nim.NVOpenAIChat",
  "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
  "options": {"uri": "http://localhost:9000/v1"}
}'

# List, get, update, delete are all available
nemo garak configs list -w default
nemo garak targets get nemotron-3.5-lightning-30b -w default
nemo garak configs delete quick-scan -w default
```

There is no CLI command for running an audit yet — the local-run path is
exposed through the SDK (below). The platform jobs service can submit
audits via the `garak.audit` job entry point.

## SDK quickstart

Every CLI verb has a matching method on `GarakPluginResource`, plus
`run(...)` for in-process execution that bypasses the jobs service. Wrap a
`NemoClient` to use it; the typed `GarakClient` exposes the raw endpoints.

```python
from nemo_garak.sdk import GarakPluginResource
from nemo_helix_plugin.client.client import NemoClient
from nemo_garak.entities import (
    AuditSystemData, AuditRunData, AuditPluginsData, AuditReportData,
)

garak = GarakPluginResource(NemoClient(base_url="http://localhost:8080", workspace="default"))

# Persist a config
cfg = garak.configs.create(
    workspace="default",
    name="quick-scan",
    system=AuditSystemData(lite=True, parallel_attempts=4),
    run=AuditRunData(generations=1),
    plugins=AuditPluginsData(probe_spec="goodside.Tag", detector_spec="auto"),
    reporting=AuditReportData(report_prefix="quick-scan"),
)

# Persist a target
tgt = garak.targets.create(
    workspace="default",
    name="nemotron-3.5-lightning-30b",
    type="nim.NVOpenAIChat",
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    options={"uri": "http://localhost:9000/v1"},
)

# Submit a K8s audit job and wait for it to finish.
job = garak.submit(
    config="quick-scan",
    target="nemotron-3.5-lightning-30b",
    workspace="default",
)
print(f"Job submitted: {job.name}")
job.wait_until_done()                          # blocks; streams logs while polling
artifacts_dir = job.download_artifacts()       # extracts garak reports to ./<job-name>/
print(f"Reports: {artifacts_dir}")

# Or run an audit locally (no jobs-service submission).
result = garak.run(
    config="quick-scan",       # workspace-qualified name strings ("ws/name") also work
    target="nemotron-3.5-lightning-30b",
    workspace="default",
)
print(f"Audit status: {result['status']}")
if result["status"] == "failed":
    print(f"Audit failed: {result.get('error', 'unknown error')}")
else:
    print(result["probes_complete"], result["probes_failed"])
    for name, ref in result["results"].items():
        print(name, ref["artifact_url"])
```

`submit()` posts the job to the K8s executor and returns a `GarakJobResource` handle.
`run()` shells out to a pre-installed garak interpreter (default
`~/.garak/.venv/bin/python`, override via `$NEMO_GARAK_PYTHON`)
and registers the resulting JSONL / HTML / hitlog reports as job results
under a temp directory managed by the local scheduler.
