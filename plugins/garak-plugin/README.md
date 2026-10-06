<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Garak Plugin

A NeMo Helix plugin which provides Garak Plugin, an LLM
vulnerability scanner service powered by [Garak](https://github.com/NVIDIA/garak)

## CLI quickstart

### Prerequisites

Before running the `nemo garak-plugin configs create` and `nemo garak-plugin targets
create` commands below, make sure you have:

- **NeMo CLI installed and platform running** — follow
  [SETUP.md](../../SETUP.md) (`make bootstrap` + `nemo setup`).
- **CLI pointed at the platform** — `nemo setup` configures
  `http://localhost:8080` automatically; otherwise set it explicitly with
  `nemo config set --base-url <url>`.
- **A workspace to operate in** — the examples use `-w default`, which
  `nemo setup` creates. Substitute another workspace name as needed.

For detailed guides and reference material, see the full garak-plugin
documentation at [`docs/garak-plugin/`](../../docs/garak-plugin/index.md).

### Managing configs and targets

Two persistent entity types — `ScanConfig` (probe / detector / reporting
settings) and `ScanTarget` (the model under test) — are managed through the
NeMo CLI:

```bash
# Create a config from a JSON file
nemo garak-plugin configs create quick-scan -w default --data-file ./quick-scan.json

# Create a target inline
nemo garak-plugin targets create nemotron-3.5-lightning-30b -w default -d '{
  "type": "nim.NVOpenAIChat",
  "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
  "options": {"uri": "http://localhost:9000/v1"}
}'

# List, get, update, delete are all available
nemo garak-plugin configs list -w default
nemo garak-plugin targets get nemotron-3.5-lightning-30b -w default
nemo garak-plugin configs delete quick-scan -w default
```

There is no CLI command for running a scan yet — the local-run path is
exposed through the SDK (below). The platform jobs service can submit
scans via the `garak-plugin.scan` job entry point.

## SDK quickstart

Every CLI verb has a matching Python SDK method on `client.garak_plugin`, plus
`client.garak_plugin.run(...)` for in-process execution that bypasses the jobs
service.

```python
from nemo_helix import NeMoHelix
from garak_plugin.entities import (
    ScanSystemData, ScanRunData, ScanPluginsData, ScanReportData,
)

client = NeMoHelix()

# Persist a config
cfg = client.garak_plugin.configs.create(
    workspace="default",
    name="quick-scan",
    system=ScanSystemData(lite=True, parallel_attempts=4),
    run=ScanRunData(generations=1),
    plugins=ScanPluginsData(probe_spec="goodside.Tag", detector_spec="auto"),
    reporting=ScanReportData(report_prefix="quick-scan"),
)

# Persist a target
tgt = client.garak_plugin.targets.create(
    workspace="default",
    name="nemotron-3.5-lightning-30b",
    type="nim.NVOpenAIChat",
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    options={"uri": "http://localhost:9000/v1"},
)

# Submit a K8s scan job and wait for it to finish.
job = client.garak_plugin.submit(
    config="quick-scan",
    target="nemotron-3.5-lightning-30b",
    workspace="default",
)
print(f"Job submitted: {job.name}")
job.wait_until_done()                          # blocks; streams logs while polling
artifacts_dir = job.download_artifacts()       # extracts garak reports to ./<job-name>/
print(f"Reports: {artifacts_dir}")

# Or run a scan locally (no jobs-service submission).
result = client.garak_plugin.run(
    config="quick-scan",       # workspace-qualified name strings ("ws/name") also work
    target="nemotron-3.5-lightning-30b",
    workspace="default",
)
print(f"Scan status: {result['status']}")
if result["status"] == "failed":
    print(f"Scan failed: {result.get('error', 'unknown error')}")
else:
    print(result["probes_complete"], result["probes_failed"])
    for name, ref in result["results"].items():
        print(name, ref["artifact_url"])
```

`submit()` posts the job to the K8s executor and returns an `GarakPluginJobResource` handle.
`run()` shells out to a pre-installed garak interpreter (default
`~/.garak-plugin/.venv/bin/python`, override via `$NEMO_GARAK_PLUGIN_GARAK_PYTHON`)
and registers the resulting JSONL / HTML / hitlog reports as job results
under a temp directory managed by the local scheduler.
