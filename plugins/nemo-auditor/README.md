<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Auditor Plugin

A NeMo Helix plugin which provides Auditor, an LLM
vulnerability scanner service powered by [Garak](https://https://github.com/NVIDIA/garak)

## CLI quickstart

### Prerequisites

Before running the `nemo auditor configs create` and `nemo auditor targets
create` commands below, make sure you have:

- **NeMo CLI installed and platform running** — follow
  [SETUP.md](../../SETUP.md) (`make bootstrap` + `nemo setup`).
- **CLI pointed at the platform** — `nemo setup` configures
  `http://localhost:8080` automatically; otherwise set it explicitly with
  `nemo config set --base-url <url>`.
- **A workspace to operate in** — the examples use `-w default`, which
  `nemo setup` creates. Substitute another workspace name as needed.

For detailed guides and reference material, see the full auditor
documentation at [`docs/auditor/`](../../docs/auditor/index.md).

### Managing configs and targets

Two persistent entity types — `AuditConfig` (probe / detector / reporting
settings) and `AuditTarget` (the model under test) — are managed through the
NeMo CLI:

```bash
# Create a config from a JSON file
nemo auditor configs create quick-scan -w default -f ./quick-scan.json

# Create a target inline
nemo auditor targets create nemotron-3.5-lightning-30b -w default -d '{
  "type": "nim.NVOpenAIChat",
  "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
  "options": {"uri": "http://localhost:9000/v1"}
}'

# List, get, update, delete are all available
nemo auditor configs list -w default
nemo auditor targets get nemotron-3.5-lightning-30b -w default
nemo auditor configs delete quick-scan -w default
```

There is no CLI command for running an audit yet — the local-run path is
exposed through the SDK (below). The platform jobs service can submit
audits via the `auditor.audit` job entry point.

## SDK quickstart

Every CLI verb has a matching Python SDK method on `client.auditor`, plus
`client.auditor.run(...)` for in-process execution that bypasses the jobs
service.

```python
from nemo_helix import NeMoHelix
from nemo_auditor.entities import (
    AuditSystemData, AuditRunData, AuditPluginsData, AuditReportData,
)

client = NeMoHelix()

# Persist a config
cfg = client.auditor.configs.create(
    workspace="default",
    name="quick-scan",
    system=AuditSystemData(lite=True, parallel_attempts=4),
    run=AuditRunData(generations=1),
    plugins=AuditPluginsData(probe_spec="goodside.Tag", detector_spec="auto"),
    reporting=AuditReportData(report_prefix="quick-scan"),
)

# Persist a target
tgt = client.auditor.targets.create(
    workspace="default",
    name="nemotron-3.5-lightning-30b",
    type="nim.NVOpenAIChat",
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    options={"uri": "http://localhost:9000/v1"},
)

# Submit a K8s audit job and wait for it to finish.
job = client.auditor.submit(
    config="quick-scan",
    target="nemotron-3.5-lightning-30b",
    workspace="default",
)
print(f"Job submitted: {job.name}")
job.wait_until_done()                          # blocks; streams logs while polling
artifacts_dir = job.download_artifacts()       # extracts garak reports to ./<job-name>/
print(f"Reports: {artifacts_dir}")

# Or run an audit locally (no jobs-service submission).
result = client.auditor.run(
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

`submit()` posts the job to the K8s executor and returns an `AuditorJobResource` handle.
`run()` shells out to a pre-installed garak interpreter (default
`~/.auditor/.venv/bin/python`, override via `$NEMO_AUDITOR_GARAK_PYTHON`)
and registers the resulting JSONL / HTML / hitlog reports as job results
under a temp directory managed by the local scheduler.
