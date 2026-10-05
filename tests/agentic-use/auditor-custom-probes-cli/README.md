<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Custom Audit with Selected Probes (CLI)

Harbor eval

Tests the agent's ability to create a custom audit configuration with specific
selected probes, create an audit target, and run an audit with the custom config.

## Known Limitations

**Audits may not run to completion in the Harbor test environment.**

The audit job execution chain requires:
1. A pre-built `nhx-auditor-tasks` Docker image (contains Garak framework)
2. Docker-in-Docker (DOOD) access via `/var/run/docker.sock`
3. A working inference endpoint for Garak probes to call

The Harbor container runs the NeMo Helix API in quickstart mode, but the
local Garak runtime may not be available.

As a result, the verifier validates:
- **Setup correctness**: config has the correct probes, target references the expected
  model, and the audit command references both config + target
- **Agent behavior**: trajectory analysis confirms the agent invoked the audit
  command and reviewed its output

It does **not** verify:
- That only the selected probes actually ran
- That results contain detailed findings
