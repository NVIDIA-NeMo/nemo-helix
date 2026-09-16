<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Manual Intake checks

These scripts are not tests. No automation runs them. Pytest does not collect them,
because their names do not start with `test_`.

Run them by hand against a local NeMo Helix that has Intake running and holds real
spans. The unit tests in `tests/test_traces.py` fake every Intake call, so they prove
the logic but never prove that Intake answers the way the logic expects. These scripts
close that gap.

Both scripts are read-only. Every call they make is a GET.

## Before you run

Start the platform and confirm that Intake is ready:

```bash
curl -sf http://localhost:8080/health/ready
curl -fsS http://localhost:8080/status | uv run --frozen python -c "import json,sys; print('intake' in json.load(sys.stdin)['services']['ready'])"
```

Ingest some traces if the platform is empty. A script that finds no spans reports that
and stops.

## The scripts

### `intake_tool_checks.py`

Exercises every tool in `nemo_experimentalist_plugin.eval_author.traces` against live data. It finds a
workspace that holds agent-scoped spans, then runs the checks against it, so it needs
no configuration. It prints one line per check and exits nonzero if any check fails.

```bash
uv run --frozen python plugins/nemo-eval-author/tests/manual/intake_tool_checks.py
```

### `intake_vocabulary_probe.py`

Asks Intake which filter fields and operators it really serves, and prints a report.
Run it after any change to the Intake span or trace filter schemas.

The docstrings of `query_spans` and `query_traces` are the only guide the agent has when
it builds a query, so a wrong docstring sends the agent down a dead end. This probe is how
those docstrings were established, and it is also what found the five span filters that
Intake published but could not serve, fixed in nemo-helix#1225. If the report and the
docstrings disagree, the docstrings are wrong.

```bash
uv run --frozen python plugins/nemo-eval-author/tests/manual/intake_vocabulary_probe.py
```

## Options

Both scripts accept `--base-url` (default `http://localhost:8080`).
`intake_tool_checks.py` also accepts `--workspace` and `--agent` to skip discovery.
