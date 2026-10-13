<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Import LangSmith runs

The importer calls `Client.list_runs(project_name=..., start_time=...)`, applies the exclusive upper
bound locally, then calls `Client.list_feedback(run_ids=...)`. It maps run hierarchy directly and
keeps `extra`, events, tags, serialized state, aggregate feedback, and other source-only values under
`langsmith.raw`. Evaluator-origin feedback becomes evaluator results; app/API human feedback becomes
labels, feedback, notes, and correction metadata while the native event remains in
`langsmith.signals`.

```bash
export LANGSMITH_API_KEY=...
uv run --with langsmith python ../scripts/import_langsmith.py \
  --project <project-name> \
  --since 2026-08-01T00:00:00Z \
  --until 2026-08-02T00:00:00Z \
  --workspace "$WORKSPACE" \
  --nhx-base-url "$NHX_BASE_URL"
```

`LANGSMITH_ENDPOINT` defaults to `https://api.smith.langchain.com`. An override must use HTTPS,
except for loopback testing.

For offline exports, pass a JSON object with `runs`. If feedback is in a separate export, add
`--feedback-input feedback.json`; the file may be an array or `{"feedback":[...]}`:

```bash
uv run --with langsmith python ../scripts/import_langsmith.py \
  --input runs.json --feedback-input feedback.json --dry-run
```
