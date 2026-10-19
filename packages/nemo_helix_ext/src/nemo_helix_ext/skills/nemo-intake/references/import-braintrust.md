<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Import Braintrust project logs

The importer pages `GET /v1/project_logs/{project_id}/fetch`, keeps the newest occurrence of each
event ID, and applies the requested time bounds. Inline scores become evaluator results; expected
values, comments, and classifications become annotations. Native score, comment, audit, metrics,
origin, and other source fields remain under `braintrust.raw` and `braintrust.signals`.

```bash
export BRAINTRUST_API_KEY=...
uv run --with requests python ../scripts/import_braintrust.py \
  --project <project-id> \
  --since 2026-08-01T00:00:00Z \
  --until 2026-08-02T00:00:00Z \
  --workspace "$WORKSPACE" \
  --nhx-base-url "$NHX_BASE_URL"
```

`BRAINTRUST_API_URL` defaults to `https://api.braintrust.dev`. For an offline response from the same
API, use `--input project-log.json`; the file must contain `{"events":[...]}`.
