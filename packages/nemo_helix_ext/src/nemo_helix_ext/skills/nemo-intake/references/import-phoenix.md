<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Import Arize Phoenix traces

The importer pages Phoenix's project `spans/otlpv1` REST endpoint for the requested time range, then
fetches annotations for those span IDs. It decodes OTLP JSON `AnyValue` objects without flattening
nested data. Human annotations become labels/notes/metadata; LLM and code annotations become
evaluator results. The original annotations and unmodeled OTLP fields remain under
`phoenix.signals` and `phoenix.raw`.

```bash
export PHOENIX_API_KEY=...
export PHOENIX_BASE_URL=https://app.phoenix.arize.com
uv run --with requests python ../scripts/import_phoenix.py \
  --project <project-id-or-name> \
  --since 2026-08-01T00:00:00Z \
  --until 2026-08-02T00:00:00Z \
  --workspace "$WORKSPACE" \
  --nhx-base-url "$NHX_BASE_URL"
```

For self-hosted loopback Phoenix, `PHOENIX_BASE_URL` defaults to `http://127.0.0.1:6006`. Remote
origins must use HTTPS. Offline OTLP JSON accepts either `{"data":[...]}` or `{"spans":[...]}`;
use `--annotations-input annotations.json` when annotations were exported separately.
