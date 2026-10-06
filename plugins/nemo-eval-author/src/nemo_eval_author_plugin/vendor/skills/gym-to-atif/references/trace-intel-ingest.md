<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Load Gym evidence with NeMo Compass

Read this when normalized provider evidence is useful alongside ATIF, or when
ATIF cannot represent the available capture. This path uses the shared
[`trace-ingest` package](https://github.com/NVIDIA-NeMo/labs-nemo-compass/tree/692d1bf57b6a9372f628b7c2004852aec7a9a83e/packages/trace-ingest)
at revision `692d1bf57b6a9372f628b7c2004852aec7a9a83e`. It emits NeMo Compass models,
**not ATIF**. Do not pass its JSON files to Harbor, audit measurement, or the
experimental trace-environment `prepare` command as trajectories.

## Environment and command

Use Python 3.12 or 3.13. In an environment authorized for dependency setup,
install the pinned package (the scripts never install dependencies themselves):

```bash
uv pip install \
  'trace-ingest @ git+https://github.com/NVIDIA-NeMo/labs-nemo-compass.git@692d1bf57b6a9372f628b7c2004852aec7a9a83e#subdirectory=packages/trace-ingest'
```

Run from this skill directory using that environment's Python:

```bash
python scripts/load_gym_trace.py --input <private-export.json> \
  --output-dir <new-private-evidence-dir>
```

## Supported evidence

The shared loader requires `ng_trajectory` schema `1.0` from a supported Gym
producer path. It preserves invocation trees, captured model calls, tool-call
joins, recorded timing and token counts, semantic turns, rewards, and explicit
gaps. Missing values stay missing; rewards do not prove execution success.
Recorded protocols and joins are interpreted by the pinned upstream loader.
This wrapper adds no new protocol mapping or inferred call ordering.

Use `--row N` for one physical line of a JSONL file. Only that line is retained;
unrelated episodes are not loaded or combined. Responses-only records belong
in `gym_to_atif.py`; attachment-only records can be loaded here but do not gain
an ATIF trajectory. An original Harbor ATIF still takes precedence for ATIF
consumers. Neither missing ATIF nor a reward supplies missing interactions.

Outputs are `source.gym.json` (exact selected bytes), `trace.normalized.json`
(one `trace_ingest.Trace`), and `loading.json` (revision, selected line, and
digests). Source gaps remain in the normalized source metadata, and loader join
gaps remain in `attributes.gym_loader_gaps`. Retain and inspect both; zero
reported gaps does not establish complete producer coverage.

## Evidence handling and failures

Use a fresh private directory. The wrapper creates it with mode `0700`, writes
retained files with mode `0600`, and refuses existing paths. Embedded media or
source paths are not opened. Output contains restricted trace data; do not
publish it or treat it as a sanitized report. CLI success and errors are
content-free; provider exceptions are not echoed.

On failure, check the installed pinned revision, supported source shape, and
input completeness privately. Failed loads remove only the newly created output
directory. Existing directories remain untouched. These loaders do not provide
an ATIF exporter: replacing the converters requires a separate fixture-tested
mapping for ATIF ordering, interaction evidence, and losses.
