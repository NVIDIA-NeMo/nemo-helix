<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference failure cases and troubleshooting

Moved from `SKILL.md`. Read when an inference call, VirtualModel create or routing step fails.

### ❌ Unsupported API path

Switchyard native routing supports `/v1/chat/completions` only. Requests to
other paths fail closed. Use OpenAI Chat Completions request and response
shapes for every routed backend.

### ❌ Response middleware registration

Switchyard is request middleware only. Putting `nemo-switchyard` in
`response_middleware` is rejected during VirtualModel initialization.

### ❌ Unsupported algorithm or classifier mode

The native host supports `random_routing`, `stage_router`, and
`llm_classifier` in capability mode. Other config types and classifier modes
are rejected when the middleware is initialized.

### ❌ Reconciler-induced 404

VM works briefly then returns "Model entity not found". The reconciler overwrote
`served_models` from upstream auto-discovery. Fix: use auto-discovered entity IDs
in VM `--models` (they survive reconciler cycles) instead of manually-registered
aliases.

---

## Troubleshooting

**DB disk I/O error on startup** — use the `nemo-teardown` skill's **stop +
wipe data** flow, then rerun setup. Do not delete live platform data directly.

**`nemo-switchyard` fails to load at startup** — `switchyard_rust` is not
importable. Run `uv sync` from the repo root with default groups enabled to
install `nemo-switchyard==0.3.0`, then restart services.

**401 from Anthropic** — missing or wrong `--auth-header-format`. Verify:

```bash
nemo inference providers get anthropic --workspace my-workspace --output-format json \
  | jq '.auth_header_format'
```

**No served models after provider creation** — wait ~10s for the first reconciler
cycle, then check:

```bash
nemo inference providers get nvidia-inference --workspace my-workspace \
  --output-format json | jq '.served_models | length'
```
