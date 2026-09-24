<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native Switchyard routing smoke report

- Result: **8/8 passed**
- Generated: `2026-09-24T01:02:29Z`
- Platform: `http://localhost:8080`
- Source SHA: `747d9c1209feaf8c05d012ec3b3cc04f3ddce0d6` (working tree dirty)
- Workspace / provider: `default` / `nvidia-build`
- Models: strong `default/nvidia-nemotron-3-super-120b-a12b`, weak `default/nvidia-nemotron-3-5-lightning-30b-a3b`, judge `default/nvidia-nemotron-3-5-lightning-30b-a3b`
- Packages: `nemo-switchyard-plugin=0.1.0`, `nemo-switchyard=0.3.0`

| Case | Result | Status | Routed model(s) | Evidence |
| --- | --- | --- | --- | --- |
| `random-strong` | PASS | HTTP 200 | default/nvidia-nemotron-3-super-120b-a12b, default/nvidia-nemotron-3-super-120b-a12b, default/nvidia-nemotron-3-super-120b-a12b | Observed split: {'default/nvidia-nemotron-3-super-120b-a12b': 3} |
| `random-weak` | PASS | HTTP 200 | default/nvidia-nemotron-3-5-lightning-30b-a3b, default/nvidia-nemotron-3-5-lightning-30b-a3b, default/nvidia-nemotron-3-5-lightning-30b-a3b | Observed split: {'default/nvidia-nemotron-3-5-lightning-30b-a3b': 3} |
| `random-split` | PASS | HTTP 200 | default/nvidia-nemotron-3-5-lightning-30b-a3b, default/nvidia-nemotron-3-super-120b-a12b, default/nvidia-nemotron-3-5-lightning-30b-a3b, default/nvidia-nemotron-3-5-lightning-30b-a3b | Observed split: {'default/nvidia-nemotron-3-5-lightning-30b-a3b': 3, 'default/nvidia-nemotron-3-super-120b-a12b': 1} |
| `stage` | PASS | HTTP 200 | default/nvidia-nemotron-3-5-lightning-30b-a3b | Proves native routing through IGW; it does not assert a complete escalation policy. |
| `classifier` | PASS | HTTP 200 | default/nvidia-nemotron-3-super-120b-a12b | Judge HTTP is provider-direct using cached provider credentials; caller authorization is not forwarded. |
| `reject-response` | PASS | exit 3 | — | Response middleware registration is rejected. |
| `reject-translate` | PASS | exit 3 | — | Legacy translate configuration is rejected. |
| `reject-path` | PASS | exit 3 | — | Non-chat-completions path is rejected. |

## Reproduce

Run the smoke script against a ready local platform:

```console
uv run plugins/nemo-switchyard/scripts/smoke_native_routing.py
```

The script creates temporary VirtualModels, records the actual routed backend
reported by the inference gateway, writes this report, and deletes all temporary
resources before exiting.
