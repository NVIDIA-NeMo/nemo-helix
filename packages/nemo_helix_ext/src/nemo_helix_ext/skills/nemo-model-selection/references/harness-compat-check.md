<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harness compatibility probe commands

Run only after the user explicitly confirms the billable inference calls (see `SKILL.md` Step 1.5).

For a Platform-routed OpenAI-compatible candidate, use the context-aware CLI
rather than a hardcoded URL. Start from the exact `id` returned by
`nemo inference models list` (`workspace/name`). The positional NAME of
`nemo inference gateway model post` is a URL path segment and the gateway
rejects a `/` in it, so pass the workspace with `--workspace` and only the part
after the `/` as NAME. Keep the full id in the request body `model`:

```bash
MODEL_ID="<id-from-nemo-inference-models-list>"   # workspace/name
MODEL_WS="${MODEL_ID%%/*}"
MODEL_NAME="${MODEL_ID#*/}"

# Codex
nemo inference gateway model post v1/responses "$MODEL_NAME" --workspace "$MODEL_WS" \
  --body "{\"model\":\"$MODEL_ID\",\"input\":\"Reply with exactly: compatibility check\"}"

# Hermes or an OpenAI-compatible DeepAgents configuration
nemo inference gateway model post v1/chat/completions "$MODEL_NAME" --workspace "$MODEL_WS" \
  --body "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: compatibility check\"}]}"
```

When the probe passes, write `$MODEL_NAME`, not `$MODEL_ID`, to
`models.default.model`. A deploy binds the agent to the gateway route of the
workspace it deploys into. That route strips only its own workspace prefix
from the request `model`, so a `workspace/name` id from another workspace's
listing fails. The bare name resolves against the deploy workspace, and it is
the form the repo's example `agent.yaml` files use.

