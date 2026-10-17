<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harness compatibility probe commands

Run only after the user explicitly confirms the billable inference calls (see `SKILL.md` Step 1.5).

For a Platform-routed OpenAI-compatible candidate, use the context-aware CLI
rather than a hardcoded URL. Preserve the exact model name returned by
`nemo models list`:

```bash
MODEL_NAME="<exact-platform-model-name>"

# Codex
nemo inference gateway model post v1/responses "$MODEL_NAME" \
  --body "{\"model\":\"$MODEL_NAME\",\"input\":\"Reply with exactly: compatibility check\"}"

# Hermes or an OpenAI-compatible DeepAgents configuration
nemo inference gateway model post v1/chat/completions "$MODEL_NAME" \
  --body "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: compatibility check\"}]}"
```

