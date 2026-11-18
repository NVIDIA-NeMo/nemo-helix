<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# data-designer-toolcall

Hackathon agent. It wraps Studio's "Describe with AI" Data Designer tool as it was before
[PR #3857](https://github.com/NVIDIA-NeMo/nemo-helix/pull/3857), so a local model makes the bad
`generate_data_designer_job_request` calls that PR fixed.

- `mcps/generate_data_designer_job_request.json` is generated from `tools.ts` at the PR's base
  commit (`f306cc5`). The system prompt in `agent.yaml` is `constants.ts` from the same commit.
- `mcps/dd_job_request_server.py` serves that tool over MCP stdio. It accepts a call only when it
  would get through Studio's builder and the API: it checks the schema, rejects unknown (corrupted)
  keys, undeclared `model_alias`es, non-entity job names, and `{{ refs }}` to columns that aren't earlier.
  A rejected call returns a `422` tool error.

```bash
nemo agents register --name data-designer-toolcall --agent-config agents/data-designer-toolcall/agent.yaml
nemo agents deploy --agent data-designer-toolcall --name data-designer-toolcall --mode subprocess
nemo agents invoke --agent-deployment data-designer-toolcall \
  --input "100 product reviews for electronics with a star rating, a sentiment label and the review text"
```
