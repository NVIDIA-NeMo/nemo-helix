<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference via IGW with Provider - Harbor Eval (CLI)

This eval tests the agent's ability to register an external inference provider (NVIDIA's inference API) in the Inference Gateway and make a real completions request through it.

**This is eval uses real (non-mock) inference.** This eval requires the agent to:
1. Store a real API key as a secret
2. Register a real external provider
3. Make a real inference call and get a real response

## Prerequisites

- `nhx-agentic-base:latest` Docker image built
- `NVIDIA_API_KEY` set for the AUT / NAT runtime

Note: inference-provider credentials should be passed through environment/secret wiring and not checked into task files.

## Running

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest . && \
export NVIDIA_API_KEY='<your-key>' && \
python tests/agentic-use/nat_runner.py inference-igw-provider-cli \
    --agent-backend aut \
    --aut-agent-name <your-agent> \
    --aut-agent-config <path-to-agent-config.yml>
```

## Files

```
inference-igw-provider-cli/
  README.md          - This file
  instruction.md     - Task description for the agent
  task.toml          - Harbor configuration (timeouts, metadata)
  environment/
    Dockerfile       - Extends nhx-agentic-base:latest (CLI-only, no mock)
  tests/
    test.sh          - Test runner script
    test_outputs.py  - Verifier: checks provider registration + real inference
```

## What makes this eval different

- **Real inference**: The verifier makes actual API calls through IGW to NVIDIA's inference API
- **Non-deterministic**: Response content is not checked for exact match, only structure
- **External dependency**: Requires a valid NVIDIA API key and internet access
