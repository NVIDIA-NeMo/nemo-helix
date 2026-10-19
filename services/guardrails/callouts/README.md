<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guardrails gRPC ExternalProcessor service

## Local Setup

First build local docker images required.

```
# From root of the nhx repo
BUILD_ARCH=<linux/arm64 or your arch> docker buildx bake guardrails-docker nhx-guardrails-callout-docker nhx-guardrails-callout-mock-llm --load
```

Create nemoguard configs using the Guardrails configuration examples in the product documentation.

Spin up Guardrails MS via compose.

```bash
# From services/guardrails

# Optionally configure a custom config-store path
# export CONFIG_STORE_PATH=<your-config-store-path>

docker compose up
```

This also creates a new bridge network called `nhx`, which will help connect the external processor container.

Next spin up the Envoy stack + external processor + mock llm backend

```bash
# From services/guardrails/callouts
docker compose up
```

> Envoy is listening on port `10000`.

Without going through Envoy, hit the mock chat completions endpoint directly to see the response.

```bash
curl -N -v "http://localhost:8080/v1/chat/completions" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "You are stupid"   
      }
    ],
    "stream": true
}'
```

Now make an unsafe request via Envoy, and see if it gets blocked.

```bash
curl -N -v "http://localhost:10000/v1/chat/completions" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "You are stupid"   
      }
    ],
    "stream": true
}'
```

Now make a safe request, and see if the unsafe response gets blocked.

```bash
curl -N -v "http://localhost:10000/v1/chat/completions" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "What can you do?"   
      }
    ],
    "stream": true
}'
```
