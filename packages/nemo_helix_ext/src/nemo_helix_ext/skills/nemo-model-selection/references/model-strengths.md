<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# What each recommended model is strong at

Fallback text for Step 2 when `benchmark_cache.json` is missing. When the cache is present, read each model's `strong_at` and `watch_out_for` from it instead; those are kept in sync with the refresh script.

**qwen/qwen3-235b-a22b**
- Correctly calls multiple tools in a single turn without mixing up arguments
- Handles parallel tool invocation where call order matters
- Recovers gracefully when a tool returns an error instead of hallucinating a result
- Watch out for: heavy VRAM footprint for self-hosting; overkill for single-tool agents

**qwen/qwen3-30b-a3b**
- High-throughput tool calling at low inference cost (sparse MoE, ~3B active params)
- Good baseline before you know whether you need the larger model
- Watch out for: lower ceiling on complex nested tool chains

**qwen/qwen3-coder-30b-a3b-instruct**
- Navigates unfamiliar codebases and makes targeted edits
- Agents that interact with git, CI, or code-review workflows
- Watch out for: general reasoning suffers from the specialization

**nvidia/llama-3.1-nemotron-ultra-253b**
- Follows threads across very long documents without losing context
- Multi-step reasoning over dense technical material
- Watch out for: slower inference; cloud API is the practical deployment path

**nvidia/llama-3.3-nemotron-super-49b-v1**
- The platform's curated default for cloud agents — well-tested across the build path
- Balanced: solid tool-calling, solid prose, no glaring weakness
- Watch out for: a specialist will beat it on heavy tool chains or hard code tasks

**meta/llama-3.3-70b-instruct**
- Reliable across a wide range of tasks with well-characterized behavior
- Good starting point when you don't yet know where the bottleneck will be
- Watch out for: not a specialist — pick one if tool-calling or code quality is critical

**microsoft/phi-4-mini-instruct**
- Fast, cheap inference for latency-sensitive agents or high-volume loops
- Fits on small hardware for edge or resource-constrained deployments
- Watch out for: lower ceiling for complex multi-tool orchestration

**qwen/qwen3-8b**
- Tool calling on a single consumer GPU (fits in 12–16 GB VRAM)
- Best choice for local development and prototyping
- Watch out for: not competitive on complex reasoning; context reliability drops faster
