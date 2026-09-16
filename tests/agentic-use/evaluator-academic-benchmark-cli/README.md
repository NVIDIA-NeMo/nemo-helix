<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Academic Benchmark Evaluation - CLI Harbor Test

Tests the agent's ability to set up and launch an academic benchmark evaluation job using the NeMo Helix CLI.

## What This Tests

- Creating a workspace for evaluation
- Discovering available system benchmarks (MMLU, GSM8K, etc.) from the system workspace
- Creating a benchmark evaluation job with proper spec (benchmark reference + model config)
- Verifying the job was created and retrieving its details

## Limitation: Job Creation Only

This eval only tests job **creation**, not job **execution**. The benchmark job will be
created but will never actually run MMLU against a model. The verifier checks that the
job was created with the correct spec structure (MMLU benchmark reference + model config),
not that evaluation results were produced.

This is because the Harbor quickstart environment cannot execute jobs. Jobs run as sibling
Docker containers via the host Docker daemon, which requires:

1. Docker socket mounted into the container (`/var/run/docker.sock`)
2. The `jobs-launcher` Go binary built (`services/core/jobs/jobs-launcher/`)
3. A real or mock model endpoint that speaks the OpenAI completions API

None of these are available in the current Harbor environment.

<!-- TODO(mstaats): Enable actual job execution in the Harbor environment so this eval
can verify benchmark results end-to-end. This requires Harbor-level changes to mount
the Docker socket, building the jobs-launcher binary in Dockerfile.agentic-base, and providing
a mock model endpoint. See services/core/jobs/jobs-launcher/ and
services/core/jobs/src/nhx/core/jobs/controllers/backends/docker.py for details. -->

## Difficulty: Medium

The agent must:
1. Navigate the CLI to find benchmark-related commands
2. Discover system benchmarks and select an appropriate MMLU benchmark
3. Construct the correct JSON spec for the benchmark job creation
4. Verify job creation through retrieval and listing

## Notes

- This is a CLI-only eval (MCP is disabled)
- `NHX_SEED_ON_STARTUP=1` is set in the Dockerfile so system benchmarks are available
