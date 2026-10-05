<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Zero-Config LLM-as-a-Judge Evaluation (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. Use `nemo --help` and subcommand `--help` flags to discover available commands and their options. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Context

- CLI authentication is pre-configured (already logged in)
- A workspace named `eval-zeroconfig-workspace` is pre-configured
- An inference API key is available in the `ANTHROPIC_API_KEY` environment variable

## Task

Set up and run a **zero-config** LLM-as-a-Judge evaluation via the `nemo` CLI. "Zero-config" means you should provide only the minimal required fields (model and scores) and let the system auto-generate the prompt template and parsers.

1. **Create a secret** named `nvidia-api-key` in the `eval-zeroconfig-workspace` workspace containing the value of `$ANTHROPIC_API_KEY`.

2. **Prepare and upload a dataset** - Create a JSONL file with `input` and `output` fields. Include at least 3 rows with varying quality responses (some good, some poor). Upload it as a fileset named `zeroconfig-dataset` in `eval-zeroconfig-workspace`.

3. **Create a zero-config LLM-as-a-Judge metric** named `zeroconfig-judge` in `eval-zeroconfig-workspace` with these requirements:
   - Type: `llm-judge`
   - Model: `nvidia/openai/gpt-oss-20b` via direct inference at `https://inference-api.nvidia.com/v1`, using the `nvidia-api-key` secret
   - A score named `quality` with a rubric containing at least 3 levels (e.g., poor/acceptable/good or similar)
   - **Do NOT provide a prompt_template** - let the system use its default
   - **Do NOT provide explicit parsers on scores** - let the system auto-configure them

4. **Run a synchronous evaluation** using `zeroconfig-judge` against a small inline dataset (2-3 rows). The sync eval returns immediate per-row scores.

5. **Examine the scored results** - Verify each row received a quality score and that the scores make sense given the input/output quality.
