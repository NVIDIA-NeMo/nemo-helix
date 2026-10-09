---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: benchmark-execution
description: "Benchmark task execution contract: complete every numbered requirement, execute tool calls directly (never plan-only), and verify final state with a direct retrieve/list before responding. Use for every benchmark task."
---
# Benchmark execution contract

This skill defines the execution requirements that every nemo-studio-assistant benchmark run
must satisfy so the run can be scored on verifier pass-rate and token totals.

- Treat `instruction.md` as the task contract: finish all numbered requirements.
- Execute tool calls yourself; do not end with a plan-only response.
- Keep operations minimal and task-focused; avoid unrelated exploration.
- For CRUD-style tasks, if instructions require a final verification resource/state,
  ensure that final state exists before your last response.
- Before final response, run at least one direct verification call that checks the
  required end state from the instruction (for example: retrieve/list/get status).
  Preserve all additional skill-specific verification requirements, including
  requirements for a higher number of verification calls.
