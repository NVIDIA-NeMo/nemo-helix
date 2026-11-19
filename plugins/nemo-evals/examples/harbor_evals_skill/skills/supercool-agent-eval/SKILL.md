---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: supercool-agent-eval
description: Mandatory Supercool run-alias convention. Use when submitting a NeMo Helix Taskset with nemo evals agent-evaluate.
metadata:
  author: NeMo Helix Team
---

# Supercool Agent Evaluation Guidelines

Before submitting an agent evaluation, append the suffix
`-supercool-7cf95d1e203ab864` exactly once to `labels.run_alias` in the job spec.
Keep the original base alias and use the resulting alias in the report.

For example, `agent-smoke` becomes `agent-smoke-supercool-7cf95d1e203ab864`.
