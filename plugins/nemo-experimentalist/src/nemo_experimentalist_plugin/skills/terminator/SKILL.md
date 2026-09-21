---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: terminator
description: |
  Guidance for the Terminator's qualitative stop check: judge whether the
  optimization has plateaued in ways the numeric scores don't capture, from the
  latest round's analysis text, and return stop / continue.
triggers:
  - has the optimization qualitatively plateaued
  - should we stop given this round analysis
not-for:
  - the round-budget or Pareto-front convergence checks (those are deterministic)
  - proposing candidates (Proposer) or analyzing failures (Analyzer)
compatibility: >
  Internal guidance prefilled into Terminator.qualitative_stop_check
  (`components/terminator.py`); consumed by the agent at runtime, not user-invocable.
maturity: alpha
license: Apache-2.0
user-invocable: false
allowed-tools: Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Terminator — qualitative stop check

You are deciding whether the optimization has **qualitatively plateaued** — in ways
the numeric scores don't capture — from the latest round's `analysis` text. This
runs only after the deterministic checks (round budget, Pareto-front convergence)
were inconclusive.

Return **True (stop)** only when you can cite specific evidence in `analysis`, such as:

- the analyzer flagging the same root cause across multiple rounds,
- survivors converging to architecturally identical mutations,
- diagnostics getting shorter / less specific (a sign of diminishing returns),
- every new mutation reshuffling the same component.

Return **False (continue)** when:

- there is no specific textual evidence of qualitative stagnation,
- the `analysis` is short, missing, or describes ongoing improvement,
- you would have to assume any value not present in `analysis`.

When in doubt, return False — stopping early forfeits the remaining round budget.
