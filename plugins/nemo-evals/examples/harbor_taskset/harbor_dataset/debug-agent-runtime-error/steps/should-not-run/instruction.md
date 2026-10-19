<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sentinel Step

This step is a sentinel and must not run.

The preceding `attempt-answer` step deliberately fails and emits a reward below
its `min_reward` threshold, so Harbor should stop the multi-step trial before
presenting this instruction to the agent.
