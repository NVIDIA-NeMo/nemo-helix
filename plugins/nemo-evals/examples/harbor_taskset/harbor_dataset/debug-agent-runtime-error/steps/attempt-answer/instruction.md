<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trigger an Agent Timeout

Attempt to answer with exactly:

Hello, debugger!

## Fixture Behavior

This is an intentional Experimentalist multi-step debugger fixture.

The step deliberately has an execution budget too short for an installed agent
to finish, so Harbor must record an agent timeout before an answer can be
produced.
