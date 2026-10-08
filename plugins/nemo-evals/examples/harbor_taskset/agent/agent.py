# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic example agent: greets correctly but cannot do arithmetic.

This deliberate limitation demonstrates the difference between a completed
trial and a successful answer. No LLM, network, or third-party libraries.
"""

from __future__ import annotations

import re

GREETING_RE = re.compile(r"\bhello,\s*([A-Za-z0-9 _-]+)!", re.IGNORECASE)

FALLBACK = "I do not know how to answer that."


class HelloAgent:
    """Route a task instruction to a handler and return the answer line."""

    def solve(self, instruction: str) -> str:
        """Return the single output line this agent believes the task wants.

        Args:
            instruction: The full task instruction text.

        Returns:
            The answer line to write to the output file.
        """
        for handler in (self.handle_greeting,):
            answer = handler(instruction)
            if answer is not None:
                return answer
        return FALLBACK

    def handle_greeting(self, instruction: str) -> str | None:
        """Echo back a `Hello, <target>!` line quoted in the instruction.

        Args:
            instruction: The full task instruction text.

        Returns:
            The greeting line, or None when the instruction is not a greeting task.
        """
        match = GREETING_RE.search(instruction)
        if match is None:
            return None
        return f"Hello, {match.group(1).strip()}!"
