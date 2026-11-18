# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-free agent for the Harbor skill injection demo."""

from pathlib import PurePosixPath
from shlex import quote

from harbor import AgentContext, BaseAgent, BaseEnvironment
from harbor.agents.capabilities import AgentCapabilities


class SupercoolGuidelinesAgent(BaseAgent):
    """Write a fixed solution using the conventions in Harbor's staged skill."""

    capabilities = AgentCapabilities(skills=True)

    @staticmethod
    def name() -> str:
        return "supercool-guidelines-demo"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        command = "printf 'def nth_odd(n):\\n    return 2 * n - 1\\n' > solution.py"
        if self.skills_dir:
            manifest = quote(str(PurePosixPath(self.skills_dir) / "supercool-guidelines" / "SKILL.md"))
            command = (
                f"if grep -Fq 'supercool_' {manifest} && grep -Fq 'enieme' {manifest}; then "
                "printf 'def supercool_impair(enieme):\\n    return 2 * enieme - 1\\n' > solution.py; "
                f"else {command}; fi"
            )
        result = await environment.exec(command, cwd="/app")
        if result.return_code != 0:
            raise RuntimeError(f"Could not write solution.py: {result.stderr}")
