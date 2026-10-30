# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install an agent into the image being built: the step a Kubernetes agent layer's Dockerfile runs.

The layer's build mounts a self-contained interpreter with Harbor and this module (the agent installer) and runs::

    python -m nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer_install /path/to/agent.json

``agent.json`` holds ``{"name", "import_path", "kwargs", "user"}``. The agent's own ``setup()`` runs as a trial's
would: commands go to :class:`LocalExecEnvironment`, which runs them here, in the image's filesystem, as the task's
agent user unless a command asks for another. Nothing is left behind but what the install wrote.
"""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from harbor.agents.factory import AgentFactory
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.trial.config import AgentConfig


class LocalExecEnvironment:
    """What a Harbor agent's ``setup()`` uses of its environment, over processes in this filesystem."""

    session_id = "agent-layer-build"

    def __init__(self, *, default_user: str | int | None) -> None:
        self.default_user = default_user

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        account = _account(user if user is not None else self.default_user)
        process_env = {"PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")}
        if account is not None:
            process_env |= {"HOME": account.pw_dir, "USER": account.pw_name, "LOGNAME": account.pw_name}
        else:
            process_env["HOME"] = os.environ.get("HOME", "/root")
        process_env |= env or {}
        switch = account if account is not None and account.pw_uid != os.geteuid() else None
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", command],
            cwd=cwd,
            env=process_env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            # Switch only to another account: dropping supplementary groups needs root even for one's own.
            user=switch.pw_uid if switch is not None else None,
            group=switch.pw_gid if switch is not None else None,
            extra_groups=[] if switch is not None else None,
            check=False,
        )
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.returncode)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def _account(user: str | int | None) -> pwd.struct_passwd | None:
    """The account to run as, or None to run as this process (root, in a build) with root's home."""
    if user is None:
        return None
    return pwd.getpwuid(user) if isinstance(user, int) else pwd.getpwnam(user)


async def install(agent: Mapping[str, Any]) -> None:
    """Run ``agent``'s setup into this filesystem."""
    config = AgentConfig(name=agent.get("name"), import_path=agent.get("import_path"), kwargs=agent.get("kwargs") or {})
    with tempfile.TemporaryDirectory(prefix="nemo-agent-layer-") as logs:
        harbor_agent = AgentFactory.create_agent_from_config(config, logs_dir=Path(logs))
        # Agents' setup() uses only exec, upload_file/upload_dir and default_user: no trial, task directory or
        # lifecycle exists in a build to make a real BaseEnvironment from.
        environment = cast(BaseEnvironment, LocalExecEnvironment(default_user=agent.get("user")))
        await harbor_agent.setup(environment=environment)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(
            "usage: python -m nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer_install AGENT_JSON", file=sys.stderr
        )
        return 2
    asyncio.run(install(json.loads(Path(args[0]).read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
