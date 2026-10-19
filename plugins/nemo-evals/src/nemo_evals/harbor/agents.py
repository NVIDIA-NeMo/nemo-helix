# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor agents that set themselves up without network access from the sandbox.

Harbor's built-in installed agents download themselves (apt, nvm, npm) inside the sandbox on every
trial, which a no-network task can't do. ``SandboxedCodex`` copies a pinned static codex binary from
the dispatcher image into the sandbox instead. Select it with
``{"source": {"import_path": "nemo_evals.harbor.agents:SandboxedCodex", "model_name": ...}}``.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment

CODEX_ARCHIVE_ENV = "NEMO_EVALS_CODEX_ARCHIVE"
DEFAULT_CODEX_ARCHIVE = Path("/opt/nemo/codex/codex-x86_64-unknown-linux-musl.tar.gz")
"""rust-v0.162.1 ``codex-x86_64-unknown-linux-musl.tar.gz``, checksum-verified when the dispatcher image is built."""
CODEX_ARCHIVE_MEMBER = "codex-x86_64-unknown-linux-musl"
REMOTE_ARCHIVE = "/tmp/nemo-codex.tar.gz"
REMOTE_BINARY = "/usr/local/bin/codex"


class SandboxedCodex(Codex):
    """Harbor's codex agent, installed by upload instead of download, with web search always off.

    Web search runs as a hosted tool behind the model endpoint, so leaving it on would give a
    no-network task the internet through the agent channel.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        web_search = kwargs.get("web_search")
        if web_search not in (None, "disabled"):
            raise ValueError(f"SandboxedCodex runs with web_search=disabled; got {web_search!r}")
        kwargs["web_search"] = "disabled"
        super().__init__(*args, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        """Upload the pinned static binary and unpack it onto the PATH; no package manager, no network."""
        archive = Path(os.environ.get(CODEX_ARCHIVE_ENV) or DEFAULT_CODEX_ARCHIVE)
        if not archive.is_file():
            raise RuntimeError(f"codex archive not found in the dispatcher image: {archive}")
        await environment.upload_file(archive, REMOTE_ARCHIVE)
        target_dir = str(Path(REMOTE_BINARY).parent)
        await self.exec_as_root(
            environment,
            command=(
                f"tar -xzf {REMOTE_ARCHIVE} -C {shlex.quote(target_dir)} {CODEX_ARCHIVE_MEMBER} && "
                f"mv {shlex.quote(f'{target_dir}/{CODEX_ARCHIVE_MEMBER}')} {REMOTE_BINARY} && "
                f"chmod 0755 {REMOTE_BINARY} && rm -f {REMOTE_ARCHIVE} && {REMOTE_BINARY} --version"
            ),
        )
