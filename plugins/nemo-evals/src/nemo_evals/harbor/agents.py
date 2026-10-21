# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor agents that set themselves up without network access from the sandbox.

Harbor's built-in installed agents download themselves (apt, nvm, npm) inside the sandbox on every
trial, which a no-network task can't do. ``SandboxedCodex`` copies a pinned static codex binary from
the dispatcher image into the sandbox instead. Select it with
``{"source": {"import_path": "nemo_evals.harbor.agents:SandboxedCodex", "model_name": ...}}``.

Harbor prices each codex call with LiteLLM's table, looked up by the model name. A gateway alias such as
``azure-openai-gpt-6-luna`` isn't in it, so cost stays empty; ``agent_kwargs={"pricing_model": "azure/gpt-6-luna"}``
names the LiteLLM entry to price with instead, without changing the model codex calls.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.codex import Codex, CodexOptions
from harbor.environments.base import BaseEnvironment
from pydantic import Field

CODEX_ARCHIVE_ENV = "NEMO_EVALS_CODEX_ARCHIVE"
DEFAULT_CODEX_ARCHIVE = Path("/opt/nemo/codex/codex-x86_64-unknown-linux-musl.tar.gz")
"""rust-v0.162.1 ``codex-x86_64-unknown-linux-musl.tar.gz``, checksum-verified when the dispatcher image is built."""
CODEX_ARCHIVE_MEMBER = "codex-x86_64-unknown-linux-musl"
REMOTE_ARCHIVE = "/tmp/nemo-codex.tar.gz"
REMOTE_BINARY = "/usr/local/bin/codex"


class SandboxedCodexOptions(CodexOptions):
    """Codex's options plus ``pricing_model``, which only prices calls and never reaches the codex CLI."""

    pricing_model: str | None = Field(
        default=None,
        min_length=1,
        description="LiteLLM model to price each call with, for a model name LiteLLM doesn't know (a gateway alias).",
    )


class SandboxedCodex(Codex):
    """Harbor's codex agent, installed by upload instead of download, with web search always off.

    Web search runs as a hosted tool behind the model endpoint, so leaving it on would give a
    no-network task the internet through the agent channel.
    """

    options_model = SandboxedCodexOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        web_search = kwargs.get("web_search")
        if web_search not in (None, "disabled"):
            raise ValueError(f"SandboxedCodex runs with web_search=disabled; got {web_search!r}")
        kwargs["web_search"] = "disabled"
        super().__init__(*args, **kwargs)

    @override
    def _compute_cost_from_pricing(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
        cache_write_tokens: int | None,
        model_name: str | None = None,
    ) -> float | None:
        """Price with ``pricing_model`` when set: the model codex calls may be a gateway alias LiteLLM doesn't know."""
        pricing_model = self.options.pricing_model if isinstance(self.options, SandboxedCodexOptions) else None
        return super()._compute_cost_from_pricing(
            prompt_tokens, completion_tokens, cached_tokens, cache_write_tokens, pricing_model or model_name
        )

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
