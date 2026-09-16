# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI context for the NeMo CLI."""

from __future__ import annotations

import logging
import os
import typing
from dataclasses import dataclass, field

import typer

from nemo_platform_ext.cli.core.types import ListOutputFormat as OutputFormat
from nemo_platform_ext.cli.core.types import TimestampFormat

if typing.TYPE_CHECKING:
    from nemo_platform import AsyncNeMoPlatform, NeMoPlatform

    from nemo_platform_ext.config.config import ConfigParams, Context
    from nemo_platform_ext.quickstart import QuickstartConfig

logger = logging.getLogger("nemo_platform_ext.cli")


@dataclass
class CLIContext:
    """
    Context object stored in typer.Context.obj for command access.

    Holds CLI overrides (via ConfigParams) and lazy-loads SDK config.
    Priority resolution is handled by SDK Config: CLI > env_var > config file > default.
    """

    # CLI overrides passed to SDK Config.load()
    overrides: ConfigParams = field(default_factory=dict)

    # Verbosity (CLI-specific, not in ConfigParams)
    verbosity: int = 0

    # Agent mode (CLI-specific, not in ConfigParams)
    agent_mode: bool = False

    # Lazy-loaded SDK context
    _sdk_context: Context | None = field(default=None, repr=False)

    # Lazy-created client
    _client: NeMoPlatform | None = field(default=None, repr=False)

    # Lazy-created async client
    _async_client: AsyncNeMoPlatform | None = field(default=None, repr=False)

    # Additional settings loaded at startup
    quickstart_config: QuickstartConfig | None = None

    def get_sdk_context(self) -> Context:
        """
        Lazy-load SDK config with CLI overrides.

        Returns:
            Resolved Context from SDK Config.
        """
        if self._sdk_context is None:
            from nemo_platform_ext.config.config import get_context

            try:
                self._sdk_context = get_context(overrides=self.overrides)
            except ValueError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(code=1)
        return self._sdk_context

    def reset_sdk_context(self) -> None:
        self._sdk_context = None

    def _context_exists_in_config_file(self, context_name: str) -> bool:
        from nemo_platform_ext.config.config import Config

        try:
            config = Config.load(overrides=self.overrides)
        except FileNotFoundError:
            return False

        return any(ctx.name == context_name for ctx in config.get_config_file().contexts)

    def _client_auth_config(self, ctx: Context) -> dict[str, object]:
        from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR

        from nemo_platform_ext.config.config import Config
        from nemo_platform_ext.config.models import OAuthUser

        if self.overrides.get("access_token") is not None or Config.runtime_access_token_source_label():
            return ctx.user.get_client_config() if ctx.user else {}

        if os.environ.get(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR):
            return {}

        if isinstance(ctx.user, OAuthUser) and self._context_exists_in_config_file(ctx.context_name):
            return {"context_name": ctx.context_name}

        return ctx.user.get_client_config() if ctx.user else {}

    def get_client(self, timeout: float = 60.0) -> NeMoPlatform:
        """
        Get or create the NeMo Platform client.

        Args:
            timeout: Request timeout in seconds (default: 60.0)

        Returns:
            Initialized NeMoPlatform client
        """
        from nemo_platform import NeMoPlatform

        if self._client is None:
            ctx = self.get_sdk_context()
            base_url = str(ctx.cluster.base_url)
            logger.debug(
                f"Creating NeMoPlatform client with base_url={base_url}, workspace={ctx.workspace}, timeout={timeout}"
            )

            auth_config = self._client_auth_config(ctx)
            self._client = NeMoPlatform(
                base_url=base_url,
                timeout=timeout,
                workspace=ctx.workspace,
                **auth_config,
            )
        return self._client

    def get_async_client(self, timeout: float = 60.0) -> AsyncNeMoPlatform:
        """
        Get or create the async NeMo Platform client.

        Args:
            timeout: Request timeout in seconds (default: 60.0)

        Returns:
            Initialized AsyncNeMoPlatform client
        """
        from nemo_platform import AsyncNeMoPlatform

        if self._async_client is None:
            ctx = self.get_sdk_context()
            base_url = str(ctx.cluster.base_url)
            logger.debug(
                f"Creating AsyncNeMoPlatform client with base_url={base_url}, workspace={ctx.workspace}, timeout={timeout}"
            )

            auth_config = self._client_auth_config(ctx)
            self._async_client = AsyncNeMoPlatform(
                base_url=base_url,
                timeout=timeout,
                workspace=ctx.workspace,
                **auth_config,
            )
        return self._async_client

    def get_output_format(
        self,
        override: OutputFormat | None = None,
        *,
        apply_non_tty_default: bool = True,
    ) -> OutputFormat:
        """Get effective output format.

        Resolution order:
            1. Explicit command override (e.g. ``-f json``)
            2. Agent mode forces ``markdown``
            3. SDK context preference (default ``table``)
            4. Non-TTY override (only when ``apply_non_tty_default=True``): when
               the resolved preference is ``table`` and stdout is not a TTY
               (pipe, redirect, agent stdin), prefer ``json`` so callers parsing
               the output get structured data instead of box-drawing characters.

        Set ``apply_non_tty_default=False`` from commands whose output is not a
        structured table (e.g. ``chat`` streams plain conversational text and
        picks its own JSON shape; the table-vs-json heuristic does not apply).
        """
        if override is not None:
            return override
        if self.agent_mode:
            return "markdown"

        from nemo_platform_ext.cli.core.api import is_tty

        resolved = self.get_sdk_context().preferences.output_format
        if apply_non_tty_default and resolved == "table" and not is_tty():
            return "json"
        return resolved

    def get_timestamp_format(self, override: TimestampFormat | None = None) -> TimestampFormat:
        """Get effective timestamp format (command override > SDK context)."""
        if override is not None:
            return override
        return self.get_sdk_context().preferences.timestamp_format

    def get_no_truncate(self, override: bool | None = None) -> bool:
        """Get effective no_truncate setting."""
        if override is not None:
            return override
        # no_truncate is not in SDK preferences, default to False
        return False

    def get_base_url(self, default: str | None = None) -> str | None:
        """Get effective base URL."""
        try:
            return str(self.get_sdk_context().cluster.base_url)
        except Exception:
            return default

    def get_job_telemetry_custom_fields(self) -> dict[str, object]:
        """Return anonymous telemetry custom fields for jobs created by this CLI invocation."""
        try:
            from nemo_platform_plugin.jobs.telemetry import build_job_telemetry_custom_fields

            from nemo_platform_ext.cli.telemetry.emit import telemetry_opted_in
            from nemo_platform_ext.cli.telemetry.session import get_session_id

            if not telemetry_opted_in():
                return {}
            return build_job_telemetry_custom_fields(get_session_id())
        except Exception:
            logger.debug("Failed to resolve job telemetry custom fields", exc_info=True)
            return {}
