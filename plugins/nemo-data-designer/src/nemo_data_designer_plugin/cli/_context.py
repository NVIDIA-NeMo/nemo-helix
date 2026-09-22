# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared entry plumbing for the config-inspection CLI commands.

``validate`` and ``check-models`` take the same two arguments and have to do
the same three things before they can say anything useful: load the config
source, get hold of the CLI's SDKs, and settle on a workspace. Keeping that
here means the two commands can't drift on error messages or on which
workspace they end up resolving against.
"""

from __future__ import annotations

from typing import Literal, cast

import data_designer.config as dd
import typer
from data_designer.cli.ui import print_error
from data_designer.cli.utils.config_loader import ConfigLoadError, load_config_builder
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.cli_state import resolve_local_cli_sdks

OutputFormat = Literal["text", "json"]


def load_builder_or_exit(config_source: str) -> dd.DataDesignerConfigBuilder:
    """Load a config builder from ``config_source``, exiting 1 if it can't be read."""
    try:
        return load_config_builder(config_source)
    except ConfigLoadError as e:
        print_error(f"Could not load config: {e}")
        raise typer.Exit(code=1) from e


def resolve_sdks_or_exit(typer_ctx: typer.Context) -> tuple[NeMoPlatform | None, AsyncNeMoPlatform | None]:
    """Resolve the CLI's SDK pair, exiting 1 when neither is configured."""
    sdk, async_sdk = resolve_local_cli_sdks(typer_ctx)
    sdk = cast("NeMoPlatform | None", sdk)
    async_sdk = cast("AsyncNeMoPlatform | None", async_sdk)

    if sdk is None and async_sdk is None:
        print_error(
            "No NeMo Platform SDK is available. Run `nemo` from a configured environment "
            "or supply credentials via the top-level CLI."
        )
        raise typer.Exit(code=1)

    return sdk, async_sdk


def resolve_workspace(
    workspace: str | None,
    *,
    sdk: NeMoPlatform | None,
    async_sdk: AsyncNeMoPlatform | None,
) -> str:
    """Pick the workspace to resolve against: explicit flag, then SDK default, then ``"default"``."""
    return (
        workspace
        or (getattr(sdk, "workspace", None) if sdk is not None else None)
        or (getattr(async_sdk, "workspace", None) if async_sdk is not None else None)
        or "default"
    )
