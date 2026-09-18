# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared client-side plumbing for the config-inspection commands.

``validate`` and ``check-models`` ask different questions of the same
configuration, but they reach the engine along the same path: resolve the
config against the platform, then — only if that pass came back clean —
stand up a real ``DataDesigner`` against the resolved providers and call a
single method on it.

This module owns that common path so :mod:`nemo_data_designer_plugin.sdk.validation`
and :mod:`nemo_data_designer_plugin.sdk.check_models` differ only in the engine
method they invoke, the exceptions they treat as an answer rather than a crash,
and how they render the result.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

import data_designer.config as dd
from data_designer.interface.data_designer import DataDesigner
from data_designer_nemo.context.engine_protocol import DataDesignerEngineContext
from data_designer_nemo.context.execution import create_execution_context
from data_designer_nemo.context.validation import create_validation_context
from data_designer_nemo.errors import NDDError
from data_designer_nemo.runnable import resolve_runnable_config
from data_designer_nemo.sdk_translation import sync_to_async_sdk
from nemo_data_designer_plugin._data_designer import create_data_designer
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform

EngineCall = Callable[[DataDesigner, dd.DataDesignerConfigBuilder], None]

# Given the sync SDK (if any), the workspace, and the filesystem roots the
# validation pass cleared, produce the context to run the engine against —
# or ``None`` to skip the engine entirely.
EngineContextFactory = Callable[[NeMoPlatform | None, str, set[str]], DataDesignerEngineContext | None]


@dataclass(frozen=True)
class EnginePassResult:
    """Outcome of :func:`run_engine_pass`.

    ``engine_ran`` is what lets callers tell "the engine had nothing to
    complain about" apart from "the engine never got to look". The two are
    worlds apart for a liveness probe, where a skipped pass that reported
    success would be exactly the false green this command exists to prevent.
    """

    resolution_errors: list[NDDError]
    engine_error: Exception | None
    engine_ran: bool


async def run_engine_pass(
    config_builder: dd.DataDesignerConfigBuilder,
    *,
    sdk: NeMoPlatform | None = None,
    async_sdk: AsyncNeMoPlatform | None = None,
    workspace: str,
    engine_call: EngineCall,
    engine_errors: tuple[type[Exception], ...],
    engine_context_factory: EngineContextFactory | None = None,
) -> EnginePassResult:
    """Resolve ``config_builder`` against the platform, then run one engine check.

    Step one is :func:`resolve_runnable_config` — the same producer pass
    ``CreateJob.to_spec`` and ``PreviewFunction.run`` execute. It never raises;
    every problem lands in the returned error buffer.

    Step two runs ``engine_call`` against a ``DataDesigner`` built from the
    resolved providers, but only when step one found nothing and
    ``engine_context_factory`` produced a context. A config we already know is
    malformed makes the engine report internal failures (e.g. ``No reader found
    for seed_type 'df'``) instead of the actionable problem, so skipping keeps
    the diagnostics focused.

    Args:
        config_builder: The Data Designer config to inspect.
        sdk: Sync NeMoPlatform SDK. Required for the engine pass, and used to
            derive ``async_sdk`` when one is not supplied.
        async_sdk: Async NeMoPlatform SDK. Built from ``sdk`` when omitted.
        workspace: Workspace used to resolve provider references and seed
            sources. Pass ``"default"`` if you have no better value.
        engine_call: Invoked as ``engine_call(data_designer, config_builder)``
            on a worker thread. The engine's APIs are sync; running them off the
            event loop keeps it unblocked.
        engine_errors: Exception types to capture and return rather than
            propagate. Anything else is a genuine bug and is left to raise.
        engine_context_factory: Builds the context the engine runs against.
            Defaults to a real execution context, which requires a sync ``sdk``
            and yields no engine pass without one.

    Returns:
        An :class:`EnginePassResult`. Check ``engine_ran`` before reading a
        clean ``engine_error`` as a pass — the engine is skipped entirely when
        resolution failed or no sync ``sdk`` was supplied.

    Raises:
        ValueError: If neither ``sdk`` nor ``async_sdk`` is provided.
    """
    if async_sdk is None:
        if sdk is None:
            raise ValueError("run_engine_pass requires either sdk= or async_sdk=")
        async_sdk = sync_to_async_sdk(sdk)

    config = config_builder.build()

    validation_ctx = create_validation_context(async_sdk, workspace)
    resolution_errors, _model_configs, model_providers = await resolve_runnable_config(validation_ctx, config)

    if resolution_errors:
        return EnginePassResult(resolution_errors=resolution_errors, engine_error=None, engine_ran=False)

    factory = engine_context_factory or _execution_context_factory
    engine_ctx = factory(sdk, workspace, validation_ctx.validated_filesystem_roots)
    if engine_ctx is None:
        return EnginePassResult(resolution_errors=resolution_errors, engine_error=None, engine_ran=False)

    try:
        with tempfile.TemporaryDirectory() as artifact_path:
            data_designer = create_data_designer(
                artifact_path=artifact_path,
                model_providers=model_providers,
                dd_ctx=engine_ctx,
            )
            await asyncio.to_thread(engine_call, data_designer, config_builder)
    except engine_errors as e:
        return EnginePassResult(resolution_errors=resolution_errors, engine_error=e, engine_ran=True)

    return EnginePassResult(resolution_errors=resolution_errors, engine_error=None, engine_ran=True)


def _execution_context_factory(
    sdk: NeMoPlatform | None,
    workspace: str,
    validated_roots: set[str],
) -> DataDesignerEngineContext | None:
    """Default factory: a real execution context, which needs a sync SDK."""
    if sdk is None:
        return None
    return create_execution_context(sdk, workspace, validated_roots=validated_roots)
