# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side validation core for Data Designer configs.

This module owns the "is this config fit to run locally / submit remotely?"
question. It is consumed by both the SDK (``DataDesignerResource.validate``)
and the CLI (``nemo data-designer validate``); the surface layer just decides
how to render the report and which exit code to use.

The validation passes here mirror what ``CreateJob.to_spec`` and
``PreviewFunction.run`` execute at submit/preview time, so a green
``ValidationReport`` is a strong (but not absolute) indicator that downstream
calls will succeed. The remote pass is a client-side simulation — it does not
hit the data-designer service.
"""

from __future__ import annotations

import asyncio
import tempfile

import data_designer.config as dd
from data_designer.config.errors import InvalidConfigError
from data_designer_nemo.context import create_execution_context, create_validation_context
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from data_designer_nemo.runnable import resolve_runnable_config
from data_designer_nemo.sdk_translation import sync_to_async_sdk
from nemo_data_designer_plugin._data_designer import create_data_designer
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from pydantic import BaseModel, Field, computed_field


class ValidationError(BaseModel):
    """A single problem surfaced by a validation pass."""

    message: str


class ValidationReport(BaseModel):
    """Top-level validation result."""

    config_source: str | None = None
    errors: list[ValidationError] = Field(default_factory=list)

    @computed_field
    @property
    def ok(self) -> bool:
        return not self.errors


def _to_validation_error(exc: Exception) -> ValidationError:
    return ValidationError(message=str(exc))


async def validate_config(
    config_builder: dd.DataDesignerConfigBuilder,
    *,
    sdk: NeMoPlatform | None = None,
    async_sdk: AsyncNeMoPlatform | None = None,
    workspace: str,
    config_source: str | None = None,
) -> ValidationReport:
    """Validate the provided ``config_builder``.

    Mirrors the work ``CreateJob.to_spec`` and ``PreviewFunction.run`` do at
    submit/preview time — via the shared :func:`resolve_runnable_config` —
    and additionally runs an engine-level compile check when a sync SDK is
    available. Never short-circuits: every sub-check that *can* be run is run,
    so a single pass surfaces every problem.

    Args:
        config_builder: The Data Designer config to validate.
        sdk: Sync NeMoPlatform SDK. Used for engine-level compile validation
            and as a fallback to derive ``async_sdk`` when one is not supplied.
        async_sdk: Async NeMoPlatform SDK. If omitted but ``sdk`` is supplied,
            an async wrapper is built via ``sync_to_async_sdk``.
        workspace: Workspace used to resolve provider references and seed
            sources for the remote context. Pass ``"default"`` if you have
            no better value.
        config_source: Informational identifier for the config source — echoed
            back through the report. Not used for any logic.

    Returns:
        A ``ValidationReport``.

    Raises:
        ValueError: If neither ``sdk`` nor ``async_sdk`` is provided.
    """
    if async_sdk is None:
        if sdk is None:
            raise ValueError("validate_config requires either sdk= or async_sdk=")
        async_sdk = sync_to_async_sdk(sdk)

    config = config_builder.build()

    validation_ctx = create_validation_context(async_sdk, workspace)

    # First run the same resolution that the job and function execute.
    runnable_errors, _model_configs, model_providers = await resolve_runnable_config(validation_ctx, config)
    errors: list[ValidationError] = [_to_validation_error(e) for e in runnable_errors]

    # Next additionally run engine-level compile via upstream ``DataDesigner.validate``,
    # the source of truth on column→alias→provider consistency.
    #
    # If we've already collected errors above (e.g. unsupported seed type,
    # unrecognized aliases, unresolved providers), the engine pass is
    # unlikely to succeed in a diagnostically useful way — the engine
    # surfaces internal failures (e.g. ``No reader found for seed_type
    # 'df'``) when the config is already known to be malformed. Skip the
    # engine check in that case so the user-facing diagnostics stay focused
    # on the actionable problems we've already found.
    if not errors and sdk is not None:
        execution_ctx = create_execution_context(
            sdk,
            workspace,
            validated_roots=validation_ctx.validated_filesystem_roots,
        )
        try:
            with tempfile.TemporaryDirectory() as artifact_path:
                data_designer = create_data_designer(
                    artifact_path=artifact_path,
                    model_providers=model_providers,
                    dd_ctx=execution_ctx,
                )
                # Engine validate is sync; run it on a worker thread so the
                # event loop stays unblocked.
                await asyncio.to_thread(data_designer.validate, config_builder)
        except (InvalidConfigError, NDDInvalidConfigError, NDDInternalError) as e:
            errors.append(_to_validation_error(e))

    return ValidationReport(config_source=config_source, errors=errors)


def validate_config_sync(
    config_builder: dd.DataDesignerConfigBuilder,
    *,
    sdk: NeMoPlatform | None = None,
    async_sdk: AsyncNeMoPlatform | None = None,
    workspace: str,
    config_source: str | None = None,
) -> ValidationReport:
    """Sync wrapper for :func:`validate_config` for SDK callers without an event loop."""
    return asyncio.run(
        validate_config(
            config_builder,
            sdk=sdk,
            async_sdk=async_sdk,
            workspace=workspace,
            config_source=config_source,
        )
    )
