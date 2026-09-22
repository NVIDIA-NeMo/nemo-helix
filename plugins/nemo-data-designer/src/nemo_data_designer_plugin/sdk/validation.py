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

Validation is deliberately limited to *internal* readiness: the config's own
structure, and whether the platform resources it names resolve. It does not
probe whether those resources respond, so a provider can resolve while still
refusing to serve the model named alongside it. That external-readiness
question belongs to :mod:`nemo_data_designer_plugin.sdk.check_models`, which
mirrors upstream's split between ``validate`` and ``check_models``.
"""

from __future__ import annotations

import asyncio

import data_designer.config as dd
from data_designer.config.errors import InvalidConfigError
from data_designer.interface.data_designer import DataDesigner
from data_designer_nemo.errors import NDDInternalError, NDDInvalidConfigError
from nemo_data_designer_plugin.sdk._engine_pass import run_engine_pass
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from pydantic import BaseModel, Field, computed_field

# The engine's compile step is the source of truth on column→alias→provider
# consistency; these are the ways it reports a config it cannot compile.
_ENGINE_ERRORS = (InvalidConfigError, NDDInvalidConfigError, NDDInternalError)


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


def _validate(data_designer: DataDesigner, config_builder: dd.DataDesignerConfigBuilder) -> None:
    data_designer.validate(config_builder)


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
    available. Never short-circuits within a pass: every sub-check that *can*
    be run is run, so a single invocation surfaces every problem it detects.

    Does not check whether the referenced models respond; see
    :func:`nemo_data_designer_plugin.sdk.check_models.check_models_config`.

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
    result = await run_engine_pass(
        config_builder,
        sdk=sdk,
        async_sdk=async_sdk,
        workspace=workspace,
        engine_call=_validate,
        engine_errors=_ENGINE_ERRORS,
    )

    # A skipped engine pass is not an error here: the resolution pass already
    # answers most of what validate promises, so an async-only caller gets a
    # narrower check rather than a failure.
    errors = [_to_validation_error(e) for e in result.resolution_errors]
    if result.engine_error is not None:
        errors.append(_to_validation_error(result.engine_error))

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
