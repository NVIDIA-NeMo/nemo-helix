# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side model health checks for Data Designer configs.

This module owns the "are the providers this config depends on actually
responsive?" question. It is consumed by both the SDK
(``DataDesignerResource.check_models``) and the CLI
(``nemo data-designer check-models``); the surface layer just decides how to
render the report and which exit code to use.

It is the external-readiness counterpart to
:mod:`nemo_data_designer_plugin.sdk.validation`: ``validate`` answers "is my
configuration well-formed, and do the platform resources it names resolve?",
while this answers "does a real generation against each referenced model
actually come back?". Neither subsumes the other — a config can resolve
perfectly and still name a model the provider will not serve, which is the
gap a provider's advertised model list cannot close and only a live probe can.

The probe itself is upstream's ``DataDesigner.check_models``, which is the same
code path the dataset builder runs at the start of ``preview`` and ``create``.
Delegating wholesale — rather than collecting aliases and probing them
ourselves — is what keeps this from drifting away from that startup gate.

One consequence of that delegation: upstream probes aliases serially and raises
on the first failure, so this reports at most one model error, unlike
``validate``, which surfaces every problem in a single pass. Forwarding the
engine's logs (see ``on_log``) is what makes that tolerable, since the log
records name each alias as it is probed.
"""

from __future__ import annotations

import asyncio

import data_designer.config as dd
from data_designer.errors import DataDesignerError
from data_designer.interface.data_designer import DataDesigner
from data_designer_nemo.context.check_models import create_check_models_context
from data_designer_nemo.context.engine_protocol import DataDesignerEngineContext
from data_designer_nemo.context.execution import create_execution_context
from nemo_data_designer_plugin.sdk._engine_logs import LogCallback, forward_engine_logs
from nemo_data_designer_plugin.sdk._engine_pass import run_engine_pass
from nemo_data_designer_plugin.sdk.logging import ensure_library_logging_handler
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from pydantic import BaseModel, Field, computed_field

# Mirrors what ``PreviewFunction.run`` treats as a failed health check.
# ``DataDesignerError`` is the base of the typed model errors
# (``ModelNotFoundError``, ``ModelAuthenticationError``, ...); ``TimeoutError``
# is raised when the engine's 180s health-check budget is exhausted.
_ENGINE_ERRORS = (DataDesignerError, TimeoutError)


class ModelCheckError(BaseModel):
    """A single problem surfaced while checking models."""

    error_type: str
    message: str


class CheckModelsReport(BaseModel):
    """Top-level model health check result."""

    config_source: str | None = None
    errors: list[ModelCheckError] = Field(default_factory=list)

    @computed_field
    @property
    def ok(self) -> bool:
        return not self.errors


def _to_model_check_error(exc: Exception) -> ModelCheckError:
    return ModelCheckError(error_type=type(exc).__name__, message=str(exc))


def _seed_type(config_builder: dd.DataDesignerConfigBuilder) -> str | None:
    seed_config = config_builder.get_seed_config()
    return None if seed_config is None else seed_config.source.seed_type


def _make_engine_context_factory(config_builder: dd.DataDesignerConfigBuilder):
    """Prefer a real execution context; fall back to a probe-only one.

    A sync SDK gives us the real thing, so the sync path behaves exactly as it
    would for a preview. An async-only caller cannot build one, and a probe
    does not need it — see
    :mod:`data_designer_nemo.context.check_models`.
    """

    def factory(
        sdk: NeMoPlatform | None,
        workspace: str,
        validated_roots: set[str],
    ) -> DataDesignerEngineContext | None:
        if sdk is not None:
            return create_execution_context(sdk, workspace, validated_roots=validated_roots)
        return create_check_models_context(seed_type=_seed_type(config_builder))

    return factory


def _check_models(data_designer: DataDesigner, config_builder: dd.DataDesignerConfigBuilder) -> None:
    # Upstream's default is a single attempt with no backoff. We deliberately
    # don't expose ``max_attempts`` / ``retry_backoff_seconds``, matching the
    # upstream CLI, which exposes neither.
    data_designer.check_models(config_builder)


async def check_models_config(
    config_builder: dd.DataDesignerConfigBuilder,
    *,
    sdk: NeMoPlatform | None = None,
    async_sdk: AsyncNeMoPlatform | None = None,
    workspace: str,
    config_source: str | None = None,
    on_log: LogCallback | None = None,
) -> CheckModelsReport:
    """Probe every model referenced by ``config_builder``.

    Resolves the config against the platform first — the same pass ``validate``
    runs — and only probes when that comes back clean, since an unresolvable
    provider cannot be probed in any meaningful way. Resolution problems are
    reported here too, so ``check-models`` is useful on its own and does not
    require the user to have run ``validate`` first.

    Requests route through the Inference Gateway using the caller's own
    credentials, so no provider API key is needed on the client.

    Args:
        config_builder: The Data Designer config whose model aliases are probed.
        sdk: Sync NeMoPlatform SDK. Used for the engine context when present.
        async_sdk: Async NeMoPlatform SDK. Derived from ``sdk`` when omitted.
            An async-only caller still probes, via a probe-only engine context.
        workspace: Workspace used to resolve provider references and seed
            sources. Pass ``"default"`` if you have no better value.
        config_source: Informational identifier echoed back through the report.
            Not used for any logic.
        on_log: Optional sink for the engine's per-alias log records, for
            callers that render logs themselves. When omitted, those records go
            to the SDK's stream handler unless logging is already configured.

    Returns:
        A ``CheckModelsReport``.

    Raises:
        ValueError: If neither ``sdk`` nor ``async_sdk`` is provided.
    """
    # The engine names each alias as it probes it, and that is the only place
    # that identity appears. A caller supplying ``on_log`` renders those records
    # itself; everyone else gets them through the SDK's usual stream handler.
    log_ctx = forward_engine_logs(on_log) if on_log is not None else ensure_library_logging_handler()

    with log_ctx:
        result = await run_engine_pass(
            config_builder,
            sdk=sdk,
            async_sdk=async_sdk,
            workspace=workspace,
            engine_call=_check_models,
            engine_errors=_ENGINE_ERRORS,
            engine_context_factory=_make_engine_context_factory(config_builder),
        )

    errors = [_to_model_check_error(e) for e in result.resolution_errors]
    if result.engine_error is not None:
        errors.append(_to_model_check_error(result.engine_error))

    # Unlike validate, this command has nothing left to report if the engine
    # never ran — every model is unprobed. Saying so explicitly keeps a skipped
    # probe from reading as a clean bill of health, which is the exact failure
    # mode this command exists to close.
    if not result.engine_ran and not errors:  # pragma: no cover - safety net
        errors.append(
            ModelCheckError(
                error_type="ModelsNotProbed",
                message=(
                    "No models were probed: the health check needs a sync NeMo Platform SDK and none was supplied."
                ),
            )
        )

    return CheckModelsReport(config_source=config_source, errors=errors)


def check_models_config_sync(
    config_builder: dd.DataDesignerConfigBuilder,
    *,
    sdk: NeMoPlatform | None = None,
    async_sdk: AsyncNeMoPlatform | None = None,
    workspace: str,
    config_source: str | None = None,
    on_log: LogCallback | None = None,
) -> CheckModelsReport:
    """Sync wrapper for :func:`check_models_config` for callers without an event loop."""
    return asyncio.run(
        check_models_config(
            config_builder,
            sdk=sdk,
            async_sdk=async_sdk,
            workspace=workspace,
            config_source=config_source,
            on_log=on_log,
        )
    )
