# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric adapter for running the Insights analyst without persistence side effects."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import lifecycle
from nemo_insights_plugin.analyst.run import run_analyst_change_set
from nemo_platform_plugin.nooa_model_client import ConfiguredModelRefs
from nemo_platform_plugin.sdk_provider import get_async_task_sdk
from nemo_platform_plugin.tasks.logging_setup import configure_task_logging
from nemo_relay import plugin as relay_plugin

logger = logging.getLogger(__name__)

ANALYST_RELAY_SCOPE = "insights-analyst"


class AnalystAdapterConfigError(ValueError):
    """The Fabric-projected analyst adapter configuration is invalid."""


class InsightsAnalystRuntime:
    """Adapter-owned runtime for one Fabric-managed analyst process."""

    def __init__(self) -> None:
        self._settings: dict[str, Any] = {}
        self._models: dict[str, contract.AgentModelConfig] = {}

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
        self._settings = dict(config.harness.settings if config.harness else {})
        self._models = dict(config.models)

    async def invoke(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        try:
            result = await self._run_with_telemetry(request, context)
        except Exception as error:
            # Log the full exception so it reaches this process's stderr, which
            # is where the job reads a failed run's diagnostics from.
            logger.exception("Insights analyst run failed.")
            return contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output={"response": str(error)},
                error=contract.AgentRunError(
                    code="insights_analyst_failed",
                    message=str(error),
                    retryable=False,
                ),
            )

        return contract.AgentRunResult(
            status=contract.AgentRunStatus.SUCCEEDED,
            output={
                "response": result.summary,
                "analyst_result": result.model_dump(mode="json"),
            },
        )

    async def _run_with_telemetry(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ):
        """Run the analysis, instrumented by Relay when Fabric asked for it.

        Fabric resolves the whole export -- endpoint, credentials, agent name --
        into a config file and points at it through ``telemetry.env``. Nothing
        about the destination is decided here; the adapter's job is to activate
        the config and let Relay carry the trajectory.
        """
        telemetry = context.telemetry
        if telemetry is None or not telemetry.relay_enabled:
            return await self._run_analysis(request)

        # Relay reads credentials for the export from the environment Fabric
        # names, so apply it before the exporter is built -- and only for this
        # invocation. The runtime is long-lived and serves many; a leftover
        # FABRIC_RELAY_CONFIG_PATH is the ambient-config hazard the bundled
        # adapters have a named guard against.
        with _applied_environment(telemetry.env):
            async with relay_plugin.plugin(_relay_plugin_config(telemetry)):
                return await self._run_analysis(request, relay_scope_name=ANALYST_RELAY_SCOPE)

    async def _run_analysis(self, request: contract.AgentRunRequest, *, relay_scope_name: str | None = None):
        target_agent = _string_setting(self._settings, "agent") or _string_setting(self._settings, "target_agent")
        if target_agent is None:
            raise AnalystAdapterConfigError("harness.settings.agent is required for the Insights analyst adapter")

        workspace = (
            _string_setting(self._settings, "workspace")
            or _string_context(request.context, "job_workspace")
            or os.environ.get("NMP_WORKSPACE")
            or "default"
        )
        base_url = (
            _string_setting(self._settings, "base_url")
            or os.environ.get("NMP_BASE_URL")
            or os.environ.get("NEMO_BASE_URL")
        )
        # Every settings read can raise, so resolve them before opening the
        # client: nothing is worth a live SDK handle that no one closes.
        ethos = _string_setting(self._settings, "ethos")
        since = _datetime_setting(self._settings, "since")
        evaluation_id = _string_setting(self._settings, "evaluation_id")
        enable_observability = bool(self._settings.get("enable_observability", True))
        model_refs = ConfiguredModelRefs(
            default=_default_model_ref(self._settings, self._models),
            fast=_fast_model_ref(self._settings, self._models),
        )
        async with get_async_task_sdk("insights") as client:
            result, _backend = await run_analyst_change_set(
                agent=target_agent,
                ethos=ethos,
                workspace=workspace,
                base_url=base_url,
                client=client,
                since=since,
                evaluation_id=evaluation_id,
                enable_observability=enable_observability,
                relay_scope_name=relay_scope_name,
                model_refs=model_refs,
            )
        return result

    async def stop(self) -> None:
        self.__init__()


@contextmanager
def _applied_environment(env: dict[str, str]) -> Iterator[None]:
    """Apply *env* for the duration of one invocation, then put it back."""
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _relay_plugin_config(telemetry: contract.RuntimeTelemetryContext) -> dict[str, Any]:
    """Read the Relay plugin config Fabric resolved for this invocation.

    ``nemo_fabric_adapters.common.load_relay_plugin_config`` does the same from
    a raw invocation payload, which a lifecycle adapter never sees -- it is
    handed the typed context instead, and ``config_path`` points at the same
    file. The helper additionally rebases ATOF file-sink directories, which
    matters only for configs this path does not produce: the agents plugin
    wires ATIF over HTTP.
    """
    if not telemetry.config_path:
        raise AnalystAdapterConfigError("Relay is enabled but Fabric supplied no config path")
    wrapper = json.loads(Path(telemetry.config_path).read_text(encoding="utf-8"))
    config = (wrapper.get("relay") or {}).get("config") or {}
    if not config.get("components"):
        raise AnalystAdapterConfigError(f"Relay config at {telemetry.config_path} declares no components")
    return config


def _string_setting(settings: dict[str, Any], key: str) -> str | None:
    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AnalystAdapterConfigError(f"harness.settings.{key} must be a non-empty string")
    return value


def _string_context(context: dict[str, Any], key: str) -> str | None:
    value = context.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _datetime_setting(settings: dict[str, Any], key: str) -> datetime | None:
    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AnalystAdapterConfigError(f"harness.settings.{key} must be an ISO-8601 datetime string")
    return datetime.fromisoformat(value)


def _default_model_ref(
    settings: dict[str, Any],
    models: dict[str, contract.AgentModelConfig],
) -> str:
    configured = _string_setting(settings, "default_model")
    if configured is not None:
        return configured
    model = models.get("default")
    if model is None:
        raise AnalystAdapterConfigError("models.default or harness.settings.default_model is required")
    return model.model


def _fast_model_ref(
    settings: dict[str, Any],
    models: dict[str, contract.AgentModelConfig],
) -> str:
    configured = _string_setting(settings, "fast_model")
    if configured is not None:
        return configured
    model = models.get("fast")
    if model is not None:
        return model.model
    return _default_model_ref(settings, models)


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""
    # Fabric spawns this as its own process and redirects its streams to the
    # run's stdout/stderr artifacts. Nothing configures logging here, so
    # without this the analyst's and Nooa's INFO output is dropped and those
    # artifacts hold only bare WARNING+ lines from ``logging.lastResort``.
    configure_task_logging()
    lifecycle.serve(InsightsAnalystRuntime, config_loader=contract.AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
