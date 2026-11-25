# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric adapter for running Compass without persistence side effects."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nemo_compass_plugin.analysis.run import run_compass_change_set
from nemo_compass_plugin.analyst import CompassAnalyst
from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common.utils import (
    reject_ambient_relay_plugin_config,
    reject_inherited_relay_plugin_config,
)
from nemo_helix_plugin.agents.plugin_managed_agents import NEMO_HELIX_CONTEXT_KEY
from nemo_helix_plugin.client_provider import get_async_task_nemo_client
from nemo_helix_plugin.nooa_model_client import ConfiguredModelRefs
from nemo_helix_plugin.tasks.logging_setup import configure_task_logging
from nemo_relay.plugin import activate as activate_relay_plugin
from pydantic import ValidationError

logger = logging.getLogger(__name__)

COMPASS_RELAY_SCOPE = "compass"


class CompassAdapterConfigError(ValueError):
    """The Fabric-projected Compass adapter configuration is invalid."""


class CompassRequestContextError(ValueError):
    """The request context is not a valid Compass run context."""


class CompassRuntime:
    """Adapter-owned runtime for one Fabric-managed Compass process."""

    async def start(self, payload: dict[str, Any]) -> None:
        del payload

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
            logger.exception("Compass run failed.")
            return contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output={"response": str(error)},
                error=contract.AgentRunError(
                    code="compass_failed",
                    message=str(error),
                    retryable=False,
                ),
            )

        return contract.AgentRunResult(
            status=contract.AgentRunStatus.SUCCEEDED,
            output={
                "response": result.summary,
                "compass_result": result.model_dump(mode="json"),
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
            reject_ambient_relay_plugin_config()
            async with activate_relay_plugin(_relay_plugin_config(telemetry)) as activation:
                reject_inherited_relay_plugin_config(activation.report)
                return await self._run_analysis(request, relay_scope_name=COMPASS_RELAY_SCOPE)

    async def _run_analysis(self, request: contract.AgentRunRequest, *, relay_scope_name: str | None = None):
        # The workspace is the job's own: Compass analyzes the workspace it runs in.
        workspace = _nemo_helix_context(request.context, "job_workspace")
        if workspace is None:
            raise CompassRequestContextError(
                f"context.{NEMO_HELIX_CONTEXT_KEY}.job_workspace is required; run Compass as an execute job"
            )
        # Parse before opening the client: nothing is worth a live SDK handle that no one closes.
        try:
            run = CompassAnalyst.parse_context(request.context)
        except ValidationError as error:
            raise CompassRequestContextError(f"the request context is not a Compass run context: {error}") from error

        # Compass's self-observability builds its own OTLP exporter from
        # the operator's local CLI config, which does not exist in a task
        # pod, so on a cluster it exports unauthenticated at best. Telemetry
        # for these runs belongs in the agent config's relay endpoints,
        # where the agents layer already carries auth; until that is wired,
        # a run must not depend on it.
        enable_observability = False

        base_url = os.environ.get("NHX_BASE_URL") or os.environ.get("NEMO_BASE_URL")

        async with get_async_task_nemo_client("compass") as client:
            result, _backend = await run_compass_change_set(
                agent=run.agent,
                ethos=run.ethos,
                workspace=workspace,
                base_url=base_url,
                client=client,
                since=run.since,
                evaluation_id=run.evaluation_id,
                enable_observability=enable_observability,
                relay_scope_name=relay_scope_name,
                model_refs=ConfiguredModelRefs(default=run.default_model, fast=run.fast_model),
            )
        return result

    async def stop(self) -> None:
        return None


@contextmanager
def _applied_environment(env: dict[str, str]) -> Generator[None, None, None]:
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
        raise CompassAdapterConfigError("Relay is enabled but Fabric supplied no config path")
    wrapper = json.loads(Path(telemetry.config_path).read_text(encoding="utf-8"))
    config = (wrapper.get("relay") or {}).get("config") or {}
    if not config.get("components"):
        raise CompassAdapterConfigError(f"Relay config at {telemetry.config_path} declares no components")
    return config


def _nemo_helix_context(context: dict[str, Any], key: str) -> str | None:
    platform = context.get(NEMO_HELIX_CONTEXT_KEY)
    value = platform.get(key) if isinstance(platform, dict) else None
    return value if isinstance(value, str) and value.strip() else None


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""
    # Fabric spawns this as its own process and redirects its streams to the
    # run's stdout/stderr artifacts. Nothing configures logging here, so
    # without this Compass's and Nooa's INFO output is dropped and those
    # artifacts hold only bare WARNING+ lines from ``logging.lastResort``.
    configure_task_logging()
    lifecycle.serve(CompassRuntime, config_loader=contract.AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
