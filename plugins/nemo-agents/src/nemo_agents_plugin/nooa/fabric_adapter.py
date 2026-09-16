# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric adapter for NOOA agents written to the platform's calling convention.

NOOA has no standardized entrypoint, so Fabric ships no generic NOOA adapter and
is right not to. This is not Fabric standardizing NOOA; it is the *platform*
declaring a calling convention over it: export one async callable taking a
:class:`~nemo_platform_plugin.agents.nooa_contract.NooaInvocation`, name it in
``harness.settings.entrypoint``, and this runtime imports and invokes it.

The runtime owns only what is universal — ``lifecycle.serve`` plumbing, Relay
telemetry activation, importing and invoking the entrypoint, and mapping
exceptions onto an ``AgentRunResult``. It deliberately does not resolve platform
model clients; see the ``models`` field on ``NooaInvocation`` for why.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract import models as contract
from nemo_fabric_adapters.common import lifecycle
from nemo_platform_plugin.agents.nooa_contract import NooaInvocation
from nemo_platform_plugin.tasks.logging_setup import configure_task_logging
from nemo_relay import plugin as relay_plugin

logger = logging.getLogger(__name__)

ENTRYPOINT_SETTING = "entrypoint"

# What a platform NOOA agent exports. One async callable, one argument.
NooaEntrypoint = Callable[[NooaInvocation], Awaitable[contract.AgentRunResult]]


class PlatformNooaAdapterConfigError(ValueError):
    """The platform NOOA adapter configuration is invalid."""


class PlatformNooaRuntime:
    """Adapter-owned runtime for one Fabric-managed platform NOOA agent process."""

    def __init__(self) -> None:
        self._settings: dict[str, Any] = {}
        self._models: dict[str, contract.AgentModelConfig] = {}

    async def start(self, payload: dict[str, Any]) -> None:
        config: contract.AgentConfig = payload["config"]
        self._settings = dict(config.harness.settings if config.harness else {})
        self._models = dict(config.models)
        # Resolve the entrypoint at start, not at first invoke: a typo in
        # `entrypoint` should fail the runtime immediately rather than after
        # Fabric has reported the agent healthy.
        _load_entrypoint(self._entrypoint_spec())

    async def invoke(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        try:
            return await self._run_with_telemetry(request, context)
        except Exception as error:
            # Log the full exception so it reaches this process's stderr, which
            # is where the job reads a failed run's diagnostics from.
            logger.exception("Platform NOOA agent run failed.")
            return contract.AgentRunResult(
                status=contract.AgentRunStatus.FAILED,
                output={"response": str(error)},
                error=contract.AgentRunError(
                    code="platform_nooa_agent_failed",
                    message=str(error),
                    retryable=False,
                ),
            )

    async def _run_with_telemetry(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        """Invoke the entrypoint, instrumented by Relay when Fabric asked for it.

        Fabric resolves the whole export -- endpoint, credentials, agent name --
        into a config file and points at it through ``telemetry.env``. Nothing
        about the destination is decided here; the adapter's job is to activate
        the config and let Relay carry the trajectory.
        """
        telemetry = context.telemetry
        if telemetry is None or not telemetry.relay_enabled:
            return await self._invoke_entrypoint(request, context)

        # Relay reads credentials for the export from the environment Fabric
        # names, so apply it before the exporter is built -- and only for this
        # invocation. The runtime is long-lived and serves many; a leftover
        # FABRIC_RELAY_CONFIG_PATH is the ambient-config hazard the bundled
        # adapters have a named guard against.
        with _applied_environment(telemetry.env):
            async with relay_plugin.plugin(_relay_plugin_config(telemetry)):
                return await self._invoke_entrypoint(request, context)

    async def _invoke_entrypoint(
        self,
        request: contract.AgentRunRequest,
        context: contract.RuntimeContext,
    ) -> contract.AgentRunResult:
        entrypoint = _load_entrypoint(self._entrypoint_spec())
        result = await entrypoint(
            NooaInvocation(
                request=request,
                context=context,
                settings=dict(self._settings),
                models=dict(self._models),
            )
        )
        if not isinstance(result, contract.AgentRunResult):
            raise PlatformNooaAdapterConfigError(
                f"{self._entrypoint_spec()!r} returned {type(result).__name__}, expected an AgentRunResult"
            )
        return result

    def _entrypoint_spec(self) -> str:
        value = self._settings.get(ENTRYPOINT_SETTING)
        if not isinstance(value, str) or not value.strip():
            raise PlatformNooaAdapterConfigError(
                f"harness.settings.{ENTRYPOINT_SETTING} is required and must be a non-empty string, "
                'formatted as "module.path:callable"'
            )
        return value.strip()

    async def stop(self) -> None:
        self.__init__()


def _load_entrypoint(spec: str) -> NooaEntrypoint:
    """Import ``module.path:callable`` from the running environment."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise PlatformNooaAdapterConfigError(
            f'harness.settings.{ENTRYPOINT_SETTING} must be formatted as "module.path:callable"; got {spec!r}'
        )
    try:
        module = importlib.import_module(module_name.strip())
    except ImportError as error:
        # The overwhelmingly common cause is that the agent's own project was
        # never installed into the image -- i.e. `nemo agents package` ran
        # without --pyproject. Say so rather than surfacing a bare ImportError.
        raise PlatformNooaAdapterConfigError(
            f"Could not import {module_name.strip()!r} for "
            f"harness.settings.{ENTRYPOINT_SETTING}={spec!r}: {error}. "
            "The module providing it must be installed in the environment running this agent; "
            "package the agent with `nemo agents package --pyproject pyproject.toml` so the "
            "project is installed into the image."
        ) from error
    try:
        entrypoint = getattr(module, attribute.strip())
    except AttributeError as error:
        raise PlatformNooaAdapterConfigError(
            f"Module {module_name.strip()!r} has no attribute {attribute.strip()!r} "
            f"for harness.settings.{ENTRYPOINT_SETTING}={spec!r}"
        ) from error
    if not callable(entrypoint):
        raise PlatformNooaAdapterConfigError(f"{spec!r} resolved to {type(entrypoint).__name__}, which is not callable")
    return entrypoint


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
        raise PlatformNooaAdapterConfigError("Relay is enabled but Fabric supplied no config path")
    wrapper = json.loads(Path(telemetry.config_path).read_text(encoding="utf-8"))
    config = (wrapper.get("relay") or {}).get("config") or {}
    if not config.get("components"):
        raise PlatformNooaAdapterConfigError(f"Relay config at {telemetry.config_path} declares no components")
    return config


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""
    # Fabric spawns this as its own process and redirects its streams to the
    # run's stdout/stderr artifacts. Nothing configures logging here, so
    # without this the agent's and Nooa's INFO output is dropped and those
    # artifacts hold only bare WARNING+ lines from ``logging.lastResort``.
    configure_task_logging()
    lifecycle.serve(PlatformNooaRuntime, config_loader=contract.AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
