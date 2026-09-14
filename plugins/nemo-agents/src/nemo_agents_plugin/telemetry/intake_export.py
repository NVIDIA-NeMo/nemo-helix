# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-wire an agent's Relay ATIF export to the platform's Intake ingest.

Filling in an export destination by hand is a poor thing to ask of anyone
writing an agent config: the reachable platform URL differs per deployment
context, and the same config should work whether it is deployed or run as a
job. So the backend wires it, and the config carries at most a name.

The two contexts differ only in how identity reaches Intake. A deployment
routes through a loopback auth-proxy sidecar that stamps the principal on the
way out. A job has one creator for its whole life and is handed that principal
directly, so it names environment variables the exporter reads instead.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import nemo_fabric as fabric
from nemo_agents_plugin.agent_config import TelemetryConfig
from pydantic import ValidationError

logger = logging.getLogger(__name__)

INTAKE_ATIF_INGEST_PATH = "/apis/intake/v2/workspaces/{workspace}/ingest/atif"


def supports_intake_atif_export(config: dict[str, Any], *, base_dir: Path | None = None) -> bool:
    """Whether the agent's adapter declares that Relay can export ATIF for it.

    Adapters advertise this in their descriptor's ``telemetry.providers``, and
    Fabric rejects a relay configuration outright for one that does not -- so
    wiring without asking turns "this agent cannot be traced" into "this agent
    cannot run". The ``atif`` output is checked too: an adapter may support
    Relay for OpenTelemetry alone, and ATIF is what this wires.

    Anything we cannot answer -- a config that will not validate, a plan that
    fails -- returns False. Those failures belong to whoever runs the agent
    next, reported with their own diagnostics; here they only mean we do not
    know enough to wire telemetry.
    """
    try:
        from nemo_agents_plugin.agent_config import AgentConfig
        from nemo_agents_plugin.fabric.translator import translate_agent_config

        agent_config = AgentConfig.model_validate(config)
        with _plan_directory(base_dir) as resolved_base:
            plan = fabric.Fabric().plan(translate_agent_config(agent_config), base_dir=resolved_base)
        descriptor = plan.to_dict().get("adapter_descriptor") or {}
        relay = descriptor.get("descriptor", descriptor).get("telemetry", {}).get("providers", {}).get("relay")
    except Exception:
        logger.warning("Could not read adapter telemetry support; the agent will run untraced.", exc_info=True)
        return False

    if relay is None:
        logger.info("Adapter does not support Relay telemetry; the agent will run untraced.")
        return False
    if "atif" not in (relay.get("outputs") or []):
        logger.info("Adapter supports Relay but not its ATIF output; the agent will run untraced.")
        return False
    return True


@contextmanager
def _plan_directory(base_dir: Path | None) -> Iterator[Path]:
    """Planning needs somewhere to resolve against; callers mid-run already have one."""
    if base_dir is not None:
        yield base_dir
        return
    with tempfile.TemporaryDirectory() as scratch:
        yield Path(scratch)


def configure_intake_atif_export(
    config: dict[str, Any],
    *,
    workspace: str,
    base_url: str,
    header_env: dict[str, str] | None = None,
) -> bool:
    """Point *config*'s ATIF export at *workspace*'s Intake ingest.

    Mutates *config* in place. Returns whether telemetry was wired.

    Takes the config as a mapping rather than an ``AgentConfig`` because the
    deployments path reaches this point with a config already resolved for its
    runtime, whose harnesses no longer round-trip through the model. The
    telemetry section does round-trip, so it is manipulated as a
    :class:`TelemetryConfig` rather than by poking at keys.

    ``telemetry.enabled`` is tri-state: unset means "wire it for me", ``False``
    is an explicit opt-out, and ``True`` turns it on while still letting the
    backend fill in anything the config left out. An agent that already names
    its own ATIF storage keeps it — an explicit destination beats an inferred
    one.

    Args:
        config: Agent config to wire, modified in place.
        workspace: Workspace whose Intake receives the trajectory.
        base_url: Platform URL reachable from wherever the agent will run.
        header_env: Header name to environment variable name, for contexts
            with no auth proxy to stamp identity. The variables must exist in
            the agent process; the values deliberately never enter the config,
            which is written into the run's artifacts.
    """
    telemetry = _telemetry_awaiting_destination(config)
    if telemetry is None:
        return False

    storage: dict[str, object] = {
        "type": "http",
        "endpoint": f"{base_url.rstrip('/')}{INTAKE_ATIF_INGEST_PATH.format(workspace=workspace)}",
    }
    if header_env:
        storage["header_env"] = dict(header_env)

    atif = dict(telemetry.atif or {})
    atif["enabled"] = True
    atif["storage"] = [storage]

    wired = telemetry.model_copy(
        update={
            "enabled": True,
            "provider": telemetry.provider or "relay",
            "agent_name": telemetry.agent_name or config.get("name"),
            "atif": atif,
        }
    )
    config["telemetry"] = wired.model_dump(exclude_none=True)
    return True


def wants_intake_atif_export(config: dict[str, Any]) -> bool:
    """Whether *config* leaves its ATIF destination for the backend to fill in.

    Reads the config's own ``telemetry`` section and nothing else, which makes
    this cheap enough to ask first. :func:`supports_intake_atif_export` resolves
    a Fabric plan to answer what the *adapter* can do; an agent that already
    opted out, or that named its own destination, should not pay for that.

    Answering True is not a promise that wiring will happen -- the adapter still
    has to support the export -- only that nothing in the config forbids it.
    """
    return _telemetry_awaiting_destination(config) is not None


def _telemetry_awaiting_destination(config: dict[str, Any]) -> TelemetryConfig | None:
    """Return the parsed telemetry when the backend should fill in a destination.

    ``None`` means leave the config alone, for any of the reasons that amount to
    "someone already decided": an unreadable section, an explicit opt-out, a
    declared ATIF destination, or ATIF declined while telemetry stays on.

    Single source of truth for that reading, so the predicate callers ask up
    front and the wiring itself cannot drift apart.
    """
    section = config.get("telemetry")
    try:
        telemetry = TelemetryConfig.model_validate(section if isinstance(section, dict) else {})
    except ValidationError as exc:
        # Leave a section we do not understand exactly as we found it. The jobs
        # path validates the whole config moments later and will report this
        # properly; deployments do not, and a telemetry key is no reason to
        # fail one.
        logger.warning("Leaving an unrecognized telemetry section unwired: %s", exc)
        return None

    if telemetry.enabled is False:
        return None
    if _declares_atif_storage(telemetry):
        return None
    if isinstance(telemetry.atif, dict) and telemetry.atif.get("enabled") is False:
        # Declining ATIF while leaving telemetry on is a real choice -- an agent
        # may want only OTel -- and overriding it would be the opposite of
        # preserving an explicit declaration.
        return None
    return telemetry


def _declares_atif_storage(telemetry: TelemetryConfig) -> bool:
    """Whether the config says where its trajectory goes.

    Only the ATIF block is consulted. Declaring an OpenTelemetry collector or an
    ATOF sink is an opinion about *that* output; leaving ATIF unset is no
    opinion about ATIF, so the backend's default applies -- the same tri-state
    reading as ``telemetry.enabled``. A config that wants no trajectory says so
    with ``atif.enabled: false``.
    """
    return isinstance(telemetry.atif, dict) and bool(telemetry.atif.get("storage"))
