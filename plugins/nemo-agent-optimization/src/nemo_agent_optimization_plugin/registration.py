# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist an optimization's result as a new agent entity plus its ethos fileset."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import httpx
import yaml
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.run_dependencies import LocalRunError

logger = logging.getLogger(__name__)


def register_optimized_agent(
    optimized: dict[str, Any],
    *,
    name: str,
    source_agent_config: dict[str, Any],
    workspace: str,
    sdk: NeMoPlatform,
) -> dict[str, Any]:
    """Create the optimized agent entity and upload its files.

    The entity is always stored as ``nemo-agents-spec-v1``.  The companion
    ``<name>-ethos`` fileset carries the optimized ``agent.yaml`` and the source
    agent's ``ETHOS.md``, matching what ``nemo agents create`` writes.
    """
    # Soft dependency: this plugin cannot declare nemo-agents-plugin (it would
    # cycle), so the package only ever arrives transitively.
    try:
        from nemo_agents_plugin.agent_config import AgentConfig
        from nemo_agents_plugin.entities import (
            AGENT_CONFIG_FILENAME,
            ETHOS_FILENAME,
            NEMO_AGENTS_SPEC_CONFIG_FORMAT,
            ethos_fileset_name,
        )
        from nemo_agents_plugin.jobs.fileset_io import upload_to_fileset
    except ImportError as exc:  # pragma: no cover - agents plugin always present for job path
        raise LocalRunError("Registering an optimized agent requires nemo-agents-plugin.") from exc

    payload = {**optimized, "name": name, "config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT}
    try:
        AgentConfig.model_validate(payload)
    except Exception as exc:
        raise LocalRunError(
            f"The optimization produced a config that is not a valid nemo-agents-spec-v1 agent: {exc}"
        ) from exc

    try:
        sdk.agents.create(
            name=name,
            config=payload,
            description=str(payload.get("description", "")),
            config_format=NEMO_AGENTS_SPEC_CONFIG_FORMAT,
            workspace=workspace,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 409:
            raise LocalRunError(
                f"An agent named {name!r} already exists in workspace {workspace!r}. "
                "Agents are not versioned, and overwriting one that may be deployed is not "
                "safe, so pick another --output-agent name."
            ) from exc
        raise

    fileset = ethos_fileset_name(name)
    try:
        with tempfile.TemporaryDirectory(prefix=f".ethos-{name}-") as staging:
            staged = Path(staging)
            (staged / AGENT_CONFIG_FILENAME).write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            ethos = source_agent_config.get("_ethos_markdown")
            if isinstance(ethos, str) and ethos.strip():
                (staged / ETHOS_FILENAME).write_text(ethos, encoding="utf-8")
            upload_to_fileset(staged, fileset=fileset, workspace=workspace, sdk=sdk)
    except Exception as exc:
        try:
            sdk.agents.delete(name, workspace=workspace)
        except Exception:
            logger.exception("Failed to roll back agent %r after fileset upload failure", name)
            raise LocalRunError(
                f"Failed to upload the ethos fileset for {name!r} and could not roll the agent "
                f"back; it may still exist. Remove it with `nemo agents delete {name}`. Cause: {exc}"
            ) from exc
        raise LocalRunError(
            f"Failed to upload the ethos fileset for {name!r}; the agent was rolled back. Cause: {exc}"
        ) from exc

    logger.info("Registered optimized agent %s/%s", workspace, name)
    return {"agent": f"{workspace}/{name}"}
