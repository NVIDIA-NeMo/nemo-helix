# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist an optimization's result as a new agent entity plus its ethos fileset."""

from __future__ import annotations

import logging
import shutil
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
    source_agent: str,
    source_workspace: str,
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
            (staged / AGENT_CONFIG_FILENAME).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            _stage_source_ethos(staged, source_agent=source_agent, source_workspace=source_workspace, sdk=sdk)
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


def _stage_source_ethos(
    staged: Path,
    *,
    source_agent: str,
    source_workspace: str,
    sdk: NeMoPlatform,
) -> None:
    """Copy the source agent's ``ETHOS.md`` in beside the optimized ``agent.yaml``.

    The optimized agent is a new entity with its own ``<name>-ethos`` fileset, and the
    tuned ``agent.yaml`` belongs next to the ethos it was built from — an optimized
    agent whose "why" is missing reads as if it never had one.

    A source agent with no ethos fileset, or one that holds no ``ETHOS.md``, is a clean
    skip: plenty of agents are registered straight from an ``agent.yaml``, and the
    optimized agent is complete without it.
    """
    from nemo_agents_plugin.entities import ETHOS_FILENAME, ethos_fileset_name

    source_fileset = ethos_fileset_name(source_agent)
    with tempfile.TemporaryDirectory(prefix=f".source-ethos-{source_agent}-") as download_dir:
        downloaded = Path(download_dir)
        try:
            sdk.files.download(
                remote_path=ETHOS_FILENAME,
                local_path=str(downloaded),
                fileset=source_fileset,
                workspace=source_workspace,
            )
        except Exception as exc:
            # "No ethos" reaches us in several shapes (missing fileset, missing file,
            # empty match) depending on the files client and transport, and none of
            # them is fatal here — so treat any failure to fetch as "there isn't one".
            logger.info(
                "No %s staged for the optimized agent: %s/%s could not be read (%s)",
                ETHOS_FILENAME,
                source_workspace,
                source_fileset,
                exc,
            )
            return

        source_ethos = downloaded / ETHOS_FILENAME
        if not source_ethos.is_file():
            logger.info(
                "No %s staged for the optimized agent: fileset %s/%s holds none",
                ETHOS_FILENAME,
                source_workspace,
                source_fileset,
            )
            return

        shutil.copyfile(source_ethos, staged / ETHOS_FILENAME)
        logger.info("Staged %s from source agent fileset %s/%s", ETHOS_FILENAME, source_workspace, source_fileset)
