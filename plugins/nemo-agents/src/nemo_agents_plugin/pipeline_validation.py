# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline config validation that reports every problem: the static checks, then workspace checks
(agents are registered and runnable, fixed context values match the agent's context_schema)."""

from __future__ import annotations

import logging
import re
from typing import Any

from nemo_agents_plugin.context_schema import context_errors
from nemo_agents_plugin.entities import Agent
from nemo_helix_plugin.agents import pipeline_template as tpl
from nemo_helix_plugin.agents.pipelines import (
    PipelineInline,
    PipelineIssue,
    PipelineStepConfig,
    PipelineValidationResult,
    collect_static_issues,
)
from nemo_helix_plugin.agents.types import NEMO_AGENTS_SPEC_CONFIG_FORMAT, EntityMetadata
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_helix_plugin.entity_naming import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nemo_helix_plugin.refs import parse_entity_ref
from pydantic import ValidationError

logger = logging.getLogger(__name__)

_METADATA_KEYS = frozenset(EntityMetadata.model_fields) - {"name"}
_NAME = re.compile(NAME_PATTERN)

ContextPath = tuple[str | int, ...]


def strip_entity_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Drop id, workspace, timestamps and other entity metadata, so a config fetched with GET can be reused."""
    return {key: value for key, value in data.items() if key not in _METADATA_KEYS}


def static_issues(data: dict[str, Any]) -> list[PipelineIssue]:
    """The checks that need no workspace, for a pipeline.yaml (or a config as returned by GET)."""
    return _parse(data)[1]


async def check_pipeline_config(
    data: dict[str, Any], *, workspace: str, entity_client: NemoEntitiesClient
) -> PipelineValidationResult:
    """Static checks, then workspace checks when those pass. Never raises for a bad config."""
    pipeline, issues = _parse(data)
    if pipeline is not None and not issues:
        for step in pipeline.steps:
            issues.extend(await _workspace_issues(step, workspace=workspace, entity_client=entity_client))
    return PipelineValidationResult(valid=not issues, issues=issues)


def _parse(data: dict[str, Any]) -> tuple[PipelineInline | None, list[PipelineIssue]]:
    body = strip_entity_metadata(data)
    name = body.pop("name", None)
    try:
        pipeline = PipelineInline.model_validate(body)
    except ValidationError as exc:
        return None, [
            PipelineIssue(
                location=_format_location(error["loc"]) or "pipeline",
                message=error["msg"].removeprefix("Value error, "),
            )
            for error in exc.errors(include_url=False)
        ]
    issues = []
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        issues.append(PipelineIssue(location="name", message=f"Set a valid pipeline name. {NAME_PATTERN_DESCRIPTION}"))
    return pipeline, issues + collect_static_issues(pipeline)


def _format_location(loc: tuple[int | str, ...]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


async def _workspace_issues(
    step: PipelineStepConfig, *, workspace: str, entity_client: NemoEntitiesClient
) -> list[PipelineIssue]:
    where = f"steps[{step.name}]"
    ref = parse_entity_ref(step.agent, default_workspace=workspace)
    try:
        agent = await entity_client.get(Agent, name=ref.name, workspace=ref.workspace)
    except NemoEntityNotFoundError:
        return [
            PipelineIssue(
                location=f"{where}.agent",
                message=f"Agent '{ref.workspace}/{ref.name}' is not registered. Register it from its agent.yaml first.",
            )
        ]
    except PermissionDeniedError as exc:
        logger.info("Could not read agent '%s/%s' for pipeline validation: %s", ref.workspace, ref.name, exc)
        return [
            PipelineIssue(
                location=f"{where}.agent",
                message=f"Agent '{ref.workspace}/{ref.name}' could not be read; check that you have access to it.",
            )
        ]
    issues = []
    if agent.config_format != NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        issues.append(
            PipelineIssue(
                location=f"{where}.agent",
                message=(
                    f"Agent '{ref.workspace}/{ref.name}' uses config format {agent.config_format!r}; pipeline "
                    f"steps run through agents.execute, which supports only {NEMO_AGENTS_SPEC_CONFIG_FORMAT!r}."
                ),
            )
        )
    if agent.context_schema is not None:
        templated: list[ContextPath] = []
        literal = _literal_context(step.context, (), templated)
        for path, message in context_errors(literal, agent.context_schema):
            if any(path[: len(t)] == t for t in templated):
                continue  # only known at run time
            location = f"{where}.context" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path)
            issues.append(
                PipelineIssue(location=location, message=f"Does not match the agent's context_schema: {message}")
            )
    return issues


def _literal_context(value: Any, path: ContextPath, templated: list[ContextPath]) -> Any:
    """*value* as the agent receives it ({% raw %} removed), recording the paths of strings with placeholders."""
    if isinstance(value, str):
        try:
            if not tpl.placeholders(value):
                return tpl.render(value, {})
        except tpl.TemplateSyntaxError:
            pass
        templated.append(path)
        return value
    if isinstance(value, dict):
        return {key: _literal_context(item, (*path, key), templated) for key, item in value.items()}
    if isinstance(value, list):
        return [_literal_context(item, (*path, index), templated) for index, item in enumerate(value)]
    return value
