# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery and dispatch for trusted ``agents.execute`` lifecycle extensions.

The *contract* an extension implements lives in
:mod:`nemo_platform_plugin.agents.execute_extensions`, so a plugin can write one
without depending on this package. This module is the host half: resolving an
installed extension kind and running it.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, ClassVar

from nemo_platform_plugin.agents.execute_extensions import (
    EXECUTE_AGENT_EXTENSION_ENTRY_POINT_GROUP,
    ExecuteAgentAfterInvokeContext,
    ExecuteAgentExtension,
)
from pydantic import BaseModel, ConfigDict, ValidationError

NOOP_EXECUTE_AGENT_EXTENSION_KIND = "noop"


class NoopExecuteAgentExtensionConfig(BaseModel):
    """The noop extension takes no configuration at all."""

    model_config = ConfigDict(extra="forbid")


class NoopExecuteAgentExtension:
    """Default extension used when no plugin extension is configured."""

    config_model: ClassVar[type[BaseModel]] = NoopExecuteAgentExtensionConfig

    def after_invoke(self, context: ExecuteAgentAfterInvokeContext) -> None:
        del context


def resolve_execute_agent_extension(kind: str) -> type[ExecuteAgentExtension]:
    """Resolve an installed trusted execute-agent extension kind."""
    return _load_execute_agent_extension(kind)


def validate_execute_agent_extension_config(kind: str, config: dict[str, Any]) -> None:
    """Reject a malformed extension config body.

    Lenient by design: an extension that declares no ``config_model`` is
    accepted unvalidated rather than rejected.
    """
    config_model = getattr(_load_execute_agent_extension(kind), "config_model", None)
    if config_model is None:
        return
    try:
        config_model.model_validate(config)
    except ValidationError as exc:
        # The caller supplied a `kind`, not a model name; lead with the kind.
        raise ValueError(f"Invalid config for agents.execute extension {kind!r}: {exc}") from exc


def run_execute_agent_after_invoke_extension(kind: str, context: ExecuteAgentAfterInvokeContext) -> None:
    """Run the ``after_invoke`` lifecycle method for an installed extension kind."""
    extension_cls = _load_execute_agent_extension(kind)
    extension_cls().after_invoke(context)


def _load_execute_agent_extension(kind: str) -> type[ExecuteAgentExtension]:
    if kind == NOOP_EXECUTE_AGENT_EXTENSION_KIND:
        return NoopExecuteAgentExtension

    matches = [
        entry_point
        for entry_point in entry_points(group=EXECUTE_AGENT_EXTENSION_ENTRY_POINT_GROUP)
        if entry_point.name == kind
    ]
    if not matches:
        raise ValueError(f"Unknown agents.execute extension {kind!r}.")

    # A kind may legitimately be declared by more than one installed
    # distribution: the aggregate ``nemo-platform`` wheel re-declares every
    # bundled plugin's entry points, so a standard install sees each of them
    # twice — once from the plugin, once from the aggregate. What must be
    # unique is the *implementation*, not the number of declarations, so
    # collapse identical targets and reject only genuine conflicts.
    targets = {entry_point.value for entry_point in matches}
    if len(targets) > 1:
        conflicting = ", ".join(sorted(targets))
        raise ValueError(f"Conflicting agents.execute extensions are registered for {kind!r}: {conflicting}.")
    return matches[0].load()
