# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery and dispatch for ``agents.execute`` post-run actions.

The *contract* an action implements lives in
:mod:`nemo_helix_plugin.agents.post_run_actions`, so a plugin can write one
without depending on this package. This module is the host half: resolving an
installed action kind and listing what is installed.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from importlib.metadata import EntryPoint, entry_points
from typing import Any, ClassVar

from nemo_helix_plugin.agents.post_run_actions import (
    POST_RUN_ACTION_ENTRY_POINT_GROUP,
    PostRunAction,
    PostRunActionContext,
)
from nemo_helix_plugin.agents.types import NOOP_POST_RUN_ACTION_KIND as NOOP_POST_RUN_ACTION_KIND
from nemo_helix_plugin.agents.types import PostRunActionInfo
from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


class NoopPostRunActionConfig(BaseModel):
    """The noop action takes no configuration at all."""

    model_config = ConfigDict(extra="forbid")


class NoopPostRunAction:
    """Default action used when a job configures none."""

    title: ClassVar[str] = "No action"
    description: ClassVar[str] = "Do nothing after the agent runs; the job's results are the agent's own."
    config_model: ClassVar[type[BaseModel]] = NoopPostRunActionConfig

    def after_invoke(self, context: PostRunActionContext) -> None:
        del context


def resolve_post_run_action(kind: str) -> type[PostRunAction]:
    """Resolve an installed post-run action kind."""
    return _load_post_run_action(kind)


def list_post_run_actions() -> list[PostRunActionInfo]:
    """Describe every installed post-run action, sorted by kind.

    A kind that cannot be loaded — conflicting registrations, or an entry point
    that fails to import — is logged and left out rather than failing the whole
    listing; creating a job with that kind still reports the real error.
    """
    registered = _registered_entry_points()
    infos: list[PostRunActionInfo] = []
    for kind in sorted(registered):
        try:
            action_cls = _load_registered_post_run_action(kind, registered)
        except Exception:
            logger.exception("Skipping post-run action %r: it could not be loaded.", kind)
            continue
        infos.append(
            PostRunActionInfo(
                kind=kind,
                # Lenient for out-of-tree actions predating the display metadata.
                title=getattr(action_cls, "title", kind),
                description=getattr(action_cls, "description", ""),
            )
        )
    return infos


def validate_post_run_action_kinds(kinds: set[str]) -> None:
    """Reject a declared kind that is not installed, or not uniquely so.

    Only the kind is checked: an action's config is chosen per run, so it is
    validated when a job sets it.
    """
    registered = _registered_entry_points()
    for kind in sorted(kinds):
        _load_registered_post_run_action(kind, registered)


def validate_post_run_action_config(
    kind: str, config: dict[str, Any], *, action: type[PostRunAction] | None = None
) -> None:
    """Reject a malformed post-run action config body.

    Pass *action* when the kind is already resolved, to validate against that
    class without looking it up again. Lenient by design: an action that
    declares no ``config_model`` is accepted unvalidated rather than rejected.
    """
    if action is None:
        action = _load_post_run_action(kind)
    config_model = getattr(action, "config_model", None)
    if config_model is None:
        return
    try:
        config_model.model_validate(config)
    except ValidationError as exc:
        # The caller supplied a `kind`, not a model name; lead with the kind.
        raise ValueError(f"Invalid config for post-run action {kind!r}: {exc}") from exc


def _registered_entry_points() -> dict[str, list[EntryPoint]]:
    registered: dict[str, list[EntryPoint]] = defaultdict(list)
    for entry_point in entry_points(group=POST_RUN_ACTION_ENTRY_POINT_GROUP):
        registered[entry_point.name].append(entry_point)
    return registered


def _load_post_run_action(kind: str) -> type[PostRunAction]:
    return _load_registered_post_run_action(kind, _registered_entry_points())


def _load_registered_post_run_action(kind: str, registered: dict[str, list[EntryPoint]]) -> type[PostRunAction]:
    matches = registered.get(kind)
    if not matches:
        installed = ", ".join(sorted(registered)) or "none"
        raise ValueError(f"Unknown post-run action {kind!r}. Installed post-run actions: {installed}.")

    # A kind may legitimately be declared by more than one installed
    # distribution: the aggregate ``nemo-helix`` wheel re-declares every
    # bundled plugin's entry points, so a standard install sees each of them
    # twice — once from the plugin, once from the aggregate. What must be
    # unique is the *implementation*, not the number of declarations, so
    # collapse identical targets and reject only genuine conflicts.
    targets = {entry_point.value for entry_point in matches}
    if len(targets) > 1:
        conflicting = ", ".join(sorted(targets))
        raise ValueError(f"Conflicting post-run actions are registered for {kind!r}: {conflicting}.")
    return matches[0].load()
