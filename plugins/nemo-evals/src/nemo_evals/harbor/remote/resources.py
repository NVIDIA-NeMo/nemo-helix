# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each sandbox of a task gets: the request's overrides (agent only), then task.toml, then defaults.

Submit checks these values against the profile's ceilings, and the dispatcher's environment renders the
same ones, so a task never runs with more than submit allowed.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from harbor.models.task.config import EnvironmentConfig, TaskConfig
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
from nemo_evals.harbor.remote.profile import ResolvedHarborSandboxPolicy, resource_defaults

SandboxRole = Literal["agent", "verifier"]


def effective_resources(
    config: EnvironmentConfig, *, overrides: Mapping[str, int], defaults: Mapping[str, int]
) -> tuple[EnvironmentConfig, tuple[str, ...]]:
    """A copy of ``config`` with ``overrides`` applied and unset ``defaults`` fields filled; and those fields.

    Never changes ``config``: Harbor builds an inherited separate verifier's config as a copy of the task's,
    which must keep task.toml's values.
    """
    update = dict(overrides)
    defaulted = tuple(field for field in defaults if field not in update and getattr(config, field) is None)
    update.update({field: defaults[field] for field in defaulted})
    return config.model_copy(update=update, deep=True), defaulted


@dataclass(frozen=True)
class SandboxResources:
    """One sandbox of a task, as it will be sized."""

    role: SandboxRole
    config: EnvironmentConfig
    sources: Mapping[str, str]  # field -> "execution.sandbox", "task.toml" or "default"

    @property
    def defaulted(self) -> tuple[str, ...]:
        return tuple(field for field, source in self.sources.items() if source == "default")


def task_sandboxes(task_config: TaskConfig, policy: ResolvedHarborSandboxPolicy) -> list[SandboxResources]:
    """The agent sandbox, and the separate verifier's when the task has one."""
    overrides = policy.resource_overrides.model_dump(exclude_none=True)
    defaults = resource_defaults(policy)
    sandboxes = [_sized("agent", task_config.environment, overrides, defaults)]
    verifier = resolve_effective_verifier_env_config(task_config, None)
    if verifier is not None:
        sandboxes.append(_sized("verifier", verifier, {}, defaults))
    return sandboxes


def _sized(
    role: SandboxRole, config: EnvironmentConfig, overrides: Mapping[str, int], defaults: Mapping[str, int]
) -> SandboxResources:
    sized, defaulted = effective_resources(config, overrides=overrides, defaults=defaults)
    sources = {
        field: "execution.sandbox" if field in overrides else "default" if field in defaulted else "task.toml"
        for field in ("cpus", "memory_mb", "storage_mb")
        if getattr(sized, field) is not None
    }
    return SandboxResources(role=role, config=sized, sources=sources)


def defaulted_resources_notice(
    tasks: Iterable[tuple[str, Mapping[str, Any]]], policy: ResolvedHarborSandboxPolicy, *, shown: int = 5
) -> str | None:
    """One submit notice naming the tasks whose sandboxes rely on default cpus or memory_mb, or None.

    ``tasks`` are (task id, task.toml config) pairs.
    """
    entries: list[str] = []
    task_ids: set[str] = set()
    for task_id, config in tasks:
        for sandbox in task_sandboxes(TaskConfig.model_validate(config), policy):
            if sandbox.defaulted:
                task_ids.add(task_id)
                values = " ".join(f"{field}={getattr(sandbox.config, field)}" for field in sandbox.defaulted)
                entries.append(f"{task_id} {sandbox.role} {values}")
    if not entries:
        return None
    listed = ", ".join(entries[:shown]) + (f", and {len(entries) - shown} more" if len(entries) > shown else "")
    count = len(task_ids)
    return (
        f"{count} {'task relies' if count == 1 else 'tasks rely'} on default sandbox resources: {listed}. "
        "Set them in task.toml or pass --cpus/--memory-mb."
    )
