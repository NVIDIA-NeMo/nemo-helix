# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility fixes for the Fabric Deep Agents runtime bundled in the image."""

import inspect
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

# The wrappers this module has installed, so applying a compatibility patch twice is a no-op.
_INSTALLED: set[Callable[..., Any]] = set()


def virtualize_skill_sources(skill_paths: Iterable[str], workspace_root: Path) -> list[str]:
    """Convert normalized host skill paths under the workspace to virtual paths."""
    resolved_root = workspace_root.resolve()
    normalized: list[str] = []
    for skill_path in skill_paths:
        path = Path(skill_path)
        if not path.is_absolute():
            normalized.append(skill_path)
            continue
        try:
            relative = path.resolve().relative_to(resolved_root)
        except ValueError:
            normalized.append(skill_path)
            continue
        normalized.append(f"/{relative.as_posix()}")
    return normalized


def apply_deepagents_skill_path_compatibility() -> None:
    """Patch legacy Fabric to honor virtual-mode skill paths under its workspace."""
    from nemo_fabric_adapters.deepagents import adapter as adapter_module

    adapter: Any = adapter_module

    # Fabric 0.2 uses typed runtime/config objects and resolves relative skill
    # paths directly against the workspace. Its adapter no longer needs this
    # payload-based compatibility patch.
    if "base_dir" in inspect.signature(adapter.resolve_backend).parameters:
        return

    common_utils = adapter.common_utils

    original_plan = common_utils.capability_plan
    if original_plan not in _INSTALLED:

        def capability_plan(payload: dict[str, Any]) -> dict[str, Any]:
            plan = original_plan(payload)
            native = plan.get("native")
            if not isinstance(native, dict) or not native.get("skill_paths"):
                return plan
            environment = common_utils.environment_payload(payload)
            configured_workspace = Path(str(environment.get("workspace") or "."))
            workspace_root = (
                configured_workspace
                if configured_workspace.is_absolute()
                else Path(common_utils.base_dir(payload)) / configured_workspace
            )
            normalized_native = {
                **native,
                "skill_paths": virtualize_skill_sources(native["skill_paths"], workspace_root),
            }
            return {**plan, "native": normalized_native}

        _INSTALLED.add(capability_plan)
        common_utils.capability_plan = capability_plan

    original = adapter.resolve_skills
    if original in _INSTALLED:
        return

    def resolve_skills(payload: dict[str, Any]) -> list[str] | None:
        skill_paths = original(payload)
        if not skill_paths:
            return None
        environment = adapter.common_utils.environment_payload(payload)
        configured_workspace = Path(str(environment.get("workspace") or "."))
        workspace_root = (
            configured_workspace
            if configured_workspace.is_absolute()
            else Path(adapter.common_utils.base_dir(payload)) / configured_workspace
        )
        return virtualize_skill_sources(skill_paths, workspace_root)

    _INSTALLED.add(resolve_skills)
    adapter.resolve_skills = resolve_skills


def apply_deepagents_mcp_env_compatibility() -> None:
    """Preserve the adapter environment when Fabric starts stdio MCP tools."""
    from nemo_fabric_adapters.deepagents import adapter as adapter_module

    adapter: Any = adapter_module
    original = adapter._mcp_connection
    if original in _INSTALLED:
        return

    def mcp_connection(name: str, spec: Any) -> dict[str, Any]:
        connection = original(name, spec)
        if connection.get("transport") == "stdio":
            connection["env"] = {**os.environ, **connection.get("env", {})}
        return connection

    _INSTALLED.add(mcp_connection)
    adapter._mcp_connection = mcp_connection


def apply_platform_skill_translation_compatibility() -> None:
    """Translate packaged relative skill roots to Deep Agents virtual paths."""
    from nemo_agents_plugin.fabric import translator as translator_module

    translator: Any = translator_module
    original = translator._skills_config
    if original in _INSTALLED:
        return

    def skills_config(config: Any) -> Any:
        translated = original(config)
        if translated is not None:
            translated.paths = [str(path) if Path(path).is_absolute() else f"/{path}" for path in translated.paths]
        return translated

    _INSTALLED.add(skills_config)
    translator._skills_config = skills_config
