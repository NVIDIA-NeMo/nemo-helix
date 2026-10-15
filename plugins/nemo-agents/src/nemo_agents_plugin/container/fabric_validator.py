# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-context validation for packaged Fabric-backed agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_agents_plugin.agent_config import AgentConfig, AgentConfigLoadError, load_agent_config
from nemo_agents_plugin.fabric.translator import FabricTranslationError, translate_agent_config
from nemo_agents_plugin.fabric.validation import FabricValidationError, plan_fabric_config

if TYPE_CHECKING:
    from nemo_fabric import FabricConfig


class FabricPackageValidationError(ValueError):
    """Raised when a Fabric agent cannot be validated for packaging."""


class FabricPackageArtifactError(FabricPackageValidationError):
    """Raised when a referenced Fabric artifact cannot be packaged."""


@dataclass(frozen=True, slots=True)
class FabricPackageValidationResult:
    """Loaded, translated, and planned Fabric package configuration."""

    agent_config: AgentConfig
    fabric_config: FabricConfig
    plan: Any


async def validate_fabric_agent_package(
    agent_config_path: Path,
    *,
    context_dir: Path,
    fabric: Any | None = None,
) -> FabricPackageValidationResult:
    """Load, translate, plan, and validate artifacts for a Fabric package."""
    try:
        agent_config = load_agent_config(agent_config_path)
        fabric_config = translate_agent_config(agent_config)
        plan = await _plan_unless_in_image_adapter(
            fabric_config,
            base_dir=agent_config_path.resolve().parent,
            fabric=fabric,
        )
    except (AgentConfigLoadError, FabricTranslationError, FabricValidationError) as error:
        raise FabricPackageValidationError(f"Fabric package validation failed: {error}") from error

    validate_fabric_package_artifacts(
        agent_config,
        agent_config_path=agent_config_path,
        context_dir=context_dir,
    )
    return FabricPackageValidationResult(
        agent_config=agent_config,
        fabric_config=fabric_config,
        plan=plan,
    )


# Adapters whose descriptor is staged INSIDE the built image rather than installed on
# the producer host, so host-side `Fabric.plan` (preinstalled resolution) cannot resolve
# them. For these the pre-build plan is skipped; translate + artifact validation still run.
_IN_IMAGE_ADAPTER_IDS = frozenset({"nvidia.fabric.pi"})


async def _plan_unless_in_image_adapter(
    fabric_config: FabricConfig,
    *,
    base_dir: Path,
    fabric: Any | None,
) -> Any | None:
    """Plan the config, unless its adapter is only installed inside the image.

    The Pi adapter is an npm package staged into the image at build time, not a host
    Python package, so host-side preinstalled resolution would always fail. Skip only
    the plan for those adapters; config translation (above) and artifact checks (below)
    still guard the package.
    """
    harness = getattr(fabric_config, "harness", None)
    if harness is not None and getattr(harness, "adapter_id", None) in _IN_IMAGE_ADAPTER_IDS:
        return None
    return await plan_fabric_config(fabric_config, base_dir=base_dir, fabric=fabric)


def validate_fabric_package_artifacts(
    config: AgentConfig,
    *,
    agent_config_path: Path,
    context_dir: Path,
) -> None:
    """Validate that config-referenced inputs will exist in the built image."""
    resolved_context = context_dir.resolve()
    resolved_config = agent_config_path.resolve()
    errors: list[str] = []

    if not resolved_config.is_relative_to(resolved_context):
        errors.append(f"agent config {agent_config_path} is outside Docker build context {resolved_context}")

    for configured_path in config.skills.paths if config.skills is not None else ():
        skill_path = Path(configured_path)
        if skill_path.is_absolute():
            errors.append(
                f"skills.paths entry {configured_path!r} must be relative to agent.yaml "
                "so it remains valid under /workspace"
            )
            continue

        resolved_skill = (resolved_config.parent / skill_path).resolve()
        if not resolved_skill.is_relative_to(resolved_context):
            errors.append(
                f"skills.paths entry {configured_path!r} resolves outside Docker build context {resolved_context}"
            )
            continue
        if not resolved_skill.exists():
            errors.append(f"skills.paths entry {configured_path!r} does not exist at {resolved_skill}")
            continue
        if not resolved_skill.is_dir():
            errors.append(f"skills.paths entry {configured_path!r} must reference a directory: {resolved_skill}")
            continue

        skill_manifest = resolved_skill / "SKILL.md"
        if not skill_manifest.is_file():
            errors.append(f"skills.paths entry {configured_path!r} does not contain SKILL.md: {resolved_skill}")
            continue
        try:
            with skill_manifest.open("rb") as manifest:
                manifest.read(1)
        except OSError as error:
            errors.append(f"skills.paths entry {configured_path!r} contains an unreadable SKILL.md: {error}")

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise FabricPackageArtifactError(f"Fabric package artifact validation failed:\n{details}")
