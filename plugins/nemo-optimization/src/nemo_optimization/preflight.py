# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight checks before dispatching an optimize study."""

from __future__ import annotations

import logging
import re
from typing import Any

from nemo_platform_plugin.client.adapter import SyncPlatformClient, client_from_platform
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.entities.base import parse_qualified_name
from nemo_platform_plugin.virtual_models.client import VirtualModelsClient

logger = logging.getLogger(__name__)

# Legacy NAT ``llms`` blocks routed through Platform IGW.
_IGW_LLM_TYPES = frozenset({"openai", "nim", "azure_openai"})
# Fabric ``models`` providers that speak through Platform's OpenAI-compatible IGW.
_FABRIC_IGW_MODEL_PROVIDERS = frozenset({"openai", "nvidia", "openai-compatible"})
_UNEXPANDED_ENV_VAR_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def preflight_validate_llm_models(
    optimize_config: dict[str, Any],
    *,
    workspace: str,
    sdk: SyncPlatformClient | None,
    agent_config: dict[str, Any] | None = None,
) -> None:
    """Validate IGW-routed model names against workspace VirtualModels.

    Accepts both legacy NAT ``llms`` blocks and Fabric ``models`` entries from
    ``optimize_config`` and an optional resolved ``agent_config``.
    """
    if sdk is None:
        return

    to_check = _collect_igw_model_names(optimize_config, agent_config=agent_config, workspace=workspace)
    if not to_check:
        return

    virtual_models = client_from_platform(sdk, VirtualModelsClient)
    missing: list[tuple[str, str]] = []
    for (target_ws, target_name), location in to_check.items():
        try:
            virtual_models.get_virtual_model(name=target_name, workspace=target_ws)
        except NotFoundError:
            missing.append((f"{target_ws}/{target_name}", location))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Could not validate model at %s (model=%r) in workspace %r: %s",
                location,
                target_name,
                target_ws,
                exc,
                exc_info=exc,
            )

    if missing:
        details = ", ".join(f"{name!r} ({location})" for name, location in missing)
        raise ValueError(
            f"The following LLM model(s) are not registered as VirtualModels in workspace {workspace!r}: {details}."
        )


def _collect_igw_model_names(
    optimize_config: dict[str, Any],
    *,
    agent_config: dict[str, Any] | None,
    workspace: str,
) -> dict[tuple[str, str], str]:
    """Return ``{(vm_workspace, vm_name): config_location}`` for IGW-bound models."""
    to_check: dict[tuple[str, str], str] = {}

    sources: list[tuple[str, dict[str, Any]]] = [("optimize_config", optimize_config)]
    if isinstance(agent_config, dict):
        sources.append(("agent_config", agent_config))

    for source_name, payload in sources:
        llms = payload.get("llms")
        if isinstance(llms, dict):
            for llm_key, llm_cfg in llms.items():
                location = f"{source_name}.llms.{llm_key}.model_name"
                _maybe_add_nat_llm(to_check, llm_cfg, location=location, workspace=workspace)

        models = payload.get("models")
        if isinstance(models, dict):
            for model_key, model_cfg in models.items():
                location = f"{source_name}.models.{model_key}.model"
                _maybe_add_fabric_model(to_check, model_cfg, location=location, workspace=workspace)

    return to_check


def _maybe_add_nat_llm(
    to_check: dict[tuple[str, str], str],
    llm_cfg: Any,
    *,
    location: str,
    workspace: str,
) -> None:
    if not isinstance(llm_cfg, dict):
        return
    if llm_cfg.get("_type") not in _IGW_LLM_TYPES:
        return
    model_name = llm_cfg.get("model_name")
    _add_model_name(to_check, model_name, location=location, workspace=workspace)


def _maybe_add_fabric_model(
    to_check: dict[tuple[str, str], str],
    model_cfg: Any,
    *,
    location: str,
    workspace: str,
) -> None:
    if not isinstance(model_cfg, dict):
        return
    provider = model_cfg.get("provider")
    if not isinstance(provider, str) or provider.lower() not in _FABRIC_IGW_MODEL_PROVIDERS:
        return
    # Explicit non-IGW endpoints (e.g. inference-api.nvidia.com) are not VirtualModels.
    if _has_external_base_url(model_cfg):
        return
    model_name = model_cfg.get("model") or model_cfg.get("model_name")
    _add_model_name(to_check, model_name, location=location, workspace=workspace)


def _has_external_base_url(model_cfg: dict[str, Any]) -> bool:
    base_url = model_cfg.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        settings = model_cfg.get("settings")
        if isinstance(settings, dict):
            base_url = settings.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    return "inference-gateway" not in base_url


def _add_model_name(
    to_check: dict[tuple[str, str], str],
    model_name: Any,
    *,
    location: str,
    workspace: str,
) -> None:
    if not isinstance(model_name, str) or not model_name:
        return
    if _UNEXPANDED_ENV_VAR_RE.search(model_name):
        return
    target_ws, target_name = parse_qualified_name(model_name, default_workspace=workspace)
    if not target_name:
        return
    to_check.setdefault((target_ws, target_name), location)
