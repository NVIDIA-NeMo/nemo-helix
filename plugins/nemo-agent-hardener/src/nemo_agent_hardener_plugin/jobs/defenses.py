# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose a chosen subset of a run's recommended defenses into deployable workflow + policy YAML.

A hardening run's ``mitigations`` artifact enumerates each individually selectable defense in
``defenses[]`` (one per ``custom_guardrail_N`` middleware plus, optionally, the hardened OpenShell
policy). The Studio "harden" flow lets the user pick a subset; this rebuilds the workflow with only the
selected guardrails and picks the hardened-vs-baseline policy, so the selection can be previewed, frozen
into a sanity-check run, and applied. Guardrails are structurally independent (a keyed global middleware
entry + a name in the attacked tool's ``middleware`` list), so dropping one is a clean delete.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

_CUSTOM_GUARDRAIL_RE = re.compile(r"^custom_guardrail_\d+$")
_POLICY_DEFENSE_ID = "openshell_policy"


def defense_ids(mitigations: dict[str, Any]) -> list[str]:
    """The ids of every selectable defense in the run's mitigations artifact (``defenses[].id``)."""
    return [d["id"] for d in mitigations.get("defenses", []) if isinstance(d, dict) and d.get("id")]


def select_defense_ids(
    all_ids: list[str], keep: list[str] | None = None, exclude: list[str] | None = None
) -> list[str]:
    """Resolve a ``keep``/``exclude`` selection over *all_ids* (order preserved).

    ``keep`` wins when given (only those ids, if they exist); else ``exclude`` drops the named ids; else all.
    """
    if keep:
        keep_set = set(keep)
        return [i for i in all_ids if i in keep_set]
    if exclude:
        exclude_set = set(exclude)
        return [i for i in all_ids if i not in exclude_set]
    return list(all_ids)


def compose_defense(mitigations: dict[str, Any], selected_ids: list[str]) -> tuple[str | None, str | None]:
    """Build ``(workflow_yaml, policy_yaml)`` from the hardened mitigations keeping only *selected_ids*.

    - Workflow: the hardened workflow with every unselected ``custom_guardrail_N`` removed. ``None`` when the
      run produced no workflow change.
    - Policy: the hardened policy when ``"openshell_policy"`` is selected, else the baseline. ``None`` when the
      run produced no policy change.
    """
    selected = set(selected_ids)
    workflow = mitigations.get("workflow") or {}
    workflow_after = workflow.get("after")
    workflow_yaml = _compose_workflow(workflow_after, selected) if isinstance(workflow_after, str) else None

    policy = mitigations.get("policy") or {}
    policy_yaml: str | None = None
    if policy:
        policy_yaml = policy.get("after") if _POLICY_DEFENSE_ID in selected else policy.get("before")

    return workflow_yaml, policy_yaml


def _compose_workflow(after_text: str, selected: set[str]) -> str:
    """Return the hardened workflow with unselected ``custom_guardrail_N`` middleware removed."""
    config = yaml.safe_load(after_text) or {}
    middleware = config.get("middleware")
    if not isinstance(middleware, dict):
        return after_text  # no guardrail middleware to prune

    removed = [
        name
        for name in list(middleware)
        if isinstance(name, str) and _CUSTOM_GUARDRAIL_RE.match(name) and name not in selected
    ]
    for name in removed:
        middleware.pop(name, None)
    _drop_middleware_refs(config, set(removed))

    # The guardrails' shared safety_llm is only needed while some custom guardrail remains.
    if not any(isinstance(k, str) and _CUSTOM_GUARDRAIL_RE.match(k) for k in middleware):
        llms = config.get("llms")
        if isinstance(llms, dict):
            llms.pop("safety_llm", None)

    return yaml.safe_dump(config, sort_keys=False)


def _drop_middleware_refs(config: dict[str, Any], removed: set[str]) -> None:
    """Remove references to *removed* guardrails from every middleware-bearing component.

    ``workflow`` is a single component dict alongside the ``functions``/``function_groups`` mappings and
    can carry its own ``middleware`` list. Missing it leaves a name pointing at a middleware we just
    deleted, and the victim then fails config validation ("middleware type not found") and never serves.
    """
    components: list[Any] = []
    for block_key in ("functions", "function_groups"):
        block = config.get(block_key)
        if isinstance(block, dict):
            components.extend(block.values())
    workflow = config.get("workflow")
    if isinstance(workflow, dict):
        components.append(workflow)

    for component in components:
        if not isinstance(component, dict):
            continue
        refs = component.get("middleware")
        if isinstance(refs, str):
            refs = [refs]
        if isinstance(refs, list):
            component["middleware"] = [ref for ref in refs if ref not in removed]
