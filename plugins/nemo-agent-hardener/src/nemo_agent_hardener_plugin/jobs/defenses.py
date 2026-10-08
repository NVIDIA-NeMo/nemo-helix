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
import tomllib
from typing import Any

import tomli_w

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
    """Build ``(guardrails_toml, policy_yaml)`` from the hardened mitigations keeping only *selected_ids*.

    - Guardrails: the hardened Relay plugin config with every unselected ``custom_guardrail_N`` entry
      removed. ``None`` when the run produced no guardrail change.
    - Policy: the hardened policy when ``"openshell_policy"`` is selected, else the baseline. ``None``
      when the run produced no policy change.
    """
    selected = set(selected_ids)
    guardrails = mitigations.get("guardrails") or {}
    after = guardrails.get("after")
    guardrails_toml = _compose_guardrails(after, selected) if isinstance(after, str) else None

    policy = mitigations.get("policy") or {}
    policy_yaml: str | None = None
    if policy:
        policy_yaml = policy.get("after") if _POLICY_DEFENSE_ID in selected else policy.get("before")

    return guardrails_toml, policy_yaml


def _compose_guardrails(after_text: str, selected: set[str]) -> str:
    """Return the hardened plugin config with unselected ``custom_guardrail_N`` entries removed.

    Simpler than the NAT version it replaces: a guardrail is one self-contained table, so pruning is a
    list filter. There is no second place referencing it, which is what ``_drop_middleware_refs`` had
    to clean up — and getting that wrong left a dangling name that stopped the victim serving.
    """
    try:
        document = tomllib.loads(after_text)
    except tomllib.TOMLDecodeError:
        return after_text
    for entry in document.get("components", []):
        config = entry.get("config") if isinstance(entry, dict) else None
        if not isinstance(config, dict):
            continue
        config["guardrails"] = [
            rail
            for rail in config.get("guardrails", [])
            if not (isinstance(rail, dict) and _CUSTOM_GUARDRAIL_RE.match(str(rail.get("name", ""))))
            or str(rail.get("name")) in selected
        ]
    return tomli_w.dumps(document)
