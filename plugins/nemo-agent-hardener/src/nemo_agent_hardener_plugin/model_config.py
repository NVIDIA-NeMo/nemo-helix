# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-selectable model configuration for a war-game.

Agent Hardener's model-driven roles collapse into three user-facing groups:

- ``attack``   — garak's red-team + detector models (the adversary).
- ``analysis`` — the defenders + the benign validator (both its synth suite-generation and its judge)
  — one shared "analysis" model.
- ``safety``   — the guardrail middleware the guardrails defender installs on the victim.

The victim's own LLM is deliberately *not* a group. In NAT it is declared in the workflow YAML, so
overriding it would mean rewriting the target's own config — the war-game measures the agent rather
than editing it. Change it in the project's workflow and re-upload.

``attack`` and ``analysis`` reach agent-hardener as subprocess env vars; ``safety`` travels in the
manifest instead (``overrides.defenders`` → the guardrails entry's ``config``), because it is consumed
by a defender rather than by the agent-hardener process.

Each group is a :class:`ModelChoice` (model name, optional custom ``base_url``, optional Secrets
name for a custom provider key). ``None`` anywhere means "use the built-in default", so an unset
config reproduces today's behavior exactly.

This module is the single source of truth shared by the entity (stored default), the job spec
(per-run override), and the API (the defaults the UI pre-fills). It imports nothing plugin-internal
so it can be depended on from anywhere without cycles.
"""

from __future__ import annotations

from nemo_platform_plugin.agent_hardener.types import (
    ANALYSIS_DEFAULT_BASE_URL,
    ANALYSIS_DEFAULT_MODEL,
    ATTACK_DEFAULT_BASE_URL,
    ATTACK_DEFAULT_MODEL,
    ModelChoice,
    ModelConfigDefaults,
    ModelGroupDefault,
    WarGameModels,
    model_config_defaults,
)

__all__ = [
    "ANALYSIS_DEFAULT_BASE_URL",
    "ANALYSIS_DEFAULT_MODEL",
    "ATTACK_DEFAULT_BASE_URL",
    "ATTACK_DEFAULT_MODEL",
    "ModelChoice",
    "ModelConfigDefaults",
    "ModelGroupDefault",
    "WarGameModels",
    "model_config_defaults",
]
