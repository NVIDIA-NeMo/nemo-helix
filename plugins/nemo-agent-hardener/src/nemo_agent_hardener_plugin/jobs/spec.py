# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The war-game job's input spec (the shape ``AgentHardenerRunJob.run``/``compile`` see)."""

from __future__ import annotations

from nemo_agent_hardener_plugin.model_config import WarGameModels
from pydantic import BaseModel


class WarGameSpec(BaseModel):
    """Canonical war-game inputs — the shape ``run()`` and ``compile()`` see.

    Supply either a saved ``manifest_id`` (the Studio path — materialized on the host from the stored
    agent ref) or a ready ``config`` manifest path (the CLI path).
    """

    config: str | None = None
    manifest_id: str | None = None
    env_file: str | None = None
    driver: str | None = None  # "service" for the Studio-driven HITL path; else the one-shot run
    stop_after_synth: bool = False  # generate/refresh the benign suite (interview+review), then stop before the attack
    # Replay recorded garak hits instead of a live attack: a fileset ref holding a garak hitlog (either
    # uploaded by the user or a prior run's saved hitlog). When set, the job replays it via `--replay <path>`.
    replay_hitlog_fileset: str | None = None
    # Override the benign suite for this run: a fileset ref holding an uploaded requests.csv. When set, it
    # replaces the manifest's suite and is passed via `--benign-suite <path>` (skips synthesis).
    benign_suite_fileset: str | None = None
    # Per-run config overrides (None = use the manifest's stored default). The launch applies these over
    # the materialized manifest without mutating it, so a run can deviate from the saved baseline.
    port: int | None = None
    defenders: list[str] | None = None
    attack_intensity: str | None = None
    rounds: int | None = None
    # Sanity-check (validate-only) mode: freeze a user-chosen set of defenses as the victim's baseline and
    # replay the recorded attacks + benign suite against it WITHOUT generating new mitigations, to measure
    # which attacks are now blocked and which benign requests are wrongly blocked. `defense_workflow` /
    # `defense_policy` are the composed YAMLs (see jobs.defenses.compose_defense); the run seeds them as the
    # baseline and forces `overrides.defenders: []` (zero defenders → no new mitigations).
    validate_only: bool = False
    defense_workflow: str | None = None
    defense_policy: str | None = None
    # Per-run model override (None = use the manifest's stored default / agent-hardener built-ins). The launch
    # merges these over the manifest's stored `models`; model names + base_urls become env vars for the
    # agent-hardener subprocess (attack→GARAK_*, analysis→AGENT_HARDENER_*) and the agent model rewrites the victim
    # LLMs. api_key_secret names are resolved to the corresponding env keys (NIM_API_KEY / INFERENCE_API_KEY).
    models: WarGameModels | None = None
    # The harden run a validate-only sanity check was launched from; recorded on the run so the Harden tab
    # can re-attach its scorecard after a reload (see AgentHardenerRun.source_run).
    source_run: str | None = None
