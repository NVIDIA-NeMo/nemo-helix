# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity definitions for the Agent Hardener plugin — stored in the NeMo Platform entity store.

A :class:`AgentHardenerRun` records one war-game run (agent targeted, victim port, manifest, outcome); a
:class:`AgentHardenerManifest` is a named, reusable war-game target scaffolded from a deployed agent or an
uploaded NAT project. ``name``/``workspace``/``created_at``/``id`` are inherited from the base and
managed by the store; only domain fields are declared here. The ``AGENT_HARDENER_*_TYPE`` constants are
the canonical entity-type strings used at every call site.
"""

from __future__ import annotations

from typing import Literal

from nemo_agent_hardener_plugin.model_config import WarGameModels
from nemo_platform_plugin.entity import NemoEntity
from pydantic import Field

AGENT_HARDENER_RUN_TYPE = "agent_hardener_run"
AGENT_HARDENER_MANIFEST_TYPE = "agent_hardener_manifest"

RunStatus = Literal["running", "completed", "failed"]


class AgentHardenerRun(NemoEntity, entity_type=AGENT_HARDENER_RUN_TYPE):
    """A record of one Agent Hardener war-game run."""

    agent: str = Field(default="", description="Targeted agent reference (workspace/name).")
    job_id: str = Field(default="", description="Platform job that drove this run (for live status/HITL).")
    port: int = Field(default=0, description="Victim port the war-game attacked.")
    manifest: str = Field(default="", description="Path to the agent-hardener.yaml manifest used.")
    manifest_id: str = Field(default="", description="Manifest this run belongs to (scopes 'replay last run').")
    status: RunStatus = Field(default="failed", description="Final run status.")
    returncode: int = Field(default=-1, description="Exit code from `agent-hardener run`.")
    summary: str = Field(default="", description="Short human-readable outcome summary.")
    error_category: str = Field(
        default="",
        description="Classified failure category when status is 'failed' (e.g. sandbox, missing_credential, "
        "manifest, network); empty for a successful run.",
    )
    error_message: str = Field(default="", description="Operator-facing failure message when the run failed.")
    error_remediation: str = Field(
        default="", description="Suggested next step to resolve the failure; empty for a successful run."
    )
    hitlog_fileset: str = Field(
        default="",
        description="Fileset ref of the garak hitlog this run produced, if any; replay a later run from it.",
    )
    events_fileset: str = Field(
        default="",
        description="Fileset ref of the run's events.jsonl, uploaded at completion for durable history.",
    )
    source_run: str = Field(
        default="",
        description="For a validate-only sanity-check run, the name of the harden run it was launched from; "
        "lets the Harden tab re-attach the scorecard on reload. Empty for normal war-game runs.",
    )


class AgentHardenerManifest(NemoEntity, entity_type=AGENT_HARDENER_MANIFEST_TYPE):
    """A named, reusable war-game target (``name`` is its id), from a registered agent or an uploaded project.

    Either way the package is persisted as a fileset the run re-downloads, so a manifest is a frozen
    target rather than a query re-evaluated each run: editing the agent does not change an existing
    manifest until it is refreshed. A project manifest has nothing to refresh *from* — its bundle is the
    upload — which is why the two sources are distinguished rather than merged.
    """

    source_type: Literal["agent", "project"] = Field(
        default="agent",
        description="Where the victim came from. The run reads this to decide which bundle field to expand.",
    )
    agent: str = Field(default="", description="Registered agent reference (workspace/name) this manifest targets.")
    project_fileset: str = Field(
        default="",
        description="Fileset ref holding the uploaded project bundle, for a 'project' manifest. The run "
        "expands this instead of ``agent_fileset``.",
    )
    agent_fileset: str = Field(
        default="",
        description="Fileset ref holding the agent package resolved from the agent — its config plus the "
        "Dockerfile that serves it. Empty on manifests created before targets were frozen; those "
        "re-resolve once, then store a ref.",
    )
    dockerfile: str = Field(
        default="",
        description="Path within the package to the Dockerfile the victim image is built from, so a manifest "
        "records which image it ran rather than only that it had one.",
    )
    binaries: list[str] = Field(
        default_factory=list,
        description="In-container glob patterns scoping which processes may egress; agent-hardener requires them "
        "because the layout of an image it did not write cannot be inferred.",
    )
    manifest_yaml: str = Field(default="", description="The resolved agent-hardener.yaml content (for display).")
    port: int = Field(default=0, description="Victim port the war-game will target.")
    secrets: list[str] = Field(default_factory=list, description="Secret names the victim agent requires.")
    egress: list[str] = Field(
        default_factory=list,
        description="Allow-listed egress host[:port] entries the victim may reach. Config-only agents "
        "keep tool hosts in packaged code, so egress discovery can't find them; without these the "
        "victim's outbound calls are dropped.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Non-secret environment variables for the victim (agent-hardener's agent.env) — a "
        "host-backend URL, a feature flag. Stored in plaintext on this entity, so never put "
        "credentials here: those belong in `secrets`, which names them and resolves the values from "
        "the platform Secrets store at run time.",
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal notes from scaffolding.")
    benign_suite: list[dict[str, str]] = Field(
        default_factory=list,
        description="Cached, reviewed benign test suite (tool,payload,label,rationale,persona rows); "
        "generated on the first run and reused/edited thereafter. Empty until generated.",
    )
    benign_interview: list[dict[str, str]] = Field(
        default_factory=list,
        description="Interview Q&A (gap,question,answer rows) captured during the last benign-suite "
        "generation, kept for display. Empty until generated.",
    )
    defenders: list[str] = Field(
        default_factory=list,
        description="Enabled defender keys ('guardrails','openshell'); empty means agent-hardener's defaults "
        "(all applicable). Materialized into the manifest's overrides.defenders at run time.",
    )
    attack_intensity: Literal["light", "standard", "thorough"] = Field(
        default="standard",
        description="Attacker (garak) effort preset, materialized into the manifest's garak block at run time.",
    )
    rounds: int = Field(
        default=1,
        ge=1,
        description="Number of iterative attack/defend/validate hardening rounds; passed to agent-hardener's "
        "`run --rounds` at run time.",
    )
    models: WarGameModels = Field(
        default_factory=WarGameModels,
        description="Stored default model selection (attack/analysis/agent groups); an unset group uses "
        "agent-hardener's built-in default. A run may override these per-launch.",
    )

    @classmethod
    def from_agent_resolution(
        cls,
        *,
        name: str,
        workspace: str,
        agent_ref: str,
        manifest_yaml: str,
        port: int,
        secrets: list[str],
        warnings: list[str],
        egress: list[str] | None = None,
        env: dict[str, str] | None = None,
        models: WarGameModels | None = None,
        agent_fileset: str = "",
    ) -> AgentHardenerManifest:
        """Build an ``agent``-source manifest entity from a resolved agent scaffold.

        Shared by ``POST /manifests`` and the refresh route so both persist the same shape from
        :func:`resolve_agent_to_manifest`'s output. ``agent_fileset`` holds the scaffold the run
        re-downloads; without it the run has to re-resolve (legacy manifests only).
        """
        return cls(
            name=name,
            workspace=workspace,
            agent=agent_ref,
            manifest_yaml=manifest_yaml,
            agent_fileset=agent_fileset,
            port=port,
            secrets=secrets,
            egress=egress or [],
            env=env or {},
            warnings=warnings,
            models=models or WarGameModels(),
        )

    @classmethod
    def from_project_upload(
        cls,
        *,
        name: str,
        workspace: str,
        project_fileset: str,
        manifest_yaml: str,
        dockerfile: str,
        binaries: list[str],
        port: int,
        secrets: list[str],
        warnings: list[str],
        egress: list[str] | None = None,
        env: dict[str, str] | None = None,
        models: WarGameModels | None = None,
    ) -> AgentHardenerManifest:
        """Build a ``project``-source manifest entity from an uploaded bundle and its derivation.

        ``agent`` stays empty: there is no registered agent behind a project manifest, and inventing a
        reference for one would make it look refreshable when nothing exists to refresh against.
        """
        return cls(
            name=name,
            workspace=workspace,
            source_type="project",
            project_fileset=project_fileset,
            manifest_yaml=manifest_yaml,
            dockerfile=dockerfile,
            binaries=binaries,
            port=port,
            secrets=secrets,
            egress=egress or [],
            env=env or {},
            warnings=warnings,
            models=models or WarGameModels(),
        )
