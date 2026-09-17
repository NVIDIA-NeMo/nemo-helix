# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the on-host ``agent-hardener.yaml`` the war-game runs against.

Materializes a saved manifest onto disk, applies the run's overrides (attacker intensity, defender
selection, port), and seeds the frozen validate-only baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from nemo_agent_hardener_plugin.agent_resolver import gateway_backend, resolve_agent_to_manifest
from nemo_agent_hardener_plugin.cli.client import base_url
from nemo_agent_hardener_plugin.entities import AGENT_HARDENER_MANIFEST_TYPE
from nemo_agent_hardener_plugin.filesets import download_and_extract_project, upload_project_dir
from nemo_agent_hardener_plugin.jobs.errors import CATEGORY_FILESET, CATEGORY_MANIFEST, AgentHardenerRunError
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.entities.client import EntitiesClient
from nemo_platform_plugin.entities.types import EntityUpdate
from nemo_platform_plugin.job_context import JobContext

logger = logging.getLogger(__name__)

#: Where Relay reads system-policy plugin config inside the victim. Duplicated from
#: ``agent_hardener.openshell.relay_victim`` rather than imported: agent-hardener is deliberately not a
#: dependency of this plugin (its garak closure conflicts with the platform's).
_RELAY_PLUGINS_UPLOAD_DEST = "/etc/nemo-relay/plugins.toml"


# Attacker effort presets → garak knobs written into the manifest's top-level `garak:` block.
# "standard" is omitted so agent-hardener's own defaults apply.
INTENSITY_GARAK: dict[str, dict[str, int]] = {
    "light": {"generations": 1, "max_attempts_per_tool": 1},
    "thorough": {"generations": 5, "max_attempts_per_tool": 10},
}

# Defender override entries mirroring agent_hardener.manifest._default_defenders (name + implementation +
# capabilities — agent-hardener's SessionConfig validator requires a non-empty `capabilities`). Selecting a
# subset replaces the default defender list via the manifest's `overrides.defenders` (agent-hardener merges
# overrides with lists replacing).
#
# An entry's `config` is NOT inert: agent-hardener's defenders manager builds each defender's context as
# `{**enriched_context, **defender.config}`, so keys placed here arrive in its `DefenderInput.context`.
# That is how the user's `safety` model reaches the guardrails defender — see `_safety_llm_entries`.
# No entry declares a static `config`; it is attached per-run.
DEFENDER_ENTRIES: dict[str, dict[str, Any]] = {
    "openshell": {
        "name": "openshell-policy-defender",
        "implementation": "agent_hardener.agents.defenders.openshell_defender_v2.openshell_defender_agent:run",
        "timeout_seconds": 300,
        "capabilities": (
            "Mitigates attacks that exploit Linux kernel security controls: network egress, filesystem "
            "read/write access, process identity (UID/GID), seccomp syscall filtering, and Landlock path "
            "restrictions. Generates and repairs OpenShell policy YAML patches."
        ),
    },
    "guardrails": {
        "name": "defender-guardrails",
        "implementation": "agent_hardener.agents.defenders.guardrails_defender_v2.guardrails_defender_agent:run",
        "timeout_seconds": 300,
        "capabilities": (
            "Mitigates prompt injection, unsafe tool invocations, sensitive content disclosure, "
            "reconnaissance commands, and untrusted content handling through LLM-generated guardrail rules."
        ),
    },
}


def _safety_model(data: dict[str, Any]) -> str | None:
    """The user's chosen guardrail ("safety" group) model, if any."""
    safety = (data.get("models") or {}).get("safety")
    model = safety.get("model") if isinstance(safety, dict) else None
    return str(model) if model else None


def _with_safety_llm(entries: list[dict[str, Any]], safety_model: str) -> list[dict[str, Any]]:
    """Attach *safety_model* to the guardrails entry's ``config``, which becomes its DefenderInput context.

    Copies each entry rather than mutating it: ``DEFENDER_ENTRIES`` is module-level, so an in-place edit
    would leak one run's choice into every later run in the same process.
    """
    guardrails_name = DEFENDER_ENTRIES["guardrails"]["name"]
    return [
        {**entry, "config": {**entry.get("config", {}), "safety_llm": safety_model}}
        if entry["name"] == guardrails_name
        else entry
        for entry in entries
    ]


def _apply_manifest_overrides(manifest: dict[str, Any], data: dict[str, Any]) -> None:
    """Re-apply the manifest's persisted war-game overrides (attacker intensity + defender selection).

    The run rebuilds the thin manifest from the agent ref, so these choices — like the victim port —
    must be re-injected here, as agent-hardener's native top-level ``garak:`` block and ``overrides.defenders``
    list. An empty defender selection leaves agent-hardener's defaults untouched.
    """
    garak = INTENSITY_GARAK.get(str(data.get("attack_intensity") or "standard"))
    if garak:
        manifest["garak"] = garak
    # Apply an explicit victim port only when set (stored on the manifest or a per-run override); otherwise
    # leave the port the agent resolver derived from the running deployment.
    if data.get("port"):
        manifest.setdefault("agent", {})["port"] = int(data["port"])
    safety_model = _safety_model(data)
    enabled = [key for key in (data.get("defenders") or []) if key in DEFENDER_ENTRIES]
    if not enabled:
        # No selection means agent-hardener's own defender defaults, which carry no `config` for us to
        # attach to — say so rather than dropping the choice quietly.
        if safety_model:
            logger.warning(
                "safety model %r not applied: no defender selection to attach it to (agent-hardener's "
                "defaults are in force). Enable the guardrails defender to use it.",
                safety_model,
            )
        return
    # No workflow gate: the guardrail is Relay plugin config Agent Hardener owns, not a file the agent has
    # to contain, so it applies to every victim. Gating on `agent.workflow` (a NAT-era leftover) meant
    # selecting defenders in Studio silently dropped this one and scored 0 blocked, while selecting
    # none — which falls through to agent-hardener's defaults — hardened normally.
    entries = [DEFENDER_ENTRIES[key] for key in enabled]
    if safety_model:
        if any(entry["name"] == DEFENDER_ENTRIES["guardrails"]["name"] for entry in entries):
            entries = _with_safety_llm(entries, safety_model)
        else:
            logger.warning(
                "safety model %r not applied: the guardrails defender is %s.",
                safety_model,
                "not enabled" if "guardrails" not in enabled else "unavailable (the agent has no workflow)",
            )
    if entries:
        manifest.setdefault("overrides", {})["defenders"] = entries


def _materialize_manifest(
    sdk: Any, manifest_id: str, ctx: JobContext, config_overrides: dict[str, Any] | None = None
) -> str:
    """Materialize a saved manifest into an on-host ``agent-hardener.yaml``; return its path.

    Both sources store their victim project as a fileset, so materializing is one path: download the
    bundle, load the stored manifest, repoint the paths that only exist on this host. An agent-source
    manifest predating that (no ``agent_fileset``) re-resolves once and stores a bundle as it goes.

    ``config_overrides`` (per-run port/defenders/attack_intensity from the launch dialog) is overlaid
    onto the stored config so the run can deviate from the manifest without persisting the change.
    """
    if sdk is None:
        raise AgentHardenerRunError(
            CATEGORY_MANIFEST, "running a saved manifest requires the platform SDK (submit the job, don't run locally)."
        )
    record = (
        client_from_platform(sdk, EntitiesClient)
        .get_entity_by_name(name=manifest_id, entity_type=AGENT_HARDENER_MANIFEST_TYPE, workspace=ctx.workspace)
        .data()
    )
    data = {**(getattr(record, "data", {}) or {}), **(config_overrides or {})}
    manifest_dir = ctx.storage.persistent
    manifest_dir.mkdir(parents=True, exist_ok=True)
    is_project = (data.get("source_type") or "agent") == "project"
    bundle = data.get("project_fileset") if is_project else data.get("agent_fileset")
    if bundle:
        manifest = _materialize_from_bundle(sdk, manifest_id, data, manifest_dir, bundle)
    elif is_project:
        raise AgentHardenerRunError(
            CATEGORY_MANIFEST, f"project manifest {manifest_id!r} is missing its project_fileset."
        )
    else:
        manifest = _materialize_legacy_agent_manifest(sdk, manifest_id, data, ctx, manifest_dir, record)
    manifest_path = manifest_dir / "agent-hardener.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return str(manifest_path)


def _materialize_from_bundle(
    sdk: Any, manifest_id: str, data: dict[str, Any], manifest_dir: Path, bundle: str
) -> dict[str, Any]:
    """Restore a manifest's stored victim bundle and repoint the parts that are host-specific.

    The single materialization path for both sources. The stored ``manifest_yaml`` is authoritative —
    nothing is re-derived from the agent — so what runs is what was frozen, and a setting cannot be
    silently lost between runs. Only three keys are rewritten, because they describe *this* host and
    *this* platform rather than the target: where the project landed, where its secrets were written,
    and the current Inference-Gateway route.
    """
    manifest_yaml = data.get("manifest_yaml")
    if not manifest_yaml:
        raise AgentHardenerRunError(CATEGORY_MANIFEST, f"manifest {manifest_id!r} has no manifest_yaml to restore.")
    try:
        project_dir = download_and_extract_project(sdk, bundle, manifest_dir)
    except AgentHardenerRunError:
        raise
    except Exception as exc:  # a fileset download/extract failure is a distinct, actionable class
        raise AgentHardenerRunError(
            CATEGORY_FILESET, f"could not download or unpack the victim bundle for manifest {manifest_id!r}: {exc}"
        ) from exc

    manifest = yaml.safe_load(manifest_yaml) or {}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("agent"), dict):
        raise AgentHardenerRunError(CATEGORY_MANIFEST, f"manifest {manifest_id!r} has malformed manifest_yaml.")
    agent = manifest["agent"]
    agent["project_dir"] = str(project_dir)
    # The bundle's own .env only carries what the project shipped; the victim's secrets (incl. operator-
    # provided ones like a host-backend URL) are materialized next to the manifest. Point secrets_file at
    # that absolute path so agent-hardener's credential provider reads them (a relative ".env" would resolve
    # against the task cwd, where no dotenv exists, and silently deliver nothing).
    agent["secrets_file"] = str((manifest_dir / ".env").resolve())
    # The gateway route is a property of the platform we are running on, not of the frozen target, so
    # a manifest created against a platform that has since moved still reaches the current one.
    # `backends` is top-level on agent-hardener's AgentManifest, not part of AgentSpec — nesting it under
    # `agent` fails validation with extra_forbidden before the victim ever starts.
    gw_backend = gateway_backend(base_url())
    if gw_backend:
        others = [b for b in (manifest.get("backends") or []) if b.get("name") != gw_backend.get("name")]
        manifest["backends"] = [*others, gw_backend]
    # Edits made after freezing (PATCH /manifests) have to reach the run; the stored YAML is the base.
    if data.get("egress"):
        agent["egress"] = list(data["egress"])
    if data.get("secrets"):
        agent["secrets"] = list(data["secrets"])
    if data.get("env"):
        agent["env"] = {**(agent.get("env") or {}), **data["env"]}
    _apply_manifest_overrides(manifest, data)
    return manifest


def _materialize_legacy_agent_manifest(
    sdk: Any, manifest_id: str, data: dict[str, Any], ctx: JobContext, manifest_dir: Path, record: Any
) -> dict[str, Any]:
    """Re-resolve a manifest saved before targets were frozen, then freeze it so this runs once.

    Upgrading here rather than in a migration keeps existing manifests working with no user action:
    the first run after the upgrade behaves exactly as before, and every run after it is frozen.
    """
    agent_ref = data.get("agent")
    if not agent_ref:
        raise AgentHardenerRunError(
            CATEGORY_MANIFEST, f"manifest {manifest_id!r} has no agent reference to materialize from."
        )
    logger.info("manifest %s predates frozen targets; re-resolving and storing a bundle", manifest_id)
    resolved = resolve_agent_to_manifest(
        agent_ref,
        sdk=sdk,
        base_url=base_url(),
        default_workspace=ctx.workspace,
        manifest_dir=manifest_dir,
        egress=data.get("egress") or None,
        secrets=data.get("secrets") or None,
    )
    _persist_upgraded_bundle(sdk, manifest_id, ctx, record, resolved)
    _apply_manifest_overrides(resolved.manifest, data)
    for warning in resolved.warnings:
        logger.warning("manifest %s: %s", manifest_id, warning)
    return resolved.manifest


def _persist_upgraded_bundle(sdk: Any, manifest_id: str, ctx: JobContext, record: Any, resolved: Any) -> None:
    """Store the freshly-resolved scaffold on the manifest; best-effort, the run proceeds regardless."""
    try:
        fileset = upload_project_dir(sdk, resolved.project_dir, workspace=ctx.workspace)
        updated = {**(getattr(record, "data", {}) or {})}
        updated["agent_fileset"] = fileset
        updated["manifest_yaml"] = yaml.safe_dump(resolved.manifest, sort_keys=False)
        client_from_platform(sdk, EntitiesClient).update_entity_by_name(
            name=manifest_id,
            entity_type=AGENT_HARDENER_MANIFEST_TYPE,
            workspace=ctx.workspace,
            body=EntityUpdate(data=updated),
        )
    except Exception:  # the war-game matters more than the upgrade; it retries next run
        logger.warning("could not freeze manifest %s on this run", manifest_id, exc_info=True)


def _seed_validation_manifest(
    manifest_path: str, defense_guardrails: str | None, defense_policy: str | None, ctx: JobContext
) -> None:
    """Rewrite a materialized manifest for a frozen validate-only run: zero defenders + composed baseline.

    Seeds the user-chosen composed guardrail set as the victim's baseline plugins.toml and, when a policy
    was chosen, points the victim at the composed OpenShell policy. Forces ``overrides.defenders: []`` so
    agent-hardener runs no defender agents — it deploys this fixed baseline and only replays + scores (see the
    frozen-validation design). Attacks/benign come from ``--replay`` + the suite.
    """
    manifest_dir = ctx.storage.persistent
    data = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    overrides = data.setdefault("overrides", {})
    overrides["defenders"] = []  # zero defenders: deploy the frozen baseline, generate nothing
    if defense_guardrails:
        guardrails_file = manifest_dir / "composed-plugins.toml"
        guardrails_file.write_text(defense_guardrails, encoding="utf-8")
        # Three keys, because agent-hardener derives three things from the plugins path: what the defenders
        # base on (`target`), what seeds the run's init/ baseline (`storage`), and what is uploaded into
        # the victim (`uploads`). Overriding only one would validate a guardrail set the victim never ran.
        overrides.setdefault("target", {})["agent_relay_plugins"] = str(guardrails_file)
        overrides.setdefault("storage", {})["victim_relay_plugins_path"] = str(guardrails_file)
        config = overrides.setdefault("victim_control", {}).setdefault("config", {})
        config["uploads"] = [f"{guardrails_file}:{_RELAY_PLUGINS_UPLOAD_DEST}"]
    if defense_policy:
        policy_file = manifest_dir / "composed-policy.yaml"
        policy_file.write_text(defense_policy, encoding="utf-8")
        overrides.setdefault("victim_control", {}).setdefault("config", {})["policy_path"] = str(policy_file)
        overrides.setdefault("storage", {})["victim_policy_path"] = str(policy_file)
    Path(manifest_path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _manifest_facts(manifest_path: str) -> tuple[str, int]:
    """Best-effort read of (agent_name, port) from the manifest for the run record."""
    try:
        data = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
        agent = data.get("agent", {}) if isinstance(data, dict) else {}
        return str(agent.get("name", "")), int(agent.get("port", 0) or 0)
    except (OSError, ValueError, yaml.YAMLError):
        return "", 0
