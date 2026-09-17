# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routes over the ``AgentHardenerManifest`` entity — named, reusable war-game targets.

Mounted at ``/apis/agent-hardener/v2/workspaces/{workspace}``. ``POST /manifests`` runs `init` and persists a
named record the operator later selects to run against; list/get/delete mirror the runs routes.

Both sources store the victim project as a fileset the run re-downloads, so a manifest is a frozen
target rather than a query re-evaluated per run: ``agent`` resolves a deployed agent and stores the
scaffold it produced, ``project`` builds from an uploaded NAT project via ``agent-hardener init --yes``
(``POST /manifests/inspect`` detects its layout first). Changes to the agent reach an existing
manifest only via ``POST /manifests/{name}/refresh`` — which ``apply-mitigation`` calls for you.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_agent_hardener_plugin._perms import AgentHardenerManifestPerms
from nemo_agent_hardener_plugin.agent_resolver import (
    AgentResolutionError,
    ResolvedManifest,
    inspect_agent,
    resolve_agent_to_manifest,
)
from nemo_agent_hardener_plugin.api.v2._filters import make_filter_dep
from nemo_agent_hardener_plugin.api.v2.schemas import (
    InspectAgentRequest,
    InspectAgentResponse,
    InspectProjectRequest,
    InspectProjectResponse,
    ManifestFilter,
    ManifestInit,
    ManifestUpdate,
    ValidateModelRequest,
    ValidateModelResponse,
)
from nemo_agent_hardener_plugin.authz import scope
from nemo_agent_hardener_plugin.cli.client import base_url
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.entities import AgentHardenerManifest
from nemo_agent_hardener_plugin.filesets import delete_fileset, download_and_extract_project, upload_project_dir
from nemo_agent_hardener_plugin.jobs._common import resolve_model_key
from nemo_agent_hardener_plugin.model_config import ModelConfigDefaults, WarGameModels, model_config_defaults
from nemo_agent_hardener_plugin.model_preflight import validate_choice
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_platform_plugin.jobs.openapi_utils import generate_openapi_extra_params
from nemo_platform_plugin.log_utils import sanitize_for_log
from nemo_platform_plugin.sdk_provider import get_platform_sdk
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

router = APIRouter()


class _SubprocessError(Exception):
    """A non-zero ``agent-hardener inspect``/``init`` exit (carries the stderr tail for the API detail)."""


# These run inside a request, on a threadpool worker. Unbounded, a wedged subprocess pins its worker
# for the process's lifetime and enough of them starve the pool — so every call gets a ceiling.
_SUBPROCESS_TIMEOUT_SECONDS = 120


class _SubprocessTimeout(Exception):
    """``agent-hardener inspect``/``init`` exceeded :data:`_SUBPROCESS_TIMEOUT_SECONDS`."""


def _run_agent_hardener(cmd: list[str], cwd: str, action: str) -> subprocess.CompletedProcess[str]:
    """Run an ``agent-hardener`` subcommand with a bounded runtime, raising on timeout or non-zero exit."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, check=False, timeout=_SUBPROCESS_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as exc:
        raise _SubprocessTimeout(f"{action} timed out after {_SUBPROCESS_TIMEOUT_SECONDS}s.") from exc
    if result.returncode != 0:
        raise _SubprocessError((result.stderr or result.stdout).strip()[-500:] or f"{action} returned no output.")
    return result


_manifest_filter_dep = make_filter_dep(ManifestFilter)


@router.get(
    "/manifests",
    tags=["Agent Hardener Manifests"],
    openapi_extra=generate_openapi_extra_params(filter_schema=ManifestFilter),
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.LIST])
async def list_manifests(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: ManifestFilter = Depends(_manifest_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> dict:
    """List saved manifests in the workspace, with pagination and an ``agent``/``source_type`` filter."""
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    try:
        result = await entity_client.list(
            AgentHardenerManifest,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_dict or None,
        )
    except Exception as exc:
        logger.exception("Failed to list agent-hardener manifests in workspace '%s'", sanitize_for_log(workspace))
        raise HTTPException(status_code=500, detail="Failed to list agent-hardener manifests.") from exc
    return {
        "data": [manifest.model_dump(mode="json") for manifest in result.data],
        "pagination": result.pagination.model_dump() if result.pagination else None,
        "sort": sort,
        "filter": filter or None,
    }


@router.get("/manifests/{name}", response_model=AgentHardenerManifest, tags=["Agent Hardener Manifests"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.READ])
async def get_manifest(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentHardenerManifest:
    """Get a single manifest by name."""
    try:
        return await entity_client.get(AgentHardenerManifest, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"AgentHardenerManifest '{name}' not found in workspace '{workspace}'."
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get agent-hardener manifest '%s'", sanitize_for_log(name))
        raise HTTPException(status_code=500, detail="Failed to get agent-hardener manifest.") from exc


@router.get("/model-config-defaults", response_model=ModelConfigDefaults, tags=["Agent Hardener Manifests"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.INSPECT])
async def get_model_config_defaults(workspace: str) -> ModelConfigDefaults:
    """The built-in per-group model defaults (attack/analysis) the create/run forms pre-fill."""
    return model_config_defaults()


@router.post("/model-config/validate", response_model=ValidateModelResponse, tags=["Agent Hardener Manifests"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.INSPECT])
async def validate_model_config(workspace: str, body: ValidateModelRequest) -> ValidateModelResponse:
    """Probe a model choice's endpoint/key (the "Test connection" affordance) and list reachable models.

    Resolves the chosen Secret to its value (if any) and lists ``{base_url}/models``. Never leaks the key —
    only the boolean verdict + the reachable model ids come back, so the UI can offer real options.
    """
    sdk = get_platform_sdk(as_service="agent-hardener", internal=True)

    def _validate() -> ValidateModelResponse:
        # Falls back to the provisioned agent-hardener key when no Secret is named — the documented meaning of
        # a null `api_key_secret`. Probing with no key at all reported 401 for every model that a run would
        # in fact reach, which made this endpoint (and Studio's "Test connection") reject valid choices.
        api_key = resolve_model_key(sdk, body.api_key_secret, workspace=workspace)
        verdict = validate_choice(body.model, body.base_url, api_key)
        return ValidateModelResponse(
            ok=verdict.ok, reason=verdict.reason, available=verdict.available, detail=verdict.detail
        )

    return await run_in_threadpool(_validate)


@router.post("/manifests/inspect", response_model=InspectProjectResponse, tags=["Agent Hardener Manifests"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.INSPECT])
async def inspect_project(
    workspace: str,
    body: InspectProjectRequest,
) -> InspectProjectResponse:
    """Detect an uploaded NAT project's layout (`agent-hardener inspect`) to pre-fill the create wizard.

    Downloads the project bundle, expands it, and runs the read-only, offline detector — no code is
    executed. Returns the discovered workflows, launch mode, name, secrets, and egress as defaults.
    """
    sdk = get_platform_sdk(as_service="agent-hardener", internal=True)
    bin_path = AgentHardenerConfig.get().agent_hardener_bin

    def _inspect() -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = download_and_extract_project(sdk, body.project_fileset, Path(tmp))
            result = _run_agent_hardener(
                [str(bin_path), "inspect", "--project-dir", str(project_dir), "--json"],
                cwd=str(project_dir),
                action="inspect",
            )
            return json.loads(result.stdout)

    try:
        detected = await run_in_threadpool(_inspect)
    except _SubprocessTimeout as exc:
        raise HTTPException(status_code=504, detail=f"Failed to inspect project: {exc}") from exc
    except _SubprocessError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to inspect project: {exc}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the uploaded project: {exc}") from exc
    return InspectProjectResponse(**detected)


@router.post("/manifests/inspect-agent", response_model=InspectAgentResponse, tags=["Agent Hardener Manifests"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.INSPECT])
async def inspect_agent_endpoint(workspace: str, body: InspectAgentRequest) -> InspectAgentResponse:
    """Derive the deployed-agent create-form defaults (victim port + secret names) for pre-fill.

    Read-only: fetches the stored agent config and its running deployment; nothing is materialized.
    """
    sdk = get_platform_sdk(as_service="agent-hardener", internal=True)

    def _inspect() -> tuple[str, int, list[str], list[str]]:
        return inspect_agent(body.agent, sdk=sdk, default_workspace=workspace)

    try:
        ref, port, secrets, warnings = await run_in_threadpool(_inspect)
    except AgentResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return InspectAgentResponse(agent=ref, port=port, secrets=secrets, warnings=warnings)


@router.post("/manifests", response_model=AgentHardenerManifest, status_code=201, tags=["Agent Hardener Manifests"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.WRITE])
async def create_manifest(
    workspace: str,
    body: ManifestInit,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentHardenerManifest:
    """`init`: build a manifest (from a deployed agent or an uploaded project) and persist it by ``name``."""
    if body.source_type == "project":
        manifest = await _build_project_manifest(workspace, body)
    else:
        manifest = await _build_agent_manifest(workspace, body)
    try:
        return await entity_client.create(manifest)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409, detail=f"Manifest '{body.name}' already exists in workspace '{workspace}'."
        ) from exc
    except Exception as exc:
        logger.exception("Failed to persist agent-hardener manifest '%s'", sanitize_for_log(body.name))
        raise HTTPException(status_code=500, detail="Failed to create agent-hardener manifest.") from exc


async def _get_manifest_or_404(entity_client: NemoEntitiesClient, workspace: str, name: str) -> AgentHardenerManifest:
    """Fetch a manifest, turning a missing entity into a 404 (shared by PATCH, refresh and DELETE)."""
    try:
        return await entity_client.get(AgentHardenerManifest, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"AgentHardenerManifest '{name}' not found in workspace '{workspace}'."
        ) from exc


async def _resolve_and_store_scaffold(
    workspace: str,
    agent_ref: str,
    *,
    egress: list[str] | None,
    port: int | None,
    secrets: list[str] | None,
) -> tuple[ResolvedManifest, str]:
    """Resolve *agent_ref* and persist its scaffold as a fileset; return the resolution and the ref.

    Resolution *writes* an installable project (``scaffold_project`` + ``materialize_workflow``), so
    the scaffold is an artifact, not a by-product. Storing it is what makes a manifest a frozen
    target: the run downloads this instead of re-resolving, so nothing it depends on can be silently
    re-derived. Shared by create and refresh — the only two ways a scaffold is produced.
    """
    sdk = get_platform_sdk(as_service="agent-hardener", internal=True)

    # resolve_agent_to_manifest is sync + network-bound (sdk.agents.get), and so is the upload;
    # keep both off the event loop. The temp dir must outlive the upload, hence one closure.
    def _resolve_and_upload() -> tuple[ResolvedManifest, str]:
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_agent_to_manifest(
                agent_ref,
                sdk=sdk,
                base_url=base_url(),
                default_workspace=workspace,
                manifest_dir=Path(tmp),
                egress=egress,
                port=port,
                secrets=secrets,
            )
            return resolved, upload_project_dir(sdk, resolved.project_dir, workspace=workspace)

    try:
        return await run_in_threadpool(_resolve_and_upload)
    except AgentResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:  # empty/unreadable scaffold from upload_project_dir
        raise HTTPException(status_code=400, detail=f"Could not store the resolved scaffold: {exc}") from exc


async def _build_agent_manifest(workspace: str, body: ManifestInit) -> AgentHardenerManifest:
    """Resolve a deployed agent into a manifest and freeze its scaffold as a fileset."""
    if not body.agent:
        raise HTTPException(status_code=422, detail="source_type 'agent' requires an 'agent' reference.")

    resolved, fileset = await _resolve_and_store_scaffold(
        workspace, body.agent, egress=body.egress, port=body.port, secrets=body.secrets
    )

    manifest = AgentHardenerManifest.from_agent_resolution(
        name=body.name,
        workspace=workspace,
        agent_ref=f"{resolved.workspace}/{resolved.agent_name}",
        manifest_yaml=yaml.safe_dump(resolved.manifest, sort_keys=False),
        agent_fileset=fileset,
        port=resolved.port,
        secrets=resolved.secrets,
        egress=body.egress or [],  # persisted, not just used for the resolve above
        env=body.env or {},
        warnings=resolved.warnings,
        models=body.models or WarGameModels(),
    )
    # resolve_agent_to_manifest has no `env` parameter, so it is absent from the YAML it produced.
    manifest.manifest_yaml = _yaml_with_agent_settings(manifest.manifest_yaml, manifest)
    return manifest


def _validate_launch_mode(body: ManifestInit) -> None:
    """Reject a launch mode we can't build, before the bundle is downloaded."""
    if body.launch_mode and body.launch_mode not in ("workflow", "byo"):
        raise HTTPException(
            status_code=422, detail=f"launch_mode must be 'workflow' or 'byo'; got {body.launch_mode!r}."
        )
    if body.launch_mode == "byo" and not body.dockerfile and not body.manifest_yaml:
        raise HTTPException(
            status_code=422,
            detail="launch_mode 'byo' requires a 'dockerfile' (or a 'manifest_yaml' that already sets one).",
        )
    if body.dockerfile and not body.binaries:
        # agent-hardener's NatVictimSpec rejects BYO without them.
        raise HTTPException(
            status_code=422,
            detail="'dockerfile' requires 'binaries' — glob patterns scoping which processes may egress, "
            "e.g. ['/app/.venv/bin/**'].",
        )


async def _build_project_manifest(workspace: str, body: ManifestInit) -> AgentHardenerManifest:
    """Build a manifest from an uploaded NAT project by shelling ``agent-hardener init --yes``.

    The bundle is expanded to a temp dir and ``init`` runs there (so ``project_dir`` resolves to ``.``);
    the war-game re-downloads the bundle and repoints ``project_dir`` at the restored copy.
    """
    fileset = body.project_fileset
    if not fileset:
        raise HTTPException(status_code=422, detail="source_type 'project' requires a 'project_fileset'.")
    _validate_launch_mode(body)

    sdk = get_platform_sdk(as_service="agent-hardener", internal=True)
    bin_path = AgentHardenerConfig.get().agent_hardener_bin
    port = body.port or 8000

    def _init() -> str:
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = download_and_extract_project(sdk, fileset, Path(tmp))
            output = Path(tmp) / "agent-hardener.yaml"
            cmd = [
                str(bin_path),
                "init",
                "--yes",
                "--force",
                "--project-dir",
                ".",
                "--name",
                body.name,
                "--port",
                str(port),
                "-o",
                str(output),
            ]
            if body.workflow:
                cmd += ["--workflow", body.workflow]
            if body.dockerfile:
                # Same containment check as `secrets_file` below. The flag keeps the relative path:
                # the run re-materializes the manifest against a different directory.
                candidate = (project_dir / body.dockerfile).resolve()
                if not candidate.is_relative_to(project_dir.resolve()):
                    raise ValueError("'dockerfile' must be inside the uploaded project.")
                if not candidate.is_file():
                    raise ValueError(f"'dockerfile' not found in the uploaded project: {body.dockerfile}")
                cmd += ["--dockerfile", body.dockerfile]
                for glob in body.binaries or []:
                    cmd += ["--binary", glob]
            if body.secrets:
                cmd += ["--secrets", ",".join(body.secrets)]
            if body.secrets_file:
                # Client-supplied, and `init` reads it on the platform host: without this it could
                # name any readable file (e.g. /proc/self/environ) and fold it into the manifest.
                candidate = (project_dir / body.secrets_file).resolve()
                if not candidate.is_relative_to(project_dir.resolve()):
                    raise ValueError("'secrets_file' must be inside the uploaded project.")
                cmd += ["--secrets-file", str(candidate)]
            for host in body.egress or []:
                cmd += ["--egress", host]
            for spec in body.backends or []:
                cmd += ["--backend", spec]
            _run_agent_hardener(cmd, cwd=str(project_dir), action="init")
            return output.read_text(encoding="utf-8")

    if body.manifest_yaml:
        # The CLI already ran agent-hardener's interactive `init` at the operator's terminal; rebuilding
        # it here with `--yes` would silently discard the answers they gave.
        manifest_yaml = body.manifest_yaml
    else:
        try:
            manifest_yaml = await run_in_threadpool(_init)
        except _SubprocessTimeout as exc:
            raise HTTPException(status_code=504, detail=f"Failed to build manifest from project: {exc}") from exc
        except _SubprocessError as exc:
            raise HTTPException(status_code=400, detail=f"Failed to build manifest from project: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Could not read the uploaded project: {exc}") from exc

    # The persisted manifest can't hold the temp project path; the run repoints it. Force project_dir='.'.
    manifest_yaml = _with_project_dir_dot(manifest_yaml)
    agent_section = _agent_section(manifest_yaml)
    if not agent_section:
        raise HTTPException(
            status_code=422, detail="manifest_yaml has no 'agent' section; not an agent-hardener manifest."
        )

    # The manifest itself is what the run executes, so the entity's fields describe it rather than
    # the request — otherwise the two disagree whenever a client omits a field agent-hardener detected.
    manifest = AgentHardenerManifest(
        name=body.name,
        workspace=workspace,
        source_type="project",
        project_fileset=fileset,
        workflow=body.workflow or str(agent_section.get("workflow") or ""),
        # Derived, not defaulted: the CLI sends a pre-built manifest_yaml with no launch_mode, so
        # defaulting to "workflow" mislabels every BYO manifest it creates.
        launch_mode=body.launch_mode or ("byo" if agent_section.get("dockerfile") else "workflow"),
        dockerfile=body.dockerfile or str(agent_section.get("dockerfile") or ""),
        binaries=body.binaries or list(agent_section.get("binaries") or []),
        manifest_yaml=manifest_yaml,
        port=body.port or int(agent_section.get("port") or port),
        secrets=body.secrets or list(agent_section.get("secrets") or []),
        egress=body.egress or list(agent_section.get("egress") or []),
        env=body.env or dict(agent_section.get("env") or {}),
        models=body.models or WarGameModels(),
    )
    if manifest.launch_mode == "byo" and not manifest.workflow:
        # agent-hardener needs `workflow` or `start_command` to launch a BYO victim, and the platform
        # never sets start_command — so this manifest would raise at run time, not merely degrade.
        raise HTTPException(
            status_code=422,
            detail="A BYO image needs a workflow to serve: none was given or detected in the project. "
            "Pass 'workflow' (the image is how the environment is built; the workflow is what gets "
            "served and hardened).",
        )
    manifest.manifest_yaml = _yaml_with_agent_settings(manifest.manifest_yaml, manifest)
    return manifest


def _agent_section(manifest_yaml: str) -> dict[str, Any]:
    """Return the manifest's ``agent`` mapping, or ``{}`` if it is absent or the YAML is unparseable."""
    try:
        data = yaml.safe_load(manifest_yaml) or {}
    except yaml.YAMLError:
        return {}
    agent = data.get("agent") if isinstance(data, dict) else None
    return agent if isinstance(agent, dict) else {}


def _with_project_dir_dot(manifest_yaml: str) -> str:
    """Return *manifest_yaml* with ``agent.project_dir`` normalized to ``.`` (unchanged if unparseable)."""
    try:
        data = yaml.safe_load(manifest_yaml) or {}
    except yaml.YAMLError:
        return manifest_yaml
    if isinstance(data, dict) and isinstance(data.get("agent"), dict):
        data["agent"]["project_dir"] = "."
        return yaml.safe_dump(data, sort_keys=False)
    return manifest_yaml


def _yaml_with_agent_settings(manifest_yaml: str, manifest: AgentHardenerManifest) -> str:
    """Return *manifest_yaml* with the manifest's stored agent settings written into it.

    The run layers these on at materialization anyway, so this is not what makes them take effect —
    it is what makes the stored YAML *honest*. Without it the manifest we show (and that `init -o`
    writes) is the frozen base rather than what will actually run, so an operator who sets `env` sees
    no trace of it and reasonably concludes it was lost.

    Unparseable YAML is returned untouched: a display concern must never cost someone their manifest.
    """
    try:
        data = yaml.safe_load(manifest_yaml) or {}
    except yaml.YAMLError:
        return manifest_yaml
    if not (isinstance(data, dict) and isinstance(data.get("agent"), dict)):
        return manifest_yaml
    agent = data["agent"]
    if manifest.port:
        agent["port"] = manifest.port
    if manifest.egress:
        agent["egress"] = list(manifest.egress)
    if manifest.secrets:
        agent["secrets"] = list(manifest.secrets)
    if manifest.env:
        agent["env"] = dict(manifest.env)
    return yaml.safe_dump(data, sort_keys=False)


@router.patch("/manifests/{name}", response_model=AgentHardenerManifest, tags=["Agent Hardener Manifests"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.WRITE])
async def update_manifest(
    workspace: str,
    name: str,
    body: ManifestUpdate,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentHardenerManifest:
    """Edit a manifest's cached benign suite, victim port, egress, or war-game settings.

    The agent source itself is immutable — re-create the manifest to point at a different agent, or
    `POST /manifests/{name}/refresh` to re-resolve the one it already targets.
    """
    existing = await _get_manifest_or_404(entity_client, workspace, name)
    if body.benign_suite is not None:
        existing.benign_suite = body.benign_suite
    if body.defenders is not None:
        existing.defenders = body.defenders
    if body.attack_intensity is not None:
        existing.attack_intensity = body.attack_intensity
    if body.rounds is not None:
        existing.rounds = body.rounds
    if body.models is not None:
        existing.models = body.models
    if body.egress is not None:
        existing.egress = body.egress
    if body.env is not None:
        existing.env = body.env
    if body.port is not None:
        existing.port = body.port
    existing.manifest_yaml = _yaml_with_agent_settings(existing.manifest_yaml, existing)
    try:
        return await entity_client.update(existing)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"AgentHardenerManifest '{name}' not found in workspace '{workspace}'."
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(status_code=409, detail=f"Manifest '{name}' was modified concurrently.") from exc


@router.post("/manifests/{name}/refresh", response_model=AgentHardenerManifest, tags=["Agent Hardener Manifests"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.WRITE])
async def refresh_manifest(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentHardenerManifest:
    """Re-resolve an agent-source manifest against the agent as it is *now*.

    A manifest is a frozen target, so edits to the agent — a new model, an added tool, a redeploy —
    do not reach it on their own. This is how you take them, deliberately, when you want the next
    run to measure the current agent rather than the one you saved.

    Everything the operator chose is kept: egress, secrets, models, defenders, intensity, rounds, and
    the cached benign suite. Only the scaffold and its rendered manifest are rebuilt.
    """
    existing = await _get_manifest_or_404(entity_client, workspace, name)
    if existing.source_type != "agent" or not existing.agent:
        raise HTTPException(
            status_code=422,
            detail=f"manifest '{name}' has no agent source to refresh from; re-upload the project instead.",
        )

    resolved, fileset = await _resolve_and_store_scaffold(
        workspace,
        existing.agent,
        egress=existing.egress or None,
        port=existing.port or None,
        secrets=existing.secrets or None,
    )
    stale = existing.agent_fileset

    existing.agent_fileset = fileset
    existing.port = resolved.port
    existing.secrets = resolved.secrets
    existing.warnings = resolved.warnings
    existing.manifest_yaml = _yaml_with_agent_settings(yaml.safe_dump(resolved.manifest, sort_keys=False), existing)
    updated = await entity_client.update(existing)

    if stale and stale != fileset:
        await run_in_threadpool(delete_fileset, get_platform_sdk(as_service="agent-hardener", internal=True), stale)
    return updated


@router.delete("/manifests/{name}", status_code=204, tags=["Agent Hardener Manifests"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerManifestPerms.WRITE])
async def delete_manifest(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete a saved manifest by name, along with the victim bundle the service created for it."""
    existing = await _get_manifest_or_404(entity_client, workspace, name)
    try:
        await entity_client.delete(AgentHardenerManifest, name=name, workspace=workspace)
    except Exception as exc:
        logger.exception("Failed to delete agent-hardener manifest '%s'", sanitize_for_log(name))
        raise HTTPException(status_code=500, detail="Failed to delete agent-hardener manifest.") from exc

    # Only after the entity is gone: a bundle with no manifest is garbage, but a manifest whose
    # bundle we deleted early would be unrunnable if the delete above had failed.
    #
    # `agent_fileset` only — the service uploads that one itself. `project_fileset` is supplied by
    # the caller and nothing stops two manifests naming the same bundle, so deleting it here would
    # break the other one. The uploader owns it and removes it.
    sdk = get_platform_sdk(as_service="agent-hardener", internal=True)
    if existing.agent_fileset:
        await run_in_threadpool(delete_fileset, sdk, existing.agent_fileset)
