# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for the Agent Hardener plugin.

Mounted on :class:`~nemo_platform.NeMoPlatform` as ``client.agent_hardener`` via the ``nemo.sdk``
entry-point. Exposes ``run(config=..., env_file=..., workspace=...)`` which executes the
``agent-hardener.war-game`` job locally, in-process, via
:meth:`~nemo_platform_plugin.scheduler.NemoJobScheduler.run_local` — mirroring the auditor
plugin's ``client.auditor.run`` — plus ``client.agent_hardener.runs`` to read run records.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence
from pathlib import Path

from nemo_agent_hardener_plugin.cli.client import base_url, make_sdk
from nemo_agent_hardener_plugin.filesets import upload_file_to_fileset
from nemo_agent_hardener_plugin.jobs.defenses import compose_defense
from nemo_agent_hardener_plugin.jobs.run import AgentHardenerRunJob
from nemo_agent_hardener_plugin.jobs.synth_benign import AgentHardenerSynthBenignJob
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.agent_hardener.client import AgentHardenerClient
from nemo_platform_plugin.agent_hardener.types import (
    AGENT_HARDENER_MANIFEST_TYPE,
    AGENT_HARDENER_RUN_TYPE,
    InspectProjectRequest,
    JsonMap,
    JsonValue,
    ManifestInit,
    ManifestUpdate,
    ValidateModelRequest,
    WarGameModels,
)
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.entities.client import EntitiesClient
from nemo_platform_plugin.entities.types import Entity, ListEntitiesQueryParams
from nemo_platform_plugin.scheduler import NemoJobScheduler
from nemo_platform_plugin.sdk import NemoPluginSDKResources
from pydantic import BaseModel, TypeAdapter

_JSON_MAP_ADAPTER = TypeAdapter(JsonMap)
ModelSelection = WarGameModels | dict[str, dict[str, str]]


def _to_json_map(value: object) -> JsonMap:
    return _JSON_MAP_ADAPTER.validate_python(value)


def _model_to_json_map(model: BaseModel) -> JsonMap:
    return _to_json_map(model.model_dump(mode="json"))


def _models_to_json_map(models: ModelSelection | None) -> JsonMap | None:
    if models is None:
        return None
    if isinstance(models, WarGameModels):
        return _model_to_json_map(models)
    return _model_to_json_map(WarGameModels.model_validate(models))


def _run_to_dict(entity: Entity) -> JsonMap:
    """Flatten an entity-store record into a flat dict for display."""
    data = _to_json_map(entity.data or {})
    data["name"] = entity.name
    created = entity.created_at
    if created is not None:
        data["created_at"] = str(created)
    return data


def _run_war_game(
    sync_sdk: NeMoPlatform,
    *,
    config: str | None,
    manifest_id: str | None,
    env_file: str | None,
    workspace: str,
    benign_suite: str | None,
    rounds: int | None = None,
    port: int | None = None,
    defenders: list[str] | None = None,
    attack_intensity: str | None = None,
    replay_hitlog_fileset: str | None = None,
    models: JsonMap | None = None,
) -> JsonMap:
    """Blocking war-game launch shared by the sync and async resources.

    ``run_local`` runs the job synchronously and the job downloads its filesets (benign suite, replay
    hitlog, materialized manifest) through the sync ``sdk``, so both entry points funnel through this
    one sync body — the async twin just runs it on a worker thread with a sync client it builds.

    Pass a local ``config`` manifest path or a saved ``manifest_id`` (which materializes the manifest and
    reuses its cached benign suite). Exactly one is required.

    The per-run overrides are applied over the materialized manifest without persisting it, mirroring
    Studio's launch dialog: a run may deviate from the saved baseline without editing the frozen target.
    ``port``/``defenders``/``attack_intensity`` need a manifest to overlay, so they are rejected
    alongside ``config`` rather than silently dropped; ``rounds`` is an ``agent-hardener run`` argument and
    applies to both paths.
    """
    if not (config or manifest_id):
        raise ValueError("agent-hardener run requires a 'config' manifest path or a 'manifest_id'.")
    overlay = {"port": port, "defenders": defenders, "attack_intensity": attack_intensity}
    unsupported = sorted(key for key, value in overlay.items() if value is not None)
    if config and unsupported:
        raise ValueError(
            f"{', '.join(unsupported)} cannot be combined with a local 'config' manifest — "
            "set them in the manifest's own `overrides:` block instead."
        )
    spec: dict[str, JsonValue] = {"config": config, "manifest_id": manifest_id, "env_file": env_file}
    spec.update({key: value for key, value in overlay.items() if value is not None})
    if rounds is not None:
        spec["rounds"] = rounds
    if replay_hitlog_fileset:
        spec["replay_hitlog_fileset"] = replay_hitlog_fileset
    if models:
        spec["models"] = models
    if benign_suite:
        spec["benign_suite_fileset"] = upload_file_to_fileset(
            sync_sdk, Path(benign_suite), workspace=workspace, prefix="benign-suite"
        )
    return _to_json_map(NemoJobScheduler().run_local(AgentHardenerRunJob, spec, workspace=workspace, sdk=sync_sdk))


def _run_synth_benign(
    sync_sdk: NeMoPlatform, *, manifest_id: str, env_file: str | None, interview: str, workspace: str
) -> JsonMap:
    """Blocking benign-suite synthesis for a saved manifest, shared by the sync and async resources.

    Materializes the manifest, runs native ``agent-hardener synth-benign`` (TTY interview), and caches the
    reviewed suite on the manifest entity through the sync ``sdk``.
    """
    spec: dict[str, JsonValue] = {"manifest_id": manifest_id, "env_file": env_file, "interview": interview}
    return _to_json_map(
        NemoJobScheduler().run_local(AgentHardenerSynthBenignJob, spec, workspace=workspace, sdk=sync_sdk)
    )


def _list_newest(entities: EntitiesClient, entity_type: str, *, workspace: str, limit: int) -> list[JsonMap]:
    """Return at most *limit* records of *entity_type*, newest first.

    ``entities.list`` returns a ``SyncDefaultPagination`` whose ``__iter__`` auto-paginates, so
    ``page_size`` bounds the *page*, not the total — iterating it walks the entire history. We ask for
    one page of *limit* and take only that page's items, which is a single request.
    """
    page = entities.list_entities(
        entity_type=entity_type,
        workspace=workspace,
        query_params=ListEntitiesQueryParams(sort="-created_at", page_size=limit),
    ).page()
    return [_run_to_dict(item) for item in itertools.islice(page.items, limit)]


class _RunsResource:
    """``client.agent_hardener.runs`` — read AgentHardenerRun records from the entity store."""

    def __init__(self, platform: NeMoPlatform) -> None:
        self._platform = platform

    def list(self, *, workspace: str = "default", limit: int = 20) -> Sequence[JsonMap]:
        return _list_newest(
            client_from_platform(self._platform, EntitiesClient),
            AGENT_HARDENER_RUN_TYPE,
            workspace=workspace,
            limit=limit,
        )

    def latest(self, *, workspace: str = "default") -> JsonMap | None:
        runs = self.list(workspace=workspace, limit=1)
        return runs[0] if runs else None


class _ManifestsResource:
    """``client.agent_hardener.manifests`` — saved AgentHardenerManifest records.

    Reads go through the entity store; writes go through the plugin's own API so the CLI and Studio
    share one implementation of manifest creation (resolution, persistence, validation).
    """

    def __init__(self, platform: NeMoPlatform) -> None:
        self._platform = platform

    def _client(self) -> AgentHardenerClient:
        return client_from_platform(self._platform, AgentHardenerClient)

    def list(self, *, workspace: str = "default", limit: int = 20) -> Sequence[JsonMap]:
        return _list_newest(
            client_from_platform(self._platform, EntitiesClient),
            AGENT_HARDENER_MANIFEST_TYPE,
            workspace=workspace,
            limit=limit,
        )

    def get(self, name: str, *, workspace: str = "default") -> JsonMap:
        """Read one saved manifest (``GET /manifests/{name}``)."""
        return _model_to_json_map(self._client().get_manifest(name=name, workspace=workspace).data())

    def validate_model(self, *, workspace: str = "default", **body: object) -> JsonMap:
        """Probe a model choice (``POST /model-config/validate``); *body* is a ``ValidateModelRequest``.

        Backs Studio's "Test connection" button and the CLI's set-time model preflight.
        """
        request = ValidateModelRequest.model_validate(body)
        return _model_to_json_map(self._client().validate_model(workspace=workspace, body=request).data())

    def create(self, *, workspace: str = "default", **body: object) -> JsonMap:
        """Create a manifest (``POST /manifests``); *body* is a ``ManifestInit``."""
        request = ManifestInit.model_validate(body)
        return _model_to_json_map(self._client().create_manifest(workspace=workspace, body=request).data())

    def update(self, name: str, *, workspace: str = "default", **body: object) -> JsonMap:
        """Edit a saved manifest (``PATCH /manifests/{name}``); *body* is a ``ManifestUpdate``."""
        request = ManifestUpdate.model_validate(body)
        return _model_to_json_map(self._client().update_manifest(name=name, workspace=workspace, body=request).data())

    def refresh(self, name: str, *, workspace: str = "default") -> JsonMap:
        """Re-resolve a frozen agent-source manifest against the agent as it is now."""
        return _model_to_json_map(self._client().refresh_manifest(name=name, workspace=workspace).data())

    def inspect_project(
        self, project_fileset: str, *, dockerfile: str | None = None, workspace: str = "default"
    ) -> JsonMap:
        """Read an uploaded project bundle (``POST /manifests/inspect-project``).

        Returns the derived manifest fields plus ``unresolved`` — the fields the project cannot state
        about itself, which the caller must supply.
        """
        request = InspectProjectRequest(project_fileset=project_fileset, dockerfile=dockerfile)
        return _model_to_json_map(self._client().inspect_project(workspace=workspace, body=request).data())


class AgentHardenerPluginResource:
    """Sync SDK namespace mounted as ``client.agent_hardener``."""

    def __init__(self, platform: NeMoPlatform) -> None:
        self._platform = platform
        self._runs: _RunsResource | None = None
        self._manifests: _ManifestsResource | None = None

    @property
    def runs(self) -> _RunsResource:
        if self._runs is None:
            self._runs = _RunsResource(self._platform)
        return self._runs

    @property
    def manifests(self) -> _ManifestsResource:
        if self._manifests is None:
            self._manifests = _ManifestsResource(self._platform)
        return self._manifests

    def run(
        self,
        *,
        config: str | None = None,
        manifest_id: str | None = None,
        env_file: str | None = None,
        workspace: str | None = None,
        benign_suite: str | None = None,
        rounds: int | None = None,
        port: int | None = None,
        defenders: list[str] | None = None,
        attack_intensity: str | None = None,
        replay_hitlog_fileset: str | None = None,
        models: ModelSelection | None = None,
    ) -> JsonMap:
        """Run the war-game locally against a local ``config`` manifest or a saved ``manifest_id``.

        A saved ``manifest_id`` materializes the manifest and reuses its cached benign suite (from a prior
        ``synth_benign``). ``benign_suite`` is a local CSV (tool,payload,label,rationale,persona) uploaded
        as a fileset and passed to agent-hardener via ``--benign-suite``, overriding the cached suite.

        ``rounds``/``port``/``defenders``/``attack_intensity``/``replay_hitlog_fileset`` are per-run
        overrides applied without persisting them on the manifest.
        """
        return _run_war_game(
            self._platform,
            config=config,
            manifest_id=manifest_id,
            env_file=env_file,
            workspace=workspace or "default",
            benign_suite=benign_suite,
            rounds=rounds,
            port=port,
            defenders=defenders,
            attack_intensity=attack_intensity,
            replay_hitlog_fileset=replay_hitlog_fileset,
            models=_models_to_json_map(models),
        )

    def synth_benign(
        self,
        *,
        manifest_id: str,
        env_file: str | None = None,
        interview: str = "interactive",
        workspace: str | None = None,
    ) -> JsonMap:
        """Synthesize a saved manifest's benign suite and cache it on the manifest.

        Shells out to native ``agent-hardener synth-benign`` (its own TTY interview). ``interview`` is
        ``"interactive"`` (prompt), ``"auto"`` (accept recommended defaults), or ``"skip"`` (rules-only).
        """
        return _run_synth_benign(
            self._platform,
            manifest_id=manifest_id,
            env_file=env_file,
            interview=interview,
            workspace=workspace or "default",
        )

    def submit(
        self,
        *,
        manifest_id: str | None = None,
        config: str | None = None,
        env_file: str | None = None,
        driver: str | None = None,
        workspace: str | None = None,
        profile: str | None = None,
    ) -> JsonMap:
        """Submit the war-game to the platform executor (remote-capable path Studio uses).

        Pass a saved ``manifest_id`` (Studio) or a ready ``config`` path. ``driver="service"`` selects
        the Studio-driven interview/review HITL; omit it for the one-shot run.
        """
        spec: dict[str, JsonValue] = {
            "manifest_id": manifest_id,
            "config": config,
            "env_file": env_file,
            "driver": driver,
        }
        return _to_json_map(
            NemoJobScheduler().submit_remote(
                AgentHardenerRunJob, spec, base_url=base_url(), workspace=workspace or "default", profile=profile
            )
        )

    def sanity_check(
        self,
        *,
        manifest_id: str,
        mitigations: JsonMap,
        selected_defense_ids: list[str],
        replay_hitlog_fileset: str,
        env_file: str | None = None,
        workspace: str | None = None,
        profile: str | None = None,
    ) -> JsonMap:
        """Submit a validate-only war-game: freeze the chosen defenses and replay the recorded attacks + benign.

        Composes the selected subset of the run's recommended defenses (guardrails + policy) into the victim's
        frozen baseline, disables the mitigation-generating defenders, and replays ``replay_hitlog_fileset``
        against it — measuring which attacks are now blocked and which benign requests are wrongly blocked. The
        produced ``validation`` job result holds the per-item verdicts.
        """
        guardrails_toml, policy_yaml = compose_defense(mitigations, selected_defense_ids)
        spec: dict[str, JsonValue] = {
            "manifest_id": manifest_id,
            "driver": "service",
            "validate_only": True,
            "replay_hitlog_fileset": replay_hitlog_fileset,
            "env_file": env_file,
            "defense_guardrails": guardrails_toml,
            "defense_policy": policy_yaml,
        }
        return _to_json_map(
            NemoJobScheduler().submit_remote(
                AgentHardenerRunJob, spec, base_url=base_url(), workspace=workspace or "default", profile=profile
            )
        )


class AsyncAgentHardenerPluginResource:
    """Async SDK namespace mounted as ``client.agent_hardener``."""

    def __init__(self, platform: AsyncNeMoPlatform) -> None:
        self._platform = platform

    async def run(
        self,
        *,
        config: str | None = None,
        manifest_id: str | None = None,
        env_file: str | None = None,
        workspace: str | None = None,
        benign_suite: str | None = None,
        rounds: int | None = None,
        port: int | None = None,
        defenders: list[str] | None = None,
        attack_intensity: str | None = None,
        replay_hitlog_fileset: str | None = None,
        models: ModelSelection | None = None,
    ) -> JsonMap:
        """Async twin of :meth:`AgentHardenerPluginResource.run`.

        ``run_local`` and the job it drives are synchronous and reach the platform through a *sync*
        client (fileset uploads/downloads, manifest materialization). We build one targeting the same
        base URL as the injected async client and run the whole blocking flow on a worker thread so the
        caller's event loop stays free. Auth mirrors the CLI's direct-mode ``make_sdk`` (fine for the
        local-platform path agent-hardener runs against).
        """
        sync_sdk = make_sdk(str(self._platform.base_url))
        return await asyncio.to_thread(
            _run_war_game,
            sync_sdk,
            config=config,
            manifest_id=manifest_id,
            env_file=env_file,
            workspace=workspace or "default",
            benign_suite=benign_suite,
            rounds=rounds,
            port=port,
            defenders=defenders,
            attack_intensity=attack_intensity,
            replay_hitlog_fileset=replay_hitlog_fileset,
            models=_models_to_json_map(models),
        )

    async def synth_benign(
        self,
        *,
        manifest_id: str,
        env_file: str | None = None,
        interview: str = "interactive",
        workspace: str | None = None,
    ) -> JsonMap:
        """Async twin of :meth:`AgentHardenerPluginResource.synth_benign` (runs the blocking flow off-loop)."""
        sync_sdk = make_sdk(str(self._platform.base_url))
        return await asyncio.to_thread(
            _run_synth_benign,
            sync_sdk,
            manifest_id=manifest_id,
            env_file=env_file,
            interview=interview,
            workspace=workspace or "default",
        )


agent_hardener_sdk_resources = NemoPluginSDKResources(
    sync_resource=AgentHardenerPluginResource,
    async_resource=AsyncAgentHardenerPluginResource,
)
