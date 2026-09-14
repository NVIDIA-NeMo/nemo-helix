# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``agent-hardener.war-game`` job — registered under ``nemo.jobs``.

Orchestrates one attack/defend/validate war-game against a deployed NAT agent by shelling out to
agent-hardener's own CLI (its own venv; agent-hardener is never imported). This module holds only the job
class: ``compile`` builds the platform job spec (pre-creating the run record for Studio's live view)
and ``run`` sequences the phases. The mechanics live in sibling modules:
:mod:`~nemo_agent_hardener_plugin.jobs.manifest` (materialize/seed the on-host manifest),
:mod:`~nemo_agent_hardener_plugin.jobs.records` (entity-store rows),
:mod:`~nemo_agent_hardener_plugin.jobs.artifacts` (results + filesets), and
:mod:`~nemo_agent_hardener_plugin.jobs.execution` (the subprocess invocation paths).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, cast

from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import _common, benign_suite
from nemo_agent_hardener_plugin.jobs.artifacts import (
    _replay_args,
    _save_composed_guardrails,
    _save_events_fileset,
    _save_hitlog_fileset,
    _save_mitigations,
    _save_validation,
    _uploaded_benign_suite,
)
from nemo_agent_hardener_plugin.jobs.errors import (
    CATEGORY_MODEL_UNAVAILABLE,
    AgentHardenerRunError,
    RunFailure,
    classify_exception,
)
from nemo_agent_hardener_plugin.jobs.execution import _run_one_shot, _run_service_driven
from nemo_agent_hardener_plugin.jobs.manifest import _manifest_facts, _materialize_manifest, _seed_validation_manifest
from nemo_agent_hardener_plugin.jobs.records import (
    _cached_benign_suite,
    _create_run,
    _manifest_models,
    _manifest_rounds,
    _precreate_run,
    _run_data,
    _run_facts,
    _update_run,
)
from nemo_agent_hardener_plugin.jobs.spec import WarGameSpec
from nemo_agent_hardener_plugin.model_config import (
    ANALYSIS_DEFAULT_BASE_URL,
    ATTACK_DEFAULT_BASE_URL,
    ModelChoice,
    WarGameModels,
)
from nemo_agent_hardener_plugin.model_preflight import validate_choice
from nemo_platform_plugin.entity_client import NemoEntitiesClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.api_factory import (
    EnvironmentVariable,
    PlatformJobSpec,
    PlatformJobStep,
    SubprocessExecutionProviderSpec,
)
from nemo_platform_plugin.jobs.constants import DEFAULT_JOB_STORAGE_PATH, PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_LOG_TAIL = 4000


def _merge_choice(default: ModelChoice | None, override: ModelChoice | None) -> ModelChoice | None:
    """Field-level merge of one model group: the per-run override wins per field, else the stored default."""
    if default is None and override is None:
        return None
    default = default or ModelChoice()
    override = override or ModelChoice()
    merged = ModelChoice(
        model=override.model or default.model,
        base_url=override.base_url or default.base_url,
        api_key_secret=override.api_key_secret or default.api_key_secret,
    )
    return merged if (merged.model or merged.base_url or merged.api_key_secret) else None


def _effective_models(sdk: Any, config: dict, ctx: JobContext) -> WarGameModels | None:
    """Resolve the run's effective model selection: the manifest's stored default merged with the override.

    Reads the stored default from the manifest record (Studio path) and merges the per-run ``models`` from
    the spec over it, field by field. Returns ``None`` when neither side selects anything (so agent-hardener's
    built-in defaults stay in force and nothing is injected).
    """
    stored_raw = _manifest_models(sdk, str(config["manifest_id"]), ctx) if config.get("manifest_id") else {}
    # `agent` was a group before the victim's LLM was ruled out of scope (it lives in the target's own
    # workflow). Stored values validate away silently, so say why they stopped applying.
    for source in (stored_raw, config.get("models") or {}):
        if isinstance(source, dict) and source.get("agent"):
            logger.warning(
                "the 'agent' model group was removed — the victim's LLM is set in its workflow, not by the "
                "war-game. Ignoring it; use the 'safety' group for the guardrail model."
            )
            break
    stored = WarGameModels.model_validate(stored_raw)
    override = WarGameModels.model_validate(config.get("models") or {})
    merged = WarGameModels(
        attack=_merge_choice(stored.attack, override.attack),
        analysis=_merge_choice(stored.analysis, override.analysis),
        safety=_merge_choice(stored.safety, override.safety),
    )
    return merged if (merged.attack or merged.analysis or merged.safety) else None


def _preflight_models(models: WarGameModels | None, *, sdk: Any, workspace: str, default_key: str | None) -> None:
    """Fail fast (before the sandbox spins up) if a user-chosen model/endpoint/key can't be reached.

    Only groups the user explicitly configured (a model name and/or a custom ``base_url``) are probed —
    the built-in defaults are known-good and left untouched. On a bad credential or a wrong model name we
    raise a classified :class:`AgentHardenerRunError` whose message lists the models those credentials *can*
    reach, so the user can correct the choice instead of guessing.

    The guardrail ("safety") group is probed against :data:`ATTACK_DEFAULT_BASE_URL` because that is the
    endpoint agent-hardener's ``guardrails_defender_v2.nodes.yaml_writer._ensure_safety_llm`` hardcodes when it
    writes ``llms.safety_llm`` — with ``${INFERENCE_API_KEY}``, the same provisioned key ``default_key`` holds.
    Worth probing precisely because its failure is silent: an unreachable safety model does not error, it
    hangs the guardrails defender until its 300s timeout, which reads as "ran, proposed no change".
    """
    if models is None:
        return
    groups = (
        ("attack", models.attack, ATTACK_DEFAULT_BASE_URL),
        ("analysis", models.analysis, ANALYSIS_DEFAULT_BASE_URL),
        ("safety", models.safety, ATTACK_DEFAULT_BASE_URL),
    )
    for label, choice, default_base_url in groups:
        if choice is None or not (choice.model or choice.base_url):
            continue
        key = _common._resolve_secret(sdk, choice.api_key_secret, workspace) if choice.api_key_secret else default_key
        verdict = validate_choice(choice.model, choice.base_url or default_base_url, key)
        if verdict.ok:
            continue
        raise AgentHardenerRunError(CATEGORY_MODEL_UNAVAILABLE, _preflight_message(label, choice, verdict))


def _preflight_message(label: str, choice: ModelChoice, verdict: Any) -> str:
    """Compose the operator-facing message for a failed model preflight (lists reachable models)."""
    endpoint = choice.base_url or "the default endpoint"
    if verdict.reason == "auth":
        return f"The {label} model credentials were rejected by {endpoint} ({verdict.detail or 'unauthorized'})."
    if verdict.reason == "unreachable":
        return f"Could not reach the {label} model endpoint {endpoint} ({verdict.detail or 'no response'})."
    if verdict.reason == "provider_error":
        return (
            f"The {label} model endpoint {endpoint} returned an error ({verdict.detail or 'unknown'}); "
            "the credentials were accepted, so this is the provider's side — retry shortly."
        )
    available = ", ".join(verdict.available[:20]) or "none"
    return (
        f"The {label} model {choice.model!r} is not available at {endpoint}. "
        f"Models reachable with these credentials: {available}."
    )


class AgentHardenerRunJob(NemoJob):
    """Run the attack/defend/validate war-game against the configured agent."""

    name = "war-game"  # CLI: `nemo agent-hardener war-game ...`; keeps `run` free for the wrapper command
    description = "Run the Agent Hardener war-game against a deployed NAT agent."
    container = "cpu-tasks"
    spec_schema: ClassVar[type[BaseModel] | None] = WarGameSpec

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,  # WarGameSpec
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        """Single subprocess step running the war-game on the host where `nemo agent-hardener setup` provisioned it.

        Subprocess (not container) executor: the war-game shells out to agent-hardener's CLI + garak venv and
        launches the Docker victim sandbox, all of which live on the provisioned host today. A Docker-capable
        container image (`CPUExecutionProviderSpec(container=...)`) is the Phase-2 swap — `run()` is unchanged.
        """
        war_game = WarGameSpec.model_validate(spec)

        # Pre-create the run record now (a Studio war-game submits a manifest_id + service driver) so the UI
        # can open its live view immediately; the worker reuses this record via `run_name` in the step config.
        run_name: str | None = None
        if war_game.driver == "service" and war_game.manifest_id and job_name and not war_game.stop_after_synth:
            run_name = await _precreate_run(
                cast(NemoEntitiesClient, entity_client),
                workspace=workspace,
                manifest_id=war_game.manifest_id,
                job_id=job_name,
                source_run=war_game.source_run or "",
            )

        environment = [
            EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH),
        ]
        # The subprocess executor forwards only PATH/VIRTUAL_ENV, but the war-game's openshell sandbox reads
        # its gateway registration from $HOME/.config/openshell and reaches Docker via $DOCKER_HOST. Forward
        # them explicitly from the provisioned host this subprocess runs on (see the executor note above).
        for name in ("HOME", "DOCKER_HOST", "XDG_CONFIG_HOME"):
            value = os.environ.get(name)
            if value:
                environment.append(EnvironmentVariable(name=name, value=value))
        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="war-game",
                    executor=SubprocessExecutionProviderSpec(
                        provider="subprocess",
                        command=["python", "-m", "nemo_agent_hardener_plugin.tasks.war_game"],
                    ),
                    config={**war_game.model_dump(mode="json"), **({"run_name": run_name} if run_name else {})},
                    environment=environment,
                ),
            ],
        )

    def run(self, config: dict, *, ctx: JobContext, sdk: Any = None, **_: Any) -> dict:
        """Run the war-game, classifying and surfacing any failure that affects the run's results.

        The whole run is wrapped in one error boundary: a classified :class:`AgentHardenerRunError` (or any
        other exception) is turned into a :class:`RunFailure`, recorded on the run entity (so the
        pre-created ``running`` row is finalized to ``failed`` with a cause, never orphaned) and logged,
        then re-surfaced as a ``failed`` result so the process exits non-zero and the platform job errors.
        """
        try:
            return self._execute(config, ctx=ctx, sdk=sdk)
        except Exception as exc:
            failure = classify_exception(exc)
            logger.exception("agent-hardener war-game failed [%s]: %s", failure.category, failure.message)
            self._record_failure(ctx, sdk, config, failure)
            return {
                "status": "failed",
                "returncode": 1,
                "error": {
                    "category": failure.category,
                    "message": failure.message,
                    "remediation": failure.remediation,
                },
            }

    def _record_failure(self, ctx: JobContext, sdk: Any, config: dict, failure: RunFailure) -> None:
        """Finalize the run record as ``failed`` with the classified error, on every channel the user sees.

        Reuses the pre-created record (``run_name``) when present so its live view resolves to the failure
        instead of a perpetual ``running``; otherwise creates a failed record now. Also reports terminal
        ``failed`` progress with the error details. Recording stays best-effort — it must not mask the cause.

        The update replaces the whole record, so the pre-created ``agent``/``port`` are read back and
        carried forward — otherwise a failure would blank the agent the run was targeting.
        """
        prepared = config.get("run_name")
        agent, port = _run_facts(sdk, workspace=ctx.workspace, name=str(prepared)) if prepared else ("", 0)
        data = _run_data(
            agent,
            port or int(config.get("port") or 0),
            str(config.get("config") or ""),
            "failed",
            1,
            ctx.job_id or "",
            manifest_id=str(config.get("manifest_id") or ""),
            source_run=str(config.get("source_run") or ""),
            failure=failure,
        )
        if prepared:
            _update_run(sdk, workspace=ctx.workspace, name=str(prepared), data=data)
        else:
            _create_run(sdk, workspace=ctx.workspace, data=data)
        self.report_progress(ctx, work_done=0, work_total=1, status="failed", details=failure.as_error_details())

    def _execute(self, config: dict, *, ctx: JobContext, sdk: Any = None) -> dict:
        plugin_config = AgentHardenerConfig.get()
        _common.require_provisioned(plugin_config)

        # Studio submits a saved manifest_id (materialized here from the stored agent ref); the CLI
        # passes a ready manifest path via `config`.
        manifest_id: str | None = None
        cached_suite: list[dict[str, str]] = []
        rounds = 1
        # Per-run config overrides from the launch dialog: apply over the manifest without persisting.
        config_overrides = {
            key: config[key] for key in ("port", "defenders", "attack_intensity") if config.get(key) is not None
        }
        # Effective model selection = the manifest's stored default merged with the per-run override. Drives
        # both the subprocess env knobs (attack/analysis groups) and the guardrail model, which travels in
        # the manifest as the guardrails defender's config (safety group). Resolved once here so materialize
        # + env stay consistent.
        models = _effective_models(sdk, config, ctx)
        if models is not None:
            config_overrides["models"] = models.model_dump(mode="json")
        model_env = _common.build_model_env(models, sdk=sdk, workspace=ctx.workspace)
        # Preflight user-chosen models against their endpoint before the (minutes-long) sandbox spin-up, so
        # a wrong model name / key fails in seconds with the list of models the credentials can actually reach.
        _preflight_models(
            models,
            sdk=sdk,
            workspace=ctx.workspace,
            # Same resolver the validate endpoint uses, so a run is never preflighted against a
            # different credential than it runs with.
            default_key=_common.resolve_model_key(sdk, None, workspace=ctx.workspace),
        )
        validate_only = bool(config.get("validate_only"))
        if config.get("manifest_id"):
            manifest_id = str(config["manifest_id"])
            manifest = _materialize_manifest(sdk, manifest_id, ctx, config_overrides)
            cached_suite = _cached_benign_suite(sdk, manifest_id, ctx)
            rounds = int(config["rounds"]) if config.get("rounds") else _manifest_rounds(sdk, manifest_id, ctx)
        elif config.get("config"):
            manifest = str(config["config"])
            # No manifest record to read a stored default from, so the spec is the only source. Resolved
            # here rather than left at 1: rounds is an `agent-hardener run` argument, not a manifest field.
            rounds = int(config["rounds"]) if config.get("rounds") else 1
        else:
            raise ValueError(
                "agent-hardener war-game requires a 'manifest_id' or a 'config' manifest path in the spec."
            )

        # Frozen sanity check: seed the chosen composed defenses as the victim baseline and force zero
        # defenders, so the replay measures the fixed defense without generating new mitigations. Always a
        # single round (validation, not iterative hardening).
        if validate_only:
            _seed_validation_manifest(manifest, config.get("defense_guardrails"), config.get("defense_policy"), ctx)
            rounds = 1
        # Studio submits no env_file; agent-hardener reads victim creds from a project dotenv, so synthesize
        # one from the operator env (which carries the provisioned INFERENCE_API_KEY) for the manifest's secrets.
        env_file = config.get("env_file")
        if not env_file:
            env_file = _common.materialize_victim_env_file(
                manifest, _common.build_subprocess_env(plugin_config), Path(manifest).parent
            )
        agent_name, port = _manifest_facts(manifest)

        # Replay mode: skip the live garak attack and replay a recorded hitlog (uploaded, or a prior run's
        # saved hitlog) against the defended victim. The fileset is downloaded here and passed as `--replay <path>`.
        replay_fileset = config.get("replay_hitlog_fileset") or None
        replay_args = _replay_args(replay_fileset, sdk, ctx)

        # Benign suite: an uploaded suite (if supplied) overrides the manifest's cached suite for this run.
        benign_override = _uploaded_benign_suite(config.get("benign_suite_fileset") or None, sdk, ctx)

        # `driver: "service"` (Studio) drives the interview/review HITL via the serve service; otherwise the
        # default one-shot `agent-hardener run` (TTY interview when interactive).
        if config.get("driver") == "service":
            outcome = _run_service_driven(
                manifest,
                env_file,
                plugin_config,
                ctx,
                sdk,
                agent_name,
                port,
                manifest_id=manifest_id,
                cached_suite=cached_suite,
                stop_after_synth=bool(config.get("stop_after_synth")),
                prepared_run_name=config.get("run_name") or None,
                rounds=rounds,
                replay_args=replay_args,
                benign_suite_override=benign_override,
                source_run=str(config.get("source_run") or ""),
                model_env=model_env,
            )
        else:
            # One-shot `agent-hardener run` consumes a suite; it never synthesizes. Prefer an uploaded override,
            # else fall back to the manifest's cached suite (from a prior `synth-benign`), written to a CSV.
            suite_for_run = benign_override
            if suite_for_run is None and cached_suite:
                suite_csv = ctx.storage.persistent / "benign-suite.csv"
                benign_suite.write_suite(suite_csv, cached_suite)
                suite_for_run = str(suite_csv)
            outcome = _run_one_shot(
                manifest,
                env_file,
                plugin_config,
                ctx,
                replay_args,
                benign_suite=suite_for_run,
                model_env=model_env,
                rounds=rounds,
            )

        # A validate-only run generates no mitigations (defenders: []); it produces the sanity-check
        # validation.json instead. A normal hardening run produces the mitigations artifact.
        if validate_only:
            _save_validation(ctx)
            # Also persist the exact composed workflow that was validated, so the Harden tab can recover it
            # after a reload and keep "Apply to Agent" enabled without re-running the check.
            _save_composed_guardrails(ctx, config.get("defense_guardrails"))
        elif not config.get("stop_after_synth"):
            _save_mitigations(ctx)

        # Record the run's garak hitlog so a later run (e.g. the Harden-tab sanity check) can replay it;
        # per-job storage doesn't survive across runs. A live attack produces a new hitlog we persist to a
        # fileset; a replay produces no new hits but carries forward the hitlog it replayed, so replay runs
        # stay sanity-checkable too. Generation-only runs (`--stop-after-synth`) have no attack hits.
        hitlog_fileset = ""
        if config.get("stop_after_synth"):
            pass
        elif replay_args:
            hitlog_fileset = replay_fileset or ""
        else:
            hitlog_fileset = _save_hitlog_fileset(sdk, ctx, ctx.workspace)

        events_fileset = _save_events_fileset(sdk, workspace=ctx.workspace, run_name=outcome.record_name or "")

        # Finalize the up-front record (service path) or create one now (one-shot). Preserve source_run from
        # the config so a validate-only sanity check stays linked to its harden run (the update replaces the
        # whole record, which would otherwise drop the pre-created link).
        data = _run_data(
            agent_name,
            port,
            manifest,
            outcome.status,
            outcome.returncode,
            ctx.job_id or "",
            hitlog_fileset,
            manifest_id or "",
            source_run=str(config.get("source_run") or ""),
            failure=outcome.failure,
            events_fileset=events_fileset,
        )
        if outcome.record_name:
            _update_run(sdk, workspace=ctx.workspace, name=outcome.record_name, data=data)
            record_name = outcome.record_name
        else:
            record_name = _create_run(sdk, workspace=ctx.workspace, data=data)

        details = {"returncode": str(outcome.returncode)}
        if outcome.failure is not None:
            details.update(outcome.failure.as_error_details())
        self.report_progress(ctx, work_done=1, work_total=1, status=outcome.status, details=details)
        result = {
            "status": outcome.status,
            "returncode": outcome.returncode,
            "log_tail": outcome.log_text[-_LOG_TAIL:],
            "results": {"agent-hardener-log": outcome.log_ref.model_dump()} if outcome.log_ref else {},
            "run_record": record_name,
        }
        if outcome.failure is not None:  # a subprocess-classified failure surfaces its cause here too
            result["error"] = {
                "category": outcome.failure.category,
                "message": outcome.failure.message,
                "remediation": outcome.failure.remediation,
            }
        return result
