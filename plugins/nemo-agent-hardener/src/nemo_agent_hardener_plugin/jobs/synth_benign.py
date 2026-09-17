# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``agent-hardener.synth`` job — synthesize a saved manifest's benign suite and cache it.

The single entry point for benign-suite synthesis, selected by ``driver``:

- ``native`` (CLI): shell out to native ``agent-hardener synth-benign`` (its own TTY interview), run locally.
- ``service`` (Studio): drive ``agent-hardener serve`` + the interview/review HITL over the platform job's
  ``status_details`` — the exact serve path the war-game uses, via
  :func:`~nemo_agent_hardener_plugin.jobs.execution._run_service_driven` with ``stop_after_synth=True``.

Both converge on :func:`~nemo_agent_hardener_plugin.jobs.records.read_and_persist_suite`, caching the reviewed
suite on the manifest entity.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, cast

from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import _common
from nemo_agent_hardener_plugin.jobs.artifacts import _save_events_fileset
from nemo_agent_hardener_plugin.jobs.errors import classify_exception
from nemo_agent_hardener_plugin.jobs.execution import RunOutcome, _run_service_driven, run_synth_benign
from nemo_agent_hardener_plugin.jobs.manifest import _manifest_facts, _materialize_manifest
from nemo_agent_hardener_plugin.jobs.records import _create_run, _run_data, _update_run, read_and_persist_suite
from nemo_agent_hardener_plugin.jobs.run import _effective_models
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


class SynthBenignSpec(BaseModel):
    """Inputs for the benign-suite synthesis phase (the shape ``run()``/``compile()`` see)."""

    manifest_id: str
    driver: str = "native"  # "native" (CLI TTY) | "service" (Studio serve HITL over status_details)
    env_file: str | None = None
    # Interview mode for the native driver: "interactive" (TTY prompts), "auto" (--yes), "skip" (--no-interactive).
    interview: str = "interactive"
    # Reused pre-created run record name (service driver); unused for native.
    run_name: str | None = None
    source_run: str | None = None


class AgentHardenerSynthBenignJob(NemoJob):
    """Synthesize and cache the benign request suite for a saved manifest (native TTY or Studio serve HITL)."""

    name = "synth"  # keeps the hand-written `nemo agent-hardener synth-benign` command unshadowed (cf. war-game/run)
    description = "Synthesize a saved manifest's benign request suite and cache it on the manifest."
    container = "cpu-tasks"
    spec_schema: ClassVar[type[BaseModel] | None] = SynthBenignSpec

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,  # SynthBenignSpec
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        """A single subprocess step running the synth task on the provisioned host (mirrors the war-game).

        The ``service`` driver's HITL is relayed through the platform job's ``status_details``, so it must run
        as a submitted job (``ctx.job_id`` set). The run record is created at runtime by ``_run_service_driven``
        (as the war-game ``stop_after_synth`` path does), so no pre-creation is needed here.
        """
        del workspace, entity_client, job_name, async_sdk, profile, options
        synth = cast(SynthBenignSpec, spec)
        environment = [EnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value=DEFAULT_JOB_STORAGE_PATH)]
        # The subprocess executor forwards only PATH/VIRTUAL_ENV; the sandbox reads its gateway registration
        # from $HOME/.config/openshell and reaches Docker via $DOCKER_HOST (same as the war-game step).
        for name in ("HOME", "DOCKER_HOST", "XDG_CONFIG_HOME"):
            value = os.environ.get(name)
            if value:
                environment.append(EnvironmentVariable(name=name, value=value))
        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="synth",
                    executor=SubprocessExecutionProviderSpec(
                        provider="subprocess",
                        command=["python", "-m", "nemo_agent_hardener_plugin.tasks.synth_benign"],
                    ),
                    config=synth.model_dump(mode="json"),
                    environment=environment,
                ),
            ],
        )

    def run(self, config: dict, *, ctx: JobContext, sdk: Any = None, **_: Any) -> dict:
        """Run synthesis, classifying any failure into an operator-facing error result."""
        try:
            return self._execute(config, ctx=ctx, sdk=sdk)
        except Exception as exc:
            failure = classify_exception(exc)
            logger.exception("agent-hardener synth-benign failed [%s]: %s", failure.category, failure.message)
            return {
                "status": "failed",
                "returncode": 1,
                "error": {
                    "category": failure.category,
                    "message": failure.message,
                    "remediation": failure.remediation,
                },
            }

    def _execute(self, config: dict, *, ctx: JobContext, sdk: Any = None) -> dict:
        plugin_config = AgentHardenerConfig.get()
        _common.require_provisioned(plugin_config)

        manifest_id = str(config["manifest_id"])
        manifest = _materialize_manifest(sdk, manifest_id, ctx)
        env = _common.build_subprocess_env(plugin_config)
        # Synthesis probes the live victim, so it needs the manifest's declared secrets. When no --env-file is
        # supplied (Studio never sends one), synthesize one from the operator env (as the war-game does).
        env_file = config.get("env_file")
        if not env_file:
            env_file = _common.materialize_victim_env_file(manifest, env, Path(manifest).parent)
        _common.check_victim_secrets(manifest, env, env_file)

        if config.get("driver") == "service":
            return self._run_service(config, ctx, sdk, plugin_config, manifest, manifest_id, env_file)

        csv_path = run_synth_benign(
            plugin_config.agent_hardener_bin,
            manifest,
            env_file,
            env,
            ctx,
            interview=str(config.get("interview") or "interactive"),
        )
        suite = read_and_persist_suite(sdk, ctx, manifest_id, csv_path)
        self.report_progress(
            ctx, work_done=1, work_total=1, status="completed", details={"suite_size": str(len(suite))}
        )
        return {"status": "completed", "returncode": 0, "manifest_id": manifest_id, "suite_size": len(suite)}

    def _run_service(
        self,
        config: dict,
        ctx: JobContext,
        sdk: Any,
        plugin_config: AgentHardenerConfig,
        manifest: str,
        manifest_id: str,
        env_file: str | None,
    ) -> dict:
        """Studio path: drive the serve interview/review HITL, persist the suite, finalize the run record."""
        models = _effective_models(sdk, config, ctx)
        model_env = _common.build_model_env(models, sdk=sdk, workspace=ctx.workspace)
        agent, port = _manifest_facts(manifest)
        outcome = _run_service_driven(
            manifest,
            env_file,
            plugin_config,
            ctx,
            sdk,
            agent,
            port,
            manifest_id=manifest_id,
            stop_after_synth=True,
            prepared_run_name=config.get("run_name") or None,
            source_run=str(config.get("source_run") or ""),
            model_env=model_env,
        )
        self._finalize_run(sdk, ctx, config, manifest, agent, port, manifest_id, outcome)
        return {
            "status": outcome.status,
            "returncode": outcome.returncode,
            "manifest_id": manifest_id,
            "run_record": outcome.record_name,
        }

    def _finalize_run(
        self,
        sdk: Any,
        ctx: JobContext,
        config: dict,
        manifest: str,
        agent: str,
        port: int,
        manifest_id: str,
        outcome: RunOutcome,
    ) -> None:
        """Finalize the run record ``_run_service_driven`` left as ``running`` (synth-only: no hitlog)."""
        events_fileset = _save_events_fileset(sdk, workspace=ctx.workspace, run_name=outcome.record_name or "")
        data = _run_data(
            agent,
            port,
            manifest,
            outcome.status,
            outcome.returncode,
            ctx.job_id or "",
            manifest_id=manifest_id,
            source_run=str(config.get("source_run") or ""),
            failure=outcome.failure,
            events_fileset=events_fileset,
        )
        if outcome.record_name:
            _update_run(sdk, workspace=ctx.workspace, name=outcome.record_name, data=data)
        else:
            _create_run(sdk, workspace=ctx.workspace, data=data)
        details = {"returncode": str(outcome.returncode)}
        if outcome.failure is not None:
            details.update(outcome.failure.as_error_details())
        self.report_progress(ctx, work_done=1, work_total=1, status=outcome.status, details=details)
