# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Fabric-backed agent-eval runtime.

``FabricAgentRuntime`` drives an agent harness (Codex, Hermes, ...) through the
NeMo Fabric Python SDK and adapts each normalized Fabric ``RunResult`` into an
:class:`AgentEvalTrial`. The harness is chosen by the supplied Fabric config's
``harness.adapter_id`` (never inferred from a model); an optional ``model`` slug
is applied as the config's default model, mirroring Fabric's own Harbor integration.

Per-task settings (workspace, model, trajectory capture) are composed directly onto
a copy of the supplied config via the SDK's config helpers (``model_copy`` +
``enable_relay`` + ``environment``). Fabric removed profile overlays in 0.1.0rc2 —
``FabricConfig`` rejects a ``profiles`` key and ``Fabric.run`` takes no ``profiles``
argument — so a run is described by exactly one complete typed config, and the
evaluator-owned per-task settings are authoritative simply by being applied last.

Every task runs in its own fresh workspace: the runtime seeds it from
``inputs['files']`` (a no-op when there are none), runs the harness in it (via
``environment.workspace``), and exposes its final file tree as ``workspace``
filesystem evidence, so workspace-reading metrics score a Fabric trial alongside
the ATIF trajectory. Any ``environment.workspace`` set in the supplied config is
overridden per task.

``nemo_fabric`` is an optional native dependency: its types are imported for
annotations under ``TYPE_CHECKING`` and the package is loaded lazily at runtime,
so this module stays importable without it.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from nemo_evaluator_sdk.agent_eval.runtimes.fabric import _common
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.hooks import FabricTaskRunHook, FabricTaskRunSession
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_receiver import OTLPReceiver
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer import (
    fold_exports,
    otlp_trace_path,
    register_trace_evidence,
    traces_dir,
)
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import (
    SKILL_MODE_CODEX_SKILLS_DIR,
    AgentSkill,
    SkillMode,
    SkillProvenance,
    SkillSet,
    install_skills,
    resolve_skill_mode,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    RunnerInfo,
    TrialMeasurements,
)
from nemo_evaluator_sdk.agent_eval.workspace_seeds import SEED_FILES_INPUT_KEY, seed_workspace
from nemo_evaluator_sdk.values.evidence import (
    CandidateEvidence,
    EvidenceDescriptor,
)
from pydantic import JsonValue

if TYPE_CHECKING:
    # Annotations use nemo_fabric's real types (single source of truth). nemo_fabric is an optional
    # native package not yet in our locked dependency set, so it is imported for typing only and
    # loaded lazily at runtime (see ``run_tasks``). Drop the ty:ignore once nemo-fabric is a
    # resolvable dependency and the checker can see it.
    from nemo_fabric import (  # ty: ignore[unresolved-import]
        Fabric,
        FabricConfig,
        RelayObservabilityConfig,
        RunOutput,
        RunResult,
    )

DEFAULT_FABRIC_TIMEOUT_S = 600
_RUNTIME_NAME = "fabric"
_MISSING_FABRIC_MSG = "FabricAgentRuntime requires the `nemo-fabric` package (native NeMo Fabric SDK)."
_MISSING_RELAY_MSG = (
    "FabricAgentRuntime trajectory capture requires the `nemo-relay` package "
    "(install `nemo-fabric[relay]`), or set capture_trajectory=False."
)

logger = logging.getLogger(__name__)

# Evidence-dir layout for trajectory capture. These subdir names are our own local layout — we create
# them and hand them to Fabric/Relay, so they are not derived from either library.
_RELAY_SUBDIR = "relay"
_ARTIFACTS_SUBDIR = "artifacts"
# Per-task workspace: where seed files are staged and where the harness reads/writes. We
# create it, point Fabric's ``environment.workspace`` at it, and expose it as ``workspace`` evidence.
_WORKSPACE_SUBDIR = "workspace"
# Per-task skill staging dir (native injection): the skill's files are resolved here and the staged
# root is added to the task config's ``skills.paths``. For codex self-injection the skill lands in the
# workspace instead (no path added).
_SKILL_SUBDIR = "skill"
# Sentinel skill path attached only to probe Fabric's capability planner for the selected adapter's
# skills routing (see ``_resolve_skill_mode``). Never staged and need not exist on disk — the planner
# just reports how it would route a skill for this adapter.
_SKILL_PROBE_PATH = "nemo-eval-skill-capability-probe"
# Evidence key + descriptor kind for the staged workspace, consumed by the
# workspace-reading metrics.
_WORKSPACE_EVIDENCE_KEY = "workspace"
_WORKSPACE_EVIDENCE_KIND = "filesystem"
# File-exporter output names we choose for the Relay ATIF/ATOF trajectory (Relay accepts these as inputs).
# ``kind`` Fabric stamps on the promoted Relay ATIF artifact; used to surface it as trace evidence.
_ATIF_ARTIFACT_KIND = "atif"


class FabricAgentRuntime:
    """AgentTaskRunner that generates trials by running tasks through NeMo Fabric.

    The harness is selected entirely by ``config["harness"]["adapter_id"]``. Across harnesses the
    config shape differs mainly in that ``adapter_id``, optional input/output schemas, and any
    harness-specific ``harness.settings``. Fabric owns each adapter's execution mechanism. See
    ``examples/fabric_harness_runtimes.py`` for full Codex and Hermes config examples.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        model: str | None = None,
        base_dir: str | Path | None = None,
        work_root: str | Path | None = None,
        timeout_s: int = DEFAULT_FABRIC_TIMEOUT_S,
        capture_trajectory: bool = True,
        trajectory_extra: Mapping[str, Any] | None = None,
        runtime_name: str = _RUNTIME_NAME,
        skills: Sequence[AgentSkill] | None = None,
        task_hook: FabricTaskRunHook | None = None,
    ) -> None:
        self._config = config
        self._model = model
        self._base_dir = Path(base_dir).expanduser() if base_dir is not None else None
        self._work_root = Path(work_root).expanduser() if work_root is not None else None
        self._timeout_s = timeout_s
        self._capture_trajectory = capture_trajectory
        self._trajectory_extra = dict(trajectory_extra) if trajectory_extra else None
        self._runtime_name = runtime_name
        self._skill_set = SkillSet(tuple(skills or ()))
        self._task_hook = task_hook

    def with_skills(self, skills: Sequence[AgentSkill]) -> FabricAgentRuntime:
        """Return a copy of this runtime with ``skills`` *added* to its skill set; ``self`` is not modified.

        Additive and chainable: ``rt.with_skills([a]).with_skills([b])`` injects both a and b. Lets an A/B
        eval derive a treated runtime from a skill-free baseline (``baseline.with_skills(the_skills)``) so
        the two arms differ in exactly the injected skills. Skill names must be unique across the combined
        set — two bundles claiming the same ``<name>/`` would collide — so re-adding a skill already
        present raises. A shallow copy suffices — the shared fields are immutable config/paths.
        """
        clone = copy.copy(self)
        clone._skill_set = self._skill_set.with_skills(skills)
        return clone

    def with_skill(self, skill: AgentSkill) -> FabricAgentRuntime:
        """Return a copy of this runtime with ``skill`` *added*; ``self`` is not modified.

        Thin wrapper over :meth:`with_skills` for the common single-skill case; equally chainable
        (``rt.with_skill(a).with_skill(b)`` injects both).
        """
        return self.with_skills([skill])

    def _adapter_id(self) -> str:
        """Harness adapter selected by the Fabric config (empty when unset)."""
        harness = self._config.get("harness") if isinstance(self._config, Mapping) else None
        adapter_id = harness.get("adapter_id") if isinstance(harness, Mapping) else None
        return str(adapter_id) if adapter_id is not None else ""

    def _effective_model(self) -> str | None:
        """The model a run will actually use, mirroring :meth:`_compose_config`'s precedence.

        ``_compose_config`` only overwrites the config's default model when ``self._model`` is set, so
        a model supplied purely through ``config`` is what runs. Reporting ``self._model`` alone would
        record ``None`` for those runs, giving two runs with *different* models identical provenance —
        the one thing this metadata exists to prevent.
        """
        if self._model:
            return self._model
        models = self._config.get("models") if isinstance(self._config, Mapping) else None
        default = models.get("default") if isinstance(models, Mapping) else None
        model = default.get("model") if isinstance(default, Mapping) else getattr(default, "model", None)
        return str(model) if model is not None else None

    def runner_info(self) -> RunnerInfo:
        """Identify this runner and the Fabric settings that shape its results."""
        return RunnerInfo(
            name=self._runtime_name,
            kind="runner",
            config={
                "model": self._effective_model(),
                "timeout_s": self._timeout_s,
                "adapter_id": self._adapter_id(),
                "skills": [skill.name for skill in self._skill_set.skills],
                # Off means no relay/ATIF exporter, so the run captures no trajectory evidence — a
                # metric that scores trajectories sees something different.
                "capture_trajectory": self._capture_trajectory,
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> Sequence[AgentEvalTrial]:
        try:
            # nemo_fabric ships a native (pyo3) core and is an optional dependency, so it is imported
            # lazily here rather than at module load.
            from nemo_fabric import Fabric, FabricConfig  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise RuntimeError(_MISSING_FABRIC_MSG) from exc

        resolved_config = config or AgentEvalRunConfig()
        # Assign a run id once per run so two runs (e.g. an A/B baseline vs. skilled variant) written
        # under the same work_root/output_dir land in distinct, non-colliding evidence trees. Callers
        # that set run_id keep their identifier.
        if resolved_config.run_id is None:
            resolved_config = resolved_config.model_copy(update={"run_id": _new_run_id()})
        agent_config = FabricConfig.from_mapping(self._config)
        # Fail fast (once) if trajectory capture is requested but the nemo-relay gateway isn't
        # importable, rather than failing every task the same way inside the per-task guard.
        if self._capture_trajectory:
            try:
                import nemo_relay.observability  # noqa: F401  # ty: ignore[unresolved-import]
            except ImportError as exc:
                raise RuntimeError(_MISSING_RELAY_MSG) from exc
        # ``Fabric`` (formerly ``FabricClient``) is a lightweight, reusable facade — not a lifecycle
        # context manager — so it is created once and reused across tasks with no cleanup.
        client = Fabric()

        # Resolve once how a skill would reach this harness (the adapter is constant across tasks) by
        # asking Fabric's own capability planner, so any adapter that declares native skills support — ours
        # or an end-user's — is picked up automatically instead of via a hardcoded allow-list. Fail fast
        # rather than silently run a skill-free trial mislabeled as "with skill", which would corrupt an
        # A/B comparison. Only touched when a skill is set, so the no-skill path is unaffected.
        skill_mode: SkillMode | None = None
        if self._skill_set.skills:
            skill_mode = self._resolve_skill_mode(client, agent_config)
            if skill_mode is None:
                adapter_id = agent_config.harness.adapter_id
                raise RuntimeError(
                    f"FabricAgentRuntime received one or more skills but adapter {adapter_id!r} has no known "
                    "skill-injection strategy: Fabric does not route skills to it natively and it is not a "
                    "codex harness. Use a skills-native or codex harness, or drop the skills."
                )

        semaphore = asyncio.Semaphore(resolved_config.parallelism)

        async def run_one(index: int, task: AgentEvalTask) -> AgentEvalTrial:
            async with semaphore:
                return await self._run_task(client, agent_config, index, task, resolved_config, skill_mode)

        return await asyncio.gather(*(run_one(index, task) for index, task in enumerate(tasks)))

    def _resolve_skill_mode(self, client: Fabric, agent_config: FabricConfig) -> SkillMode | None:
        """Ask Fabric how a skill would reach the selected harness, or ``None`` if it can't.

        Probes Fabric's capability planner: plan a copy of the config with a sentinel skill path attached
        (it need not exist on disk) and read how the adapter routes skills. Querying the authoritative
        source at runtime means adapters that declare native skills support — ours or an end-user's — are
        detected without a hardcoded list. See :func:`~...skills.resolve_skill_mode`.
        """
        probe_config = agent_config.model_copy(deep=True)
        probe_config.add_skill_path(_SKILL_PROBE_PATH)
        plan = client.plan(probe_config, base_dir=self._base_dir)
        return resolve_skill_mode(
            capability_plan=plan.capability_plan,
            adapter_id=agent_config.harness.adapter_id,
        )

    async def _run_task(
        self,
        client: Fabric,
        agent_config: FabricConfig,
        index: int,
        task: AgentEvalTask,
        config: AgentEvalRunConfig,
        skill_mode: SkillMode | None,
    ) -> AgentEvalTrial:
        # nemo_fabric is already imported+validated in ``run_tasks``; this is a cached sys.modules
        # lookup, not a re-load, so the types are used where they're constructed instead of threaded down.
        from nemo_fabric import RunRequest  # ty: ignore[unresolved-import]

        evidence_dir = self._evidence_dir(index, task, config)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # Every task runs in its own fresh workspace: seed any ``inputs['files']`` into it (a no-op when
        # there are none), point the harness at it, and expose it as ``workspace`` filesystem evidence —
        # a uniform per-task dir that maps cleanly onto a per-task container volume later. Seeding runs
        # inside the guarded block so a bad seed (a path escaping the workspace, an unresolvable fileset)
        # fails just this task, not the whole run; it is synchronous and may block (a fileset handler
        # downloads), so it is offloaded off the shared event loop.
        workspace_dir = evidence_dir / _WORKSPACE_SUBDIR
        workspace_dir.mkdir(parents=True, exist_ok=True)
        skill_provenances: list[SkillProvenance] = []
        hook_session = FabricTaskRunSession()
        hook_extras: dict[str, Any] | None = None
        trace_receiver: OTLPReceiver | None = None
        try:
            # Inside the guarded block: a port it cannot bind costs this trial its trace, like any
            # other per-task failure, rather than aborting every task in the gather.
            trace_receiver = self._start_trace_receiver(evidence_dir)
            # Stage seed files into the workspace for their on-disk side effect; the prompt is the task
            # instruction only, so the returned paths are unused.
            await asyncio.to_thread(seed_workspace, workspace_dir, task.inputs.get(SEED_FILES_INPUT_KEY))

            # Inject the skill set (if any) for this task. A native harness gets each staged bundle added
            # to the config's ``skills.paths``; codex self-injection stages each bundle into the
            # workspace and adds no path. One provenance per skill is stamped on the trial for the A/B
            # diff. Blocking file I/O, off the event loop.
            skill_paths: list[str] = []
            if self._skill_set.skills and skill_mode is not None:
                installation = await asyncio.to_thread(
                    install_skills,
                    skills=self._skill_set.skills,
                    adapter_id=agent_config.harness.adapter_id,
                    mode=skill_mode,
                    workspace_dir=workspace_dir,
                    skill_stage_dir=(evidence_dir / _SKILL_SUBDIR).resolve(),
                )
                skill_provenances = installation.provenances
                skill_paths = installation.skill_paths

            # Everything the run needs lives in one typed config: Fabric no longer layers profile
            # overlays, so the per-task workspace/model/trajectory settings are composed on last and are
            # authoritative by construction. ``add_skill_path`` appends, so config-declared skills survive.
            task_config = self._compose_config(
                agent_config, evidence_dir, workspace_dir, task=task, trace_receiver=trace_receiver
            )
            for skill_path in skill_paths:
                task_config.add_skill_path(skill_path)

            if self._task_hook is not None:
                task_config = self._task_hook.prepare(
                    config=task_config,
                    task=task,
                    evidence_dir=evidence_dir,
                    workspace_dir=workspace_dir,
                    session=hook_session,
                )

            result = await asyncio.wait_for(
                # ``Fabric.run`` folds the per-invocation input + request id into a ``RunRequest``.
                client.run(
                    task_config,
                    base_dir=self._base_dir,
                    request=RunRequest(input=task.agent_prompt(), request_id=task.id),
                ),
                timeout=self._timeout_s,
            )
            # Always try to harvest MCP binding results. Hermes often ends with
            # ``completed=false`` / empty finals after a successful tool call; the binding
            # audit is still the authoritative analyzer output for scoring.
            if self._task_hook is not None:
                try:
                    hook_extras = self._task_hook.after_success(task=task, result=result, session=hook_session)
                except Exception as exc:  # noqa: BLE001 - binding harvest must not abort the batch
                    logger.warning("Fabric task hook after_success failed: %s", exc)
                    if result.status == "succeeded":
                        raise
                    hook_extras = None
        except TimeoutError as exc:
            return self._failed_trial(
                task,
                evidence_dir,
                exc,
                extra_metadata=self._skill_metadata(skill_provenances),
                measurements=_atif_measurements(_relay_atif_path(evidence_dir)),
            )
        except Exception as exc:  # noqa: BLE001 - a task failure must not abort the whole run
            return self._failed_trial(
                task,
                evidence_dir,
                exc,
                extra_metadata=self._skill_metadata(skill_provenances),
                measurements=_atif_measurements(_relay_atif_path(evidence_dir)),
            )
        finally:
            if trace_receiver is not None:
                # Stopped before the fold, and both before `_to_trial` reads the evidence: the
                # spans arrive on the exporter's own schedule, and a trace folded before the last
                # flush looks complete.
                trace_receiver.__exit__(None, None, None)
                # Guarded because this runs in `finally`, where a raise escapes the handlers above
                # and aborts the whole gather instead of failing this one task.
                try:
                    fold_exports(traces_dir(evidence_dir))
                except Exception as exc:  # noqa: BLE001 - any fold failure costs the trace, not the trial
                    logger.warning("Could not fold the OTLP trace for task %s: %s", task.id, exc)
            if self._task_hook is not None:
                try:
                    self._task_hook.cleanup(session=hook_session)
                except Exception:  # noqa: BLE001 - hook cleanup must not mask the trial outcome
                    pass
            # Codex self-injection staged each bundle *inside* the workspace so the harness could discover
            # it. Remove them once the run is over (it is already captured in the trajectory) so the injected
            # files don't linger in the durable workspace and, on any path that exposes it as filesystem
            # evidence, read as agent output and skew workspace-reading metrics. In ``finally`` so a
            # timed-out or errored run cleans up too, not just the success path. Best-effort per
            # ``_remove_injected_bundle``; a no-op for native mode and when nothing was staged.
            if skill_mode == SKILL_MODE_CODEX_SKILLS_DIR:
                for provenance in skill_provenances:
                    await asyncio.to_thread(_remove_injected_bundle, workspace_dir, provenance["location"])

        return self._to_trial(
            task,
            result,
            evidence_dir,
            workspace_dir,
            skill_provenances=skill_provenances,
            hook_extras=hook_extras,
        )

    @staticmethod
    def _skill_metadata(provenances: list[SkillProvenance]) -> dict[str, Any]:
        """Trial-metadata fields describing the injected skill set (the A/B provenance).

        ``skills`` is the full list of injected-skill provenances (empty = baseline). ``skill`` keeps the
        historical single-provenance field — the lone provenance for a one-skill run, else ``None`` — so
        single-skill consumers (e.g. ``SkillUsedMetric``) and existing trials keep working unchanged.
        """
        return {"skill": provenances[0] if len(provenances) == 1 else None, "skills": provenances}

    def _to_trial(
        self,
        task: AgentEvalTask,
        result: RunResult,
        evidence_dir: Path,
        workspace_dir: Path,
        skill_provenances: list[SkillProvenance] | None = None,
        hook_extras: Mapping[str, Any] | None = None,
    ) -> AgentEvalTrial:
        # Persist the full normalized Fabric result so graders (and debugging) can see the raw
        # envelope, and expose it as an evidence descriptor.
        result_path = evidence_dir / "fabric_result.json"
        result_path.write_text(json.dumps(result.to_mapping(), indent=2, default=str), encoding="utf-8")

        extras = dict(hook_extras) if hook_extras else {}
        measurements = _atif_measurements(_atif_artifact_path(result))
        base_metadata: dict[str, Any] = {
            "runtime": self._runtime_name,
            "harness": result.harness,
            "adapter_id": result.adapter_id,
            "adapter_kind": result.adapter_kind,
            "invocation_id": result.invocation_id,
            "agent_model": self._model,
            # Skill provenance (name + content hash + injection mode) for the A/B diff.
            **self._skill_metadata(skill_provenances or []),
            **extras,
        }

        if result.status != "succeeded":
            # Hermes may report a non-success final message after a successful MCP tool
            # call. Prefer the binding audit result over a hard fail when present.
            binding_result = _first_mcp_binding_result(extras)
            analysis = binding_result if binding_result is not None else extras.get("analyzer_analysis")
            if analysis is not None:
                base_metadata = {
                    **base_metadata,
                    "fabric_status": result.status,
                    "recovered_from_mcp_binding": True,
                }
                return AgentEvalTrial(
                    id=f"{task.id}:fabric",
                    task_id=task.id,
                    status=AgentEvalTrialStatus.COMPLETED,
                    output=AgentOutput(
                        output_text=json.dumps(analysis, default=str),
                        response=_normalize_output(result.output),
                        metadata={**base_metadata, "evidence_dir": str(evidence_dir)},
                    ),
                    evidence=self._evidence(result, result_path, workspace_dir),
                    measurements=measurements,
                    metadata={**base_metadata, "generated": True, "agent_ok": True},
                )
            return self._failed_trial(
                task, evidence_dir, _result_error(result), extra_metadata=base_metadata, measurements=measurements
            )

        # Fabric wraps the output in a ``RunOutput`` mapping (RunOutput response contract, #52),
        # which is not itself a JSON value; normalize it to a plain mapping so it round-trips through the
        # trial's ``JsonValue``-typed response.
        output = _normalize_output(result.output)
        # Author / mcp_run_binding hooks may attach a structured result. Prefer that when the
        # harness returns an empty final message after a successful tool call.
        output_text = _extract_output_text(output)
        if not output_text or not str(output_text).strip():
            binding_result = _first_mcp_binding_result(extras)
            analysis = binding_result if binding_result is not None else extras.get("analyzer_analysis")
            if analysis is not None:
                output_text = json.dumps(analysis, default=str)
        return AgentEvalTrial(
            id=f"{task.id}:fabric",
            task_id=task.id,
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(
                output_text=output_text,
                response=output,
                metadata={**base_metadata, "evidence_dir": str(evidence_dir)},
            ),
            evidence=self._evidence(result, result_path, workspace_dir),
            measurements=measurements,
            # AgentPhaseSuccessMetric reads agent_ok to score whether the agent phase finished cleanly
            # (an explicit bool, not just trial status).
            metadata={**base_metadata, "generated": True, "agent_ok": True},
        )

    def _evidence(self, result: RunResult, result_path: Path, workspace_dir: Path) -> CandidateEvidence:
        # The workspace is a host directory the harness ran in, so its final file tree is available on
        # disk — expose it as filesystem evidence so workspace-reading metrics can score a Fabric trial.
        descriptors: dict[str, EvidenceDescriptor] = {
            "result": EvidenceDescriptor(kind="json", format="json", ref=str(result_path)),
            _WORKSPACE_EVIDENCE_KEY: EvidenceDescriptor(kind=_WORKSPACE_EVIDENCE_KIND, ref=str(workspace_dir)),
        }
        atif_path: Path | None = None
        for artifact in result.artifacts.artifacts:
            descriptors[artifact.name] = EvidenceDescriptor(
                kind=artifact.kind or "file",
                ref=str(artifact.path),
                metadata={"media_type": artifact.media_type},
            )
            # Surface the Relay ATIF trajectory under the standard trace evidence key so graders
            # that consume a normalized trajectory find it.
            if artifact.kind == _ATIF_ARTIFACT_KIND:
                atif_path = Path(artifact.path)

        register_trace_evidence(descriptors, atif=atif_path, otlp=otlp_trace_path(result_path.parent))
        return CandidateEvidence(
            descriptors=descriptors,
            metadata={
                "runtime": self._runtime_name,
                "harness": result.harness,
                "telemetry": [
                    {"provider": ref.provider, "kind": ref.kind, "uri": ref.uri, "trace_id": ref.trace_id}
                    for ref in result.telemetry
                ],
                "events": [{"kind": event.kind, "message": event.message} for event in result.events],
            },
        )

    def _failed_trial(
        self,
        task: AgentEvalTask,
        evidence_dir: Path,
        error: Exception | Mapping[str, Any],
        extra_metadata: Mapping[str, Any] | None = None,
        measurements: TrialMeasurements | None = None,
    ) -> AgentEvalTrial:
        if isinstance(error, Mapping):
            error_type = str(error.get("code") or error.get("stage") or "FabricError")
            error_message = str(error.get("message") or error)
        else:
            error_type = error.__class__.__name__
            error_message = str(error)
        error_path = evidence_dir / "error.json"
        error_path.write_text(json.dumps({"error_type": error_type, "error": error_message}) + "\n", encoding="utf-8")
        return AgentEvalTrial(
            id=f"{task.id}:fabric",
            task_id=task.id,
            status=AgentEvalTrialStatus.FAILED,
            output=None,
            evidence=CandidateEvidence(
                descriptors={"error": EvidenceDescriptor(kind="error", format="json", ref=str(error_path))},
                metadata={"runtime": self._runtime_name},
            ),
            measurements=measurements if measurements is not None else TrialMeasurements(),
            metadata={
                **(dict(extra_metadata) if extra_metadata else {}),
                "runtime": self._runtime_name,
                "agent_ok": False,
                "error_type": error_type,
                "error": error_message,
            },
        )

    def _start_trace_receiver(self, evidence_dir: Path) -> OTLPReceiver | None:
        """A running receiver for this task's OTLP trace, or None when trajectory capture is off."""
        if not self._capture_trajectory:
            return None
        return OTLPReceiver(traces_dir(evidence_dir)).__enter__()

    def _compose_config(
        self,
        agent_config: FabricConfig,
        evidence_dir: Path,
        workspace_dir: Path,
        task: AgentEvalTask,
        trace_receiver: OTLPReceiver | None = None,
    ) -> FabricConfig:
        # nemo_fabric is already imported+validated in ``run_tasks``; this is a cached sys.modules
        # lookup, not a re-load, so the type is used where it's constructed instead of threaded down.
        from nemo_fabric import EnvironmentConfig, ModelConfig  # ty: ignore[unresolved-import]

        # Copy the base config and apply this task's workspace, model, and trajectory settings directly
        # onto it. These land last, so they override anything the supplied config declared.
        cfg = agent_config.model_copy(deep=True)

        # Point the harness at this task's staged workspace (the codex-cli adapter resolves its cwd from
        # it). ``provider="local"`` is required by the native planner. Any config-supplied
        # environment.workspace is overridden per task.
        environment = cfg.environment or EnvironmentConfig(provider="local")
        environment.provider = environment.provider or "local"
        environment.workspace = str(workspace_dir.resolve())
        cfg.environment = environment

        # Apply the model as the config's default (mirrors nemo_fabric.integrations.harbor).
        if self._model:
            provider = self._model.split("/", maxsplit=1)[0] if "/" in self._model else "openai"
            cfg.models["default"] = ModelConfig(provider=provider, model=self._model)

        if self._capture_trajectory:
            # Enable Relay's ATIF/ATOF file exporter under this task's durable evidence dir, and pin the
            # Fabric artifact root so the promoted ``trajectory-*.atif.json`` persists. Requires the
            # ``nemo-relay`` gateway on PATH in the runtime. Stamp the task id (and any caller
            # ``trajectory_extra``) onto ATIF ``extra`` so optimizer trials can join traces to rows.
            relay_dir = evidence_dir / _RELAY_SUBDIR
            artifacts_dir = evidence_dir / _ARTIFACTS_SUBDIR
            relay_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            row_extra = {"nemo.optimizer.row_id": task.id} if task.id else None
            cfg.enable_relay(
                output_dir=str(relay_dir),
                observability=self._relay_config(relay_dir, extra=row_extra, trace_receiver=trace_receiver),
            )
            cfg.runtime.artifacts = str(artifacts_dir)
            cfg.environment.artifacts = str(artifacts_dir)

        return cfg

    def _relay_config(
        self,
        relay_dir: Path,
        extra: Mapping[str, Any] | None = None,
        trace_receiver: OTLPReceiver | None = None,
    ) -> RelayObservabilityConfig:
        atif_extra: dict[str, Any] | None = None
        if self._trajectory_extra or extra:
            atif_extra = {**(self._trajectory_extra or {}), **(dict(extra) if extra else {})}
        return _common.relay_observability(
            relay_dir=str(relay_dir),
            agent_name=self._runtime_name,
            agent_version=_common.FABRIC_AGENT_VERSION,
            extra=atif_extra,
            otlp_endpoint=trace_receiver.endpoint if trace_receiver is not None else None,
        )

    def _evidence_dir(self, index: int, task: AgentEvalTask, config: AgentEvalRunConfig) -> Path:
        root = self._work_root
        if root is None:
            root = (config.work_dir or Path.cwd()) / "evidence" / "fabric"
        # The run id isolates this run's evidence from other runs sharing the same root (A/B baseline
        # vs. skilled); run_tasks always populates it, so the fallback only guards a direct call.
        run_id = config.run_id or _new_run_id()
        safe_task_id = _safe_path_name(task.id)
        task_dir = f"{index:06d}-{safe_task_id}" if safe_task_id else f"task-{index:06d}"
        return Path(root) / _safe_path_name(run_id) / task_dir


def _remove_injected_bundle(workspace_dir: Path, location: str) -> None:
    """Remove the Codex-injected skill subtree from ``workspace_dir`` and prune emptied parents.

    ``location`` is workspace-relative (``.agents/skills/<name>``). Best-effort: the skill was already
    captured in the run's trajectory, so SkillUsedMetric (which reads the trace, not the workspace) is
    unaffected, and any filesystem error here must not fail an otherwise-successful trial.
    """
    workspace_root = workspace_dir.resolve()
    injected = (workspace_dir / location).resolve()
    # Guard against a location escaping the workspace (defensive; provenance is evaluator-authored).
    if workspace_root not in injected.parents or not injected.exists():
        return
    shutil.rmtree(injected, ignore_errors=True)
    # Prune now-empty reserved parents (``.agents/skills``, ``.agents``) but never the workspace itself.
    parent = injected.parent
    while parent != workspace_root and parent.is_dir():
        try:
            parent.rmdir()  # only succeeds while empty
        except OSError:
            break
        parent = parent.parent


def _normalize_output(output: RunOutput | JsonValue) -> JsonValue:
    """Unwrap a Fabric ``RunResult.output`` into the plain JSON value the trial response stores.

    Newer Fabric wraps output in a ``RunOutput`` (the RunOutput response contract), which is a
    ``Mapping``; copy it into a plain dict (equivalent to its ``to_mapping()``). Raw/older JSON outputs
    are already JSON values and pass through unchanged.
    """
    if isinstance(output, Mapping):
        return dict(output)
    return output


def _first_mcp_binding_result(extras: Mapping[str, Any]) -> Any | None:
    """Return the first ``mcp_bindings.<server>.result`` payload, if any."""
    bindings = extras.get("mcp_bindings")
    if not isinstance(bindings, Mapping):
        return None
    for entry in bindings.values():
        if isinstance(entry, Mapping) and "result" in entry:
            return entry.get("result")
    return None


def _extract_output_text(output: object) -> str | None:
    """Pull the user-visible message out of a Fabric ``RunResult.output`` (JSON-shaped).

    Harness outputs vary; adapters commonly nest the final message under ``response`` (the codex-cli
    adapter does). Prefer a string ``response``/``output_text``, else stringify the whole value.
    """
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if isinstance(output, Mapping):
        for key in ("response", "output_text", "text", "message"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    return json.dumps(output, default=str)


def _result_error(result: RunResult) -> Mapping[str, Any]:
    error = result.error
    if error is None:
        return {"code": result.status, "message": "Fabric run did not succeed"}
    return {"stage": error.stage, "code": error.code, "message": error.message}


def _atif_artifact_path(result: RunResult) -> Path | None:
    """Path of the ATIF trajectory Fabric promoted as an artifact, if any."""
    for artifact in result.artifacts.artifacts:
        if artifact.kind == _ATIF_ARTIFACT_KIND:
            return Path(artifact.path)
    return None


def _relay_atif_path(evidence_dir: Path) -> Path | None:
    """Path of the Relay-written ATIF trajectory, used when no ``RunResult`` exists (timeout/error).

    Relay's filename template is per-session, so more than one file can land when subagents emit
    their own sessions. Picking one under-reports and summing double-counts a root that already
    aggregates, so anything other than a single match reports nothing rather than a wrong number.
    """
    matches = sorted((evidence_dir / _RELAY_SUBDIR).glob(_common.ATIF_FILENAME_TEMPLATE.format(session_id="*")))
    if len(matches) == 1:
        return matches[0]
    if matches:
        logger.warning("Fabric token capture: %d ATIF trajectories under %s; skipping", len(matches), evidence_dir)
    return None


def _atif_measurements(path: Path | None) -> TrialMeasurements:
    """Build typed measurements from a Relay ATIF trajectory.

    Each field is resolved on its own: a valid trajectory-level ``final_metrics``
    value wins; when absent, the matching per-step values are summed. An
    explicitly invalid final value or step contributor poisons only that field.
    This per-field fallback matters most on timeout, when Relay may flush a
    partial trajectory.

    ``cache_creation_tokens`` has no ATIF source and stays unset. A missing or
    unreadable trajectory yields empty measurements and never fails the trial.
    """
    if path is None:
        return TrialMeasurements()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Fabric token capture: unreadable ATIF trajectory %s (%s)", path, exc)
        return TrialMeasurements()
    if not isinstance(payload, Mapping):
        return TrialMeasurements()

    final_metrics = payload.get("final_metrics")
    final = final_metrics if isinstance(final_metrics, Mapping) else {}
    fields = {
        "prompt_tokens": ("total_prompt_tokens", "prompt_tokens", False),
        "completion_tokens": ("total_completion_tokens", "completion_tokens", False),
        "cache_read_tokens": ("total_cached_tokens", "cached_tokens", False),
        "cost_usd": ("total_cost_usd", "cost_usd", True),
    }
    resolved: dict[str, int | float] = {}
    for sdk_key, (final_key, step_key, is_float) in fields.items():
        if final_key in final and final[final_key] is not None:
            value = final[final_key]
            if _valid_atif_measurement(value, is_float=is_float):
                resolved[sdk_key] = float(value) if is_float else int(value)
            else:
                logger.warning(
                    "Fabric token capture: invalid %s=%r in %s; omitting %s", final_key, value, path, sdk_key
                )
            continue
        value, valid = _sum_step_metric(payload, step_key, is_float=is_float)
        if valid and value is not None:
            resolved[sdk_key] = value
        elif not valid:
            logger.warning("Fabric token capture: invalid step %s in %s; omitting %s", step_key, path, sdk_key)
    return TrialMeasurements.model_validate(resolved)


def _sum_step_metric(payload: Mapping[str, Any], key: str, *, is_float: bool) -> tuple[int | float | None, bool]:
    """Sum one per-step ATIF metric, poisoning an explicitly invalid contributor."""
    values: list[int | float] = []
    steps = payload.get("steps")
    if steps is None:
        return None, True
    if not isinstance(steps, list):
        return None, False
    for step in steps:
        metrics = step.get("metrics") if isinstance(step, Mapping) else None
        if not isinstance(metrics, Mapping) or key not in metrics or metrics[key] is None:
            continue
        value = metrics[key]
        if not _valid_atif_measurement(value, is_float=is_float):
            return None, False
        values.append(value)
    if not values:
        return None, True
    try:
        total = math.fsum(float(value) for value in values) if is_float else sum(int(value) for value in values)
    except OverflowError:
        return None, False
    if not _valid_atif_measurement(total, is_float=is_float):
        return None, False
    return (float(total) if is_float else int(total), True)


def _valid_atif_measurement(value: Any, *, is_float: bool) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return False
    if not is_float and not isinstance(value, int):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _safe_path_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value).strip(".-")[:120]


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"fabric-{timestamp}-{uuid4().hex[:8]}"
