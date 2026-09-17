# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Fabric-backed agent-eval runtime.

``FabricAgentRuntime`` drives an agent harness (Codex, Hermes, ...) through the
NeMo Fabric Python SDK and adapts each normalized Fabric ``RunResult`` into an
:class:`AgentEvalTrial`. The harness is chosen by the supplied Fabric config's
``harness.adapter_id`` (never inferred from a model); an optional ``model`` slug
is applied as the config's default model, mirroring Fabric's own Harbor integration.

Where the harness process runs is chosen by ``sandbox``:

* **No sandbox** (the default): ``Fabric().run(...)`` in-process on the host.
* **A** :class:`~nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base.SandboxProvider`: each task runs
  inside a fresh sandbox from that provider and its ``/out`` tree is downloaded into the task's
  evidence dir (:mod:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric._sandbox_execution`).

Both modes lay evidence out the same way — ``fabric_result.json``, ``workspace/``, ``relay/``,
``traces/`` under one per-task dir — and map it through one trial-building step, so a metric sees the
same ``result``, ``trace`` and ``workspace`` evidence whichever mode produced the trial.

Per-task settings (workspace, model, trajectory capture) are composed onto a copy of the supplied
config. Fabric removed profile overlays in 0.1.0rc2, so a run is described by exactly one complete
typed config, and the evaluator-owned per-task settings are authoritative simply by being applied last.

Every task runs in its own fresh workspace: the runtime seeds it from ``inputs['files']`` (a no-op
when there are none), runs the harness in it, and exposes its final file tree as ``workspace``
filesystem evidence. Any ``environment.workspace`` set in the supplied config is overridden per task.

``nemo_fabric`` is an optional native dependency: its types are imported for annotations under
``TYPE_CHECKING`` and the package is loaded lazily at runtime, so this module stays importable
without it.
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
from nemo_evaluator_sdk.agent_eval.runtimes.fabric._sandbox_execution import SandboxExecution
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.image import ensure_fabric_image
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
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import SandboxProvider
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    RunnerInfo,
    TrialMeasurements,
)
from nemo_evaluator_sdk.agent_eval.workspace_seeds import SEED_FILES_INPUT_KEY, seed_workspace
from nemo_evaluator_sdk.resolver_protocols import SecretResolver
from nemo_evaluator_sdk.resolvers import LocalSecretResolver
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_LOGS,
    CandidateEvidence,
    EvidenceDescriptor,
)

if TYPE_CHECKING:
    # Annotations use nemo_fabric's real types (single source of truth). nemo_fabric is an optional
    # native package not yet in our locked dependency set, so it is imported for typing only and
    # loaded lazily at runtime (see ``run_tasks``). Drop the ty:ignore once nemo-fabric is a
    # resolvable dependency and the checker can see it.
    from nemo_fabric import (  # ty: ignore[unresolved-import]
        Fabric,
        FabricConfig,
        RelayObservabilityConfig,
    )

DEFAULT_FABRIC_TIMEOUT_S = 600
_RUNTIME_NAME = "fabric"
_MISSING_FABRIC_MSG = "FabricAgentRuntime requires the `nemo-fabric` package (native NeMo Fabric SDK)."
_MISSING_RELAY_MSG = (
    "FabricAgentRuntime trajectory capture requires the `nemo-relay` package "
    "(install `nemo-fabric[relay]`), or set capture_trajectory=False."
)

logger = logging.getLogger(__name__)

# Evidence-dir layout. These subdir names are our own local layout — we create them and hand them to
# Fabric/Relay, so they are not derived from either library.
_RELAY_SUBDIR = "relay"
_ARTIFACTS_SUBDIR = "artifacts"
_WORKSPACE_SUBDIR = "workspace"
_LOGS_SUBDIR = "logs"
_RESULT_FILENAME = "fabric_result.json"
# Per-task skill staging dir (native injection): the skill's files are resolved here and the staged
# root is added to the task config's ``skills.paths``. For codex self-injection the skill lands in the
# workspace instead (no path added).
_SKILL_SUBDIR = "skill"
# Sentinel skill path attached only to probe Fabric's capability planner for the selected adapter's
# skills routing (see ``_resolve_skill_mode``). Never staged and need not exist on disk — the planner
# just reports how it would route a skill for this adapter.
_SKILL_PROBE_PATH = "nemo-eval-skill-capability-probe"
_WORKSPACE_EVIDENCE_KEY = "workspace"
_WORKSPACE_EVIDENCE_KIND = "filesystem"


class FabricAgentRuntime:
    """AgentTaskRunner that generates trials by running tasks through NeMo Fabric.

    The harness is selected entirely by ``config["harness"]["adapter_id"]``. Across harnesses the
    config shape differs mainly in that ``adapter_id``, optional input/output schemas, and any
    harness-specific ``harness.settings``. Fabric owns each adapter's execution mechanism. See
    ``examples/fabric_harness_runtimes.py`` for full Codex and Hermes config examples.

    Pass ``sandbox=`` to run each task inside a sandbox from that provider instead of on the host.
    ``image`` and ``secrets`` only apply there; ``base_dir`` only applies on the host.
    """

    def __init__(
        self,
        config: FabricConfig | Mapping[str, Any],
        *,
        model: str | None = None,
        base_dir: str | Path | None = None,
        work_root: str | Path | None = None,
        timeout_s: int = DEFAULT_FABRIC_TIMEOUT_S,
        capture_trajectory: bool = True,
        trajectory_extra: Mapping[str, Any] | None = None,
        runtime_name: str = _RUNTIME_NAME,
        skills: Sequence[AgentSkill] | None = None,
        sandbox: SandboxProvider | None = None,
        image: str | None = None,
        secrets: Mapping[str, SecretRef] | None = None,
    ) -> None:
        if sandbox is None:
            if image is not None:
                raise ValueError("image= selects the sandbox image; pass sandbox=<SandboxProvider> with it")
            if secrets:
                raise ValueError("secrets= are injected into a sandbox; pass sandbox=<SandboxProvider> with them")
        elif base_dir is not None:
            raise ValueError("base_dir is not supported in sandbox mode: the config is seeded into /in")
        self._config = _common.to_mapping(config)
        self._model = model
        self._base_dir = Path(base_dir).expanduser() if base_dir is not None else None
        self._work_root = Path(work_root).expanduser() if work_root is not None else None
        self._timeout_s = timeout_s
        self._capture_trajectory = capture_trajectory
        self._trajectory_extra = dict(trajectory_extra) if trajectory_extra else None
        self._runtime_name = runtime_name
        self._skill_set = SkillSet(tuple(skills or ()))
        self._sandbox = sandbox
        # Optional prebuilt image: the trial runs inside it, so it must contain the Fabric CLI + adapter.
        # None -> stock harness-agnostic image built on first run.
        self._image = image
        # ``secrets`` maps the env-var name a Fabric harness reads its credential from (declared by the
        # adapter's ``requirements.env``) to a SecretRef. The runner only *declares* them; the resolver
        # is owned by the orchestrator (see ``resolve_secrets``), mirroring ``MetricWithSecrets``.
        self._secrets = dict(secrets or {})
        self._resolved_env: dict[str, str] = {}
        self._secrets_resolved = False

    def with_skills(self, skills: Sequence[AgentSkill]) -> FabricAgentRuntime:
        """Return a copy of this runtime with ``skills`` *added* to its skill set; ``self`` is not modified.

        Additive and chainable: ``rt.with_skills([a]).with_skills([b])`` injects both a and b. Lets an A/B
        eval derive a treated runtime from a skill-free baseline (``baseline.with_skills(the_skills)``) so
        the two arms differ in exactly the injected skills. Skill names must be unique across the combined
        set — two bundles claiming the same ``<name>/`` would collide — so re-adding a skill already
        present raises. A shallow copy suffices — the shared fields are immutable config/paths/provider.
        (``run_tasks`` disposes a sandbox provider on completion, so an A/B run over two arms should give
        each arm its own provider.)
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

    async def resolve_secrets(self, secret_resolver: SecretResolver) -> None:
        """Resolve declared ``SecretRef``\\ s to values, keyed by the env var each harness reads.

        Mirrors ``MetricWithSecrets.resolve_secrets``: the resolver is owned by the orchestrator (the
        AgentEvaluator / execution backend), not the runner. Call before :meth:`run_tasks`; a standalone
        ``run_tasks`` falls back to local env resolution when this was not called.
        """
        env: dict[str, str] = {}
        for env_var, secret_ref in self._secrets.items():
            value = await secret_resolver.resolve_secret(secret_ref)
            if value is None:
                raise ValueError(f"could not resolve secret {secret_ref.root!r} for env var {env_var!r}")
            env[env_var] = value
        self._resolved_env = env
        self._secrets_resolved = True

    def _adapter_id(self) -> str:
        """Harness adapter selected by the Fabric config (empty when unset)."""
        harness = self._config.get("harness")
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
        models = self._config.get("models")
        default = models.get("default") if isinstance(models, Mapping) else None
        model = default.get("model") if isinstance(default, Mapping) else getattr(default, "model", None)
        return str(model) if model is not None else None

    def runner_info(self) -> RunnerInfo:
        """Identify this runner and the Fabric settings that shape its results.

        Records the sandbox provider's name only — never ``self._secrets``, which is persisted nowhere.
        """
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
                "sandbox": self._sandbox.name if self._sandbox is not None else None,
                "image": self._image,
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> Sequence[AgentEvalTrial]:
        resolved_config = config or AgentEvalRunConfig()
        # Assign a run id once per run so two runs (e.g. an A/B baseline vs. skilled variant) written
        # under the same work_root/output_dir land in distinct, non-colliding evidence trees. Callers
        # that set run_id keep their identifier.
        if resolved_config.run_id is None:
            resolved_config = resolved_config.model_copy(update={"run_id": _new_run_id()})

        sandbox_execution: SandboxExecution | None = None
        host: tuple[Fabric, FabricConfig] | None = None
        try:
            if self._sandbox is None:
                host = self._open_host()
            else:
                sandbox_execution = await self._open_sandbox(self._sandbox)

            # Resolve once how a skill would reach this harness (the adapter is constant across tasks) by
            # asking Fabric's own capability planner, so any adapter that declares native skills support —
            # ours or an end-user's — is picked up automatically instead of via a hardcoded allow-list. Fail
            # fast rather than silently run a skill-free trial mislabeled as "with skill", which would
            # corrupt an A/B comparison. Blocking pyo3 planning, so off the shared event loop.
            skill_mode: SkillMode | None = None
            if self._skill_set.skills:
                skill_mode = await asyncio.to_thread(self._resolve_skill_mode)
                if skill_mode is None:
                    raise RuntimeError(
                        f"FabricAgentRuntime received one or more skills but adapter {self._adapter_id()!r} has "
                        "no known skill-injection strategy: Fabric does not route skills to it natively and it "
                        "is not a codex harness. Use a skills-native or codex harness, or drop the skills."
                    )

            semaphore = asyncio.Semaphore(resolved_config.parallelism)

            async def run_one(index: int, task: AgentEvalTask) -> AgentEvalTrial:
                async with semaphore:
                    evidence_dir = self._evidence_dir(index, task, resolved_config)
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    if sandbox_execution is not None:
                        run = await sandbox_execution.run_task(task, evidence_dir, skill_mode)
                    else:
                        assert host is not None
                        run = await self._run_task_on_host(*host, task, evidence_dir, skill_mode)
                    return await self._finish_task(task, evidence_dir, run, skill_mode)

            return await asyncio.gather(*(run_one(index, task) for index, task in enumerate(tasks)))
        finally:
            if self._sandbox is not None:
                # Each sandbox tears itself down; the provider is shared across the batch, so its
                # process-wide resources are disposed once here, even when setup itself raised.
                await self._sandbox.aclose()

    def _open_host(self) -> tuple[Fabric, FabricConfig]:
        try:
            # nemo_fabric ships a native (pyo3) core and is an optional dependency, so it is imported
            # lazily here rather than at module load.
            from nemo_fabric import Fabric, FabricConfig  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise RuntimeError(_MISSING_FABRIC_MSG) from exc
        agent_config = FabricConfig.from_mapping(self._config)
        # Fail fast (once) if trajectory capture is requested but the nemo-relay gateway isn't
        # importable, rather than failing every task the same way inside the per-task guard.
        if self._capture_trajectory:
            try:
                import nemo_relay.observability  # noqa: F401  # ty: ignore[unresolved-import]
            except ImportError as exc:
                raise RuntimeError(_MISSING_RELAY_MSG) from exc
        # ``Fabric`` is a lightweight, reusable facade — not a lifecycle context manager — so it is
        # created once and reused across tasks with no cleanup.
        return Fabric(), agent_config

    async def _open_sandbox(self, provider: SandboxProvider) -> SandboxExecution:
        if self._image is None:
            # Build-if-missing; a first build compiles nemo-fabric (minutes), so keep it off the event loop.
            self._image = await asyncio.to_thread(ensure_fabric_image)
        if self._secrets and not self._secrets_resolved:
            # No orchestrator resolved our secrets (standalone run) — fall back to local env resolution.
            await self.resolve_secrets(LocalSecretResolver())
        return SandboxExecution(
            config=self._config,
            provider=provider,
            image=self._image,
            env=self._resolved_env,
            model=self._model,
            timeout_s=self._timeout_s,
            capture_trajectory=self._capture_trajectory,
            trajectory_extra=self._trajectory_extra,
            runtime_name=self._runtime_name,
            skills=self._skill_set,
        )

    def _resolve_skill_mode(self) -> SkillMode | None:
        """Ask Fabric how a skill would reach the selected harness, or ``None`` if it can't.

        Probes Fabric's capability planner: plan a copy of the config with a sentinel skill path attached
        (it need not exist on disk) and read how the adapter routes skills. Querying the authoritative
        source at runtime means adapters that declare native skills support — ours or an end-user's — are
        detected without a hardcoded list. See :func:`~...skills.resolve_skill_mode`.
        """
        try:
            from nemo_fabric import Fabric, FabricConfig  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise RuntimeError(_MISSING_FABRIC_MSG) from exc
        probe_config = FabricConfig.from_mapping(self._config)
        probe_config.add_skill_path(_SKILL_PROBE_PATH)
        plan = Fabric().plan(probe_config, base_dir=self._base_dir)
        return resolve_skill_mode(capability_plan=plan.capability_plan, adapter_id=self._adapter_id())

    async def _run_task_on_host(
        self,
        client: Fabric,
        agent_config: FabricConfig,
        task: AgentEvalTask,
        evidence_dir: Path,
        skill_mode: SkillMode | None,
    ) -> _common.TaskRun:
        # nemo_fabric is already imported+validated in ``_open_host``; this is a cached sys.modules
        # lookup, not a re-load, so the types are used where they're constructed instead of threaded down.
        from nemo_fabric import RunRequest  # ty: ignore[unresolved-import]

        workspace_dir = evidence_dir / _WORKSPACE_SUBDIR
        workspace_dir.mkdir(parents=True, exist_ok=True)
        run = _common.TaskRun(workspace_dir=workspace_dir, relay_dir=evidence_dir / _RELAY_SUBDIR)
        trace_receiver: OTLPReceiver | None = None
        try:
            # Inside the guarded block: a port it cannot bind costs this trial its trace, like any
            # other per-task failure, rather than aborting every task in the gather.
            trace_receiver = self._start_trace_receiver(evidence_dir)
            # Seeding runs inside the guarded block so a bad seed (a path escaping the workspace, an
            # unresolvable fileset) fails just this task, not the whole run; it is synchronous and may
            # block (a fileset handler downloads), so it is offloaded off the shared event loop.
            await asyncio.to_thread(seed_workspace, workspace_dir, task.inputs.get(SEED_FILES_INPUT_KEY))

            # A native harness gets each staged bundle added to the config's ``skills.paths``; codex
            # self-injection stages each bundle into the workspace and adds no path.
            skill_paths: list[str] = []
            if self._skill_set.skills and skill_mode is not None:
                installation = await asyncio.to_thread(
                    install_skills,
                    skills=self._skill_set.skills,
                    adapter_id=self._adapter_id(),
                    mode=skill_mode,
                    workspace_dir=workspace_dir,
                    skill_stage_dir=(evidence_dir / _SKILL_SUBDIR).resolve(),
                )
                run.skill_provenances = installation.provenances
                skill_paths = installation.skill_paths

            # ``add_skill_path`` appends, so config-declared skills survive.
            task_config = self._compose_config(
                agent_config, evidence_dir, workspace_dir, task=task, trace_receiver=trace_receiver
            )
            for skill_path in skill_paths:
                task_config.add_skill_path(skill_path)

            result = await asyncio.wait_for(
                # ``Fabric.run`` folds the per-invocation input + request id into a ``RunRequest``.
                client.run(
                    task_config,
                    base_dir=self._base_dir,
                    request=RunRequest(input=task.agent_prompt(), request_id=task.id),
                ),
                timeout=self._timeout_s,
            )
            run.result = _common.ResultView.from_result(result)
        except Exception as exc:  # noqa: BLE001 - a task failure must not abort the whole run
            run.error = exc
        finally:
            if trace_receiver is not None:
                # Stopped before the fold, and both before the trial reads the evidence: the
                # spans arrive on the exporter's own schedule, and a trace folded before the last
                # flush looks complete.
                trace_receiver.__exit__(None, None, None)
                # Guarded because this runs in `finally`, where a raise escapes the handler above
                # and aborts the whole gather instead of failing this one task.
                try:
                    fold_exports(traces_dir(evidence_dir))
                except Exception as exc:  # noqa: BLE001 - any fold failure costs the trace, not the trial
                    logger.warning("Could not fold the OTLP trace for task %s: %s", task.id, exc)
        return run

    async def _finish_task(
        self, task: AgentEvalTask, evidence_dir: Path, run: _common.TaskRun, skill_mode: SkillMode | None
    ) -> AgentEvalTrial:
        # Codex self-injection staged each bundle *inside* the workspace so the harness could discover
        # it. Remove them once the run is over (it is already captured in the trajectory) so the injected
        # files don't read as agent output to workspace-reading metrics. Runs for failed tasks too.
        if skill_mode == SKILL_MODE_CODEX_SKILLS_DIR:
            for provenance in run.skill_provenances:
                await asyncio.to_thread(_remove_injected_bundle, run.workspace_dir, provenance["location"])
        skill_metadata = self._skill_metadata(run.skill_provenances)
        atif_path = _atif_path(run)
        measurements = _atif_measurements(atif_path)
        if run.error is not None or run.result is None:
            return self._failed_trial(
                task,
                evidence_dir,
                run.error if run.error is not None else RuntimeError("Fabric produced no result"),
                extra_metadata={**run.metadata, **skill_metadata},
                measurements=measurements,
            )
        return self._to_trial(task, run, run.result, evidence_dir, atif_path=atif_path, measurements=measurements)

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
        run: _common.TaskRun,
        result: _common.ResultView,
        evidence_dir: Path,
        *,
        atif_path: Path | None,
        measurements: TrialMeasurements,
    ) -> AgentEvalTrial:
        # Persist the full normalized Fabric result so graders (and debugging) can see the raw
        # envelope, and expose it as an evidence descriptor.
        result_path = evidence_dir / _RESULT_FILENAME
        result_path.write_text(json.dumps(result.payload, indent=2, default=str), encoding="utf-8")

        base_metadata: dict[str, Any] = {
            "runtime": self._runtime_name,
            "harness": result.harness,
            "adapter_id": result.adapter_id,
            "adapter_kind": result.adapter_kind,
            "invocation_id": result.invocation_id,
            "agent_model": self._model,
            **run.metadata,
            # Skill provenance (name + content hash + injection mode) for the A/B diff.
            **self._skill_metadata(run.skill_provenances),
        }

        if result.status != "succeeded":
            return self._failed_trial(
                task, evidence_dir, result.failure(), extra_metadata=base_metadata, measurements=measurements
            )

        output_text = _common.extract_output_text(result.output)
        return AgentEvalTrial(
            id=f"{task.id}:{self._runtime_name}",
            task_id=task.id,
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(
                output_text=output_text,
                response=result.output,
                metadata={**base_metadata, "evidence_dir": str(evidence_dir)},
            ),
            evidence=self._evidence(run, result, result_path, evidence_dir, atif_path=atif_path),
            measurements=measurements,
            # AgentPhaseSuccessMetric reads agent_ok to score whether the agent phase finished cleanly
            # (an explicit bool, not just trial status).
            metadata={**base_metadata, "generated": True, "agent_ok": True},
        )

    def _evidence(
        self,
        run: _common.TaskRun,
        result: _common.ResultView,
        result_path: Path,
        evidence_dir: Path,
        *,
        atif_path: Path | None,
    ) -> CandidateEvidence:
        descriptors: dict[str, EvidenceDescriptor] = {
            "result": EvidenceDescriptor(kind="json", format="json", ref=str(result_path)),
        }
        # The workspace is where the harness ran, so its final file tree is on disk — expose it as
        # filesystem evidence so workspace-reading metrics can score a Fabric trial.
        if run.workspace_dir.is_dir():
            descriptors[_WORKSPACE_EVIDENCE_KEY] = EvidenceDescriptor(
                kind=_WORKSPACE_EVIDENCE_KIND, ref=str(run.workspace_dir)
            )
        logs_dir = evidence_dir / _LOGS_SUBDIR
        if logs_dir.is_dir():
            descriptors[EVIDENCE_LOGS] = EvidenceDescriptor(kind="logs", ref=str(logs_dir))
        for artifact in result.artifacts:
            descriptors[artifact.name] = EvidenceDescriptor(
                kind=artifact.kind or "file",
                ref=artifact.path,
                metadata={"media_type": artifact.media_type},
            )
        register_trace_evidence(descriptors, atif=atif_path, otlp=otlp_trace_path(evidence_dir))
        return CandidateEvidence(
            descriptors=descriptors,
            metadata={
                "runtime": self._runtime_name,
                "harness": result.harness,
                **run.metadata,
                "telemetry": list(result.telemetry),
                "events": list(result.events),
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
        return _common.build_failed_trial(
            task,
            evidence_dir,
            error,
            runtime_name=self._runtime_name,
            trial_id_suffix=self._runtime_name,
            extra_metadata=extra_metadata,
            measurements=measurements,
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
        # nemo_fabric is already imported+validated in ``_open_host``; this is a cached sys.modules
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
            root = (config.work_dir or Path.cwd()) / "evidence" / _RUNTIME_NAME
        # The run id isolates this run's evidence from other runs sharing the same root (A/B baseline
        # vs. skilled); run_tasks always populates it, so the fallback only guards a direct call.
        run_id = config.run_id or _new_run_id()
        return Path(root) / _common.safe_path_name(run_id) / _common.task_subdir_name(index, task.id)


def _remove_injected_bundle(workspace_dir: Path, location: str) -> None:
    """Remove the Codex-injected skill subtree from ``workspace_dir`` and prune emptied parents.

    ``location`` is workspace-relative (``.agents/skills/<name>``). Best-effort: the skill was already
    captured in the run's trajectory, so SkillUsedMetric (which reads the trace, not the workspace) is
    unaffected, and any filesystem error here must not fail an otherwise-successful trial.
    """
    if not workspace_dir.is_dir():
        return
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


def _atif_path(run: _common.TaskRun) -> Path | None:
    """The ATIF trajectory to grade this task from: the promoted artifact, else Relay's own output.

    Relay's filename template is per-session, so more than one file can land under ``relay/`` when
    subagents emit their own sessions. Picking one under-reports and summing double-counts a root
    that already aggregates, so anything other than a single match reports nothing rather than a
    wrong trajectory.
    """
    if run.result is not None:
        for artifact in run.result.artifacts:
            if artifact.kind == _common.ATIF_ARTIFACT_KIND:
                return Path(artifact.path)
    if not run.relay_dir.is_dir():
        return None
    matches = sorted(run.relay_dir.rglob(_common.ATIF_FILENAME_TEMPLATE.format(session_id="*")))
    if len(matches) == 1:
        return matches[0]
    if matches:
        logger.warning("Fabric token capture: %d ATIF trajectories under %s; skipping", len(matches), run.relay_dir)
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


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"fabric-{timestamp}-{uuid4().hex[:8]}"
