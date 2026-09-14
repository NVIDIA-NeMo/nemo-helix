# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Containerized NeMo Fabric agent-eval runtime.

``FabricContainerRuntime`` is the sandboxed sibling of
:class:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime.FabricAgentRuntime`: instead of
running the Fabric harness on the host filesystem, it runs it **inside a sandbox** (Docker now,
Kubernetes/agent-sandbox later) through the provider-neutral
:class:`~nemo_evaluator_sdk.agent_eval.runtimes.sandbox.api.AsyncSandbox` seam.

Per task it:

1. seeds ``/in`` with the composed Fabric agent config and framed input, plus the task's workspace
   seed files;
2. runs one Fabric invocation through a seeded in-sandbox driver
   (:mod:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.sandbox_driver`), which writes a normalized
   ``RunResult`` to stdout and the workspace + Relay ATIF trajectory under a fixed ``/out`` layout;
3. downloads ``/out`` across the boundary into the durable per-task evidence dir; and
4. maps it into the shared :class:`CandidateEvidence` contract the eval metrics consume — ``result``
   (json), ``trace`` (ATIF), plus ``workspace`` (filesystem) and ``logs`` — so the workspace-file,
   held-out ``run_verifier``, and trajectory metrics score container trials with no metric changes.
   (``FabricAgentRuntime`` surfaces ``workspace``/``logs`` only when Fabric promotes them as artifacts;
   the container always captures them from the ``/out`` tree, so its evidence is a superset.)

Relay writes ATIF **inside the image** (no host gateway), which removes the bare-``python3`` /
``tomli_w`` adapter-interpreter problem the host runtime has to work around.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import shlex
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from nemo_evaluator_sdk.agent_eval.runtimes.fabric import _common, otlp_receiver
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.image import ensure_fabric_image
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_receiver import READY_FILENAME, TRACES_PATH
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer import (
    TRACES_SUBDIR,
    fold_exports,
    otlp_trace_path,
    register_trace_evidence,
    traces_dir,
)
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import (
    CODEX_SKILLS_DIR,
    SKILL_MODE_CODEX_SKILLS_DIR,
    AgentSkill,
    SkillInjectionError,
    SkillMode,
    SkillProvenance,
    SkillSet,
    resolve_skill_mode,
    stage_skills_seed,
)
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.api import AsyncSandbox
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import SandboxExecResult, SandboxProvider, SandboxSpec
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, RunnerInfo
from nemo_evaluator_sdk.agent_eval.workspace_seeds import SEED_FILES_INPUT_KEY, seed_workspace
from nemo_evaluator_sdk.resolver_protocols import SecretResolver
from nemo_evaluator_sdk.resolvers import LocalSecretResolver
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_LOGS,
    CandidateEvidence,
    EvidenceDescriptor,
)
from pydantic import JsonValue

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # nemo_fabric is an optional native dep (see FabricAgentRuntime); imported for typing only. Configs
    # are consumed structurally via ``to_mapping()`` at runtime, so this module stays importable without it.
    from nemo_fabric import FabricConfig  # ty: ignore[unresolved-import]

# Default per-task exec budget. Timeout is really task-specific (see AALGO-323 to move it onto
# AgentEvalTask); until then it is an internal default rather than a runtime-construction knob.
DEFAULT_FABRIC_TIMEOUT_S = 600
_RUNTIME_NAME = "fabric_container"
_MISSING_FABRIC_MSG = (
    "FabricContainerRuntime skill injection requires the `nemo-fabric` package (native NeMo Fabric SDK) "
    "on the host to resolve how a skill reaches the selected adapter; the container otherwise runs Fabric "
    "only inside the sandbox."
)

# Fixed in-container layout. The runtime seeds ``/in`` (agent config, input), execs Fabric's
# driver, and reads the produced ``/out`` subtree back across the boundary.
_IN_DIR = "/in"
_OUT_DIR = "/out"
_WORKSPACE_DIR = f"{_OUT_DIR}/workspace"
_RELAY_DIR = f"{_OUT_DIR}/relay"
_ARTIFACTS_DIR = f"{_OUT_DIR}/artifacts"
_LOGS_DIR = f"{_OUT_DIR}/logs"
_RESULT_PATH = f"{_OUT_DIR}/fabric_result.json"
_TRACES_DIR = f"{_OUT_DIR}/{TRACES_SUBDIR}"
_RECEIVER_PATH = f"{_IN_DIR}/otlp_receiver.py"
#: OTLP/HTTP's default port, on the sandbox's own loopback. Nothing else in the image binds it.
_OTLP_PORT = 4318
_OTLP_ENDPOINT = f"http://127.0.0.1:{_OTLP_PORT}{TRACES_PATH}"
_FABRIC_STDERR = f"{_LOGS_DIR}/fabric-stderr.txt"
_AGENT_PATH = f"{_IN_DIR}/agent.json"
_INPUT_PATH = f"{_IN_DIR}/input.txt"
#: Seeded as source rather than imported: the driver runs against the sandbox's Fabric, not the host's.
_DRIVER_SOURCE = Path(__file__).with_name("sandbox_driver.py")
_DRIVER_PATH = f"{_IN_DIR}/{_DRIVER_SOURCE.name}"
# In-sandbox root for a natively-injected skill bundle. It lives under ``/in`` (not ``/out``), so it is
# never part of the downloaded ``/out`` evidence — only codex-mode skills, which must sit in the workspace
# for the harness to self-discover them, need post-download cleanup.
_SKILLS_DIR = f"{_IN_DIR}/skills"
# Sentinel skill path attached only to probe Fabric's capability planner for the selected adapter's skills
# routing (mirrors the host runtime). Never staged and need not exist on disk.
_SKILL_PROBE_PATH = "nemo-eval-skill-capability-probe"


class FabricContainerRuntime:
    """AgentTaskRunner that generates trials by running Fabric tasks inside a sandbox."""

    def __init__(
        self,
        config: FabricConfig | Mapping[str, Any],
        *,
        provider: SandboxProvider,
        secrets: Mapping[str, SecretRef] = {},
        image: str | None = None,
        skills: Sequence[AgentSkill] | None = None,
    ) -> None:
        # The Fabric agent is fully described by its ``FabricConfig`` (harness + model + runtime); it is
        # consumed structurally as a mapping to cross the sandbox boundary as JSON.
        self._config = _to_mapping(config)
        self._provider = provider
        # ``secrets`` maps the env-var name a Fabric harness reads its credential from (declared by the
        # adapter's ``requirements.env``) to a SecretRef. The runner only *declares* them; the resolver
        # is owned by the orchestrator (see ``resolve_secrets``), mirroring ``MetricWithSecrets``.
        self._secrets = dict(secrets)
        self._resolved_env: dict[str, str] = {}
        self._secrets_resolved = False
        # Optional prebuilt image: the trial runs inside it, so it must contain the Fabric CLI + adapter.
        # None -> stock harness-agnostic image built on first run.
        self._image: str | None = image
        # Optional agent skills injected per task (A/B: baseline vs. treated via ``with_skills``). How they
        # reach the harness is resolved once per run (the adapter is constant across the taskset) in
        # ``run_tasks``; only touched when a skill is set, so the no-skill path stays dependency-free. Names
        # must be unique — each stages to its own ``<name>/`` bundle, so a repeat would collide.
        self._skill_set = SkillSet(tuple(skills or ()))

    def with_skills(self, skills: Sequence[AgentSkill]) -> FabricContainerRuntime:
        """Return a copy of this runtime with ``skills`` *added* to its skill set; ``self`` is not modified.

        Mirrors :meth:`FabricAgentRuntime.with_skills`: additive and chainable
        (``rt.with_skills([a]).with_skills([b])`` injects both), so an A/B eval derives a treated runtime
        from a skill-free baseline (``baseline.with_skills(the_skills)``) and the arms differ in exactly the
        injected skills. Names must be unique across the combined set (colliding ``<name>/`` bundles), so
        re-adding a present skill raises. A shallow copy suffices — the shared fields are immutable
        config/paths/provider. (``run_tasks`` disposes the injected provider on completion, so an A/B run
        over two arms should give each arm its own provider.)
        """
        clone = copy.copy(self)
        clone._skill_set = self._skill_set.with_skills(skills)
        return clone

    def with_skill(self, skill: AgentSkill) -> FabricContainerRuntime:
        """Return a copy of this runtime with ``skill`` *added*; ``self`` is not modified.

        Thin wrapper over :meth:`with_skills` for the single-skill case; equally chainable
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

    def runner_info(self) -> RunnerInfo:
        """Identify this runner and the Fabric container settings that shape its results.

        Records the provider only — never ``self._secrets``, which is persisted nowhere.
        """
        return RunnerInfo(
            name="fabric_container",
            kind="runner",
            config={
                "provider": self._provider.name,
                "image": self._image,
                "adapter_id": self._adapter_id(),
                "skills": [skill.name for skill in self._skill_set.skills],
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> Sequence[AgentEvalTrial]:
        resolved_config = config or AgentEvalRunConfig()
        semaphore = asyncio.Semaphore(resolved_config.parallelism)

        async def run_one(index: int, task: AgentEvalTask, skill_mode: SkillMode | None) -> AgentEvalTrial:
            async with semaphore:
                logger.info("running task", extra={"index": index + 1, "task_id": task.id})
                result = await self._run_task(index, task, resolved_config, skill_mode)
                logger.info("task completed", extra={"index": index + 1, "task_id": task.id})
                return result

        try:
            # Provision the harness-agnostic Fabric image once, build-if-missing (a first build compiles
            # nemo-fabric — minutes); keep the blocking build off the shared event loop. Inside the guard
            # so the provider is disposed even if provisioning or secret resolution raises.
            if self._image is None:
                self._image = await asyncio.to_thread(ensure_fabric_image)
            if self._secrets and not self._secrets_resolved:
                # No orchestrator resolved our secrets (standalone run) — fall back to local env resolution.
                await self.resolve_secrets(LocalSecretResolver())
            # Resolve once (the adapter is constant across the taskset) how a skill reaches this harness, by
            # probing Fabric's capability planner — the same authoritative routing the host runtime uses.
            # Fail fast rather than silently run a skill-free trial mislabeled "with skill". Blocking pyo3
            # planning, so keep it off the shared event loop; only reached when a skill is set.
            skill_mode = await asyncio.to_thread(self._resolve_skill_mode) if self._skill_set.skills else None
            if self._skill_set.skills and skill_mode is None:
                raise RuntimeError(
                    f"FabricContainerRuntime received one or more skills but adapter {self._adapter_id()!r} "
                    "has no known skill-injection strategy: Fabric does not route skills to it natively and "
                    "it is not a codex harness. Use a skills-native or codex harness, or drop the skills."
                )
            return await asyncio.gather(*(run_one(index, task, skill_mode) for index, task in enumerate(tasks)))
        finally:
            # Each sandbox tears itself down; the provider is shared across the batch, so its
            # process-wide resources are disposed once here, when the batch completes.
            await self._provider.aclose()

    async def _run_task(
        self, index: int, task: AgentEvalTask, config: AgentEvalRunConfig, skill_mode: SkillMode | None
    ) -> AgentEvalTrial:
        evidence_dir = self._evidence_dir(index, task, config)
        out_dir = evidence_dir / "out"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # The whole per-task flow — framing input, seeding, exec, download, and parsing the result — is
        # guarded so any failure (bad seed, sandbox crash, unreadable result) fails only this task's
        # trial rather than aborting the gathered batch.
        skill_provenances: list[SkillProvenance] = []
        try:
            seed_files, skill_provenances = self._seed_files(task, skill_mode)
            spec = SandboxSpec(
                image=self._image, workdir=_WORKSPACE_DIR, env=dict(self._resolved_env), files=seed_files
            )
            async with AsyncSandbox(self._provider, spec) as sandbox:
                await sandbox.start()
                await self._seed_workspace(sandbox, task)
                result = await sandbox.exec(self._fabric_command(), timeout_s=DEFAULT_FABRIC_TIMEOUT_S)
                await sandbox.download_dir(_OUT_DIR, out_dir)
            # After download, because folding needs protobuf and the sandbox image is not
            # guaranteed to have it.
            folded = await asyncio.to_thread(fold_exports, traces_dir(out_dir))
            if not folded:
                # Zero also means the receiver never started -- no `python3`, or the port taken --
                # and that failure is invisible from the trial otherwise.
                logger.warning(
                    "No OTLP export was captured for task %s; its trace falls back to ATIF. See %s.",
                    task.id,
                    out_dir / "logs" / "otlp-receiver.log",
                )
            # Codex self-injection seeds each bundle inside the workspace so the harness discovers it during
            # the run; drop them from the downloaded evidence before the workspace is exposed (else the
            # injected files read as agent output to workspace-reading metrics). Native staging lives under
            # /in, which is never downloaded, so it never pollutes the evidence.
            if skill_mode == SKILL_MODE_CODEX_SKILLS_DIR:
                for provenance in skill_provenances:
                    await asyncio.to_thread(_remove_injected_bundle, out_dir / "workspace", provenance["location"])
            return self._to_trial(task, out_dir, evidence_dir, result, skill_provenances=skill_provenances)
        except Exception as exc:  # noqa: BLE001 - a task failure must not abort the whole run
            # Stamp runtime + image + skills even on failures before _to_trial (startup/seeding/download).
            return self._failed_trial(
                task, evidence_dir, exc, extra_metadata={**self._base_metadata(), **_skill_metadata(skill_provenances)}
            )

    def _resolve_skill_mode(self) -> SkillMode | None:
        """Ask Fabric how a skill would reach the selected harness, or ``None`` if it can't.

        Mirrors :meth:`FabricAgentRuntime._resolve_skill_mode`: plan a copy of the config with a sentinel
        skill path attached (it need not exist on disk) and read how the adapter routes skills from the
        capability plan. Querying the authoritative planner at runtime means any adapter that declares
        native skills support — ours or an end-user's — is picked up without a hardcoded list. ``nemo_fabric``
        is imported lazily on the host (only when a skill is set), so the no-skill path never needs it.
        """
        try:
            from nemo_fabric import Fabric, FabricConfig  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise RuntimeError(_MISSING_FABRIC_MSG) from exc
        probe_config = FabricConfig.from_mapping(self._config)
        probe_config.add_skill_path(_SKILL_PROBE_PATH)
        plan = Fabric().plan(probe_config)
        return resolve_skill_mode(capability_plan=plan.capability_plan, adapter_id=self._adapter_id())

    def _adapter_id(self) -> str:
        """The harness adapter id declared by the config mapping (for provenance + error messages)."""
        harness = self._config.get("harness")
        adapter_id = harness.get("adapter_id") if isinstance(harness, Mapping) else None
        return str(adapter_id) if adapter_id is not None else ""

    def _fabric_command(self) -> str:
        """The driver invocation: pre-create the /out dirs Fabric chdirs into, run, capture stdout."""
        run = f"python3 {shlex.quote(_DRIVER_PATH)} {shlex.quote(_AGENT_PATH)} {shlex.quote(_INPUT_PATH)}"
        ready = shlex.quote(f"{_TRACES_DIR}/{READY_FILENAME}")
        # Wait for the file the receiver writes once bound: Relay's first export must not arrive
        # before there is anything listening.
        return (
            f"mkdir -p {_WORKSPACE_DIR} {_RELAY_DIR} {_ARTIFACTS_DIR} {_LOGS_DIR} {_TRACES_DIR} && "
            f"python3 {shlex.quote(_RECEIVER_PATH)} {shlex.quote(_TRACES_DIR)} {_OTLP_PORT} "
            f"> {shlex.quote(f'{_LOGS_DIR}/otlp-receiver.log')} 2>&1 & "
            f"RECEIVER=$!; "
            f"for _ in $(seq 1 50); do [ -f {ready} ] && break; sleep 0.1; done; "
            f"{run} > {shlex.quote(_RESULT_PATH)} 2> {shlex.quote(_FABRIC_STDERR)}; "
            f"STATUS=$?; "
            f"kill $RECEIVER 2>/dev/null; wait $RECEIVER 2>/dev/null; "
            f"exit $STATUS"
        )

    def _seed_files(
        self, task: AgentEvalTask, skill_mode: SkillMode | None
    ) -> tuple[dict[str, str], list[SkillProvenance]]:
        """Return (files to seed into the sandbox, skill provenances).

        Everything — the runtime's in-container settings and any natively-injected skill paths — is
        composed into the single agent config here, written as JSON for the driver to load. When skills are injected each bundle is also rendered into the seed set
        at the harness's in-sandbox discovery path (native: ``/in/skills/<name>``; codex:
        ``<workspace>/.agents/skills/<name>``).
        """
        skill_paths: list[str] = []
        provenances: list[SkillProvenance] = []
        files: dict[str, str] = {
            _INPUT_PATH: task.agent_prompt(),
            _DRIVER_PATH: _DRIVER_SOURCE.read_text(encoding="utf-8"),
            _RECEIVER_PATH: _receiver_source(),
        }
        if self._skill_set.skills and skill_mode is not None:
            if skill_mode == SKILL_MODE_CODEX_SKILLS_DIR:
                _check_codex_skill_collision(self._skill_set.skills, task.inputs.get(SEED_FILES_INPUT_KEY) or {})
            seed = stage_skills_seed(
                skills=self._skill_set.skills,
                adapter_id=self._adapter_id(),
                mode=skill_mode,
                workspace_dir=_WORKSPACE_DIR,
                skills_dir=_SKILLS_DIR,
            )
            files.update(seed.files)
            skill_paths = seed.skill_paths
            provenances = seed.provenances
        files[_AGENT_PATH] = json.dumps(self._composed_config(skill_paths))
        return files, provenances

    def _composed_config(self, skill_paths: Sequence[str] = ()) -> dict[str, Any]:
        """The supplied agent config with the runtime's in-container settings merged on last.

        Mirrors the host runtime's ``_compose_config``: the workspace, artifact roots, trajectory
        telemetry, and any natively-injected skill paths are evaluator-owned, so they are applied over
        whatever the caller's config declared. Stays plain dicts rather than round-tripping through the
        host's ``FabricConfig`` — the sandbox may run a different Fabric build, so the config is only
        required to survive JSON transport, not to validate against the host's schema.

        Injected skill paths are APPENDED to the config's own ``skills.paths`` — mirroring
        ``FabricConfig.add_skill_path`` — so skills the caller preconfigured survive injection and the
        treated A/B arm differs from the baseline by exactly the injected skills.
        """
        config = dict(self._config)

        # Each section is spread over the caller's, so sibling keys survive — pinning
        # ``runtime.artifacts`` must not drop configured input/output schemas or timeouts.
        config["runtime"] = {**_section(config, "runtime"), "artifacts": _ARTIFACTS_DIR}
        # ``provider: local`` is required by the native planner in the container (it does not inject the
        # Python default), and the workspace pins the harness cwd to the retrievable /out subtree.
        config["environment"] = {
            **_section(config, "environment"),
            "provider": "local",
            "workspace": _WORKSPACE_DIR,
            "artifacts": _ARTIFACTS_DIR,
        }
        # Relay's ATIF/ATOF file exporter, spanning whatever keys Fabric currently serializes it into.
        # ``agent_name`` distinguishes this runtime from the host one; ``agent_version`` records the
        # agent framework and so matches the host's value, letting an ATIF consumer group both
        # runtimes' traces. (Neither is a real version yet.)
        config.update(
            _common.relay_telemetry_fragment(
                config,
                relay_dir=_RELAY_DIR,
                agent_name=_RUNTIME_NAME,
                agent_version=_common.FABRIC_AGENT_VERSION,
                otlp_endpoint=_OTLP_ENDPOINT,
            )
        )

        declared_paths = _section(config, "skills").get("paths") or []
        merged_paths = list(dict.fromkeys([*(str(path) for path in declared_paths), *skill_paths]))
        if merged_paths:
            config["skills"] = {**_section(config, "skills"), "paths": merged_paths}
        return config

    async def _seed_workspace(self, sandbox: AsyncSandbox, task: AgentEvalTask) -> None:
        seeds = task.inputs.get(SEED_FILES_INPUT_KEY)
        if not seeds:
            return
        # Transient host-side staging (a tmpdir, not part of the evidence bundle): seed with the SDK
        # handlers, then upload across the boundary. seed_workspace is synchronous and a handler may do
        # blocking I/O (e.g. a fileset download), so run it off the event loop shared by concurrent tasks.
        with tempfile.TemporaryDirectory(prefix="nemo-fabric-seed-") as staging_dir:
            staging = Path(staging_dir)
            await asyncio.to_thread(seed_workspace, staging, seeds)
            await sandbox.upload_dir(staging, _WORKSPACE_DIR)

    def _base_metadata(self) -> dict[str, object]:
        """Metadata stamped on every trial from this runtime (success or failure), incl. the resolved image."""
        return {"runtime": _RUNTIME_NAME, "image": self._image, "sandbox_provider": self._provider.name}

    def _to_trial(
        self,
        task: AgentEvalTask,
        out_dir: Path,
        evidence_dir: Path,
        result: SandboxExecResult,
        *,
        skill_provenances: list[SkillProvenance] | None = None,
    ) -> AgentEvalTrial:
        # Skill provenance (name + content hash + injection mode) rides on every trial for the A/B diff:
        # a ``skills`` list plus the historical lone ``skill`` field, matching the host FabricAgentRuntime.
        base_metadata = {**self._base_metadata(), **_skill_metadata(skill_provenances or [])}

        # Gate on the exec outcome first: a timed-out or non-zero driver run is untrustworthy even
        # when a stale/partial fabric_result.json is left behind (the shell ``>`` redirect truncates the
        # file regardless), so never grade such a run off that file.
        if result.error_type or result.return_code != 0:
            stderr = _read_text(out_dir / "logs" / "fabric-stderr.txt") or (result.stderr or "")
            detail = stderr.strip() or result.error_type or f"exit code {result.return_code}"
            return self._failed_trial(
                task,
                evidence_dir,
                RuntimeError(f"fabric run failed in the sandbox: {detail}"),
                extra_metadata=base_metadata,
            )

        result_path = out_dir / "fabric_result.json"
        result_payload = _read_json(result_path)
        # The driver writes a normalized RunResult object (a failed harness run still produces one, with
        # status != "succeeded"). A missing, non-object, or unreadable payload means no usable result.
        if not isinstance(result_payload, Mapping):
            stderr = _read_text(out_dir / "logs" / "fabric-stderr.txt") or (result.stderr or "")
            return self._failed_trial(
                task,
                evidence_dir,
                RuntimeError(f"the sandbox produced no usable Fabric result: {stderr.strip()}"),
                extra_metadata=base_metadata,
            )

        status = str(result_payload.get("status"))
        if status != "succeeded":
            return self._failed_trial(task, evidence_dir, _result_error(result_payload), extra_metadata=base_metadata)

        return AgentEvalTrial(
            id=f"{task.id}:fabric_container",
            task_id=task.id,
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(
                # ``response`` is the RunResult *output* payload (matching the host FabricAgentRuntime),
                # not the whole normalized envelope, so metrics reading ``sample.response`` see one shape.
                output_text=_common.extract_output_text(result_payload.get("output")),
                response=cast(JsonValue, result_payload.get("output")),
                metadata={**base_metadata, "evidence_dir": str(evidence_dir)},
            ),
            evidence=self._evidence(out_dir, result_path),
            metadata={**base_metadata, "generated": True, "agent_ok": True},
        )

    def _evidence(self, out_dir: Path, result_path: Path) -> CandidateEvidence:
        descriptors: dict[str, EvidenceDescriptor] = {
            "result": EvidenceDescriptor(kind="json", format="json", ref=str(result_path)),
        }
        workspace_dir = out_dir / "workspace"
        if workspace_dir.is_dir():
            descriptors["workspace"] = EvidenceDescriptor(kind="filesystem", ref=str(workspace_dir))
        logs_dir = out_dir / "logs"
        if logs_dir.is_dir():
            descriptors[EVIDENCE_LOGS] = EvidenceDescriptor(kind="logs", ref=str(logs_dir))
        register_trace_evidence(descriptors, atif=_find_atif(out_dir / "relay"), otlp=otlp_trace_path(out_dir))
        return CandidateEvidence(
            descriptors=descriptors,
            metadata={"runtime": _RUNTIME_NAME, "sandbox_provider": self._provider.name, "image": self._image},
        )

    def _failed_trial(
        self,
        task: AgentEvalTask,
        evidence_dir: Path,
        error: Exception | Mapping[str, object],
        *,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> AgentEvalTrial:
        # Bind this runtime's name + trial-id suffix to the shared FAILED-trial builder.
        return _common.build_failed_trial(
            task,
            evidence_dir,
            error,
            runtime_name=_RUNTIME_NAME,
            trial_id_suffix=_RUNTIME_NAME,
            extra_metadata=extra_metadata,
        )

    def _evidence_dir(self, index: int, task: AgentEvalTask, config: AgentEvalRunConfig) -> Path:
        # Evidence lands under the run's output dir (like every other runtime); the container's own
        # working state lives at /out inside the sandbox and is downloaded here.
        root = (config.work_dir or Path.cwd()) / "evidence" / "fabric_container"
        return root / _common.task_subdir_name(index, task.id)


def _receiver_source() -> str:
    """The shared receiver's own source, seeded into the sandbox and run there as a script."""
    return (Path(otlp_receiver.__file__)).read_text(encoding="utf-8")


def _to_mapping(config: FabricConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a typed Fabric config or a plain mapping to a plain dict for JSON transport."""
    # A typed Fabric config exposes ``to_mapping()``; a plain mapping is used as-is. Both are
    # str-keyed at runtime, but the getattr + optional (unresolved) ``FabricConfig`` type defeat static
    # narrowing, so cast the known-good source before building the dict.
    to_mapping = getattr(config, "to_mapping", None)
    source = to_mapping() if callable(to_mapping) else config
    return dict(cast(Mapping[str, Any], source))


def _skill_metadata(provenances: list[SkillProvenance]) -> dict[str, object]:
    """Trial-metadata fields describing the injected skill set (the A/B provenance).

    ``skills`` is the full list of injected-skill provenances (empty = baseline). ``skill`` keeps the
    historical single-provenance field — the lone provenance for a one-skill run, else ``None`` — so
    single-skill consumers (e.g. ``SkillUsedMetric``) and existing trials/tests keep working unchanged.
    Mirrors ``FabricAgentRuntime._skill_metadata``.
    """
    return {"skill": provenances[0] if len(provenances) == 1 else None, "skills": provenances}


def _check_codex_skill_collision(skills: Sequence[AgentSkill], task_files: Mapping[str, object]) -> None:
    """Raise if a task seed file targets the same bundle dir as a runtime-injected codex skill.

    ``.agents/skills/`` holds skills from two independent, equally valid sources: the runtime
    ``skills`` parameter (the A/B knob — staged into the workspace before the sandbox starts) and
    the task's own ``files`` inputs (skills the task definition always ships — uploaded after it
    starts). Tasks are free to seed their own skills there; only writing the *same*
    ``.agents/skills/<name>/`` from both sources is a conflict, since the task upload lands second
    and would overwrite the injected bundle, leaving the stamped provenance hash describing content
    the agent never saw. Fail that case rather than emit a silently mislabeled A/B trial.
    """
    for skill in skills:
        injected_bundle = PurePosixPath(CODEX_SKILLS_DIR) / skill.name
        for rel_path in task_files:
            seed = PurePosixPath(rel_path)
            if seed == injected_bundle or injected_bundle in seed.parents:
                raise SkillInjectionError(
                    f"task seed file {str(rel_path)!r} writes into {str(injected_bundle)!r}, which is "
                    f"also injected as the runtime skill {skill.name!r}; the task upload would overwrite "
                    "the injected bundle. Inject this skill via the runtime ``skills`` parameter or ship "
                    "it in the task's files, not both"
                )


def _remove_injected_bundle(workspace_dir: Path, location: str) -> None:
    """Remove the Codex-injected skill subtree from a downloaded ``workspace`` dir and prune emptied parents.

    ``location`` is workspace-relative (``.agents/skills/<name>``). Best-effort and mirrors the host
    runtime's cleanup: the skill was already captured in the run's trajectory, so SkillUsedMetric (which
    reads the trace, not the workspace) is unaffected, and any filesystem error here must not fail an
    otherwise-successful trial.
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


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """A top-level config section as a plain dict — ``{}`` when absent or not a mapping."""
    value = config.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def _find_atif(relay_dir: Path) -> Path | None:
    # Relay nests the trajectory under a per-run subdir (relay/runtime-<id>/trajectory-*.atif.json),
    # so search recursively rather than only relay's direct children.
    if not relay_dir.is_dir():
        return None
    matches = sorted(relay_dir.rglob("trajectory-*.atif.json"))
    return matches[0] if matches else None


def _read_json(path: Path) -> JsonValue | None:
    if not path.is_file():
        return None
    # A truncated/binary/unreadable result (e.g. a crashed CLI that left partial or non-UTF-8 bytes)
    # is treated as "no usable result" rather than propagating and aborting the batch.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else ""
    except (UnicodeDecodeError, OSError):
        return ""


def _result_error(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        return {"code": "FabricError", "message": "Fabric run did not produce a result"}
    error = payload.get("error")
    if isinstance(error, Mapping):
        return {"stage": error.get("stage"), "code": error.get("code"), "message": error.get("message")}
    return {"code": payload.get("status"), "message": "Fabric run did not succeed"}
