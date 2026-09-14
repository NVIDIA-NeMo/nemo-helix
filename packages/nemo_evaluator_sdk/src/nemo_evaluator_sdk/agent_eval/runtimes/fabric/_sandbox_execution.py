# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sandbox execution mode of :class:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime.FabricAgentRuntime`.

Runs the Fabric harness **inside a sandbox** (Docker now, Kubernetes/agent-sandbox later) through the
provider-neutral :class:`~nemo_evaluator_sdk.agent_eval.runtimes.sandbox.api.AsyncSandbox` seam.

Per task it:

1. seeds ``/in`` with the composed Fabric agent config and framed input, plus the task's workspace
   seed files;
2. runs one Fabric invocation through a seeded in-sandbox driver
   (:mod:`~nemo_evaluator_sdk.agent_eval.runtimes.fabric.sandbox_driver`), which writes a normalized
   ``RunResult`` to stdout and the workspace + Relay ATIF trajectory under a fixed ``/out`` layout;
3. downloads ``/out`` across the boundary into the task's evidence dir, where the host mode would
   have written the same files, so the runtime maps both modes into one trial shape.

Relay writes ATIF **inside the image** (no host gateway), which removes the bare-``python3`` /
``tomli_w`` adapter-interpreter problem the host mode has to work around.
"""

from __future__ import annotations

import asyncio
import json
import logging
import posixpath
import shlex
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

from nemo_evaluator_sdk.agent_eval.runtimes.fabric import _common, otlp_receiver
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_receiver import READY_FILENAME, TRACES_PATH
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer import TRACES_SUBDIR, fold_exports, traces_dir
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import (
    CODEX_SKILLS_DIR,
    SKILL_MODE_CODEX_SKILLS_DIR,
    AgentSkill,
    SkillInjectionError,
    SkillMode,
    SkillProvenance,
    SkillSet,
    stage_skills_seed,
)
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.api import AsyncSandbox
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import SandboxProvider, SandboxSpec
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.workspace_seeds import SEED_FILES_INPUT_KEY, seed_workspace
from pydantic import JsonValue

logger = logging.getLogger(__name__)

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


class SandboxExecution:
    """Runs each task of one ``run_tasks`` batch inside a fresh sandbox from ``provider``."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        provider: SandboxProvider,
        image: str,
        env: Mapping[str, str],
        model: str | None,
        timeout_s: int,
        capture_trajectory: bool,
        trajectory_extra: Mapping[str, Any] | None,
        runtime_name: str,
        skills: SkillSet,
    ) -> None:
        self._config = config
        self._provider = provider
        self._image = image
        self._env = dict(env)
        self._model = model
        self._timeout_s = timeout_s
        self._capture_trajectory = capture_trajectory
        self._trajectory_extra = dict(trajectory_extra) if trajectory_extra else None
        self._runtime_name = runtime_name
        self._skill_set = skills

    def metadata(self) -> dict[str, Any]:
        """Trial metadata that identifies where the harness ran."""
        return {"image": self._image, "sandbox_provider": self._provider.name}

    async def run_task(self, task: AgentEvalTask, evidence_dir: Path, skill_mode: SkillMode | None) -> _common.TaskRun:
        run = _common.TaskRun(
            workspace_dir=evidence_dir / "workspace", relay_dir=evidence_dir / "relay", metadata=self.metadata()
        )
        try:
            seed_files, run.skill_provenances = self._seed_files(task, skill_mode)
            spec = SandboxSpec(image=self._image, workdir=_WORKSPACE_DIR, env=dict(self._env), files=seed_files)
            async with AsyncSandbox(self._provider, spec) as sandbox:
                await sandbox.start()
                await self._seed_workspace(sandbox, task)
                exec_result = await sandbox.exec(self._fabric_command(), timeout_s=self._timeout_s)
                await sandbox.download_dir(_OUT_DIR, evidence_dir)
            if self._capture_trajectory:
                # After download, because folding needs protobuf and the sandbox image is not
                # guaranteed to have it.
                folded = await asyncio.to_thread(fold_exports, traces_dir(evidence_dir))
                if not folded:
                    # Zero also means the receiver never started -- no `python3`, or the port taken --
                    # and that failure is invisible from the trial otherwise.
                    logger.warning(
                        "No OTLP export was captured for task %s; its trace falls back to ATIF. See %s.",
                        task.id,
                        evidence_dir / "logs" / "otlp-receiver.log",
                    )
            stderr = _read_text(evidence_dir / "logs" / "fabric-stderr.txt") or (exec_result.stderr or "")
            # A timed-out or non-zero driver run is untrustworthy even when a stale/partial
            # fabric_result.json is left behind (the shell ``>`` redirect truncates the file regardless).
            if exec_result.error_type or exec_result.return_code != 0:
                detail = stderr.strip() or exec_result.error_type or f"exit code {exec_result.return_code}"
                raise RuntimeError(f"fabric run failed in the sandbox: {detail}")
            payload = _read_json(evidence_dir / "fabric_result.json")
            # The driver writes a normalized RunResult object (a failed harness run still produces one,
            # with status != "succeeded"). A missing, non-object, or unreadable payload means no usable result.
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"the sandbox produced no usable Fabric result: {stderr.strip()}")
            run.result = _common.ResultView.from_mapping(
                cast(Mapping[str, Any], payload), artifact_path=lambda path: _downloaded_path(path, evidence_dir)
            )
        except Exception as exc:  # noqa: BLE001 - a task failure must not abort the whole run
            run.error = exc
        return run

    def _adapter_id(self) -> str:
        harness = self._config.get("harness")
        adapter_id = harness.get("adapter_id") if isinstance(harness, Mapping) else None
        return str(adapter_id) if adapter_id is not None else ""

    def _fabric_command(self) -> str:
        """The driver invocation: pre-create the /out dirs Fabric chdirs into, run, capture stdout."""
        run = f"python3 {shlex.quote(_DRIVER_PATH)} {shlex.quote(_AGENT_PATH)} {shlex.quote(_INPUT_PATH)}"
        run = f"{run} > {shlex.quote(_RESULT_PATH)} 2> {shlex.quote(_FABRIC_STDERR)}"
        mkdir = f"mkdir -p {_WORKSPACE_DIR} {_RELAY_DIR} {_ARTIFACTS_DIR} {_LOGS_DIR} {_TRACES_DIR}"
        if not self._capture_trajectory:
            return f"{mkdir} && {run}"
        ready = shlex.quote(f"{_TRACES_DIR}/{READY_FILENAME}")
        # Wait for the file the receiver writes once bound: Relay's first export must not arrive
        # before there is anything listening.
        return (
            f"{mkdir} && "
            f"python3 {shlex.quote(_RECEIVER_PATH)} {shlex.quote(_TRACES_DIR)} {_OTLP_PORT} "
            f"> {shlex.quote(f'{_LOGS_DIR}/otlp-receiver.log')} 2>&1 & "
            f"RECEIVER=$!; "
            f"for _ in $(seq 1 50); do [ -f {ready} ] && break; sleep 0.1; done; "
            f"{run}; "
            f"STATUS=$?; "
            f"kill $RECEIVER 2>/dev/null; wait $RECEIVER 2>/dev/null; "
            f"exit $STATUS"
        )

    def _seed_files(
        self, task: AgentEvalTask, skill_mode: SkillMode | None
    ) -> tuple[dict[str, str], list[SkillProvenance]]:
        """Return (files to seed into the sandbox, skill provenances).

        Everything — the runtime's in-container settings and any natively-injected skill paths — is
        composed into the single agent config here, written as JSON for the driver to load. When skills
        are injected each bundle is also rendered into the seed set at the harness's in-sandbox
        discovery path (native: ``/in/skills/<name>``; codex: ``<workspace>/.agents/skills/<name>``).
        """
        skill_paths: list[str] = []
        provenances: list[SkillProvenance] = []
        files: dict[str, str] = {
            _INPUT_PATH: task.agent_prompt(),
            _DRIVER_PATH: _DRIVER_SOURCE.read_text(encoding="utf-8"),
        }
        if self._capture_trajectory:
            files[_RECEIVER_PATH] = _receiver_source()
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
        row_extra = {"nemo.optimizer.row_id": task.id} if task.id else {}
        atif_extra = {**(self._trajectory_extra or {}), **row_extra} or None
        files[_AGENT_PATH] = json.dumps(self._composed_config(skill_paths, atif_extra=atif_extra))
        return files, provenances

    def _composed_config(
        self, skill_paths: Sequence[str] = (), *, atif_extra: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """The supplied agent config with the runtime's in-container settings merged on last.

        The workspace, artifact roots, default model, trajectory telemetry, and any natively-injected
        skill paths are evaluator-owned, so they are applied over whatever the caller's config declared.
        Stays plain dicts rather than round-tripping through the host's ``FabricConfig`` — the sandbox
        may run a different Fabric build, so the config is only required to survive JSON transport, not
        to validate against the host's schema.

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
        if self._model:
            provider = self._model.split("/", maxsplit=1)[0] if "/" in self._model else "openai"
            config["models"] = {
                **_section(config, "models"),
                "default": {"provider": provider, "model": self._model},
            }
        if self._capture_trajectory:
            config.update(
                _common.relay_telemetry_fragment(
                    config,
                    relay_dir=_RELAY_DIR,
                    agent_name=self._runtime_name,
                    agent_version=_common.FABRIC_AGENT_VERSION,
                    extra=atif_extra,
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


def _downloaded_path(sandbox_path: str, evidence_dir: Path) -> str | None:
    """Where an artifact written under ``/out`` landed after the download, or None if it was not under it.

    The downloaded tree is sandbox output, so a symlink in it may point anywhere on the host; the
    candidate is resolved and must still sit inside the evidence dir before it becomes an evidence ref.
    """
    try:
        relative = PurePosixPath(sandbox_path).relative_to(_OUT_DIR)
    except ValueError:
        return None
    if ".." in relative.parts:
        return None
    root = evidence_dir.resolve()
    candidate = evidence_dir.joinpath(*relative.parts)
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        return None
    return str(candidate)


def _receiver_source() -> str:
    """The shared receiver's own source, seeded into the sandbox and run there as a script."""
    return (Path(otlp_receiver.__file__)).read_text(encoding="utf-8")


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
            # ``seed_workspace`` normalizes the key before writing, so compare the same canonical form.
            seed = PurePosixPath(posixpath.normpath(str(rel_path)))
            if seed == injected_bundle or injected_bundle in seed.parents:
                raise SkillInjectionError(
                    f"task seed file {str(rel_path)!r} writes into {str(injected_bundle)!r}, which is "
                    f"also injected as the runtime skill {skill.name!r}; the task upload would overwrite "
                    "the injected bundle. Inject this skill via the runtime ``skills`` parameter or ship "
                    "it in the task's files, not both"
                )


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """A top-level config section as a plain dict — ``{}`` when absent or not a mapping."""
    value = config.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


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
