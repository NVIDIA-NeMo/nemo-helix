# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``harbor_opensandbox`` runtime: Harbor 0.20 trials in OpenSandbox sandboxes.

Harbor runs as a detached child of the evaluation Job, exactly as the ``sandbox_k8s`` process
path does, but its environment is ``NemoOpenSandboxEnvironment``. That subclass creates one
OpenSandbox sandbox per trial under a default-deny egress policy and verifies the applied
policy before the trial starts.

This backend owns the Harbor ``environment`` block. The operator template and evaluation
profile choose agents, retries and timeouts. They cannot choose the environment class, the
egress allowlist, the ownership metadata or the connection settings. The trusted allowlist is
the operator's model endpoint plus the operator's host list. Tasks can narrow it, never widen
it.

Ownership metadata carries this deployment's ID and the evaluation ID. Cancellation, failure
and success all end in the same cleanup: stop the runner, then kill every sandbox that
carries both labels.
"""

from __future__ import annotations

import json
import logging
import os
import string
import subprocess
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import tomlkit
import yaml

from scaled_evals.api.framework_versions import HARBOR_OPENSANDBOX_HARBOR_VERSION, HARBOR_OPENSANDBOX_RUNTIME
from scaled_evals.api.settings import settings
from scaled_evals.dispatch.credentials import merged_env_file
from scaled_evals.dispatch.paths import setting_evaluation_dir
from scaled_evals.dispatch.runtime_backend import (
    CallableRuntimeBackend,
    RuntimeBackendCapabilities,
    RuntimeBackendRegistration,
)
from scaled_evals.dispatch.sandbox_k8s import (
    _INITIAL_USER_TURNS_ENV,
    StatusReader,
    _deep_merge_harbor_config,
    _harbor_result_path,
    _harbor_run_argv,
    _inject_extra_skills,
    _is_dataset_only_harbor_profile,
    _normalize_harbor_profile_config,
    _patch_instruction,
    _save_extra_skill_materials_artifact,
    _save_instruction_artifact,
    _spawn_detached,
    _stage_task_tree,
    _staged_task_name,
    _stop_detached_runner,
    _task_image_ref_for_sandbox,
    apply_agent_timeout_floor,
    load_env_file,
    make_sandbox_k8s_status_reader,
    summarize_harbor_result,
)
from scaled_evals.harbor_opensandbox_cleanup import (
    APPLIED_EGRESS_FILENAME,
    BENCHMARK_RUN_METADATA_KEY,
    DEPLOYMENT_METADATA_KEY,
    EVALUATION_METADATA_KEY,
    ownership_selector,
    validate_selector,
)
from scaled_evals.harbor_opensandbox_cleanup import connection_env as _sdk_connection_env
from scaled_evals.models.runtime import LaunchHandle, LaunchSpec, RuntimeStatus

LOG = logging.getLogger(__name__)

NEMO_OPENSANDBOX_IMPORT_PATH = "scaled_evals.harbor_opensandbox_environment:NemoOpenSandboxEnvironment"
PROVENANCE_FILENAME = "nemo-opensandbox-provenance.json"
SUPPORTED_NETWORK_POLICIES = ("default_deny",)
CLEANUP_MODULE = "scaled_evals.harbor_opensandbox_cleanup"

# Template kwargs that tune Harbor's adapter without touching isolation. Everything else in the
# environment block is set by this backend or rejected.
_TEMPLATE_ENVIRONMENT_KWARGS = frozenset(
    {"use_server_proxy", "request_timeout_sec", "ready_timeout_sec", "health_check_poll_interval_sec"}
)
_BACKEND_OWNED_TOP_LEVEL = ("environment", "datasets", "tasks", "jobs_dir", "job_name")
_COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")

CleanupRunner = Callable[[Sequence[str], Mapping[str, str], float], subprocess.CompletedProcess[str]]
# (argv, cwd, log_path, env) -> None. Injected in tests so nothing is spawned.
EnvRunner = Callable[[list[str], Path, Path, Mapping[str, str]], None]


class HarborOpenSandboxBackend(CallableRuntimeBackend):
    name = HARBOR_OPENSANDBOX_RUNTIME

    def __init__(
        self,
        *,
        submitter: Callable[[LaunchSpec], LaunchHandle] | None = None,
        status_reader: StatusReader | None = None,
        terminator: Callable[[LaunchHandle], None] | None = None,
    ) -> None:
        super().__init__(
            name=self.name,
            summarizer=summarize_harbor_result,
            submitter=submitter,
            status_reader=status_reader,
            terminator=terminator,
            launch_unavailable=(
                "harbor_opensandbox is disabled; set HARBOR_OPENSANDBOX_ENABLED=true on a "
                "deployment whose Jobs can reach an OpenSandbox server"
            ),
        )


def _split_hosts(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def trusted_allowed_hosts() -> list[str]:
    """Model endpoint hosts first, then the operator list, without duplicates."""
    hosts: list[str] = []
    for host in [
        *_split_hosts(settings.harbor_opensandbox_model_endpoint_hosts),
        *_split_hosts(settings.harbor_opensandbox_allowed_hosts),
    ]:
        if host not in hosts:
            hosts.append(host)
    return hosts


def preflight(spec: LaunchSpec) -> None:
    """Reject evaluations this runtime cannot isolate before any work is staged."""
    if spec.framework != "harbor":
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} runs Harbor evaluations only, not {spec.framework!r}")
    if spec.framework_version != HARBOR_OPENSANDBOX_HARBOR_VERSION:
        raise ValueError(
            f"{HARBOR_OPENSANDBOX_RUNTIME} requires Harbor {HARBOR_OPENSANDBOX_HARBOR_VERSION}, "
            f"got {spec.framework_version!r}"
        )
    if spec.network_policy not in SUPPORTED_NETWORK_POLICIES:
        raise ValueError(
            f"{HARBOR_OPENSANDBOX_RUNTIME} supports network_policy "
            f"{', '.join(SUPPORTED_NETWORK_POLICIES)}, not {spec.network_policy!r}"
        )
    if spec.network_policy_config:
        raise ValueError(
            f"{HARBOR_OPENSANDBOX_RUNTIME} takes its egress allowlist from operator settings; "
            "omit network_policy_config"
        )
    if spec.agent_bundle:
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} does not support agent bundles yet")
    if spec.switchyard is not None:
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} does not support Switchyard leases yet")
    if spec.harbor_dataset_image_imports:
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} requires an uploaded task pack, not dataset image imports")
    if not (spec.image_ref and spec.image_digest):
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} requires a prebuilt task image with a recorded digest")
    if not spec.tarball_object_key:
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} requires an uploaded task pack")
    if _is_dataset_only_harbor_profile(spec.harbor_config):
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} does not support dataset-only Harbor profiles")


def preflight_staged_task(task_dir: Path, trusted_hosts: Sequence[str]) -> None:
    """Checks that need the staged task tree: compose, and task-declared egress."""
    for name in _COMPOSE_FILENAMES:
        if (task_dir / "environment" / name).exists() or (task_dir / name).exists():
            raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} runs single-container tasks; the task ships {name}")
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        raise ValueError(f"staged task has no task.toml: {task_dir}")
    environment = tomlkit.parse(task_toml.read_text(encoding="utf-8")).get("environment") or {}
    if environment.get("network_mode") == "allowlist":
        requested = [str(host) for host in environment.get("allowed_hosts") or []]
        extra = sorted(set(requested) - set(trusted_hosts))
        if extra:
            raise ValueError(
                f"task requests egress to {extra}, which the operator allowlist for "
                f"{HARBOR_OPENSANDBOX_RUNTIME} does not include"
            )


def bind_task_image(task_dir: Path, image_ref: str) -> None:
    """Point the staged task at its prebuilt image; OpenSandbox never builds images."""
    task_toml = task_dir / "task.toml"
    document = tomlkit.parse(task_toml.read_text(encoding="utf-8"))
    environment = document.get("environment")
    if environment is None:
        environment = tomlkit.table()
        document["environment"] = environment
    if not isinstance(environment, MutableMapping):
        raise ValueError(f"task [environment] must be a TOML table: {task_toml}")
    environment["docker_image"] = image_ref
    temporary = task_toml.with_suffix(".toml.tmp")
    temporary.write_text(tomlkit.dumps(document), encoding="utf-8")
    os.replace(temporary, task_toml)


def _profile_overrides(profile_config: Mapping[str, Any]) -> dict[str, Any]:
    template_text, profile_env = _normalize_harbor_profile_config(profile_config)
    if not template_text:
        return {}
    rendered = string.Template(template_text).safe_substitute(profile_env)
    loaded = yaml.safe_load(rendered) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Harbor profile config must be a mapping")
    owned = [key for key in _BACKEND_OWNED_TOP_LEVEL if key in loaded]
    if owned:
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} sets {', '.join(owned)} itself; remove it from the profile")
    return loaded


def ownership_metadata(spec: LaunchSpec) -> dict[str, str]:
    metadata = ownership_selector(
        deployment_id=settings.harbor_opensandbox_deployment_id,
        evaluation_id=spec.evaluation_id,
    )
    if spec.benchmark_run_id:
        metadata[BENCHMARK_RUN_METADATA_KEY] = spec.benchmark_run_id
    return metadata


def render_harbor_config(
    template_text: str,
    spec: LaunchSpec,
    *,
    task_path: Path,
    jobs_dir: str,
    trusted_hosts: Sequence[str],
) -> dict[str, Any]:
    """Merge template and profile, then write the environment block this backend owns."""
    template = yaml.safe_load(template_text) or {}
    if not isinstance(template, dict):
        raise ValueError("harbor_opensandbox config template must be a mapping")
    template_environment = template.pop("environment", None) or {}
    if not isinstance(template_environment, Mapping):
        raise ValueError("harbor_opensandbox template environment must be a mapping")
    template_kwargs = dict(template_environment.get("kwargs") or {})
    rejected = sorted(set(template_kwargs) - _TEMPLATE_ENVIRONMENT_KWARGS)
    if rejected:
        raise ValueError(f"harbor_opensandbox template environment.kwargs may not set {', '.join(rejected)}")
    for key in ("datasets", "tasks"):
        template.pop(key, None)

    config = _deep_merge_harbor_config(template, _profile_overrides(spec.harbor_config))
    config["job_name"] = spec.evaluation_id
    config["jobs_dir"] = jobs_dir
    config["n_attempts"] = spec.n_attempts
    config["n_concurrent_trials"] = spec.parallelism
    config["tasks"] = [{"path": str(task_path)}]
    config["environment"] = {
        "import_path": NEMO_OPENSANDBOX_IMPORT_PATH,
        "delete": True,
        "kwargs": {
            **template_kwargs,
            "protocol": settings.harbor_opensandbox_protocol,
            "sandbox_timeout_sec": settings.harbor_opensandbox_sandbox_timeout_seconds,
            "trusted_allowed_hosts": list(trusted_hosts),
            "egress_verification": settings.harbor_opensandbox_egress_verification,
            "metadata": ownership_metadata(spec),
        },
    }
    return config


def connection_env(base: Mapping[str, str], env_file: str | None) -> dict[str, str]:
    """Process env plus the operator env file, with ``OPEN_SANDBOX_*`` aliased to the SDK names."""
    env = dict(base)
    if env_file:
        env.update(load_env_file(Path(env_file).expanduser()))
    try:
        return _sdk_connection_env(env)
    except ValueError as exc:
        raise RuntimeError(
            f"{HARBOR_OPENSANDBOX_RUNTIME} needs OpenSandbox connection settings; {exc} "
            "(set them in the dispatcher environment or HARBOR_OPENSANDBOX_ENV_FILE)"
        ) from exc


def _opensandbox_sdk_version(harbor_dir: Path) -> str | None:
    for dist_info in sorted((harbor_dir / ".venv").glob("lib/python*/site-packages/opensandbox-*.dist-info")):
        return dist_info.name.removeprefix("opensandbox-").removesuffix(".dist-info")
    return None


def _write_env_file(path: Path, values: Mapping[str, str]) -> Path:
    base = path.with_suffix(".base.env")
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("")
    base.chmod(0o600)
    return merged_env_file(source_env_file=base, output_env_file=path, credential_env=values)


def _spawn_with_env(argv: list[str], cwd: Path, log_path: Path, env: Mapping[str, str]) -> None:
    _spawn_detached(argv, cwd, log_path, env=env)


def make_harbor_opensandbox_submitter(
    *,
    harbor_dir: str,
    config_path: str,
    work_dir: str,
    jobs_dir: str,
    env_file: str | None = None,
    runner: EnvRunner | None = None,
    environ: Mapping[str, str] | None = None,
) -> Callable[[LaunchSpec], LaunchHandle]:
    runner = runner or _spawn_with_env
    default_harbor = Path(harbor_dir).expanduser()
    template_path = Path(config_path).expanduser()
    work = Path(work_dir).expanduser()

    def submit(spec: LaunchSpec) -> LaunchHandle:
        preflight(spec)
        harbor_env = connection_env(os.environ if environ is None else environ, env_file)
        selected_harbor = Path(spec.harbor_dir or default_harbor).expanduser()
        run_dir = work / spec.evaluation_id
        task_dir = run_dir / _staged_task_name(spec)
        if spec.tarball_object_key is None or not _stage_task_tree(spec.tarball_object_key, task_dir):
            raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} task pack contains no Harbor task tree")
        trusted_hosts = trusted_allowed_hosts()
        preflight_staged_task(task_dir, trusted_hosts)
        if spec.extra_skill_object_keys:
            materials = _inject_extra_skills(task_dir, spec.extra_skill_object_keys)
            _save_extra_skill_materials_artifact(materials, spec.evaluation_id, harbor_dir=selected_harbor)
        _patch_instruction(task_dir, spec.instruction_prefix, spec.instruction_postfix)
        _save_instruction_artifact(task_dir, spec.evaluation_id, harbor_dir=selected_harbor)
        agent_timeout_apply = (
            apply_agent_timeout_floor(task_dir, spec.agent_timeout_floor_sec)
            if spec.agent_timeout_floor_sec is not None
            else None
        )
        task_image_ref = _task_image_ref_for_sandbox(spec)
        bind_task_image(task_dir, task_image_ref)

        config = render_harbor_config(
            template_path.read_text(),
            spec,
            task_path=task_dir,
            jobs_dir=jobs_dir,
            trusted_hosts=trusted_hosts,
        )
        rendered_path = run_dir / "harbor-config.yaml"
        rendered_path.write_text(yaml.safe_dump(config, sort_keys=False))

        agent_env = dict(spec.credential_env)
        if spec.initial_user_turns:
            agent_env[_INITIAL_USER_TURNS_ENV] = json.dumps(spec.initial_user_turns, separators=(",", ":"))
        agent_env_file = _write_env_file(run_dir / "harbor.env", agent_env)
        log_path = run_dir / "harbor.log"
        argv = [*_harbor_run_argv(selected_harbor), "-c", str(rendered_path), "--env-file", str(agent_env_file), "-y"]
        provenance = {
            "harbor_version": spec.framework_version,
            "opensandbox_sdk_version": _opensandbox_sdk_version(selected_harbor),
            "environment_import_path": NEMO_OPENSANDBOX_IMPORT_PATH,
            "task_image_ref": task_image_ref,
            "task_image_digest": spec.image_digest,
            "network_policy": spec.network_policy,
            "trusted_allowed_hosts": trusted_hosts,
            "egress_verification": settings.harbor_opensandbox_egress_verification,
        }
        LOG.info(
            "dispatch %s: harbor_opensandbox launch trusted_hosts=%s credential_env keys=%s",
            spec.evaluation_id,
            trusted_hosts,
            sorted(spec.credential_env),
        )
        runner(argv, selected_harbor, log_path, harbor_env)
        return LaunchHandle(
            backend=HARBOR_OPENSANDBOX_RUNTIME,
            external_id=spec.evaluation_id,
            raw={
                "config": str(rendered_path),
                "log": str(log_path),
                "pid_file": str(log_path.with_suffix(f"{log_path.suffix}.pid")),
                "exit_file": str(log_path.with_suffix(f"{log_path.suffix}.exit.json")),
                "argv": argv,
                "harbor_dir": str(selected_harbor),
                "agent_timeout_apply": agent_timeout_apply,
                "task_image_ref": task_image_ref,
                "ownership": ownership_metadata(spec),
                "provenance": provenance,
            },
        )

    return submit


def collect_applied_egress(job_dir: Path) -> list[dict[str, Any]]:
    """Read the per-trial records ``NemoOpenSandboxEnvironment`` wrote after verification."""
    records: list[dict[str, Any]] = []
    for path in sorted(job_dir.glob(f"*/{APPLIED_EGRESS_FILENAME}")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            LOG.warning("unreadable applied-egress record %s", path)
            continue
        records.append(
            {
                "trial": path.parent.name,
                "sandbox_id": record.get("sandbox_id"),
                "network_mode": record.get("network_mode"),
                "policy_sha256": record.get("policy_sha256"),
            }
        )
    return records


def write_provenance(job_dir: Path, handle: LaunchHandle, *, cleanup: Mapping[str, Any] | None = None) -> None:
    applied = collect_applied_egress(job_dir)
    document = {
        **dict(handle.raw.get("provenance") or {}),
        "ownership": handle.raw.get("ownership"),
        "applied_egress": applied,
        "applied_policy_sha256s": sorted({str(item["policy_sha256"]) for item in applied if item["policy_sha256"]}),
    }
    if cleanup is not None:
        document["cleanup"] = dict(cleanup)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / PROVENANCE_FILENAME).write_text(json.dumps(document, indent=2, sort_keys=True))


def make_harbor_opensandbox_status_reader(*, harbor_dir: str, jobs_dir: str) -> StatusReader:
    read_harbor = make_sandbox_k8s_status_reader(harbor_dir=harbor_dir, jobs_dir=jobs_dir)

    def read(handle: LaunchHandle) -> RuntimeStatus:
        status = read_harbor(handle)
        if status.phase in {"succeeded", "failed"}:
            job_dir = _harbor_result_path(handle, harbor_dir=harbor_dir, jobs_dir=jobs_dir).parent
            try:
                write_provenance(job_dir, handle)
            except OSError as exc:
                LOG.warning("harbor_opensandbox provenance for %s failed: %s", handle.external_id, exc)
        return status

    return read


def _run_cleanup_process(
    argv: Sequence[str], env: Mapping[str, str], timeout_s: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), env=dict(env), capture_output=True, text=True, timeout=timeout_s, check=False)


def make_harbor_opensandbox_terminator(
    *,
    harbor_dir: str,
    jobs_dir: str,
    env_file: str | None = None,
    cleanup_runner: CleanupRunner | None = None,
    environ: Mapping[str, str] | None = None,
) -> Callable[[LaunchHandle], None]:
    """Stop Harbor, then kill every sandbox this evaluation owns; raise if any survive."""
    cleanup_runner = cleanup_runner or _run_cleanup_process

    def terminate(handle: LaunchHandle) -> None:
        failures: list[str] = []
        try:
            _stop_detached_runner(handle)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            failures.append(f"harbor runner termination failed: {exc}")

        selector = dict(handle.raw.get("ownership") or {})
        report: dict[str, Any]
        try:
            validate_selector(selector)
            if selector.get(DEPLOYMENT_METADATA_KEY) != settings.harbor_opensandbox_deployment_id:
                raise ValueError("launch handle belongs to another deployment")
            if selector.get(EVALUATION_METADATA_KEY) != handle.external_id:
                raise ValueError("launch handle ownership does not match its evaluation")
            selected_harbor = Path(str(handle.raw.get("harbor_dir") or harbor_dir)).expanduser()
            timeout_s = float(settings.harbor_opensandbox_cleanup_timeout_seconds)
            argv = [
                str(selected_harbor / ".venv" / "bin" / "python"),
                "-m",
                CLEANUP_MODULE,
                "--protocol",
                settings.harbor_opensandbox_protocol,
                "--timeout",
                str(timeout_s),
            ]
            for key, value in sorted(selector.items()):
                if key in (DEPLOYMENT_METADATA_KEY, EVALUATION_METADATA_KEY):
                    argv += ["--selector", f"{key}={value}"]
            env = connection_env(os.environ if environ is None else environ, env_file)
            completed = cleanup_runner(argv, env, timeout_s + 60)
            report = _parse_cleanup_report(completed.stdout)
            report["exit_code"] = completed.returncode
            if completed.returncode != 0:
                detail = report.get("error") or f"sandboxes still live: {report.get('remaining')}"
                failures.append(f"OpenSandbox cleanup failed: {detail}")
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            report = {"error": f"{type(exc).__name__}: {exc}"}
            failures.append(f"OpenSandbox cleanup failed: {exc}")

        job_dir = _harbor_result_path(handle, harbor_dir=harbor_dir, jobs_dir=jobs_dir).parent
        try:
            write_provenance(job_dir, handle, cleanup=report)
        except OSError as exc:
            LOG.warning("harbor_opensandbox provenance for %s failed: %s", handle.external_id, exc)
        if failures:
            raise RuntimeError("; ".join(failures))

    return terminate


def _parse_cleanup_report(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        try:
            loaded = json.loads(line)
        except ValueError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {"error": f"cleanup printed no report: {stdout[-400:]!r}"}


def validate_settings() -> None:
    if not settings.harbor_opensandbox_enabled:
        return
    if not settings.harbor_opensandbox_config_path:
        raise RuntimeError("HARBOR_OPENSANDBOX_ENABLED is set but HARBOR_OPENSANDBOX_CONFIG_PATH is not")
    if not Path(settings.harbor_opensandbox_config_path).expanduser().is_file():
        raise RuntimeError(f"HARBOR_OPENSANDBOX_CONFIG_PATH does not exist: {settings.harbor_opensandbox_config_path}")
    env_file = settings.harbor_opensandbox_env_file
    if env_file and not Path(env_file).expanduser().is_file():
        raise RuntimeError(f"HARBOR_OPENSANDBOX_ENV_FILE does not exist: {env_file}")
    if not settings.harbor_opensandbox_deployment_id.strip():
        raise RuntimeError("HARBOR_OPENSANDBOX_DEPLOYMENT_ID must be non-empty")
    if not settings.harbor_opensandbox_model_endpoint_hosts.strip():
        LOG.warning("harbor_opensandbox has no model endpoint host; trials can reach only the operator allowlist")


def build_backend() -> HarborOpenSandboxBackend:
    if not settings.harbor_opensandbox_enabled:
        return HarborOpenSandboxBackend()
    validate_settings()
    assert settings.harbor_opensandbox_config_path is not None
    return HarborOpenSandboxBackend(
        submitter=make_harbor_opensandbox_submitter(
            harbor_dir=settings.harbor_dir,
            config_path=settings.harbor_opensandbox_config_path,
            work_dir=settings.harbor_opensandbox_work_dir,
            jobs_dir=settings.harbor_opensandbox_jobs_dir,
            env_file=settings.harbor_opensandbox_env_file,
        ),
        status_reader=make_harbor_opensandbox_status_reader(
            harbor_dir=settings.harbor_dir,
            jobs_dir=settings.harbor_opensandbox_jobs_dir,
        ),
        terminator=make_harbor_opensandbox_terminator(
            harbor_dir=settings.harbor_dir,
            jobs_dir=settings.harbor_opensandbox_jobs_dir,
            env_file=settings.harbor_opensandbox_env_file,
        ),
    )


def _artifact_root(evaluation_id: str) -> Path:
    return Path(settings.harbor_dir).expanduser() / settings.harbor_opensandbox_jobs_dir / evaluation_id


def register_runtime_backends(registry: Any) -> None:
    registry.register(
        RuntimeBackendRegistration(
            name=HARBOR_OPENSANDBOX_RUNTIME,
            factory=build_backend,
            validate=validate_settings,
            description="Harbor 0.20 trials in default-deny OpenSandbox sandboxes.",
            capabilities=RuntimeBackendCapabilities(
                artifact_root=_artifact_root,
                dispatch_work_dir=setting_evaluation_dir(settings, "harbor_opensandbox_work_dir"),
                dispatch_log_name="harbor.log",
                supported_network_policies=SUPPORTED_NETWORK_POLICIES,
            ),
        )
    )
