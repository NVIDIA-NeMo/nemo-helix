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
from nhx_sandbox.opensandbox_policy import canonical_egress_target
from tomlkit.items import InlineTable, Table

from scaled_evals.api.framework_versions import HARBOR_OPENSANDBOX_HARBOR_VERSION, HARBOR_OPENSANDBOX_RUNTIME
from scaled_evals.api.settings import settings
from scaled_evals.dispatch.credentials import merged_env_file
from scaled_evals.dispatch.paths import setting_evaluation_dir
from scaled_evals.dispatch.runtime_backend import (
    CallableRuntimeBackend,
    IncompatibleTaskError,
    RuntimeBackendCapabilities,
    RuntimeBackendRegistration,
)
from scaled_evals.dispatch.sandbox_k8s import (
    _INITIAL_USER_TURNS_ENV,
    StatusReader,
    _deep_merge_harbor_config,
    _harbor_result_path,
    _harbor_run_argv,
    _image_ref_for_sandbox,
    _inject_extra_skills,
    _is_dataset_only_harbor_profile,
    _normalize_harbor_profile_config,
    _patch_instruction,
    _process_start_ticks,
    _save_extra_skill_materials_artifact,
    _save_instruction_artifact,
    _spawn_detached,
    _stage_task_tree,
    _staged_task_name,
    _task_image_ref_for_sandbox,
    _terminate_process_group,
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

# Harbor environment class the rendered config loads for every trial.
NEMO_OPENSANDBOX_IMPORT_PATH = "scaled_evals.harbor_opensandbox_environment:NemoOpenSandboxEnvironment"
# Run-level artifact of per-sandbox egress records, read back by the evidence builder for provenance.
APPLIED_EGRESS_SUMMARY_FILENAME = "scaled-evals-applied-egress.json"
# Evaluation network policies this runtime accepts.
SUPPORTED_NETWORK_POLICIES = ("default_deny",)
# Module the terminator runs, with the Harbor runner's interpreter, to kill an evaluation's sandboxes.
CLEANUP_MODULE = "scaled_evals.harbor_opensandbox_cleanup"

# Template kwargs that tune Harbor's adapter without touching isolation. Everything else in the
# environment block is set by this backend or rejected.
_TEMPLATE_ENVIRONMENT_KWARGS = frozenset(
    {"use_server_proxy", "request_timeout_sec", "ready_timeout_sec", "health_check_poll_interval_sec"}
)
# Top-level Harbor config keys this backend sets; an evaluation profile that sets any of them is rejected.
_BACKEND_OWNED_TOP_LEVEL = ("environment", "datasets", "tasks", "jobs_dir", "job_name")
# File names that mark a multi-container (Compose) task, which one sandbox per trial can't run.
_COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")

# (argv, env, timeout_s) -> completed process. Injected in tests so nothing is spawned.
CleanupRunner = Callable[[Sequence[str], Mapping[str, str], float], subprocess.CompletedProcess[str]]
# (argv, cwd, log_path, env) -> None. Injected in tests so nothing is spawned.
EnvRunner = Callable[[list[str], Path, Path, Mapping[str, str]], None]


class HarborOpenSandboxBackend(CallableRuntimeBackend):
    """Runtime backend wiring the submitter, status reader and terminator built in this module.

    With no submitter (the runtime is disabled), launches fail with a message naming the setting to enable.
    """

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
    """Split a comma-separated host setting, dropping blanks."""
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
    # NemoOpenSandboxEnvironment subclasses one specific Harbor release.
    if spec.framework != "harbor":
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} runs Harbor evaluations only, not {spec.framework!r}")
    if spec.framework_version != HARBOR_OPENSANDBOX_HARBOR_VERSION:
        raise ValueError(
            f"{HARBOR_OPENSANDBOX_RUNTIME} requires Harbor {HARBOR_OPENSANDBOX_HARBOR_VERSION}, "
            f"got {spec.framework_version!r}"
        )

    # Egress is always default-deny plus the operator allowlist; evaluations can't configure it.
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

    # sandbox_k8s features this runtime hasn't implemented.
    if spec.agent_bundle:
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} does not support agent bundles yet")

    # The task must be an uploaded pack with a prebuilt image: submit stages the pack, and
    # OpenSandbox can only run existing images.
    if spec.harbor_dataset_image_imports:
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} requires an uploaded task pack, not dataset image imports")
    if not (spec.image_ref and spec.image_digest):
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} requires a prebuilt task image with a recorded digest")
    if not spec.tarball_object_key:
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} requires an uploaded task pack")
    if _is_dataset_only_harbor_profile(spec.harbor_config):
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} does not support dataset-only Harbor profiles")


def _verifier_mode(verifier: Any) -> str | None:
    """Harbor's verifier mode for one ``[verifier]`` table, before inheritance; ``None`` if unset.

    Mirrors ``harbor.models.task.verifier_mode._resolve_mode``: an explicit ``environment_mode``
    wins, and a ``[verifier.environment]`` table on its own implies ``separate``.
    """
    if not isinstance(verifier, Mapping):
        return None
    mode = verifier.get("environment_mode")
    if mode is not None:
        return str(mode)
    if verifier.get("environment") is not None:
        return "separate"
    return None


def uses_separate_verifier(document: Mapping[str, Any]) -> bool:
    """True when any verify pass of the task runs in its own sandbox, as Harbor 0.20 resolves it."""
    # Harbor's default is "shared": the verifier runs inside the agent's sandbox.
    task_mode = _verifier_mode(document.get("verifier")) or "shared"

    steps = document.get("steps") or []
    if not steps:
        return task_mode == "separate"

    # Multi-step task: each step verifies on its own, using its own mode if set, else the task's.
    for step in steps:
        step_verifier = step.get("verifier") if isinstance(step, Mapping) else None
        if (_verifier_mode(step_verifier) or task_mode) == "separate":
            return True
    return False


def preflight_staged_task(
    task_dir: Path,
    trusted_hosts: Sequence[str],
    *,
    verifier_image_ref: str | None = None,
    verifier_image_digest: str | None = None,
) -> None:
    """Checks that need the staged task tree: compose, the separate verifier image, and task-declared egress."""
    # Each trial gets exactly one sandbox, so multi-container (Compose) tasks can't run.
    for name in _COMPOSE_FILENAMES:
        if (task_dir / "environment" / name).exists() or (task_dir / name).exists():
            raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} runs single-container tasks; the task ships {name}")

    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        raise ValueError(f"staged task has no task.toml: {task_dir}")
    document = tomlkit.parse(task_toml.read_text(encoding="utf-8"))

    # A separate verifier runs from the image built from tests/, which the task revision must
    # record. Without it Harbor would verify in a copy of the agent image, which has no tests.
    if uses_separate_verifier(document):
        if document.get("steps"):
            raise IncompatibleTaskError(
                f"{HARBOR_OPENSANDBOX_RUNTIME} does not support multi-step tasks with a separate verifier yet"
            )
        if not (verifier_image_ref and verifier_image_digest):
            raise IncompatibleTaskError(
                'the task runs its verifier separately ([verifier] environment_mode = "separate"), '
                "but its revision has no verifier image with a recorded digest; finalize a new "
                "revision with verifier_image_ref and verifier_image_digest for the image built from tests/"
            )

    # A task may narrow egress to a subset of the operator allowlist, but asking for any other
    # host fails the evaluation instead of silently dropping it.
    environment = document.get("environment") or {}
    if environment.get("network_mode") == "allowlist":
        requested = [str(host) for host in environment.get("allowed_hosts") or []]
        trusted_targets = {canonical_egress_target(host) for host in trusted_hosts}
        extra = sorted({host for host in requested if canonical_egress_target(host) not in trusted_targets})
        if extra:
            raise ValueError(
                f"task requests egress to {extra}, which the operator allowlist for "
                f"{HARBOR_OPENSANDBOX_RUNTIME} does not include"
            )


def bind_task_image(task_dir: Path, image_ref: str, *, verifier_image_ref: str | None = None) -> bool:
    """Point the staged task at its prebuilt images; OpenSandbox never builds images.

    ``verifier_image_ref`` is bound only when the task runs its verifier separately; any image the
    task itself names for the verifier is replaced. Returns whether it was bound.
    """
    # tomlkit keeps the rest of the author's task.toml (comments, ordering) as-is.
    task_toml = task_dir / "task.toml"
    document = tomlkit.parse(task_toml.read_text(encoding="utf-8"))

    # Agent image: always bound, creating [environment] if the task has none.
    environment = document.get("environment")
    if environment is None:
        environment = tomlkit.table()
        document["environment"] = environment
    if not isinstance(environment, MutableMapping):
        raise ValueError(f"task [environment] must be a TOML table: {task_toml}")
    environment["docker_image"] = image_ref

    # Verifier image: only for a verifier that runs in its own sandbox. A shared verifier
    # runs inside the agent sandbox, so there is nothing to bind.
    verifier_bound = verifier_image_ref is not None and uses_separate_verifier(document)
    if verifier_bound:
        verifier = document.get("verifier")
        if not isinstance(verifier, MutableMapping):
            raise ValueError(f"task [verifier] must be a TOML table: {task_toml}")

        verifier_environment = verifier.get("environment")
        if verifier_environment is None:
            # Without [verifier.environment], Harbor verifies in a copy of [environment].
            # Make that copy explicit, so the verifier keeps the task's resources and
            # network mode and only the image changes.
            verifier_environment = tomlkit.table()
            plain = environment.unwrap() if isinstance(environment, Table | InlineTable) else dict(environment)
            for key, value in plain.items():
                verifier_environment[key] = value
            verifier["environment"] = verifier_environment
        if not isinstance(verifier_environment, MutableMapping):
            raise ValueError(f"task [verifier.environment] must be a TOML table: {task_toml}")

        # Replace any image the task file names: only the platform's image was verified and pinned.
        verifier_environment["docker_image"] = verifier_image_ref

    # Write to a temporary file and swap it in, so a crash never leaves a half-written task.toml.
    temporary = task_toml.with_suffix(".toml.tmp")
    temporary.write_text(tomlkit.dumps(document), encoding="utf-8")
    os.replace(temporary, task_toml)
    return verifier_bound


def _profile_overrides(profile_config: Mapping[str, Any]) -> dict[str, Any]:
    """Render the evaluation's Harbor profile into config overrides, rejecting keys this backend owns."""
    # The profile is YAML text with $VARIABLES, filled from the profile's own env values.
    template_text, profile_env = _normalize_harbor_profile_config(profile_config)
    if not template_text:
        return {}
    rendered = string.Template(template_text).safe_substitute(profile_env)
    loaded = yaml.safe_load(rendered) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Harbor profile config must be a mapping")

    # Profiles tune agents, retries and timeouts; they can't touch what this backend sets.
    owned = [key for key in _BACKEND_OWNED_TOP_LEVEL if key in loaded]
    if owned:
        raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} sets {', '.join(owned)} itself; remove it from the profile")
    return loaded


def ownership_metadata(spec: LaunchSpec) -> dict[str, str]:
    """Labels every sandbox of this evaluation carries: the cleanup selector, plus the benchmark run if any."""
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

    # From the operator template's environment block, keep only the adapter tuning kwargs
    # (proxy use, request and readiness timeouts). Anything else is rejected, not ignored.
    template_environment = template.pop("environment", None) or {}
    if not isinstance(template_environment, Mapping):
        raise ValueError("harbor_opensandbox template environment must be a mapping")
    template_kwargs = dict(template_environment.get("kwargs") or {})
    rejected = sorted(set(template_kwargs) - _TEMPLATE_ENVIRONMENT_KWARGS)
    if rejected:
        raise ValueError(f"harbor_opensandbox template environment.kwargs may not set {', '.join(rejected)}")

    # The run executes exactly the staged task, so drop any task sources the template lists.
    for key in ("datasets", "tasks"):
        template.pop(key, None)

    # Profile values override the template. Then set the per-evaluation job fields.
    config = _deep_merge_harbor_config(template, _profile_overrides(spec.harbor_config))
    config["job_name"] = spec.evaluation_id
    config["jobs_dir"] = jobs_dir
    config["n_attempts"] = spec.n_attempts
    config["n_concurrent_trials"] = spec.parallelism
    config["tasks"] = [{"path": str(task_path)}]

    # The environment block is written last and only here: the sandbox class, egress allowlist,
    # verification mode and ownership labels all come from operator settings. "delete" makes
    # Harbor remove each sandbox when its trial ends.
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
    # Values in the operator env file win over the process environment.
    env = dict(base)
    if env_file:
        env.update(load_env_file(Path(env_file).expanduser()))

    # Raise a message that names where to set the connection settings, not just which one is missing.
    try:
        return _sdk_connection_env(env)
    except ValueError as exc:
        raise RuntimeError(
            f"{HARBOR_OPENSANDBOX_RUNTIME} needs OpenSandbox connection settings; {exc} "
            "(set them in the dispatcher environment or HARBOR_OPENSANDBOX_ENV_FILE)"
        ) from exc


def _opensandbox_sdk_version(harbor_dir: Path) -> str | None:
    """Read the OpenSandbox SDK version installed in a Harbor runner venv, for provenance."""
    for dist_info in sorted((harbor_dir / ".venv").glob("lib/python*/site-packages/opensandbox-*.dist-info")):
        return dist_info.name.removeprefix("opensandbox-").removesuffix(".dist-info")
    return None


def _write_env_file(path: Path, values: Mapping[str, str]) -> Path:
    """Write the agent env file Harbor loads with ``--env-file``, owner-readable only."""
    # Reuse merged_env_file for its value quoting. It needs a source file to append to, and this
    # runtime has none, so start from an empty one (also what Harbor gets when there are no credentials).
    base = path.with_suffix(".base.env")
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("")
    base.chmod(0o600)
    return merged_env_file(source_env_file=base, output_env_file=path, credential_env=values)


def _spawn_with_env(argv: list[str], cwd: Path, log_path: Path, env: Mapping[str, str]) -> None:
    """Default ``EnvRunner``: start Harbor detached with the OpenSandbox connection environment."""
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
    """Build the submitter: preflight, stage the task, render the Harbor config, then start Harbor.

    The returned handle carries the ownership labels cleanup selects on and the launch-time facts
    provenance records.
    """
    runner = runner or _spawn_with_env
    default_harbor = Path(harbor_dir).expanduser()
    template_path = Path(config_path).expanduser()
    work = Path(work_dir).expanduser()

    def submit(spec: LaunchSpec) -> LaunchHandle:
        # Reject unsupported evaluations, and missing OpenSandbox connection settings, before
        # anything is downloaded or written.
        preflight(spec)
        harbor_env = connection_env(os.environ if environ is None else environ, env_file)

        # The evaluation may pin its own Harbor runner venv; otherwise use the deployment default.
        selected_harbor = Path(spec.harbor_dir or default_harbor).expanduser()
        run_dir = work / spec.evaluation_id
        task_dir = run_dir / _staged_task_name(spec)

        # Download the task pack and unpack its Harbor task tree into the run directory.
        # Everything below edits this staged copy, never the uploaded pack.
        staged = spec.tarball_object_key is not None and _stage_task_tree(spec.tarball_object_key, task_dir)
        if not staged:
            raise ValueError(f"{HARBOR_OPENSANDBOX_RUNTIME} task pack contains no Harbor task tree")

        # Checks that need the task files: no Compose tasks, a verifier image for a separate
        # verifier, and no task egress beyond the operator allowlist.
        trusted_hosts = trusted_allowed_hosts()
        preflight_staged_task(
            task_dir,
            trusted_hosts,
            verifier_image_ref=spec.verifier_image_ref,
            verifier_image_digest=spec.verifier_image_digest,
        )

        # Apply the evaluation's task customizations, shared with sandbox_k8s: extra skill files,
        # instruction prefix/postfix, and the agent timeout floor. The skill list and final
        # instruction are saved as run artifacts; the timeout change is kept for the launch handle.
        if spec.extra_skill_object_keys:
            materials = _inject_extra_skills(task_dir, spec.extra_skill_object_keys)
            _save_extra_skill_materials_artifact(
                materials, spec.evaluation_id, harbor_dir=selected_harbor, jobs_dir=jobs_dir
            )
        _patch_instruction(task_dir, spec.instruction_prefix, spec.instruction_postfix)
        _save_instruction_artifact(task_dir, spec.evaluation_id, harbor_dir=selected_harbor, jobs_dir=jobs_dir)
        agent_timeout_apply = (
            apply_agent_timeout_floor(task_dir, spec.agent_timeout_floor_sec)
            if spec.agent_timeout_floor_sec is not None
            else None
        )

        # OpenSandbox can't build images, so point task.toml at the images recorded at finalize.
        # Both are pinned to their recorded digests.
        task_image_ref = _task_image_ref_for_sandbox(spec)
        verifier_image_ref = (
            _image_ref_for_sandbox(spec.verifier_image_ref, spec.verifier_image_digest)
            if spec.verifier_image_ref
            else None
        )

        # A shared-verifier task never uses the verifier image, even if the revision has one.
        # Clear it so provenance doesn't claim a verifier image that wasn't used.
        if not bind_task_image(task_dir, task_image_ref, verifier_image_ref=verifier_image_ref):
            verifier_image_ref = None

        # Render the Harbor config: the operator template plus the evaluation profile, with the
        # environment block (sandbox class, egress allowlist, ownership labels) set by this backend.
        config = render_harbor_config(
            template_path.read_text(),
            spec,
            task_path=task_dir,
            jobs_dir=jobs_dir,
            trusted_hosts=trusted_hosts,
        )
        rendered_path = run_dir / "harbor-config.yaml"
        rendered_path.write_text(yaml.safe_dump(config, sort_keys=False))

        # Harbor gets its environment from two places:
        # - harbor_env is the process environment it's spawned with, carrying the OpenSandbox API
        #   connection settings the environment class uses to create sandboxes.
        # - agent_env is written to an owner-only file that Harbor loads with --env-file. It holds
        #   the model credentials and any scripted user turns that agents read.
        agent_env = dict(spec.credential_env)
        if spec.initial_user_turns:
            agent_env[_INITIAL_USER_TURNS_ENV] = json.dumps(spec.initial_user_turns, separators=(",", ":"))
        agent_env_file = _write_env_file(run_dir / "harbor.env", agent_env)

        log_path = run_dir / "harbor.log"
        argv = [*_harbor_run_argv(selected_harbor), "-c", str(rendered_path), "--env-file", str(agent_env_file), "-y"]

        # Launch-time facts the provenance manifest records; stored on the launch handle below.
        provenance = {
            "harbor_version": spec.framework_version,
            "opensandbox_sdk_version": _opensandbox_sdk_version(selected_harbor),
            "environment_import_path": NEMO_OPENSANDBOX_IMPORT_PATH,
            "task_image_ref": task_image_ref,
            "task_image_digest": spec.image_digest,
            "verifier_image_ref": verifier_image_ref,
            "verifier_image_digest": spec.verifier_image_digest if verifier_image_ref else None,
            "network_policy": spec.network_policy,
            "trusted_allowed_hosts": trusted_hosts,
            "egress_verification": settings.harbor_opensandbox_egress_verification,
        }

        # Only credential names are logged, never values.
        LOG.info(
            "dispatch %s: harbor_opensandbox launch trusted_hosts_count=%d credential_env keys=%s",
            spec.evaluation_id,
            len(trusted_hosts),
            sorted(spec.credential_env),
        )

        # Start Harbor detached and return immediately; the status reader polls it from here on.
        runner(argv, selected_harbor, log_path, harbor_env)

        # The worker saves the handle in the evaluations table. The status reader and terminator
        # use the pid, exit and log paths plus harbor_dir; cleanup selects sandboxes by
        # "ownership"; the evidence builder reads "provenance" and "agent_timeout_apply".
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
    # Harbor gives each trial its own directory under the job dir. Only sandboxes that passed
    # verification have a record; unreadable ones are skipped so one bad file can't hide the rest.
    records: list[dict[str, Any]] = []
    for path in sorted(job_dir.glob(f"*/{APPLIED_EGRESS_FILENAME}")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            LOG.warning("unreadable applied-egress record %s", path)
            continue

        # Keep just the fields provenance reports; the directory name identifies the trial.
        records.append(
            {
                "trial": path.parent.name,
                "sandbox_id": record.get("sandbox_id"),
                "network_mode": record.get("network_mode"),
                "policy_sha256": record.get("policy_sha256"),
            }
        )
    return records


def write_applied_egress_summary(job_dir: Path) -> None:
    """Collect the per-trial records into one artifact the evidence builder reads into provenance."""
    document = {"sandboxes": collect_applied_egress(job_dir)}
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / APPLIED_EGRESS_SUMMARY_FILENAME).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def make_harbor_opensandbox_status_reader(
    *, harbor_dir: str, jobs_dir: str, artifact_root: str | None = None
) -> StatusReader:
    """Read Harbor's status like ``sandbox_k8s``, and write the applied-egress summary once the run ends."""
    read_harbor = make_sandbox_k8s_status_reader(harbor_dir=harbor_dir, jobs_dir=jobs_dir, artifact_root=artifact_root)

    def read(handle: LaunchHandle) -> RuntimeStatus:
        # Harbor writes the same result.json as under sandbox_k8s, so reuse that runtime's reader.
        status = read_harbor(handle)

        # Once Harbor has finished, every trial has written its record, so summarize them now.
        # A failed write is logged, not raised: it must not change the evaluation's outcome.
        if status.phase in {"succeeded", "failed"}:
            job_dir = _harbor_result_path(
                handle, harbor_dir=harbor_dir, jobs_dir=jobs_dir, artifact_root=artifact_root
            ).parent
            try:
                write_applied_egress_summary(job_dir)
            except OSError as exc:
                LOG.warning("harbor_opensandbox applied-egress summary for %s failed: %s", handle.external_id, exc)
        return status

    return read


def _stop_detached_runner(handle: LaunchHandle) -> None:
    """Stop the detached ``harbor run`` recorded in ``handle.raw["pid_file"]``, if it is still running.

    The detached runner deletes its pid file on exit, so a missing file means there is nothing to stop.
    Renaming the file first makes concurrent terminators race for it rather than both signalling.

    Raises:
        RuntimeError: The recorded pid now belongs to a different process, or its group is not detached.
        OSError, ValueError, KeyError: The pid file is unreadable or malformed.
    """
    pid_path_value = handle.raw.get("pid_file")
    if not isinstance(pid_path_value, str) or not pid_path_value:
        return
    pid_path = Path(pid_path_value)
    claimed_pid_path = pid_path.with_name(f"{pid_path.name}.terminating")
    try:
        pid_path.rename(claimed_pid_path)
    except FileNotFoundError:
        return
    try:
        identity = json.loads(claimed_pid_path.read_text())
        pid = int(identity["pid"])
        expected_start = identity.get("start_ticks")
        if expected_start is None or _process_start_ticks(pid) != expected_start:
            raise RuntimeError(f"refusing to terminate reused runner pid {pid}")
        _terminate_process_group(pid)
    finally:
        claimed_pid_path.unlink(missing_ok=True)


def _run_cleanup_process(
    argv: Sequence[str], env: Mapping[str, str], timeout_s: float
) -> subprocess.CompletedProcess[str]:
    """Default ``CleanupRunner``: run the cleanup module and capture its JSON report."""
    return subprocess.run(list(argv), env=dict(env), capture_output=True, text=True, timeout=timeout_s, check=False)


def make_harbor_opensandbox_terminator(
    *,
    harbor_dir: str,
    jobs_dir: str,
    artifact_root: str | None = None,
    env_file: str | None = None,
    cleanup_runner: CleanupRunner | None = None,
    environ: Mapping[str, str] | None = None,
) -> Callable[[LaunchHandle], None]:
    """Stop Harbor, then kill every sandbox this evaluation owns; raise if any survive."""
    cleanup_runner = cleanup_runner or _run_cleanup_process

    def terminate(handle: LaunchHandle) -> None:
        # Every step below runs even if an earlier one fails. Failures are collected and raised
        # together at the end, so a stuck Harbor process never stops sandbox cleanup.
        failures: list[str] = []

        # Stop Harbor first, so it can't create new sandboxes while cleanup is killing them.
        try:
            _stop_detached_runner(handle)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            failures.append(f"harbor runner termination failed: {exc}")

        selector = dict(handle.raw.get("ownership") or {})
        try:
            # Only kill sandboxes labelled with this deployment and this evaluation. A handle whose
            # labels don't match could otherwise delete another deployment's or evaluation's sandboxes.
            validate_selector(selector)
            if selector.get(DEPLOYMENT_METADATA_KEY) != settings.harbor_opensandbox_deployment_id:
                raise ValueError("launch handle belongs to another deployment")
            if selector.get(EVALUATION_METADATA_KEY) != handle.external_id:
                raise ValueError("launch handle ownership does not match its evaluation")

            # Run cleanup with the Harbor runner's interpreter: the OpenSandbox SDK is installed
            # only in that venv. Select on exactly the deployment and evaluation labels; the
            # optional benchmark-run label would only narrow the match.
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

            # The process timeout gives the module a minute beyond its own wait for sandboxes to die.
            # On a non-zero exit its JSON report says why: an error, or the sandboxes still live.
            env = connection_env(os.environ if environ is None else environ, env_file)
            completed = cleanup_runner(argv, env, timeout_s + 60)
            if completed.returncode != 0:
                report = _parse_cleanup_report(completed.stdout)
                detail = report.get("error") or f"sandboxes still live: {report.get('remaining')}"
                failures.append(f"OpenSandbox cleanup failed: {detail}")
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            failures.append(f"OpenSandbox cleanup failed: {exc}")

        # Also write the summary here: a cancelled run never reaches a terminal phase in the status reader.
        job_dir = _harbor_result_path(
            handle, harbor_dir=harbor_dir, jobs_dir=jobs_dir, artifact_root=artifact_root
        ).parent
        try:
            write_applied_egress_summary(job_dir)
        except OSError as exc:
            LOG.warning("harbor_opensandbox applied-egress summary for %s failed: %s", handle.external_id, exc)

        if failures:
            raise RuntimeError("; ".join(failures))

    return terminate


def _parse_cleanup_report(stdout: str) -> dict[str, Any]:
    """Return the last JSON object the cleanup module printed, or an error entry if there is none."""
    # The report is the module's final stdout line; anything before it is log output.
    for line in reversed(stdout.strip().splitlines()):
        try:
            loaded = json.loads(line)
        except ValueError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {"error": f"cleanup printed no report: {stdout[-400:]!r}"}


def validate_settings() -> None:
    """Fail at startup, not at first launch, when the runtime is enabled but misconfigured."""
    if not settings.harbor_opensandbox_enabled:
        return

    # Files the submitter reads on every launch must exist.
    if not settings.harbor_opensandbox_config_path:
        raise RuntimeError("HARBOR_OPENSANDBOX_ENABLED is set but HARBOR_OPENSANDBOX_CONFIG_PATH is not")
    if not Path(settings.harbor_opensandbox_config_path).expanduser().is_file():
        raise RuntimeError(f"HARBOR_OPENSANDBOX_CONFIG_PATH does not exist: {settings.harbor_opensandbox_config_path}")
    env_file = settings.harbor_opensandbox_env_file
    if env_file and not Path(env_file).expanduser().is_file():
        raise RuntimeError(f"HARBOR_OPENSANDBOX_ENV_FILE does not exist: {env_file}")

    # Cleanup selects sandboxes by deployment ID; an empty ID would match other deployments' sandboxes.
    if not settings.harbor_opensandbox_deployment_id.strip():
        raise RuntimeError("HARBOR_OPENSANDBOX_DEPLOYMENT_ID must be non-empty")

    # Valid but probably a mistake: the fallback reads harbor_dir, which may not be the runner Harbor wrote under.
    if not settings.harbor_opensandbox_artifact_root:
        LOG.warning(
            "harbor_opensandbox has no HARBOR_OPENSANDBOX_ARTIFACT_ROOT; reading job output from "
            "HARBOR_DIR/HARBOR_OPENSANDBOX_JOBS_DIR, which misses runs written under another Harbor runner"
        )

    # Valid but probably a mistake: without a model endpoint host, agents can't reach their model.
    if not settings.harbor_opensandbox_model_endpoint_hosts.strip():
        LOG.warning("harbor_opensandbox has no model endpoint host; trials can reach only the operator allowlist")


def build_backend() -> HarborOpenSandboxBackend:
    """Build the backend from settings; when disabled, one that refuses every launch."""
    if not settings.harbor_opensandbox_enabled:
        return HarborOpenSandboxBackend()

    # validate_settings guarantees the config path is set; the assert narrows the type.
    validate_settings()
    assert settings.harbor_opensandbox_config_path is not None

    # All three share the Harbor runner and jobs dir, so they agree on where each run's files live.
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
            artifact_root=settings.harbor_opensandbox_artifact_root,
        ),
        terminator=make_harbor_opensandbox_terminator(
            harbor_dir=settings.harbor_dir,
            jobs_dir=settings.harbor_opensandbox_jobs_dir,
            artifact_root=settings.harbor_opensandbox_artifact_root,
            env_file=settings.harbor_opensandbox_env_file,
        ),
    )


def _artifact_root(evaluation_id: str) -> Path:
    """Harbor's job directory for one evaluation; the worker uploads it as the evaluation's artifacts."""
    if settings.harbor_opensandbox_artifact_root:
        return Path(settings.harbor_opensandbox_artifact_root).expanduser() / evaluation_id
    return Path(settings.harbor_dir).expanduser() / settings.harbor_opensandbox_jobs_dir / evaluation_id


def register_runtime_backends(registry: Any) -> None:
    """Plugin entry point the runtime registry calls to register ``harbor_opensandbox``."""
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
