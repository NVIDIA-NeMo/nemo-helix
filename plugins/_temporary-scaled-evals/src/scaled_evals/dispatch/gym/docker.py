# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch Gym Daytona work as one-shot Docker containers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scaled_evals.api.settings import settings
from scaled_evals.dispatch.credentials import write_env_file
from scaled_evals.dispatch.gym.common import (
    build_run_and_collect_command,
    harness_root_from_env_file,
    materialize_gym_launch_env,
    resolve_env_file_path,
    validate_gym_launch_contract,
)
from scaled_evals.dispatch.gym.profile import gym_profile_env
from scaled_evals.dispatch.paths import resolve_host_env_file
from scaled_evals.dispatch.runtime_backend import LaunchHandle, LaunchSpec, RuntimeStatus
from scaled_evals.dispatch.sandbox_k8s import StatusReader, load_env_file

RUNNER_ENV_IN_CONTAINER = Path("/run/daytona.env")
RUNNER_WORK_MOUNT = "/work"
GYM_SOURCE_REVISION_LABEL = "com.nvidia.nemo-gym.revision"
GYM_PACKAGE_VERSION_LABEL = "com.nvidia.nemo-gym.version"


def host_env_file_path(
    container_env_file: Path,
    *,
    explicit_host_env_file: str | None = None,
) -> Path:
    """Map a harness env path inside the API container to a host path for ``docker run -v``."""
    return resolve_host_env_file(
        container_env_file,
        explicit_host_env_file=explicit_host_env_file,
        host_root=settings.scaled_evals_host_dir,
    )


def harness_run_and_collect_entrypoint(container_env_file: Path) -> list[str]:
    """Return the harness ``run_and_collect.sh`` path for the gym-runner image."""
    harness = harness_root_from_env_file(container_env_file)
    if harness and (harness / "run_and_collect.sh").is_file():
        return [str(harness / "run_and_collect.sh")]
    if harness and str(harness).startswith("/harness/"):
        return [str(harness / "run_and_collect.sh")]
    return ["/harness/gym-sandbox-daytona/run_and_collect.sh"]


def build_docker_run_and_collect_argv(
    *,
    evaluation_id: str,
    work_dir_in_runner: Path,
    env_file_in_runner: Path = RUNNER_ENV_IN_CONTAINER,
) -> list[str]:
    """Arguments passed to the harness ``run_and_collect.sh`` inside gym-runner."""
    return build_run_and_collect_command(
        env_file=env_file_in_runner,
        evaluation_id=evaluation_id,
        work_dir=work_dir_in_runner,
    )


def launch_gym_runner_container(
    *,
    image: str,
    evaluation_id: str,
    env_file_host: Path | None,
    work_volume: str,
    command: list[str],
    entrypoint: list[str] | None = None,
    shm_size: str | None = None,
) -> str:
    """Start a detached gym-runner container; return the container id."""
    import docker

    client = docker.from_env()
    volumes: dict = {
        work_volume: {"bind": RUNNER_WORK_MOUNT, "mode": "rw"},
    }
    if env_file_host is not None:
        volumes[str(env_file_host)] = {"bind": str(RUNNER_ENV_IN_CONTAINER), "mode": "ro"}
    run_kwargs: dict = {
        "image": image,
        "entrypoint": entrypoint or ["/harness/gym-sandbox-daytona/run_and_collect.sh"],
        "command": command,
        "detach": True,
        "name": f"gym-{evaluation_id}",
        "volumes": volumes,
        "environment": {
            "UV_NO_PROJECT": "1",
            "RAY_TMPDIR": "/tmp/ray",
            "RAY_ENABLE_DASHBOARD": "0",
            "RAY_USAGE_STATS_ENABLED": "0",
        },
        "remove": False,
    }
    if shm_size:
        run_kwargs["shm_size"] = shm_size
    container = client.containers.run(**run_kwargs)
    return container.id


def inspect_gym_runner_container_identity(container_id: str) -> dict[str, Any]:
    """Read non-secret immutable identity from the image Docker actually launched."""
    import docker

    container = docker.from_env().containers.get(container_id)
    image = container.image
    attrs = image.attrs if isinstance(image.attrs, Mapping) else {}
    repo_digests = [str(value) for value in attrs.get("RepoDigests") or []]
    config = attrs.get("Config")
    labels = config.get("Labels", {}) if isinstance(config, Mapping) else {}
    if not isinstance(labels, Mapping):
        labels = {}
    observed_digest = next(
        (value.rsplit("@", 1)[1] for value in repo_digests if "@sha256:" in value),
        None,
    )
    return {
        "observed_runner_image_id": str(getattr(image, "id", "") or "") or None,
        "observed_runner_image_digest": observed_digest,
        "observed_runner_repo_digests": repo_digests,
        "observed_gym_source_revision": str(labels.get(GYM_SOURCE_REVISION_LABEL) or "") or None,
        "observed_gym_package_version": str(labels.get(GYM_PACKAGE_VERSION_LABEL) or "") or None,
    }


def make_gym_docker_submitter(
    *,
    backend_name: str,
    image: str | None,
    env_file: str,
    work_dir: str,
    work_volume: str,
    host_env_file: str | None = None,
    runner: Callable[[list[str], Path, Path], None] | None = None,
) -> Callable[[LaunchSpec], LaunchHandle]:
    """Build a submitter that launches one gym-runner container per evaluation."""
    envf = resolve_env_file_path(env_file)
    work = Path(work_dir).expanduser().resolve()
    entrypoint = harness_run_and_collect_entrypoint(envf)

    def submit(spec: LaunchSpec) -> LaunchHandle:
        validate_gym_launch_contract(spec, backend_name=backend_name)
        selected_image = spec.runner_image_ref
        if not selected_image and spec.allow_live_runner_fallback:
            selected_image = image
        if not selected_image:
            raise RuntimeError(
                "gym runner image is missing from the execution snapshot; "
                "only legacy evaluations may fall back to GYM_RUNNER_IMAGE"
            )
        selected_image = _digest_pinned_image_ref(selected_image, spec.runner_image_digest)
        eval_work = work / spec.evaluation_id
        work.mkdir(parents=True, exist_ok=True)
        eval_work.mkdir(exist_ok=False)
        log_path = eval_work / "gym.log"
        output_jsonl = str(eval_work / "rollouts.jsonl")
        target_env = load_env_file(envf)
        if spec.framework_config:
            target_env.update(gym_profile_env(spec.framework_config))
        target_env, launch_metadata = materialize_gym_launch_env(
            spec,
            target_env,
            eval_work=eval_work,
            runner_task_path=Path(RUNNER_WORK_MOUNT) / spec.evaluation_id / "task",
        )
        write_env_file(eval_work / "target.env", {**target_env, **spec.credential_env})

        runner_work = Path(RUNNER_WORK_MOUNT) / spec.evaluation_id
        command = build_docker_run_and_collect_argv(
            evaluation_id=spec.evaluation_id,
            work_dir_in_runner=runner_work,
            env_file_in_runner=runner_work / "target.env",
        )

        observed_identity: dict[str, Any] = {}

        def _live_runner(_argv: list[str], _cwd: Path, log: Path) -> None:
            container_id = launch_gym_runner_container(
                image=selected_image,
                evaluation_id=spec.evaluation_id,
                env_file_host=None,
                work_volume=work_volume,
                command=command,
                entrypoint=entrypoint,
                shm_size=settings.gym_runner_shm_size,
            )
            try:
                observed_identity.update(inspect_gym_runner_container_identity(container_id))
                if spec.runner_image_digest and selected_image.endswith(f"@{spec.runner_image_digest}"):
                    observed_identity["observed_runner_image_digest"] = spec.runner_image_digest
                _validate_observed_gym_identity(spec, observed_identity)
            except Exception:
                _remove_failed_gym_container(container_id)
                raise
            log.write_text(
                f"launched gym-runner container {container_id}\n"
                f"image: {selected_image}\n"
                f"entrypoint: {' '.join(entrypoint)}\n"
                f"command: {' '.join(command)}\n"
                f"logs: docker logs gym-{spec.evaluation_id}\n"
                f"work: {eval_work}\n"
            )

        (runner or _live_runner)(command, work, log_path)

        return LaunchHandle(
            backend=backend_name,
            external_id=spec.evaluation_id,
            raw={
                "argv": command,
                "log": str(log_path),
                "output_jsonl": output_jsonl,
                "gym_runner_image": selected_image,
                "expected_runner_image_digest": spec.runner_image_digest,
                "expected_gym_source_revision": spec.runner_source_revision,
                "expected_gym_package_version": spec.runner_package_version,
                **{key: value for key, value in observed_identity.items() if value is not None},
                "command": "run_and_collect",
                "effective_runtime_settings": launch_metadata,
                "docker": True,
                "runner_container_name": _gym_runner_container_name(spec.evaluation_id),
                "runner_stop_timeout_s": settings.gym_runner_teardown_timeout_seconds,
                "remote_teardown": {
                    "strategy": "gym_runner_sigterm",
                    "providers": ["daytona", "opensandbox"],
                },
            },
        )

    return submit


def _digest_pinned_image_ref(image_ref: str, digest: str | None) -> str:
    if not digest:
        return image_ref
    if "@" in image_ref:
        current = image_ref.rsplit("@", 1)[1]
        if current != digest:
            raise RuntimeError(f"gym runner image reference digest {current!r} does not match {digest!r}")
        return image_ref
    last_slash = image_ref.rfind("/")
    last_colon = image_ref.rfind(":")
    repository = image_ref[:last_colon] if last_colon > last_slash else image_ref
    return f"{repository}@{digest}"


def _validate_observed_gym_identity(
    spec: LaunchSpec,
    observed: Mapping[str, Any],
) -> None:
    checks = (
        (
            "Gym source revision",
            spec.runner_source_revision,
            observed.get("observed_gym_source_revision"),
        ),
        (
            "Gym package version",
            spec.runner_package_version,
            observed.get("observed_gym_package_version"),
        ),
    )
    for label, expected, actual in checks:
        if expected and actual != expected:
            raise RuntimeError(f"{label} mismatch: expected {expected!r}, observed {actual!r}")
    expected_digest = spec.runner_image_digest
    observed_digest = observed.get("observed_runner_image_digest")
    if expected_digest and observed_digest and observed_digest != expected_digest:
        raise RuntimeError(
            f"Gym runner image digest mismatch: expected {expected_digest!r}, observed {observed_digest!r}"
        )


def _remove_failed_gym_container(container_id: str) -> None:
    import docker

    with suppress(Exception):
        docker.from_env().containers.get(container_id).remove(force=True)


def rollouts_jsonl_to_result_envelope(path: Path, *, evaluation_id: str) -> dict[str, Any]:
    """Convert ``ng_collect_rollouts`` output to a Harbor-shaped result envelope."""
    rollouts: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            rollouts.append(json.loads(stripped))

    rewards = [float(r["reward"]) for r in rollouts if r.get("reward") is not None]
    reward_buckets: dict[str, list[str]] = {}
    exception_stats: dict[str, int] = {}
    for index, rollout in enumerate(rollouts):
        trial_id = str(rollout.get("id") or f"rollout-{index}")
        if rollout.get("reward") is not None:
            reward = str(float(rollout["reward"]))
            reward_buckets.setdefault(reward, []).append(trial_id)
        if _rollout_has_error(rollout):
            category = _rollout_exception_category(rollout)
            exception_stats[category] = exception_stats.get(category, 0) + 1
    aggregate_metrics = _load_rollouts_aggregate_metrics(path)
    row_errors = sum(1 for r in rollouts if _rollout_has_error(r))
    aggregate_errors = _aggregate_eval_error_count(aggregate_metrics, total_rollouts=len(rollouts))
    n_errored = max(row_errors, aggregate_errors)
    n_completed = max(0, len(rollouts) - n_errored)
    mean_reward = sum(rewards) / len(rewards) if rewards else None
    finished_at = datetime.now(tz=UTC).isoformat()
    metric: dict[str, Any] = {}
    if mean_reward is not None:
        metric["mean"] = mean_reward
    metric.update(_aggregate_key_metrics(aggregate_metrics))

    return {
        "id": evaluation_id,
        "finished_at": finished_at,
        "n_total_trials": len(rollouts) if rollouts else None,
        "stats": {
            "n_completed_trials": n_completed if rollouts else None,
            "n_errored_trials": n_errored if rollouts else None,
            "evals": {
                "gym_rollouts": {
                    "n_trials": len(rollouts),
                    "n_errors": n_errored,
                    "metrics": [metric] if metric else [],
                    "reward_stats": {"reward": reward_buckets},
                    "exception_stats": exception_stats,
                }
            },
        },
        "source": "gym_rollouts_jsonl",
        "output_jsonl": str(path),
    }


def _load_rollouts_aggregate_metrics(rollouts_path: Path) -> list[dict[str, Any]]:
    """Read Gym's aggregate metrics sidecar when ``ng_collect_rollouts`` wrote one."""
    metrics_path = rollouts_path.with_name("rollouts_aggregate_metrics.json")
    if not metrics_path.is_file():
        return []
    try:
        raw = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _rollout_has_error(rollout: Mapping[str, Any]) -> bool:
    """Best-effort row-level Gym error detection.

    Some Gym agents only expose eval errors in the aggregate sidecar, but direct
    rollout fields are still useful for older/newer harnesses.
    """
    status = str(rollout.get("status") or "").lower()
    if status in {"error", "errored", "failed"}:
        return True
    if rollout.get("error"):
        return True
    response = rollout.get("response")
    if isinstance(response, Mapping):
        response_status = str(response.get("status") or "").lower()
        if response_status in {"error", "errored", "failed"}:
            return True
        if response.get("error"):
            return True
    return False


def _rollout_exception_category(rollout: Mapping[str, Any]) -> str:
    """Extract a stable diagnostic category without retaining secret/error text."""
    candidates = [rollout.get("exception_info"), rollout.get("error")]
    response = rollout.get("response")
    if isinstance(response, Mapping):
        candidates.extend((response.get("exception_info"), response.get("error")))
    for value in candidates:
        if isinstance(value, Mapping):
            for key in ("type", "name", "class", "exception_type"):
                category = value.get(key)
                if isinstance(category, str) and category:
                    return category
        elif isinstance(value, str) and value:
            # Keep only an exception-like leading token, never full messages.
            token = value.split(":", 1)[0].split()[-1]
            if token.endswith(("Error", "Exception")):
                return token
    return "GymRolloutError"


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _aggregate_eval_error_count(aggregate_metrics: list[dict[str, Any]], *, total_rollouts: int) -> int:
    """Return Gym's aggregate eval-error count, estimating from rate if needed."""
    count = 0
    max_rate = 0.0
    for item in aggregate_metrics:
        agent_metrics = item.get("agent_metrics")
        if isinstance(agent_metrics, Mapping):
            if (value := _as_int(agent_metrics.get("eval_error_rollout_count"))) is not None:
                count += value
            if (rate := _as_float(agent_metrics.get("eval_error_rate"))) is not None:
                max_rate = max(max_rate, rate)
        key_metrics = item.get("key_metrics")
        if isinstance(key_metrics, Mapping) and (rate := _as_float(key_metrics.get("eval_error_rate"))) is not None:
            max_rate = max(max_rate, rate)

    if count:
        return min(count, total_rollouts) if total_rollouts > 0 else count
    if max_rate <= 0 or total_rollouts <= 0:
        return 0
    return max(1, min(total_rollouts, round((max_rate / 100.0) * total_rollouts)))


def _aggregate_key_metrics(aggregate_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Promote useful Gym aggregate fields into the persisted summary metric."""
    promoted: dict[str, float] = {}
    for item in aggregate_metrics:
        key_metrics = item.get("key_metrics")
        if not isinstance(key_metrics, Mapping):
            continue
        for key in ("eval_error_rate", "tests_status_rate", "resolved_task_rate"):
            value = _as_float(key_metrics.get(key))
            if value is not None:
                promoted[key] = max(promoted.get(key, value), value)
    return promoted


def gym_error_summary(result: Mapping[str, Any]) -> str | None:
    """Summarize Gym eval errors that should make a run operationally failed."""
    stats = result.get("stats")
    if not isinstance(stats, Mapping):
        return None
    n_errored = stats.get("n_errored_trials") or 0
    if not n_errored:
        return None
    total = result.get("n_total_trials")
    head = f"{n_errored}/{total} rollouts errored" if total else f"{n_errored} rollouts errored"
    evals = stats.get("evals")
    if isinstance(evals, Mapping):
        gym_rollouts = evals.get("gym_rollouts")
        if isinstance(gym_rollouts, Mapping):
            metrics = gym_rollouts.get("metrics")
            if isinstance(metrics, list) and metrics and isinstance(metrics[0], Mapping):
                rate = _as_float(metrics[0].get("eval_error_rate"))
                if rate is not None:
                    return f"{head}; eval_error_rate={rate:g}%"
    return head


def _status_from_gym_result(result: dict[str, Any]) -> RuntimeStatus:
    if error_summary := gym_error_summary(result):
        return RuntimeStatus(
            phase="failed",
            detail=f"gym run finished with errored rollouts ({error_summary})",
            raw=result,
        )
    if not result.get("n_total_trials"):
        return RuntimeStatus(
            phase="failed",
            detail="gym run produced no rollouts",
            raw=result,
        )
    return RuntimeStatus(phase="succeeded", raw=result)


def _gym_runner_container_name(evaluation_id: str) -> str:
    return f"gym-{evaluation_id}"


def make_gym_docker_status_reader(*, work_dir: str) -> StatusReader:
    """Poll gym-runner container exit and read rollouts from the shared work volume."""
    work = Path(work_dir).expanduser().resolve()

    def read(handle: LaunchHandle) -> RuntimeStatus:
        eval_id = handle.external_id
        eval_work = work / eval_id
        output_jsonl = eval_work / "rollouts.jsonl"

        from docker.errors import NotFound

        import docker

        client = docker.from_env()
        container_name = _gym_runner_container_name(eval_id)
        try:
            container = client.containers.get(container_name)
        except NotFound:
            if output_jsonl.is_file():
                result = rollouts_jsonl_to_result_envelope(output_jsonl, evaluation_id=eval_id)
                return _status_from_gym_result(result)
            return RuntimeStatus(phase="running", detail="awaiting gym-runner container")

        container.reload()
        state = container.attrs.get("State") or {}
        if state.get("Running"):
            return RuntimeStatus(phase="running", detail="gym-runner container running")

        exit_code = state.get("ExitCode")
        if exit_code not in (0, None):
            tail = ""
            try:
                logs = container.logs(tail=80).decode("utf-8", errors="replace")
                tail = logs[-800:] if len(logs) > 800 else logs
            except Exception:  # noqa: BLE001 — container may already be removed
                run_log = eval_work / "ng_run.log"
                gym_log = eval_work / "gym.log"
                if run_log.is_file():
                    text = run_log.read_text(errors="replace")
                    tail = text[-800:] if len(text) > 800 else text
                elif gym_log.is_file():
                    text = gym_log.read_text(errors="replace")
                    tail = text[-800:] if len(text) > 800 else text
            return RuntimeStatus(
                phase="failed",
                detail=f"gym-runner exited {exit_code}: {tail}",
            )

        if not output_jsonl.is_file():
            run_log = eval_work / "ng_run.log"
            detail = "gym-runner exited 0 but rollouts.jsonl missing"
            if run_log.is_file():
                detail = f"{detail}; see {run_log}"
            return RuntimeStatus(phase="failed", detail=detail)

        result = rollouts_jsonl_to_result_envelope(output_jsonl, evaluation_id=eval_id)
        return _status_from_gym_result(result)

    return read


def make_gym_docker_terminator() -> Callable[[LaunchHandle], None]:
    """Gracefully stop and remove the one-shot gym-runner container.

    Gym sandbox providers are owned by the runner process. A force-remove skips
    the harness ``TERM``/``EXIT`` traps and can leave Daytona/OpenSandbox
    sandboxes alive. Stopping first sends SIGTERM and waits for Gym to close its
    remote provider handles; remove is only the final local-container cleanup.
    """

    def teardown(handle: LaunchHandle) -> None:
        from docker.errors import NotFound

        import docker

        client = docker.from_env()
        container_name = str(handle.raw.get("runner_container_name") or _gym_runner_container_name(handle.external_id))
        stop_timeout = handle.raw.get("runner_stop_timeout_s")
        if not isinstance(stop_timeout, int) or stop_timeout < 0:
            stop_timeout = settings.gym_runner_teardown_timeout_seconds
        try:
            container = client.containers.get(container_name)
        except NotFound:
            pass
        else:
            container.stop(timeout=stop_timeout)
            container.remove(force=True)

    return teardown
