# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gym-runner Docker dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.dispatch.gym.docker import (
    harness_run_and_collect_entrypoint,
    host_env_file_path,
    inspect_gym_runner_container_identity,
    launch_gym_runner_container,
    make_gym_docker_status_reader,
    make_gym_docker_terminator,
    rollouts_jsonl_to_result_envelope,
)
from scaled_evals.dispatch.runtime_backend import LaunchHandle


def test_launch_gym_runner_container_sets_shm_size(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeContainer:
        id = "deadbeef"

    def fake_run(**kwargs: object) -> FakeContainer:
        captured.update(kwargs)
        return FakeContainer()

    fake_client = MagicMock()
    fake_client.containers.run = fake_run
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    container_id = launch_gym_runner_container(
        image="scaled-evals-gym-runner:dev",
        evaluation_id="ev_test123",
        env_file_host=Path("/tmp/daytona.env"),
        work_volume="scaled-evals-gym-sandbox-work",
        command=["--env-file", "/run/daytona.env"],
        entrypoint=["/harness/gym-daytona/run_and_collect.sh"],
        shm_size="2g",
    )

    assert container_id == "deadbeef"
    assert captured["shm_size"] == "2g"
    assert captured["name"] == "gym-ev_test123"
    assert captured["entrypoint"] == ["/harness/gym-daytona/run_and_collect.sh"]
    assert captured["environment"] == {
        "UV_NO_PROJECT": "1",
        "RAY_TMPDIR": "/tmp/ray",
        "RAY_ENABLE_DASHBOARD": "0",
        "RAY_USAGE_STATS_ENABLED": "0",
    }


def test_inspect_gym_runner_container_identity_reads_digest_and_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeImage:
        id = "sha256:image-id"
        attrs = {
            "RepoDigests": ["registry.example/gym@sha256:" + "a" * 64],
            "Config": {
                "Labels": {
                    "com.nvidia.nemo-gym.revision": "b" * 40,
                    "com.nvidia.nemo-gym.version": "0.4.0",
                }
            },
        }

    class FakeContainer:
        image = FakeImage()

    fake_client = MagicMock()
    fake_client.containers.get.return_value = FakeContainer()
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    identity = inspect_gym_runner_container_identity("container-id")

    assert identity["observed_runner_image_id"] == "sha256:image-id"
    assert identity["observed_runner_image_digest"] == "sha256:" + "a" * 64
    assert identity["observed_gym_source_revision"] == "b" * 40
    assert identity["observed_gym_package_version"] == "0.4.0"


def test_gym_docker_submitter_persists_remote_teardown_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scaled_evals.api.settings import settings
    from scaled_evals.dispatch.gym.docker import make_gym_docker_submitter
    from scaled_evals.dispatch.runtime_backend import LaunchSpec

    env_file = tmp_path / "daytona.env"
    env_file.write_text("GYM_AGENT_NAME=mini_swe_agent_2\n", encoding="utf-8")
    monkeypatch.setattr(settings, "gym_runner_teardown_timeout_seconds", 17)

    submit = make_gym_docker_submitter(
        backend_name="gym_sandbox_daytona",
        image="scaled-evals-gym-runner:dev",
        env_file=str(env_file),
        work_dir=str(tmp_path / "work"),
        work_volume="scaled-evals-gym-sandbox-work",
        runner=lambda _argv, _cwd, _log: None,
    )

    handle = submit(
        LaunchSpec(
            evaluation_id="ev_test123",
            name="test",
            framework="harbor",
            runner_image_ref="registry.example/gym:frozen",
            image_ref="image:tag",
            parallelism=1,
        )
    )

    assert handle.raw["gym_runner_image"] == "registry.example/gym:frozen"
    assert handle.raw["runner_container_name"] == "gym-ev_test123"
    assert handle.raw["runner_stop_timeout_s"] == 17
    assert handle.raw["remote_teardown"] == {
        "strategy": "gym_runner_sigterm",
        "providers": ["daytona", "opensandbox"],
    }


def test_gym_docker_submitter_writes_materialized_launchspec_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scaled_evals.dispatch.gym.docker import make_gym_docker_submitter
    from scaled_evals.dispatch.runtime_backend import LaunchSpec

    env_file = tmp_path / "daytona.env"
    env_file.write_text("GYM_AGENT_NAME=chart-wide-agent\n", encoding="utf-8")
    monkeypatch.setattr(
        "scaled_evals.dispatch.gym.common._stage_task_tree",
        lambda _object_key, dest: dest.mkdir(parents=True) or dest,
    )
    submit = make_gym_docker_submitter(
        backend_name="gym_sandbox_daytona",
        image="scaled-evals-gym-runner:dev",
        env_file=str(env_file),
        work_dir=str(tmp_path / "work"),
        work_volume="scaled-evals-gym-sandbox-work",
        runner=lambda _argv, _cwd, _log: None,
    )

    handle = submit(
        LaunchSpec(
            evaluation_id="ev_docker_contract",
            name="test",
            framework="nemo_gym",
            runner_image_ref="registry.example/gym:frozen",
            image_ref="registry.example/task:1",
            parallelism=4,
            tarball_object_key="tasks/task/rev/1/tarball.tar.gz",
            framework_config={
                "schema_version": "1",
                "command": "run_and_collect",
                "config_paths": ["/opt/gym/configs/smoke.yaml"],
                "agent_name": "profile-agent",
            },
        )
    )

    rendered = (tmp_path / "work" / "ev_docker_contract" / "target.env").read_text(encoding="utf-8")
    assert "GYM_AGENT_NAME=profile-agent" in rendered
    assert "TASK_IMAGE=registry.example/task:1" in rendered
    assert "TASK_PATH=/work/ev_docker_contract/task" in rendered
    assert "++num_samples_in_parallel=4" in rendered
    assert handle.raw["effective_runtime_settings"]["task_path"] == "/work/ev_docker_contract/task"


def test_gym_docker_submitter_rejects_live_image_for_snapshot_run(tmp_path: Path) -> None:
    from scaled_evals.dispatch.gym.docker import make_gym_docker_submitter
    from scaled_evals.dispatch.runtime_backend import LaunchSpec

    env_file = tmp_path / "daytona.env"
    env_file.write_text("GYM_AGENT_NAME=mini_swe_agent_2\n", encoding="utf-8")
    submit = make_gym_docker_submitter(
        backend_name="gym_sandbox_daytona",
        image="mutable-live:latest",
        env_file=str(env_file),
        work_dir=str(tmp_path / "work"),
        work_volume="gym-work",
        runner=lambda _argv, _cwd, _log: None,
    )

    with pytest.raises(RuntimeError, match="missing from the execution snapshot"):
        submit(
            LaunchSpec(
                evaluation_id="ev_snapshot",
                name="snapshot",
                framework="nemo_gym",
                image_ref="task:tag",
                parallelism=1,
            )
        )


def test_gym_docker_submitter_digest_pins_snapshotted_image(tmp_path: Path) -> None:
    from scaled_evals.dispatch.gym.docker import make_gym_docker_submitter
    from scaled_evals.dispatch.runtime_backend import LaunchSpec

    digest = "sha256:" + "d" * 64
    env_file = tmp_path / "daytona.env"
    env_file.write_text("GYM_AGENT_NAME=mini_swe_agent_2\n", encoding="utf-8")
    submit = make_gym_docker_submitter(
        backend_name="gym_sandbox_daytona",
        image="mutable-live:latest",
        env_file=str(env_file),
        work_dir=str(tmp_path / "work"),
        work_volume="gym-work",
        runner=lambda _argv, _cwd, _log: None,
    )

    handle = submit(
        LaunchSpec(
            evaluation_id="ev_snapshot",
            name="snapshot",
            framework="nemo_gym",
            runner_image_ref="registry.example:5000/team/gym:latest",
            runner_image_digest=digest,
            image_ref="task:tag",
            parallelism=1,
        )
    )

    assert handle.raw["gym_runner_image"] == f"registry.example:5000/team/gym@{digest}"


def test_gym_docker_terminator_stops_before_remove(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeContainer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def stop(self, *, timeout: int) -> None:
            self.calls.append(("stop", timeout))

        def remove(self, *, force: bool) -> None:
            self.calls.append(("remove", force))

    fake_container = FakeContainer()
    fake_client = MagicMock()
    fake_client.containers.get.return_value = fake_container
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    terminate = make_gym_docker_terminator()
    terminate(
        LaunchHandle(
            backend="gym_sandbox_opensandbox",
            external_id="ev_test123",
            raw={"runner_container_name": "gym-ev_test123", "runner_stop_timeout_s": 12},
        )
    )

    fake_client.containers.get.assert_called_once_with("gym-ev_test123")
    assert fake_container.calls == [("stop", 12), ("remove", True)]


def test_gym_docker_terminator_tolerates_missing_container(monkeypatch: pytest.MonkeyPatch) -> None:
    from docker.errors import NotFound

    fake_client = MagicMock()
    fake_client.containers.get.side_effect = NotFound("missing")
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    terminate = make_gym_docker_terminator()
    terminate(LaunchHandle(backend="gym_sandbox_daytona", external_id="ev_missing"))

    fake_client.containers.get.assert_called_once_with("gym-ev_missing")


def test_gym_docker_terminator_surfaces_stop_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeContainer:
        remove = MagicMock()

        def stop(self, *, timeout: int) -> None:
            raise TimeoutError(f"timed out after {timeout}s")

    fake_client = MagicMock()
    fake_client.containers.get.return_value = FakeContainer()
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    terminate = make_gym_docker_terminator()
    with pytest.raises(TimeoutError, match="timed out after 60s"):
        terminate(LaunchHandle(backend="gym_sandbox_daytona", external_id="ev_timeout"))


def test_rollouts_jsonl_to_result_envelope_averages_rewards(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps({"reward": 1.0, "id": "a"}) + "\n" + json.dumps({"reward": 0.0, "id": "b"}) + "\n")
    result = rollouts_jsonl_to_result_envelope(rollouts, evaluation_id="ev_test")
    assert result["n_total_trials"] == 2
    assert result["stats"]["n_completed_trials"] == 2
    assert result["stats"]["evals"]["gym_rollouts"]["metrics"][0]["mean"] == 0.5
    assert result["stats"]["evals"]["gym_rollouts"]["reward_stats"]["reward"] == {
        "0.0": ["b"],
        "1.0": ["a"],
    }


def test_rollouts_jsonl_to_result_envelope_counts_aggregate_eval_errors(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps({"reward": 0.0, "response": {"status": "completed"}}) + "\n")
    (tmp_path / "rollouts_aggregate_metrics.json").write_text(
        json.dumps(
            [
                {
                    "agent_metrics": {
                        "eval_error_rollout_count": 1,
                        "eval_error_rate": 100.0,
                    },
                    "key_metrics": {
                        "eval_error_rate": 100.0,
                        "tests_status_rate": 0.0,
                        "resolved_task_rate": 0.0,
                    },
                }
            ]
        )
    )

    result = rollouts_jsonl_to_result_envelope(rollouts, evaluation_id="ev_test")

    assert result["stats"]["n_completed_trials"] == 0
    assert result["stats"]["n_errored_trials"] == 1
    gym_rollouts = result["stats"]["evals"]["gym_rollouts"]
    assert gym_rollouts["n_errors"] == 1
    assert gym_rollouts["metrics"][0]["mean"] == 0.0
    assert gym_rollouts["metrics"][0]["eval_error_rate"] == 100.0


def test_gym_docker_status_reader_running_while_container_up(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    eval_id = "ev_test123"
    eval_work = tmp_path / eval_id
    eval_work.mkdir()

    class FakeContainer:
        attrs = {"State": {"Running": True, "ExitCode": None}}

        def reload(self) -> None:
            pass

    fake_client = MagicMock()
    fake_client.containers.get.return_value = FakeContainer()
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    read = make_gym_docker_status_reader(work_dir=str(tmp_path))
    handle = LaunchHandle(backend="gym_sandbox_daytona", external_id=eval_id)
    assert read(handle).phase == "running"


def test_gym_docker_status_reader_succeeded_on_exit_with_rollouts(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    eval_id = "ev_test123"
    eval_work = tmp_path / eval_id
    eval_work.mkdir()
    rollouts = eval_work / "rollouts.jsonl"
    rollouts.write_text(json.dumps({"reward": 0.0}) + "\n")

    class FakeContainer:
        attrs = {"State": {"Running": False, "ExitCode": 0}}

        def reload(self) -> None:
            pass

        def logs(self, *, tail: int) -> bytes:
            return b"done"

    fake_client = MagicMock()
    fake_client.containers.get.return_value = FakeContainer()
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    read = make_gym_docker_status_reader(work_dir=str(tmp_path))
    handle = LaunchHandle(backend="gym_sandbox_daytona", external_id=eval_id)
    status = read(handle)
    assert status.phase == "succeeded"
    assert status.raw["stats"]["evals"]["gym_rollouts"]["metrics"][0]["mean"] == 0.0


def test_gym_docker_status_reader_fails_on_empty_rollouts(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    eval_id = "ev_test123"
    eval_work = tmp_path / eval_id
    eval_work.mkdir()
    (eval_work / "rollouts.jsonl").write_text("")

    class FakeContainer:
        attrs = {"State": {"Running": False, "ExitCode": 0}}

        def reload(self) -> None:
            pass

    fake_client = MagicMock()
    fake_client.containers.get.return_value = FakeContainer()
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    read = make_gym_docker_status_reader(work_dir=str(tmp_path))
    handle = LaunchHandle(backend="gym_sandbox_daytona", external_id=eval_id)
    status = read(handle)

    assert status.phase == "failed"
    assert status.detail == "gym run produced no rollouts"
    assert status.raw["n_total_trials"] is None


def test_gym_docker_status_reader_failed_on_zero_exit_with_eval_errors(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    eval_id = "ev_test123"
    eval_work = tmp_path / eval_id
    eval_work.mkdir()
    (eval_work / "rollouts.jsonl").write_text(json.dumps({"reward": 0.0, "response": {"status": "completed"}}) + "\n")
    (eval_work / "rollouts_aggregate_metrics.json").write_text(
        json.dumps(
            [
                {
                    "agent_metrics": {
                        "eval_error_rollout_count": 1,
                        "eval_error_rate": 100.0,
                    },
                    "key_metrics": {"eval_error_rate": 100.0},
                }
            ]
        )
    )

    class FakeContainer:
        attrs = {"State": {"Running": False, "ExitCode": 0}}

        def reload(self) -> None:
            pass

    fake_client = MagicMock()
    fake_client.containers.get.return_value = FakeContainer()
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    read = make_gym_docker_status_reader(work_dir=str(tmp_path))
    handle = LaunchHandle(backend="gym_sandbox_opensandbox", external_id=eval_id)
    status = read(handle)
    assert status.phase == "failed"
    assert "1/1 rollouts errored" in (status.detail or "")
    assert "eval_error_rate=100%" in (status.detail or "")
    assert status.raw["stats"]["n_errored_trials"] == 1


def test_gym_docker_status_reader_failed_on_nonzero_exit(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    eval_id = "ev_test123"
    eval_work = tmp_path / eval_id
    eval_work.mkdir()
    (eval_work / "ng_run.log").write_text("server timeout in ng_run")

    class FakeContainer:
        attrs = {"State": {"Running": False, "ExitCode": 1}}

        def reload(self) -> None:
            pass

        def logs(self, *, tail: int) -> bytes:
            raise RuntimeError("container removed")

    fake_client = MagicMock()
    fake_client.containers.get.return_value = FakeContainer()
    monkeypatch.setattr("docker.from_env", lambda: fake_client)

    read = make_gym_docker_status_reader(work_dir=str(tmp_path))
    handle = LaunchHandle(backend="gym_sandbox_daytona", external_id=eval_id)
    status = read(handle)
    assert status.phase == "failed"
    assert "exited 1" in (status.detail or "")
    assert "server timeout" in (status.detail or "")


def test_harness_run_and_collect_entrypoint_selects_harness() -> None:
    sandbox_env = Path("/harness/gym-sandbox-daytona/targets/daytona.env")
    opensandbox_env = Path("/harness/gym-sandbox-opensandbox/targets/opensandbox.env")
    daytona_env = Path("/harness/gym-daytona/targets/daytona.env")
    assert harness_run_and_collect_entrypoint(sandbox_env) == ["/harness/gym-sandbox-daytona/run_and_collect.sh"]
    assert harness_run_and_collect_entrypoint(opensandbox_env) == [
        "/harness/gym-sandbox-opensandbox/run_and_collect.sh"
    ]
    assert harness_run_and_collect_entrypoint(daytona_env) == ["/harness/gym-daytona/run_and_collect.sh"]


def test_host_env_file_path_anchors_explicit_relative_path_to_host_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scaled_evals.api.settings import settings

    monkeypatch.setattr(settings, "scaled_evals_host_dir", str(tmp_path))
    assert (
        host_env_file_path(
            Path("/harness/unrelated/targets/config.env"),
            explicit_host_env_file="examples/custom/host.env",
        )
        == (tmp_path / "examples/custom/host.env").resolve()
    )


def test_host_env_file_path_preserves_explicit_absolute_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scaled_evals.api.settings import settings

    monkeypatch.setattr(settings, "scaled_evals_host_dir", "/ignored")
    explicit = tmp_path / "host.env"

    assert (
        host_env_file_path(
            Path("/harness/unrelated/targets/config.env"),
            explicit_host_env_file=str(explicit),
        )
        == explicit.resolve()
    )


def test_host_env_file_path_uses_generic_harness_mapping_only_as_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scaled_evals.api.settings import settings

    monkeypatch.setattr(settings, "scaled_evals_host_dir", str(tmp_path))
    monkeypatch.setattr(settings, "gym_daytona_host_env_file", "/wrong/daytona.env")
    monkeypatch.setattr(settings, "gym_sandbox_daytona_host_env_file", "/wrong/sandbox.env")
    monkeypatch.setattr(settings, "gym_sandbox_opensandbox_host_env_file", "/wrong/open.env")

    assert (
        host_env_file_path(Path("/harness/unrelated/targets/config.env"))
        == (tmp_path / "examples/unrelated/targets/config.env").resolve()
    )
