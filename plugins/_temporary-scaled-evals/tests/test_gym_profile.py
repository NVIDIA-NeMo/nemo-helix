# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gym framework-profile validation and harness materialization tests."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.dispatch.gym.common import make_gym_submitter
from scaled_evals.dispatch.gym.profile import gym_profile_env, validate_gym_profile_config
from scaled_evals.models.gym_identity import gym_run_identity
from scaled_evals.models.runtime import LaunchSpec


def _profile(*, agent_name: str = "mini_swe_agent_2", limit: int = 2) -> dict:
    return {
        "schema_version": "1",
        "command": "run_and_collect",
        "config_paths": [
            "/harness/gym-sandbox-opensandbox/configs/mini_swe_agent_opensandbox_smoke.yaml",
            "responses_api_models/openai_model/configs/openai_model.yaml",
        ],
        "agent_name": agent_name,
        "limit": limit,
        "num_samples_in_parallel": 4,
        "responses_create_params": {"temperature": 0.5},
        "overrides": {"skip_venv_if_present": True},
    }


def _spec(profile: dict, *, evaluation_id: str = "ev_gym_profile") -> LaunchSpec:
    return LaunchSpec(
        evaluation_id=evaluation_id,
        name="profiled gym",
        framework="nemo_gym",
        image_ref="registry.example/task:1",
        parallelism=1,
        framework_config=profile,
        credential_env={"POLICY_API_KEY": "secret"},
    )


def test_gym_profile_translates_canonical_fields_to_harness_env() -> None:
    env = gym_profile_env(_profile(agent_name="experimental_agent", limit=7))

    assert env["GYM_AGENT_NAME"] == "experimental_agent"
    assert env["GYM_COMMAND"] == "run_and_collect"
    assert env["GYM_LIMIT"] == "7"
    assert env["GYM_CONFIG_PATHS"].startswith("/harness/gym-sandbox-opensandbox/")
    assert "++num_samples_in_parallel=4" in env["GYM_EXTRA_OVERRIDES"]
    assert "++limit=7" in env["GYM_EXTRA_OVERRIDES"]
    assert "++responses_create_params.temperature=0.5" in env["GYM_EXTRA_OVERRIDES"]
    assert "++skip_venv_if_present=true" in env["GYM_EXTRA_OVERRIDES"]


@pytest.mark.parametrize(
    "patch",
    [
        {"OPENSANDBOX_DOMAIN": "https://attacker.invalid"},
        {"overrides": {"sandbox_provider.opensandbox.connection.domain": "attacker"}},
        {"overrides": {"policy_base_url": "https://attacker.invalid"}},
        {"config_paths": ["../../etc/passwd"]},
        {"config_paths": ["/etc/passwd"]},
    ],
)
def test_gym_profile_rejects_substrate_overrides_and_unsafe_paths(patch: dict) -> None:
    profile = {**_profile(), **patch}
    with pytest.raises(ValueError):
        validate_gym_profile_config(profile)


def test_gym_submitter_materializes_each_evaluations_snapshotted_profile(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    (harness / "targets").mkdir(parents=True)
    script = harness / "run_and_collect.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    env_file = harness / "targets" / "opensandbox.env"
    env_file.write_text(
        "OPENSANDBOX_DOMAIN=https://operator.example\nGYM_AGENT_NAME=chart-wide-agent-must-not-win\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def runner(argv, _cwd, _log):  # noqa: ANN001, ANN202
        calls.append(argv)

    submit = make_gym_submitter(
        backend_name="gym_sandbox_opensandbox",
        gym_dir=str(tmp_path),
        env_file=str(env_file),
        work_dir=str(tmp_path / "work"),
        runner=runner,
    )
    submit(_spec(_profile(agent_name="agent_one")))
    submit(
        _spec(
            _profile(agent_name="agent_two", limit=9),
            evaluation_id="ev_gym_profile_two",
        )
    )

    target = tmp_path / "work" / "ev_gym_profile" / "target.env"
    rendered = target.read_text(encoding="utf-8")
    assert "GYM_AGENT_NAME=agent_one" in rendered
    assert "chart-wide-agent-must-not-win" not in rendered
    assert "OPENSANDBOX_DOMAIN=https://operator.example" in rendered
    assert "POLICY_API_KEY=secret" in rendered
    assert calls[0][calls[0].index("--env-file") + 1] == str(target)
    second_target = tmp_path / "work" / "ev_gym_profile_two" / "target.env"
    second_rendered = second_target.read_text(encoding="utf-8")
    assert "GYM_AGENT_NAME=agent_two" in second_rendered
    assert "GYM_LIMIT=9" in second_rendered
    assert calls[1][calls[1].index("--env-file") + 1] == str(second_target)


def test_gym_submitter_materializes_launchspec_task_and_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = tmp_path / "harness"
    (harness / "targets").mkdir(parents=True)
    (harness / "run_and_collect.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    env_file = harness / "targets" / "opensandbox.env"
    env_file.write_text("GYM_AGENT_NAME=chart-wide-agent\n", encoding="utf-8")
    staged_calls: list[Path] = []

    def fake_stage(_object_key: str, dest: Path) -> Path:
        dest.mkdir(parents=True)
        (dest / "task.toml").write_text("[task]\n", encoding="utf-8")
        staged_calls.append(dest)
        return dest

    monkeypatch.setattr("scaled_evals.dispatch.gym.common._stage_task_tree", fake_stage)
    monkeypatch.setattr(
        "scaled_evals.dispatch.gym.common._inject_extra_skills",
        lambda _task_tree, keys: [
            {"object_key": key, "status": "staged", "staged_path": "environment/skills/x/SKILL.md"} for key in keys
        ],
    )

    submit = make_gym_submitter(
        backend_name="gym_sandbox_opensandbox",
        gym_dir=str(tmp_path),
        env_file=str(env_file),
        work_dir=str(tmp_path / "work"),
        runner=lambda _argv, _cwd, _log: None,
    )
    handle = submit(
        LaunchSpec(
            evaluation_id="ev_gym_contract",
            name="profiled gym",
            framework="nemo_gym",
            image_ref="registry.example/task:1",
            image_digest="sha256:" + "a" * 64,
            parallelism=3,
            n_attempts=2,
            tarball_object_key="tasks/task/rev/1/tarball.tar.gz",
            extra_skill_object_keys=["skills/debug/SKILL.md"],
            framework_config=_profile(agent_name="agent_contract"),
        )
    )

    target = tmp_path / "work" / "ev_gym_contract" / "target.env"
    rendered = target.read_text(encoding="utf-8")
    assert "GYM_AGENT_NAME=agent_contract" in rendered
    assert "TASK_IMAGE=registry.example/task:1" in rendered
    assert "TASK_PATH=" + str(staged_calls[0]) in rendered
    assert "SCALED_EVALS_TASK_PACK_OBJECT_KEY=tasks/task/rev/1/tarball.tar.gz" in rendered
    assert "++num_samples_in_parallel=3" in rendered
    assert "++num_repeats=2" in rendered
    assert handle.raw["effective_runtime_settings"]["task_pack_staged"] is True
    materials = handle.raw["effective_runtime_settings"]["extra_skill_materials"]
    assert materials[0]["status"] == "staged"


def test_gym_submitter_rejects_unsupported_network_policy(tmp_path: Path) -> None:
    env_file = tmp_path / "daytona.env"
    env_file.write_text("GYM_AGENT_NAME=mini_swe_agent_2\n", encoding="utf-8")
    submit = make_gym_submitter(
        backend_name="gym_sandbox_opensandbox",
        gym_dir=str(tmp_path),
        env_file=str(env_file),
        work_dir=str(tmp_path / "work"),
        runner=lambda _argv, _cwd, _log: None,
    )

    with pytest.raises(RuntimeError, match="does not support network_policy='default_deny'"):
        submit(
            LaunchSpec(
                evaluation_id="ev_gym_network",
                name="network policy",
                framework="nemo_gym",
                image_ref="registry.example/task:1",
                parallelism=1,
                network_policy="default_deny",
            )
        )


def test_gym_submitter_rejects_instruction_patch_without_uploaded_pack(tmp_path: Path) -> None:
    env_file = tmp_path / "daytona.env"
    env_file.write_text("GYM_AGENT_NAME=mini_swe_agent_2\n", encoding="utf-8")
    submit = make_gym_submitter(
        backend_name="gym_sandbox_opensandbox",
        gym_dir=str(tmp_path),
        env_file=str(env_file),
        work_dir=str(tmp_path / "work"),
        runner=lambda _argv, _cwd, _log: None,
    )

    with pytest.raises(RuntimeError, match="task-tree mutations require a staged uploaded task"):
        submit(
            LaunchSpec(
                evaluation_id="ev_gym_instruction",
                name="instruction patch",
                framework="nemo_gym",
                image_ref="registry.example/task:1",
                parallelism=1,
                instruction_prefix="read this first",
            )
        )


def test_gym_submitter_rejects_instruction_patch_when_pack_has_no_task_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / "daytona.env"
    env_file.write_text("GYM_AGENT_NAME=mini_swe_agent_2\n", encoding="utf-8")
    monkeypatch.setattr("scaled_evals.dispatch.gym.common._stage_task_tree", lambda *_args: None)
    submit = make_gym_submitter(
        backend_name="gym_sandbox_opensandbox",
        gym_dir=str(tmp_path),
        env_file=str(env_file),
        work_dir=str(tmp_path / "work"),
        runner=lambda _argv, _cwd, _log: None,
    )

    with pytest.raises(RuntimeError, match="task-tree mutations require an uploaded task pack"):
        submit(
            LaunchSpec(
                evaluation_id="ev_gym_instruction",
                name="instruction patch",
                framework="nemo_gym",
                image_ref="registry.example/task:1",
                parallelism=1,
                tarball_object_key="tasks/task/rev/1/tarball.tar.gz",
                instruction_postfix="finish here",
            )
        )


def test_ng_collect_profile_requires_an_input_file() -> None:
    profile = {**_profile(), "command": "ng_collect_rollouts"}
    with pytest.raises(ValueError, match="input_jsonl_fpath is required"):
        validate_gym_profile_config(profile)


def test_gym_identity_records_the_snapshotted_profile_agent() -> None:
    profile = _profile(agent_name="agent_two")
    profile.pop("limit")
    profile.pop("num_samples_in_parallel")
    snapshot = {
        "schema_version": "scaled-evals-execution-inputs-v1",
        "captured_at": "2026-07-13T00:00:00+00:00",
        "evaluation": {
            "framework": "nemo_gym",
            "framework_profile_id": "cfg_gym",
            "runtime": "gym_sandbox_opensandbox",
            "runner_metadata": {},
        },
        "task": {},
        "profiles": {
            "framework": {
                "id": "cfg_gym",
                "type": "gym",
                "config": profile,
            }
        },
        "credentials": {},
        "submission_identity": {},
    }

    identity = gym_run_identity(
        {
            "execution_snapshot": snapshot,
            "parallelism": 8,
            "n_attempts": 2,
            "dispatch_job_name": "job-one",
            "dispatch_job_uid": "uid-one",
            "backend_handle": {
                "raw": {
                    "command": "ng_e2e_collect_rollouts",
                    "process": True,
                    "process_owner_pod": "pod-one",
                }
            },
        }
    )

    assert identity is not None
    assert identity["agent_path"] == "agent_two"
    assert identity["profile"]["command_verification"] == "mismatch"
    assert identity["profile"]["effective_limit"] is None
    assert identity["profile"]["effective_num_samples_in_parallel"] is None
    assert identity["executor"] == {
        "mode": "process",
        "dispatch_job_name": "job-one",
        "dispatch_job_uid": "uid-one",
        "runner_pod_name": "pod-one",
        "runner_pod_name_source": "backend-handle",
    }

    matched = gym_run_identity(
        {
            "execution_snapshot": snapshot,
            "backend_handle": {
                "raw": {
                    "command": "run_and_collect",
                    "process": True,
                    "effective_runtime_settings": {
                        "parallelism": 5,
                        "n_attempts": 3,
                        "task_image_ref": "registry.example/task:1",
                        "network_policy": "unrestricted",
                    },
                }
            },
        }
    )
    assert matched is not None
    assert matched["profile"]["command_verification"] == "matched"
    assert matched["profile"]["effective_limit"] == 1
    assert matched["profile"]["effective_num_samples_in_parallel"] == 5
    assert matched["profile"]["control_plane_attempts"] == 3
    assert matched["runtime_settings"]["task_image_ref"] == "registry.example/task:1"
