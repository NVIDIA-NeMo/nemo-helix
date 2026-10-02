# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Gym environment package the staging step assembles for a registered agent."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from nemo_evaluator.api.schemas import AgentRef
from nemo_evaluator.jobs.gym_environment_package import (
    parse_environment_manifest,
    validate_environment_manifest_against_listing,
)
from nemo_evaluator.jobs.gym_registered_agent_package import (
    ENVIRONMENT_MOUNT_PATH,
    GymRegisteredAgentPackageSpec,
    download_wheels,
    host_gym_constraints,
    wheel_download_command,
    write_registered_agent_package,
)
from pytest_mock import MockerFixture

_CONFIG = {
    "metadata": {"name": "calc"},
    "harness": {"adapter_id": "nvidia.fabric.langchain.deepagents"},
    "runtime": {"timeout_seconds": 900},
    "skills": {"paths": ["skills/a"]},
}


def _spec(**overrides) -> GymRegisteredAgentPackageSpec:
    fields = dict(
        agent=AgentRef(root="dev/Calc-Agent"),
        resolved_config=_CONFIG,
        requirements=["nemo-fabric[deepagents,relay]==0.3.0", "mcp==1.29.0"],
    )
    fields.update(overrides)
    return GymRegisteredAgentPackageSpec(**fields)


def _fake_download(calls: list):
    def download(requirements, constraints, destination: Path, python_version: str, platform):
        calls.append((list(requirements), list(constraints), python_version, platform))
        (destination / "nemo_fabric-0.3.0-py3-none-any.whl").write_bytes(b"")

    return download


def _listing(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]


def test_the_package_is_a_valid_wheels_v1_environment_running_the_platform_component(tmp_path: Path) -> None:
    root = tmp_path / "environment"
    root.mkdir()
    ethos = tmp_path / "ethos"
    (ethos / "skills" / "a").mkdir(parents=True)
    (ethos / "skills" / "a" / "SKILL.md").write_text("# a")
    calls: list = []

    config_path = write_registered_agent_package(root, _spec(), agent_files=ethos, download=_fake_download(calls))

    assert config_path == "responses_api_agents/nemo_registered_agent/configs/registered_calc_agent.yaml"
    manifest = parse_environment_manifest((root / "nemo-environment.yaml").read_text())
    assert manifest.format == "wheels-v1" and list(manifest.config_paths) == [config_path]
    validate_environment_manifest_against_listing(manifest, _listing(root))
    component = root / "responses_api_agents" / "nemo_registered_agent"
    assert "class NeMoRegisteredAgent(" in (component / "app.py").read_text()
    assert (component / "requirements.txt").read_text().splitlines() == [
        "nemo-fabric[deepagents,relay]==0.3.0",
        "mcp==1.29.0",
    ]
    assert (component / "agents" / "registered_calc_agent" / "skills" / "a" / "SKILL.md").is_file()
    instance = yaml.safe_load((root / config_path).read_text())["registered_calc_agent"]["responses_api_agents"][
        "nemo_registered_agent"
    ]
    assert instance["fabric_config"] == _CONFIG
    assert (
        instance["fabric_config_base_dir"]
        == f"{ENVIRONMENT_MOUNT_PATH}/responses_api_agents/nemo_registered_agent/agents/registered_calc_agent"
    )
    assert instance["resources_server"] == {
        "type": "resources_servers",
        "name": "???",
    }  # bound by the resolver's Hydra override
    assert instance["model_server"] == {"type": "responses_api_models", "name": "policy_model"}
    assert "timeout" not in instance  # the agent's own deadline travels inside fabric_config.runtime
    assert calls == [(["nemo-fabric[deepagents,relay]==0.3.0", "mcp==1.29.0"], host_gym_constraints(), "3.13", None)]


def test_a_users_wheels_environment_is_extended_and_a_native_one_refused(tmp_path: Path) -> None:
    root = tmp_path / "environment"
    (root / "resources_servers" / "greet" / "configs").mkdir(parents=True)
    (root / "resources_servers" / "greet" / "configs" / "greet.yaml").write_text("greet: {}\n")
    (root / "wheels").mkdir()
    (root / "wheels" / "greet-1.0-py3-none-any.whl").write_bytes(b"")
    (root / "nemo-environment.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "wheels-v1",
                "config_paths": ["resources_servers/greet/configs/greet.yaml"],
                "metadata": {"name": "greet"},
            }
        )
    )

    config_path = write_registered_agent_package(root, _spec(), agent_files=None, download=_fake_download([]))

    manifest = parse_environment_manifest((root / "nemo-environment.yaml").read_text())
    assert list(manifest.config_paths) == ["resources_servers/greet/configs/greet.yaml", config_path]
    assert manifest.metadata.name == "greet"
    validate_environment_manifest_against_listing(manifest, _listing(root))
    instance = yaml.safe_load((root / config_path).read_text())["registered_calc_agent"]["responses_api_agents"][
        "nemo_registered_agent"
    ]
    assert instance["fabric_config_base_dir"] is None  # no Ethos files, nothing to resolve relative paths against

    native = tmp_path / "native"
    native.mkdir()
    (native / "nemo-environment.yaml").write_text(
        yaml.safe_dump(
            {"format": "native-v1", "config_paths": ["resources_servers/x/configs/x.yaml"], "metadata": {"name": "x"}}
        )
    )
    with pytest.raises(ValueError, match="wheels-v1"):
        write_registered_agent_package(native, _spec(), agent_files=None, download=_fake_download([]))


def test_the_wheelhouse_is_downloaded_for_the_hosts_interpreter_not_the_job_containers(tmp_path: Path) -> None:
    """pip matches platform tags exactly, so both manylinux tags a wheel may carry are requested."""
    command = wheel_download_command(["nemo-fabric[relay]==0.3.0"], tmp_path / "pins.txt", tmp_path, "3.13", "arm64")
    assert command[command.index("--constraint") + 1] == str(tmp_path / "pins.txt")
    assert command[command.index("--python-version") + 1] == "3.13"
    assert command[command.index("--abi") + 1] == "cp313"
    assert "--only-binary=:all:" in command
    assert [command[i + 1] for i, a in enumerate(command) if a == "--platform"] == [
        "manylinux2014_aarch64",
        "manylinux_2_28_aarch64",
    ]
    assert command[-1] == "nemo-fabric[relay]==0.3.0"
    unconstrained = wheel_download_command(["x"], None, tmp_path, "3.13", None)
    assert "--platform" in unconstrained  # the container's own arch, still Linux tags
    assert "--constraint" not in unconstrained


def test_download_wheels_hands_pip_a_constraints_file_and_cleans_it_up(tmp_path: Path, mocker) -> None:
    """The host's Gym pins travel to pip as a constraints file that lives only for the invocation."""
    seen: dict[str, object] = {}

    def run(command, **kwargs):
        idx = command.index("--constraint")
        seen["constraints"] = Path(command[idx + 1]).read_text()
        return mocker.Mock(returncode=0, stderr="")

    mocker.patch("nemo_evaluator.jobs.gym_registered_agent_package.subprocess.run", side_effect=run)
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()

    download_wheels(["nemo-fabric[relay]==0.3.0"], ["openai<=2.7.2", "httpx<1"], wheelhouse, "3.13", None)

    assert seen["constraints"] == "openai<=2.7.2\nhttpx<1\n"
    assert not (tmp_path / ".wheel-constraints.txt").exists()


def test_the_bundled_host_lock_is_the_gym_host_images_lock() -> None:
    """The plugin ships a copy of docker/locks/nhx-gym-host/uv.lock; a Gym bump in the image must copy it over again."""
    repo_lock = Path(__file__).resolve().parents[3] / "docker" / "locks" / "nhx-gym-host" / "uv.lock"
    bundled = (
        Path(__file__).resolve().parents[1] / "src" / "nemo_evaluator" / "gym_registered_agent" / "nhx-gym-host.uv.lock"
    )
    assert bundled.read_bytes() == repo_lock.read_bytes(), (
        "nemo_evaluator/gym_registered_agent/nhx-gym-host.uv.lock differs from docker/locks/nhx-gym-host/uv.lock; "
        "copy the lock over: cp docker/locks/nhx-gym-host/uv.lock "
        "plugins/nemo-evaluator/src/nemo_evaluator/gym_registered_agent/nhx-gym-host.uv.lock"
    )
    pins = host_gym_constraints()
    assert any(pin.startswith("nemo-gym==") for pin in pins)
    assert any(pin.startswith("openai==") for pin in pins)
    assert all("==" in pin for pin in pins) and pins == sorted(pins)
    assert not any(pin.startswith("nemo-sandboxed-gym==") or pin.startswith("nhx-gym-host==") for pin in pins)


def test_lock_entries_that_are_not_distributions_carry_no_pin() -> None:
    lock = """
version = 1
[[package]]
name = "nhx-gym-host"
version = "0.0.0"
source = { virtual = "." }
[[package]]
name = "nemo-sandboxed-gym"
version = "0.1.0"
source = { directory = "../../packages/sandboxed_gym" }
[[package]]
name = "openai"
version = "2.7.2"
source = { registry = "https://pypi.org/simple" }
"""
    assert host_gym_constraints(lock) == ["openai==2.7.2"]


def test_gym_takes_the_harness_extra_alone_and_lets_the_host_pin_its_companions() -> None:
    from nemo_evaluator.jobs.fabric_harness_packages import fabric_harness_requirements

    with_companions = fabric_harness_requirements("nvidia.fabric.langchain.deepagents")
    alone = fabric_harness_requirements("nvidia.fabric.langchain.deepagents", companions=False)
    assert alone == with_companions[:1] and alone[0].startswith("nemo-fabric[deepagents,relay]==")
    assert any(r.startswith("mcp==") for r in with_companions)


def test_a_user_package_that_already_ships_the_component_is_refused(tmp_path: Path) -> None:
    """The writer adds to a user's package; it must not silently replace a component or config the user put there."""
    root = tmp_path / "env"
    (root / "responses_api_agents" / "nemo_registered_agent").mkdir(parents=True)
    (root / "nemo-environment.yaml").write_text("format: wheels-v1\nconfig_paths: []\nmetadata: {name: g}\n")

    with pytest.raises(ValueError, match="already contains responses_api_agents/nemo_registered_agent"):
        write_registered_agent_package(root, _spec(), agent_files=None, download=_fake_download([]))


def test_a_stalled_wheelhouse_download_is_cut_off(tmp_path: Path, mocker: MockerFixture) -> None:
    import subprocess

    mocker.patch(
        "nemo_evaluator.jobs.gym_registered_agent_package.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="pip download", timeout=1),
    )
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()

    with pytest.raises(RuntimeError, match="did not finish within"):
        download_wheels(["nemo-fabric[relay]==0.3.0"], ["openai==2.7.2"], wheelhouse, "3.13", None)
    assert not (tmp_path / ".wheel-constraints.txt").exists()
