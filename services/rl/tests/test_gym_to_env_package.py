# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "grpo-examples" / "gym_to_env_package.py"
SPEC = importlib.util.spec_from_file_location("gym_to_env_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _touch_wheels(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).touch()


def test_referenced_servers_contribute_requirements_and_not_their_trees(tmp_path: Path) -> None:
    gym = tmp_path / "Gym"
    selected = gym / "resources_servers" / "math_with_judge"
    selected.mkdir(parents=True)
    (selected / "requirements.txt").write_text("math-verify==0.8.0\n", encoding="utf-8")
    agent = gym / "responses_api_agents" / "simple_agent"
    agent.mkdir(parents=True)
    (agent / "app.py").write_text("pass\n", encoding="utf-8")
    (agent / "requirements.txt").write_text("-e nemo-gym[dev] @ ../../\nhttpx==0.28.1\n", encoding="utf-8")
    model = gym / "responses_api_models" / "vllm_model"
    model.mkdir(parents=True)
    (model / "pyproject.toml").write_text(
        '[project]\nname = "vllm-model"\nversion = "0.0.1"\ndependencies = ["nemo-gym[dev]", "nixl==0.1.0"]\n',
        encoding="utf-8",
    )
    out = tmp_path / "pkg"
    out.mkdir()
    config = out / "math_with_judge.yaml"
    config.write_text(
        "math_with_judge:\n"
        "  resources_servers:\n"
        "    math_with_judge:\n"
        "      entrypoint: app.py\n"
        "math_with_judge_simple_agent:\n"
        "  responses_api_agents:\n"
        "    simple_agent:\n"
        "      entrypoint: app.py\n",
        encoding="utf-8",
    )
    policy = out / "policy_model.yaml"
    policy.write_text(
        "policy_model:\n  responses_api_models:\n    vllm_model:\n      entrypoint: app.py\n",
        encoding="utf-8",
    )

    lines = MODULE.referenced_server_requirements(
        gym,
        [config, policy],
        Path("resources_servers/math_with_judge"),
    )

    assert lines == ["httpx==0.28.1", "nixl==0.1.0"]
    assert not (out / "responses_api_agents").exists()
    assert not (out / "responses_api_models").exists()


def test_server_closure_uses_the_selected_server_only(tmp_path: Path) -> None:
    server = tmp_path / "resources_servers" / "math_with_judge"
    server.mkdir(parents=True)
    (server / "requirements.txt").write_text(
        "-e nemo-gym[dev] @ ../../\nmath-verify==0.8.0\n# comment\n\ndatasets\n"
        "nemo-gym==0.7.0rc0\nnemo-gym-extras==1.0\n",
        encoding="utf-8",
    )

    lines = MODULE.server_closure_requirements(server, "2.56.1", "2.44.0")

    assert lines[:2] == ["ray[default]==2.56.1", "openai==2.44.0"]
    assert "math-verify==0.8.0" in lines
    assert "datasets" in lines
    assert "nemo-gym-extras==1.0" in lines
    assert "nemo-gym==0.7.0rc0" not in lines


def _zip_wheel(path: Path, *names: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"")


def test_checkout_gym_wheel_is_dropped_and_the_library_wheel_stays(tmp_path: Path) -> None:
    _zip_wheel(tmp_path / "nemo_gym-0.7.0rc0-py3-none-any.whl", "nemo_gym/__init__.py")
    _zip_wheel(tmp_path / "nemo_gym-9.9.9-py3-none-any.whl", "resources_servers/math_with_judge/app.py")
    (tmp_path / "verifiers-0.3.1-py3-none-any.whl").touch()

    MODULE.drop_image_gym_wheel(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "nemo_gym-0.7.0rc0-py3-none-any.whl",
        "verifiers-0.3.1-py3-none-any.whl",
    ]


def test_library_gym_project_copies_the_library_only(tmp_path: Path) -> None:
    gym = tmp_path / "Gym"
    (gym / "nemo_gym").mkdir(parents=True)
    (gym / "nemo_gym" / "package_info.py").write_text("MAJOR = 0\n", encoding="utf-8")
    (gym / "resources_servers" / "math_with_judge").mkdir(parents=True)
    (gym / "resources_servers" / "math_with_judge" / "app.py").write_text("pass\n", encoding="utf-8")
    (gym / "pyproject.toml").write_text(
        '[project]\nname = "nemo-gym"\ndependencies = ["anthropic<=0.109.2"]\n'
        '[project.optional-dependencies]\ndev = ["mypy>=1.8.0"]\n',
        encoding="utf-8",
    )

    dest = tmp_path / "stage"
    MODULE.write_library_gym_project(gym, dest, "0.7.0rc0")

    assert (dest / "nemo_gym" / "package_info.py").is_file()
    assert not (dest / "resources_servers").exists()
    text = (dest / "pyproject.toml").read_text(encoding="utf-8")
    assert "anthropic<=0.109.2" in text
    assert "mypy>=1.8.0" in text
    assert 'include = ["nemo_gym", "nemo_gym.*"]' in text


def test_required_wheel_versions_accept_supported_hydra_stack(tmp_path: Path) -> None:
    _touch_wheels(
        tmp_path,
        "hydra_core-1.3.2-py3-none-any.whl",
        "omegaconf-2.3.0-py3-none-any.whl",
    )

    MODULE.validate_required_wheel_versions(tmp_path)


def test_required_wheel_versions_reject_backtracked_hydra_stack(tmp_path: Path) -> None:
    _touch_wheels(
        tmp_path,
        "hydra_core-0.11.3-py3-none-any.whl",
        "omegaconf-1.4.1-py3-none-any.whl",
    )

    with pytest.raises(SystemExit, match="hydra-core.*0.11.3.*omegaconf.*1.4.1"):
        MODULE.validate_required_wheel_versions(tmp_path)


def test_rl_dependency_policy_reads_uv_tables(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv]
override-dependencies = ["fastapi[standard]>=0.133.0,<0.137.0"]
constraint-dependencies = ["urllib3>=2.7.0"]
""",
        encoding="utf-8",
    )

    overrides, constraints = MODULE.rl_dependency_policy(tmp_path)

    assert overrides == ["fastapi[standard]>=0.133.0,<0.137.0"]
    assert constraints == ["urllib3>=2.7.0"]


def test_setuptools_override_keeps_the_pkg_resources_ceiling(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.uv]
override-dependencies = ["setuptools>=80.10.2"]
constraint-dependencies = ["urllib3>=2.7.0"]
""",
        encoding="utf-8",
    )

    overrides, _constraints = MODULE.rl_dependency_policy(tmp_path)

    assert overrides == ["setuptools<81,>=80.10.2"]


def test_rl_dependency_policy_rejects_a_checkout_without_uv_limits(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.uv]\nmanaged = true\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="override-dependencies"):
        MODULE.rl_dependency_policy(tmp_path)


def test_package_info_version_keeps_the_prerelease_suffix(tmp_path: Path) -> None:
    (tmp_path / "nemo_gym").mkdir()
    (tmp_path / "nemo_gym" / "package_info.py").write_text(
        'MAJOR = 0\nMINOR = 7\nPATCH = 0\nPRE_RELEASE = "rc0"\n',
        encoding="utf-8",
    )

    assert MODULE.nemo_gym_version(tmp_path) == "0.7.0rc0"


def test_locked_versions_pin_the_lock_and_the_gym_edge(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text(
        """
[[package]]
name = "nemo-gym"
dependencies = [
    { name = "openai", version = "2.44.0" },
    { name = "ray", extra = ["default"] },
]

[[package]]
name = "openai"
version = "2.6.1"

[[package]]
name = "openai"
version = "2.44.0"

[[package]]
name = "ray"
version = "2.56.1"

[[package]]
name = "aiohttp"
version = "3.14.3"

[[package]]
name = "torch"
version = "2.11.0"

[[package]]
name = "torch"
version = "2.13.0"
""",
        encoding="utf-8",
    )

    pins = MODULE.locked_package_versions(tmp_path)

    assert pins == {"openai": "2.44.0", "ray": "2.56.1", "aiohttp": "3.14.3"}
    assert MODULE.lock_requirement_lines(pins, ["ray[default]>=2.56.1"]) == [
        "aiohttp==3.14.3",
        "openai==2.44.0",
        "ray==2.56.1",
    ]
    assert MODULE.lock_requirement_lines({"setuptools": "84.0.0", "aiohttp": "3.14.3"}, ["setuptools<81"]) == [
        "aiohttp==3.14.3",
    ]
    assert MODULE.align_requirement_to_lock("aiohttp>=3.14.1", {"aiohttp": "3.14.3"}) == "aiohttp>=3.14.1"
    assert MODULE.lock_requirement_lines(
        {"math-verify": "0.9.0", "latex2sympy2-extended": "1.11.0", "aiohttp": "3.14.3"},
        ["math-verify==0.8.0"],
        {"math-verify": {"latex2sympy2-extended"}},
        expand_dependencies_of=["math-verify==0.8.0"],
    ) == ["aiohttp==3.14.3"]
