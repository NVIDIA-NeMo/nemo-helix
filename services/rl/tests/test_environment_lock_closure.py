# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep unrelated training-image extras out of isolated Gym environment resolves."""

import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest
from nhx.rl.tasks.environment import convert


def _rl_checkout(root: Path) -> Path:
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[tool.uv]\noverride-dependencies = ["setuptools<81"]\nconstraint-dependencies = ["urllib3>=2"]\n'
    )
    (root / "uv.lock").write_text(
        """[[package]]
name = "nemo-gym"
version = "0.5.0"
dependencies = [{ name = "openai" }, { name = "ray", extra = ["default"] }]
[package.optional-dependencies]
dev = [{ name = "build-tool" }]
sandbox = [{ name = "e2b" }]

[[package]]
name = "openai"
version = "2.6.1"

[[package]]
name = "ray"
version = "2.56.1"
[package.optional-dependencies]
default = [{ name = "grpcio" }]

[[package]]
name = "grpcio"
version = "1.70.0"

[[package]]
name = "build-tool"
version = "1.0"

[[package]]
name = "e2b"
version = "2.61.0"
dependencies = [{ name = "connectrpc" }]

[[package]]
name = "connectrpc"
version = "0.12.1"
"""
    )
    return root


def test_environment_lock_keeps_required_transitives_and_requested_extras(tmp_path: Path) -> None:
    root = _rl_checkout(tmp_path / "RL")

    pins = convert.environment_lock_versions(root, ["nemo-gym[dev]"])

    assert pins == {
        "nemo-gym": "0.5.0",
        "openai": "2.6.1",
        "ray": "2.56.1",
        "grpcio": "1.70.0",
        "build-tool": "1.0",
    }


def test_environment_lock_retains_sandbox_pins_when_requested(tmp_path: Path) -> None:
    root = _rl_checkout(tmp_path / "RL")

    pins = convert.environment_lock_versions(root, ["nemo-gym[sandbox]"])

    assert pins["e2b"] == "2.61.0"
    assert pins["connectrpc"] == "0.12.1"
    assert "build-tool" not in pins


def _wheel(directory: Path, name: str, version: str, requirements: tuple[str, ...] = ()) -> None:
    normalized = name.replace("-", "_")
    info = f"{normalized}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\nProvides-Extra: dev\n"
    metadata += "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements)
    with ZipFile(directory / f"{normalized}-{version}-py3-none-any.whl", "w") as wheel:
        wheel.writestr(f"{info}/METADATA", metadata)
        wheel.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        wheel.writestr(f"{info}/RECORD", "")


def test_adapter_closure_resolves_connectrpc_without_unrelated_sandbox_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the real uv solver, offline, with the same conflicting dependency bounds."""
    root = _rl_checkout(tmp_path / "RL")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "nemo-gym", "0.5.0", ("openai", "ray", 'build-tool; extra == "dev"'))
    _wheel(wheels, "openai", "2.6.1")
    _wheel(wheels, "ray", "2.56.1")
    _wheel(wheels, "build-tool", "1.0")
    _wheel(wheels, "prime-sandboxes", "0.2.39", ("connectrpc>=0.11.1,<0.12",))
    _wheel(wheels, "connectrpc", "0.11.1")
    _wheel(wheels, "connectrpc", "0.12.1")

    def compile_offline(command: list[str]) -> None:
        assert command[:3] == ["uv", "pip", "compile"]
        subprocess.run([*command, "--no-index", "--find-links", str(wheels)], check=True)

    monkeypatch.setattr(convert, "_run_build_step", compile_offline)

    resolved = convert._compile_pinned_requirements(
        tmp_path / "work", ["nemo-gym[dev]==0.5.0", "prime-sandboxes==0.2.39"], nemo_rl_root=root
    ).read_text()

    assert "connectrpc==0.11.1" in resolved
    assert "connectrpc==0.12.1" not in resolved
    assert "openai==2.6.1" in resolved
    assert "ray==2.56.1" in resolved
