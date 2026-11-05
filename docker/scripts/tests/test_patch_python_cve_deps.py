# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location("patch_python_cve_deps", ROOT / "docker/scripts/patch-python-cve-deps.py")
assert SPEC is not None and SPEC.loader is not None
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)
OVERRIDES = ROOT / "docker/python-cve-overrides.txt"


@pytest.mark.parametrize("installed", [[], ["urllib3"], ["fsspec", "datasets"], ["urllib3", "fsspec", "datasets"]])
def test_only_installed_packages_are_requested(monkeypatch, installed: list[str]) -> None:
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"]
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(installed))

    monkeypatch.setattr(patch.subprocess, "run", run)
    python = Path("/venv/bin/python")
    extra = Path("/preserve-torch.txt")
    patch.patch_environment(python, [OVERRIDES, extra])

    assert calls[0][0] == str(python)
    if not installed:
        assert len(calls) == 1
        return
    command = calls[1]
    assert command[:7] == ["uv", "--no-config", "pip", "install", "--python", str(python), "--no-cache"]
    assert command[-len(installed) :] == [
        spec
        for name, spec in (
            ("urllib3", "urllib3>=2.8.0,<3"),
            ("fsspec", "fsspec>=2026.6.0,<=2026.7.0"),
            ("datasets", "datasets>=5.1.0,<6"),
        )
        if name in installed
    ]
    assert command.count("--overrides") == 2
    assert ("--reinstall-package" in command) == ("fsspec" in installed)
    assert str(extra) in command


@pytest.mark.parametrize("fail_at", [1, 2])
def test_probe_and_install_failures_abort_the_build(monkeypatch, fail_at: int) -> None:
    calls = 0

    def run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout='["urllib3"]')

    monkeypatch.setattr(patch.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        patch.patch_environment(Path("/venv/bin/python"), [OVERRIDES])


def test_nested_venvs_keep_their_interpreter_paths(tmp_path: Path, monkeypatch) -> None:
    interpreters = []
    for name in ("host", "components/agents/simple/.venv", "components/models/inference/.venv"):
        python = tmp_path / name / "bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        interpreters.append(python)
    monkeypatch.setattr(sys, "argv", ["patch", "--root", str(tmp_path), "--overrides", str(OVERRIDES)])
    patched = []
    monkeypatch.setattr(patch, "patch_environment", lambda python, overrides: patched.append(python))

    patch.main()

    assert patched == sorted(interpreters)
    assert Path(sys.executable) not in patched


def test_empty_environment_root_fails_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["patch", "--root", str(tmp_path), "--overrides", str(OVERRIDES)])
    with pytest.raises(SystemExit, match="2"):
        patch.main()


@pytest.mark.parametrize(
    ("dockerfile", "environment"),
    [
        ("docker/Dockerfile.nhx-tasks", "--python /app/.venv/bin/python"),
        ("docker/Dockerfile.nhx-agent-hardener-tasks", "--root /opt/agent-hardener/garak-venv"),
        ("docker/gym-host/Dockerfile", '--root "${NEMO_GYM_VENV_DIR}"'),
        ("docker/Dockerfile.nhx-unsloth-training", "--python ${VIRTUAL_ENV}/bin/python"),
        ("docker/automodel/Dockerfile.nhx-automodel-training", "--python ${VIRTUAL_ENV}/bin/python"),
    ],
)
def test_requested_images_patch_their_runtime_environment(dockerfile: str, environment: str) -> None:
    text = (ROOT / dockerfile).read_text()
    assert "source=docker/python-cve-overrides.txt" in text
    assert environment in text
    assert "python /tmp/patch-python-cve-deps.py" in text
    if "/usr/local/bin/collect-python-sdists.py \\" in text:
        assert text.index("python /tmp/patch-python-cve-deps.py") < text.index(
            "/usr/local/bin/collect-python-sdists.py \\"
        )


@pytest.mark.parametrize("package,floor", [("urllib3", (2, 8, 0)), ("fsspec", (2026, 6, 0))])
def test_gym_host_lock_retains_fixed_versions(package: str, floor: tuple[int, ...]) -> None:
    lock = tomllib.loads((ROOT / "docker/locks/nhx-gym-host/uv.lock").read_text())
    version = next(entry["version"] for entry in lock["package"] if entry["name"] == package)
    assert tuple(map(int, version.split("."))) >= floor


def test_gym_drops_vulnerable_archives_before_copying_cache() -> None:
    text = (ROOT / "docker/gym-host/Dockerfile").read_text()
    assert text.index("python /tmp/patch-python-cve-deps.py") < text.index("uv cache clean urllib3 fsspec datasets")
    assert text.index("uv cache clean urllib3 fsspec datasets") < text.index("# Copy the matching uv cache")


@pytest.mark.parametrize(
    "dockerfile", ["docker/Dockerfile.nhx-unsloth-training", "docker/automodel/Dockerfile.nhx-automodel-training"]
)
def test_shadowed_ngc_fsspec_removed_only_after_installing_fixed_copy(dockerfile: str) -> None:
    text = (ROOT / dockerfile).read_text()
    assert text.index("python /tmp/patch-python-cve-deps.py") < text.index(
        "source=docker/removals/files/final/training-fsspec.txt"
    )
    patterns = (ROOT / "docker/removals/files/final/training-fsspec.txt").read_text()
    assert "/usr/local/lib/python3.12/dist-packages/fsspec\n" in patterns
    assert "/usr/local/lib/python3.12/dist-packages/fsspec-*.dist-info" in patterns
