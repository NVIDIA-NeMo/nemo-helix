# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=9.0.3,<10", "pyyaml>=6,<7"]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for consolidated CI coverage and isolated wheel installs."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github/workflows/ci.yaml").read_text())["jobs"]
HELPER = ROOT / ".github/scripts/test-wheel-install.sh"


def test_packaging_keeps_all_interpreters_and_install_methods() -> None:
    job = JOBS["wheel-build"]
    assert job["strategy"] == {"fail-fast": False, "matrix": {"python-version": ["3.12", "3.13", "3.14"]}}
    assert "if" not in job  # Packaging remains unconditional.
    builds = [step for step in job["steps"] if step.get("uses") == "./.github/actions/build-nemo-helix-wheel"]
    assert [step["with"]["package"] for step in builds] == ["nemo-helix", "nemo-helix-plugin"]
    assert len({step["with"]["out-dir"] for step in builds}) == 2
    installs = [step for step in job["steps"] if "test-wheel-install.sh" in step.get("run", "")]
    assert [step["run"].split()[-2:] for step in installs] == [
        ["nemo-helix", "venv"],
        ["nemo-helix", "tool"],
        ["nemo-helix-plugin", "venv"],
    ]
    for step in installs:
        assert "!cancelled()" in step["if"]
        assert ".outcome == 'success'" in step["if"]
    uploads = [step for step in job["steps"] if step.get("uses", "").startswith("actions/upload-artifact@")]
    assert {step["with"]["name"] for step in uploads} == {
        "nemo-helix-wheel-py${{ matrix.python-version }}",
        "nemo-helix-plugin-wheel-py${{ matrix.python-version }}",
    }


def test_aggregate_has_no_dangling_dependencies_and_includes_groups() -> None:
    for job in JOBS.values():
        needs = job.get("needs", [])
        assert set([needs] if isinstance(needs, str) else needs) <= JOBS.keys()
    assert {
        "wheel-build",
        "web-static",
        "web-typecheck",
        "uv-lock",
        "policy-wasm",
        "openshell-tests",
        "kind-cpu-e2e",
    } <= set(JOBS["ci-status"]["needs"])
    assert JOBS["python-unit-test-status"]["name"] == "Python unit tests"
    assert JOBS["lint"]["name"] == "Lint all"


def test_web_checks_keep_triggers_order_and_failure_independence() -> None:
    for job_id in ("web-static", "web-typecheck"):
        condition = JOBS[job_id]["if"]
        assert "!cancelled()" in condition
        assert "github.event_name == 'workflow_dispatch'" in condition
        assert "needs.changes.outputs.web-studio == 'true'" in condition
        assert ("needs.changes.outputs.openapi == 'true'" in condition) == (job_id == "web-typecheck")
    steps = JOBS["web-typecheck"]["steps"]
    names = [step.get("name") for step in steps]
    assert names.index("Typecheck web packages") < names.index("Generate SDK from OpenAPI specs")
    for step in steps:
        if step.get("id") == "sdk" or "regenerated SDK" in step.get("name", ""):
            assert "!cancelled()" in step["if"]
    steps = JOBS["web-static"]["steps"]
    assert [step["run"] for step in steps if step.get("name") in ("Check formatting", "Lint", "Check studio deps")] == [
        "pnpm format",
        "pnpm lint",
        "pnpm deps:studio",
    ]
    for step in steps:
        if step.get("name") in ("Lint", "Check studio deps", "Regenerate and check plugin surface types"):
            assert "!cancelled()" in step["if"]
            assert "steps.deps.outcome == 'success'" in step["if"]


def test_openshell_preserves_individual_suite_filters() -> None:
    job = JOBS["openshell-tests"]
    for suite in ("deployments", "jobs"):
        condition = f"needs.changes.outputs.{suite}-openshell == 'true'"
        assert condition in job["if"]
        step = next(step for step in job["steps"] if step.get("run") == f"make test-{suite}-openshell")
        assert condition in step["if"]
        assert (
            f"needs.changes.outputs.{'jobs' if suite == 'deployments' else 'deployments'}-openshell" not in step["if"]
        )
        assert "needs.changes.outputs.deps == 'true'" in step["if"]
        assert "github.event_name == 'workflow_dispatch'" in step["if"]
        assert "!cancelled()" in step["if"]


@pytest.fixture
def install_env(tmp_path: Path) -> dict[str, str]:
    """Fake only external commands: run the real helper, without network installs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        '#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$*" >> "$CALLS"\n'
        'if [[ "$1" == venv ]]; then\n'
        '  mkdir -p "$2/bin"\n'
        '  printf \'#!/usr/bin/env bash\\nprintf "%%s\\n" "$*" >> "$CALLS"\\n\' > "$2/bin/python"\n'
        '  chmod +x "$2/bin/python"\nfi\n'
        'exit "${UV_EXIT_CODE:-0}"\n'
    )
    uv.chmod(0o755)
    workspace = tmp_path / "checkout"
    (workspace / "script").mkdir(parents=True)
    (workspace / "script/test-nemo-cli.sh").write_text(
        'set -eu\n[[ -z "${PYTHONPATH+x}" && -z "${VIRTUAL_ENV+x}" ]]\n'
        '[[ "$NHX_AUTH_ENABLED" == false ]]\n'
        '[[ "$NHX_DATA_DIR" == "$PWD/nemo-data" ]]\n'
        '[[ "$UV_TOOL_DIR" == "$PWD/tools" ]]\n'
        '[[ "$UV_TOOL_BIN_DIR" == "$PWD/tool-bin" ]]\n'
        '[[ "$RUNNER_TEMP" == "$PWD" ]]\n'
        'printf "cli:%s:%s\\n" "$PWD" "$PATH" >> "$CALLS"\n'
    )
    wheel = tmp_path / "test.whl"
    wheel.touch()
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GITHUB_WORKSPACE": str(workspace),
        "RUNNER_TEMP": str(tmp_path),
        "PYTHON_VERSION": "3.14",
        "WHEEL": str(wheel),
        "CALLS": str(tmp_path / "calls"),
        "PYTHONPATH": "must-not-leak",
        "VIRTUAL_ENV": "must-not-leak",
    }


@pytest.mark.parametrize(
    "package,method", [("nemo-helix", "venv"), ("nemo-helix", "tool"), ("nemo-helix-plugin", "venv")]
)
def test_wheel_installs_are_constrained_and_isolated(install_env: dict[str, str], package: str, method: str) -> None:
    for _ in range(2):
        subprocess.run(
            ["bash", str(HELPER), package, method], env=install_env, check=True, capture_output=True, text=True
        )
    calls = Path(install_env["CALLS"]).read_text()
    assert calls.count("install --no-config") == 2
    assert f"{install_env['WHEEL']}[services]" in calls if package == "nemo-helix" else "[services]" not in calls
    constraint = "nemo-helix-services" if package == "nemo-helix" else "nemo-helix-plugin"
    assert f"tools/wheel-constraints/{constraint}.txt" in calls
    assert "--python 3.14" in calls
    directories = list(Path(install_env["RUNNER_TEMP"]).glob(f"wheelcheck-{package}-{method}.*"))
    assert len(directories) == 2
    if package == "nemo-helix":
        for directory in directories:
            expected_bin = directory / ("tool-bin" if method == "tool" else ".venv/bin")
            assert f"cli:{directory}:{expected_bin}:" in calls
    else:
        for module in ("nemo_helix_plugin", "nemo_helix_plugin.cli", "nemo_helix_plugin.commands"):
            assert f"import {module}" in calls


def test_wheel_install_failure_is_not_swallowed(install_env: dict[str, str]) -> None:
    result = subprocess.run(
        ["bash", str(HELPER), "nemo-helix", "tool"],
        env={**install_env, "UV_EXIT_CODE": "17"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 17
    assert "cli:" not in Path(install_env["CALLS"]).read_text()


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [__file__, "-q", "-c", os.devnull, "-p", "no:cacheprovider", "--confcutdir", str(Path(__file__).parent)]
        )
    )
