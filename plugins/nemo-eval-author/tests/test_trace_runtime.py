# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability probes must judge provider behavior, not version ordering."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from harbor.models.task.config import TaskConfig
from harbor.models.task.task import Task
from harbor.models.trial.result import TrialResult
from pydantic import BaseModel, ConfigDict, Field

_SCRIPT = Path(__file__).resolve().parents[1] / "skills/eval-author-trace-environment/scripts/trace_environment.py"


class ModeResult(BaseModel):
    verifier_environment_mode: Literal["shared", "separate"] | None = None


class MissingModeResult(BaseModel):
    model_config = ConfigDict(extra="allow")


class ExcludedModeResult(BaseModel):
    verifier_environment_mode: Literal["shared", "separate"] | None = Field(default=None, exclude=True)


class SharedOnlyResult(BaseModel):
    verifier_environment_mode: Literal["shared"] | None = None


@pytest.fixture
def helper() -> Any:
    spec = importlib.util.spec_from_file_location("trace_runtime_probe_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def provider(monkeypatch: pytest.MonkeyPatch, helper: Any, result_model: type[BaseModel], label: str) -> None:
    original_import = helper.importlib.import_module
    original_version = helper.importlib.metadata.version
    modules = {
        "harbor.models.task.config": SimpleNamespace(TaskConfig=TaskConfig),
        "harbor.models.trial.result": SimpleNamespace(TrialResult=result_model),
        "harbor.models.task.task": SimpleNamespace(Task=Task),
    }

    def load(name: str, package: str | None = None) -> Any:
        return modules[name] if name in modules else original_import(name, package)

    monkeypatch.setattr(helper.importlib, "import_module", load)
    monkeypatch.setattr(
        helper.importlib.metadata, "version", lambda name: label if name == "harbor" else original_version(name)
    )


@pytest.mark.parametrize("label", ["0.1.0", "0.22.0", "99.0.0", "development-build"])
def test_runtime_accepts_capabilities_independent_of_version(
    helper: Any, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    provider(monkeypatch, helper, ModeResult, label)
    result = helper._check_runtime(argparse.Namespace(task_dir=None))
    assert result["valid"] is True
    assert result["harbor_version"] == label
    assert result["execution_verified"] is False
    assert result["scope"] == "single_step_proof"


@pytest.mark.parametrize("result_model", [MissingModeResult, ExcludedModeResult, SharedOnlyResult])
def test_runtime_rejects_missing_dropped_or_narrowed_mode(
    helper: Any, monkeypatch: pytest.MonkeyPatch, result_model: type[BaseModel]
) -> None:
    provider(monkeypatch, helper, result_model, "99.0.0")
    result = helper._check_runtime(argparse.Namespace(task_dir=None))
    assert result["valid"] is False
    assert any(check["name"] == "trial_verifier_mode" and not check["passed"] for check in result["checks"])


def test_runtime_does_not_treat_unknown_config_extras_as_support(helper: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    provider(monkeypatch, helper, ModeResult, "99.0.0")

    class ExtraConfig(BaseModel):
        model_config = ConfigDict(extra="allow")

    class PermissiveTaskConfig(BaseModel):
        environment: ExtraConfig
        verifier: ExtraConfig

    # An old provider can preserve arbitrary fields while not interpreting them.
    original_import = helper.importlib.import_module
    monkeypatch.setattr(
        helper.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(TaskConfig=PermissiveTaskConfig)
            if name == "harbor.models.task.config"
            else original_import(name)
        ),
    )
    result = helper._check_runtime(argparse.Namespace(task_dir=None))
    assert result["valid"] is False
    assert any(not check["passed"] for check in result["checks"] if "config" in check["name"])


def test_runtime_missing_provider_is_a_structured_finding(helper: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(helper.importlib, "import_module", missing)
    result = helper._check_runtime(argparse.Namespace(task_dir=None))
    assert result["valid"] is False
    assert result["checks"][0]["name"] == "provider_imports"


def test_runtime_metadata_absence_does_not_override_capabilities(helper: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    provider(monkeypatch, helper, ModeResult, "development")

    def missing(name: str) -> str:
        raise helper.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(helper.importlib.metadata, "version", missing)
    result = helper._check_runtime(argparse.Namespace(task_dir=None))
    assert result["harbor_version"] is None
    assert result["valid"] is True


def test_runtime_cli_reports_actual_installed_provider_without_writes(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "check-runtime"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    report = json.loads(result.stdout)
    assert result.returncode == (0 if report["valid"] else 1)
    mode = next(check for check in report["checks"] if check["name"] == "trial_verifier_mode")
    if "verifier_environment_mode" not in TrialResult.model_fields:
        assert mode["passed"] is False
    assert report["execution_verified"] is False
    assert list(tmp_path.iterdir()) == []


def test_task_preflight_rejects_multistep_shape(helper: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider(monkeypatch, helper, ModeResult, "development")
    original_import = helper.importlib.import_module

    class MultiStepTask:
        checksum = "probe-checksum"
        config = SimpleNamespace(steps=[SimpleNamespace(name="step-one")])

        def __init__(self, task_dir: Path):
            pass

    monkeypatch.setattr(
        helper.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(Task=MultiStepTask) if name == "harbor.models.task.task" else original_import(name)
        ),
    )
    workspace = tmp_path / "runtime-probe"
    workspace.mkdir()
    result = helper._check_runtime(argparse.Namespace(task_dir=workspace))
    assert result["valid"] is False
    assert any(check["name"] == "task_proof_shape" and not check["passed"] for check in result["checks"])


def test_runtime_provider_exception_is_not_an_unhandled_traceback(helper: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def incompatible(name: str) -> Any:
        raise RuntimeError("provider initialization changed")

    monkeypatch.setattr(helper.importlib, "import_module", incompatible)
    result = helper._check_runtime(argparse.Namespace(task_dir=None))
    assert result["valid"] is False
    assert "RuntimeError" in result["checks"][0]["message"]
