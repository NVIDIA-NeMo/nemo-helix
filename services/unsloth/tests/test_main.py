# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the training container entrypoint.

The container ``CMD`` is ``python -m nhx.unsloth.tasks.training``. It
expects the platform Jobs runner to mount a step-config JSON file via
``NEMO_JOB_STEP_CONFIG_FILE_PATH`` and then invokes
``train_sft`` against the paths the file_io step downloaded to.

These tests cover the failure path (no env var → exit 2 with a hint to
submit via the customization CLI). The happy path requires the
``[unsloth]`` extra in the test env, so we don't exercise it here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from nhx.unsloth.app.jobs.training.schemas import TrainingStepConfig
from nhx.unsloth.schemas import (
    DatasetSpec,
    LoRAParams,
    ModelLoadSpec,
    OutputResponse,
    TrainingSpec,
    UnslothJobOutput,
)
from nhx.unsloth.tasks.training.__main__ import main


def _step_config() -> TrainingStepConfig:
    spec = UnslothJobOutput(
        model=ModelLoadSpec(name="default/base"),
        dataset=DatasetSpec(path="default/training"),
        training=TrainingSpec(lora=LoRAParams()),
        schedule={"max_steps": 1},
        output=OutputResponse(name="r", type="adapter", save_method="lora", fileset="r"),
    )
    return TrainingStepConfig(
        spec=spec,
        model_path="/var/run/scratch/job/model",
        dataset_path="/var/run/scratch/job/dataset",
        output_path="/var/run/scratch/job/output_model",
    )


class TestEntrypointCachePaths:
    """The entrypoint must redirect unsloth's compile cache + HF cache off the
    root-owned WORKDIR onto the job's writable ephemeral storage before
    ``train_sft`` triggers ``import unsloth``."""

    def test_sets_writable_cache_env_under_storage(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from nhx.common.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
        from nhx.unsloth.tasks.training.backends import unsloth_sft

        config_file = tmp_path / "step.json"
        config_file.write_text(json.dumps(_step_config().model_dump(mode="json")))

        monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(config_file))
        monkeypatch.setenv(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, str(tmp_path / "job"))
        monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)

        captured: dict[str, str | None] = {}

        def _stub_train_sft(*_args: object, **_kwargs: object) -> dict[str, object]:
            captured["UNSLOTH_COMPILE_LOCATION"] = os.environ.get("UNSLOTH_COMPILE_LOCATION")
            captured["HF_HOME"] = os.environ.get("HF_HOME")
            return {}

        # main() does `from ...unsloth_sft import train_sft` at call time, so
        # patching the module attribute is enough.
        monkeypatch.setattr(unsloth_sft, "train_sft", _stub_train_sft)

        rc = main()

        assert rc == 0
        ephemeral = tmp_path / "job" / "ephemeral"
        assert captured["UNSLOTH_COMPILE_LOCATION"] == str(ephemeral / "unsloth_compiled_cache")
        assert captured["HF_HOME"] == str(ephemeral / "hf")
        # The dirs must actually exist (unsloth_zoo only makedirs lazily).
        assert (ephemeral / "unsloth_compiled_cache").is_dir()
        assert (ephemeral / "hf").is_dir()

    def test_remaps_legacy_storage_paths_to_runtime_mount(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from nhx.common.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
        from nhx.unsloth.tasks.training.backends import unsloth_sft

        config = _step_config().model_copy(
            update={
                "model_path": "/run/scratch/job/model",
                "dataset_path": "/run/scratch/job/dataset",
                "output_path": "/run/scratch/job/output_model",
            },
        )
        config_file = tmp_path / "step.json"
        config_file.write_text(json.dumps(config.model_dump(mode="json")))

        storage = tmp_path / "job"
        monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(config_file))
        monkeypatch.setenv(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, str(storage))
        monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)

        captured: dict[str, object] = {}

        def _stub_train_sft(*_args: object, **kwargs: object) -> dict[str, object]:
            captured.update(
                {
                    "model_path": kwargs["model_path"],
                    "dataset_path": kwargs["dataset_path"],
                    "output_path": kwargs["output_path"],
                },
            )
            return {}

        monkeypatch.setattr(unsloth_sft, "train_sft", _stub_train_sft)

        rc = main()

        assert rc == 0
        assert captured == {
            "model_path": str(storage / "model"),
            "dataset_path": str(storage / "dataset"),
            "output_path": str(storage / "output_model"),
        }

    def test_uses_legacy_storage_env_for_runtime_mount(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from nhx.common.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
        from nhx.customization_common.service.path_utils import (
            CURRENT_PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            LEGACY_PERSISTENT_JOB_STORAGE_PATH_ENVVARS,
        )
        from nhx.unsloth.tasks.training.backends import unsloth_sft

        config_file = tmp_path / "step.json"
        config_file.write_text(json.dumps(_step_config().model_dump(mode="json")))

        storage = tmp_path / "legacy-job"
        monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(config_file))
        monkeypatch.delenv(CURRENT_PERSISTENT_JOB_STORAGE_PATH_ENVVAR, raising=False)
        monkeypatch.delenv(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, raising=False)
        monkeypatch.setenv(LEGACY_PERSISTENT_JOB_STORAGE_PATH_ENVVARS[0], str(storage))
        monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)

        captured: dict[str, str | None] = {}

        def _stub_train_sft(*_args: object, **_kwargs: object) -> dict[str, object]:
            captured["UNSLOTH_COMPILE_LOCATION"] = os.environ.get("UNSLOTH_COMPILE_LOCATION")
            captured["HF_HOME"] = os.environ.get("HF_HOME")
            return {}

        monkeypatch.setattr(unsloth_sft, "train_sft", _stub_train_sft)

        rc = main()

        assert rc == 0
        ephemeral = storage / "ephemeral"
        assert captured["UNSLOTH_COMPILE_LOCATION"] == str(ephemeral / "unsloth_compiled_cache")
        assert captured["HF_HOME"] == str(ephemeral / "hf")


class TestEntrypointWithoutStepConfig:
    def test_returns_2_when_step_config_env_missing(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", raising=False)
        rc = main()
        assert rc == 2
        err = capsys.readouterr().err
        # Friendly hint redirects the user to the submit CLI path.
        assert "nemo customization unsloth --job-json" in err

    def test_module_invocation_exits_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Smoke: ``python -m nhx.unsloth.tasks.training`` exits 2 without a step config."""
        env = {k: v for k, v in __import__("os").environ.items()}
        env.pop("NEMO_JOB_STEP_CONFIG_FILE_PATH", None)
        result = subprocess.run(
            [sys.executable, "-m", "nhx.unsloth.tasks.training"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert result.returncode == 2
        assert "nemo customization unsloth --job-json" in result.stderr


def test_a_paused_run_exits_with_the_pause_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint path reaches training under the job's storage, and a pause exits 75."""
    from nhx.common.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
    from nhx.unsloth.app.constants import DEFAULT_CHECKPOINT_PATH
    from nhx.unsloth.tasks.training.backends import unsloth_sft

    config = _step_config().model_copy(update={"checkpoint_path": DEFAULT_CHECKPOINT_PATH})
    config_file = tmp_path / "step.json"
    config_file.write_text(json.dumps(config.model_dump(mode="json")))
    storage = tmp_path / "job"
    monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(config_file))
    monkeypatch.setenv(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, str(storage))
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    captured: dict[str, object] = {}

    def _stub_train_sft(*_args: object, **kwargs: object) -> dict[str, object]:
        captured["checkpoint_path"] = kwargs["checkpoint_path"]
        return {"paused": True, "global_step": 7}

    monkeypatch.setattr(unsloth_sft, "train_sft", _stub_train_sft)

    assert main() == 75
    assert captured["checkpoint_path"] == str(storage / "training" / "checkpoints")


class TestEntrypointFailureReport:
    """A failed step reports its classified cause, so a rerun can warn when it would fail the same way."""

    def _run_failing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, *, reporting_fails: bool = False
    ) -> tuple[int, list]:
        from nhx.common.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
        from nhx.unsloth.tasks.training import progress
        from nhx.unsloth.tasks.training.backends import unsloth_sft

        config_file = tmp_path / "step.json"
        config_file.write_text(json.dumps(_step_config().model_dump(mode="json")))
        monkeypatch.setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", str(config_file))
        monkeypatch.setenv(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, str(tmp_path / "job"))

        def _fail(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise error

        reports: list = []

        class _Reporter:
            def __init__(self, _job_ctx: object) -> None:
                if reporting_fails:
                    raise RuntimeError("Jobs service unreachable")

            def report_error(self, error_details: dict, status_details: dict) -> None:
                reports.append((error_details, status_details))

            def close(self) -> None: ...

        monkeypatch.setattr(unsloth_sft, "train_sft", _fail)
        monkeypatch.setattr(progress, "JobsServiceProgressReporter", _Reporter)
        return main(), reports

    def test_an_out_of_memory_failure_is_reported_as_not_rerunnable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rc, reports = self._run_failing(
            tmp_path, monkeypatch, RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        )

        assert rc == 1
        ((error_details, status_details),) = reports
        assert error_details["cause"] == "gpu_out_of_memory"
        assert status_details["resumable"] is False

    def test_a_failure_to_report_still_fails_the_step(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        rc, reports = self._run_failing(tmp_path, monkeypatch, RuntimeError("boom"), reporting_fails=True)

        assert rc == 1
        assert reports == []
