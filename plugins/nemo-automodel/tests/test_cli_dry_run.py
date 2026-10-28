# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo customization automodel --dry-run`` and ``preflight``."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from nemo_automodel_plugin.contributor import AutomodelContributor
from typer.testing import CliRunner

_CHAT = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}


@pytest.fixture(autouse=True)
def _platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://nhx.test")


def _job_json(tmp_path: Path, **spec: object) -> Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"training": {"training_type": "sft"}, **spec}))
    return path


def _report(passed: bool) -> dict:
    return {
        "spec": {},
        "steps": [{"name": "training", "provider": "gpu", "profile": "default", "image": "automodel:1"}],
        "checks": [
            {"name": "execution profile", "passed": True, "message": "Configured: gpu/default."},
            {"name": "training dataset", "passed": passed, "message": "ws/d#train.jsonl line 3 ..."},
        ],
        "passed": passed,
    }


@pytest.mark.parametrize(("passed", "exit_code"), [(True, 0), (False, 1)])
def test_dry_run_reports_the_platform_checks_and_submits_nothing(tmp_path: Path, passed: bool, exit_code: int) -> None:
    job = _job_json(tmp_path, model="default/m", dataset={"training": "default/d"})
    with (
        patch(
            "nhx.customization_common.cli.dry_run.NemoJobScheduler.dry_run_remote", return_value=_report(passed)
        ) as dry_run,
        patch("nemo_helix_plugin.scheduler.NemoJobScheduler.submit_remote") as submit,
    ):
        result = CliRunner().invoke(
            AutomodelContributor().get_cli(), ["--job-json", str(job), "--dry-run", "--profile", "a100"]
        )

    assert result.exit_code == exit_code, result.output
    assert json.loads(result.output) == _report(passed)
    assert dry_run.call_args.kwargs["profile"] == "a100"
    assert dry_run.call_args.args[1]["dataset"] == {"training": "default/d"}
    submit.assert_not_called()


def test_dry_run_with_upload_checks_local_rows_and_uploads_nothing(tmp_path: Path) -> None:
    job = _job_json(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(json.dumps(_CHAT) + "\n" + json.dumps({"instruction": "q"}) + "\n")
    with (
        patch("nhx.customization_common.cli.overrides.run_uploads") as uploads,
        patch("nhx.customization_common.cli.dry_run.NemoJobScheduler.dry_run_remote") as dry_run,
    ):
        result = CliRunner().invoke(
            AutomodelContributor().get_cli(),
            ["--job-json", str(job), "--upload-model", "Qwen/Qwen3-0.6B", "--upload-dataset", str(data), "--dry-run"],
        )

    assert result.exit_code == 1, result.output
    checks = json.loads(result.output)["checks"]
    assert checks[0] == {"name": "job JSON", "passed": True, "message": "Matches the schema."}
    assert checks[1]["passed"] is False
    assert checks[1]["message"].startswith(f"{data}/train.jsonl line 2 does not conform")
    uploads.assert_not_called()
    dry_run.assert_not_called()


def test_dry_run_cannot_wait(tmp_path: Path) -> None:
    job = _job_json(tmp_path, model="default/m", dataset={"training": "default/d"})
    result = CliRunner().invoke(AutomodelContributor().get_cli(), ["--job-json", str(job), "--dry-run", "--wait"])
    assert result.exit_code == 2
    assert "--dry-run creates no job to wait for" in result.output


def test_preflight_passes_with_a_gpu_profile() -> None:
    profiles = [SimpleNamespace(provider="gpu", profile="default", backend="docker")]
    jobs = MagicMock()
    jobs.__enter__.return_value.list_execution_profiles.return_value = profiles
    with patch("nhx.customization_common.cli.dry_run.JobsClient", return_value=jobs):
        result = CliRunner().invoke(AutomodelContributor().get_cli(), ["preflight"])

    assert result.exit_code == 0, result.output
    assert [check["message"] for check in json.loads(result.output)["checks"]] == [
        "Reachable at http://nhx.test.",
        "Usable: gpu/default (docker).",
    ]


def test_preflight_fails_without_the_platform() -> None:
    jobs = MagicMock()
    jobs.__enter__.return_value.list_execution_profiles.side_effect = ConnectionError("refused")
    with patch("nhx.customization_common.cli.dry_run.JobsClient", return_value=jobs):
        result = CliRunner().invoke(AutomodelContributor().get_cli(), ["preflight"])

    assert result.exit_code == 1
    [check] = json.loads(result.output)["checks"]
    assert check["message"].startswith("Cannot reach http://nhx.test: refused.")


def test_table_output_reads_as_one_line_per_check(tmp_path: Path) -> None:
    job = _job_json(tmp_path, model="default/m", dataset={"training": "default/d"})
    with (
        patch("nhx.customization_common.cli.dry_run.NemoJobScheduler.dry_run_remote", return_value=_report(False)),
        patch("nhx.customization_common.cli.dry_run.resolve_output_format", return_value="table"),
    ):
        result = CliRunner().invoke(AutomodelContributor().get_cli(), ["--job-json", str(job), "--dry-run"])

    assert result.output.splitlines() == [
        "OK   execution profile: Configured: gpu/default.",
        "FAIL training dataset: ws/d#train.jsonl line 3 ...",
        "     step training: gpu/default, image automodel:1",
        "Dry run failed. Nothing was created.",
    ]
