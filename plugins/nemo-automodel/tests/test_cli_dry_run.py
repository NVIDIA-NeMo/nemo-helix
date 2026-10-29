# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo customization automodel --dry-run`` and ``preflight``."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import respx
from nemo_automodel_plugin.contributor import AutomodelContributor
from nemo_helix_plugin.jobs.types import ExecutionProfileGPU, ExecutionProfileGPUs, ExecutionProfileGPUStatus
from typer.testing import CliRunner, Result

_CHAT = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
_CREATE = "http://nhx.test/apis/customization/v2/workspaces/default/automodel/jobs"


@pytest.fixture(autouse=True)
def _platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://nhx.test")


def _job_json(tmp_path: Path, **spec: object) -> Path:
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"training": {"training_type": "sft"}, **spec}))
    return path


def _checks(passed: bool) -> list[dict]:
    return [
        {"name": "execution profile", "passed": True, "message": "Configured: gpu/default."},
        {"name": "training dataset", "passed": passed, "message": "ws/d#train.jsonl line 3 ..."},
    ]


def _mock_create(passed: bool) -> respx.Route:
    """The create route with ?dry_run=true: 200 with the unsaved job, or 422 listing every check."""
    route = respx.post(_CREATE, params={"dry_run": "true"})
    if passed:
        return route.respond(200, json={"name": "j", "spec": {"m": 1}, "status_details": {"checks": _checks(True)}})
    detail = {"message": "Dry run failed: training dataset. Nothing was created.", "checks": _checks(False)}
    return route.respond(422, json={"detail": detail})


@pytest.mark.parametrize(("passed", "exit_code"), [(True, 0), (False, 1)])
@respx.mock
def test_dry_run_submits_with_the_flag_and_reports_every_check(tmp_path: Path, passed: bool, exit_code: int) -> None:
    job = _job_json(tmp_path, model="default/m", dataset={"training": "default/d"})
    create = _mock_create(passed)

    result = CliRunner().invoke(
        AutomodelContributor().get_cli(), ["--job-json", str(job), "--dry-run", "--profile", "a100"]
    )

    assert result.exit_code == exit_code, result.output
    report = json.loads(result.output)
    assert (report["checks"], report["passed"]) == (_checks(passed), passed)
    body = json.loads(create.calls.last.request.content)
    assert (body["profile"], body["spec"]["dataset"]) == ("a100", {"training": "default/d"})


@respx.mock
def test_a_dry_run_error_that_is_not_a_check_prints_as_submit_would(tmp_path: Path) -> None:
    job = _job_json(tmp_path, model="default/m", dataset={"training": "default/d"})
    respx.post(_CREATE, params={"dry_run": "true"}).respond(404, json={"detail": "Model default/m not found."})

    result = CliRunner().invoke(AutomodelContributor().get_cli(), ["--job-json", str(job), "--dry-run"])

    assert result.exit_code == 2
    assert "Model default/m not found." in result.output


def test_dry_run_with_upload_checks_local_rows_and_uploads_nothing(tmp_path: Path) -> None:
    job = _job_json(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(json.dumps(_CHAT) + "\n" + json.dumps({"instruction": "q"}) + "\n")
    with (
        patch("nhx.customization_common.cli.overrides.run_uploads") as uploads,
        patch("nhx.customization_common.cli.dry_run.NemoJobScheduler.submit_remote") as submit,
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
    submit.assert_not_called()


def test_dry_run_cannot_wait(tmp_path: Path) -> None:
    job = _job_json(tmp_path, model="default/m", dataset={"training": "default/d"})
    result = CliRunner().invoke(AutomodelContributor().get_cli(), ["--job-json", str(job), "--dry-run", "--wait"])
    assert result.exit_code == 2
    assert "--dry-run creates no job to wait for" in result.output


def _preflight_with_gpus(status: ExecutionProfileGPUStatus, message: str | None = None) -> Result:
    profiles = [SimpleNamespace(provider="gpu", profile="default", backend="docker")]
    found = [ExecutionProfileGPU(name="NVIDIA H100 80GB HBM3", compute_capability="9.0", count=8)]
    gpus = ExecutionProfileGPUs(
        provider="gpu",
        profile="default",
        backend="docker",
        status=status,
        gpus=found if status == "available" else [],
        message=message,
    )
    jobs = MagicMock()
    jobs.__enter__.return_value.list_execution_profiles.return_value = profiles
    jobs.__enter__.return_value.get_execution_profile_gpus.return_value.data.return_value = [gpus]
    with patch("nhx.customization_common.cli.dry_run.JobsClient", return_value=jobs):
        return CliRunner().invoke(AutomodelContributor().get_cli(), ["preflight"])


def test_preflight_passes_with_a_gpu_profile_that_has_gpus() -> None:
    result = _preflight_with_gpus("available")

    assert result.exit_code == 0, result.output
    assert [check["message"] for check in json.loads(result.output)["checks"]] == [
        "Reachable at http://nhx.test.",
        "Usable: gpu/default (docker).",
        "gpu/default: 8x NVIDIA H100 80GB HBM3 (compute 9.0).",
    ]


def test_preflight_fails_when_the_gpu_profile_has_no_gpus() -> None:
    result = _preflight_with_gpus(
        "none", "Docker cannot give containers a GPU: the NVIDIA Container Toolkit is not set up."
    )

    assert result.exit_code == 1
    gpus = json.loads(result.output)["checks"][-1]
    assert (gpus["name"], gpus["passed"]) == ("GPUs", False)
    assert "NVIDIA Container Toolkit" in gpus["message"]


def test_preflight_passes_when_gpus_cannot_be_read() -> None:
    result = _preflight_with_gpus("unknown", "The jobs API may not list nodes.")

    assert result.exit_code == 0, result.output
    assert (
        json.loads(result.output)["checks"][-1]["message"]
        == "gpu/default: GPUs not checked (The jobs API may not list nodes.)."
    )


def test_preflight_fails_without_the_platform() -> None:
    jobs = MagicMock()
    jobs.__enter__.return_value.list_execution_profiles.side_effect = ConnectionError("refused")
    with patch("nhx.customization_common.cli.dry_run.JobsClient", return_value=jobs):
        result = CliRunner().invoke(AutomodelContributor().get_cli(), ["preflight"])

    assert result.exit_code == 1
    [check] = json.loads(result.output)["checks"]
    assert check["message"].startswith("Cannot reach http://nhx.test: refused.")


@respx.mock
def test_table_output_reads_as_one_line_per_check(tmp_path: Path) -> None:
    job = _job_json(tmp_path, model="default/m", dataset={"training": "default/d"})
    _mock_create(False)
    with patch("nhx.customization_common.cli.dry_run.resolve_output_format", return_value="table"):
        result = CliRunner().invoke(AutomodelContributor().get_cli(), ["--job-json", str(job), "--dry-run"])

    assert result.output.splitlines() == [
        "OK   execution profile: Configured: gpu/default.",
        "FAIL training dataset: ws/d#train.jsonl line 3 ...",
        "Dry run failed. Nothing was created.",
    ]
