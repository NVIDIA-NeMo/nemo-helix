# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo customization automodel --dry-run``."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import respx
from nemo_automodel_plugin.contributor import AutomodelContributor
from typer.testing import CliRunner

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
