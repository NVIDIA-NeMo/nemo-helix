# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for what ``submit`` reports after creating a job.

These drive the real Typer command, so they cover the shared override in
``nmp.customization_common.cli.overrides`` as the three backends use it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_automodel_plugin.contributor import AutomodelContributor
from nemo_automodel_plugin.jobs.jobs import AutomodelJob
from nmp.customization_common.cli.tracking import FollowResult
from typer.testing import CliRunner

FIXTURES = Path(__file__).parent / "fixtures"
JOB_JSON = FIXTURES / "minimal_sft_lora.json"


@pytest.fixture
def stub_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make submit_remote return a job without touching the network."""

    def fake_submit_remote(
        _scheduler,
        job_cls: type,
        spec_data: dict,
        base_url: str | None = None,
        workspace: str = "default",
        profile: str | None = None,
        options: dict | None = None,
        metadata: dict | None = None,
        http_client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        return {"name": "automodel-1a2b3c", "status": "created"}

    monkeypatch.setattr(
        "nemo_platform_plugin.commands.NemoJobScheduler.submit_remote",
        fake_submit_remote,
    )
    monkeypatch.setattr(
        "nemo_platform_plugin.discovery.discover_jobs",
        lambda: {"customization.automodel.jobs": AutomodelJob},
    )


def _run(*args: str) -> Any:
    return CliRunner().invoke(
        AutomodelContributor().get_cli(),
        ["submit", str(JOB_JSON), "--base-url", "https://nmp.test", *args],
    )


@pytest.mark.usefixtures("stub_submit")
class TestTrackingMessage:
    def test_stdout_stays_parsable_json(self) -> None:
        """Scripts read the job off stdout; the tracking message must not land there."""
        result = _run()
        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == {"name": "automodel-1a2b3c", "status": "created"}

    def test_tracking_message_goes_to_stderr_once(self) -> None:
        result = _run("--workspace", "acme")
        assert "nemo jobs watch automodel-1a2b3c --workspace acme" in result.stderr
        assert result.stderr.count("nemo jobs get-status automodel-1a2b3c") == 1

    def test_does_not_suggest_flags_that_no_longer_apply(self) -> None:
        """The job is already submitted, so --wait and --watch stay on submit --help."""
        stderr = _run().stderr
        assert "Job submitted. Track it with:" in stderr
        assert "--wait" not in stderr
        assert "--watch" not in stderr


@pytest.mark.usefixtures("stub_submit")
class TestWaitAndWatch:
    def test_wait_follows_the_job_without_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_follow(**kwargs: Any) -> FollowResult:
            captured.update(kwargs)
            return FollowResult.SUCCEEDED

        monkeypatch.setattr("nmp.customization_common.cli.overrides.follow_job", fake_follow)
        result = _run("--wait", "--workspace", "acme", "--poll-interval", "7")
        assert result.exit_code == 0, result.stderr
        assert captured["job_name"] == "automodel-1a2b3c"
        assert captured["workspace"] == "acme"
        assert captured["base_url"] == "https://nmp.test"
        assert captured["include_logs"] is False
        assert captured["poll_interval"] == 7

    def test_watch_asks_for_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_follow(**kwargs: Any) -> FollowResult:
            captured.update(kwargs)
            return FollowResult.SUCCEEDED

        monkeypatch.setattr("nmp.customization_common.cli.overrides.follow_job", fake_follow)
        assert _run("--watch").exit_code == 0
        assert captured["include_logs"] is True

    def test_failed_job_exits_non_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "nmp.customization_common.cli.overrides.follow_job",
            lambda **_: FollowResult.FAILED,
        )
        assert _run("--wait").exit_code == 1

    def test_interrupt_exits_130(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "nmp.customization_common.cli.overrides.follow_job",
            lambda **_: FollowResult.INTERRUPTED,
        )
        assert _run("--wait").exit_code == 130

    def test_wait_and_watch_together_is_rejected(self) -> None:
        result = _run("--wait", "--watch")
        assert result.exit_code == 2
        assert "not both" in result.stderr

    def test_no_flags_does_not_follow_the_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(**_: Any) -> FollowResult:
            raise AssertionError("submit must not block without --wait or --watch")

        monkeypatch.setattr("nmp.customization_common.cli.overrides.follow_job", _fail)
        assert _run().exit_code == 0


@pytest.fixture
def unnamed_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """A submit response that carries no job name."""

    def fake_submit_remote(*_args: Any, **_kwargs: Any) -> dict:
        return {"status": "created"}

    monkeypatch.setattr(
        "nemo_platform_plugin.commands.NemoJobScheduler.submit_remote",
        fake_submit_remote,
    )
    monkeypatch.setattr(
        "nmp.customization_common.cli.overrides.follow_job",
        lambda **_: pytest.fail("must not follow a job it cannot name"),
    )


@pytest.mark.usefixtures("stub_submit", "unnamed_job")
@pytest.mark.parametrize("flag", ["--wait", "--watch"])
def test_missing_job_name_fails_the_follow(flag: str) -> None:
    """Exit 0 means the job completed; a job that cannot be followed is not guessed at."""
    result = _run(flag)
    assert result.exit_code == 1
    assert "no job name" in result.stderr


@pytest.mark.usefixtures("stub_submit", "unnamed_job")
def test_missing_job_name_without_follow_still_succeeds() -> None:
    assert _run().exit_code == 0


@pytest.fixture
def submitted_spec(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the spec that reaches the scheduler."""
    captured: dict[str, Any] = {}

    def fake_submit_remote(_scheduler, job_cls, spec_data, **kwargs: Any) -> dict:
        captured.update(spec_data)
        return {"name": "automodel-1a2b3c"}

    monkeypatch.setattr(
        "nemo_platform_plugin.commands.NemoJobScheduler.submit_remote",
        fake_submit_remote,
    )
    monkeypatch.setattr(
        "nemo_platform_plugin.discovery.discover_jobs",
        lambda: {"customization.automodel.jobs": AutomodelJob},
    )
    return captured


@pytest.fixture
def stub_uploads(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace resource creation, keeping the flag wiring under test."""
    seen: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> Any:
        from nmp.customization_common.cli.uploads import UploadReport

        seen.update(kwargs)
        report = UploadReport()
        if kwargs.get("model_source") is not None:
            report.model_ref = "default/created-model"
        if kwargs.get("dataset_source") is not None:
            report.dataset_ref = "default/created-dataset"
        return report

    monkeypatch.setattr("nmp.customization_common.cli.overrides.create_resources", fake_create)
    monkeypatch.setattr(
        "nmp.customization_common.cli.overrides.resolve_submit_base_url",
        lambda *a, **k: "https://nmp.test",
    )
    monkeypatch.setattr(
        "nmp.customization_common.cli.overrides.resolve_submit_auth_headers",
        lambda *a, **k: {},
    )
    return seen


def _write_job(tmp_path: Path, **fields: Any) -> Path:
    """Write a job JSON holding *fields* plus the training block every job needs.

    A field an upload flag will create must be absent; anything a flag will not
    create has to be present, because the API requires both model and dataset.
    """
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"training": {"training_type": "sft"}, **fields}))
    return path


def _run_job(job: Path, *args: str) -> Any:
    return CliRunner().invoke(
        AutomodelContributor().get_cli(),
        ["submit", str(job), "--base-url", "https://nmp.test", *args],
    )


class TestUploadFlags:
    """The created references must be what reaches submit_remote."""

    def test_upload_model_only(
        self, tmp_path: Path, submitted_spec: dict[str, Any], stub_uploads: dict[str, Any]
    ) -> None:
        job = _write_job(tmp_path, dataset={"training": "default/keep"})
        result = _run_job(job, "--upload-model", "Qwen/Qwen3-1.7B")
        assert result.exit_code == 0, result.stderr
        assert stub_uploads["model_source"] == "Qwen/Qwen3-1.7B"
        assert stub_uploads["dataset_source"] is None
        assert submitted_spec["model"] == "default/created-model"
        assert submitted_spec["dataset"]["training"] == "default/keep"

    def test_upload_dataset_only(
        self, tmp_path: Path, submitted_spec: dict[str, Any], stub_uploads: dict[str, Any]
    ) -> None:
        job = _write_job(tmp_path, model="default/keep")
        result = _run_job(job, "--upload-dataset", "./data")
        assert result.exit_code == 0, result.stderr
        assert stub_uploads["model_source"] is None
        assert submitted_spec["dataset"]["training"] == "default/created-dataset"
        assert submitted_spec["model"] == "default/keep"

    def test_both_together(self, tmp_path: Path, submitted_spec: dict[str, Any], stub_uploads: dict[str, Any]) -> None:
        result = _run_job(_write_job(tmp_path), "--upload-model", "Qwen/Qwen3-1.7B", "--upload-dataset", "./data")
        assert result.exit_code == 0, result.stderr
        assert submitted_spec["model"] == "default/created-model"
        assert submitted_spec["dataset"]["training"] == "default/created-dataset"

    def test_exist_ok_reaches_both(
        self, tmp_path: Path, submitted_spec: dict[str, Any], stub_uploads: dict[str, Any]
    ) -> None:
        """One --exist-ok covers whichever sources were given."""
        result = _run_job(_write_job(tmp_path), "--upload-model", "m", "--upload-dataset", "./d", "--exist-ok")
        assert result.exit_code == 0, result.stderr
        assert stub_uploads["exist_ok"] is True

    def test_hf_token_secret_is_passed_through(
        self, tmp_path: Path, submitted_spec: dict[str, Any], stub_uploads: dict[str, Any]
    ) -> None:
        job = _write_job(tmp_path, dataset={"training": "default/keep"})
        result = _run_job(job, "--upload-model", "meta-llama/Llama-3.1-8B", "--hf-token-secret", "hf-token")
        assert result.exit_code == 0, result.stderr
        assert stub_uploads["hf_token_secret"] == "hf-token"

    def test_no_flag_creates_nothing(self, submitted_spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(**_: Any) -> Any:
            raise AssertionError("nothing must be created without a flag")

        monkeypatch.setattr("nmp.customization_common.cli.overrides.create_resources", _fail)
        assert _run().exit_code == 0
        assert "created" not in str(submitted_spec)

    def test_failure_reports_cleanly_and_does_not_submit(
        self, tmp_path: Path, submitted_spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nmp.customization_common.cli.uploads import UploadError

        def _boom(**_: Any) -> Any:
            raise UploadError("Fileset default/train already exists. Pass --exist-ok to reuse it as it is.")

        monkeypatch.setattr("nmp.customization_common.cli.overrides.create_resources", _boom)
        monkeypatch.setattr(
            "nmp.customization_common.cli.overrides.resolve_submit_base_url",
            lambda *a, **k: "https://nmp.test",
        )
        monkeypatch.setattr(
            "nmp.customization_common.cli.overrides.resolve_submit_auth_headers",
            lambda *a, **k: {},
        )
        result = _run_job(_write_job(tmp_path, model="default/keep"), "--upload-dataset", "./data")
        assert result.exit_code == 2
        assert "already exists" in result.stderr
        assert "Traceback" not in result.stderr
        assert submitted_spec == {}, "a failed creation must not submit the job"


class TestJobJsonMayOmitUploadedFields:
    """model and dataset stay required by the API, so the CLI fills them in first."""

    @pytest.fixture
    def minimal_job(self, tmp_path: Path) -> Path:
        """A job JSON with neither model nor dataset."""
        path = tmp_path / "job.json"
        path.write_text(json.dumps({"training": {"training_type": "sft"}}))
        return path

    def _run_minimal(self, job: Path, *args: str) -> Any:
        return CliRunner().invoke(
            AutomodelContributor().get_cli(),
            ["submit", str(job), "--base-url", "https://nmp.test", *args],
        )

    def test_both_fields_can_be_omitted_when_both_are_uploaded(
        self, minimal_job: Path, submitted_spec: dict[str, Any], stub_uploads: dict[str, Any]
    ) -> None:
        result = self._run_minimal(minimal_job, "--upload-model", "Qwen/Qwen3-1.7B", "--upload-dataset", "./data")
        assert result.exit_code == 0, result.stderr
        assert submitted_spec["model"] == "default/created-model"
        assert submitted_spec["dataset"]["training"] == "default/created-dataset"

    def test_a_field_with_no_matching_flag_is_still_required(
        self, minimal_job: Path, submitted_spec: dict[str, Any], stub_uploads: dict[str, Any]
    ) -> None:
        result = self._run_minimal(minimal_job, "--upload-dataset", "./data")
        assert result.exit_code == 2
        assert "model" in result.stderr
        assert "Traceback" not in result.stderr

    def test_the_rest_of_the_spec_is_checked_before_anything_is_created(
        self, tmp_path: Path, submitted_spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An error elsewhere must not cost the user an upload."""
        job = tmp_path / "job.json"
        job.write_text(json.dumps({"training": {"training_type": "sft", "nonsense": 1}}))

        def _fail(**_: Any) -> Any:
            raise AssertionError("must not create resources for a spec that cannot validate")

        monkeypatch.setattr("nmp.customization_common.cli.overrides.create_resources", _fail)
        monkeypatch.setattr(
            "nmp.customization_common.cli.overrides.resolve_submit_base_url",
            lambda *a, **k: "https://nmp.test",
        )
        monkeypatch.setattr(
            "nmp.customization_common.cli.overrides.resolve_submit_auth_headers",
            lambda *a, **k: {},
        )
        result = self._run_minimal(job, "--upload-model", "Qwen/Qwen3-1.7B")
        assert result.exit_code == 2
        assert "nonsense" in result.stderr

    def test_malformed_json_is_reported_cleanly(self, tmp_path: Path) -> None:
        job = tmp_path / "job.json"
        job.write_text("{not json")
        result = self._run_minimal(job, "--upload-model", "Qwen/Qwen3-1.7B")
        assert result.exit_code == 2
        assert "Traceback" not in result.stderr


class TestConflictingReferences:
    """A reference set in both the job JSON and a flag must be refused, not overwritten."""

    def _run_with(self, job: Path, *args: str) -> Any:
        return CliRunner().invoke(
            AutomodelContributor().get_cli(),
            ["submit", str(job), "--base-url", "https://nmp.test", *args],
        )

    def test_a_filled_job_json_plus_a_flag_is_refused(
        self, submitted_spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail(**_: Any) -> Any:
            raise AssertionError("must not create resources when the reference already exists")

        monkeypatch.setattr("nmp.customization_common.cli.overrides.create_resources", _fail)
        result = self._run_with(JOB_JSON, "--upload-model", "Qwen/Qwen3-1.7B")

        assert result.exit_code == 2
        assert "already sets model" in result.stderr
        assert submitted_spec == {}, "a refused submit must not reach the platform"

    def test_the_stale_validation_field_is_caught(
        self, tmp_path: Path, submitted_spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reported bug: training was replaced while validation kept pointing elsewhere."""
        job = tmp_path / "job.json"
        job.write_text(
            json.dumps(
                {
                    "dataset": {"training": "default/old", "validation": "default/old"},
                    "training": {"training_type": "sft"},
                }
            )
        )
        monkeypatch.setattr(
            "nmp.customization_common.cli.overrides.create_resources",
            lambda **_: pytest.fail("must not upload"),
        )
        result = self._run_with(job, "--upload-dataset", "./data")
        assert result.exit_code == 2
        assert "dataset.validation" in result.stderr

    def test_only_the_flagged_field_is_checked(
        self, tmp_path: Path, submitted_spec: dict[str, Any], stub_uploads: dict[str, Any]
    ) -> None:
        """A model in the file is fine when only the dataset is being uploaded."""
        job = tmp_path / "job.json"
        job.write_text(json.dumps({"model": "default/keep", "training": {"training_type": "sft"}}))
        result = self._run_with(job, "--upload-dataset", "./data")

        assert result.exit_code == 0, result.stderr
        assert submitted_spec["model"] == "default/keep"

    def test_an_empty_job_json_fills_both_dataset_fields(
        self, tmp_path: Path, submitted_spec: dict[str, Any], stub_uploads: dict[str, Any]
    ) -> None:
        job = tmp_path / "job.json"
        job.write_text(json.dumps({"training": {"training_type": "sft"}}))
        result = self._run_with(job, "--upload-model", "Qwen/Qwen3-0.6B", "--upload-dataset", "./data")

        assert result.exit_code == 0, result.stderr
        assert submitted_spec["dataset"] == {
            "training": "default/created-dataset",
            "validation": "default/created-dataset",
        }
