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

    def test_tracking_message_goes_to_stderr(self) -> None:
        result = _run("--workspace", "acme")
        assert "Submitted automodel job automodel-1a2b3c to workspace acme." in result.stderr
        assert "nemo jobs watch automodel-1a2b3c" in result.stderr

    def test_does_not_suggest_flags_that_no_longer_apply(self) -> None:
        """The job is already submitted, so --wait and --watch stay on submit --help."""
        stderr = _run().stderr
        assert "Submitted automodel job" in stderr
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
        assert result.exit_code != 0
        assert "not both" in result.stderr

    def test_no_flags_does_not_follow_the_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(**_: Any) -> FollowResult:
            raise AssertionError("submit must not block without --wait or --watch")

        monkeypatch.setattr("nmp.customization_common.cli.overrides.follow_job", _fail)
        assert _run().exit_code == 0


@pytest.mark.usefixtures("stub_submit")
def test_missing_job_name_is_not_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job id comes from the submit response, never from listing recent jobs."""

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
    result = _run("--wait")
    assert result.exit_code == 0, result.stderr
    assert "Submitted" not in result.stderr
