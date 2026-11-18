# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``AuditJob.run`` — exercises the garak invocation and
result-collection plumbing without actually shelling out to garak."""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from nemo_garak.config import GarakPluginConfig
from nemo_garak.entities import (
    AuditConfig,
    AuditPluginsData,
    AuditReportData,
    AuditRunData,
    AuditSystemData,
)
from nemo_garak.entities import (
    AuditTarget as AuditTargetEntity,
)
from nemo_garak.jobs.audit import (
    GARAK_PYTHON_ENVVAR,
    AuditInputSpec,
    AuditJob,
    AuditSpec,
    GarakFailure,
    _aggregate_reports,
    _collect_report_artifacts,
    _divide_and_write_confs,
    _garak_config_dict,
    _gateway_token,
    _resolve_garak_python,
    _rewrite_options_uris,
    _uses_inference_gateway,
)
from nemo_garak.jobs.publication import (
    IntakePublicationSpec,
    PublicationFailedError,
    PublicationOutcome,
    PublicationSpec,
)
from nemo_helix_plugin.client.auth import StaticToken
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.config import clear_nemo_config_override, set_nemo_config_override
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError
from nemo_helix_plugin.job_context import JobContext, StoragePaths
from nemo_helix_plugin.job_results import LocalJobResults
from nemo_helix_plugin.jobs.schemas import HelixJobStatus

# The probe name returned by parse_plugin_spec for "encoding.InjectAscii85".
_PROBE_NAME = "encoding.InjectAscii85"
_PROBE_FULL = f"probes.{_PROBE_NAME}"


def _make_ctx(tmp_path: Path) -> JobContext:
    ephemeral = tmp_path / "ephemeral"
    persistent = tmp_path / "persistent"
    ephemeral.mkdir()
    persistent.mkdir()
    return JobContext(
        workspace="default",
        storage=StoragePaths(ephemeral=ephemeral, persistent=persistent),
        results=LocalJobResults(persistent / "results"),
        job_id=None,
    )


def _make_config(**overrides) -> AuditConfig:
    defaults: dict = {
        "name": "test-cfg",
        "workspace": "default",
        "description": "test",
        "system": AuditSystemData(lite=True, parallel_attempts=1),
        "run": AuditRunData(generations=1),
        "plugins": AuditPluginsData(probe_spec="encoding.InjectAscii85"),
        "reporting": AuditReportData(report_prefix="run1", report_dir="garak_runs"),
    }
    defaults.update(overrides)
    return AuditConfig(**defaults)


def _make_target(**overrides) -> AuditTargetEntity:
    defaults: dict = {
        "name": "test-tgt",
        "workspace": "default",
        "type": "test",
        "model": "test.Blank",
        "options": {},
    }
    defaults.update(overrides)
    return AuditTargetEntity(**defaults)


def _make_spec_dict(**overrides) -> dict:
    cfg = overrides.pop("config", _make_config())
    tgt = overrides.pop("target", _make_target())
    d = {
        "config": cfg.model_dump(mode="json"),
        "target": tgt.model_dump(mode="json"),
    }
    d.update(overrides)
    return d


def _plant_reports(persistent: Path, prefix: str, kinds: tuple[str, ...], report_dir: str = "garak_runs") -> None:
    """Plant fake aggregated garak report files where ``_collect_report_artifacts`` will look."""
    d = persistent / "garak" / report_dir
    d.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        (d / f"{prefix}{kind}").write_text(f"fake-{kind}")


def _plant_probe_success(
    persistent: Path,
    probe_name: str,
    report_prefix: str = "run1",
    report_dir: str = "garak_runs",
) -> None:
    """Plant the per-probe HTML success marker that AuditJob uses to detect probe completion."""
    marker = persistent / "running" / probe_name / "garak" / report_dir / f"{report_prefix}.report.html"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_garak_python(tmp_path: Path, monkeypatch) -> Path:
    """Create an empty file standing in for the garak interpreter and point the
    job at it via the env var. AuditJob only checks existence, never executes
    it (subprocess.run is patched separately)."""
    interp = tmp_path / "garak-python"
    interp.touch()
    monkeypatch.setenv("NEMO_GARAK_PYTHON", str(interp))
    return interp


@pytest.fixture
def fake_parse_plugin_spec():
    """Patch garakapi.parse_plugin_spec to return a single known probe without
    requiring a real garak plugin cache."""
    with patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as mock:
        mock.return_value = ([_PROBE_FULL], [])
        yield mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestGarakConfigDict:
    def test_drops_entity_base_fields_and_description(self) -> None:
        cfg = _make_config()
        out = _garak_config_dict(cfg)
        assert set(out.keys()) == {"system", "run", "plugins", "reporting"}
        for forbidden in ("name", "workspace", "description", "id", "entity_type"):
            assert forbidden not in out

    def test_preserves_nested_values(self) -> None:
        cfg = _make_config(
            system=AuditSystemData(lite=False, parallel_attempts=8),
            run=AuditRunData(generations=5, eval_threshold=0.7),
        )
        out = _garak_config_dict(cfg)
        assert out["system"]["parallel_attempts"] == 8
        assert out["run"]["generations"] == 5
        assert out["run"]["eval_threshold"] == 0.7


# ---------------------------------------------------------------------------
# _divide_and_write_confs
# ---------------------------------------------------------------------------


class TestDivideAndWriteConfs:
    def test_writes_one_yaml_per_probe(self, tmp_path: Path) -> None:
        todo = tmp_path / "todo"
        todo.mkdir()
        config_dict = {
            "plugins": {"probe_spec": "encoding", "detector_spec": "auto"},
            "run": {"probe_tags": None},
            "system": {},
            "reporting": {},
        }
        with patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as mock:
            mock.return_value = (["probes.encoding.InjectAscii85", "probes.encoding.InjectBase16"], [])
            _divide_and_write_confs(config_dict, todo)

        yamls = sorted(todo.glob("*.yaml"))
        assert len(yamls) == 2
        assert {y.stem for y in yamls} == {"encoding.InjectAscii85", "encoding.InjectBase16"}

    def test_per_probe_yaml_has_single_probe_spec(self, tmp_path: Path) -> None:
        todo = tmp_path / "todo"
        todo.mkdir()
        config_dict = {
            "plugins": {"probe_spec": "encoding.InjectAscii85", "detector_spec": "auto"},
            "run": {"probe_tags": None},
        }
        with patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as mock:
            mock.return_value = ([_PROBE_FULL], [])
            _divide_and_write_confs(config_dict, todo)

        loaded = yaml.safe_load((todo / f"{_PROBE_NAME}.yaml").read_text())
        assert loaded["plugins"]["probe_spec"] == _PROBE_NAME
        # Other plugin keys are preserved.
        assert loaded["plugins"]["detector_spec"] == "auto"

    def test_raises_on_empty_activated_list(self, tmp_path: Path) -> None:
        todo = tmp_path / "todo"
        todo.mkdir()
        with patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as mock:
            mock.return_value = ([], [])
            with pytest.raises(GarakFailure, match="No probes found"):
                _divide_and_write_confs({"plugins": {"probe_spec": "nonexistent"}, "run": {}}, todo)

    def test_raises_on_unknown_probes(self, tmp_path: Path) -> None:
        todo = tmp_path / "todo"
        todo.mkdir()
        with patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as mock:
            mock.return_value = ([], ["bad.probe"])
            with pytest.raises(GarakFailure, match="Invalid probe"):
                _divide_and_write_confs({"plugins": {"probe_spec": "bad.probe"}, "run": {}}, todo)

    def test_passes_probe_tags_to_parse_plugin_spec(self, tmp_path: Path) -> None:
        todo = tmp_path / "todo"
        todo.mkdir()
        config_dict = {"plugins": {"probe_spec": "all"}, "run": {"probe_tags": "owasp:llm06"}}
        with patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as mock:
            mock.return_value = ([_PROBE_FULL], [])
            _divide_and_write_confs(config_dict, todo)

        mock.assert_called_once_with("all", "probes", "owasp:llm06")

    def test_normalises_none_probe_tags_to_empty_string(self, tmp_path: Path) -> None:
        todo = tmp_path / "todo"
        todo.mkdir()
        config_dict = {"plugins": {"probe_spec": "encoding"}, "run": {"probe_tags": None}}
        with patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as mock:
            mock.return_value = ([_PROBE_FULL], [])
            _divide_and_write_confs(config_dict, todo)

        mock.assert_called_once_with("encoding", "probes", "")


# ---------------------------------------------------------------------------
# _aggregate_reports
# ---------------------------------------------------------------------------


class TestAggregateReports:
    def test_returns_false_when_no_completed_jsonls(self, tmp_path: Path, fake_garak_python: Path) -> None:
        (tmp_path / "complete").mkdir()
        result = _aggregate_reports(tmp_path, "garak_runs", "run1", str(fake_garak_python))
        assert result is False

    def test_calls_aggregate_reports_and_report_digest(self, tmp_path: Path, fake_garak_python: Path) -> None:
        # Plant a per-probe JSONL so the function has something to aggregate.
        jsonl = tmp_path / "complete" / _PROBE_NAME / "garak" / "garak_runs" / "run1.report.jsonl"
        jsonl.parent.mkdir(parents=True)
        jsonl.write_text("{}")

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            result = _aggregate_reports(tmp_path, "garak_runs", "run1", str(fake_garak_python))

        assert result is True
        assert mock_run.call_count == 2
        cmds = [call.args[0] for call in mock_run.call_args_list]
        assert any("aggregate_reports" in " ".join(c) for c in cmds)
        assert any("report_digest" in " ".join(c) for c in cmds)

    def test_raises_garak_failure_when_aggregate_subprocess_fails(
        self, tmp_path: Path, fake_garak_python: Path
    ) -> None:
        jsonl = tmp_path / "complete" / _PROBE_NAME / "garak" / "garak_runs" / "run1.report.jsonl"
        jsonl.parent.mkdir(parents=True)
        jsonl.write_text("{}")

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stderr="boom")
            with pytest.raises(GarakFailure, match="aggregate_reports failed"):
                _aggregate_reports(tmp_path, "garak_runs", "run1", str(fake_garak_python))

    def test_concatenates_hitlogs(self, tmp_path: Path, fake_garak_python: Path) -> None:
        for probe in ("probeA", "probeB"):
            d = tmp_path / "complete" / probe / "garak" / "garak_runs"
            d.mkdir(parents=True)
            (d / "run1.report.jsonl").write_text("{}")
            (d / "run1.hitlog.jsonl").write_bytes(b"hit-" + probe.encode())

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            _aggregate_reports(tmp_path, "garak_runs", "run1", str(fake_garak_python))

        hitlog = tmp_path / "garak" / "garak_runs" / "run1.hitlog.jsonl"
        assert hitlog.exists()
        contents = hitlog.read_bytes()
        assert b"hit-probeA" in contents
        assert b"hit-probeB" in contents


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------


class TestAuditJobRun:
    def test_invokes_garak_with_expected_argv_and_env(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        spec = _make_spec_dict()

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            # returncode=0 but no HTML planted → probe "fails", GarakFailure caught internally.
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            AuditJob().run(spec, ctx=ctx)

        # One call: the probe invocation (aggregation not reached after failure).
        assert mock_run.call_count == 1
        call_args = mock_run.call_args
        argv = call_args.args[0]

        assert argv[0] == str(fake_garak_python)
        assert argv[1:3] == ["-m", "garak"]
        assert "--config" in argv
        cfg_idx = argv.index("--config")
        # Per-probe config is copied into running/<probe>/config.yaml.
        assert argv[cfg_idx + 1].endswith("config.yaml")
        assert ["--target_type", "test"] == argv[argv.index("--target_type") : argv.index("--target_type") + 2]
        assert ["--target_name", "test.Blank"] == argv[argv.index("--target_name") : argv.index("--target_name") + 2]
        # No options on the default target → no --generator_option_file.
        assert "--generator_option_file" not in argv

        env = call_args.kwargs["env"]
        for key in ("NIM_API_KEY", "OPENAI_API_KEY", "REST_API_KEY", "OPENAICOMPATIBLE_API_KEY"):
            assert env[key]
        # XDG_DATA_HOME is the per-probe running dir, not the persistent root.
        assert env["XDG_DATA_HOME"] == str(ctx.storage.persistent / "running" / _PROBE_NAME)
        assert env["GARAK_LOG_FILE"].endswith("garak.log")

    def test_yaml_config_only_has_garak_sections(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        spec = _make_spec_dict(config=_make_config(description="will-be-stripped"))

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            AuditJob().run(spec, ctx=ctx)

        # The per-probe config is moved to failed/ after the probe fails (no HTML).
        probe_config = ctx.storage.persistent / "failed" / _PROBE_NAME / "config.yaml"
        assert probe_config.exists()
        loaded = yaml.safe_load(probe_config.read_text())
        assert set(loaded.keys()) == {"system", "run", "plugins", "reporting"}
        assert "description" not in loaded
        assert "name" not in loaded

    def test_target_options_written_when_present_and_flag_added(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        spec = _make_spec_dict(target=_make_target(options={"endpoint": "https://example.invalid", "key_env": "X"}))

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            AuditJob().run(spec, ctx=ctx)

        # Target options are written to persistent storage (survives pause/resume).
        opts_path = ctx.storage.persistent / "target_options.json"
        assert opts_path.exists()
        assert json.loads(opts_path.read_text()) == {"endpoint": "https://example.invalid", "key_env": "X"}

        argv = mock_run.call_args.args[0]
        assert "--generator_option_file" in argv
        assert argv[argv.index("--generator_option_file") + 1].endswith("target_options.json")

    def test_missing_garak_interpreter_raises_clear_error(self, tmp_path: Path, monkeypatch) -> None:
        ctx = _make_ctx(tmp_path)
        monkeypatch.setenv("NEMO_GARAK_PYTHON", str(tmp_path / "does-not-exist"))
        with pytest.raises(FileNotFoundError, match="garak interpreter not found"):
            AuditJob().run(_make_spec_dict(), ctx=ctx)

    def test_garak_interperter_resolution_env_override(self, monkeypatch) -> None:
        monkeypatch.delenv(GARAK_PYTHON_ENVVAR, raising=False)
        assert len(_resolve_garak_python()) > 0
        env_path = "env_path"
        monkeypatch.setenv(GARAK_PYTHON_ENVVAR, env_path)
        garak_pythons = _resolve_garak_python()
        assert len(garak_pythons) == 1
        assert garak_pythons[0] == env_path

    def test_completed_run_collects_all_three_artifacts(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        spec = _make_spec_dict()

        def fake_run(cmd, **kwargs):
            # Plant the per-probe HTML success marker so the probe is treated as complete.
            xdg = kwargs["env"]["XDG_DATA_HOME"]
            marker = Path(xdg) / "garak" / "garak_runs" / "run1.report.html"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok")
            return subprocess.CompletedProcess(args=[], returncode=0)

        with (
            patch("nemo_garak.jobs.audit.run_subprocess", side_effect=fake_run),
            patch("nemo_garak.jobs.audit._aggregate_reports", return_value=True),
        ):
            _plant_reports(ctx.storage.persistent, "run1", (".report.jsonl", ".report.html", ".hitlog.jsonl"))
            result = AuditJob().run(spec, ctx=ctx)

        assert result["status"] == "completed"
        assert result["probes_complete"] == 1
        assert set(result["results"].keys()) == {"report-jsonl", "report-html", "report-hitlog-jsonl"}
        for ref in result["results"].values():
            assert ref["artifact_url"].startswith("file://")
            assert Path(ref["artifact_url"][len("file://") :]).exists()

    def test_failed_run_returns_failed_status(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        spec = _make_spec_dict()

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=2)
            result = AuditJob().run(spec, ctx=ctx)

        assert result["status"] == "failed"
        assert "results" in result

    def test_failed_run_with_no_artifacts_still_returns_envelope(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        spec = _make_spec_dict()

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            result = AuditJob().run(spec, ctx=ctx)

        assert result["status"] == "failed"
        assert result["results"] == {}

    def test_uses_custom_report_prefix_and_dir(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        cfg = _make_config(reporting=AuditReportData(report_prefix="custom-prefix", report_dir="custom_dir"))
        spec = _make_spec_dict(config=cfg)

        def fake_run(cmd, **kwargs):
            xdg = kwargs["env"]["XDG_DATA_HOME"]
            marker = Path(xdg) / "garak" / "custom_dir" / "custom-prefix.report.html"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok")
            return subprocess.CompletedProcess(args=[], returncode=0)

        with (
            patch("nemo_garak.jobs.audit.run_subprocess", side_effect=fake_run),
            patch("nemo_garak.jobs.audit._aggregate_reports", return_value=True),
        ):
            _plant_reports(ctx.storage.persistent, "custom-prefix", (".report.jsonl",), report_dir="custom_dir")
            result = AuditJob().run(spec, ctx=ctx)

        assert result["status"] == "completed"
        assert "report-jsonl" in result["results"]

    # -----------------------------------------------------------------------
    # Scratch space and per-probe directory management
    # -----------------------------------------------------------------------

    def test_first_run_creates_scratch_directories(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            AuditJob().run(_make_spec_dict(), ctx=ctx)

        persistent = ctx.storage.persistent
        for d in ("todo", "running", "complete", "failed", "failed_probe_logs"):
            assert (persistent / d).exists(), f"Expected {d}/ to be created"

    def test_successful_probe_lands_in_complete(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)

        def fake_run(cmd, **kwargs):
            xdg = kwargs["env"]["XDG_DATA_HOME"]
            marker = Path(xdg) / "garak" / "garak_runs" / "run1.report.html"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok")
            return subprocess.CompletedProcess(args=[], returncode=0)

        with (
            patch("nemo_garak.jobs.audit.run_subprocess", side_effect=fake_run),
            patch("nemo_garak.jobs.audit._aggregate_reports", return_value=False),
        ):
            AuditJob().run(_make_spec_dict(), ctx=ctx)

        assert (ctx.storage.persistent / "complete" / _PROBE_NAME).is_dir()
        # todo YAML removed after probe completes.
        assert not list((ctx.storage.persistent / "todo").glob("*.yaml"))

    def test_failed_probe_lands_in_failed(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            AuditJob().run(_make_spec_dict(), ctx=ctx)

        assert (ctx.storage.persistent / "failed" / _PROBE_NAME).is_dir()

    def test_probe_succeeds_on_retry(self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec) -> None:
        ctx = _make_ctx(tmp_path)
        call_count = 0

        def fake_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Second attempt plants the success marker.
                xdg = kwargs["env"]["XDG_DATA_HOME"]
                marker = Path(xdg) / "garak" / "garak_runs" / "run1.report.html"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("ok")
                return subprocess.CompletedProcess(args=[], returncode=0)
            return subprocess.CompletedProcess(args=[], returncode=1)

        spec = _make_spec_dict(max_probe_retries=1)
        with (
            patch("nemo_garak.jobs.audit.run_subprocess", side_effect=fake_run),
            patch("nemo_garak.jobs.audit._aggregate_reports", return_value=False),
        ):
            result = AuditJob().run(spec, ctx=ctx)

        assert result["probes_complete"] == 1
        assert (ctx.storage.persistent / "complete" / _PROBE_NAME).is_dir()
        # The failed-probe log directory was created for the first (failed) attempt.
        assert (ctx.storage.persistent / "failed_probe_logs" / _PROBE_NAME).is_dir()

    def test_retries_exhausted_fail_job_returns_failed(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        spec = _make_spec_dict(max_probe_retries=1, fail_job_on_retries_exhausted=True)

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            result = AuditJob().run(spec, ctx=ctx)

        assert result["status"] == "failed"
        assert _PROBE_NAME in result["error"]
        assert mock_run.call_count == 2  # 1 attempt + 1 retry

    def test_retries_exhausted_continue_gives_partial_result(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        """With fail_job_on_retries_exhausted=False and two probes, exhausted retries
        on the first probe are skipped and the second probe still runs."""
        ctx = _make_ctx(tmp_path)

        with patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as mock_spec:
            mock_spec.return_value = (
                ["probes.encoding.InjectAscii85", "probes.encoding.InjectBase16"],
                [],
            )
            call_order: list[str] = []

            def fake_run(cmd, **kwargs):
                xdg = kwargs["env"]["XDG_DATA_HOME"]
                probe_dir = Path(xdg)
                if "InjectBase16" in str(probe_dir):
                    # Second probe succeeds.
                    marker = probe_dir / "garak" / "garak_runs" / "run1.report.html"
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text("ok")
                    call_order.append("success")
                    return subprocess.CompletedProcess(args=[], returncode=0)
                call_order.append("fail")
                return subprocess.CompletedProcess(args=[], returncode=1)

            spec = _make_spec_dict(fail_job_on_retries_exhausted=False)
            with (
                patch("nemo_garak.jobs.audit.run_subprocess", side_effect=fake_run),
                patch("nemo_garak.jobs.audit._aggregate_reports", return_value=False),
            ):
                result = AuditJob().run(spec, ctx=ctx)

        assert result["status"] == "partial"
        assert result["probes_complete"] == 1
        assert result["probes_failed"] == 1

    def test_all_probes_fail_returns_failed(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        spec = _make_spec_dict(fail_job_on_retries_exhausted=False)

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            result = AuditJob().run(spec, ctx=ctx)

        assert result["status"] == "failed"
        assert "All probes failed" in result["error"]

    # -----------------------------------------------------------------------
    # Pause / resume
    # -----------------------------------------------------------------------

    def test_resume_skips_initialization(self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec) -> None:
        """When todo/ already exists, _divide_and_write_confs must not be called again."""
        ctx = _make_ctx(tmp_path)
        persistent = ctx.storage.persistent

        # Simulate a prior run that left one probe in todo/.
        (persistent / "todo").mkdir(parents=True)
        (persistent / "running").mkdir()
        (persistent / "complete").mkdir()
        (persistent / "failed").mkdir()
        (persistent / "failed_probe_logs").mkdir()
        cfg_dict = _garak_config_dict(_make_config())
        (persistent / "todo" / f"{_PROBE_NAME}.yaml").write_text(yaml.safe_dump(cfg_dict))

        with (
            patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as mock_spec,
            patch("nemo_garak.jobs.audit.run_subprocess") as mock_run,
        ):
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            AuditJob().run(_make_spec_dict(), ctx=ctx)
            # parse_plugin_spec must not be called — initialization was skipped.
            mock_spec.assert_not_called()

    def test_resume_requeues_interrupted_probe(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        """A probe directory left in running/ is re-queued to todo/ on resume."""
        ctx = _make_ctx(tmp_path)
        persistent = ctx.storage.persistent

        (persistent / "todo").mkdir(parents=True)
        (persistent / "running").mkdir()
        (persistent / "complete").mkdir()
        (persistent / "failed").mkdir()
        (persistent / "failed_probe_logs").mkdir()
        # Simulate an interrupted probe: running/ has the probe dir with a config.
        probe_dir = persistent / "running" / _PROBE_NAME
        probe_dir.mkdir()
        cfg_dict = _garak_config_dict(_make_config())
        (probe_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict))

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            AuditJob().run(_make_spec_dict(), ctx=ctx)

        # The probe was re-queued and executed (appears in failed/ because returncode=1).
        assert (persistent / "failed" / _PROBE_NAME).is_dir()
        # running/<probe> was cleaned up after re-queue and execution.
        assert not (persistent / "running" / _PROBE_NAME).exists()

    # -----------------------------------------------------------------------
    # SIGTERM handler
    # -----------------------------------------------------------------------

    def test_sigterm_handler_aggregates_partial_results(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        """Directly invoke the SIGTERM closure and verify it calls aggregation."""
        ctx = _make_ctx(tmp_path)
        captured_handler: list = []

        original_signal = signal.signal

        def capture_signal(signum, handler):
            if signum == signal.SIGTERM:
                captured_handler.append(handler)
            return original_signal(signum, handler)

        with (
            patch("nemo_garak.jobs.audit.run_subprocess") as mock_run,
            patch("nemo_garak.jobs.audit.signal", side_effect=capture_signal),
            patch("nemo_garak.jobs.audit._aggregate_reports", return_value=False) as mock_agg,
            patch("nemo_garak.jobs.audit.sys_exit") as mock_exit,
        ):
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            AuditJob().run(_make_spec_dict(), ctx=ctx)

            assert captured_handler, "SIGTERM handler was not registered"
            # Invoke the handler inside the patch context so sys.exit remains mocked.
            captured_handler[0](signal.SIGTERM, None)
            mock_agg.assert_called()
            mock_exit.assert_called_with(0)


# ---------------------------------------------------------------------------
# AuditJob.compile — profile defaulting
# ---------------------------------------------------------------------------


class TestCompileProfileDefault:
    def teardown_method(self) -> None:
        clear_nemo_config_override(GarakPluginConfig)

    def _compile(self, profile: str | None) -> dict:
        result = asyncio.run(
            AuditJob.compile(
                workspace="default",
                spec=_make_config(),
                entity_client=None,
                job_name=None,
                async_sdk=cast(AsyncNemoClient, None),
                profile=profile,
            )
        )
        return cast(dict, result)

    def test_defaults_to_default_profile_when_unset(self) -> None:
        """ "default" is registered out of the box everywhere (Docker and Kubernetes),
        so audit jobs work without extra config in CI/k8s."""
        result = self._compile(None)
        assert result["steps"][0]["executor"]["profile"] == "default"

    def test_explicit_profile_overrides_default(self) -> None:
        result = self._compile("gpu-pool")
        assert result["steps"][0]["executor"]["profile"] == "gpu-pool"

    def test_plugin_config_overrides_unset_profile(self) -> None:
        """Deployments (e.g. local dev) that need a dedicated container-backed profile
        can point audit jobs at it via the garak plugin config."""
        set_nemo_config_override(GarakPluginConfig(job_executor_profile="garak"))
        result = self._compile(None)
        assert result["steps"][0]["executor"]["profile"] == "garak"

    def test_explicit_profile_wins_over_plugin_config(self) -> None:
        set_nemo_config_override(GarakPluginConfig(job_executor_profile="garak"))
        result = self._compile("gpu-pool")
        assert result["steps"][0]["executor"]["profile"] == "gpu-pool"


# ---------------------------------------------------------------------------
# Schema-level tests
# ---------------------------------------------------------------------------


class TestAuditSpec:
    def test_rejects_extra_top_level_fields(self) -> None:
        with pytest.raises(ValueError):
            AuditSpec.model_validate(
                {
                    "config": _make_config().model_dump(mode="json"),
                    "target": _make_target().model_dump(mode="json"),
                    "extra": "not-allowed",
                }
            )

    def test_requires_config_and_target(self) -> None:
        with pytest.raises(ValueError):
            AuditSpec.model_validate({"target": _make_target().model_dump(mode="json")})
        with pytest.raises(ValueError):
            AuditSpec.model_validate({"config": _make_config().model_dump(mode="json")})

    def test_default_task_options(self) -> None:
        spec = AuditSpec.model_validate(_make_spec_dict())
        assert spec.max_probe_retries == 0
        assert spec.fail_job_on_retries_exhausted is True

    def test_rejects_negative_max_probe_retries(self) -> None:
        with pytest.raises(ValueError):
            AuditSpec.model_validate({**_make_spec_dict(), "max_probe_retries": -1})


# ---------------------------------------------------------------------------
# Optional smoke: real ~/.garak venv reachable
# ---------------------------------------------------------------------------


GARAK_VENV_PYTHON = Path("~/.garak/.venv/bin/python").expanduser()


@pytest.mark.skipif(
    not GARAK_VENV_PYTHON.exists(),
    reason="garak venv not present at ~/.garak/.venv",
)
def test_garak_venv_is_reachable() -> None:
    """Sanity: the dev assumption that ``~/.garak/.venv`` ships garak holds."""
    completed = subprocess.run(
        [str(GARAK_VENV_PYTHON), "-c", "import garak; print(garak.__version__)"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip(), "expected a version string from garak"


# ---------------------------------------------------------------------------
# _collect_report_artifacts
# ---------------------------------------------------------------------------


class TestCollectReportArtifacts:
    def test_skips_missing_files(self, tmp_path: Path) -> None:
        results = LocalJobResults(tmp_path / "results")
        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        # Only the jsonl exists.
        (report_dir / "run1.report.jsonl").write_text("hi")

        artifacts = _collect_report_artifacts(report_dir, "run1", results)
        assert set(artifacts.keys()) == {"report-jsonl"}

    def test_returns_refs_with_local_file_urls(self, tmp_path: Path) -> None:
        results = LocalJobResults(tmp_path / "results")
        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        (report_dir / "run1.report.jsonl").write_text("a")
        (report_dir / "run1.report.html").write_text("b")

        artifacts = _collect_report_artifacts(report_dir, "run1", results)
        for name, ref in artifacts.items():
            assert ref["name"] == name
            assert ref["artifact_url"].startswith("file://")


# ---------------------------------------------------------------------------
# _rewrite_options_uris — nhx_uri_spec resolution via the platform SDK
# ---------------------------------------------------------------------------


def _mock_sdk(uri: str = "https://igw.example.invalid/v1") -> MagicMock:
    """Return a MagicMock that mimics the SDK calls _rewrite_options_uris uses.

    The mock exposes ``models_client``, the typed ``ModelsClient`` that
    ``client_from_platform(sdk, ModelsClient)`` returns in production. Its
    ``get_provider`` returns a ``.data()``-bearing provider, and
    ``get_provider_route_openai_url`` returns ``uri``.
    """
    sdk = MagicMock()

    class _Resp:
        def __init__(self, value):
            self._value = value

        def data(self):
            return self._value

    provider = MagicMock()

    models_client = MagicMock()
    models_client.get_provider.return_value = _Resp(provider)
    models_client.get_provider_route_openai_url.return_value = uri
    sdk.models_client = models_client
    return sdk


@pytest.fixture(autouse=True)
def _patch_client_from_platform():
    """Route ``client_from_platform(sdk, ModelsClient)`` in ``audit`` back to
    ``sdk.models_client`` so tests control the typed client directly."""
    with patch(
        "nemo_garak.jobs.audit.client_from_platform",
        side_effect=lambda sdk_or_async, _cls: sdk_or_async.models_client,
    ):
        yield


class TestRewriteOptionsUris:
    def test_replaces_nhx_uri_spec_at_top_level_nim(self) -> None:
        options = {
            "nim": {
                "skip_seq_start": "<think>",
                "skip_seq_end": "</think>",
                "max_tokens": 4000,
                "nhx_uri_spec": {
                    "inference_gateway": {"workspace": "default", "provider": "build"},
                },
            }
        }
        sdk = _mock_sdk("https://replaced-url")
        _rewrite_options_uris(options, sdk)

        assert options == {
            "nim": {
                "skip_seq_start": "<think>",
                "skip_seq_end": "</think>",
                "max_tokens": 4000,
                "uri": "https://replaced-url",
            }
        }
        sdk.models_client.get_provider.assert_called_once_with(workspace="default", name="build")

    def test_replaces_at_nested_openai_compatible(self) -> None:
        options = {
            "openai": {
                "OpenAICompatible": {
                    "nhx_uri_spec": {
                        "inference_gateway": {"workspace": "default", "provider": "openai"},
                    }
                }
            }
        }
        sdk = _mock_sdk("https://replaced-url")
        _rewrite_options_uris(options, sdk)
        assert options == {"openai": {"OpenAICompatible": {"uri": "https://replaced-url"}}}

    def test_no_op_when_no_sentinel(self) -> None:
        options = {
            "nim": {
                "skip_seq_start": "<think>",
                "max_tokens": 4000,
                "uri": "https://dont-replace-me",
            }
        }
        sdk = _mock_sdk()
        _rewrite_options_uris(options, sdk)
        assert options == {
            "nim": {
                "skip_seq_start": "<think>",
                "max_tokens": 4000,
                "uri": "https://dont-replace-me",
            }
        }
        sdk.models_client.get_provider.assert_not_called()

    def test_no_sdk_calls_when_options_have_no_sentinel_at_all(self) -> None:
        options = {"a": {"b": {"c": "leaf"}}, "d": "string"}
        sdk = _mock_sdk()
        _rewrite_options_uris(options, sdk)
        assert options == {"a": {"b": {"c": "leaf"}}, "d": "string"}
        sdk.models_client.get_provider.assert_not_called()

    def test_raises_on_missing_provider(self) -> None:
        options = {"nim": {"nhx_uri_spec": {"inference_gateway": {"workspace": "default"}}}}
        with pytest.raises(ValueError, match="Invalid nhx_uri_spec"):
            _rewrite_options_uris(options, _mock_sdk())

    def test_raises_on_missing_workspace(self) -> None:
        options = {"nim": {"nhx_uri_spec": {"inference_gateway": {"provider": "build"}}}}
        with pytest.raises(ValueError, match="Invalid nhx_uri_spec"):
            _rewrite_options_uris(options, _mock_sdk())

    def test_raises_on_missing_inference_gateway_key(self) -> None:
        options = {"nim": {"nhx_uri_spec": {"some_other_resolver": {}}}}
        with pytest.raises(ValueError, match="Invalid nhx_uri_spec"):
            _rewrite_options_uris(options, _mock_sdk())

    def test_raises_on_uri_and_sentinel_conflict(self) -> None:
        options = {
            "nim": {
                "uri": "https://this-should-not-exist",
                "nhx_uri_spec": {
                    "inference_gateway": {"workspace": "default", "provider": "build"},
                },
            }
        }
        with pytest.raises(ValueError, match="both 'uri' and 'nhx_uri_spec'"):
            _rewrite_options_uris(options, _mock_sdk())

    def test_raises_when_sentinel_present_but_sdk_is_none(self) -> None:
        options = {
            "nim": {
                "nhx_uri_spec": {
                    "inference_gateway": {"workspace": "default", "provider": "build"},
                }
            }
        }
        with pytest.raises(RuntimeError, match="requires a connected platform SDK"):
            _rewrite_options_uris(options, None)

    def test_raises_when_both_sdk_and_async_sdk_are_none(self) -> None:
        options = {
            "nim": {
                "nhx_uri_spec": {
                    "inference_gateway": {"workspace": "default", "provider": "build"},
                }
            }
        }
        with pytest.raises(RuntimeError, match="requires a connected platform SDK"):
            _rewrite_options_uris(options, None, async_sdk=None)

    def test_resolves_nhx_uri_spec_via_async_sdk(self) -> None:
        options = {
            "nim": {
                "max_tokens": 32,
                "nhx_uri_spec": {
                    "inference_gateway": {"workspace": "default", "provider": "nvidia-inference-api"},
                },
            }
        }
        async_sdk = MagicMock()
        async_sdk.models_client = MagicMock()
        async_sdk.models_client.get_provider = AsyncMock(return_value=MagicMock(data=lambda: MagicMock()))
        async_sdk.models_client.get_provider_route_openai_url.return_value = "https://igw-async.example/v1"

        _rewrite_options_uris(options, sdk=None, async_sdk=async_sdk)

        assert options == {"nim": {"max_tokens": 32, "uri": "https://igw-async.example/v1"}}
        async_sdk.models_client.get_provider.assert_called_once_with(workspace="default", name="nvidia-inference-api")

    def test_wraps_sdk_lookup_failure_in_runtimeerror(self) -> None:
        sdk = _mock_sdk()
        sdk.models_client.get_provider.side_effect = LookupError("no such provider")
        options = {
            "nim": {
                "nhx_uri_spec": {
                    "inference_gateway": {"workspace": "default", "provider": "ghost"},
                }
            }
        }
        with pytest.raises(RuntimeError, match="Failed to resolve inference gateway provider"):
            _rewrite_options_uris(options, sdk)


# ---------------------------------------------------------------------------
# AuditJob.run — end-to-end with nhx_uri_spec
# ---------------------------------------------------------------------------


class TestAuditJobIGW:
    def test_run_writes_options_with_resolved_uri_and_drops_sentinel(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        ctx = _make_ctx(tmp_path)
        target = _make_target(
            options={
                "nim": {
                    "max_tokens": 4000,
                    "nhx_uri_spec": {
                        "inference_gateway": {"workspace": "default", "provider": "build"},
                    },
                }
            }
        )
        spec = _make_spec_dict(target=target)
        sdk = _mock_sdk("https://igw-resolved.example/v1")

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            AuditJob().run(spec, ctx=ctx, sdk=sdk)

        # Options are written to persistent storage (not ephemeral).
        opts_path = ctx.storage.persistent / "target_options.json"
        assert opts_path.exists()
        on_disk = json.loads(opts_path.read_text())
        assert on_disk == {
            "nim": {
                "max_tokens": 4000,
                "uri": "https://igw-resolved.example/v1",
            }
        }
        # And the original validated spec is untouched.
        assert "nhx_uri_spec" in target.options["nim"]

    def test_run_without_sdk_when_no_sentinel_works(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        """sdk=None is fine when options carry no nhx_uri_spec."""
        ctx = _make_ctx(tmp_path)
        spec = _make_spec_dict(target=_make_target(options={"nim": {"max_tokens": 100}}))

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            AuditJob().run(spec, ctx=ctx)  # no sdk kwarg

        on_disk = json.loads((ctx.storage.persistent / "target_options.json").read_text())
        assert on_disk == {"nim": {"max_tokens": 100}}

    def test_run_resolves_nhx_uri_spec_via_async_sdk(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        """async_sdk path: nhx_uri_spec is rewritten when sdk=None but async_sdk is provided."""
        ctx = _make_ctx(tmp_path)
        target = _make_target(
            options={
                "nim": {
                    "max_tokens": 32,
                    "nhx_uri_spec": {
                        "inference_gateway": {"workspace": "default", "provider": "nvidia-inference-api"},
                    },
                }
            }
        )
        spec = _make_spec_dict(target=target)

        async_sdk = MagicMock()
        async_sdk.models_client = MagicMock()
        async_sdk.models_client.get_provider = AsyncMock(return_value=MagicMock(data=MagicMock()))
        async_sdk.models_client.get_provider_route_openai_url.return_value = "https://igw-async.example/v1"

        with patch("nemo_garak.jobs.audit.run_subprocess") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            AuditJob().run(spec, ctx=ctx, sdk=None, async_sdk=async_sdk)

        opts_path = ctx.storage.persistent / "target_options.json"
        assert opts_path.exists()
        on_disk = json.loads(opts_path.read_text())
        assert on_disk == {"nim": {"max_tokens": 32, "uri": "https://igw-async.example/v1"}}
        assert "nhx_uri_spec" in target.options["nim"]  # original spec untouched


# ---------------------------------------------------------------------------
# AuditInputSpec — schema-level validation (inline / ref / mixed)
# ---------------------------------------------------------------------------


class TestAuditInputSpec:
    def test_inline_inline(self) -> None:
        validated = AuditInputSpec.model_validate(
            {
                "config": _make_config().model_dump(mode="json"),
                "target": _make_target().model_dump(mode="json"),
            }
        )
        assert isinstance(validated.config, AuditConfig)
        assert isinstance(validated.target, AuditTargetEntity)

    def test_ref_ref(self) -> None:
        validated = AuditInputSpec.model_validate({"config": "my-cfg", "target": "my-tgt"})
        assert validated.config == "my-cfg"
        assert validated.target == "my-tgt"

    def test_inline_ref_mixed(self) -> None:
        validated = AuditInputSpec.model_validate(
            {"config": _make_config().model_dump(mode="json"), "target": "prod/my-tgt"}
        )
        assert isinstance(validated.config, AuditConfig)
        assert validated.target == "prod/my-tgt"

    def test_ref_inline_mixed(self) -> None:
        validated = AuditInputSpec.model_validate(
            {"config": "my-cfg", "target": _make_target().model_dump(mode="json")}
        )
        assert validated.config == "my-cfg"
        assert isinstance(validated.target, AuditTargetEntity)

    def test_workspace_qualified_string_preserved(self) -> None:
        validated = AuditInputSpec.model_validate({"config": "prod/my-cfg", "target": "qa/my-tgt"})
        assert validated.config == "prod/my-cfg"
        assert validated.target == "qa/my-tgt"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError):
            AuditInputSpec.model_validate({"config": "", "target": "my-tgt"})

    def test_rejects_whitespace_only_string(self) -> None:
        # strip_whitespace + min_length=1 means "   " collapses to "" → rejected.
        with pytest.raises(ValueError):
            AuditInputSpec.model_validate({"config": "   ", "target": "my-tgt"})

    def test_rejects_extra_field(self) -> None:
        with pytest.raises(ValueError):
            AuditInputSpec.model_validate(
                {"config": "my-cfg", "target": "my-tgt", "extra": "no"},
            )

    def test_requires_config_and_target(self) -> None:
        with pytest.raises(ValueError):
            AuditInputSpec.model_validate({"target": "my-tgt"})
        with pytest.raises(ValueError):
            AuditInputSpec.model_validate({"config": "my-cfg"})

    def test_default_task_options(self) -> None:
        spec = AuditInputSpec.model_validate({"config": "my-cfg", "target": "my-tgt"})
        assert spec.max_probe_retries == 0
        assert spec.fail_job_on_retries_exhausted is True

    def test_custom_task_options(self) -> None:
        spec = AuditInputSpec.model_validate(
            {"config": "my-cfg", "target": "my-tgt", "max_probe_retries": 3, "fail_job_on_retries_exhausted": False}
        )
        assert spec.max_probe_retries == 3
        assert spec.fail_job_on_retries_exhausted is False


# ---------------------------------------------------------------------------
# AuditJob.to_spec — name resolution via entity_client
# ---------------------------------------------------------------------------


def _run_to_spec(input_spec: AuditInputSpec, *, workspace: str, entity_client) -> AuditSpec:
    """Sync wrapper around the async classmethod for use in synchronous tests."""
    return cast(
        AuditSpec,
        asyncio.run(
            AuditJob.to_spec(
                input_spec,
                workspace=workspace,
                entity_client=entity_client,
                async_sdk=None,
                is_local=True,
            )
        ),
    )


class TestToSpec:
    def test_inline_inline_is_identity_and_no_lookups(self) -> None:
        cfg = _make_config()
        tgt = _make_target()
        client = AsyncMock()

        out = _run_to_spec(AuditInputSpec(config=cfg, target=tgt), workspace="default", entity_client=client)
        assert out.config is cfg
        assert out.target is tgt
        client.get.assert_not_awaited()

    def test_ref_ref_resolves_both_with_default_workspace(self) -> None:
        resolved_cfg = _make_config(name="resolved-cfg")
        resolved_tgt = _make_target(name="resolved-tgt")

        client = AsyncMock()

        async def fake_get(entity_class, **kwargs):
            return resolved_cfg if entity_class is AuditConfig else resolved_tgt

        client.get = AsyncMock(side_effect=fake_get)

        out = _run_to_spec(
            AuditInputSpec(config="my-cfg", target="my-tgt"),
            workspace="default",
            entity_client=client,
        )

        assert out.config is resolved_cfg
        assert out.target is resolved_tgt
        # Both calls used the runtime-default workspace.
        calls = client.get.await_args_list
        assert len(calls) == 2
        assert {c.args[0] for c in calls} == {AuditConfig, AuditTargetEntity}
        for c in calls:
            assert c.kwargs["workspace"] == "default"
        assert {c.kwargs["name"] for c in calls} == {"my-cfg", "my-tgt"}

    def test_workspace_qualified_string_overrides_default(self) -> None:
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[_make_config(workspace="prod"), _make_target(workspace="qa")])

        _run_to_spec(
            AuditInputSpec(config="prod/cfg-a", target="qa/tgt-b"),
            workspace="default",
            entity_client=client,
        )

        calls = client.get.await_args_list
        ws_by_name = {c.kwargs["name"]: c.kwargs["workspace"] for c in calls}
        assert ws_by_name == {"cfg-a": "prod", "tgt-b": "qa"}

    def test_unqualified_uses_runtime_workspace(self) -> None:
        client = AsyncMock()
        client.get = AsyncMock(side_effect=[_make_config(workspace="dev"), _make_target(workspace="dev")])

        _run_to_spec(
            AuditInputSpec(config="cfg", target="tgt"),
            workspace="dev",
            entity_client=client,
        )

        calls = client.get.await_args_list
        for c in calls:
            assert c.kwargs["workspace"] == "dev"

    def test_mixed_inline_config_ref_target(self) -> None:
        inline_cfg = _make_config(name="inline-cfg")
        resolved_tgt = _make_target(name="resolved-tgt")
        client = AsyncMock()
        client.get = AsyncMock(return_value=resolved_tgt)

        out = _run_to_spec(
            AuditInputSpec(config=inline_cfg, target="my-tgt"),
            workspace="default",
            entity_client=client,
        )

        assert out.config is inline_cfg
        assert out.target is resolved_tgt
        # Only one entity-client call: target.
        client.get.assert_awaited_once()
        await_args = client.get.await_args
        assert await_args is not None
        assert await_args.args[0] is AuditTargetEntity

    def test_not_found_wraps_in_runtimeerror_with_kind_and_path(self) -> None:
        client = AsyncMock()
        client.get = AsyncMock(side_effect=NemoEntityNotFoundError("no such entity"))

        with pytest.raises(RuntimeError, match=r"audit config 'prod/missing-cfg' not found"):
            _run_to_spec(
                AuditInputSpec(config="prod/missing-cfg", target=_make_target()),
                workspace="default",
                entity_client=client,
            )

    def test_target_not_found_message_uses_target_kind(self) -> None:
        client = AsyncMock()
        client.get = AsyncMock(side_effect=NemoEntityNotFoundError("nope"))

        with pytest.raises(RuntimeError, match=r"audit target 'default/missing-tgt' not found"):
            _run_to_spec(
                AuditInputSpec(config=_make_config(), target="missing-tgt"),
                workspace="default",
                entity_client=client,
            )

    def test_falls_back_to_async_sdk_when_client_is_none(self) -> None:
        """Local-mode scheduler adapts its async SDK to the typed entities client."""
        resolved_cfg = _make_config(name="from-sdk")
        resolved_tgt = _make_target(name="from-sdk")
        async_sdk = MagicMock(entities=MagicMock())
        typed_client = MagicMock()

        with (
            patch("nemo_garak.jobs.audit.client_from_platform", return_value=typed_client) as mock_adapter,
            patch("nemo_garak.jobs.audit.NemoEntitiesClient") as mock_cls,
        ):
            client = AsyncMock()
            client.get = AsyncMock(side_effect=[resolved_cfg, resolved_tgt])
            mock_cls.return_value = client

            out = cast(
                AuditSpec,
                asyncio.run(
                    AuditJob.to_spec(
                        AuditInputSpec(config="cfg", target="tgt"),
                        workspace="default",
                        entity_client=None,
                        async_sdk=async_sdk,
                        is_local=True,
                    )
                ),
            )

        mock_adapter.assert_called_once_with(async_sdk, AsyncEntitiesClient)
        mock_cls.assert_called_once_with(typed_client)
        assert out.config is resolved_cfg
        assert out.target is resolved_tgt

    def test_raises_when_neither_client_nor_sdk_present_and_refs_used(self) -> None:
        """No client + name refs = unresolvable; clear error."""
        with pytest.raises(RuntimeError, match=r"no platform client was injected"):
            asyncio.run(
                AuditJob.to_spec(
                    AuditInputSpec(config="cfg", target="tgt"),
                    workspace="default",
                    entity_client=None,
                    async_sdk=None,
                    is_local=True,
                )
            )

    def test_no_client_required_when_both_inline(self) -> None:
        """Inline-only specs don't need any client at all."""
        out = asyncio.run(
            AuditJob.to_spec(
                AuditInputSpec(config=_make_config(), target=_make_target()),
                workspace="default",
                entity_client=None,
                async_sdk=None,
                is_local=True,
            )
        )
        assert isinstance(out, AuditSpec)

    def test_task_options_pass_through_to_spec(self) -> None:
        """max_probe_retries and fail_job_on_retries_exhausted are forwarded to AuditSpec."""
        out = cast(
            AuditSpec,
            asyncio.run(
                AuditJob.to_spec(
                    AuditInputSpec(
                        config=_make_config(),
                        target=_make_target(),
                        max_probe_retries=3,
                        fail_job_on_retries_exhausted=False,
                    ),
                    workspace="default",
                    entity_client=None,
                    async_sdk=None,
                    is_local=True,
                )
            ),
        )
        assert out.max_probe_retries == 3
        assert out.fail_job_on_retries_exhausted is False


# ---------------------------------------------------------------------------
# Inference Gateway credentials
# ---------------------------------------------------------------------------

_GATEWAY = {"nhx_uri_spec": {"inference_gateway": {"workspace": "default", "provider": "build"}}}
_TOKEN = "helix-access-token-value"
_KEY_VARS = ("NIM_API_KEY", "OPENAI_API_KEY", "REST_API_KEY", "OPENAICOMPATIBLE_API_KEY")


class _RotatingToken:
    """Token provider that hands out a new token on every call, like a refreshing one."""

    def __init__(self) -> None:
        self.calls = 0

    def get_access_token(self) -> str:
        self.calls += 1
        return f"token-{self.calls}"


def _token_sdk(auth: object = _TOKEN) -> NemoClient:
    """A real ``NemoClient`` (so token resolution is real) with the models lookup stubbed."""
    sdk = NemoClient(base_url="https://helix.example", workspace="default", auth=auth)  # ty: ignore[invalid-argument-type]
    sdk.models_client = _mock_sdk("https://helix.example/gateway/v1").models_client  # ty: ignore[unresolved-attribute]
    return sdk


def _succeeding_garak(envs: list[dict[str, str]]):
    """A ``run_subprocess`` stand-in that records each env and marks the probe complete."""

    def fake_run(cmd, **kwargs):
        envs.append(kwargs["env"])
        marker = Path(kwargs["env"]["XDG_DATA_HOME"]) / "garak" / "garak_runs" / "run1.report.html"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return subprocess.CompletedProcess(args=[], returncode=0)

    return fake_run


class TestGatewayToken:
    def test_detects_the_gateway_sentinel_at_any_depth(self) -> None:
        assert _uses_inference_gateway({"nim": _GATEWAY})
        assert _uses_inference_gateway({"openai": {"OpenAICompatible": _GATEWAY}})
        assert not _uses_inference_gateway({"nim": {"uri": "https://integrate.api.nvidia.com/v1"}})
        assert not _uses_inference_gateway({})

    def test_token_comes_from_either_client(self) -> None:
        async_sdk = AsyncNemoClient(base_url="https://helix.example", workspace="default", auth=StaticToken(_TOKEN))
        assert _gateway_token(_token_sdk(), None) == _TOKEN
        assert _gateway_token(None, async_sdk) == _TOKEN
        assert _gateway_token(None, None) is None

    def test_no_token_when_the_client_authenticates_with_headers(self) -> None:
        # The task client on a cluster without workload identity sends trusted principal headers.
        headers = {"X-NHX-Principal-Id": "service:garak"}
        sdk = NemoClient(base_url="https://helix.example", workspace="default", default_headers=headers)
        assert _gateway_token(sdk, None) is None

    def test_token_is_resolved_again_on_each_call(self) -> None:
        sdk = _token_sdk(auth=_RotatingToken())
        assert [_gateway_token(sdk, None), _gateway_token(sdk, None)] == ["token-1", "token-2"]

    def test_resolution_failure_returns_none_without_leaking(self, caplog) -> None:
        class _Broken:
            def get_access_token(self) -> str:
                raise RuntimeError(f"refresh failed for {_TOKEN}")

        with caplog.at_level("WARNING"):
            assert _gateway_token(_token_sdk(auth=_Broken()), None) is None
        assert "Could not get an access token" in caplog.text
        assert _TOKEN not in caplog.text

    def test_refuses_a_cleartext_remote_endpoint(self) -> None:
        # The client will not send its token over plain HTTP to a remote host, so neither do we.
        sdk = NemoClient(base_url="http://helix.example", workspace="default", auth=_TOKEN)
        assert _gateway_token(sdk, None) is None


class TestAuditJobGatewayCredentials:
    @pytest.fixture(autouse=True)
    def _no_keys_in_env(self, monkeypatch, fake_garak_python: Path) -> None:
        for key in _KEY_VARS:
            monkeypatch.delenv(key, raising=False)

    def _run(self, tmp_path: Path, *, sdk: object = None, **target: Any) -> tuple[dict, list[dict[str, str]]]:
        envs: list[dict[str, str]] = []
        spec = _make_spec_dict(target=_make_target(**{"type": "nim", "model": "m", **target}))
        with (
            patch("nemo_garak.jobs.audit.run_subprocess", side_effect=_succeeding_garak(envs)),
            patch("nemo_garak.jobs.audit._aggregate_reports", return_value=False),
        ):
            result = AuditJob().run(spec, ctx=_make_ctx(tmp_path), sdk=sdk or _token_sdk())  # ty: ignore[invalid-argument-type]
        return result, envs

    @pytest.mark.parametrize(
        ("target_type", "options_key", "key_var"),
        [
            ("nim.NVOpenAIChat", "nim", "NIM_API_KEY"),
            ("openai.OpenAICompatible", "openai", "OPENAICOMPATIBLE_API_KEY"),
            ("openai", "openai", "OPENAI_API_KEY"),
            ("rest.RestGenerator", "rest", "REST_API_KEY"),
        ],
    )
    def test_gateway_target_gets_the_client_token_as_its_generator_key(
        self, tmp_path: Path, fake_parse_plugin_spec, target_type: str, options_key: str, key_var: str
    ) -> None:
        _, envs = self._run(tmp_path, type=target_type, options={options_key: _GATEWAY})
        # Only the target's generator gets the platform token.
        assert {key: envs[0][key] for key in _KEY_VARS} == {**dict.fromkeys(_KEY_VARS, "NOT_SET"), key_var: _TOKEN}

    def test_gateway_token_replaces_a_key_already_in_the_environment(
        self, tmp_path: Path, fake_parse_plugin_spec, monkeypatch
    ) -> None:
        monkeypatch.setenv("NIM_API_KEY", "nvapi-for-build-nvidia-com")
        _, envs = self._run(tmp_path, options={"nim": _GATEWAY})
        assert envs[0]["NIM_API_KEY"] == _TOKEN

    @pytest.mark.parametrize(
        ("target", "sdk"),
        [
            # A target that does not go through the gateway must never be handed a platform token.
            ({"options": {"nim": {"uri": "https://integrate.api.nvidia.com/v1"}}}, None),
            # A generator whose key variable is not known.
            ({"type": "test.Blank", "options": {"test": _GATEWAY}}, None),
            # A client with no bearer token.
            ({"options": {"nim": _GATEWAY}}, _mock_sdk()),
        ],
    )
    def test_placeholder_is_kept_when_there_is_no_token_to_give(
        self, tmp_path: Path, fake_parse_plugin_spec, target: dict, sdk: object
    ) -> None:
        _, envs = self._run(tmp_path, sdk=sdk, **target)
        assert {envs[0][key] for key in _KEY_VARS} == {"NOT_SET"}

    def test_token_is_refreshed_for_every_probe(self, tmp_path: Path) -> None:
        with patch("nemo_garak.jobs.audit.garakapi.parse_plugin_spec") as parse:
            parse.return_value = (["probes.dan.Dan_11_0", "probes.encoding.InjectAscii85"], [])
            _, envs = self._run(tmp_path, sdk=_token_sdk(auth=_RotatingToken()), options={"nim": _GATEWAY})
        assert len({env["NIM_API_KEY"] for env in envs}) == 2

    def test_token_is_refreshed_for_a_retry(self, tmp_path: Path, fake_parse_plugin_spec) -> None:
        keys: list[str] = []

        def failing_run(cmd, **kwargs):
            keys.append(kwargs["env"]["NIM_API_KEY"])
            return subprocess.CompletedProcess(args=[], returncode=1)

        target = _make_target(type="nim", model="m", options={"nim": _GATEWAY})
        with patch("nemo_garak.jobs.audit.run_subprocess", side_effect=failing_run):
            AuditJob().run(
                _make_spec_dict(target=target, max_probe_retries=1),
                ctx=_make_ctx(tmp_path),
                sdk=_token_sdk(auth=_RotatingToken()),
            )
        assert len(set(keys)) == 2

    def test_token_is_never_logged_or_written_to_disk(self, tmp_path: Path, fake_parse_plugin_spec, caplog) -> None:
        with caplog.at_level("DEBUG"):
            result, _ = self._run(tmp_path, options={"nim": _GATEWAY})
        assert _TOKEN not in caplog.text
        assert _TOKEN not in json.dumps(result)
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert _TOKEN not in path.read_text(errors="ignore"), path


# ---------------------------------------------------------------------------
# Publication to Intake
# ---------------------------------------------------------------------------

_PUBLICATION = {"intake": {"evaluation_id": "qwen-garak"}}


class TestAuditJobPublication:
    def _run(self, tmp_path: Path, *, publish: dict[str, Any] | None = None, sdk=None, **spec_overrides):
        """Run a one-probe audit that completes, with the publisher patched."""
        ctx = _make_ctx(tmp_path)
        _plant_reports(ctx.storage.persistent, "run1", (".report.jsonl", ".report.html"))
        outcome = PublicationOutcome(status=HelixJobStatus.COMPLETED, evaluation_id="qwen-garak", session_count=2)
        with (
            patch("nemo_garak.jobs.audit.run_subprocess", side_effect=_succeeding_garak([])),
            patch("nemo_garak.jobs.audit._aggregate_reports", return_value=True),
            patch("nemo_garak.jobs.audit.publish_audit_report", **(publish or {"return_value": outcome})) as pub,
        ):
            result = AuditJob().run(_make_spec_dict(**spec_overrides), ctx=ctx, sdk=sdk)
        return result, pub, ctx

    def test_no_publication_block_publishes_nothing(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        result, publish, _ = self._run(tmp_path)
        publish.assert_not_called()
        assert "publication" not in result

    def test_publishes_the_aggregated_report_after_artifacts_are_saved(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        sdk = MagicMock()
        config = _make_config(run=AuditRunData(generations=1, eval_threshold=0.7))
        target = _make_target(name="qwen-target", model="qwen/qwen3.8-27b")

        result, publish, ctx = self._run(tmp_path, sdk=sdk, publication=_PUBLICATION, config=config, target=target)

        assert result["status"] == "completed"
        assert set(result["results"]) == {"report-jsonl", "report-html"}
        assert result["publication"]["status"] == "completed"
        assert result["publication"]["session_count"] == 2
        call = publish.call_args
        assert call.args[0] == ctx.storage.persistent / "garak" / "garak_runs" / "run1.report.jsonl"
        assert call.kwargs == {
            "spec": IntakePublicationSpec(evaluation_id="qwen-garak"),
            "workspace": "default",
            "target_name": "qwen-target",
            "model_name": "qwen/qwen3.8-27b",
            "eval_threshold": 0.7,
            "client": sdk,
            "async_client": None,
        }

    def test_required_publication_failure_fails_the_job_but_keeps_the_reports(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        failed = PublicationOutcome(
            status=HelixJobStatus.ERROR, evaluation_id="qwen-garak", session_count=1, error="1 of 2 session(s) failed"
        )
        result, _, _ = self._run(
            tmp_path, publish={"side_effect": PublicationFailedError(failed)}, publication=_PUBLICATION
        )

        assert result["status"] == "failed"
        assert result["error"] == "Publication to Intake failed: 1 of 2 session(s) failed"
        assert set(result["results"]) == {"report-jsonl", "report-html"}
        assert result["publication"]["session_count"] == 1

    def test_optional_publication_failure_keeps_the_job_successful(
        self, tmp_path: Path, fake_garak_python: Path, fake_parse_plugin_spec
    ) -> None:
        failed = PublicationOutcome(status=HelixJobStatus.ERROR, evaluation_id="qwen-garak", error="Intake is down")
        result, _, _ = self._run(
            tmp_path,
            publish={"return_value": failed},
            publication={"intake": {"evaluation_id": "qwen-garak", "required": False}},
        )

        assert result["status"] == "completed"
        assert result["publication"]["status"] == "error"
        assert result["publication"]["error"] == "Intake is down"


class TestPublicationOnSpecs:
    def test_input_spec_accepts_publication_and_defaults_to_none(self) -> None:
        spec = AuditInputSpec.model_validate({"config": "cfg", "target": "tgt", "publication": _PUBLICATION})
        assert spec.publication == PublicationSpec(intake=IntakePublicationSpec(evaluation_id="qwen-garak"))
        assert AuditInputSpec(config="cfg", target="tgt").publication is None
        assert AuditSpec.model_validate(_make_spec_dict()).publication is None

    def test_input_spec_rejects_unknown_publication_fields(self) -> None:
        with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
            AuditInputSpec.model_validate(
                {"config": "cfg", "target": "tgt", "publication": {"intake": {"evaluation": "x"}}}
            )

    def test_to_spec_carries_publication_through(self) -> None:
        publication = PublicationSpec(intake=IntakePublicationSpec(evaluation_id="qwen-garak", required=False))
        input_spec = AuditInputSpec(config=_make_config(), target=_make_target(), publication=publication)
        resolved = _run_to_spec(input_spec, workspace="default", entity_client=None)
        # The compiled step config is what the task container validates.
        assert AuditSpec.model_validate(resolved.model_dump(mode="json")).publication == publication
