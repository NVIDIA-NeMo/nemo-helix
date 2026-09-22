# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo agents usage`` CLI integration via :class:`CliRunner`."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from nemo_agents_plugin.cli import AgentsCLI
from nemo_platform_plugin.client.client import NemoClient
from typer.testing import CliRunner

runner = CliRunner()


class _FakeSDKUser:
    def __init__(self, token: str | None) -> None:
        self._token = token

    def get_client_config(self) -> dict[str, object]:
        if self._token is None:
            return {}
        return {"default_headers": {"Authorization": f"Bearer {self._token}"}}


class _FakeSDKContext:
    def __init__(self, base_url: str, token: str | None) -> None:
        self.user = _FakeSDKUser(token)
        self.cluster = type("_Cluster", (), {"base_url": base_url})()


class _FakeCLIContext:
    """Minimal stand-in for ``CLIContext`` (typer.Context.obj)."""

    def __init__(self, base_url: str = "http://config-host:9999", token: str | None = "cfg-token") -> None:
        self._sdk = _FakeSDKContext(base_url, token)

    def get_sdk_context(self) -> _FakeSDKContext:
        return self._sdk

    def get_base_url(self, default: str | None = None) -> str | None:
        return str(self._sdk.cluster.base_url)


@pytest.fixture
def app():
    """Build the actual ``nemo agents`` Typer app (with ``usage`` registered)."""
    return AgentsCLI().get_cli()


def test_usage_show_with_run_dir(app, tmp_run_dir: Path) -> None:
    """``nemo agents usage show <run-dir>/`` emits a single-task report."""
    result = runner.invoke(app, ["usage", "show", str(tmp_run_dir)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "v0"
    assert payload["task"]["task"] == "workspace-basic-mcp"
    assert payload["task"]["total_tokens"] == 2000
    # No --total-params: compute_units is null but the field is rendered.
    assert payload["task"]["compute_units"] is None


def test_usage_show_with_total_params_populates_compute_units(app, tmp_run_dir: Path) -> None:
    """``--total-params`` populates compute_units = tokens × total_params."""
    result = runner.invoke(
        app,
        ["usage", "show", str(tmp_run_dir), "--total-params", "8.0"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # 2000 tokens × 8B total = 16000 compute units
    assert payload["task"]["compute_units"] == 16000


def test_usage_show_with_natjobs_dir(app, tmp_natjobs_dir: Path) -> None:
    """``nemo agents usage show <nat-jobs/>`` emits a batch report.

    With null-token runs in the batch, totals are ``None`` rather than a
    misleading partial sum.
    """
    result = runner.invoke(app, ["usage", "show", str(tmp_natjobs_dir)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["runs"]) == 4
    assert payload["null_token_runs"] == 3
    assert payload["total_tokens_total"] is None
    assert payload["compute_units_total"] is None


def test_usage_show_with_invalid_path_exits_nonzero(app, tmp_path: Path) -> None:
    """A non-existent ref produces a clean error message and exit code 1."""
    result = runner.invoke(app, ["usage", "show", str(tmp_path / "no-such-dir")])

    assert result.exit_code == 1
    assert "error" in result.output.lower()


def test_usage_with_no_args_prints_help(app) -> None:
    """Running ``usage`` with no arguments emits help and exits with non-zero (typer convention)."""
    result = runner.invoke(app, ["usage"])

    # typer exits 2 when no subcommand provided and no_args_is_help=True
    assert result.exit_code in (0, 2)
    assert "show" in result.output.lower()


def test_usage_show_with_fileset_ref_uses_sdk(app, tmp_natjobs_dir: Path, fake_sdk_factory) -> None:
    """A bare-name ref classifies as FilesetRef and routes through the SDK."""
    fake = fake_sdk_factory(tmp_natjobs_dir)

    with patch("nemo_agents_plugin.usage.cli._build_sdk", return_value=fake):
        result = runner.invoke(
            app,
            ["usage", "show", "my-fileset", "--workspace", "ws-test"],
        )

    assert result.exit_code == 0, result.output
    # The resolved-target banner goes to stderr; stdout stays clean JSON.
    payload = json.loads(result.stdout)
    assert len(payload["runs"]) == 4
    # Typed contract: one list_files call, then a download_file per file.
    assert fake.files.list_calls == [{"name": "my-fileset", "workspace": "ws-test"}]
    assert len(fake.files.calls) == 4
    assert all(c["name"] == "my-fileset" and c["workspace"] == "ws-test" for c in fake.files.calls)
    assert all(c["path"].endswith("/result.json") for c in fake.files.calls)


def test_usage_show_batch_partial_null_tokens_nulls_compute_units_total(
    app, tmp_path: Path, fixtures_dir: Path
) -> None:
    """A batch with any null-token run nulls compute_units_total even when scored.

    Without this gate, a run with prompt_tokens=null but total_tokens populated
    would null token totals (parser rule) while leaving compute_units_total
    populated (would-be naive scoring) — directly contradicting the documented
    "totals null if any run has missing usage" invariant.
    """
    natjobs = tmp_path / "nat-jobs"
    natjobs.mkdir()
    full = natjobs / "20260429T220000Z-full"
    full.mkdir()
    (full / "result.json").write_text((fixtures_dir / "result-ok-with-tokens.json").read_text())
    partial = natjobs / "20260429T230000Z-partial"
    partial.mkdir()
    (partial / "result.json").write_text(
        '{"task": "partial", "timestamp": "20260429T230000Z", '
        '"metrics": {"prompt_tokens": null, "completion_tokens": 500, "total_tokens": 2500}}'
    )

    result = runner.invoke(app, ["usage", "show", str(natjobs), "--total-params", "8.0"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["null_token_runs"] == 1
    assert payload["total_tokens_total"] is None
    assert payload["compute_units_total"] is None


def test_usage_show_rejects_non_positive_total_params(app, tmp_run_dir: Path) -> None:
    """``--total-params 0`` and negative values are rejected at the option layer."""
    for bad in ("0", "0.0", "-8"):
        result = runner.invoke(app, ["usage", "show", str(tmp_run_dir), "--total-params", bad])
        assert result.exit_code != 0
        assert "must be > 0" in result.output


def test_usage_show_rejects_non_finite_total_params(app, tmp_run_dir: Path) -> None:
    """NaN/inf bypass ``<= 0`` and would crash inside int(round(...)); reject at the option layer."""
    for bad in ("nan", "inf", "-inf"):
        result = runner.invoke(app, ["usage", "show", str(tmp_run_dir), "--total-params", bad])
        assert result.exit_code != 0, (bad, result.output)
        # Rich may wrap the panel between "finite" and "positive"; assert on
        # the unambiguous root-word and the offending value.
        assert "finite" in result.output, (bad, result.output)
        assert bad in result.output, (bad, result.output)


def test_usage_show_sdk_download_failure_exits_cleanly(app, monkeypatch) -> None:
    """An SDK download error exits 1 with a clean message — no Python traceback."""

    class BoomFiles:
        def list_files(self, *, workspace=None, name, query_params=None):
            listing = [type("_Entry", (), {"path": "result.json"})()]

            class Resp:
                def data(self):
                    return type("_Listing", (), {"data": listing})()

            return Resp()

        def download_file(self, *, workspace=None, name, path):
            raise RuntimeError("simulated SDK failure: fileset not found")

    sdk = type("_BoomSDK", (), {"build_files_client": lambda self: BoomFiles()})()
    monkeypatch.setattr(
        "nemo_agents_plugin.usage.sources.fileset.client_from_platform",
        lambda _platform, _client_cls: BoomFiles(),
    )
    with patch("nemo_agents_plugin.usage.cli._build_sdk", return_value=sdk):
        result = runner.invoke(app, ["usage", "show", "missing-fileset"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "failed to download fileset" in result.output


def test_usage_show_unreadable_result_json_exits_cleanly(app, tmp_path: Path) -> None:
    """A chmod-000 ``result.json`` exits 1 with a clean message — no Python traceback.

    OS-level read failures must surface as ``UsageParseError`` so the CLI's
    catch tuple turns them into a clean error line, not a stack trace.
    """
    bad = tmp_path / "result.json"
    bad.write_text("{}")
    bad.chmod(0o000)
    try:
        result = runner.invoke(app, ["usage", "show", str(bad)])
    finally:
        bad.chmod(0o644)

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "cannot read file" in result.output


def test_usage_show_batch_all_runs_tokened_sums_compute_units(app, tmp_path: Path, fixtures_dir: Path) -> None:
    """A fully-tokened batch with --total-params sums compute_units_total
    over every run rather than nulling it."""
    natjobs = tmp_path / "nat-jobs"
    natjobs.mkdir()
    src = (fixtures_dir / "result-ok-with-tokens.json").read_text()
    for i, ts in enumerate(("20260429T220000Z", "20260429T230000Z", "20260429T240000Z")):
        run = natjobs / f"{ts}-task-{i}"
        run.mkdir()
        (run / "result.json").write_text(src.replace("20260429T220000Z", ts))

    result = runner.invoke(app, ["usage", "show", str(natjobs), "--total-params", "8.0"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # Each fixture is 2000 total_tokens × 8B = 16000; 3 runs → 48000.
    assert payload["total_tokens_total"] == 6000
    assert payload["compute_units_total"] == 48000
    assert payload["null_token_runs"] == 0


def test_usage_show_rejects_empty_workspace(app, tmp_path: Path, fake_sdk_factory) -> None:
    """``--workspace ''`` is rejected at the FilesetRef layer (symmetric with empty name)."""
    fake = fake_sdk_factory(tmp_path)

    with patch("nemo_agents_plugin.usage.cli._build_sdk", return_value=fake):
        result = runner.invoke(app, ["usage", "show", "my-fileset", "--workspace", ""])

    assert result.exit_code == 1
    assert "workspace must be non-empty" in result.output


def test_usage_show_resolves_bare_local_dir_when_it_exists(
    app, tmp_path: Path, monkeypatch, fixtures_dir: Path
) -> None:
    """A bare-name ref that resolves locally is treated as a local path.

    classify_output_target's default-to-fileset rule is correct for output
    targets but wrong for input — users naturally type bare local names
    (``nat-jobs/``, ``result.json``).  When the path exists locally, take
    it; only fall through to fileset for misses.
    """
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "nat-jobs-local"
    run.mkdir()
    (run / "result.json").write_text((fixtures_dir / "result-ok-with-tokens.json").read_text())

    result = runner.invoke(app, ["usage", "show", "nat-jobs-local"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["task"]["task"] == "workspace-basic-mcp"


def test_usage_show_path_shaped_missing_does_not_try_fileset(app, tmp_path: Path) -> None:
    """A path-shaped ref (``./missing``) errors locally instead of attempting fileset."""
    result = runner.invoke(app, ["usage", "show", str(tmp_path / "no-such-dir")])

    assert result.exit_code == 1
    assert "local path does not exist" in result.output.lower()


def test_usage_show_fileset_rewrites_source_dirs(app, tmp_natjobs_dir: Path, fake_sdk_factory) -> None:
    """Fileset-resolved reports replace tempdir source_dirs with synthetic ``<ref>/<rel>``."""
    fake = fake_sdk_factory(tmp_natjobs_dir)

    with patch("nemo_agents_plugin.usage.cli._build_sdk", return_value=fake):
        result = runner.invoke(
            app,
            ["usage", "show", "my-fileset"],
        )

    assert result.exit_code == 0, result.output
    # The resolved-target banner goes to stderr; stdout stays clean JSON.
    payload = json.loads(result.stdout)
    # source_dir should be the synthetic <ref>/<rel> form, not the dead tempdir
    for run in payload["runs"]:
        assert run["source_dir"].startswith("my-fileset/")
        # And shouldn't reference any tempdir prefix
        assert "/var/folders/" not in run["source_dir"]
        assert "/.usage-" not in run["source_dir"]


def test_usage_show_fileset_single_run_rel_dot(app, tmp_path: Path, fake_sdk_factory, fixtures_dir: Path) -> None:
    """Fileset with result.json directly at root → source_dir is the bare ref."""
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "result.json").write_text((fixtures_dir / "result-ok-with-tokens.json").read_text())
    fake = fake_sdk_factory(staged)

    with patch("nemo_agents_plugin.usage.cli._build_sdk", return_value=fake):
        result = runner.invoke(app, ["usage", "show", "single-fileset"])

    assert result.exit_code == 0, result.output
    # The resolved-target banner goes to stderr; stdout stays clean JSON.
    payload = json.loads(result.stdout)
    # rel == "." branch: no slash + rel suffix; just the bare ref.
    assert payload["task"]["source_dir"] == "single-fileset"


def test_fileset_path_stages_nested_paths_preserving_relative(app, tmp_path: Path, fake_sdk_factory) -> None:
    """fileset_path stages fileset files under their relative (nested) paths.

    Each file is downloaded to ``tmp/<relative-path>`` rather than flattened to
    the staging root, so a fileset with subdirectories round-trips intact.
    """
    staged = tmp_path / "staged-fileset"
    nested = staged / "configs" / "v1"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text('{"a": 1}')
    (staged / "top.txt").write_text("top")
    fake = fake_sdk_factory(staged)

    from nemo_agents_plugin.usage.sources.fileset import FilesetRef, fileset_path

    with fileset_path(FilesetRef("nested-fs"), sdk=fake, workspace="default") as staged_dir:
        assert (staged_dir / "top.txt").read_text() == "top"
        assert (staged_dir / "configs" / "v1" / "config.json").read_text() == '{"a": 1}'

    # The fake recorded per-file downloads under their relative paths.
    assert {c["path"] for c in fake.files.calls} == {"top.txt", "configs/v1/config.json"}


def test_usage_show_rejects_multi_segment_fileset_ref(app, tmp_path: Path, fake_sdk_factory) -> None:
    """A multi-segment fileset ref (``ws/sub/path``) errors cleanly, not as a tempdir crash."""
    fake = fake_sdk_factory(tmp_path)

    with patch("nemo_agents_plugin.usage.cli._build_sdk", return_value=fake):
        result = runner.invoke(app, ["usage", "show", "ws/sub/path"])

    assert result.exit_code == 1
    assert "must not contain '/'" in result.output


def test_usage_show_rejects_empty_or_dot_relative_fileset_name(app, tmp_path: Path, fake_sdk_factory) -> None:
    """Trailing slash, ``.``, and ``..`` produce names rejected at the FilesetRef layer."""
    fake = fake_sdk_factory(tmp_path)

    for ref in ("my-fileset/", "ws/.", "ws/.."):
        with patch("nemo_agents_plugin.usage.cli._build_sdk", return_value=fake):
            result = runner.invoke(app, ["usage", "show", ref])
        assert result.exit_code == 1, (ref, result.output)
        assert "must be a real fileset name" in result.output, (ref, result.output)


def test_usage_show_fileset_builds_sdk_with_context_base_url_and_auth(app, tmp_natjobs_dir: Path) -> None:
    """A fileset ref builds the platform client with the shared context's base URL + auth token.

    Pins P0 parity for ``usage show``: it must honor ``nemo config`` /
    ``NMP_BASE_URL`` and attach the ``Authorization`` bearer token, instead
    of defaulting to localhost with no auth.
    """
    captured: dict[str, object] = {}

    @contextmanager
    def fake_fileset_path(_ref, *, sdk, workspace):
        assert isinstance(sdk, NemoClient)
        captured["base_url"] = sdk.base_url
        captured["default_headers"] = sdk.default_headers
        yield tmp_natjobs_dir

    with patch("nemo_agents_plugin.usage.cli.fileset_path", fake_fileset_path):
        result = runner.invoke(
            app,
            ["usage", "show", "my-fileset"],
            obj=_FakeCLIContext(base_url="http://config-host:9999", token="tkn"),
        )

    assert result.exit_code == 0, result.output
    assert captured["base_url"] == "http://config-host:9999"
    assert captured["default_headers"] == {"Authorization": "Bearer tkn"}
    assert "Targeting http://config-host:9999" in (result.stderr or "")


def test_usage_show_with_workspace_qualified_fileset_ref(app, tmp_natjobs_dir: Path, fake_sdk_factory) -> None:
    """A ``ws/name`` ref overrides the default workspace."""
    fake = fake_sdk_factory(tmp_natjobs_dir)

    with patch("nemo_agents_plugin.usage.cli._build_sdk", return_value=fake):
        result = runner.invoke(
            app,
            ["usage", "show", "other-ws/eval-results"],
        )

    assert result.exit_code == 0, result.output
    assert fake.files.list_calls == [{"name": "eval-results", "workspace": "other-ws"}]
    assert all(c["name"] == "eval-results" and c["workspace"] == "other-ws" for c in fake.files.calls)
