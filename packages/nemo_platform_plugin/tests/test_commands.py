# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_platform_plugin.commands.

Phase 1 MR 1.2c replaces the single-verb ``nemo <plugin> <job>`` CLI with a
three-verb sub-group — ``run`` / ``submit`` / ``explain`` — and makes the
bare form print usage and exit non-zero. These tests pin:

- Each job registers a sub-group under ``<job-name>``.
- The bare sub-group (no verb) prints usage and exits with status 1.
- ``run`` parses ``--config`` / ``--config-file`` and executes the job
  in-process via :class:`NemoJobScheduler`.
- ``submit`` / ``explain`` delegate to the scheduler stubs and exit 2 with
  a helpful error message in phase 1 (until MR 1.3 / MR 1.4 wire them).
- Invalid JSON on ``--config`` / ``--config-file`` exits 1 cleanly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, ClassVar, cast

import httpx
import pytest
import typer
from nemo_platform_plugin.commands import add_function_commands, add_job_commands
from nemo_platform_plugin.discovery import discover, discover_manifests
from nemo_platform_plugin.function import NemoFunction
from nemo_platform_plugin.function_context import FunctionContext
from nemo_platform_plugin.functions.frames import Done, Heartbeat
from nemo_platform_plugin.job import NemoJob
from pydantic import BaseModel, ValidationInfo, model_validator
from typer.testing import CliRunner

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape codes from Rich/Typer output for robust matching."""
    return _ANSI_RE.sub("", text)


@pytest.fixture(autouse=True)
def clear_discovery_cache():
    discover.cache_clear()
    discover_manifests.cache_clear()
    yield
    discover.cache_clear()
    discover_manifests.cache_clear()


# ---------------------------------------------------------------------------
# Fixture jobs
# ---------------------------------------------------------------------------


class _GreetJob(NemoJob):
    name = "greet"
    description = "Return a greeting."

    def run(self, config: dict) -> dict:
        return {"message": f"Hello, {config.get('name', 'world')}!"}


class _FailJob(NemoJob):
    name = "fail"
    description = "Always raises."

    def run(self, config: dict) -> dict:
        raise RuntimeError("job exploded")


class _RunNamedJob(NemoJob):
    name = "run"
    description = "A job whose name collides with the local run verb."

    def run(self, config: dict) -> dict:
        return config


class _FlatGreetJob(_GreetJob):
    generate_legacy_verbs: ClassVar[bool] = False


runner = CliRunner()


def _typer_context_with_obj(obj: object | None) -> typer.Context:
    return cast(typer.Context, SimpleNamespace(obj=obj))


def _app_with_jobs(*job_classes: type[NemoJob]) -> typer.Typer:
    """Return a fresh Typer app with the given jobs injected.

    A no-op callback is added so Typer keeps the app as a multi-command group
    rather than collapsing it into a single-function app.
    """
    app = typer.Typer()

    @app.callback()
    def _noop() -> None:
        pass

    jobs = {f"plugin.{cls.name}": cls for cls in job_classes}
    add_job_commands(app, jobs)
    return app


# ---------------------------------------------------------------------------
# Sub-group registration
# ---------------------------------------------------------------------------


class TestSubgroupRegistration:
    def test_registers_subgroup_for_each_job(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["--help"])
        assert "greet" in result.output

    def test_jobs_rich_help_panel_label(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["--help"])
        assert "Jobs" in result.output

    def test_registers_multiple_jobs(self) -> None:
        app = _app_with_jobs(_GreetJob, _FailJob)
        result = runner.invoke(app, ["--help"])
        assert "greet" in result.output
        assert "fail" in result.output

    def test_no_jobs_leaves_app_unchanged(self) -> None:
        app = typer.Typer()

        @app.command()
        def existing() -> None:
            """Existing command."""

        add_job_commands(app, {})
        result = runner.invoke(app, ["--help"])
        assert "existing" in result.output

    def test_subgroup_lists_three_verbs(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "submit" in result.output
        assert "explain" in result.output

    def test_subgroup_help_includes_description(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "--help"])
        assert "Return a greeting." in result.output

    def test_job_verb_help_text_does_not_repeat_job_name(self) -> None:
        app = _app_with_jobs(_RunNamedJob)
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "Run locally, in-process." in result.output
        assert "Submit to a cluster." in result.output
        assert "Show input/output schemas." in result.output
        assert "Run run locally" not in result.output
        assert "schemas for run" not in result.output

    def test_non_legacy_job_registers_flat_submit_command(self) -> None:
        app = _app_with_jobs(_FlatGreetJob)
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "greet" in result.output
        assert "Return a greeting." in result.output

        help_result = runner.invoke(app, ["greet", "--help"])
        assert help_result.exit_code == 0
        output = _plain(help_result.output)
        assert "COMMAND" not in output
        assert "Return a greeting." in output
        assert "--base-url" in output
        assert "--profile" in output
        assert "explain" not in output


# ---------------------------------------------------------------------------
# Bare form — usage + non-zero exit
# ---------------------------------------------------------------------------


class TestBareFormBreaks:
    def test_bare_job_name_exits_non_zero(self) -> None:
        """`nemo <plugin> <job>` with no verb must exit non-zero."""
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet"])
        assert result.exit_code != 0

    def test_bare_job_name_prints_usage(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet"])
        # Usage should mention at least one of the verbs so the user knows
        # what to type next.
        assert "run" in result.output or "submit" in result.output


# ---------------------------------------------------------------------------
# run verb
# ---------------------------------------------------------------------------


class TestRunVerb:
    def test_runs_job_with_json_config(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "run", "--config", '{"name": "Claude"}'])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output == {"message": "Hello, Claude!"}

    def test_runs_job_with_default_config(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "run"])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output == {"message": "Hello, world!"}

    def test_runs_job_with_config_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text('{"name": "File"}')
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "run", "--config-file", str(config_file)])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output == {"message": "Hello, File!"}

    def test_config_file_takes_precedence_over_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text('{"name": "FromFile"}')
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(
            app,
            [
                "greet",
                "run",
                "--config",
                '{"name": "Ignored"}',
                "--config-file",
                str(config_file),
            ],
        )
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["message"] == "Hello, FromFile!"

    def test_invalid_json_exits_with_error(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "run", "--config", "not-json"])
        assert result.exit_code != 0

    def test_job_exception_propagates(self) -> None:
        app = _app_with_jobs(_FailJob)
        with pytest.raises(RuntimeError, match="job exploded"):
            runner.invoke(app, ["fail", "run", "--config", "{}"], catch_exceptions=False)


# ---------------------------------------------------------------------------
# submit verb — phase 1 MR 1.2c stubs
# ---------------------------------------------------------------------------


class TestSubmitVerb:
    @pytest.mark.parametrize(
        ("args", "env_base_url", "context_base_url", "expected_base_url"),
        [
            (
                ["--base-url", "http://from-flag:9999", "--cluster", "configured-cluster"],
                "http://from-env:1234",
                "http://from-context:7777",
                "http://from-flag:9999",
            ),
            (
                ["--cluster", "configured-cluster"],
                "http://from-env:1234",
                "http://from-context:7777",
                "http://from-cluster:8888",
            ),
            ([], "http://from-env:1234", "http://from-context:7777", "http://from-context:7777"),
            ([], "http://from-env:1234", None, "http://from-env:1234"),
            ([], None, None, "http://localhost:8080"),
        ],
    )
    def test_submit_host_resolution_precedence(
        self,
        monkeypatch,
        args: list[str],
        env_base_url: str | None,
        context_base_url: str | None,
        expected_base_url: str,
    ) -> None:
        captured: dict[str, object] = {}

        def _capture(_self, _job_cls, _spec, *, base_url=None, **_kwargs) -> dict:
            captured["base_url"] = base_url
            return {"id": "job-123"}

        class _State:
            def __init__(self, resolved_base_url: str | None) -> None:
                self._resolved_base_url = resolved_base_url

            def get_base_url(self, default: str | None = None) -> str | None:
                return self._resolved_base_url if self._resolved_base_url is not None else default

        class _FakeConfig:
            def get_config_file(self) -> SimpleNamespace:
                return SimpleNamespace(
                    clusters=[SimpleNamespace(name="configured-cluster", base_url="http://from-cluster:8888")]
                )

        if env_base_url is None:
            monkeypatch.delenv("NMP_BASE_URL", raising=False)
        else:
            monkeypatch.setenv("NMP_BASE_URL", env_base_url)
        monkeypatch.setattr("nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote", _capture)
        monkeypatch.setattr("nemo_platform_ext.config.config.Config.load", lambda: _FakeConfig())

        app = _app_with_jobs(_GreetJob)
        state = _State(context_base_url)
        result = runner.invoke(app, ["greet", "submit", *args], obj=state)

        assert result.exit_code == 0, result.output
        assert captured == {"base_url": expected_base_url}

    def test_submit_returns_exit_code_2_on_connect_error(self, monkeypatch) -> None:
        request = httpx.Request("POST", "http://test/apis/tests/v2/workspaces/default/jobs/greet")

        def _raise_connect(*_args, **_kwargs) -> dict:
            raise httpx.ConnectError("Connection refused", request=request)

        monkeypatch.setattr("nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote", _raise_connect)

        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "submit", "--base-url", "http://test"])

        assert result.exit_code == 2
        combined = (result.output or "") + (result.stderr or "")
        assert "Connection refused" in combined
        assert "Request: POST http://test/apis/tests/v2/workspaces/default/jobs/greet" in combined
        assert "Target: tests API route /apis/tests/v2/workspaces/default/jobs/greet" in combined
        assert "nemo config view" in combined
        assert "Traceback" not in combined

    def test_submit_accepts_profile_and_cluster_flags(self) -> None:
        """Flags must be declared so MR 1.3 only needs to fill in behavior."""
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "submit", "--help"])
        assert result.exit_code == 0
        output = _plain(result.output)
        assert "--profile" in output
        assert "--cluster" in output

    def test_submit_passes_cli_auth_headers(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _capture(_self, _job_cls, _spec, headers=None, **_kwargs) -> dict:
            captured["headers"] = headers
            return {"id": "job-123"}

        class _State:
            def get_sdk_context(self) -> SimpleNamespace:
                return SimpleNamespace(
                    user=SimpleNamespace(
                        get_client_config=lambda: {
                            "default_headers": {"Authorization": "Bearer test-token"},
                        }
                    )
                )

        monkeypatch.setattr("nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote", _capture)

        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(
            app,
            ["greet", "submit", "--base-url", "http://127.0.0.1:8080"],
            obj=_State(),
        )

        assert result.exit_code == 0, result.output
        assert captured["headers"] == {"Authorization": "Bearer test-token"}

    def test_submit_passes_telemetry_custom_fields_as_metadata(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _capture(_self, _job_cls, _spec, *, metadata=None, **_kwargs) -> dict:
            captured["metadata"] = metadata
            return {"id": "job-123"}

        class _State:
            def get_job_telemetry_custom_fields(self) -> dict[str, object]:
                return {"_nemo_telemetry": {"session_id": "session-123"}}

        monkeypatch.setattr("nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote", _capture)

        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(
            app,
            ["greet", "submit", "--base-url", "http://127.0.0.1:8080"],
            obj=_State(),
        )

        assert result.exit_code == 0, result.output
        assert captured["metadata"] == {"custom_fields": {"_nemo_telemetry": {"session_id": "session-123"}}}

    def test_non_legacy_job_name_submits_remotely(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _capture(_self, job_cls, spec, *, base_url=None, workspace=None, **_kwargs) -> dict:
            captured.update(
                {
                    "job_cls": job_cls,
                    "spec": spec,
                    "base_url": base_url,
                    "workspace": workspace,
                }
            )
            return {"id": "job-123"}

        monkeypatch.setattr("nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote", _capture)

        app = _app_with_jobs(_FlatGreetJob)
        result = runner.invoke(
            app,
            ["greet", "--spec", '{"name": "Ada"}', "--base-url", "http://platform", "--workspace", "team-alpha"],
        )

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"id": "job-123"}
        assert captured == {
            "job_cls": _FlatGreetJob,
            "spec": {"name": "Ada"},
            "base_url": "http://platform",
            "workspace": "team-alpha",
        }

        assert runner.invoke(app, ["greet", "run"]).exit_code != 0
        assert runner.invoke(app, ["greet", "submit"]).exit_code != 0


# ---------------------------------------------------------------------------
# explain verb — phase 1 MR 1.2c stubs
# ---------------------------------------------------------------------------


class TestExplainVerb:
    def test_explain_works_without_cluster(self) -> None:
        """explain reads locally — no base_url, no cluster needed."""
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "explain"])
        assert result.exit_code == 0
        bundle = json.loads(result.output)
        # endpoint is a template — {workspace} stays as a literal placeholder.
        assert "{workspace}" in bundle["endpoint"]
        assert bundle["endpoint"].endswith("/jobs/greet")
        assert bundle["profile"] is None
        assert bundle["profile_providers"] == []
        assert bundle["options"] == {}

    def test_explain_annotates_with_profile(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "explain", "--profile", "research"])
        assert result.exit_code == 0
        bundle = json.loads(result.output)
        assert bundle["profile"] == "research"

    def test_explain_accepts_profile_and_cluster_flags(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "explain", "--help"])
        assert result.exit_code == 0
        output = _plain(result.output)
        assert "--profile" in output
        assert "--cluster" in output


# ---------------------------------------------------------------------------
# --spec / --spec-file rename (MR 1.3b)
# ---------------------------------------------------------------------------


class TestSpecFlagRename:
    def test_run_accepts_spec_flag(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "run", "--spec", '{"name": "Claude"}'])
        assert result.exit_code == 0
        assert json.loads(result.output) == {"message": "Hello, Claude!"}

    def test_run_accepts_spec_file_yaml(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("name: FromYaml\n")
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "run", "--spec-file", str(spec_file)])
        assert result.exit_code == 0
        assert json.loads(result.output)["message"] == "Hello, FromYaml!"

    def test_run_missing_spec_file_exits_cleanly(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "missing.yaml"
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(
            app,
            ["greet", "run", "--spec-file", str(missing_file)],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        combined = (result.output or "") + (result.stderr or "")
        assert "invalid spec" in combined
        assert str(missing_file) in combined

    def test_run_directory_spec_file_exits_cleanly(self, tmp_path: Path) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(
            app,
            ["greet", "run", "--spec-file", str(tmp_path)],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        combined = (result.output or "") + (result.stderr or "")
        assert "invalid spec" in combined
        assert str(tmp_path) in combined

    def test_run_config_alias_still_works(self) -> None:
        """--config remains as a deprecated alias for --spec during the transition."""
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "run", "--config", '{"name": "Legacy"}'])
        assert result.exit_code == 0
        assert json.loads(result.output)["message"] == "Hello, Legacy!"


# ---------------------------------------------------------------------------
# submit options passthrough (MR 1.3b)
# ---------------------------------------------------------------------------


class TestSubmitOptionsPassthrough:
    def test_submit_accepts_dash_o_flag(self) -> None:
        """The --help for submit must list -o so users can discover it."""
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "submit", "--help"])
        assert result.exit_code == 0
        output = _plain(result.output)
        assert "-o" in output
        assert "--options-file" in output
        assert "--spec-file" in output

    def test_submit_malformed_dash_o_exits_cleanly(self) -> None:
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(
            app,
            [
                "greet",
                "submit",
                "--profile",
                "research",
                "-o",
                "slurm.partition",  # missing =value
            ],
        )
        # Exit 1 from the options parser (malformed input), with error text.
        assert result.exit_code != 0
        combined = (result.output or "") + (result.stderr or "")
        assert "KEY=VALUE" in combined or "invalid -o entry" in combined

    def test_submit_malformed_options_file_exits_cleanly(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "opts.yaml"
        bad_file.write_text("slurm:\n  - not a mapping\n")
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(
            app,
            [
                "greet",
                "submit",
                "--options-file",
                str(bad_file),
            ],
        )
        # The options file has a list under slurm; that merges fine (slurm
        # becomes a list). Server-side would 422; the CLI itself tolerates
        # the shape. Instead test a truly malformed case: top-level scalar.
        scalar_file = tmp_path / "opts2.yaml"
        scalar_file.write_text("just-a-string")
        result = runner.invoke(
            app,
            [
                "greet",
                "submit",
                "--options-file",
                str(scalar_file),
            ],
        )
        assert result.exit_code != 0
        combined = (result.output or "") + (result.stderr or "")
        assert "top-level mapping" in combined

    def test_submit_missing_options_file_exits_cleanly(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "missing-options.yaml"
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(
            app,
            [
                "greet",
                "submit",
                "--options-file",
                str(missing_file),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        combined = (result.output or "") + (result.stderr or "")
        assert "invalid options" in combined
        assert str(missing_file) in combined


# ---------------------------------------------------------------------------
# NemoFunction CLI — fixtures
# ---------------------------------------------------------------------------


class _GreetSpec(BaseModel):
    name: str


class _GreetResponse(BaseModel):
    message: str


class _GreetFunction(NemoFunction[_GreetSpec]):
    name: ClassVar[str] = "greet"
    description: ClassVar[str] = "Say hello to a name."
    spec_schema: ClassVar[type[BaseModel]] = _GreetSpec

    async def run(self, spec: _GreetSpec) -> _GreetResponse:
        return _GreetResponse(message=f"Hello, {spec.name}!")


class _CountSpec(BaseModel):
    upto: int


class _CountFunction(NemoFunction[_CountSpec]):
    name: ClassVar[str] = "count"
    spec_schema: ClassVar[type[BaseModel]] = _CountSpec

    async def run(self, spec: _CountSpec) -> AsyncIterator[BaseModel]:
        for i in range(spec.upto):
            yield Heartbeat()
            del i
        yield Done()


class _WorkspaceSpec(BaseModel):
    pass


class _WorkspaceFunction(NemoFunction[_WorkspaceSpec]):
    name: ClassVar[str] = "echo-workspace"
    spec_schema: ClassVar[type[BaseModel]] = _WorkspaceSpec

    async def run(self, spec: _WorkspaceSpec, *, ctx: FunctionContext) -> dict:
        del spec
        return {"workspace": ctx.workspace}


class _LocalityFunction(NemoFunction[_WorkspaceSpec]):
    name: ClassVar[str] = "echo-locality"
    spec_schema: ClassVar[type[BaseModel]] = _WorkspaceSpec

    async def run(self, spec: _WorkspaceSpec, *, is_local: bool) -> dict:
        del spec
        return {"is_local": is_local}


class _FlatGreetFunction(_GreetFunction):
    generate_legacy_verbs: ClassVar[bool] = False


def _app_with_functions(*function_classes: type[NemoFunction]) -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _noop() -> None:
        pass

    fns = {f"plugin.{cls.name}": cls for cls in function_classes}
    add_function_commands(app, fns)
    return app


# ---------------------------------------------------------------------------
# Function sub-group registration
# ---------------------------------------------------------------------------


class TestFunctionSubgroupRegistration:
    def test_registers_subgroup_for_each_function(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["--help"])
        assert "greet" in result.output

    def test_subgroup_lists_two_verbs_no_explain(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "submit" in result.output
        # Functions deliberately do NOT get an `explain` verb — schemas
        # are introspected through `--help`.
        assert "explain" not in result.output

    def test_functions_rich_help_panel_label(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["--help"])
        assert "Functions" in result.output

    def test_bare_function_name_exits_non_zero(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet"])
        assert result.exit_code != 0

    def test_non_legacy_function_registers_flat_submit_command(self) -> None:
        app = _app_with_functions(_FlatGreetFunction)
        help_result = runner.invoke(app, ["greet", "--help"])
        assert help_result.exit_code == 0
        output = _plain(help_result.output)
        assert "COMMAND" not in output
        assert "--base-url" in output
        assert "--request-id" in output


# ---------------------------------------------------------------------------
# Function `run` verb
# ---------------------------------------------------------------------------


class TestFunctionRunVerb:
    def test_runs_function_with_json_spec(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "run", "--spec", '{"name": "Claude"}'])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"message": "Hello, Claude!"}

    def test_runs_function_with_spec_file(self, tmp_path: Path) -> None:
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("name: FromYaml\n")
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "run", "--spec-file", str(spec_file)])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["message"] == "Hello, FromYaml!"

    def test_invalid_spec_exits_with_error(self) -> None:
        # Validation runs against spec_schema before run is awaited, so an
        # unknown-type value surfaces as a clean Typer exit, not a traceback.
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "run", "--spec", "{}"])
        assert result.exit_code != 0
        combined = (result.output or "") + (result.stderr or "")
        assert "invalid spec" in combined

    def test_non_object_spec_exits_with_error(self) -> None:
        # ``--spec '[]'`` is syntactically valid JSON but not a mapping;
        # the deep-merge with per-field overlays would otherwise raise
        # a raw TypeError. Reject it at load time with the same clean
        # exit code 1 the malformed-JSON path uses.
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "run", "--spec", "[]"])
        assert result.exit_code == 1
        combined = (result.output or "") + (result.stderr or "")
        assert "invalid spec" in combined

    def test_run_streams_async_generator_frames_one_per_line(self) -> None:
        app = _app_with_functions(_CountFunction)
        result = runner.invoke(app, ["count", "run", "--spec", '{"upto": 2}'])
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.splitlines() if line.strip()]
        # 2 heartbeats + 1 terminator frame
        assert len(lines) == 3
        kinds = [json.loads(line)["kind"] for line in lines]
        assert kinds == ["heartbeat", "heartbeat", "done"]

    def test_run_injects_function_context_when_signature_asks(self) -> None:
        # Functions opt in to FunctionContext by name; --workspace flows
        # straight into ctx.workspace.
        app = _app_with_functions(_WorkspaceFunction)
        result = runner.invoke(
            app,
            ["echo-workspace", "run", "--spec", "{}", "--workspace", "team-alpha"],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"workspace": "team-alpha"}

    def test_run_injects_is_local_true_when_signature_asks(self) -> None:
        app = _app_with_functions(_LocalityFunction)
        result = runner.invoke(app, ["echo-locality", "run", "--spec", "{}"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"is_local": True}

    def test_run_validates_spec_with_local_context(self) -> None:
        class _LocalContextSpec(BaseModel):
            @model_validator(mode="before")
            @classmethod
            def require_local_context(cls, data: Any, info: ValidationInfo) -> Any:
                context = info.context
                if not (isinstance(context, dict) and context.get("is_local") is True):
                    raise ValueError("missing local validation context")
                return data

        class _LocalContextFunction(NemoFunction[_LocalContextSpec]):
            name: ClassVar[str] = "local-context"
            spec_schema: ClassVar[type[BaseModel]] = _LocalContextSpec

            async def run(self, spec: _LocalContextSpec) -> dict:
                del spec
                return {"ok": True}

        app = _app_with_functions(_LocalContextFunction)
        result = runner.invoke(app, ["local-context", "run", "--spec", "{}"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"ok": True}


# ---------------------------------------------------------------------------
# Function `submit` verb
# ---------------------------------------------------------------------------


class TestFunctionSubmitVerb:
    def test_submit_help_lists_expected_flags(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "submit", "--help"])
        assert result.exit_code == 0
        output = _plain(result.output)
        assert "--spec" in output
        assert "--spec-file" in output
        assert "--cluster" in output
        assert "--base-url" in output
        assert "--workspace" in output
        # No `--profile` / `-o` for functions: those are job-only knobs.
        assert "--profile" not in output

    def test_submit_invalid_spec_exits_before_network(self, monkeypatch) -> None:
        # Spec validation happens before any HTTP machinery is touched, so
        # a bad spec returns 1 (not 2 — that's reserved for network errors).
        called: list[bool] = []

        def _fail(*_args, **_kwargs) -> None:
            called.append(True)
            raise AssertionError("HTTP layer must not be reached for invalid spec")

        monkeypatch.setattr("nemo_platform_plugin.commands._post_function_submit", _fail)

        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "submit", "--spec", "{}"])
        assert result.exit_code == 1
        assert called == []

    def test_submit_non_object_spec_exits_before_network(self, monkeypatch) -> None:
        # Same guard as ``run``: a non-mapping ``--spec`` would otherwise
        # crash the merge step before validation has a chance to fail.
        called: list[bool] = []

        def _fail(*_args, **_kwargs) -> None:
            called.append(True)
            raise AssertionError("HTTP layer must not be reached for invalid spec")

        monkeypatch.setattr("nemo_platform_plugin.commands._post_function_submit", _fail)

        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "submit", "--spec", "[]"])
        assert result.exit_code == 1
        assert called == []
        combined = (result.output or "") + (result.stderr or "")
        assert "invalid spec" in combined

    def test_submit_posts_to_canonical_url_and_prints_json(self, monkeypatch) -> None:
        captured_url: list[str] = []
        captured_body: list[dict] = []
        captured_headers: list[dict[str, str]] = []

        def _fake_post(url: str, body: dict, *, headers: dict, timeout: float = 30.0, **_kwargs) -> None:
            captured_url.append(url)
            captured_body.append(body)
            captured_headers.append(dict(headers))
            del timeout
            typer.echo(json.dumps({"message": "ok"}))

        monkeypatch.setattr("nemo_platform_plugin.commands._post_function_submit", _fake_post)

        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(
            app,
            [
                "greet",
                "submit",
                "--spec",
                '{"name": "Ada"}',
                "--base-url",
                "http://my-platform:9090",
                "--workspace",
                "team-alpha",
                "--request-id",
                "req-42",
            ],
        )
        assert result.exit_code == 0, result.output
        # Canonical URL shape from `resources-jobs-functions.md`. The
        # plugin segment is derived from the function's module name —
        # tests/test_commands.py lives under `tests`, so the segment
        # falls back to that (good — it pins the convention).
        assert captured_url[0].endswith("/v2/workspaces/team-alpha/greet")
        assert "/apis/" in captured_url[0]
        assert captured_body[0] == {"name": "Ada"}
        assert captured_headers[0]["X-Request-ID"] == "req-42"

    def test_submit_streams_ndjson_lines(self, monkeypatch) -> None:
        # Build an httpx.MockTransport that returns a chunked NDJSON body
        # so we exercise the real streaming branch in _post_function_submit.
        ndjson_body = json.dumps({"kind": "heartbeat"}) + "\n" + json.dumps({"kind": "done"}) + "\n"

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=ndjson_body,
                headers={"content-type": "application/x-ndjson"},
            )

        transport = httpx.MockTransport(_handler)

        original_client = httpx.Client

        def _client_factory(*args, **kwargs):
            kwargs.setdefault("transport", transport)
            return original_client(*args, **kwargs)

        monkeypatch.setattr("nemo_platform_plugin.commands.httpx.Client", _client_factory)

        app = _app_with_functions(_CountFunction)
        result = runner.invoke(
            app,
            ["count", "submit", "--spec", '{"upto": 1}', "--base-url", "http://test"],
        )
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.splitlines() if line.strip()]
        kinds = [json.loads(line)["kind"] for line in lines]
        assert kinds == ["heartbeat", "done"]

    def test_submit_returns_exit_code_2_on_http_error(self, monkeypatch) -> None:
        # Regression: ``client.stream`` opens the response unbuffered,
        # so the error formatter that reads ``exc.response.text`` used
        # to blow up with ``ResponseNotRead`` and propagate a raw
        # traceback to the user instead of the structured error
        # message.
        #
        # Crucially, this handler uses ``stream=httpx.ByteStream(...)``
        # rather than the more obvious ``json=...`` / ``content=...``
        # — those constructors pre-populate ``_content`` on the
        # response, which means the buggy code reads from the
        # pre-buffered bytes and never reproduces ``ResponseNotRead``.
        # ``ByteStream`` defers the bytes until ``response.read()`` is
        # called, matching production wire behaviour.
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(b'{"detail": "boom"}'),
            )

        transport = httpx.MockTransport(_handler)
        original_client = httpx.Client

        def _client_factory(*args, **kwargs):
            kwargs.setdefault("transport", transport)
            return original_client(*args, **kwargs)

        monkeypatch.setattr("nemo_platform_plugin.commands.httpx.Client", _client_factory)

        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(
            app,
            ["greet", "submit", "--spec", '{"name": "x"}', "--base-url", "http://test"],
        )
        # Exit 2 marks "transport / server reported failure" — distinct
        # from exit 1 (CLI-side validation) so wrapper scripts can branch.
        assert result.exit_code == 2
        combined = (result.output or "") + (result.stderr or "")
        assert "500" in combined
        assert "boom" in combined
        assert "Request: POST http://test/" in combined
        assert "Target:" not in combined

    def test_submit_uses_nmp_base_url_env_when_no_flags(self, monkeypatch) -> None:
        captured_url: list[str] = []

        def _fake_post(url: str, body: dict, *, headers: dict, timeout: float = 30.0, **_kwargs) -> None:  # noqa: ARG001
            captured_url.append(url)

        monkeypatch.setattr("nemo_platform_plugin.commands._post_function_submit", _fake_post)
        monkeypatch.setenv("NMP_BASE_URL", "http://from-env:1234")

        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(
            app,
            ["greet", "submit", "--spec", '{"name": "x"}'],
        )
        assert result.exit_code == 0, result.output
        assert captured_url[0].startswith("http://from-env:1234/")

    def test_non_legacy_function_name_submits_remotely(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _fake_post(url: str, body: dict, *, headers: dict, timeout: float = 30.0, **_kwargs) -> None:
            captured.update({"url": url, "body": body, "headers": headers})
            del timeout
            typer.echo(json.dumps({"message": "ok"}))

        monkeypatch.setattr("nemo_platform_plugin.commands._post_function_submit", _fake_post)

        app = _app_with_functions(_FlatGreetFunction)
        result = runner.invoke(
            app,
            [
                "greet",
                "--spec",
                '{"name": "Ada"}',
                "--base-url",
                "http://platform",
                "--workspace",
                "team-alpha",
            ],
        )

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"message": "ok"}
        assert str(captured["url"]).startswith("http://platform/apis/")
        assert str(captured["url"]).endswith("/v2/workspaces/team-alpha/greet")
        assert captured["body"] == {"name": "Ada"}

        assert runner.invoke(app, ["greet", "run"]).exit_code != 0
        assert runner.invoke(app, ["greet", "submit"]).exit_code != 0


# ---------------------------------------------------------------------------
# Function URL derivation
# ---------------------------------------------------------------------------


class TestApiSegmentForFunction:
    """Pin how the ``submit`` URL maps a function class to its API segment.

    The platform mounts each plugin's functions under the ``<plugin>``
    half of its ``nemo.functions`` entry-point key (``<plugin>.<fn>``),
    so the CLI has to derive the same prefix or every ``submit`` 404s.
    The authoritative source is the registered entry-point — module
    paths are only consulted when the function isn't installed
    (in-process tests, scratch invocations).
    """

    def test_uses_registered_entry_point_key(self, monkeypatch) -> None:
        from nemo_platform_plugin.commands import _api_segment_for_function

        class _Spec(BaseModel):
            pass

        class _Fn(NemoFunction[_Spec]):
            name: ClassVar[str] = "greet"
            spec_schema: ClassVar[type[BaseModel]] = _Spec

            async def run(self, spec: _Spec) -> dict:
                del spec
                return {}

        # Pretend this class is the value behind a ``my-plugin.greet``
        # entry-point. Patching ``discover_functions`` keeps the test
        # hermetic and exercises exactly the lookup branch — the
        # ``-plugin`` suffix in the key is kept verbatim, even though
        # the package layout would suggest a different segment.
        _Fn.__module__ = "nemo_my_plugin.functions.greet"
        monkeypatch.setattr(
            "nemo_platform_plugin.discovery.discover_functions",
            lambda: {"my-plugin.greet": _Fn},
        )
        assert _api_segment_for_function(_Fn) == "my-plugin"

    def test_falls_back_to_module_when_not_registered(self, monkeypatch) -> None:
        from nemo_platform_plugin.commands import _api_segment_for_function

        class _Spec(BaseModel):
            pass

        class _Fn(NemoFunction[_Spec]):
            name: ClassVar[str] = "greet"
            spec_schema: ClassVar[type[BaseModel]] = _Spec

            async def run(self, spec: _Spec) -> dict:
                del spec
                return {}

        # No entry-point match → module-name fallback. The fallback no
        # longer strips ``_plugin``; an unregistered class in a
        # ``nemo_<name>_plugin`` package keeps the suffix so the
        # CLI fails loudly rather than 404ing against the wrong URL.
        _Fn.__module__ = "nemo_example_plugin.functions.greet"
        monkeypatch.setattr(
            "nemo_platform_plugin.discovery.discover_functions",
            lambda: {},
        )
        assert _api_segment_for_function(_Fn) == "example-plugin"

    def test_handles_missing_nemo_prefix(self, monkeypatch) -> None:
        from nemo_platform_plugin.commands import _api_segment_for_function

        class _Spec(BaseModel):
            pass

        class _Fn(NemoFunction[_Spec]):
            name: ClassVar[str] = "x"
            spec_schema: ClassVar[type[BaseModel]] = _Spec

            async def run(self, spec: _Spec) -> dict:
                del spec
                return {}

        # In-tree code outside ``nemo_*`` keeps its module name as-is
        # (kebab-cased) so tests with inline classes still produce a
        # stable, predictable segment.
        _Fn.__module__ = "tests.fixtures.things"
        monkeypatch.setattr(
            "nemo_platform_plugin.discovery.discover_functions",
            lambda: {},
        )
        assert _api_segment_for_function(_Fn) == "tests"


# ---------------------------------------------------------------------------
# Function CLI — auto-generated per-field flags from spec_schema
# ---------------------------------------------------------------------------


class _NestedTarget(BaseModel):
    url: str
    timeout_seconds: int = 30


class _NestedSpec(BaseModel):
    name: str
    target: _NestedTarget


class _NestedFunction(NemoFunction[_NestedSpec]):
    name: ClassVar[str] = "ping"
    description: ClassVar[str] = "Ping a nested target."
    spec_schema: ClassVar[type[BaseModel]] = _NestedSpec

    async def run(self, spec: _NestedSpec) -> dict:
        return {
            "name": spec.name,
            "url": spec.target.url,
            "timeout": spec.target.timeout_seconds,
        }


class TestFunctionAutoSpecFlags:
    """Per-field flags auto-derived from a function's ``spec_schema``.

    The wiring is shared with the jobs CLI via
    :mod:`nemo_platform_plugin._spec_flags`; these tests pin the wiring on the
    function side specifically so a future refactor can't silently
    break the function ergonomics.
    """

    def test_run_help_lists_one_flag_per_scalar_leaf(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "run", "--help"])
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "--name" in plain
        assert "Function Spec" in plain
        # The epilog tells the user where the flags came from — without
        # this, "schema discovery" still requires reading source.
        assert "GreetSpec" in plain

    def test_run_accepts_per_field_flag(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "run", "--name", "Razvan"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"message": "Hello, Razvan!"}

    def test_per_field_flag_overlays_on_top_of_spec(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(
            app,
            [
                "greet",
                "run",
                "--spec",
                '{"name": "from-spec"}',
                "--name",
                "from-flag",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"message": "Hello, from-flag!"}

    def test_nested_field_uses_dotted_flag_name(self) -> None:
        # ``target.url`` and ``target.timeout-seconds`` are the canonical
        # rendering of a nested submodel field. Underscores within a
        # segment kebab-case; the dot between segments is preserved.
        app = _app_with_functions(_NestedFunction)
        result = runner.invoke(app, ["ping", "run", "--help"])
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "--name" in plain
        assert "--target.url" in plain
        assert "--target.timeout-seconds" in plain

    def test_nested_field_overlay_round_trips(self) -> None:
        app = _app_with_functions(_NestedFunction)
        result = runner.invoke(
            app,
            [
                "ping",
                "run",
                "--name",
                "site-a",
                "--target.url",
                "https://example.test",
                "--target.timeout-seconds",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {
            "name": "site-a",
            "url": "https://example.test",
            "timeout": 5,
        }

    def test_workspace_field_in_spec_does_not_collide_with_static_flag(self) -> None:
        # A spec field literally named ``workspace`` would alias the
        # static ``--workspace`` flag (which feeds ``ctx.workspace``,
        # not the spec). The reserved-flag set drops it from the
        # auto-generated panel; users still pass it via --spec.
        class _ConfusingSpec(BaseModel):
            workspace: str = "default-ws"

        class _ConfusingFunction(NemoFunction[_ConfusingSpec]):
            name: ClassVar[str] = "confuse"
            spec_schema: ClassVar[type[BaseModel]] = _ConfusingSpec

            async def run(self, spec: _ConfusingSpec) -> dict:
                return {"in_spec": spec.workspace}

        app = _app_with_functions(_ConfusingFunction)
        help_result = runner.invoke(app, ["confuse", "run", "--help"])
        plain = _plain(help_result.output)
        # The flag exists exactly once — the static-input version
        # under the "Spec Source" panel, not duplicated under the
        # "Function Spec" panel.
        assert plain.count("--workspace") == 1

        # And --spec still wins for the spec-side workspace value.
        run_result = runner.invoke(
            app,
            ["confuse", "run", "--spec", '{"workspace": "from-spec"}', "--workspace", "ctx-ws"],
        )
        assert run_result.exit_code == 0, run_result.output
        assert json.loads(run_result.output) == {"in_spec": "from-spec"}

    def test_submit_help_lists_per_field_flags_under_function_spec_panel(self) -> None:
        app = _app_with_functions(_GreetFunction)
        result = runner.invoke(app, ["greet", "submit", "--help"])
        plain = _plain(result.output)
        assert "--name" in plain
        assert "Function Spec" in plain
        # `--workspace` is a submission-side flag (URL segment), so it
        # stays in the Submission panel and isn't auto-derived even if
        # a spec field happened to share the name.
        assert "Submission" in plain

    def test_no_spec_schema_fields_falls_back_to_no_flags_epilog(self) -> None:
        # Functions with an empty spec_schema (or only unsupported
        # types) get the "no per-field flags" epilog so users still
        # learn how to pass values.
        class _EmptySpec(BaseModel):
            pass

        class _EmptyFunction(NemoFunction[_EmptySpec]):
            name: ClassVar[str] = "noop"
            spec_schema: ClassVar[type[BaseModel]] = _EmptySpec

            async def run(self, spec: _EmptySpec) -> dict:
                del spec
                return {}

        app = _app_with_functions(_EmptyFunction)
        result = runner.invoke(app, ["noop", "run", "--help"])
        plain = _plain(result.output)
        assert "Function Spec" not in plain
        assert "no per-field flags" in plain


# ---------------------------------------------------------------------------
# Job CLI — auto-generated per-field flags from spec_schema
# ---------------------------------------------------------------------------


class _GreetJobSpec(BaseModel):
    name: str = "world"
    loud: bool = False


class _GreetSpecJob(NemoJob):
    name = "greet-spec"
    description = "Return a greeting validated against a schema."
    spec_schema: ClassVar[type[BaseModel]] = _GreetJobSpec

    def run(self, config: dict) -> dict:
        spec = _GreetJobSpec.model_validate(config)
        message = f"Hello, {spec.name}!"
        if spec.loud:
            message = message.upper()
        return {"message": message}


class _NestedJobTarget(BaseModel):
    url: str
    timeout_seconds: int = 30


class _NestedJobSpec(BaseModel):
    name: str
    target: _NestedJobTarget


class _NestedSpecJob(NemoJob):
    name = "ping-spec"
    description = "Ping a nested target."
    spec_schema: ClassVar[type[BaseModel]] = _NestedJobSpec

    def run(self, config: dict) -> dict:
        spec = _NestedJobSpec.model_validate(config)
        return {
            "name": spec.name,
            "url": spec.target.url,
            "timeout": spec.target.timeout_seconds,
        }


class TestJobAutoSpecFlags:
    """Per-field flags auto-derived from a job's ``spec_schema`` on ``run``/``submit``.

    Mirrors :class:`TestFunctionAutoSpecFlags` so the two CLIs stay in
    lockstep — the wiring is shared via :mod:`nemo_platform_plugin._spec_flags`.
    """

    def test_run_help_lists_one_flag_per_scalar_leaf(self) -> None:
        app = _app_with_jobs(_GreetSpecJob)
        result = runner.invoke(app, ["greet-spec", "run", "--help"])
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "--name" in plain
        assert "--loud" in plain
        assert "Job Spec" in plain

    def test_run_accepts_per_field_flag(self) -> None:
        app = _app_with_jobs(_GreetSpecJob)
        result = runner.invoke(app, ["greet-spec", "run", "--name", "Razvan"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {"message": "Hello, Razvan!"}

    def test_per_field_flag_overlays_on_top_of_spec(self) -> None:
        # --spec sets the base; --name overlays on top per the
        # documented precedence (spec-file → spec → per-field).
        app = _app_with_jobs(_GreetSpecJob)
        result = runner.invoke(
            app,
            [
                "greet-spec",
                "run",
                "--spec",
                '{"name": "from-spec", "loud": true}',
                "--name",
                "from-flag",
            ],
        )
        assert result.exit_code == 0, result.output
        # Loud carries over from --spec; name is overridden by the flag.
        assert json.loads(result.output) == {"message": "HELLO, FROM-FLAG!"}

    def test_nested_field_uses_dotted_flag_name(self) -> None:
        app = _app_with_jobs(_NestedSpecJob)
        result = runner.invoke(app, ["ping-spec", "run", "--help"])
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "--name" in plain
        assert "--target.url" in plain
        assert "--target.timeout-seconds" in plain

    def test_nested_field_overlay_round_trips(self) -> None:
        app = _app_with_jobs(_NestedSpecJob)
        result = runner.invoke(
            app,
            [
                "ping-spec",
                "run",
                "--name",
                "site-a",
                "--target.url",
                "https://example.test",
                "--target.timeout-seconds",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {
            "name": "site-a",
            "url": "https://example.test",
            "timeout": 5,
        }

    def test_workspace_field_in_spec_does_not_collide_with_static_flag(self) -> None:
        # A spec field literally named ``workspace`` would alias the
        # ``submit``-side ``--workspace`` flag (which feeds the URL
        # segment, not the spec). The reserved-flag set drops it from
        # the auto-generated ``submit`` panel; users still pass it via
        # --spec. ``run`` doesn't reserve ``workspace`` (it has no such
        # static flag), so the auto-flag still appears there.
        class _WorkspaceSpec(BaseModel):
            workspace: str = "default-ws"

        class _WorkspaceJob(NemoJob):
            name = "ws-confuse"
            spec_schema: ClassVar[type[BaseModel]] = _WorkspaceSpec

            def run(self, config: dict) -> dict:
                return {"in_spec": _WorkspaceSpec.model_validate(config).workspace}

        app = _app_with_jobs(_WorkspaceJob)
        submit_help = runner.invoke(app, ["ws-confuse", "submit", "--help"])
        plain = _plain(submit_help.output)
        # The flag exists exactly once on ``submit`` — the static
        # submission-side version under the "Submission" panel, not
        # duplicated under the "Job Spec" panel.
        assert plain.count("--workspace") == 1

    def test_submit_help_lists_per_field_flags_under_job_spec_panel(self) -> None:
        app = _app_with_jobs(_GreetSpecJob)
        result = runner.invoke(app, ["greet-spec", "submit", "--help"])
        plain = _plain(result.output)
        assert "--name" in plain
        assert "Job Spec" in plain
        # ``--profile`` / ``-o`` / ``--workspace`` remain visible under
        # the Submission panel — auto-flags don't displace static ones.
        assert "Submission" in plain
        assert "--profile" in plain

    def test_no_spec_schema_renders_only_static_panels(self) -> None:
        # Schema-less jobs (``_GreetJob``) don't surface a "Job Spec"
        # panel — the user passes values exclusively via --spec /
        # --spec-file under the "Spec Source" panel.
        app = _app_with_jobs(_GreetJob)
        result = runner.invoke(app, ["greet", "run", "--help"])
        plain = _plain(result.output)
        assert "Job Spec" not in plain
        assert "Spec Source" in plain

    def test_input_spec_schema_drives_auto_flags_when_declared(self) -> None:
        # When a job declares both ``input_spec_schema`` and
        # ``spec_schema``, the input shape is what the user types — so
        # that's the schema the CLI must walk.
        class _InputShape(BaseModel):
            target_name: str

        class _CanonicalShape(BaseModel):
            resolved_id: str

        class _TwoShapeJob(NemoJob):
            name = "two-shape"
            input_spec_schema: ClassVar[type[BaseModel]] = _InputShape
            spec_schema: ClassVar[type[BaseModel]] = _CanonicalShape

            def run(self, config: dict) -> dict:
                return {"got": config}

        app = _app_with_jobs(_TwoShapeJob)
        result = runner.invoke(app, ["two-shape", "run", "--help"])
        plain = _plain(result.output)
        # The flag follows ``input_spec_schema``, not ``spec_schema``.
        assert "--target-name" in plain
        assert "--resolved-id" not in plain
