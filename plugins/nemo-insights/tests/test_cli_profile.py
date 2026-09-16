# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import typer
from nemo_helix import NeMoHelixError
from nemo_insights_plugin import cli
from nemo_insights_plugin.analyst.cli import AnalystCLI
from nemo_insights_plugin.contracts.checks import CheckResult
from nemo_insights_plugin.contracts.profile import DEFAULT_BASE_URL
from nemo_insights_plugin.preflight import AnalysisProbes
from nooa import GenerationError
from typer.testing import CliRunner

runner = CliRunner()
_PROFILE_ENV_KEYS = ("NHX_BASE_URL", "TEST_PROFILE_ENV")


class AnalystRecorder:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    async def __call__(self, **kwargs: object) -> str:
        self.kwargs = kwargs
        return "analysis-summary"


@pytest.fixture
def app() -> typer.Typer:
    return AnalystCLI().get_cli()


@pytest.fixture(autouse=True)
def restore_profile_env() -> Iterator[None]:
    original = {key: os.environ.get(key) for key in _PROFILE_ENV_KEYS}
    missing = {key for key in _PROFILE_ENV_KEYS if key not in os.environ}
    yield
    for key in _PROFILE_ENV_KEYS:
        if key in missing:
            os.environ.pop(key, None)
        else:
            value = original[key]
            if value is not None:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def quiet_preflight(monkeypatch: pytest.MonkeyPatch, restore_profile_env: None) -> None:
    async def queryable(base_url: str, workspace: str, agent: str) -> bool:
        return True

    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        AnalysisProbes(
            http_ok=lambda base_url: True,
            workspace_ok=queryable,
        ),
    )
    monkeypatch.setattr(cli, "make_client", lambda base_url: object())
    monkeypatch.setattr(
        cli,
        "check_models",
        lambda: [
            CheckResult(
                name="agent-models",
                group="models",
                status="pass",
                severity="required",
                message="default=default/gpt-5; fast=default/gpt-5-mini",
            )
        ],
    )
    monkeypatch.setattr(
        "nemo_insights_plugin.preflight.configured_model_refs",
        lambda: SimpleNamespace(default="default/gpt-5", fast="default/gpt-5-mini"),
    )


@pytest.fixture
def profile_tree(tmp_path: Path) -> Path:
    (tmp_path / "optimizer.yaml").write_text(
        "agent: flight-planner\n"
        "task_template: ./evals/task_template\n"
        "datasets:\n  train: ./evals/train\n  validation: ./evals/validation\n"
        "workspace: flight-workspace\n",
        encoding="utf-8",
    )
    (tmp_path / "ETHOS.md").write_text("# Flight planner", encoding="utf-8")
    return tmp_path


def test_analyze_runs_flag_free_from_profile(app: typer.Typer, profile_tree: Path, monkeypatch) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None
    assert recorder.kwargs["agent"] == "flight-planner"
    assert recorder.kwargs["workspace"] == "flight-workspace"
    assert recorder.kwargs["ethos"] == "# Flight planner"
    assert recorder.kwargs["insights_output"] is None, "a discovered profile must not divert writes off the platform"


def test_no_local_only_flag_is_exposed(app: typer.Typer, profile_tree: Path, monkeypatch) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run", "--local-only"])

    assert result.exit_code != 0
    assert recorder.kwargs is None


def test_insights_file_output_is_a_mirror_not_a_redirect(
    app: typer.Typer, profile_tree: Path, tmp_path: Path, monkeypatch
) -> None:
    recorder = AnalystRecorder()
    output = tmp_path / "mirror.yaml"
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run", "--insights-file-output", str(output)])

    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None
    assert recorder.kwargs["insights_output"] == output
    assert "local_only" not in recorder.kwargs, "the CLI must never ask for local-only persistence"


def test_unusable_mirror_directory_drops_the_mirror_and_still_analyzes(
    app: typer.Typer, profile_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The platform is the source of truth; a bad local path must not cost the run."""
    recorder = AnalystRecorder()
    output = tmp_path / "unwritable" / "insights.yaml"
    original_mkdir = Path.mkdir

    def refuse_mirror_dir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if self == output.parent:
            raise PermissionError(13, "Permission denied")
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", refuse_mirror_dir)
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run", "--insights-file-output", str(output)])

    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None, "the analysis must still run"
    assert recorder.kwargs["insights_output"] is None
    assert "insights mirror disabled" in result.stderr
    assert "still written to the platform" in result.stderr


def test_analyze_flags_override_profile(app: typer.Typer, profile_tree: Path, monkeypatch) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run", "--agent", "other", "--workspace", "other-ws"])

    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None
    assert recorder.kwargs["agent"] == "other"
    assert recorder.kwargs["workspace"] == "other-ws"


def test_profile_env_is_loaded_before_base_url_resolution(app: typer.Typer, profile_tree: Path, monkeypatch) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.delenv("NHX_BASE_URL", raising=False)
    (profile_tree / ".env").write_text("NHX_BASE_URL=https://platform.example\n", encoding="utf-8")
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None
    assert recorder.kwargs["base_url"] == "https://platform.example"


def test_analyze_renders_invalid_profile_env_as_command_error(
    app: typer.Typer,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = AnalystRecorder()
    probe_calls: list[str] = []

    async def record_workspace_probe(base_url: str, workspace: str, agent: str) -> bool:
        probe_calls.append("workspace")
        return True

    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        AnalysisProbes(
            http_ok=lambda base_url: probe_calls.append("http") or True,
            workspace_ok=record_workspace_probe,
        ),
    )
    env_file = profile_tree / ".env"
    env_file.write_bytes(b"KEY=\xff")
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    error_lines = [line for line in result.stderr.splitlines() if line.startswith("Error:")]
    assert len(error_lines) == 1
    assert error_lines[0].startswith(f"Error: Could not read environment file {env_file}:")
    assert "Check that the file is readable UTF-8 text, then retry." in error_lines[0]
    assert "Traceback" not in result.output
    assert recorder.kwargs is None
    assert probe_calls == []


def test_doctor_renders_invalid_profile_env_as_command_error(
    app: typer.Typer,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = AnalystRecorder()
    probe_calls: list[str] = []

    async def record_workspace_probe(base_url: str, workspace: str, agent: str) -> bool:
        probe_calls.append("workspace")
        return True

    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        AnalysisProbes(
            http_ok=lambda base_url: probe_calls.append("http") or True,
            workspace_ok=record_workspace_probe,
        ),
    )
    env_file = profile_tree / ".env"
    env_file.write_bytes(b"KEY=\xff")
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    error_lines = [line for line in result.stderr.splitlines() if line.startswith("Error:")]
    assert len(error_lines) == 1
    assert error_lines[0].startswith(f"Error: Could not read environment file {env_file}:")
    assert "Check that the file is readable UTF-8 text, then retry." in error_lines[0]
    assert "Traceback" not in result.output
    assert recorder.kwargs is None
    assert probe_calls == []


@pytest.mark.parametrize(
    ("arguments", "environment", "profile_env", "expected"),
    [
        (
            ["--base-url", "https://flag.example"],
            {"NHX_BASE_URL": "https://process.example"},
            "NHX_BASE_URL=https://profile.example\n",
            "https://flag.example",
        ),
        ([], {}, "NHX_BASE_URL=https://profile.example\n", "https://profile.example"),
        (
            [],
            {"NHX_BASE_URL": "https://process.example"},
            "NHX_BASE_URL=https://profile.example\n",
            "https://process.example",
        ),
        ([], {"NEMO_BASE_URL": "https://legacy.example"}, None, DEFAULT_BASE_URL),
    ],
    ids=["explicit", "profile-env", "process-env", "legacy-ignored"],
)
def test_doctor_resolves_base_url_after_profile_env_loading(
    app: typer.Typer,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    environment: dict[str, str],
    profile_env: str | None,
    expected: str,
) -> None:
    http_urls: list[str] = []
    workspace_urls: list[str] = []

    async def record_workspace_probe(base_url: str, workspace: str, agent: str) -> bool:
        workspace_urls.append(base_url)
        return True

    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        AnalysisProbes(
            http_ok=lambda base_url: http_urls.append(base_url) or True,
            workspace_ok=record_workspace_probe,
        ),
    )
    monkeypatch.delenv("NHX_BASE_URL", raising=False)
    monkeypatch.delenv("NEMO_BASE_URL", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    if profile_env is not None:
        (profile_tree / ".env").write_text(profile_env, encoding="utf-8")
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor", *arguments])

    assert result.exit_code == 0, result.output
    assert http_urls == [expected]
    assert workspace_urls == [expected]


@pytest.mark.parametrize(
    ("arguments", "environment", "expected"),
    [
        (
            ["--base-url", "https://flag.example"],
            {"NHX_BASE_URL": "https://nhx.example", "NEMO_BASE_URL": "https://legacy.example"},
            "https://flag.example",
        ),
        (
            [],
            {"NHX_BASE_URL": "https://nhx.example", "NEMO_BASE_URL": "https://legacy.example"},
            "https://nhx.example",
        ),
        ([], {"NEMO_BASE_URL": "https://legacy.example"}, DEFAULT_BASE_URL),
    ],
)
def test_base_url_precedence_uses_only_nhx_base_url(
    app: typer.Typer,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    environment: dict[str, str],
    expected: str,
) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.delenv("NHX_BASE_URL", raising=False)
    monkeypatch.delenv("NEMO_BASE_URL", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run", *arguments])

    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None
    assert recorder.kwargs["base_url"] == expected


def test_explicit_profile_is_used_outside_profile_directory(
    app: typer.Typer, profile_tree: Path, tmp_path: Path, monkeypatch
) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "--profile", str(profile_tree / "optimizer.yaml")])

    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None
    assert recorder.kwargs["agent"] == "flight-planner"


def test_malformed_explicit_profile_errors(app: typer.Typer, tmp_path: Path, monkeypatch) -> None:
    profile = tmp_path / "optimizer.yaml"
    profile.write_text("agent: ''\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "--profile", str(profile)])

    assert result.exit_code != 0
    assert "Invalid profile" in result.output


def test_malformed_discovered_profile_warns_when_flags_are_complete(
    app: typer.Typer, tmp_path: Path, monkeypatch
) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    (tmp_path / "optimizer.yaml").write_text("agent: ''\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "--agent", "other", "--workspace", "other-ws"])

    assert result.exit_code == 0, result.output
    assert "warning:" in result.output
    assert "Invalid profile" in result.output


def test_malformed_discovered_profile_warns_with_agent_only(app: typer.Typer, tmp_path: Path, monkeypatch) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    (tmp_path / "optimizer.yaml").write_text("agent: ''\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "--agent", "other"])

    assert result.exit_code == 0, result.output
    assert "warning:" in result.output
    assert "Invalid profile" in result.output
    assert recorder.kwargs is not None
    assert recorder.kwargs["workspace"] == "default"


def test_malformed_discovered_profile_still_loads_env_file(app: typer.Typer, tmp_path: Path, monkeypatch) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.delenv("TEST_PROFILE_ENV", raising=False)
    (tmp_path / "optimizer.yaml").write_text("agent: ''\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TEST_PROFILE_ENV=from-env-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "--agent", "other"])

    assert result.exit_code == 0, result.output
    assert os.environ["TEST_PROFILE_ENV"] == "from-env-file"
    assert recorder.kwargs is not None


def test_malformed_discovered_profile_errors_without_agent(app: typer.Typer, tmp_path: Path, monkeypatch) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    (tmp_path / "optimizer.yaml").write_text("agent: ''\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "--workspace", "other-ws"])

    assert result.exit_code != 0
    assert "Invalid profile" in result.output
    assert recorder.kwargs is None


def test_missing_profile_and_agent_errors(app: typer.Typer, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run"])

    assert result.exit_code != 0
    assert "No --agent given and no optimizer.yaml profile found" in result.output


def test_analyze_blocks_before_runner_when_model_configuration_is_missing(
    app: typer.Typer, profile_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = AnalystRecorder()
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.setattr(
        cli,
        "check_models",
        lambda: [
            CheckResult(
                name="agent-models",
                group="models",
                status="fail",
                severity="required",
                message="No default model is configured",
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        AnalysisProbes(
            http_ok=lambda base_url: True,
            workspace_ok=lambda base_url, workspace, agent: _queryable(),
        ),
    )
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    assert "No default model is configured" in result.output
    assert recorder.kwargs is None


def test_analyze_runs_only_the_model_configuration_check(
    app: typer.Typer, profile_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = AnalystRecorder()
    probe_calls: list[str] = []

    async def record_workspace_probe(base_url: str, workspace: str, agent: str) -> bool:
        probe_calls.append("workspace")
        return False

    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        AnalysisProbes(
            http_ok=lambda base_url: probe_calls.append("http") or False,
            workspace_ok=record_workspace_probe,
        ),
    )
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None
    assert probe_calls == []


@pytest.mark.parametrize(
    "error",
    [
        NeMoHelixError("Intake SDK failed"),
        httpx.ConnectError("Intake unavailable", request=httpx.Request("GET", "https://platform.example")),
    ],
)
def test_analyze_renders_expected_platform_failures_without_traceback(
    app: typer.Typer,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    async def fail_analysis(**kwargs: object) -> str:
        raise error

    monkeypatch.setattr(cli, "run_analyst", fail_analysis)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    error_lines = [line for line in result.stderr.splitlines() if line.startswith("Error:")]
    assert len(error_lines) == 1
    assert "analysis failed" in error_lines[0]
    assert "--base-url/NHX_BASE_URL" in error_lines[0]
    assert "Traceback" not in result.output


def test_analyze_renders_generation_error_with_model_and_usage_guidance(
    app: typer.Typer,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_analysis(**kwargs: object) -> str:
        raise GenerationError("request limit exceeded")

    monkeypatch.setattr(cli, "run_analyst", fail_analysis)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 1
    error_lines = [line for line in result.stderr.splitlines() if line.startswith("Error:")]
    assert error_lines == [
        "Error: analyst run failed: request limit exceeded. "
        "Check inference model access and credentials, then retry or adjust usage limits."
    ]
    assert "--base-url/NHX_BASE_URL" not in result.stderr
    assert "Intake availability" not in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_analyze_constructor_failure_warns_then_exits_cleanly(
    app: typer.Typer,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    attempts = 0

    def fail_to_construct(base_url: str | None) -> object:
        nonlocal attempts
        attempts += 1
        raise error_type("invalid\nremote client context")

    monkeypatch.setattr(cli, "make_client", fail_to_construct)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run"])

    assert attempts == 1
    assert result.exit_code == 1
    error_lines = [line for line in result.stderr.splitlines() if line.startswith("Error:")]
    assert error_lines == [
        "Error: analysis failed: invalid remote client context. "
        "Check --base-url/NHX_BASE_URL, authentication, workspace, and Intake availability."
    ]
    assert "analyst run failed" not in result.stderr
    assert "Traceback" not in result.output
    assert "During handling of the above exception" not in result.output


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"insights: [\n", "valid YAML"),
        (b"- id: first\n", "YAML mapping"),
        (b"scalar\n", "YAML mapping"),
        (b"\xff\xfe", "UTF-8"),
    ],
    ids=["malformed-yaml", "list-root", "scalar-root", "invalid-utf8"],
)
def test_analyze_rejects_invalid_existing_insights_file_before_runner(
    app: typer.Typer,
    profile_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    expected: str,
) -> None:
    recorder = AnalystRecorder()
    output = tmp_path / "explicit-insights.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.chdir(profile_tree)
    arguments = ["run", "--insights-file-output", str(output)]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    error_lines = [line for line in result.stderr.splitlines() if line.startswith("Error:")]
    assert len(error_lines) == 1
    assert f"insights file {output}" in error_lines[0]
    assert expected in error_lines[0]
    assert "Traceback" not in result.output
    assert recorder.kwargs is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"insights: null\n", "`insights` must be a list"),
        (b"insights: 42\n", "`insights` must be a list"),
        (b"insights: records\n", "`insights` must be a list"),
        (b"insights: {id: one}\n", "`insights` must be a list"),
        (
            b"insights:\n  - {id: one}\n  - broken\n",
            "`insights` item 2 must be a YAML mapping",
        ),
    ],
    ids=["null", "numeric", "scalar", "mapping", "scalar-list-item"],
)
def test_analyze_rejects_invalid_insights_records_before_runner(
    app: typer.Typer,
    profile_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    expected: str,
) -> None:
    recorder = AnalystRecorder()
    output = tmp_path / "explicit-insights.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.chdir(profile_tree)
    arguments = ["run", "--insights-file-output", str(output)]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    error_lines = [line for line in result.stderr.splitlines() if line.startswith("Error:")]
    assert len(error_lines) == 1
    assert f"insights file {output}" in error_lines[0]
    assert expected in error_lines[0]
    assert "Traceback" not in result.output
    assert recorder.kwargs is None


def test_analyze_accepts_existing_insights_file_without_insights_key(
    app: typer.Typer,
    profile_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = AnalystRecorder()
    output = tmp_path / "explicit-insights.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("metadata: retained\n", encoding="utf-8")
    monkeypatch.setattr(cli, "run_analyst", recorder)
    monkeypatch.chdir(profile_tree)
    arguments = ["run", "--insights-file-output", str(output)]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None


@pytest.mark.parametrize("command", ["doctor", "run"])
def test_commands_reject_invalid_utf8_ethos(
    app: typer.Typer,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    (profile_tree / "ETHOS.md").write_bytes(b"\xff\xfe")
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, [command])

    assert result.exit_code == 1
    assert "ethos" in result.output.lower()
    assert "UTF-8" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["doctor", "run"])
def test_commands_reject_unreadable_ethos(
    app: typer.Typer,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    ethos = profile_tree / "ETHOS.md"
    original_read_text = Path.read_text

    def deny_ethos_read(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path == ethos:
            raise PermissionError("permission denied")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", deny_ethos_read)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, [command])

    assert result.exit_code == 1
    assert "ethos" in result.output.lower()
    assert "permission denied" in result.output
    assert "ensure the file is readable and encoded as UTF-8" in result.output
    assert "Traceback" not in result.output


async def _queryable() -> bool:
    return True


def test_doctor_exits_nonzero_for_missing_profile(app: typer.Typer, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "no optimizer.yaml found" in result.output


def test_doctor_missing_profile_still_runs_environment_checks(
    app: typer.Typer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http_urls: list[str] = []
    workspace_urls: list[str] = []

    async def record_workspace_probe(base_url: str, workspace: str, agent: str) -> bool:
        workspace_urls.append(base_url)
        return True

    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        AnalysisProbes(
            http_ok=lambda base_url: http_urls.append(base_url) or True,
            workspace_ok=record_workspace_probe,
        ),
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor", "--base-url", "https://flag.example"])

    assert result.exit_code == 1
    assert "no optimizer.yaml found" in result.output
    assert "Models\n  ✓ default=default/gpt-5; fast=default/gpt-5-mini" in result.output
    assert http_urls == ["https://flag.example"]
    assert workspace_urls == []


def test_doctor_reports_healthy_profile(app: typer.Typer, profile_tree: Path, monkeypatch) -> None:
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "Profile\n  ✓ profile for agent 'flight-planner'" in result.output
    assert "Models\n  ✓ default=default/gpt-5; fast=default/gpt-5-mini" in result.output
