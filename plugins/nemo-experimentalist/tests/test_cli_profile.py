# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI profile-driven behavior for experiment."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_experimentalist_plugin import cli, preflight
from nemo_experimentalist_plugin.preflight import Probes
from nemo_insights_plugin.contracts.profile import DEFAULT_BASE_URL
from typer.testing import CliRunner

runner = CliRunner()


def quiet_probes() -> Probes:
    return Probes(
        run_cmd=lambda argv: (0, "ok"),
        http_ok=lambda url: True,
    )


def write_task_toml(profile_tree: Path) -> None:
    (profile_tree / "evals" / "task_template" / "task.toml").write_text('[task]\nname = "org/x__t"\n', encoding="utf-8")


@pytest.fixture(autouse=True)
def _quiet_preflight(monkeypatch):
    """Deterministic probes for every test; tests override with their own
    monkeypatch.setattr when they exercise specific probe behavior."""
    monkeypatch.setattr(cli, "_PREFLIGHT_PROBES", quiet_probes())


@dataclass
class FakePlatformClient:
    base_url: str
    closed: bool = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_platform_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "make_client", FakePlatformClient)


@pytest.fixture()
def app():
    return cli.ExperimentalistCLI().get_cli()


def write_harbor_wrapper(agent_dir: Path) -> None:
    (agent_dir / "harbor_wrapper.py").write_text("class WrappedAgent:\n    pass\n", encoding="utf-8")


@pytest.fixture()
def profile_tree(tmp_path: Path) -> Path:
    (tmp_path / "evals" / "task_template").mkdir(parents=True)
    for split in ("train", "val"):
        (tmp_path / "evals" / split / "task-1").mkdir(parents=True)
        (tmp_path / "evals" / split / "task-1" / "task.toml").write_text("", encoding="utf-8")
    write_harbor_wrapper(tmp_path)
    (tmp_path / "optimizer.yaml").write_text(
        "agent: flight-planner\n"
        "task_template: ./evals/task_template\n"
        "datasets:\n  train: ./evals/train\n  validation: ./evals/val\n",
        encoding="utf-8",
    )
    return tmp_path


class RunRecorder:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    @property
    def call_kwargs(self) -> dict[str, Any]:
        assert self.kwargs is not None
        return self.kwargs

    async def __call__(self, **kwargs) -> str:
        self.kwargs = kwargs
        return "ok"


def test_experiment_insight_only_with_profile(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.setattr(cli, "_PREFLIGHT_PROBES", quiet_probes())
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "--insight", "ins-1", "-o", str(profile_tree / "out")])
    assert result.exit_code == 0, result.output
    assert recorder.call_kwargs["insight"] == "ins-1"
    assert recorder.call_kwargs["train_dataset"].uri == str((profile_tree / "evals" / "train").resolve())
    assert recorder.call_kwargs["agent"] == str(profile_tree.resolve())


def test_experiment_explicit_profile_flag(app, profile_tree: Path, monkeypatch, tmp_path: Path) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.setattr(cli, "_PREFLIGHT_PROBES", quiet_probes())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = runner.invoke(
        app,
        [
            "run",
            "--insight",
            "ins-1",
            "--profile",
            str(profile_tree / "optimizer.yaml"),
            "-o",
            str(elsewhere / "o"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert recorder.call_kwargs["workspace"] == "default"


def test_experiment_no_profile_missing_flags_errors_with_skeleton(app, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", "--insight", "ins-1", "-o", str(tmp_path / "o")])
    assert result.exit_code == 1
    assert "optimizer.yaml" in result.output
    assert "--train-dataset" in result.output


def test_experiment_all_flags_no_profile_still_works(app, tmp_path: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.setattr(cli, "_PREFLIGHT_PROBES", quiet_probes())
    for sub in ("agent", "t", "v"):
        (tmp_path / sub).mkdir()
    write_harbor_wrapper(tmp_path / "agent")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "--agent",
            str(tmp_path / "agent"),
            "--train-dataset",
            str(tmp_path / "t"),
            "--validation-dataset",
            str(tmp_path / "v"),
            "-o",
            str(tmp_path / "o"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert recorder.call_kwargs["insight"] is None


def test_profileless_experiment_blocks_missing_effective_agent_path(app, tmp_path: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    for sub in ("train", "validation"):
        (tmp_path / sub).mkdir()
    missing_agent = tmp_path / "missing-agent"
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--agent",
            str(missing_agent),
            "--train-dataset",
            str(tmp_path / "train"),
            "--validation-dataset",
            str(tmp_path / "validation"),
            "-o",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert recorder.kwargs is None
    assert f"agent source dir missing: {missing_agent}" in result.output


def test_profileless_experiment_blocks_missing_effective_task_template(app, tmp_path: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    for sub in ("agent", "train", "validation"):
        (tmp_path / sub).mkdir()
    missing_template = tmp_path / "missing-template"
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--agent",
            str(tmp_path / "agent"),
            "--insight",
            "ins-1",
            "--train-dataset",
            str(tmp_path / "train"),
            "--validation-dataset",
            str(tmp_path / "validation"),
            "--task-template",
            str(missing_template),
            "-o",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert recorder.kwargs is None
    assert f"task_template dir missing: {missing_template}" in result.output


def test_profileless_implicit_experiment_dirs_retry_collisions_and_stay_unique(
    app, tmp_path: Path, monkeypatch
) -> None:
    fixed_now = cli.datetime(2026, 7, 14, 12, 34, 56, tzinfo=cli.UTC)

    class FrozenDateTime:
        @staticmethod
        def now(*, tz):
            assert tz is cli.UTC
            return fixed_now

    collision_hex = "a" * 32
    first_hex = "b" * 32
    second_hex = "c" * 32
    generated_uuids = iter(
        [
            cli.uuid.UUID(hex=collision_hex),
            cli.uuid.UUID(hex=collision_hex),
            cli.uuid.UUID(hex=first_hex),
            cli.uuid.UUID(hex=first_hex),
            cli.uuid.UUID(hex=second_hex),
        ]
    )
    output_root = tmp_path / "tmp"
    colliding_dir = output_root / f"20260714-123456-{collision_hex}"
    colliding_dir.mkdir(parents=True)

    recorder = RunRecorder()
    monkeypatch.setattr(cli, "datetime", FrozenDateTime)
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: next(generated_uuids))
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    for sub in ("agent", "t", "v"):
        (tmp_path / sub).mkdir()
    write_harbor_wrapper(tmp_path / "agent")
    monkeypatch.chdir(tmp_path)
    args = [
        "run",
        "--agent",
        str(tmp_path / "agent"),
        "--train-dataset",
        str(tmp_path / "t"),
        "--validation-dataset",
        str(tmp_path / "v"),
    ]

    first_result = runner.invoke(app, args)
    first_dir = Path(recorder.call_kwargs["experiment_dir"])
    second_result = runner.invoke(app, args)
    second_dir = Path(recorder.call_kwargs["experiment_dir"])

    expected_first = Path("tmp") / f"20260714-123456-{first_hex}"
    expected_second = Path("tmp") / f"20260714-123456-{second_hex}"
    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert first_dir == expected_first
    assert second_dir == expected_second
    assert first_dir != second_dir
    assert (tmp_path / first_dir).is_dir()
    assert (tmp_path / second_dir).is_dir()
    assert f"Experiment dir: {expected_first}" in first_result.output
    assert f"Experiment dir: {expected_second}" in second_result.output


def test_experiment_insight_id_selects_from_analyst_file(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.setattr(cli, "_PREFLIGHT_PROBES", quiet_probes())
    insights = profile_tree / "insights.yaml"
    items = [
        {"id": "i-a", "title": "A", "description": "d", "agent": "flight-planner", "status": "open", "trace_refs": []},
        {"id": "i-b", "title": "B", "description": "d", "agent": "flight-planner", "status": "open", "trace_refs": []},
    ]
    insights.write_text(json.dumps({"insights": items}), encoding="utf-8")
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(
        app,
        ["run", "--insight", str(insights), "--insight-id", "i-b", "-o", str(profile_tree / "o")],
    )
    assert result.exit_code == 0, result.output
    selected = json.loads(Path(recorder.call_kwargs["insight"]).read_text(encoding="utf-8"))
    assert selected["id"] == "i-b"


def test_doctor_healthy_exits_zero(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    monkeypatch.setattr(cli, "_PREFLIGHT_PROBES", quiet_probes())
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "✓" in result.output


def test_doctor_fails_when_run_client_bootstrap_fails(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)

    def fail_client(_base_url: str) -> FakePlatformClient:
        raise httpx.UnsupportedProtocol("invalid cached auth context")

    monkeypatch.setattr(cli, "make_client", fail_client)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor", "--base-url", "https://platform.example"])

    assert result.exit_code == 1
    assert "Platform client initialization failed (UnsupportedProtocol)" in result.output
    assert "invalid cached auth context" not in result.output


def test_doctor_bootstraps_and_closes_run_client(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    clients: list[FakePlatformClient] = []

    def record_client(base_url: str) -> FakePlatformClient:
        client = FakePlatformClient(base_url)
        clients.append(client)
        return client

    monkeypatch.setattr(cli, "make_client", record_client)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor", "--base-url", "https://platform.example"])

    assert result.exit_code == 0, result.output
    assert clients == [FakePlatformClient("https://platform.example", closed=True)]
    assert "Platform client initialized with the effective authentication path" in result.output


def test_doctor_no_profile_exits_one_with_skeleton(app, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_PREFLIGHT_PROBES", quiet_probes())
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "optimizer.yaml" in result.output


def test_doctor_invalid_utf8_profile_is_structured_without_traceback(
    app,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "optimizer.yaml").write_bytes(b"agent: \xff\n")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Profile\n  ✗ Could not parse profile" in result.output
    assert "readable UTF-8 YAML" in result.output
    assert "Traceback" not in result.output


def test_doctor_invalid_utf8_task_toml_is_structured_without_traceback(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (profile_tree / "evals" / "task_template" / "task.toml").write_bytes(b"[task]\nname = \xff\n")
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor", "--insight", "ins-1"])

    assert result.exit_code == 1
    assert "Artifacts\n" in result.output
    assert "✗ task.toml unreadable or does not parse" in result.output
    assert "readable UTF-8 TOML" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "fragment",
    ["pkg//agent", ":(top,glob)**", "../outside"],
    ids=["malformed", "pathspec", "traversal"],
)
def test_doctor_invalid_git_agent_path_is_structured_without_traceback(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    fragment: str,
) -> None:
    profile = profile_tree / "optimizer.yaml"
    profile.write_text(
        profile.read_text(encoding="utf-8") + f'agent_source: "https://host/g/repo.git#{fragment}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "✗ agent path must be a normalized relative POSIX path" in result.output
    assert fragment not in result.output
    assert "Traceback" not in result.output


def test_doctor_redacts_platform_display_url(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    platform_base = "https://platform-user:platform-secret@platform.example:9443/api"  # trufflehog:ignore
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor", "--base-url", platform_base])

    assert result.exit_code == 0, result.output
    assert "https://***@platform.example:9443/api reachable" in result.output
    assert not any(secret in result.output for secret in ("platform-user", "platform-secret"))


def test_experiment_hard_fails_on_required_check(app, profile_tree: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        Probes(
            run_cmd=lambda argv: (1, "docker down"),
            http_ok=lambda url: True,
        ),
    )
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "--insight", "ins-1", "-o", str(profile_tree / "o")])
    assert result.exit_code == 1
    assert recorder.kwargs is None  # run never started
    assert "✗" in result.output


def test_experiment_advisory_warns_but_runs(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        Probes(
            run_cmd=lambda argv: (0, "ok"),
            http_ok=lambda url: False,  # platform unreachable → advisory
        ),
    )
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "--insight", "ins-1", "-o", str(profile_tree / "o")])
    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None  # run proceeded


def test_experiment_effective_storage_requires_git_before_runner(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)

    def missing_git(argv: list[str]) -> tuple[int, str]:
        if argv == ["git", "--version"]:
            return 127, "git not found"
        return 0, "ok"

    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        Probes(
            run_cmd=missing_git,
            http_ok=lambda url: True,
        ),
    )
    config = profile_tree / "experiment.yaml"
    config.write_text("storage:\n  archive_candidates: true\n", encoding="utf-8")
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(
        app,
        ["run", "--no-insight", "--config", str(config), "-o", str(profile_tree / "o")],
    )

    assert result.exit_code == 1
    assert "'git' is not available" in result.output
    assert recorder.kwargs is None


def test_discovered_profile_is_announced(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "--insight", "ins-1", "-o", str(profile_tree / "out")])
    assert result.exit_code == 0, result.output
    assert "Using profile:" in result.output
    assert "flight-planner" in result.output


def test_explicit_profile_is_not_announced(app, profile_tree: Path, monkeypatch, tmp_path: Path) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    elsewhere = tmp_path / "elsewhere2"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = runner.invoke(
        app,
        [
            "run",
            "--insight",
            "ins-1",
            "--profile",
            str(profile_tree / "optimizer.yaml"),
            "-o",
            str(elsewhere / "o"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Using profile:" not in result.output


def test_missing_models_reported_before_any_resolution(app, profile_tree: Path, monkeypatch) -> None:
    # Phase-1 ordering: the grouped model report must surface before resolution.
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        Probes(run_cmd=lambda argv: (0, "ok"), http_ok=lambda url: True),
    )
    monkeypatch.setattr(
        preflight,
        "configured_model_refs",
        lambda: (_ for _ in ()).throw(ValueError("No default model is configured")),
    )
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "--insight", "ins-1", "-o", str(profile_tree / "o")])
    assert result.exit_code == 1
    assert recorder.kwargs is None
    assert "No default model is configured" in result.output
    assert "nemo setup" in result.output


def test_insightless_run_skips_template_checks(app, profile_tree: Path, monkeypatch) -> None:
    # profile_tree's template dir has no task.toml — a Mode 2 (no insight) run
    # never reads the template, so preflight must not fail on it.
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "-o", str(profile_tree / "o")])
    assert result.exit_code == 0, result.output
    assert recorder.kwargs is not None
    assert recorder.call_kwargs["insight"] is None


def test_empty_config_flag_is_an_error_not_silent_fallback(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    empty = profile_tree / "empty.yaml"
    empty.write_text("# comments only\n", encoding="utf-8")
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "--insight", "ins-1", "--config", str(empty), "-o", str(profile_tree / "o")])
    assert result.exit_code == 1
    assert "is empty" in result.output
    assert recorder.kwargs is None


def test_doctor_multi_insight_file_requires_selector(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    insights = profile_tree / "insights.yaml"
    items = [
        {"id": "i-a", "title": "A", "agent": "flight-planner"},
        {"id": "i-b", "title": "B", "agent": "flight-planner"},
    ]
    insights.write_text(json.dumps({"insights": items}), encoding="utf-8")
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["doctor", "--insight", str(insights)])
    assert result.exit_code == 1
    assert "--insight-id" in result.output


def test_doctor_multi_insight_agent_mismatch_fails(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    insights = profile_tree / "insights.yaml"
    items = [{"id": "i-a", "title": "A", "agent": "someone-else"}]
    insights.write_text(json.dumps({"insights": items}), encoding="utf-8")
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["doctor", "--insight", str(insights)])
    assert result.exit_code == 1
    assert "someone-else" in result.output


def test_profile_dir_env_file_is_loaded_for_experiment(app, profile_tree: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.delenv("NEMO_OPT_TEST_DOTENV", raising=False)
    (profile_tree / ".env").write_text("NEMO_OPT_TEST_DOTENV=from-env-file\n", encoding="utf-8")
    monkeypatch.chdir(profile_tree)

    try:
        result = runner.invoke(app, ["run", "-o", str(profile_tree / "out")])
        assert result.exit_code == 0, result.output
        assert "Loaded .env" in result.output
        assert os.environ["NEMO_OPT_TEST_DOTENV"] == "from-env-file"
    finally:
        os.environ.pop("NEMO_OPT_TEST_DOTENV", None)


def test_experiment_loads_nhx_base_url_from_profile_env(app, profile_tree: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.delenv("NHX_BASE_URL", raising=False)
    (profile_tree / ".env").write_text(
        "NHX_BASE_URL=https://platform.example\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(
        app,
        ["run", "--no-insight", "-o", str(profile_tree / "out")],
    )

    assert result.exit_code == 0, result.output
    client = recorder.call_kwargs["client"]
    assert client.base_url == "https://platform.example"
    assert client.closed


def test_doctor_loads_nhx_base_url_from_profile_env(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    urls: list[str] = []

    def record_http(url: str) -> bool:
        urls.append(url)
        return True

    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        Probes(
            run_cmd=lambda argv: (0, "ok"),
            http_ok=record_http,
        ),
    )
    monkeypatch.delenv("NHX_BASE_URL", raising=False)
    (profile_tree / ".env").write_text(
        "NHX_BASE_URL=https://platform.example\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "https://platform.example/health/ready" in urls


def test_default_experiment_dir_is_profile_scoped_and_announced(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "--insight", "ins-1"])
    assert result.exit_code == 0, result.output
    assert "Experiment dir:" in result.output
    experiment_dir = Path(recorder.call_kwargs["experiment_dir"])
    assert experiment_dir.is_relative_to(profile_tree / ".nemo-optimizer" / "experiments")


def test_default_experiment_dir_retries_collision_and_reserves(app, profile_tree: Path, monkeypatch) -> None:
    fixed_now = cli.datetime(2026, 7, 14, 12, 34, 56, tzinfo=cli.UTC)

    class FrozenDateTime:
        @staticmethod
        def now(*, tz):
            assert tz is cli.UTC
            return fixed_now

    collision_hex = "a" * 32
    unique_hex = "b" * 32
    generated_uuids = iter(
        [
            cli.uuid.UUID(hex=collision_hex),
            cli.uuid.UUID(hex=collision_hex),
            cli.uuid.UUID(hex=unique_hex),
        ]
    )
    experiments_root = profile_tree / ".nemo-optimizer" / "experiments"
    colliding_dir = experiments_root / f"20260714-123456-{collision_hex}"
    colliding_dir.mkdir(parents=True)

    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "datetime", FrozenDateTime)
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: next(generated_uuids))
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run", "--insight", "ins-1"])

    expected = experiments_root / f"20260714-123456-{unique_hex}"
    assert result.exit_code == 0, result.output
    assert Path(recorder.call_kwargs["experiment_dir"]) == expected
    assert expected.is_dir()
    assert colliding_dir.is_dir()


def test_default_experiment_dir_is_not_reserved_before_required_preflight(app, profile_tree: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        Probes(
            run_cmd=lambda argv: (1, "docker down"),
            http_ok=lambda url: True,
        ),
    )
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["run", "--insight", "ins-1"])

    assert result.exit_code == 1
    assert recorder.kwargs is None
    assert not (profile_tree / ".nemo-optimizer" / "experiments").exists()


def test_explicit_experiment_dir_still_wins(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "--insight", "ins-1", "-o", str(profile_tree / "custom")])
    assert result.exit_code == 0, result.output
    assert "Experiment dir:" not in result.output
    assert Path(recorder.call_kwargs["experiment_dir"]) == profile_tree / "custom"


def test_experiment_defaults_to_shared_insights_file(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    shared = profile_tree / ".nemo-optimizer" / "insights.yaml"
    shared.parent.mkdir(parents=True)
    shared.write_text(
        json.dumps({"insights": [{"id": "i-1", "title": "T", "agent": "flight-planner", "trace_refs": []}]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "-o", str(profile_tree / "o")])
    assert result.exit_code == 0, result.output
    assert "Insight file:" in result.output
    selected = json.loads(Path(recorder.call_kwargs["insight"]).read_text(encoding="utf-8"))
    assert selected["id"] == "i-1"


def test_experiment_selected_insight_uses_shared_default_without_stale_warning(
    app, profile_tree: Path, monkeypatch
) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    shared = profile_tree / ".nemo-optimizer" / "insights.yaml"
    shared.parent.mkdir(parents=True)
    shared.write_text(
        json.dumps(
            {
                "insights": [
                    {"id": "i-a", "title": "A", "agent": "flight-planner"},
                    {"id": "i-b", "title": "B", "agent": "flight-planner"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(
        app,
        ["run", "--insight-id", "1", "-o", str(profile_tree / "o")],
    )
    assert result.exit_code == 0, result.output
    selected = json.loads(Path(recorder.call_kwargs["insight"]).read_text(encoding="utf-8"))
    assert selected["id"] == "i-b"
    assert "will require --insight-id" not in result.output


def test_experiment_insight_id_requires_resolved_insight(app, profile_tree: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(
        app,
        ["run", "--insight-id", "0", "-o", str(profile_tree / "o")],
    )
    assert result.exit_code == 1
    assert "--insight-id" in result.output
    assert "local multi-insight" in result.output
    assert recorder.kwargs is None


def test_experiment_insight_id_rejects_platform_id(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(
        app,
        [
            "run",
            "--insight",
            "ins-platform",
            "--insight-id",
            "0",
            "-o",
            str(profile_tree / "o"),
        ],
    )
    assert result.exit_code == 1
    assert "local multi-insight" in result.output
    assert recorder.kwargs is None


def test_experiment_insight_id_matching_single_local_insight_passes(app, profile_tree: Path, monkeypatch) -> None:
    write_task_toml(profile_tree)
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    single = profile_tree / "insights.yaml"
    single.write_text(
        json.dumps({"insights": [{"id": "i-1", "title": "T", "agent": "flight-planner"}]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(
        app,
        [
            "run",
            "--insight",
            str(single),
            "--insight-id",
            "i-1",
            "-o",
            str(profile_tree / "o"),
        ],
    )
    assert result.exit_code == 0, result.output
    selected = json.loads(Path(recorder.call_kwargs["insight"]).read_text(encoding="utf-8"))
    assert selected["id"] == "i-1"


def test_experiment_no_insight_skips_existing_shared_default(app, profile_tree: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    shared = profile_tree / ".nemo-optimizer" / "insights.yaml"
    shared.parent.mkdir(parents=True)
    shared.write_text(
        json.dumps({"insights": [{"id": "i-1", "title": "T", "agent": "flight-planner"}]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(
        app,
        ["run", "--no-insight", "-o", str(profile_tree / "o")],
    )
    assert result.exit_code == 0, result.output
    assert recorder.call_kwargs["insight"] is None
    assert "Insight disabled" in result.output
    assert "Insight file:" not in result.output


@pytest.mark.parametrize(
    "insight_args",
    [
        ["--no-insight", "--insight", "ins-1"],
        ["--no-insight", "--insight-id", "0"],
    ],
)
def test_experiment_insight_conflict_is_clean(
    app,
    profile_tree: Path,
    monkeypatch,
    insight_args: list[str],
) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(
        app,
        ["run", *insight_args, "-o", str(profile_tree / "o")],
    )
    assert result.exit_code == 1
    assert "--no-insight cannot be combined" in result.output
    assert recorder.kwargs is None


def test_experiment_insight_id_help_documents_zero_based_index(app) -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    help_text = " ".join(result.output.replace("│", " ").split())
    assert "exact id/title first" in help_text
    assert "zero-based index" in help_text
    assert "local multi-insight" in help_text


def test_experiment_mismatched_insight_blocked_before_resolution(app, profile_tree: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    bad = profile_tree / "insight.json"
    bad.write_text(json.dumps({"id": "i-x", "title": "T", "agent": "someone-else"}), encoding="utf-8")
    monkeypatch.chdir(profile_tree)
    result = runner.invoke(app, ["run", "--insight", str(bad), "-o", str(profile_tree / "o")])
    assert result.exit_code == 1
    assert "someone-else" in result.output
    assert recorder.kwargs is None  # blocked in phase-1 preflight, before resolution


def test_profileless_mode2_without_agent_stops_before_runner(app, tmp_path: Path, monkeypatch) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    for sub in ("train", "validation"):
        (tmp_path / sub).mkdir()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--train-dataset",
            str(tmp_path / "train"),
            "--validation-dataset",
            str(tmp_path / "validation"),
            "-o",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "--agent" in result.output
    assert recorder.kwargs is None


def test_doctor_validates_implicit_shared_insight_and_selector(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_task_toml(profile_tree)
    shared = profile_tree / ".nemo-optimizer" / "insights.yaml"
    shared.parent.mkdir(parents=True)
    shared.write_text(
        json.dumps(
            {
                "insights": [
                    {"id": "i-a", "title": "A", "agent": "flight-planner"},
                    {"id": "i-b", "title": "B", "agent": "flight-planner"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(profile_tree)

    unselected = runner.invoke(app, ["doctor"])
    selected = runner.invoke(app, ["doctor", "--insight-id", "i-b"])

    assert unselected.exit_code == 1
    assert "pass --insight-id" in unselected.output
    assert selected.exit_code == 0, selected.output
    assert "2 insights" in selected.output


def test_doctor_validates_implicit_shared_insight_agent(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_task_toml(profile_tree)
    shared = profile_tree / ".nemo-optimizer" / "insights.yaml"
    shared.parent.mkdir(parents=True)
    shared.write_text(
        json.dumps({"insights": [{"id": "i-x", "title": "Other", "agent": "someone-else"}]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "someone-else" in result.output


def test_doctor_validates_effective_ethos_readability(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_task_toml(profile_tree)
    ethos = profile_tree / "ETHOS.md"
    ethos.write_bytes(b"\xff")
    with (profile_tree / "optimizer.yaml").open("a", encoding="utf-8") as profile_file:
        profile_file.write("ethos: ./ETHOS.md\n")
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Could not read ethos" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("config_body", "expected"),
    [
        (None, "Could not read experiment_config"),
        ("storage: [\n", "Could not read experiment_config"),
        ("max_rounds: banana\n", "Invalid experiment config"),
    ],
    ids=["missing", "malformed-yaml", "invalid-value"],
)
def test_doctor_validates_full_effective_experiment_config(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_body: str | None,
    expected: str,
) -> None:
    write_task_toml(profile_tree)
    with (profile_tree / "optimizer.yaml").open("a", encoding="utf-8") as profile_file:
        profile_file.write("experiment_config: ./experiment.yaml\n")
    if config_body is not None:
        (profile_tree / "experiment.yaml").write_text(config_body, encoding="utf-8")
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert expected in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["run", "doctor"])
def test_missing_evaluator_entrypoint_blocks_both_commands(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """Doctor reports the same missing wrapper that stops a run, not a clean bill."""
    write_task_toml(profile_tree)
    (profile_tree / "harbor_wrapper.py").unlink()
    monkeypatch.setattr(cli, "run_experimentalist", RunRecorder())
    monkeypatch.chdir(profile_tree)
    args = [command]
    if command == "run":
        args += ["--no-insight", "-o", str(profile_tree / "out")]

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert "evaluator cannot import 'harbor_wrapper'" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["run", "doctor"])
def test_env_file_encoding_failure_is_clean_and_actionable(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    write_task_toml(profile_tree)
    (profile_tree / ".env").write_bytes(b"NHX_BASE_URL=\xff")
    monkeypatch.chdir(profile_tree)
    args = [command]
    if command == "run":
        args += ["--no-insight", "-o", str(profile_tree / "out")]

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert "Could not read environment file" in result.output
    assert "UTF-8" in result.output
    assert "Traceback" not in result.output


def test_doctor_env_file_permission_failure_is_required_and_actionable(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_task_toml(profile_tree)
    env_file = profile_tree / ".env"
    env_file.write_text("NHX_BASE_URL=https://profile.example\n", encoding="utf-8")
    original_read_text = Path.read_text

    def deny_env_file(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if path == env_file:
            raise PermissionError("permission denied by test")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", deny_env_file)
    monkeypatch.chdir(profile_tree)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Could not read environment file" in result.output
    assert "permission denied" in result.output
    assert "check that the file is readable UTF-8" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("explicit", "shell", "profile_env", "expected"),
    [
        ("https://explicit.example", "https://shell.example", "https://profile.example", "https://explicit.example"),
        (None, "https://shell.example", "https://profile.example", "https://shell.example"),
        (None, None, "https://profile.example", "https://profile.example"),
        (None, None, None, DEFAULT_BASE_URL),
    ],
    ids=["explicit", "shell", "profile-env", "default-ignores-nemo"],
)
def test_experiment_nhx_base_url_precedence_and_ignores_nemo_base_url(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit: str | None,
    shell: str | None,
    profile_env: str | None,
    expected: str,
) -> None:
    recorder = RunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", recorder)
    monkeypatch.setenv("NEMO_BASE_URL", "https://legacy-must-be-ignored.example")
    if shell is None:
        monkeypatch.delenv("NHX_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("NHX_BASE_URL", shell)
    if profile_env is not None:
        (profile_tree / ".env").write_text(f"NHX_BASE_URL={profile_env}\n", encoding="utf-8")
    monkeypatch.chdir(profile_tree)
    args = ["run", "--no-insight", "-o", str(profile_tree / "out")]
    if explicit is not None:
        args += ["--base-url", explicit]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    client = recorder.call_kwargs["client"]
    assert client.base_url == expected
    assert client.closed


@pytest.mark.parametrize(
    ("explicit", "shell", "profile_env", "expected"),
    [
        ("https://explicit.example", "https://shell.example", "https://profile.example", "https://explicit.example"),
        (None, "https://shell.example", "https://profile.example", "https://shell.example"),
        (None, None, "https://profile.example", "https://profile.example"),
        (None, None, None, DEFAULT_BASE_URL),
    ],
    ids=["explicit", "shell", "profile-env", "default-ignores-nemo"],
)
def test_doctor_nhx_base_url_precedence_and_ignores_nemo_base_url(
    app,
    profile_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit: str | None,
    shell: str | None,
    profile_env: str | None,
    expected: str,
) -> None:
    write_task_toml(profile_tree)
    urls: list[str] = []
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        Probes(
            run_cmd=lambda argv: (0, "ok"),
            http_ok=lambda url: not urls.append(url),
        ),
    )
    monkeypatch.setenv("NEMO_BASE_URL", "https://legacy-must-be-ignored.example")
    if shell is None:
        monkeypatch.delenv("NHX_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("NHX_BASE_URL", shell)
    if profile_env is not None:
        (profile_tree / ".env").write_text(f"NHX_BASE_URL={profile_env}\n", encoding="utf-8")
    monkeypatch.chdir(profile_tree)
    args = ["doctor"]
    if explicit is not None:
        args += ["--base-url", explicit]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert urls == [f"{expected}/health/ready"]


def test_experiment_help_names_insights_writer(app) -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0, result.output
    help_text = " ".join(result.output.replace("│", " ").split())
    assert "nemo agents analyst run" in help_text
    assert "writes by default" in help_text
