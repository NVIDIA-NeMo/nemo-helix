# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from click.testing import Result
from nemo_experimentalist_plugin import cli
from nemo_experimentalist_plugin.entities import DatasetRef
from nemo_experimentalist_plugin.experimentalist.strategies.evolutionary import EvolutionaryOptimizerConfig
from nemo_experimentalist_plugin.preflight import Probes
from nemo_platform_plugin.client.client import AsyncNemoClient
from nooa import GenerationError
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def quiet_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-preflight runs inside the experiment flow; make it pass deterministically.

    These tests pin the CLI→runner contract, not preflight behavior (that lives
    in test_cli_profile.py / test_preflight.py), so probes never hit the system.
    """
    monkeypatch.setattr(
        cli,
        "_PREFLIGHT_PROBES",
        Probes(
            run_cmd=lambda argv: (0, "ok"),
            http_ok=lambda url: True,
        ),
    )


@pytest.fixture(autouse=True)
def hermetic_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run from an isolated directory: profile discovery walks up from cwd, so
    an optimizer.yaml in any ancestor of the checkout must never leak in."""
    monkeypatch.chdir(tmp_path)


class FakePlatformClient(AsyncNemoClient):
    def __init__(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(500, request=request, json={"detail": "fake client"})
        )
        super().__init__(
            base_url="http://platform.test",
            workspace="default",
            http_client=httpx.AsyncClient(transport=transport),
        )
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        await super().close()


@pytest.fixture(autouse=True)
def platform_client(monkeypatch: pytest.MonkeyPatch) -> FakePlatformClient:
    client = FakePlatformClient()
    monkeypatch.setattr(cli, "make_client", lambda _base_url: client)
    return client


@dataclass
class CapturedExperimentRun:
    agent: str | Path
    train_dataset: DatasetRef
    validation_dataset: DatasetRef
    task_template: DatasetRef | None
    experiment_dir: Path
    workspace: str
    client: AsyncNemoClient
    config: EvolutionaryOptimizerConfig
    insight: Path | str | None
    ethos: str | None
    framework_skills_dirs: list[Path] | None


@dataclass
class ExperimentCliPaths:
    agent: Path
    train: Path
    validation: Path
    template: Path
    experiment: Path

    def args(self) -> list[str]:
        return [
            "--agent",
            str(self.agent),
            "--train-dataset",
            str(self.train),
            "--validation-dataset",
            str(self.validation),
            "--task-template",
            str(self.template),
            "--experiment-dir",
            str(self.experiment),
        ]


class ExperimentRunRecorder:
    def __init__(self, result: str = "experiment-summary") -> None:
        self.captured: CapturedExperimentRun | None = None
        self.result = result

    async def __call__(
        self,
        *,
        agent: str | Path,
        train_dataset: DatasetRef,
        validation_dataset: DatasetRef,
        experiment_dir: Path,
        workspace: str,
        client: AsyncNemoClient,
        config: EvolutionaryOptimizerConfig,
        task_template: DatasetRef | None = None,
        insight: Path | str | None = None,
        ethos: str | None = None,
        framework_skills_dirs: list[Path] | None = None,
    ) -> str:
        self.captured = CapturedExperimentRun(
            agent=agent,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            task_template=task_template,
            experiment_dir=experiment_dir,
            workspace=workspace,
            client=client,
            config=config,
            insight=insight,
            ethos=ethos,
            framework_skills_dirs=framework_skills_dirs,
        )
        return self.result


def _make_dir(path: Path) -> Path:
    path.mkdir()
    return path


def _make_paths(tmp_path: Path) -> ExperimentCliPaths:
    template = _make_dir(tmp_path / "template")
    (template / "task.toml").write_text('[task]\nname = "org/test-template"\n', encoding="utf-8")
    (template / "instruction.md").write_text("Test task instructions.\n", encoding="utf-8")
    agent = _make_dir(tmp_path / "agent")
    (agent / "harbor_wrapper.py").write_text("class WrappedAgent:\n    pass\n", encoding="utf-8")
    return ExperimentCliPaths(
        agent=agent,
        train=_make_dir(tmp_path / "train"),
        validation=_make_dir(tmp_path / "validation"),
        template=template,
        experiment=tmp_path / "experiment",
    )


def _run_experiment(args: list[str]) -> Result:
    app = cli.ExperimentalistCLI().get_cli()
    return CliRunner().invoke(app, ["run", *args])


async def _fail_if_runner_starts(**_: object) -> str:
    raise AssertionError("invalid CLI input must not start the experimentalist runner")


def test_cli_help_exposes_only_run_and_doctor() -> None:
    app = cli.ExperimentalistCLI().get_cli()
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "doctor" in result.output
    assert "analyze" not in result.output
    assert "analysis" not in result.output


@pytest.mark.parametrize(
    ("config_body", "expected_config", "expected_output"),
    [
        pytest.param(
            "max_rounds: 2\noutcome_evaluator_config:\n  max_attempts: 3\n",
            EvolutionaryOptimizerConfig.model_validate(
                {"max_rounds": 2, "outcome_evaluator_config": {"max_attempts": 3}}
            ),
            "configured-summary",
            id="with-config",
        ),
        pytest.param(
            None,
            EvolutionaryOptimizerConfig(),
            "default-config-summary",
            id="without-config",
        ),
    ],
)
def test_experiment_cli_passes_dataset_driven_contract_to_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_client: FakePlatformClient,
    config_body: str | None,
    expected_config: EvolutionaryOptimizerConfig,
    expected_output: str,
) -> None:
    paths = _make_paths(tmp_path)
    insight_file = tmp_path / "insight.json"
    insight_file.write_text("{}", encoding="utf-8")
    runner = ExperimentRunRecorder(result=expected_output)
    monkeypatch.setattr(cli, "run_experimentalist", runner)
    args = [
        *paths.args(),
        "--workspace",
        "workspace-a",
        "--base-url",
        "http://platform.test",
        "--insight",
        str(insight_file),
    ]

    if config_body is not None:
        # NOT named optimizer.yaml: that would collide with profile discovery
        # from the hermetic cwd — a run config is not a profile.
        config_path = tmp_path / "run-config.yaml"
        config_path.write_text(config_body, encoding="utf-8")
        args.extend(["--config", str(config_path)])

    result = _run_experiment(args)

    assert result.exit_code == 0
    assert result.output.strip() == expected_output
    assert runner.captured is not None
    # The CLI passes --agent through verbatim (a str); resolution / git-cloning
    # happens inside the loop via backend.get_agent_code().
    assert runner.captured.agent == str(paths.agent)
    # A local --insight file is normalized to scratch JSON (the local backend
    # reads insight files with json.loads, so YAML-authored files must not
    # reach it raw); platform ids still pass through verbatim.
    assert runner.captured.insight == str(paths.experiment / "resolved" / "insight.json")
    captured_insight = runner.captured.insight
    assert captured_insight is not None
    assert Path(captured_insight).read_text(encoding="utf-8").strip() == "{}"
    # Datasets and the task template are forwarded as DatasetRef URI handles,
    # tagged with a stable id the evaluator adapter uses when building datasets.
    assert runner.captured.train_dataset == DatasetRef(uri=str(paths.train), metadata={"id": "train"})
    assert runner.captured.validation_dataset == DatasetRef(uri=str(paths.validation), metadata={"id": "validation"})
    assert runner.captured.task_template == DatasetRef(uri=str(paths.template), metadata={"id": "task-template"})
    assert runner.captured.experiment_dir == paths.experiment
    assert runner.captured.workspace == "workspace-a"
    assert runner.captured.client is platform_client
    assert platform_client.closed
    assert runner.captured.config == expected_config


@pytest.mark.parametrize(
    ("extra_args", "config_body", "expected_exit_code", "expected_errors"),
    [
        pytest.param(
            (),
            "max_rounds: [",
            1,
            ("while parsing a flow node",),
            id="malformed-yaml",
        ),
        pytest.param(
            (),
            "max_rounds: not-an-int\n",
            1,
            ("max_rounds", "Input should be a valid integer"),
            id="invalid-schema",
        ),
    ],
)
def test_experiment_cli_reports_expected_errors_without_starting_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: tuple[str, ...],
    config_body: str | None,
    expected_exit_code: int,
    expected_errors: tuple[str, ...],
) -> None:
    paths = _make_paths(tmp_path)
    args = [*paths.args(), *extra_args]

    if config_body is not None:
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(config_body, encoding="utf-8")
        args.extend(["--config", str(config_path)])

    monkeypatch.setattr(cli, "run_experimentalist", _fail_if_runner_starts)

    result = _run_experiment(args)

    assert result.exit_code == expected_exit_code
    for expected_error in expected_errors:
        assert expected_error in result.output


def test_experiment_cli_exits_nonzero_when_evaluation_has_no_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_client: FakePlatformClient,
) -> None:
    async def fail_no_scores(**_: object) -> str:
        raise ValueError("Evaluation agent-validation produced no scoreable metrics from 3 trial(s)")

    paths = _make_paths(tmp_path)
    monkeypatch.setattr(cli, "run_experimentalist", fail_no_scores)

    result = _run_experiment(paths.args())

    assert result.exit_code == 1
    assert "produced no scoreable metrics from 3 trial(s)" in result.output
    assert platform_client.closed


def test_experiment_cli_reports_a_failed_generation_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_client: FakePlatformClient,
) -> None:
    """A step the model cannot finish exits 1 and repeats what the harness reported."""

    async def fail_generation(**_: object) -> str:
        raise GenerationError(
            "return_result validation failed after 3 attempts.\nExpected: list[TrialSelection]\nGot: str"
        )

    paths = _make_paths(tmp_path)
    monkeypatch.setattr(cli, "run_experimentalist", fail_generation)

    result = _run_experiment(paths.args())

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Expected: list[TrialSelection]" in result.output
    assert platform_client.closed


def test_experiment_cli_forwards_insight_platform_id_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _make_paths(tmp_path)
    runner = ExperimentRunRecorder()
    monkeypatch.setattr(cli, "run_experimentalist", runner)
    # A non-file value (a platform insight id) is accepted by --insight and forwarded
    # verbatim; the backend routes it to the platform at get_insight time.
    args = [*paths.args(), "--insight", "insight-remote-123"]

    result = _run_experiment(args)

    assert result.exit_code == 0
    assert runner.captured is not None
    assert runner.captured.insight == "insight-remote-123"
