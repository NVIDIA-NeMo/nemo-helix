# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import stat
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("scaled_evals")

from scaled_evals import harbor_opensandbox_cleanup as cleanup
from scaled_evals.api.framework_versions import resolve_framework_runner
from scaled_evals.api.settings import settings
from scaled_evals.dispatch import harbor_opensandbox as backend
from scaled_evals.models.runtime import LaunchHandle, LaunchSpec

DIGEST = "sha256:" + "a" * 64
DEPLOYMENT = "dep-test"


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "harbor_opensandbox_deployment_id", DEPLOYMENT)
    monkeypatch.setattr(settings, "harbor_opensandbox_model_endpoint_hosts", "igw.example.internal")
    monkeypatch.setattr(settings, "harbor_opensandbox_allowed_hosts", "pypi.org, igw.example.internal")
    monkeypatch.setattr(settings, "harbor_opensandbox_protocol", "http")
    monkeypatch.setattr(settings, "harbor_opensandbox_egress_verification", "default_action")
    monkeypatch.setattr(settings, "harbor_opensandbox_cleanup_timeout_seconds", 30)
    monkeypatch.setattr(settings, "sandbox_k8s_task_image_reference_mode", "digest")


def _spec(**overrides: Any) -> LaunchSpec:
    values: dict[str, Any] = {
        "evaluation_id": "ev_os1",
        "benchmark_run_id": "br_1",
        "name": "oracle",
        "task_slug": "hello-world",
        "framework": "harbor",
        "framework_version": "0.20.0",
        "image_ref": "nvcr.io/org/task:v1",
        "image_digest": DIGEST,
        "parallelism": 2,
        "n_attempts": 3,
        "network_policy": "default_deny",
        "tarball_object_key": "packs/hello.tar.gz",
    }
    values.update(overrides)
    return LaunchSpec(**values)


def _write_task(task_dir: Path, environment: str = "") -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(f'version = "1.0"\n\n[environment]\n{environment}')
    (task_dir / "instruction.md").write_text("Say hello.\n")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"framework_version": "0.13.2"}, "requires Harbor 0.20.0"),
        ({"network_policy": "unrestricted"}, "supports network_policy default_deny"),
        ({"network_policy": "scoped_egress"}, "supports network_policy default_deny"),
        ({"network_policy_config": {"allowed_hosts": ["x"]}}, "omit network_policy_config"),
        ({"agent_bundle": {"object_key": "b"}}, "agent bundles"),
        ({"image_digest": None}, "recorded digest"),
        ({"tarball_object_key": None}, "uploaded task pack"),
        ({"framework": "gym"}, "Harbor evaluations only"),
    ],
)
def test_preflight_rejects_what_it_cannot_isolate(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        backend.preflight(_spec(**overrides))


def test_preflight_accepts_supported_evaluation() -> None:
    backend.preflight(_spec())


@pytest.mark.parametrize("compose", ["environment/docker-compose.yaml", "compose.yml"])
def test_staged_preflight_rejects_compose(tmp_path: Path, compose: str) -> None:
    _write_task(tmp_path)
    (tmp_path / compose).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / compose).write_text("services: {}\n")

    with pytest.raises(ValueError, match="single-container"):
        backend.preflight_staged_task(tmp_path, ["pypi.org"])


def test_staged_preflight_rejects_task_egress_outside_operator_allowlist(tmp_path: Path) -> None:
    _write_task(tmp_path, 'network_mode = "allowlist"\nallowed_hosts = ["pypi.org", "evil.example"]\n')

    with pytest.raises(ValueError, match="evil.example"):
        backend.preflight_staged_task(tmp_path, ["pypi.org"])


def test_staged_preflight_accepts_task_narrowing(tmp_path: Path) -> None:
    _write_task(tmp_path, 'network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n')

    backend.preflight_staged_task(tmp_path, ["pypi.org", "igw.example.internal"])


def test_trusted_hosts_put_model_endpoint_first_without_duplicates() -> None:
    assert backend.trusted_allowed_hosts() == ["igw.example.internal", "pypi.org"]


def test_render_owns_environment_block(tmp_path: Path) -> None:
    template = yaml.safe_dump(
        {
            "agents": [{"name": "oracle"}],
            "environment": {"type": "docker", "kwargs": {"use_server_proxy": True, "ready_timeout_sec": 60}},
            "tasks": [{"path": "/baked/task"}],
        }
    )

    config = backend.render_harbor_config(
        template, _spec(), task_path=tmp_path / "task", jobs_dir="jobs/os", trusted_hosts=["igw", "pypi.org"]
    )

    assert config["agents"] == [{"name": "oracle"}]
    assert config["job_name"] == "ev_os1"
    assert config["jobs_dir"] == "jobs/os"
    assert config["n_attempts"] == 3
    assert config["n_concurrent_trials"] == 2
    assert config["tasks"] == [{"path": str(tmp_path / "task")}]
    environment = config["environment"]
    assert environment["import_path"] == backend.NEMO_OPENSANDBOX_IMPORT_PATH
    assert "type" not in environment
    assert environment["delete"] is True
    assert environment["kwargs"] == {
        "use_server_proxy": True,
        "ready_timeout_sec": 60,
        "protocol": "http",
        "sandbox_timeout_sec": settings.harbor_opensandbox_sandbox_timeout_seconds,
        "trusted_allowed_hosts": ["igw", "pypi.org"],
        "egress_verification": "default_action",
        "metadata": {
            cleanup.DEPLOYMENT_METADATA_KEY: DEPLOYMENT,
            cleanup.EVALUATION_METADATA_KEY: "ev_os1",
            cleanup.BENCHMARK_RUN_METADATA_KEY: "br_1",
        },
    }


def test_shipped_oracle_template_renders(tmp_path: Path) -> None:
    template = (
        Path(__file__).parents[1] / "examples/tasks/hello-world/configs/harbor_oracle_opensandbox_runtime.yaml"
    ).read_text()

    config = backend.render_harbor_config(
        template, _spec(), task_path=tmp_path, jobs_dir="jobs/os", trusted_hosts=["igw"]
    )

    assert config["agents"] == [{"name": "oracle"}]
    assert config["environment"]["import_path"] == backend.NEMO_OPENSANDBOX_IMPORT_PATH
    assert config["environment"]["kwargs"]["use_server_proxy"] is True


@pytest.mark.parametrize("kwarg", ["api_key", "domain", "volumes", "trusted_allowed_hosts", "metadata"])
def test_render_rejects_security_kwargs_in_template(tmp_path: Path, kwarg: str) -> None:
    template = yaml.safe_dump({"environment": {"kwargs": {kwarg: "x"}}})

    with pytest.raises(ValueError, match=kwarg):
        backend.render_harbor_config(template, _spec(), task_path=tmp_path, jobs_dir="j", trusted_hosts=[])


@pytest.mark.parametrize("key", ["environment", "tasks", "datasets", "jobs_dir"])
def test_render_rejects_profile_owned_keys(tmp_path: Path, key: str) -> None:
    spec = _spec(harbor_config={"harbor_config": yaml.safe_dump({key: {"import_path": "evil:Env"}})})

    with pytest.raises(ValueError, match=key):
        backend.render_harbor_config("agents: []\n", spec, task_path=tmp_path, jobs_dir="j", trusted_hosts=[])


def test_render_merges_profile_agents_with_substitution(tmp_path: Path) -> None:
    profile = {"harbor_config": "agents:\n  - name: ${AGENT}\n", "env": {"AGENT": "claude-code"}}

    config = backend.render_harbor_config(
        "agents: [{name: oracle}]\n", _spec(harbor_config=profile), task_path=tmp_path, jobs_dir="j", trusted_hosts=[]
    )

    assert config["agents"] == [{"name": "claude-code"}]


def test_connection_env_aliases_platform_names_and_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "os.env"
    env_file.write_text("OPEN_SANDBOX_API_KEY=from-file\n")

    env = backend.connection_env({"OPEN_SANDBOX_DOMAIN": "os.svc"}, str(env_file))

    assert env["OPENSANDBOX_DOMAIN"] == "os.svc"
    assert env["OPENSANDBOX_API_KEY"] == "from-file"


def test_connection_env_requires_domain_and_key() -> None:
    with pytest.raises(RuntimeError, match="OPENSANDBOX_API_KEY"):
        backend.connection_env({"OPENSANDBOX_DOMAIN": "os.svc"}, None)


def _submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: LaunchSpec | None = None, task_env: str = ""):
    harbor_dir = tmp_path / "harbor"
    site = harbor_dir / ".venv" / "lib" / "python3.12" / "site-packages" / "opensandbox-0.1.16.dist-info"
    site.mkdir(parents=True)
    template = tmp_path / "template.yaml"
    template.write_text("agents: [{name: oracle}]\n")

    def fake_stage(_key: str, dest: Path) -> Path:
        _write_task(dest, task_env)
        return dest

    monkeypatch.setattr(backend, "_stage_task_tree", fake_stage)
    calls: list[tuple[list[str], Path, Path, Mapping[str, str]]] = []
    submit = backend.make_harbor_opensandbox_submitter(
        harbor_dir=str(harbor_dir),
        config_path=str(template),
        work_dir=str(tmp_path / "work"),
        jobs_dir="jobs/os",
        runner=lambda argv, cwd, log, env: calls.append((argv, cwd, log, env)),
        environ={"OPEN_SANDBOX_DOMAIN": "os.svc", "OPEN_SANDBOX_API_KEY": "secret-key", "PATH": "/bin"},
    )
    handle = submit(spec or _spec(credential_env={"NVIDIA_API_KEY": "nv-key"}))
    return handle, calls


def test_submit_stages_binds_renders_and_spawns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handle, calls = _submit(tmp_path, monkeypatch)

    assert len(calls) == 1
    argv, cwd, log_path, env = calls[0]
    assert cwd == tmp_path / "harbor"
    assert env["OPENSANDBOX_DOMAIN"] == "os.svc"
    assert env["OPENSANDBOX_API_KEY"] == "secret-key"
    assert "NVIDIA_API_KEY" not in env

    task_dir = tmp_path / "work" / "ev_os1" / "hello-world"
    task = tomllib.loads((task_dir / "task.toml").read_text())
    assert task["environment"]["docker_image"] == f"nvcr.io/org/task@{DIGEST}"

    rendered_path = Path(handle.raw["config"])
    rendered_text = rendered_path.read_text()
    assert "secret-key" not in rendered_text
    assert yaml.safe_load(rendered_text)["tasks"] == [{"path": str(task_dir)}]

    env_file = Path(argv[argv.index("--env-file") + 1])
    assert "NVIDIA_API_KEY=nv-key" in env_file.read_text()
    assert "secret-key" not in env_file.read_text()
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600

    assert handle.backend == "harbor_opensandbox"
    assert handle.external_id == "ev_os1"
    assert handle.raw["log"] == str(log_path)
    assert handle.raw["ownership"] == {
        cleanup.DEPLOYMENT_METADATA_KEY: DEPLOYMENT,
        cleanup.EVALUATION_METADATA_KEY: "ev_os1",
        cleanup.BENCHMARK_RUN_METADATA_KEY: "br_1",
    }
    assert handle.raw["provenance"]["opensandbox_sdk_version"] == "0.1.16"
    assert handle.raw["provenance"]["harbor_version"] == "0.20.0"
    assert handle.raw["provenance"]["task_image_digest"] == DIGEST


def test_submit_rejects_before_spawning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="evil.example"):
        _submit(tmp_path, monkeypatch, task_env='network_mode = "allowlist"\nallowed_hosts = ["evil.example"]\n')
    assert not (tmp_path / "work" / "ev_os1" / "harbor-config.yaml").exists()


def _handle(tmp_path: Path, **ownership_overrides: str) -> LaunchHandle:
    ownership = {
        cleanup.DEPLOYMENT_METADATA_KEY: DEPLOYMENT,
        cleanup.EVALUATION_METADATA_KEY: "ev_os1",
        cleanup.BENCHMARK_RUN_METADATA_KEY: "br_1",
        **ownership_overrides,
    }
    return LaunchHandle(
        backend="harbor_opensandbox",
        external_id="ev_os1",
        raw={
            "harbor_dir": str(tmp_path / "harbor"),
            "pid_file": str(tmp_path / "missing.pid"),
            "ownership": ownership,
            "provenance": {"harbor_version": "0.20.0"},
        },
    )


def _write_applied(tmp_path: Path, trial: str, sha: str, *, sandbox_id: str | None = None, role: str = "agent") -> None:
    trial_dir = tmp_path / "harbor" / "jobs" / "ev_os1" / trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    sandbox_id = sandbox_id or f"sb-{trial}"
    record = {"sandbox_id": sandbox_id, "role": role, "network_mode": "public", "policy_sha256": sha, "policy": {}}
    (trial_dir / cleanup.applied_egress_filename(sandbox_id)).write_text(json.dumps(record))


class _CleanupRecorder:
    def __init__(self, returncode: int = 0, report: Mapping[str, Any] | None = None) -> None:
        self.calls: list[tuple[list[str], Mapping[str, str]]] = []
        self.returncode = returncode
        self.report = report if report is not None else {"killed": ["sb-1"], "failed": [], "remaining": []}

    def __call__(
        self, argv: Sequence[str], env: Mapping[str, str], _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), env))
        return subprocess.CompletedProcess(list(argv), self.returncode, stdout=json.dumps(self.report) + "\n")


def _terminator(tmp_path: Path, recorder: _CleanupRecorder):
    return backend.make_harbor_opensandbox_terminator(
        harbor_dir=str(tmp_path / "harbor"),
        jobs_dir="jobs",
        cleanup_runner=recorder,
        environ={"OPENSANDBOX_DOMAIN": "os.svc", "OPENSANDBOX_API_KEY": "k"},
    )


def _applied_summary(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "harbor" / "jobs" / "ev_os1" / backend.APPLIED_EGRESS_SUMMARY_FILENAME).read_text())


def _cleanup_report(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "harbor" / "jobs" / "ev_os1" / backend.CLEANUP_REPORT_FILENAME).read_text())


def test_terminator_records_a_clean_cleanup(tmp_path: Path) -> None:
    recorder = _CleanupRecorder(report={"killed": [], "failed": [], "remaining": []})

    _terminator(tmp_path, recorder)(_handle(tmp_path))

    assert _cleanup_report(tmp_path) == {
        "status": "clean",
        "exit_code": 0,
        "killed": [],
        "failed": [],
        "remaining": [],
        "error": None,
    }


def test_terminator_records_a_failed_cleanup_before_raising(tmp_path: Path) -> None:
    recorder = _CleanupRecorder(returncode=2, report={"error": "ConnectError: refused"})

    with pytest.raises(RuntimeError, match="ConnectError: refused"):
        _terminator(tmp_path, recorder)(_handle(tmp_path))

    report = _cleanup_report(tmp_path)
    assert report["status"] == "failed"
    assert report["exit_code"] == 2
    assert report["error"] == "ConnectError: refused"


def test_worker_uploads_the_cleanup_report_written_after_the_artifact_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scaled_evals.api import s3
    from scaled_evals.dispatch.worker import Dispatcher

    uploads: list[tuple[Path, str]] = []
    monkeypatch.setattr(s3, "upload_file", lambda path, key, *, content_type: uploads.append((path, key)) or 1)
    backend.write_cleanup_report(tmp_path, {"status": "clean"})

    assert Dispatcher._upload_cleanup_report_warn("ev_os1", "harbor_opensandbox", tmp_path) is None
    assert Dispatcher._upload_cleanup_report_warn("ev_os1", "sandbox_k8s", tmp_path) is None
    assert Dispatcher._upload_cleanup_report_warn("ev_os1", "harbor_opensandbox", tmp_path / "missing") is None
    assert uploads == [
        (
            tmp_path / backend.CLEANUP_REPORT_FILENAME,
            s3.evaluation_artifact_key("ev_os1", backend.CLEANUP_REPORT_FILENAME),
        )
    ]


def test_terminator_runs_scoped_cleanup_in_harbor_venv_and_writes_applied_egress(tmp_path: Path) -> None:
    _write_applied(tmp_path, "trial-a", "1" * 64)
    _write_applied(tmp_path, "trial-b", "1" * 64)
    recorder = _CleanupRecorder()

    _terminator(tmp_path, recorder)(_handle(tmp_path))

    argv, env = recorder.calls[0]
    assert argv[:3] == [str(tmp_path / "harbor" / ".venv" / "bin" / "python"), "-m", backend.CLEANUP_MODULE]
    selectors = [argv[i + 1] for i, item in enumerate(argv) if item == "--selector"]
    assert selectors == [
        f"{cleanup.DEPLOYMENT_METADATA_KEY}={DEPLOYMENT}",
        f"{cleanup.EVALUATION_METADATA_KEY}=ev_os1",
    ]
    assert env["OPENSANDBOX_API_KEY"] == "k"
    assert _applied_summary(tmp_path)["sandboxes"] == [
        {
            "trial": "trial-a",
            "sandbox_id": "sb-trial-a",
            "role": "agent",
            "network_mode": "public",
            "policy_sha256": "1" * 64,
        },
        {
            "trial": "trial-b",
            "sandbox_id": "sb-trial-b",
            "role": "agent",
            "network_mode": "public",
            "policy_sha256": "1" * 64,
        },
    ]


def test_applied_egress_keeps_both_sandboxes_of_a_separate_verifier_trial(tmp_path: Path) -> None:
    _write_applied(tmp_path, "trial-a", "1" * 64, sandbox_id="sb-agent", role="agent")
    _write_applied(tmp_path, "trial-a", "2" * 64, sandbox_id="sb-verifier", role="verifier")

    records = backend.collect_applied_egress(tmp_path / "harbor" / "jobs" / "ev_os1")

    assert sorted((item["sandbox_id"], item["role"], item["policy_sha256"]) for item in records) == [
        ("sb-agent", "agent", "1" * 64),
        ("sb-verifier", "verifier", "2" * 64),
    ]
    assert {item["trial"] for item in records} == {"trial-a"}


def test_terminator_raises_and_still_writes_applied_egress_when_sandboxes_survive(tmp_path: Path) -> None:
    _write_applied(tmp_path, "trial-a", "1" * 64)
    recorder = _CleanupRecorder(returncode=1, report={"killed": [], "failed": [], "remaining": ["sb-9"]})

    with pytest.raises(RuntimeError, match="sb-9"):
        _terminator(tmp_path, recorder)(_handle(tmp_path))

    assert [item["trial"] for item in _applied_summary(tmp_path)["sandboxes"]] == ["trial-a"]
    report = _cleanup_report(tmp_path)
    assert report["status"] == "failed"
    assert report["remaining"] == ["sb-9"]


@pytest.mark.parametrize(
    "ownership",
    [
        {cleanup.DEPLOYMENT_METADATA_KEY: "someone-else"},
        {cleanup.EVALUATION_METADATA_KEY: "ev_other"},
        {cleanup.EVALUATION_METADATA_KEY: ""},
    ],
)
def test_terminator_refuses_foreign_or_incomplete_ownership(tmp_path: Path, ownership: dict[str, str]) -> None:
    recorder = _CleanupRecorder()

    with pytest.raises(RuntimeError, match="OpenSandbox cleanup failed"):
        _terminator(tmp_path, recorder)(_handle(tmp_path, **ownership))

    assert recorder.calls == []
    assert _cleanup_report(tmp_path)["status"] == "failed"


def test_status_reader_writes_applied_egress_when_terminal(tmp_path: Path) -> None:
    job_dir = tmp_path / "harbor" / "jobs" / "ev_os1"
    _write_applied(tmp_path, "trial-a", "2" * 64)
    (job_dir / "result.json").write_text(
        json.dumps({"finished_at": "2026-09-28T00:00:00Z", "n_total_trials": 1, "stats": {"n_errored_trials": 0}})
    )
    read = backend.make_harbor_opensandbox_status_reader(harbor_dir=str(tmp_path / "harbor"), jobs_dir="jobs")

    status = read(_handle(tmp_path))

    assert status.phase == "succeeded"
    assert [item["policy_sha256"] for item in _applied_summary(tmp_path)["sandboxes"]] == ["2" * 64]


def test_backend_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "harbor_opensandbox_enabled", False)

    with pytest.raises(NotImplementedError, match="HARBOR_OPENSANDBOX_ENABLED"):
        backend.build_backend().launch(_spec())


def test_enabled_backend_requires_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "harbor_opensandbox_enabled", True)
    monkeypatch.setattr(settings, "harbor_opensandbox_config_path", None)

    with pytest.raises(RuntimeError, match="HARBOR_OPENSANDBOX_CONFIG_PATH"):
        backend.build_backend()


def test_runtime_pins_harbor_020() -> None:
    assert resolve_framework_runner("harbor", None, runtime="harbor_opensandbox").version == "0.20.0"
    with pytest.raises(ValueError, match="supports only Harbor 0.20.0"):
        resolve_framework_runner("harbor", "0.13.2", runtime="harbor_opensandbox")
    with pytest.raises(ValueError, match="requires framework 'harbor'"):
        resolve_framework_runner("gym", None, runtime="harbor_opensandbox")
