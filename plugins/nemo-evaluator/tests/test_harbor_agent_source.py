# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import stat
import tarfile
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from nemo_evaluator.harbor import agent_source as source
from nemo_evaluator.jobs.agent_spec import HarborRunnerTarget
from nemo_evaluator.jobs.runner_targets import UnsubmittableRunnerError, runner_to_target
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner, HarborRuntimeConfig
from nemo_evaluator_sdk.values import SecretRef


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "wrapper.py").write_text("from .helper import answer\n")
    (root / "helper.py").write_text("answer = 42\n")
    (root / "run.sh").write_text("#!/bin/sh\necho hello\n")
    (root / "run.sh").chmod(0o755)
    (root / "empty").mkdir()
    (root / ".env").write_text("SECRET=never-upload")
    (root / ".codex").mkdir()
    (root / ".codex" / "auth.json").write_text("secret")
    return root


@pytest.fixture
def files():
    objects = {}
    client = Mock()
    client.with_headers.return_value = client

    def upload(*, path, content, **kwargs):
        objects[path] = b"".join(content)
        return Mock()

    def download(*, path, **kwargs):
        @contextmanager
        def stream(**kwargs):
            yield iter([objects[path]])

        return Mock(stream=stream)

    client.upload_file.side_effect = upload
    client.download_file.side_effect = download
    return client, objects


def test_inventory_capture_and_relocation(root, tmp_path):
    inventory = source.inspect_agent_source(root)
    assert set(inventory.excluded) == {".env", ".codex/auth.json"}
    first, second = tmp_path / "a.gz", tmp_path / "b.gz"
    assert source.capture_agent_source(inventory, first) == source.capture_agent_source(inventory, second)
    root.rename(tmp_path / "moved")
    destination = tmp_path / "extracted"
    destination.mkdir()
    source.extract_agent_source(first, destination)
    assert (destination / "empty").is_dir()
    assert (destination / "run.sh").stat().st_mode & stat.S_IXUSR
    assert not (destination / ".env").exists()
    assert (destination / "wrapper.py").read_text() == "from .helper import answer\n"


def test_publication_unique_prefix_and_worker_cleanup(root, files, tmp_path):
    client, objects = files
    first = source.publish_agent_source(root, files_client=client, fileset_ref="default/agents")
    second = source.publish_agent_source(root, files_client=client, fileset_ref="default/agents")
    assert first.sha256 == second.sha256
    assert first.fileset_ref != second.fileset_ref
    with source.prepared_agent_source(first, parent=tmp_path / "worker", files_client=client) as directory:
        assert directory.name == "agent"
        assert (directory / "helper.py").is_file()
    assert not directory.exists()
    objects[first.fileset_ref.split("#")[1]] = b"tampered"
    with pytest.raises(ValueError, match="checksum"):
        with source.prepared_agent_source(first, parent=tmp_path / "worker", files_client=client):
            pytest.fail("Cannot prepare tampered source")
    assert not list((tmp_path / "worker").iterdir())


def test_policy_overrides_and_jobs_exclusion(root):
    (root / "outputs").mkdir()
    (root / "outputs" / "result").write_text("output")
    inventory = source.inspect_agent_source(
        root, options=source.AgentSourceOptions(include_files=(".env",)), jobs_dir=root / "outputs"
    )
    assert any(entry.path == ".env" and entry.overridden for entry in inventory.entries)
    assert "outputs" in inventory.excluded
    with pytest.raises(ValueError, match="missing"):
        source.inspect_agent_source(root, options=source.AgentSourceOptions(include_files=("missing",)))


def test_selected_symlink_and_changed_file_fail(root, tmp_path):
    inventory = source.inspect_agent_source(root)
    (root / "helper.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        source.capture_agent_source(inventory, tmp_path / "changed.gz")
    (root / "link").symlink_to(root / "helper.py")
    with pytest.raises(ValueError, match="symlink"):
        source.inspect_agent_source(root)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        source.inspect_agent_source(alias)


@pytest.mark.parametrize(
    "name,type_,mode",
    [
        ("../escape", tarfile.REGTYPE, 0o644),
        ("/escape", tarfile.REGTYPE, 0o644),
        ("link", tarfile.SYMTYPE, 0o644),
        ("hard", tarfile.LNKTYPE, 0o644),
        ("fifo", tarfile.FIFOTYPE, 0o644),
        ("setuid", tarfile.REGTYPE, 0o4755),
    ],
)
def test_malicious_archive_rejected(tmp_path, name, type_, mode):
    path = tmp_path / "bad.gz"
    with tarfile.open(path, "w:gz") as archive:
        header = tarfile.TarInfo(name)
        header.type, header.mode = type_, mode
        archive.addfile(header)
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(ValueError):
        source.extract_agent_source(path, destination)


def test_archive_limits_include_extensions_and_gzip_stream(tmp_path, monkeypatch):
    path = tmp_path / "bad.gz"
    header = tarfile.TarInfo("pax")
    header.type, header.size = tarfile.XHDTYPE, source.MAX_EXTENSION_BYTES + 1
    path.write_bytes(gzip.compress(header.tobuf() + b"x" * header.size))
    with pytest.raises(ValueError, match="extension"):
        source.extract_agent_source(path, tmp_path)
    monkeypatch.setattr(source, "MAX_STREAM_BYTES", 1024)
    path.write_bytes(gzip.compress(b"\0" * 2048))
    with pytest.raises(ValueError, match="stream"):
        source.extract_agent_source(path, tmp_path)


@pytest.mark.parametrize("names", [("same", "same"), ("Dir", "dir"), ("Dir", "dir/file")])
def test_case_colliding_paths(tmp_path, names):
    path = tmp_path / "bad.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            header = tarfile.TarInfo(name)
            header.type, header.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(header)
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(ValueError, match="[Cc]ase|[Dd]uplicate"):
        source.extract_agent_source(path, destination)


def test_native_conversion_preserves_public_env_timeouts_and_references(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-never-appear")
    config = HarborRuntimeConfig(
        jobs_dir=tmp_path,
        agent_name="codex",
        agent_model_name="model",
        agent_kwargs={"version": "1.2.3"},
        agent_env={"ENDPOINT": "http://provider"},
        agent_env_from_host=["OPENAI_API_KEY"],
        **{field: 2.5 for field in HarborRuntimeConfig.model_fields if field.endswith("timeout_multiplier")},
    )
    runner = HarborAgentTaskRunner(config=config)
    with pytest.raises(UnsubmittableRunnerError, match="OPENAI_API_KEY"):
        runner_to_target(runner)
    target = runner_to_target(runner, env_secrets={"OPENAI_API_KEY": SecretRef("default/key")})
    assert isinstance(target, HarborRunnerTarget)
    assert target.agent_env == config.agent_env
    for field in HarborRuntimeConfig.model_fields:
        if field.endswith("timeout_multiplier"):
            assert getattr(target, field) == 2.5
    assert "must-never-appear" not in target.model_dump_json()
    exported = runner.export_submission_config()
    exported.agent_env["X"] = "Y"
    assert "X" not in runner.export_submission_config().agent_env


@pytest.mark.parametrize("field,value", [("job_name", "local"), ("force_rerun", True)])
def test_local_resume_settings_rejected(tmp_path, field, value):
    runner = HarborAgentTaskRunner(config=HarborRuntimeConfig(jobs_dir=tmp_path, **{field: value}))
    with pytest.raises(UnsubmittableRunnerError, match=field):
        runner_to_target(runner)


@pytest.mark.parametrize(
    "field,value",
    [("dataset_path", "/tmp/data"), ("task_names", []), ("job_dir", "/tmp/job"), ("run_job", lambda: None)],
)
def test_process_local_overrides_rejected(tmp_path, field, value):
    runner = HarborAgentTaskRunner(config=HarborRuntimeConfig(jobs_dir=tmp_path), **{field: value})
    with pytest.raises(UnsubmittableRunnerError, match=field):
        runner_to_target(runner)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_timeouts_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        HarborRuntimeConfig(jobs_dir=tmp_path, timeout_multiplier=value)
    with pytest.raises(ValueError):
        HarborRunnerTarget(agent_timeout_multiplier=value)


@pytest.mark.parametrize("env", [{"INVALID-NAME": "x"}, {"X": "${HOST_SECRET}"}])
def test_invalid_public_env_rejected(tmp_path, env):
    with pytest.raises(ValueError):
        HarborRuntimeConfig(jobs_dir=tmp_path, agent_env=env)
    with pytest.raises(ValueError):
        HarborRunnerTarget(agent_env=env)


def test_public_secret_name_collision(tmp_path):
    with pytest.raises(ValueError, match="overlaps"):
        HarborRuntimeConfig(jobs_dir=tmp_path, agent_env={"X": "public"}, agent_env_from_host=["X"])
    with pytest.raises(ValueError, match="overlaps"):
        HarborRunnerTarget(agent_env={"X": "public"}, env_secrets={"X": SecretRef("default/secret")})


@pytest.mark.asyncio
async def test_source_validation_precedes_all_task_shapes(monkeypatch):
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator.jobs.agent_spec import AgentEvalInputSpec

    monkeypatch.setattr(
        "nemo_evaluator.jobs.agent_evaluate.get_config", lambda: Mock(harbor_agent_source_enabled=False)
    )
    descriptor = source.HarborAgentSource(fileset_ref="default/agents#one/agent.tar.gz", sha256="a" * 64)
    for tasks in ("default/taskset", ["default/task"], [{"id": "one", "intent": "test"}]):
        spec = AgentEvalInputSpec.model_validate(
            {
                "tasks": tasks,
                "target": {"kind": "harbor", "agent_import_path": "wrapper:Agent", "agent_source": descriptor},
            }
        )
        with pytest.raises(ValueError, match="disabled"):
            await AgentEvalJob.to_spec(spec, workspace="default", entity_client=None, async_sdk=None, is_local=True)


def test_generated_cli_agent_dir_uses_merged_spec(root, monkeypatch):
    import json

    import typer
    from nemo_evaluator.cli import EvaluatorPluginCLI
    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_platform_plugin.commands import add_job_commands
    from typer.testing import CliRunner

    published = []
    submitted = []
    descriptor = source.HarborAgentSource(fileset_ref="default/agents#one/agent.tar.gz", sha256="a" * 64)

    def publish(path, **kwargs):
        published.append(path.resolve())
        return descriptor

    monkeypatch.setattr(source, "publish_agent_source", publish)
    monkeypatch.setattr(
        "nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote",
        lambda self, cls, spec, **kwargs: submitted.append(spec) or {"name": "job"},
    )
    app = typer.Typer()

    @app.callback()
    def main():
        pass

    add_job_commands(app, {"evaluator.agent-evaluate": AgentEvalJob}, cli=EvaluatorPluginCLI())
    runner = CliRunner()
    help_result = runner.invoke(app, ["agent-evaluate", "submit", "--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "agent-dir" in help_result.output
    monkeypatch.chdir(root.parent)
    spec = {"tasks": "default/tasks", "target": {"kind": "harbor", "agent_import_path": "wrapper:Agent"}}
    result = runner.invoke(app, ["agent-evaluate", "submit", "--spec", json.dumps(spec), "--agent-dir", root.name])
    assert result.exit_code == 0, (result.output, result.exception)
    assert published == [root]
    assert submitted[0]["target"]["agent_source"] == descriptor.model_dump()
    assert descriptor.model_dump_json() in result.stderr
    assert json.loads(result.stdout) == {"name": "job"}
    spec["target"]["agent_source"] = descriptor.model_dump()
    result = runner.invoke(app, ["agent-evaluate", "submit", "--spec", json.dumps(spec), "--agent-dir", root.name])
    assert result.exit_code != 0
    assert len(published) == 1

    import httpx

    def lose_response(self, cls, spec, **kwargs):
        submitted.append(spec)
        raise httpx.ReadTimeout("response lost")

    monkeypatch.setattr("nemo_platform_plugin.scheduler.NemoJobScheduler.submit_remote", lose_response)
    del spec["target"]["agent_source"]
    result = runner.invoke(app, ["agent-evaluate", "submit", "--spec", json.dumps(spec), "--agent-dir", root.name])
    assert result.exit_code == 2
    assert len(submitted) == 2  # One successful submission and one uncertain submission; no retry.
    assert descriptor.model_dump_json() in result.stderr
    assert "inspect job state before resubmitting" in result.stderr
    assert result.stderr.index(descriptor.model_dump_json()) < result.stderr.index("response lost")


def test_publication_unknown_outcome_requires_readback(root, files):
    from nemo_platform_plugin.client.errors import NemoTransportError

    client, objects = files
    upload = client.upload_file.side_effect

    def lost_response(**kwargs):
        upload(**kwargs)
        raise NemoTransportError(__import__("httpx").ReadError("lost response"))

    client.upload_file.side_effect = lost_response
    descriptor = source.publish_agent_source(root, files_client=client, fileset_ref="default/agents")
    assert descriptor.fileset_ref.split("#")[1] in objects


def test_source_import_failure_cleans_scope(root, tmp_path, monkeypatch):
    import asyncio
    import sys

    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _build_native_job

    pytest.importorskip("harbor")
    import harbor.job

    create = Mock(side_effect=AssertionError("must fail before trial creation"))
    monkeypatch.setattr(harbor.job.Job, "create", create)
    (root / "wrapper.py").write_text("import nonexistent_worker_dependency_673\n")
    config = HarborRuntimeConfig(jobs_dir=tmp_path / "jobs", agent_dir=root, agent_import_path="wrapper:Agent")
    before = set(sys.modules)
    _, run = _build_native_job(config, tmp_path / "dataset", None)
    with pytest.raises(ModuleNotFoundError, match="nonexistent_worker_dependency"):
        asyncio.run(run())
    assert not [key for key in set(sys.modules) - before if key.startswith("source_")]
    create.assert_not_called()


def test_sdk_publishes_only_after_validation_and_retains_receipt(root, files, monkeypatch):
    from nemo_evaluator.api.schemas import TasksetRef
    from nemo_evaluator.sdk._executor import _SyncEvaluatorPluginExecutor
    from nemo_platform_plugin.evaluator.client import EvaluatorClient

    client, _ = files
    executor = _SyncEvaluatorPluginExecutor(client=EvaluatorClient(base_url="http://localhost", workspace="default"))
    executor._files_client = client
    executor.create_agent_eval = Mock(side_effect=RuntimeError("job response lost"))
    runner = HarborAgentTaskRunner(
        config=HarborRuntimeConfig(
            jobs_dir=root / "jobs", agent_dir=root, agent_import_path="wrapper:Agent", agent_env_from_host=["KEY"]
        )
    )
    with pytest.raises(UnsubmittableRunnerError, match="KEY"):
        executor.submit_agent_eval(tasks=TasksetRef("default/tasks"), target=runner)
    client.upload_file.assert_not_called()
    with pytest.raises(RuntimeError, match="response lost") as error:
        executor.submit_agent_eval(
            tasks=TasksetRef("default/tasks"), target=runner, env_secrets={"KEY": SecretRef("default/key")}
        )
    executor.create_agent_eval.assert_called_once()
    note = " ".join(error.value.__notes__)
    assert "sha256" in note and "retained" in note
    spec = executor.create_agent_eval.call_args.kwargs["spec"]
    assert spec.target.agent_source is not None
    assert str(root) not in spec.model_dump_json()


@pytest.mark.asyncio
async def test_env_timeouts_worker_and_harbor_roundtrip(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nemo_evaluator.jobs.agent_evaluate import AgentEvalJob
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _build_native_job

    pytest.importorskip("harbor")
    import harbor.job

    captured = []

    class FakeJob:
        @classmethod
        async def create(cls, config):
            captured.append(config)
            return cls()

        async def run(self):
            pass

    monkeypatch.setattr(harbor.job, "Job", FakeJob)
    monkeypatch.setenv("KEY", "secret-value")
    target = HarborRunnerTarget(
        agent_env={"ENDPOINT": "http://provider"},
        env_secrets={"KEY": SecretRef("default/key")},
        **{field: 3.0 for field in HarborRuntimeConfig.model_fields if field.endswith("timeout_multiplier")},
    )
    runner, _, _ = AgentEvalJob._resolve_target(target, SimpleNamespace(storage=SimpleNamespace(persistent=tmp_path)))
    config = runner.export_submission_config()
    _, run = _build_native_job(config, tmp_path / "tasks", None)
    await run()
    [job] = captured
    assert job.agents[0].env == {"ENDPOINT": "http://provider", "KEY": "${KEY}"}
    assert "secret-value" not in job.model_dump_json()
    for field in HarborRuntimeConfig.model_fields:
        if field.endswith("timeout_multiplier"):
            assert getattr(job, field) == 3.0


def test_uncertain_harbor_submission_never_inherits_post_retries(tmp_path):
    import httpx
    from nemo_evaluator.api.schemas import TasksetRef
    from nemo_evaluator.sdk._executor import _SyncEvaluatorPluginExecutor
    from nemo_platform_plugin.client.errors import NemoTransportError
    from nemo_platform_plugin.client.types import RetryPolicy
    from nemo_platform_plugin.evaluator.client import EvaluatorClient

    requests = []

    def lose_response(request):
        requests.append(request)
        raise httpx.ReadError("accepted, response lost", request=request)

    with httpx.Client(transport=httpx.MockTransport(lose_response)) as http:
        client = EvaluatorClient(
            base_url="http://localhost", workspace="default", http_client=http, retry=RetryPolicy(max_retries=3)
        )
        executor = _SyncEvaluatorPluginExecutor(client=client)
        with pytest.raises(NemoTransportError):
            executor.submit_agent_eval(
                tasks=TasksetRef("default/tasks"),
                target=HarborAgentTaskRunner(config=HarborRuntimeConfig(jobs_dir=tmp_path)),
            )
    assert len(requests) == 1


@pytest.mark.parametrize("limit,value", [("MAX_ENTRIES", 1), ("MAX_FILE_BYTES", 1), ("MAX_PAYLOAD_BYTES", 1)])
def test_capture_policy_limits_fail_before_upload(root, files, monkeypatch, limit, value):
    client, _ = files
    monkeypatch.setattr(source, limit, value)
    with pytest.raises(ValueError, match="limit"):
        source.publish_agent_source(root, files_client=client, fileset_ref="default/agents")
    client.create_fileset.assert_not_called()
    client.upload_file.assert_not_called()


def test_attempts_own_distinct_source_lifetimes(root, files, tmp_path):
    client, _ = files
    descriptor = source.publish_agent_source(root, files_client=client, fileset_ref="default/agents")
    with source.prepared_agent_source(descriptor, parent=tmp_path / "worker", files_client=client) as first:
        with source.prepared_agent_source(descriptor, parent=tmp_path / "worker", files_client=client) as second:
            assert first != second
            assert first.is_dir() and second.is_dir()
        assert first.is_dir() and not second.exists()
    assert not first.exists()


@pytest.mark.parametrize("directory", ["private", ".env.d", "credentials.pem", ".aws"])
def test_exact_override_does_not_include_excluded_siblings(root, tmp_path, directory):
    (root / directory / "nested").mkdir(parents=True)
    (root / directory / "nested" / "allowed.txt").write_text("allowed")
    (root / directory / "nested" / "secret.txt").write_text("excluded")
    (root / directory / "production").write_text("excluded")
    allowed = f"{directory}/nested/allowed.txt"
    inventory = source.inspect_agent_source(
        root, options=source.AgentSourceOptions(exclude_globs=("private",), include_files=(allowed,))
    )
    assert allowed in {entry.path for entry in inventory.entries}
    assert f"{directory}/nested/secret.txt" in inventory.excluded
    assert f"{directory}/production" in inventory.excluded
    archive = tmp_path / "override.tar.gz"
    source.capture_agent_source(inventory, archive)
    with tarfile.open(archive) as bundle:
        assert allowed in bundle.getnames()
        assert f"{directory}/nested/secret.txt" not in bundle.getnames()
        assert f"{directory}/production" not in bundle.getnames()


def test_public_env_changes_cache_identity_but_secret_values_do_not(tmp_path, monkeypatch):
    from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import _cache_stamp

    config = HarborRuntimeConfig(
        jobs_dir=tmp_path / "jobs", agent_env={"ENDPOINT": "first"}, agent_env_from_host=["KEY"]
    )
    first = _cache_stamp(config, tmp_path, [])
    monkeypatch.setenv("KEY", "rotated")
    assert _cache_stamp(config, tmp_path, []) == first
    config.agent_env["ENDPOINT"] = "second"
    assert _cache_stamp(config, tmp_path, []) != first
