# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import MagicMock, patch

import pytest
import yaml
from nemo_optimization.jobs.optimize import OptimizeJob
from nemo_optimization.schemas.optimize import FILESET_REQUIRED, OptimizeSpec, OptimizeSubmitSpec
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.jobs.exceptions import (
    PlatformJobCompilationError,
    PlatformJobDependencyUnavailableError,
)
from nemo_platform_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    DockerJobExecutionProfileConfig,
    SubprocessJobExecutionProfile,
)
from nemo_platform_plugin.refs import FilesetRef
from nemo_platform_plugin.run_dependencies import LocalRunError
from nemo_platform_plugin.scheduler import NemoJobScheduler
from pydantic import ValidationError

FABRIC_AGENT = {
    "schema_version": "fabric.agent/v1alpha1",
    "metadata": {"name": "hermes-optimize-demo"},
}
MINIMAL_CONFIG = {"optimizer": {"numeric": {"enabled": True}}}

SUBPROCESS_PROFILE = SubprocessJobExecutionProfile(profile="default")
CPU_PROFILE = DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())


def write_config(directory: Path, config: dict[str, Any], name: str = "optimize.yml") -> str:
    """Write *config* into *directory* and return its absolute path."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    return str(path)


@contextlib.contextmanager
def profiles(*execution_profiles: Any) -> Iterator[None]:
    """Patch the Jobs client so ``compile`` sees exactly *execution_profiles*."""

    async def _get_execution_profiles() -> Any:
        return SimpleNamespace(data=lambda: list(execution_profiles))

    client = MagicMock()
    client.get_execution_profiles = _get_execution_profiles
    with patch("nemo_optimization.jobs.optimize.client_from_platform", return_value=client):
        yield


async def compile_spec(spec: OptimizeSpec, *, workspace: str = "default", profile: str | None = None) -> Any:
    return await OptimizeJob.compile(
        workspace=workspace,
        spec=spec,
        entity_client=MagicMock(),
        job_name=None,
        async_sdk=MagicMock(),
        profile=profile,
    )


def staged_spec(**overrides: Any) -> OptimizeSpec:
    return OptimizeSpec.model_validate(
        {"optimize_config": "optimize.yml", "optimize_config_fileset": "default/opt-bundle", **overrides}
    )


# ---------------------------------------------------------------------------
# compile — fileset requirement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compile_stamps_the_fileset_ref_into_the_step_config() -> None:
    with profiles(SUBPROCESS_PROFILE):
        platform_spec = await compile_spec(staged_spec(), workspace="staging")

    step = next(iter(platform_spec["steps"]))
    assert step["name"] == "optimize"
    assert step["config"]["workspace"] == "staging"
    assert step["config"]["optimize_config_fileset"] == "default/opt-bundle"
    assert step["config"]["optimize_config"] == "optimize.yml"


@pytest.mark.asyncio
async def test_compile_requires_a_staged_fileset() -> None:
    spec = OptimizeSpec(optimize_config="/abs/optimize.yml")
    with pytest.raises(PlatformJobCompilationError, match="prepare-fileset"):
        await compile_spec(spec)


def test_spec_rejects_absolute_config_alongside_a_fileset() -> None:
    with pytest.raises(ValidationError, match="relative to the fileset root"):
        OptimizeSpec(optimize_config="/abs/optimize.yml", optimize_config_fileset=FilesetRef("opt-bundle"))


@pytest.mark.parametrize("config_path", ["../escape.yml", "~/optimize.yml", "C:\\bundle\\optimize.yml"])
def test_spec_rejects_config_paths_that_escape_the_fileset(config_path: str) -> None:
    with pytest.raises(ValidationError, match="relative to the fileset root"):
        OptimizeSpec(optimize_config=config_path, optimize_config_fileset=FilesetRef("opt-bundle"))


def test_spec_rejects_a_malformed_fileset_ref() -> None:
    with pytest.raises(ValidationError, match="'name' or 'workspace/name'"):
        OptimizeSpec(optimize_config="optimize.yml", optimize_config_fileset=FilesetRef("ws/fs/extra"))


def test_spec_requires_a_config_location() -> None:
    with pytest.raises(ValidationError):
        OptimizeSpec.model_validate({})


def test_submit_spec_requires_fileset_for_remote_requests() -> None:
    with pytest.raises(ValidationError, match="prepare-fileset") as missing:
        OptimizeSubmitSpec.model_validate({"optimize_config": "optimize.yml"})
    assert FILESET_REQUIRED in str(missing.value)

    with pytest.raises(ValidationError, match="prepare-fileset") as explicit_none:
        OptimizeSubmitSpec.model_validate({"optimize_config": "optimize.yml", "optimize_config_fileset": None})
    assert FILESET_REQUIRED in str(explicit_none.value)


def test_submit_spec_allows_missing_fileset_for_local_scheduler() -> None:
    spec = OptimizeSubmitSpec.model_validate({"optimize_config": "/abs/optimize.yml"}, context={"is_local": True})

    assert spec.optimize_config == "/abs/optimize.yml"
    assert spec.optimize_config_fileset is None


# ---------------------------------------------------------------------------
# compile — executor selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compile_prefers_the_subprocess_profile() -> None:
    with profiles(SUBPROCESS_PROFILE, CPU_PROFILE):
        platform_spec = await compile_spec(staged_spec())

    executor = next(iter(platform_spec["steps"]))["executor"]
    assert executor["provider"] == "subprocess"
    assert executor["profile"] == "default"
    assert executor["command"] == ["python", "-m", "nemo_optimization.tasks.optimize"]


@pytest.mark.asyncio
async def test_compile_falls_back_to_the_cpu_profile_with_a_task_image() -> None:
    with (
        profiles(CPU_PROFILE),
        patch("nemo_optimization.jobs.optimize.get_qualified_image", return_value="reg.example/nmp-cpu-tasks:test"),
    ):
        platform_spec = await compile_spec(staged_spec())

    executor = next(iter(platform_spec["steps"]))["executor"]
    assert executor["provider"] == "cpu"
    assert executor["profile"] == "default"
    assert executor["container"]["image"] == "reg.example/nmp-cpu-tasks:test"
    assert [*executor["container"]["entrypoint"], *executor["container"]["command"]] == [
        "python",
        "-m",
        "nemo_optimization.tasks.optimize",
    ]


@pytest.mark.asyncio
async def test_compile_matches_the_requested_profile_name() -> None:
    """A subprocess backend registered under another name must not capture 'high-mem'."""
    with (
        profiles(
            SUBPROCESS_PROFILE,
            DockerJobExecutionProfile(provider="cpu", profile="high-mem", config=DockerJobExecutionProfileConfig()),
        ),
        patch("nemo_optimization.jobs.optimize.get_qualified_image", return_value="reg.example/nmp-cpu-tasks:test"),
    ):
        platform_spec = await compile_spec(staged_spec(), profile="high-mem")

    executor = next(iter(platform_spec["steps"]))["executor"]
    assert executor["provider"] == "cpu"
    assert executor["profile"] == "high-mem"


@pytest.mark.asyncio
async def test_compile_reports_available_profiles_when_none_match() -> None:
    with profiles(DockerJobExecutionProfile(provider="gpu", profile="a100", config=DockerJobExecutionProfileConfig())):
        with pytest.raises(PlatformJobCompilationError, match=r"Available profiles: \['gpu/a100'\]"):
            await compile_spec(staged_spec())


@pytest.mark.asyncio
async def test_compile_is_retryable_when_jobs_is_unreachable() -> None:
    import httpx
    from nemo_platform_plugin.client.errors import NemoTransportError

    async def _boom() -> Any:
        raise NemoTransportError(httpx.ConnectError("connection refused", request=httpx.Request("GET", "http://x")))

    client = MagicMock()

    client.get_execution_profiles = _boom
    with (
        patch("nemo_optimization.jobs.optimize.client_from_platform", return_value=client),
        pytest.raises(PlatformJobDependencyUnavailableError, match="temporarily unavailable"),
    ):
        await compile_spec(staged_spec())


# ---------------------------------------------------------------------------
# run — local (absolute path) mode
# ---------------------------------------------------------------------------


def test_run_dispatches_a_local_fabric_config(tmp_path: Path, ctx: JobContext) -> None:
    optimize_config = write_config(tmp_path, {**FABRIC_AGENT, **MINIMAL_CONFIG})

    with patch(
        "nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", return_value={"status": "completed"}
    ) as dispatch:
        result = OptimizeJob().run({"optimize_config": optimize_config, "workspace": "default"}, ctx=ctx)

    assert result["status"] == "completed"
    kwargs = dispatch.call_args.kwargs
    assert kwargs["agent_config"] is None
    assert kwargs["optimize_config"]["optimizer"]["numeric"]["enabled"] is True


def test_scheduler_run_local_preserves_workspace_for_absolute_config_without_fileset(tmp_path: Path) -> None:
    optimize_config = write_config(tmp_path, {**FABRIC_AGENT, **MINIMAL_CONFIG})
    observed: dict[str, str] = {}

    def _preflight(*args: Any, workspace: str, **kwargs: Any) -> None:
        del args, kwargs
        observed["workspace"] = workspace

    with (
        patch("nemo_optimization.jobs.optimize.preflight_validate_llm_models", side_effect=_preflight),
        patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", return_value={"status": "completed"}),
    ):
        result = NemoJobScheduler().run_local(
            OptimizeJob,
            {"optimize_config": optimize_config},
            workspace="research",
        )

    assert result["status"] == "completed"
    assert observed["workspace"] == "research"


def test_run_leaves_the_working_directory_alone_in_local_mode(tmp_path: Path, ctx: JobContext) -> None:
    optimize_config = write_config(tmp_path, MINIMAL_CONFIG)
    cwd = Path.cwd()
    observed: dict[str, Path] = {}

    def _dispatch(**kwargs: Any) -> dict[str, Any]:
        observed["cwd"] = Path.cwd()
        return {"status": "completed"}

    with patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=_dispatch):
        OptimizeJob().run({"optimize_config": optimize_config, "workspace": "default"}, ctx=ctx)

    assert observed["cwd"] == cwd


def test_run_expands_env_vars_in_the_config(tmp_path: Path, ctx: JobContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIMIZE_TEST_MODEL", "demo-model")
    optimize_config = write_config(tmp_path, {"models": {"default": {"model": "${OPTIMIZE_TEST_MODEL}"}}})

    with patch(
        "nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", return_value={"status": "completed"}
    ) as dispatch:
        OptimizeJob().run({"optimize_config": optimize_config, "workspace": "default"}, ctx=ctx)

    assert dispatch.call_args.kwargs["optimize_config"]["models"]["default"]["model"] == "demo-model"


def test_run_resolves_platform_agent_before_dispatch(tmp_path: Path, ctx: JobContext) -> None:
    optimize_config = write_config(tmp_path, MINIMAL_CONFIG)

    platform_agent = {
        "config_format": "nemo-agents-spec-v1",
        "name": "react-agent",
        "default_harness": "hermes",
        "harnesses": {
            "hermes": {
                "kind": "hermes",
                "model": {
                    "provider": "openai",
                    "model": "demo-model",
                    "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                    "api_key_env": "NEMO_AGENTS_IGW_API_KEY",
                },
                "settings": {"max_tokens": 256, "reasoning_config": {"effort": "none"}},
            }
        },
        "instructions": {"system": {"content": "Be brief."}},
        "environment": {"provider": "local", "workspace": "./workspace", "artifacts": "./artifacts"},
        "models": {
            "judge": {
                "provider": "openai",
                "model": "demo-model",
                "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
                "api_key_env": "NEMO_AGENTS_IGW_API_KEY",
            }
        },
    }

    class _StubAgents:
        def get(self, *, name: str, workspace: str) -> dict[str, Any]:
            assert name == "react-agent"
            assert workspace == "default"
            return {"config": platform_agent}

    class _StubSDK:
        agents = _StubAgents()

    with patch(
        "nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", return_value={"status": "completed"}
    ) as dispatch:
        OptimizeJob().run(
            {"optimize_config": optimize_config, "workspace": "default", "agent": "react-agent"},
            ctx=ctx,
            sdk=cast(NeMoPlatform, _StubSDK()),
        )

    agent_config = dispatch.call_args.kwargs["agent_config"]
    assert agent_config["schema_version"] == "fabric.agent/v1alpha1"
    assert agent_config["harness"]["adapter_id"] == "nvidia.fabric.hermes"
    assert agent_config["models"]["default"]["model"] == "demo-model"
    assert agent_config["models"]["judge"]["model"] == "demo-model"


def test_run_rejects_endpoint_agent(tmp_path: Path, ctx: JobContext) -> None:
    optimize_config = write_config(tmp_path, MINIMAL_CONFIG)

    with pytest.raises(LocalRunError, match="Endpoint URL / URI optimize mode has been removed"):
        OptimizeJob().run(
            {"optimize_config": optimize_config, "workspace": "default", "agent": "http://localhost:8080"},
            ctx=ctx,
        )


# ---------------------------------------------------------------------------
# run — staged (fileset) mode
# ---------------------------------------------------------------------------


def bundle_sdk(bundle: dict[str, str], *, downloaded: dict[str, Any] | None = None) -> NeMoPlatform:
    """An SDK stub whose ``files.download`` materializes *bundle* (relative path → contents)."""

    class _StubFiles:
        def download(self, *, local_path: str, fileset: str, workspace: str) -> None:
            if downloaded is not None:
                downloaded.update(fileset=fileset, workspace=workspace)
            for relative, contents in bundle.items():
                target = Path(local_path) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents)

    class _StubSDK:
        files = _StubFiles()

    return cast(NeMoPlatform, _StubSDK())


def test_run_stages_the_config_from_the_fileset(ctx: JobContext) -> None:
    downloaded: dict[str, Any] = {}
    sdk = bundle_sdk({"configs/optimize.yml": yaml.safe_dump(MINIMAL_CONFIG)}, downloaded=downloaded)

    with patch(
        "nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", return_value={"status": "completed"}
    ) as dispatch:
        result = OptimizeJob().run(
            {
                "optimize_config": "configs/optimize.yml",
                "optimize_config_fileset": "default/opt-bundle",
                "workspace": "default",
            },
            ctx=ctx,
            sdk=sdk,
        )

    assert result["status"] == "completed"
    assert downloaded == {"fileset": "opt-bundle", "workspace": "default"}
    assert dispatch.call_args.kwargs["optimize_config"]["optimizer"]["numeric"]["enabled"] is True


def test_run_resolves_relative_assets_against_the_staged_bundle(ctx: JobContext) -> None:
    """Relative dataset / base_dir entries must resolve inside the download, not the task's cwd."""
    config = {
        **MINIMAL_CONFIG,
        "eval": {
            "general": {"dataset": {"file_path": "data/rows.json"}},
            "fabric": {"base_dir": "."},
        },
    }
    sdk = bundle_sdk(
        {
            "optimize.yml": yaml.safe_dump(config),
            "data/rows.json": json.dumps([{"question": "q", "answer": "a"}]),
        }
    )
    observed: dict[str, Any] = {}

    def _dispatch(**kwargs: Any) -> dict[str, Any]:
        dataset = kwargs["optimize_config"]["eval"]["general"]["dataset"]["file_path"]
        observed["rows"] = json.loads(Path(dataset).read_text())
        observed["cwd"] = Path.cwd()
        return {"status": "completed"}

    cwd = Path.cwd()
    with patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=_dispatch):
        OptimizeJob().run(
            {
                "optimize_config": "optimize.yml",
                "optimize_config_fileset": "opt-bundle",
                "workspace": "default",
            },
            ctx=ctx,
            sdk=sdk,
        )

    assert observed["rows"] == [{"question": "q", "answer": "a"}]
    assert observed["cwd"] != cwd
    # The chdir is undone even though the study ran inside it.
    assert Path.cwd() == cwd


def test_run_restores_the_working_directory_when_the_study_raises(ctx: JobContext) -> None:
    sdk = bundle_sdk({"optimize.yml": yaml.safe_dump(MINIMAL_CONFIG)})
    cwd = Path.cwd()

    with (
        patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=RuntimeError("study blew up")),
        pytest.raises(RuntimeError, match="study blew up"),
    ):
        OptimizeJob().run(
            {
                "optimize_config": "optimize.yml",
                "optimize_config_fileset": "opt-bundle",
                "workspace": "default",
            },
            ctx=ctx,
            sdk=sdk,
        )

    assert Path.cwd() == cwd


def test_run_rejects_a_staged_config_missing_from_the_fileset(ctx: JobContext) -> None:
    sdk = bundle_sdk({"other.yml": yaml.safe_dump(MINIMAL_CONFIG)})

    with pytest.raises(FileNotFoundError, match="was not found in fileset"):
        OptimizeJob().run(
            {
                "optimize_config": "optimize.yml",
                "optimize_config_fileset": "opt-bundle",
                "workspace": "default",
            },
            ctx=ctx,
            sdk=sdk,
        )


def test_run_rejects_a_staged_config_without_an_sdk(ctx: JobContext) -> None:
    with pytest.raises(LocalRunError, match="requires a 'sdk: NeMoPlatform'"):
        OptimizeJob().run(
            {
                "optimize_config": "optimize.yml",
                "optimize_config_fileset": "opt-bundle",
                "workspace": "default",
            },
            ctx=ctx,
        )


# ---------------------------------------------------------------------------
# run — dataset staged from its own fileset
# ---------------------------------------------------------------------------


def _config_with_dataset(dataset: Any) -> dict[str, Any]:
    return {
        **MINIMAL_CONFIG,
        "eval": {"general": {"dataset": dataset, "max_concurrency": 1}},
    }


def test_run_stages_dataset_from_fileset_ref(tmp_path: Path, ctx: JobContext) -> None:
    downloaded: dict[str, Any] = {}

    class _StubFiles:
        def download(self, *, local_path: str, fileset: str, workspace: str) -> None:
            downloaded.update(fileset=fileset, workspace=workspace)
            (Path(local_path) / "rows.json").write_text('[{"question": "q", "answer": "a"}]')

    class _StubSDK:
        files = _StubFiles()

    optimize_config = write_config(tmp_path, _config_with_dataset({"file_path": "default/evals#rows.json"}))

    with patch(
        "nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", return_value={"status": "completed"}
    ) as dispatch:
        OptimizeJob().run(
            {"optimize_config": optimize_config, "workspace": "default"},
            ctx=ctx,
            sdk=cast(NeMoPlatform, _StubSDK()),
        )

    assert downloaded == {"fileset": "evals", "workspace": "default"}
    staged = dispatch.call_args.kwargs["optimize_config"]["eval"]["general"]["dataset"]["file_path"]
    assert staged.endswith("rows.json")
    assert Path(staged).is_absolute()
    # Sibling keys survive the rewrite.
    assert dispatch.call_args.kwargs["optimize_config"]["eval"]["general"]["max_concurrency"] == 1


def test_run_leaves_plain_dataset_path_untouched(tmp_path: Path, ctx: JobContext) -> None:
    optimize_config = write_config(tmp_path, _config_with_dataset({"file_path": "/data/rows.json"}))

    with patch(
        "nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", return_value={"status": "completed"}
    ) as dispatch:
        OptimizeJob().run({"optimize_config": optimize_config, "workspace": "default"}, ctx=ctx)

    dataset = dispatch.call_args.kwargs["optimize_config"]["eval"]["general"]["dataset"]
    assert dataset == {"file_path": "/data/rows.json"}


# ---------------------------------------------------------------------------
# run — publishing study artifacts
# ---------------------------------------------------------------------------


def _write_study_artifacts(ctx: JobContext) -> Path:
    """Stand in for what a backend leaves behind under <persistent>/results."""
    artifacts = ctx.storage.persistent / "results" / "optimizer_results"
    artifacts.mkdir(parents=True)
    (artifacts / "study_summary.json").write_text('{"status": "completed"}')
    (artifacts / "optimized_config.yml").write_text("optimizer: {}\n")
    return artifacts


def test_run_publishes_results_to_fileset(tmp_path: Path, ctx: JobContext) -> None:
    uploaded: dict[str, Any] = {}

    class _StubFiles:
        def upload(self, *, local_path: str, fileset: str, workspace: str, fileset_auto_create: bool) -> Any:
            uploaded.update(
                local_path=local_path,
                fileset=fileset,
                workspace=workspace,
                auto_create=fileset_auto_create,
                names=sorted(p.name for p in Path(local_path).rglob("*") if p.is_file()),
            )
            return SimpleNamespace(name=fileset)

    class _StubSDK:
        files = _StubFiles()

    optimize_config = write_config(tmp_path, MINIMAL_CONFIG)

    def _dispatch(**kwargs: Any) -> dict[str, Any]:
        _write_study_artifacts(ctx)
        return {"status": "completed", "best_trial": 3}

    with patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=_dispatch):
        result = OptimizeJob().run(
            {"optimize_config": optimize_config, "workspace": "default", "output": "tuned-results"},
            ctx=ctx,
            sdk=cast(NeMoPlatform, _StubSDK()),
        )

    assert uploaded["fileset"] == "tuned-results"
    assert uploaded["workspace"] == "default"
    assert uploaded["auto_create"] is True
    # Trailing slash uploads contents, not the dir itself.
    assert uploaded["local_path"].endswith("/")
    assert uploaded["names"] == ["optimized_config.yml", "study_summary.json"]
    # The study's own summary survives alongside the new pointer.
    assert result["best_trial"] == 3
    assert result["output"] == {"type": "fileset", "fileset": "default/tuned-results"}


def test_run_publishes_results_to_local_dir(tmp_path: Path, ctx: JobContext) -> None:
    dest = tmp_path / "published"
    optimize_config = write_config(tmp_path, MINIMAL_CONFIG)

    def _dispatch(**kwargs: Any) -> dict[str, Any]:
        _write_study_artifacts(ctx)
        return {"status": "completed"}

    with patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=_dispatch):
        result = OptimizeJob().run(
            {"optimize_config": optimize_config, "workspace": "default", "output": str(dest)},
            ctx=ctx,
        )

    assert (dest / "optimizer_results" / "study_summary.json").is_file()
    assert result["output"] == {"type": "local_dir", "path": str(dest.resolve())}


def test_run_publishes_staged_results_after_leaving_the_bundle(ctx: JobContext, tmp_path: Path) -> None:
    """Publishing happens outside the chdir, so a relative --output lands where the caller meant."""
    sdk_bundle = bundle_sdk({"optimize.yml": yaml.safe_dump(MINIMAL_CONFIG)})
    dest = tmp_path / "published"

    def _dispatch(**kwargs: Any) -> dict[str, Any]:
        _write_study_artifacts(ctx)
        return {"status": "completed"}

    with patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=_dispatch):
        result = OptimizeJob().run(
            {
                "optimize_config": "optimize.yml",
                "optimize_config_fileset": "opt-bundle",
                "workspace": "default",
                "output": str(dest),
            },
            ctx=ctx,
            sdk=sdk_bundle,
        )

    assert (dest / "optimizer_results" / "study_summary.json").is_file()
    assert result["output"] == {"type": "local_dir", "path": str(dest.resolve())}


def test_run_without_output_publishes_nothing(tmp_path: Path, ctx: JobContext) -> None:
    optimize_config = write_config(tmp_path, MINIMAL_CONFIG)

    def _dispatch(**kwargs: Any) -> dict[str, Any]:
        _write_study_artifacts(ctx)
        return {"status": "completed"}

    with patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=_dispatch):
        result = OptimizeJob().run({"optimize_config": optimize_config, "workspace": "default"}, ctx=ctx)

    assert result == {"status": "completed"}


def test_run_does_not_publish_when_study_fails(tmp_path: Path, ctx: JobContext) -> None:
    """A crashed study must not leave partial artifacts in the target fileset."""

    class _StubFiles:
        def upload(self, **kwargs: Any) -> Any:
            raise AssertionError("upload must not run when the study raises")

    class _StubSDK:
        files = _StubFiles()

    optimize_config = write_config(tmp_path, MINIMAL_CONFIG)

    with (
        patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=RuntimeError("study blew up")),
        pytest.raises(RuntimeError, match="study blew up"),
    ):
        OptimizeJob().run(
            {"optimize_config": optimize_config, "workspace": "default", "output": "tuned-results"},
            ctx=ctx,
            sdk=cast(NeMoPlatform, _StubSDK()),
        )


def test_run_rejects_fileset_output_without_sdk(tmp_path: Path, ctx: JobContext) -> None:
    optimize_config = write_config(tmp_path, MINIMAL_CONFIG)

    def _dispatch(**kwargs: Any) -> dict[str, Any]:
        _write_study_artifacts(ctx)
        return {"status": "completed"}

    with (
        patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", side_effect=_dispatch),
        pytest.raises(LocalRunError, match="requires a 'sdk: NeMoPlatform'"),
    ):
        OptimizeJob().run(
            {"optimize_config": optimize_config, "workspace": "default", "output": "tuned-results"},
            ctx=ctx,
        )


def test_run_reports_missing_artifacts_on_publish(tmp_path: Path, ctx: JobContext) -> None:
    """A backend that claims success but writes nothing is a bug, not an empty upload."""
    optimize_config = write_config(tmp_path, MINIMAL_CONFIG)

    with (
        patch("nemo_optimization.jobs.optimize.OptimizeRouter.dispatch", return_value={"status": "completed"}),
        pytest.raises(FileNotFoundError, match="wrote no artifacts"),
    ):
        OptimizeJob().run(
            {
                "optimize_config": optimize_config,
                "workspace": "default",
                "output": str(tmp_path / "published"),
            },
            ctx=ctx,
        )


def test_optimize_task_module_is_importable() -> None:
    """Both executors invoke this module by name; a rename must fail here, not in a job."""
    import importlib

    from nemo_optimization.jobs.optimize import OPTIMIZE_TASK_MODULE

    assert importlib.import_module(OPTIMIZE_TASK_MODULE) is not None
