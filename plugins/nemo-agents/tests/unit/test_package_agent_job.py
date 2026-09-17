# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from nemo_agents_plugin.entities import NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT, Agent
from nemo_agents_plugin.jobs.package_agent import (
    PACKAGE_RESULT_NAME,
    PackageAgentInput,
    PackageAgentJob,
    PackageAgentSpec,
)
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError
from nemo_platform_plugin.jobs.exceptions import (
    PlatformJobCompilationError,
    PlatformJobDependencyUnavailableError,
)
from nemo_platform_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    DockerJobExecutionProfileConfig,
    SubprocessJobExecutionProfile,
)
from pydantic import ValidationError

FABRIC_CONFIG: dict[str, Any] = {"config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT, "harness": {"type": "deepagents"}}


class _StubEntityClient:
    def __init__(self, agent: Agent | None) -> None:
        self._agent = agent

    async def get(self, _cls: type, *, name: str, workspace: str) -> Agent:
        if self._agent is None:
            raise NemoEntityNotFoundError(f"no agent {name} in {workspace}")
        return self._agent


def _agent(config_format: str = NEMO_AGENTS_SPEC_CONFIG_FORMAT) -> Agent:
    return Agent(name="my-agent", workspace="default", config=FABRIC_CONFIG, config_format=config_format)


def _subprocess_profile(profile: str = "default") -> SubprocessJobExecutionProfile:
    return SubprocessJobExecutionProfile(profile=profile)


def _docker_profile(profile: str = "default") -> DockerJobExecutionProfile:
    return DockerJobExecutionProfile(profile=profile, config=DockerJobExecutionProfileConfig())


def _stub_ctx(tmp_path: Path) -> Any:
    ctx = MagicMock()
    ctx.storage.ephemeral = tmp_path
    ctx.results.save = MagicMock(return_value=MagicMock(model_dump=lambda: {"name": PACKAGE_RESULT_NAME}))
    return ctx


def _patch_profiles(profiles: list[Any] | Exception):
    """Patch the jobs-client lookup ``compile`` uses to discover execution profiles."""
    client = MagicMock()
    if isinstance(profiles, Exception):
        client.get_execution_profiles = MagicMock(side_effect=profiles)
    else:
        response = MagicMock()
        response.data = MagicMock(return_value=profiles)

        async def _get() -> Any:
            return response

        client.get_execution_profiles = _get
    return patch("nemo_agents_plugin.jobs.package_agent.client_from_platform", return_value=client)


class TestToSpec:
    async def test_resolves_agent_config_inline(self) -> None:
        spec = await PackageAgentJob.to_spec(
            PackageAgentInput(agent="my-agent"),
            workspace="default",
            entity_client=_StubEntityClient(_agent()),
            async_sdk=None,
            is_local=False,
        )

        assert isinstance(spec, PackageAgentSpec)
        assert spec.workspace == "default"
        assert spec.agent_config == FABRIC_CONFIG

    async def test_missing_agent_fails_at_submit(self) -> None:
        with pytest.raises(PlatformJobCompilationError, match="not found in workspace"):
            await PackageAgentJob.to_spec(
                PackageAgentInput(agent="ghost"),
                workspace="default",
                entity_client=_StubEntityClient(None),
                async_sdk=None,
                is_local=False,
            )

    async def test_nat_workflow_agent_is_rejected(self) -> None:
        with pytest.raises(PlatformJobCompilationError, match="packaging supports"):
            await PackageAgentJob.to_spec(
                PackageAgentInput(agent="my-agent"),
                workspace="default",
                entity_client=_StubEntityClient(_agent(NAT_WORKFLOW_CONFIG_FORMAT)),
                async_sdk=None,
                is_local=False,
            )

    async def test_knobs_survive_resolution(self) -> None:
        spec = await PackageAgentJob.to_spec(
            PackageAgentInput(agent="my-agent", tag="custom:1.0", allow_root=True, python_version="3.12"),
            workspace="default",
            entity_client=_StubEntityClient(_agent()),
            async_sdk=None,
            is_local=False,
        )

        assert isinstance(spec, PackageAgentSpec)
        assert (spec.tag, spec.allow_root, spec.python_version) == ("custom:1.0", True, "3.12")


class TestCompile:
    async def _compile(self, **kwargs: Any) -> Any:
        return await PackageAgentJob.compile(
            workspace="default",
            spec=PackageAgentSpec(agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG),
            entity_client=None,
            job_name=None,
            async_sdk=MagicMock(),
            **kwargs,
        )

    async def test_emits_host_subprocess_step(self) -> None:
        with _patch_profiles([_subprocess_profile()]):
            platform_spec = await self._compile()

        step = next(iter(platform_spec["steps"]))
        assert step["executor"]["provider"] == "subprocess"
        assert step["executor"]["command"] == ["python", "-m", "nemo_agents_plugin.tasks.package"]

    async def test_url_workspace_overwrites_spec_workspace(self) -> None:
        with _patch_profiles([_subprocess_profile()]):
            platform_spec = await PackageAgentJob.compile(
                workspace="default",
                spec=PackageAgentSpec(agent="my-agent", workspace="attacker", agent_config=FABRIC_CONFIG),
                entity_client=None,
                job_name=None,
                async_sdk=MagicMock(),
            )

        step = next(iter(platform_spec["steps"]))
        assert step["config"]["workspace"] == "default"

    async def test_non_subprocess_backend_is_rejected(self) -> None:
        with _patch_profiles([_docker_profile()]):
            with pytest.raises(PlatformJobCompilationError, match="does not resolve to a subprocess backend"):
                await self._compile()

    async def test_rejection_names_the_local_fallback(self) -> None:
        with _patch_profiles([]):
            with pytest.raises(PlatformJobCompilationError, match="nemo agents package"):
                await self._compile()

    async def test_named_profile_must_match(self) -> None:
        with _patch_profiles([_subprocess_profile("other")]):
            with pytest.raises(PlatformJobCompilationError):
                await self._compile(profile="default")

    async def test_unreachable_jobs_service_is_retryable(self) -> None:
        with _patch_profiles(RuntimeError("connection refused")):
            with pytest.raises(PlatformJobDependencyUnavailableError, match="Retry the submission"):
                await self._compile()

    async def test_missing_sdk_does_not_promise_a_retry(self) -> None:
        with pytest.raises(PlatformJobDependencyUnavailableError, match="resubmitting will not help"):
            await PackageAgentJob.compile(
                workspace="default",
                spec=PackageAgentSpec(agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG),
                entity_client=None,
                job_name=None,
                async_sdk=None,
            )


class TestRun:
    def test_builds_from_staged_ethos_fileset(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        async def _stage(
            *, workspace: str, agent_name: str, agent_config: dict, base_dir: Path, files_client: Any
        ) -> None:
            captured["workspace"] = workspace
            captured["agent_name"] = agent_name
            (base_dir / "skills").mkdir()
            (base_dir / "skills" / "SKILL.md").write_text("# staged\n")

        def _build(agent_config_path: Path, **kwargs: Any) -> str:
            captured["config_path"] = agent_config_path
            captured["rendered"] = yaml.safe_load(agent_config_path.read_text())
            captured["staged_skill"] = (agent_config_path.parent / "skills" / "SKILL.md").exists()
            captured["kwargs"] = kwargs
            return "my-agent-abc123:26.08.21"

        with (
            patch("nemo_agents_plugin.runner.fabric_artifact_staging.stage_fabric_ethos_dir", _stage),
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image", _build),
        ):
            result = PackageAgentJob().run(
                PackageAgentSpec(
                    agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG, tag="my-agent-abc123:26.08.21"
                ).model_dump(mode="json"),
            )

        assert result == {"image": "my-agent-abc123:26.08.21", "agent": "my-agent", "published": ""}
        assert captured["workspace"] == "default"
        assert captured["rendered"] == FABRIC_CONFIG
        assert captured["staged_skill"], "agent.yaml must be written after staging clears the tree"

    def test_build_context_is_removed(self, tmp_path: Path) -> None:
        seen: dict[str, Path] = {}

        async def _stage(**kwargs: Any) -> None:
            return None

        def _build(agent_config_path: Path, **kwargs: Any) -> str:
            seen["dir"] = agent_config_path.parent
            return "x:latest"

        with (
            patch("nemo_agents_plugin.runner.fabric_artifact_staging.stage_fabric_ethos_dir", _stage),
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image", _build),
        ):
            PackageAgentJob().run(
                PackageAgentSpec(agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG).model_dump(
                    mode="json"
                ),
            )

        assert not seen["dir"].exists()

    def test_image_tag_is_published_through_the_results_api(self, tmp_path: Path) -> None:
        async def _stage(**kwargs: Any) -> None:
            return None

        ctx = _stub_ctx(tmp_path)

        with (
            patch("nemo_agents_plugin.runner.fabric_artifact_staging.stage_fabric_ethos_dir", _stage),
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image", lambda p, **k: "my-agent:1.0"),
        ):
            result = PackageAgentJob().run(
                PackageAgentSpec(agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG).model_dump(
                    mode="json"
                ),
                ctx=ctx,
            )

        name, path = ctx.results.save.call_args.args
        assert name == PACKAGE_RESULT_NAME
        assert json.loads(Path(path).read_text()) == {
            "image": "my-agent:1.0",
            "agent": "my-agent",
            "published": "",
        }
        assert result["status"] == "completed"
        assert result["image"] == "my-agent:1.0"
        assert result[PACKAGE_RESULT_NAME] == {"name": PACKAGE_RESULT_NAME}

    def test_staged_dockerignore_cannot_exclude_required_artifacts(self, tmp_path: Path) -> None:
        seen: dict[str, bool] = {}

        async def _stage(*, base_dir: Path, **kwargs: Any) -> None:
            (base_dir / ".dockerignore").write_text("agent.yaml\nskills/\n")

        def _build(agent_config_path: Path, **kwargs: Any) -> str:
            seen["ignore"] = (agent_config_path.parent / ".dockerignore").exists()
            return "x:latest"

        with (
            patch("nemo_agents_plugin.runner.fabric_artifact_staging.stage_fabric_ethos_dir", _stage),
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image", _build),
        ):
            PackageAgentJob().run(
                PackageAgentSpec(agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG).model_dump(
                    mode="json"
                ),
            )

        assert not seen["ignore"]

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [
            ("my-agent:1.0", "nemo-agents/default/my-agent:1.0"),
            (None, "nemo-agents/default/derived:1.0"),
        ],
    )
    def test_image_is_nested_under_the_workspace_namespace(self, tag: str | None, expected: str) -> None:
        captured: dict[str, Any] = {}

        async def _stage(**kwargs: Any) -> None:
            return None

        def _build(agent_config_path: Path, **kwargs: Any) -> str:
            captured["namespace"] = kwargs["tag_namespace"]
            captured["tag"] = kwargs["tag"]
            resolved = kwargs["tag"] or "derived:1.0"
            return f"{kwargs['tag_namespace']}/{resolved}"

        with (
            patch("nemo_agents_plugin.runner.fabric_artifact_staging.stage_fabric_ethos_dir", _stage),
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image", _build),
        ):
            result = PackageAgentJob().run(
                PackageAgentSpec(agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG, tag=tag).model_dump(
                    mode="json"
                ),
            )

        assert captured["namespace"] == "nemo-agents/default"
        assert result["image"] == expected

    def test_build_failure_propagates(self) -> None:
        from nemo_agents_plugin.container.errors import ImageBuildError

        async def _stage(**kwargs: Any) -> None:
            return None

        def _build(agent_config_path: Path, **kwargs: Any) -> str:
            raise ImageBuildError("Docker build failed: no such host")

        with (
            patch("nemo_agents_plugin.runner.fabric_artifact_staging.stage_fabric_ethos_dir", _stage),
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image", _build),
        ):
            with pytest.raises(ImageBuildError, match="no such host"):
                PackageAgentJob().run(
                    PackageAgentSpec(agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG).model_dump(
                        mode="json"
                    ),
                )


class TestTaskEntrypointWiring:
    """The build context comes from an async fileset download, so the entrypoint
    must hand ``run`` an async SDK — without it the image silently loses skills."""

    def test_entrypoint_passes_async_sdk(self) -> None:
        import nemo_agents_plugin.tasks.package.__main__ as entrypoint

        with (
            patch.object(entrypoint, "get_task_sdk", return_value="sync"),
            patch.object(entrypoint, "get_async_task_sdk", return_value="async") as async_sdk,
            patch.object(entrypoint, "run_task", return_value=0) as run_task,
        ):
            assert entrypoint.main() == 0

        async_sdk.assert_called_once_with("agents")
        assert run_task.call_args.kwargs["async_sdk"] == "async"

    def test_missing_client_warns_about_dropped_artifacts(self, caplog) -> None:
        async def _noop(**kwargs: Any) -> None:
            return None

        with (
            patch("nemo_agents_plugin.runner.fabric_artifact_staging.stage_fabric_ethos_dir", _noop),
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image", lambda p, **k: "x:latest"),
            caplog.at_level("WARNING"),
        ):
            PackageAgentJob().run(
                PackageAgentSpec(agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG).model_dump(
                    mode="json"
                ),
                async_sdk=None,
            )

        assert "my-agent-ethos" in caplog.text


class TestDockerfileInjection:
    """`base_image_*` / `*_version` are interpolated into the Dockerfile unescaped,
    and the submitter is a remote caller — a newline would append build instructions."""

    @pytest.mark.parametrize(
        "field",
        ["base_image_url", "base_image_tag", "python_version", "uv_version"],
    )
    def test_newline_payload_is_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            payload: dict[str, Any] = {field: "noble\nRUN echo pwned"}
            PackageAgentInput(agent="my-agent", **payload)

    @pytest.mark.parametrize(
        "field",
        ["base_image_url", "base_image_tag", "python_version", "uv_version"],
    )
    def test_whitespace_payload_is_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            payload: dict[str, Any] = {field: "noble RUN echo pwned"}
            PackageAgentInput(agent="my-agent", **payload)

    def test_realistic_values_are_accepted(self) -> None:
        spec = PackageAgentInput(
            agent="my-agent",
            base_image_url="nvcr.io/nvidia/base/ubuntu",
            base_image_tag="noble-20260217",
            python_version="3.13",
            uv_version="0.9.14",
        )
        assert spec.base_image_url == "nvcr.io/nvidia/base/ubuntu"

    def test_registry_with_port_is_accepted(self) -> None:
        assert PackageAgentInput(agent="a", base_image_url="localhost:5000/base/ubuntu").base_image_url


class TestCliSurface:
    def test_job_name_does_not_shadow_the_local_package_command(self) -> None:
        """`nemo agents package` is an existing command; the generated job sub-group
        mounts by `name` onto the same Typer app and would win."""
        assert PackageAgentJob.name != "package"

    def test_rest_collection_path_is_pinned(self) -> None:
        assert PackageAgentJob.job_collection_path == "/jobs/package"


class TestPublishedPackagingContract:
    """The generated `nemo-platform[nemo-agents-plugin]` extra mirrors this plugin's
    base dependencies only, so packaging deps declared as an extra would leave a
    PyPI install advertising `agents.package-agent` but unable to run it."""

    @pytest.mark.parametrize("dependency", ["python-on-whales", "jinja2"])
    def test_packaging_dependency_is_a_base_requirement(self, dependency: str) -> None:
        import tomllib

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
        assert any(spec.startswith(dependency) for spec in declared)

    @pytest.mark.parametrize("dependency", ["python-on-whales", "jinja2"])
    def test_published_platform_extra_includes_packaging_dependency(self, dependency: str) -> None:
        import tomllib

        repo_root = Path(__file__).resolve().parents[4]
        pyproject = repo_root / "packages" / "nemo_platform" / "pyproject.toml"
        if not pyproject.exists():
            pytest.skip("vendored nemo-platform wrapper is not present in this checkout")
        extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
        assert any(spec.startswith(dependency) for spec in extras["nemo-agents-plugin"])


class TestTagNamespace:
    """Docker tags are daemon-global; the auth boundary is the workspace."""

    @pytest.mark.parametrize(
        "tag",
        [
            "nvcr.io/nvidia/nemo-platform:latest",
            "other-workspace/my-agent:1.0",
            "../escape:1.0",
            "my-agent:1.0\nRUN echo pwned",
        ],
    )
    def test_tag_cannot_escape_the_namespace(self, tag: str) -> None:
        with pytest.raises(ValidationError):
            PackageAgentInput(agent="my-agent", tag=tag)

    @pytest.mark.parametrize("tag", ["my-agent", "my-agent:1.0", "my.agent_v2:26.08.21"])
    def test_plain_names_are_accepted(self, tag: str) -> None:
        assert PackageAgentInput(agent="my-agent", tag=tag).tag == tag

    async def test_workspace_outside_the_docker_grammar_is_rejected(self) -> None:
        with _patch_profiles([_subprocess_profile()]):
            with pytest.raises(PlatformJobCompilationError, match="cannot be used as an image namespace"):
                await PackageAgentJob.compile(
                    workspace="team+eng",
                    spec=PackageAgentSpec(agent="my-agent", workspace="team+eng", agent_config=FABRIC_CONFIG),
                    entity_client=None,
                    job_name=None,
                    async_sdk=MagicMock(),
                )


class TestPublish:
    @staticmethod
    def _run(spec: PackageAgentSpec, push: Any, *, image_id: str = "sha256:deadbeef", resolve_id: Any = None) -> dict:
        async def _stage(**kwargs: Any) -> None:
            return None

        with (
            patch("nemo_agents_plugin.runner.fabric_artifact_staging.stage_fabric_ethos_dir", _stage),
            patch("nemo_agents_plugin.container.builder.build_fabric_agent_image", lambda p, **k: "my-agent:1.0"),
            patch("nemo_agents_plugin.container.builder.resolve_image_id", resolve_id or (lambda tag: image_id)),
            patch("nemo_agents_plugin.container.publisher.docker_push", push),
        ):
            return PackageAgentJob().run(spec.model_dump(mode="json"))

    def test_no_registry_skips_the_push(self) -> None:
        push = MagicMock()
        resolve_id = MagicMock()
        result = self._run(
            PackageAgentSpec(agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG),
            push,
            resolve_id=resolve_id,
        )

        push.assert_not_called()
        # No registry to push to, so there's nothing to resolve an image ID for either.
        resolve_id.assert_not_called()
        assert result["published"] == ""
        assert result["image"] == "my-agent:1.0"

    def test_registry_pushes_the_built_tag(self) -> None:
        push = MagicMock(return_value="nvcr.io/my-org/my-agent:1.0")
        result = self._run(
            PackageAgentSpec(
                agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG, registry="nvcr.io/my-org"
            ),
            push,
        )

        assert push.call_args.kwargs["local_tag"] == "my-agent:1.0"
        assert push.call_args.kwargs["registry"] == "nvcr.io/my-org"
        assert result["published"] == "nvcr.io/my-org/my-agent:1.0"

    def test_push_publishes_by_resolved_image_id_not_the_mutable_tag(self) -> None:
        """Guards the tag/push race: a concurrent job could rebind the shared local tag

        between build and push, so the push must address the image by the ID resolved
        right after *this* job's own build — not by the (daemon-global) tag name.
        """
        push = MagicMock(return_value="nvcr.io/my-org/my-agent:1.0")
        self._run(
            PackageAgentSpec(
                agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG, registry="nvcr.io/my-org"
            ),
            push,
            image_id="sha256:abc123",
        )

        assert push.call_args.kwargs["source_ref"] == "sha256:abc123"
        # local_tag is still forwarded — used for the default push_tag / progress text.
        assert push.call_args.kwargs["local_tag"] == "my-agent:1.0"

    def test_explicit_push_tag_is_forwarded(self) -> None:
        push = MagicMock(return_value="nvcr.io/my-org/nemo-agents/default/renamed:2.0")
        self._run(
            PackageAgentSpec(
                agent="my-agent",
                workspace="default",
                agent_config=FABRIC_CONFIG,
                registry="nvcr.io/my-org",
                push_tag="nvcr.io/my-org/nemo-agents/default/renamed:2.0",
            ),
            push,
        )

        assert push.call_args.kwargs["push_tag"] == "nvcr.io/my-org/nemo-agents/default/renamed:2.0"

    def test_push_failure_propagates(self) -> None:
        from nemo_agents_plugin.container.errors import ImagePublishError

        push = MagicMock(side_effect=ImagePublishError("Docker push failed: denied"))
        with pytest.raises(ImagePublishError, match="denied"):
            self._run(
                PackageAgentSpec(
                    agent="my-agent", workspace="default", agent_config=FABRIC_CONFIG, registry="nvcr.io/my-org"
                ),
                push,
            )


class TestPublishInputValidation:
    def test_push_tag_without_registry_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires 'registry'"):
            PackageAgentInput(agent="my-agent", push_tag="nvcr.io/my-org/x:1.0")

    @pytest.mark.parametrize("field", ["registry", "push_tag"])
    def test_control_characters_are_rejected(self, field: str) -> None:
        payload: dict[str, Any] = {"registry": "nvcr.io/my-org", field: "nvcr.io/my-org\nRUN echo pwned"}
        with pytest.raises(ValidationError):
            PackageAgentInput(agent="my-agent", **payload)

    def test_realistic_references_are_accepted(self) -> None:
        spec = PackageAgentInput(
            agent="my-agent", registry="localhost:5000/team", push_tag="localhost:5000/team/my-agent:1.0"
        )

        assert spec.registry == "localhost:5000/team"


class TestPushTagWorkspaceScoping:
    def test_push_tag_outside_the_workspace_namespace_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be nested under 'nvcr.io/my-org/nemo-agents/default/'"):
            PackageAgentSpec(
                agent="my-agent",
                workspace="default",
                agent_config=FABRIC_CONFIG,
                registry="nvcr.io/my-org",
                push_tag="nvcr.io/my-org/renamed:2.0",
            )

    def test_push_tag_under_another_workspace_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be nested under 'nvcr.io/my-org/nemo-agents/default/'"):
            PackageAgentSpec(
                agent="my-agent",
                workspace="default",
                agent_config=FABRIC_CONFIG,
                registry="nvcr.io/my-org",
                push_tag="nvcr.io/my-org/nemo-agents/other-workspace/my-agent:1.0",
            )

    def test_push_tag_under_a_different_registry_is_rejected(self) -> None:
        """The workspace-namespace segment alone isn't enough — push_tag must also start

        with the declared 'registry', or it could silently redirect to a different
        registry the host happens to also be authenticated to.
        """
        with pytest.raises(ValidationError, match="must be nested under 'nvcr.io/my-org/nemo-agents/default/'"):
            PackageAgentSpec(
                agent="my-agent",
                workspace="default",
                agent_config=FABRIC_CONFIG,
                registry="nvcr.io/my-org",
                push_tag="other-registry.example.com/nemo-agents/default/renamed:2.0",
            )

    def test_push_tag_with_extra_nesting_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be nested under 'nvcr.io/my-org/nemo-agents/default/'"):
            PackageAgentSpec(
                agent="my-agent",
                workspace="default",
                agent_config=FABRIC_CONFIG,
                registry="nvcr.io/my-org",
                push_tag="nvcr.io/my-org/nemo-agents/default/nested/renamed:2.0",
            )

    def test_push_tag_scoped_to_the_workspace_is_accepted(self) -> None:
        spec = PackageAgentSpec(
            agent="my-agent",
            workspace="default",
            agent_config=FABRIC_CONFIG,
            registry="nvcr.io/my-org",
            push_tag="nvcr.io/my-org/nemo-agents/default/renamed:2.0",
        )

        assert spec.push_tag == "nvcr.io/my-org/nemo-agents/default/renamed:2.0"

    def test_push_tag_without_workspace_is_not_checked(self) -> None:
        """No workspace to scope against yet (e.g. constructing a bare PackageAgentInput)."""
        spec = PackageAgentSpec(
            agent="my-agent",
            agent_config=FABRIC_CONFIG,
            registry="nvcr.io/my-org",
            push_tag="nvcr.io/my-org/renamed:2.0",
        )

        assert spec.push_tag == "nvcr.io/my-org/renamed:2.0"
