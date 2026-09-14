# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PackageAgentJob — build a container image for an agent that lives on the platform.

Registered under ``nemo.jobs`` as ``agents.package-agent``, serving
``/jobs/package``. The REST equivalent of
``nemo agents package``, for agents whose source of truth is the
``{agent}-ethos`` fileset rather than a directory on the submitter's laptop.

The build runs as a **host subprocess**, not in a container — the same shape
every other agents job uses. That is deliberate: the Fabric Dockerfile relies
on BuildKit cache mounts, so the step needs a real Docker CLI, and the platform
host running ``nemo services run`` already has one. It also means packaging is
unavailable wherever the subprocess executor isn't registered (notably
``runtime = kubernetes``); :meth:`PackageAgentJob.compile` rejects those
submissions up front rather than letting them fail opaquely at schedule time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import yaml
from nemo_agents_plugin.entities import (
    AGENT_CONFIG_FILENAME,
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    Agent,
    ethos_fileset_name,
)
from nemo_platform import AsyncNeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.job_results import ResultRef
from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from nemo_platform_plugin.jobs.client import AsyncJobsClient
from nemo_platform_plugin.jobs.exceptions import (
    PlatformJobCompilationError,
    PlatformJobDependencyUnavailableError,
)
from nemo_platform_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

_SUBPROCESS_PROVIDER: Literal["subprocess"] = "subprocess"
_DEFAULT_PROFILE = "default"

PACKAGE_RESULT_NAME = "package_result"

#: Docker reference grammar, anchored so no newline can smuggle in a build instruction.
#: Optional registry host with port, then one or more path components.
IMAGE_REPOSITORY_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(:[0-9]+)?(/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$"
IMAGE_TAG_PATTERN = r"^[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}$"
VERSION_PATTERN = r"^[0-9]+(\.[0-9]+){0,2}$"
#: The repository grammar with an optional ``:tag``, so the trailing ``$`` is dropped first.
IMAGE_REFERENCE_PATTERN = IMAGE_REPOSITORY_PATTERN[:-1] + r"(:[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127})?$"

#: Docker tags are daemon-global but the auth boundary is the workspace, so every
#: image is nested under ``{TAG_NAMESPACE}/{workspace}/``.
TAG_NAMESPACE = "nemo-agents"

#: One name component plus optional tag — no ``/``, so a submitted value cannot
#: climb out of that namespace.
IMAGE_NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]*(:[a-zA-Z0-9_][a-zA-Z0-9._-]{0,127})?$"

#: Narrower than the platform's entity names, which still permit ``@`` and ``+``.
_DOCKER_PATH_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

_HOST_BUILD_REQUIREMENT = (
    "Agent packaging builds an image with the host Docker CLI, so it requires a "
    "registered subprocess execution profile. Build locally instead with "
    "`nemo agents package --agent <agent.yaml>` and pass the resulting tag to "
    "`nemo agents deploy --image`."
)


class PackageAgentInput(BaseModel):
    """What a caller POSTs: an agent name plus the packaging knobs."""

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(description="Name of the Agent entity to package.")
    tag: str | None = Field(
        default=None,
        pattern=IMAGE_NAME_PATTERN,
        description=(
            "Image name and optional tag, always nested under "
            "'nemo-agents/{workspace}/'. Defaults to "
            "'{agent_name}-{agent_id}:{agent_version}'."
        ),
    )
    # These four are interpolated into the rendered Dockerfile unescaped
    # (``ARG BASE_IMAGE_URL={{ base_image_url }}``), and unlike the CLI the
    # submitter here is a remote API caller. Without a strict grammar a value
    # containing a newline appends attacker-chosen build instructions that the
    # host daemon then executes. Constrain them at the API boundary.
    base_image_url: str | None = Field(
        default=None,
        pattern=IMAGE_REPOSITORY_PATTERN,
        description="Base image repository override (e.g. 'nvcr.io/nvidia/base/ubuntu').",
    )
    base_image_tag: str | None = Field(
        default=None,
        pattern=IMAGE_TAG_PATTERN,
        description="Base image tag override (e.g. 'noble-20260217').",
    )
    python_version: str | None = Field(
        default=None,
        pattern=VERSION_PATTERN,
        description="Python version baked into the image (e.g. '3.13').",
    )
    uv_version: str | None = Field(
        default=None,
        pattern=VERSION_PATTERN,
        description="uv version baked into the image (e.g. '0.9.14').",
    )
    allow_root: bool = Field(default=False, description="Run the agent as root instead of the 'agent' user.")
    sandbox_runtime: str | None = Field(
        default=None, description="Render an image compatible with a sandbox runtime (e.g. 'openshell')."
    )
    agent_version: str | None = Field(default=None, description="OCI image version label override.")
    agent_author: str | None = Field(default=None, description="OCI image authors label override.")
    skip_validation: bool = Field(default=False, description="Bypass Fabric package validation before building.")
    registry: str | None = Field(
        default=None,
        pattern=IMAGE_REPOSITORY_PATTERN,
        description=(
            "Push the built image to this registry (e.g. 'nvcr.io/my-org'). The host executing "
            "the push must already be authenticated to it: the local machine for direct CLI "
            "invocation, or the platform's job-execution host for remote execution or the REST API. "
            "Credentials are never "
            "accepted over this API."
        ),
    )
    push_tag: str | None = Field(
        default=None,
        pattern=IMAGE_REFERENCE_PATTERN,
        description=(
            "Fully-qualified remote tag. Defaults to '<registry>/<image>', where <image> is the "
            "namespaced local reference 'nemo-agents/{workspace}/{tag}'. Requires 'registry'. Must "
            "start with '<registry>/nemo-agents/{workspace}/' — Docker tags are daemon-global while "
            "the auth boundary here is the workspace, so an unscoped push_tag would let this "
            "workspace overwrite another workspace's image, or redirect the push to a registry "
            "other than the one declared."
        ),
    )

    @model_validator(mode="after")
    def _push_tag_needs_a_registry(self) -> PackageAgentInput:
        if self.push_tag and not self.registry:
            raise ValueError("'push_tag' requires 'registry'.")
        return self


class PackageAgentSpec(PackageAgentInput):
    """Canonical spec — the agent's config resolved inline by :meth:`PackageAgentJob.to_spec`."""

    workspace: str = Field(default="", description="Workspace owning the agent and its spec fileset.")
    agent_config: dict[str, Any] = Field(default_factory=dict, description="Resolved agent config.")

    @model_validator(mode="after")
    def _push_tag_stays_in_the_workspace_namespace(self) -> PackageAgentSpec:
        # 'workspace' isn't known on PackageAgentInput (it comes from the URL / --workspace,
        # not the request body), so this check can only run here, once to_spec()/run_local()
        # has stamped it onto the spec.
        if self.push_tag and self.registry and self.workspace:
            # Anchored on 'registry' too, not just the 'nemo-agents/{workspace}/' segment —
            # otherwise push_tag could silently redirect to a registry other than the one
            # the caller declared (and is presumably authenticated to).
            expected_prefix = f"{self.registry.rstrip('/')}/{TAG_NAMESPACE}/{self.workspace}/"
            remote_name = self.push_tag.removeprefix(expected_prefix)
            if not self.push_tag.startswith(expected_prefix) or not remote_name or "/" in remote_name:
                raise ValueError(
                    f"'push_tag' must be nested under '{expected_prefix}' (e.g. "
                    f"'{expected_prefix}<name>'). Docker tags are daemon-global while the "
                    "auth boundary here is the workspace; an unscoped push_tag would let this "
                    "workspace overwrite another workspace's image on the shared host."
                )
        return self


class PackageAgentJob(NemoJob):
    """Build a container image for a platform-resident agent."""

    # NOT "package": the generated job sub-group mounts onto the same Typer app
    # that already owns `nemo agents package`, and the later registration wins —
    # naming this "package" makes the local packaging flags unreachable.
    # The REST collection path is pinned below so the API surface stays /jobs/package.
    name: ClassVar[str] = "package-agent"
    job_collection_path: ClassVar[str | None] = "/jobs/package"
    description: ClassVar[str] = "Build a container image for an agent stored on the platform."
    container: ClassVar[str] = "cpu-tasks"
    generate_legacy_verbs: ClassVar[bool] = False
    input_spec_schema: ClassVar[type[BaseModel] | None] = PackageAgentInput
    spec_schema: ClassVar[type[BaseModel] | None] = PackageAgentSpec

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: object,
        is_local: bool,
    ) -> BaseModel:
        """Resolve the agent name into its inline config.

        Reads the entity at submit time so a missing or non-Fabric agent fails
        the POST rather than the build.
        """
        del is_local
        assert isinstance(input_spec, PackageAgentInput), (
            f"PackageAgentJob.to_spec received unexpected input type: {type(input_spec).__name__}"
        )
        client = cls._resolve_entity_client(entity_client, async_sdk)
        try:
            agent = await client.get(Agent, name=input_spec.agent, workspace=workspace)
        except NemoEntityNotFoundError as exc:
            raise PlatformJobCompilationError(
                f"Agent '{input_spec.agent}' not found in workspace '{workspace}'."
            ) from exc

        if agent.config_format != NEMO_AGENTS_SPEC_CONFIG_FORMAT:
            raise PlatformJobCompilationError(
                f"Agent '{input_spec.agent}' has config_format '{agent.config_format}'; platform-side "
                f"packaging supports '{NEMO_AGENTS_SPEC_CONFIG_FORMAT}' only. NAT workflows package from "
                "a source checkout — use `nemo agents package` locally."
            )

        return PackageAgentSpec(
            **input_spec.model_dump(),
            workspace=workspace,
            agent_config=agent.config,
        )

    @staticmethod
    def _resolve_entity_client(entity_client: object, async_sdk: object) -> NemoEntitiesClient:
        """Return an entity client from whatever the scheduler injected."""
        if entity_client is not None:
            return cast(NemoEntitiesClient, entity_client)
        if async_sdk is not None:
            return NemoEntitiesClient(client_from_platform(cast(AsyncNeMoPlatform, async_sdk), AsyncEntitiesClient))
        raise PlatformJobCompilationError(
            "Packaging requires a platform client to resolve the agent entity, but none was injected."
        )

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> PlatformJobSpec:
        """Single-step PlatformJobSpec running ``nemo_agents_plugin.tasks.package`` on the host."""
        del entity_client, job_name, options
        from nemo_platform_plugin.jobs.api_factory import PlatformJobStep, SubprocessExecutionProviderSpec

        assert isinstance(spec, PackageAgentSpec), (
            f"PackageAgentJob.compile received unexpected spec type: {type(spec).__name__}"
        )
        cls._require_namespaceable_workspace(workspace)
        resolved_profile = profile or _DEFAULT_PROFILE
        await cls._require_subprocess_profile(resolved_profile, async_sdk)

        spec_dict = spec.model_dump(mode="json")
        # URL workspace is the auth boundary; overwrite whatever to_spec set.
        spec_dict["workspace"] = workspace

        return PlatformJobSpec(
            steps=[
                PlatformJobStep(
                    name="package",
                    executor=SubprocessExecutionProviderSpec(
                        provider=_SUBPROCESS_PROVIDER,
                        profile=resolved_profile,
                        command=["python", "-m", "nemo_agents_plugin.tasks.package"],
                    ),
                    config=spec_dict,
                ),
            ],
        )

    @staticmethod
    def _require_namespaceable_workspace(workspace: str) -> None:
        """Reject a workspace that cannot be spelled as a Docker path component."""
        if not _DOCKER_PATH_COMPONENT.match(workspace):
            raise PlatformJobCompilationError(
                f"Workspace '{workspace}' cannot be used as an image namespace: Docker path "
                "components allow only lowercase letters, digits, '.', '_' and '-'. Package this "
                "agent from a workspace with a Docker-safe name, or build locally with "
                "`nemo agents package`."
            )

    @staticmethod
    async def _require_subprocess_profile(profile: str, async_sdk: object) -> None:
        """Reject the submission unless *profile* resolves to a host subprocess backend."""
        if async_sdk is None:
            raise PlatformJobDependencyUnavailableError(
                f"Unable to resolve execution profile '{profile}': no platform client was injected. "
                "This is a scheduler wiring fault rather than a transient one — resubmitting will not "
                "help until the Jobs service is restarted with a platform client."
            )
        try:
            profiles = (
                await client_from_platform(cast(AsyncNeMoPlatform, async_sdk), AsyncJobsClient).get_execution_profiles()
            ).data()
        except Exception as exc:
            raise PlatformJobDependencyUnavailableError(
                f"Unable to resolve execution profile '{profile}': the Jobs service is temporarily "
                "unavailable. Retry the submission."
            ) from exc

        # The concrete profile type fixes the backend to "subprocess".
        if any(
            isinstance(execution_profile, SubprocessJobExecutionProfile) and execution_profile.profile == profile
            for execution_profile in profiles
        ):
            return

        raise PlatformJobCompilationError(
            f"Execution profile '{profile}' does not resolve to a subprocess backend. {_HOST_BUILD_REQUIREMENT}"
        )

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext | None = None,
        async_sdk: AsyncNeMoPlatform | None = None,
    ) -> dict:
        """Stage the agent's spec fileset into a temp build context, build, and optionally push."""
        from nemo_agents_plugin.container.builder import build_fabric_agent_image, resolve_image_id
        from nemo_agents_plugin.container.publisher import docker_push

        cfg = PackageAgentSpec.model_validate(config)
        with tempfile.TemporaryDirectory(prefix="nemo-agent-package-") as tmp:
            build_dir = Path(tmp)
            asyncio.run(self._stage(cfg, build_dir, async_sdk))
            self._drop_staged_dockerignore(build_dir)
            agent_config_path = build_dir / AGENT_CONFIG_FILENAME
            agent_config_path.write_text(
                yaml.safe_dump(cfg.agent_config, sort_keys=False),
                encoding="utf-8",
            )
            image = build_fabric_agent_image(
                agent_config_path,
                tag=cfg.tag,
                base_image_url=cfg.base_image_url,
                base_image_tag=cfg.base_image_tag,
                python_version=cfg.python_version,
                uv_version=cfg.uv_version,
                allow_root=cfg.allow_root,
                sandbox_runtime=cfg.sandbox_runtime,
                agent_version=cfg.agent_version,
                agent_author=cfg.agent_author,
                tag_namespace=f"{TAG_NAMESPACE}/{cfg.workspace}",
                skip_validation=cfg.skip_validation,
                on_progress=logger.info,
            )
            # Resolved immediately after the build call, not from `run()`-external
            # code between here and the push below: a concurrent job rebuilding
            # the same (daemon-global) tag name can't rebind what we publish.
            image_id = resolve_image_id(image) if cfg.registry else None

        published = ""
        if cfg.registry:
            published = docker_push(
                local_tag=image,
                registry=cfg.registry,
                push_tag=cfg.push_tag,
                source_ref=image_id,
                on_progress=logger.info,
            )

        payload = {"image": image, "agent": cfg.agent, "published": published}
        ref = self._save_package_result(ctx, payload)
        if ref is None:
            return payload
        return {"status": "completed", **payload, PACKAGE_RESULT_NAME: ref.model_dump()}

    @staticmethod
    def _drop_staged_dockerignore(build_dir: Path) -> None:
        """Discard a fileset-supplied ``.dockerignore``.

        Validation reads the staged tree off disk but Docker applies exclusions
        afterwards, so one excluding ``agent.yaml`` would validate, build, then
        fail at container start.
        """
        staged = build_dir / ".dockerignore"
        if not staged.exists():
            return
        logger.warning("Discarding .dockerignore from the Ethos fileset; the managed one is used instead.")
        staged.unlink()

    @staticmethod
    def _save_package_result(ctx: JobContext | None, payload: dict[str, str | None]) -> ResultRef | None:
        """Publish *payload* through the results API so ``/results`` can serve the tag."""
        if ctx is None:
            logger.warning("No job context available; the image tag will not be retrievable from the results API.")
            return None
        path = ctx.storage.ephemeral / f"{PACKAGE_RESULT_NAME}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return ctx.results.save(PACKAGE_RESULT_NAME, path)

    @staticmethod
    async def _stage(cfg: PackageAgentSpec, build_dir: Path, async_sdk: AsyncNeMoPlatform | None) -> None:
        """Download the ``{agent}-ethos`` fileset into *build_dir*.

        Must run before ``agent.yaml`` is written — staging clears the tree first.
        """
        from nemo_agents_plugin.runner.fabric_artifact_staging import stage_fabric_ethos_dir

        if async_sdk is None:
            logger.warning(
                "No platform client available; packaging agent %r from its stored agent.yaml alone. "
                "Skills, MCP servers, and prompts in the %r fileset will be missing from the image.",
                cfg.agent,
                ethos_fileset_name(cfg.agent),
            )

        await stage_fabric_ethos_dir(
            workspace=cfg.workspace,
            agent_name=cfg.agent,
            agent_config=cfg.agent_config,
            base_dir=build_dir,
            files_client=client_from_platform(async_sdk, AsyncFilesClient) if async_sdk is not None else None,
        )
