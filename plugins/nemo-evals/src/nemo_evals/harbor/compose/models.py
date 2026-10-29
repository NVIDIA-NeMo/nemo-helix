# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the compose parser returns and raises: the typed task, its error, and the collector of problems."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ComposeError(ValueError):
    """A task whose compose file can't run as services in a sandbox. ``code`` is stable and shown to users."""

    def __init__(self, code: str, detail: str, *, compose_path: str | None = None, paths: tuple[str, ...] = ()) -> None:
        """Build the message ``<code>: <detail> (<paths>) in <compose_path>``, and keep each part on the error."""
        message = f"{code}: {detail}"
        if paths:
            message += f" ({', '.join(paths)})"
        if compose_path:
            message += f" in {compose_path}"
        super().__init__(message)

        self.code = code
        self.detail = detail
        self.compose_path = compose_path
        self.paths = paths


class ComposeProblems:
    """Everything wrong with one compose file, collected while it is parsed and raised as one :class:`ComposeError`.

    Collecting rather than raising at the first problem lets an author fix everything in one pass. A problem is
    *unsupported* (valid Compose that a pod can't run) or *invalid* (something Compose itself would refuse).
    """

    def __init__(self) -> None:
        """Start with nothing found."""
        self._unsupported: list[str] = []
        self._invalid: list[str] = []

    def unsupported(self, path: str) -> None:
        """Record a feature at ``path`` that Compose has and a pod doesn't."""
        self._unsupported.append(path)

    def invalid(self, path: str, why: str) -> None:
        """Record a value at ``path`` that Compose itself would refuse, and why."""
        self._invalid.append(f"{path}: {why}")

    def raise_if_any(self, compose_path: str) -> None:
        """Raise one error for everything recorded; do nothing when nothing was.

        Any invalid value makes the error ``compose_invalid``, since the file is broken whatever runs it. Unsupported
        features found alongside are still listed in its paths, so the author sees every problem in one pass.
        """
        if not self._invalid and not self._unsupported:
            return

        unsupported = tuple(dict.fromkeys(self._unsupported))
        if not self._invalid:
            raise ComposeError(
                "compose_unsupported",
                "these can't run as sandbox services",
                compose_path=compose_path,
                paths=unsupported,
            )

        detail = "; ".join(self._invalid)
        if unsupported:
            detail += "; and these can't run as sandbox services"
        raise ComposeError("compose_invalid", detail, compose_path=compose_path, paths=unsupported)


class ComposeImage(BaseModel):
    """A service that runs a published image, as ``image: postgres:18``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["image"] = "image"
    reference: str = Field(description="The full reference, as Docker would pull it: `docker.io/library/postgres:18`.")


class ComposeBuild(BaseModel):
    """A service built from the archive, as ``build: {context: ./api}``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["build"] = "build"
    context_path: str = Field(
        description="The build context, relative to the task root and always under `environment/`: `environment/api`."
    )
    dockerfile: str = Field(default="Dockerfile", description="The Dockerfile, relative to `context_path`.")
    args: dict[str, str] = Field(default_factory=dict, description="Compose `build.args`: the build's `ARG` values.")
    target: str | None = Field(default=None, description="Compose `build.target`: the Dockerfile stage to build.")
    no_cache: bool = Field(default=False, description="Compose `build.no_cache`: build without the layer cache.")


ComposeImageSource = Annotated[ComposeImage | ComposeBuild, Field(discriminator="kind")]


class ComposePort(BaseModel):
    """A container port, and the compose key it came from (for error messages)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    port: int = Field(description="The port the container listens on; in a pod, every container shares it.")
    source: str = Field(description="The compose entry that declared it, such as `ports[0]`, to name it in errors.")


class ComposeReadiness(BaseModel):
    """A healthcheck command, and the longest Compose would let the service take to become healthy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exec_: list[str] = Field(
        alias="exec", min_length=1, description="The healthcheck's command, as an argument list run in the container."
    )
    budget_sec: float = Field(
        description=(
            "How long Compose would wait for the service to turn healthy, `start_period + interval * retries`. "
            "The sandbox waits at least this long before giving up."
        )
    )


class ComposeVolumeMount(BaseModel):
    """A named volume mounted into a container."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="A volume declared under the compose file's top-level `volumes`.")
    mount_path: str = Field(description="Where it appears inside the container.")
    read_only: bool = Field(default=False, description="Whether the mount ends in `:ro` or sets `read_only: true`.")


class ComposeService(BaseModel):
    """One service other than ``main``: its image source, how it starts, and how it runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="The service's compose name. Other containers in the pod reach it by this name.")
    source: ComposeImageSource = Field(description="Where its image comes from: a published image or a build.")
    platform: str | None = Field(
        default=None, description="Compose `platform`, as written: the platform its image is built or pulled for."
    )
    role: Literal["sidecar", "run_once", "after_main"] = Field(
        default="sidecar",
        description=(
            "When it runs, worked out from `depends_on`: `run_once` if something waits for it to finish, such as "
            "a migration; `after_main` if it waits for `main`; otherwise `sidecar`, which starts before `main` and "
            "keeps running."
        ),
    )
    depends_on: dict[str, str] = Field(
        default_factory=dict, description="Each service this one waits for, `main` included, and the compose condition."
    )
    command: list[str] | None = Field(default=None, description="Compose `entrypoint`; Kubernetes calls it `command`.")
    args: list[str] | None = Field(default=None, description="Compose `command`; Kubernetes calls it `args`.")
    env: dict[str, str] = Field(
        default_factory=dict, description="Its environment: its `env_file` entries, overridden by `environment`."
    )
    ports: list[ComposePort] = Field(
        default_factory=list, description="The ports it `expose`s or publishes, so the pod can check for clashes."
    )
    readiness: ComposeReadiness | None = Field(
        default=None, description="Its healthcheck; None when it has none, or when it is `run_once`."
    )
    limits: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Kubernetes resource limits (`cpu`, `memory`) from `cpus`, `mem_limit` and `deploy.resources.limits`."
        ),
    )
    requests: dict[str, str] = Field(
        default_factory=dict, description="Kubernetes resource requests from `deploy.resources.reservations`."
    )
    volume_mounts: list[ComposeVolumeMount] = Field(default_factory=list, description="The named volumes it mounts.")
    shm_size: str | None = Field(default=None, description="The size of `/dev/shm`, as a Kubernetes quantity.")


class ComposeMainSettings(BaseModel):
    """What the task's compose file and Dockerfile say about ``main``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    depends_on: dict[str, str] = Field(
        default_factory=dict, description="Each service `main` waits for, and the condition."
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Variables the compose file adds to `main`'s environment."
    )
    readiness: ComposeReadiness | None = Field(
        default=None, description="`main`'s healthcheck. The sandbox waits for it before the agent starts."
    )
    add_capabilities: list[Literal["SYS_PTRACE"]] = Field(
        default_factory=list,
        description=(
            "Linux capabilities from `cap_add`. Only `SYS_PTRACE` is allowed: broader ones could bypass egress."
        ),
    )
    volume_mounts: list[ComposeVolumeMount] = Field(
        default_factory=list, description="The named volumes `main` shares with services."
    )
    entrypoint: list[str] | None = Field(
        default=None, description="The last exec-form `ENTRYPOINT` of `main`'s Dockerfile, or the compose override."
    )
    ports: list[ComposePort] = Field(
        default_factory=list,
        description=(
            "The ports `main` `expose`s or publishes. The sandbox doesn't publish them; they're kept so a service "
            "or the sandbox's own containers can't listen on the same port in the shared pod network."
        ),
    )
    command: list[str] | None = Field(
        default=None,
        description="Compose `command` on `main`, which runs after its entrypoint in place of Harbor's keep-alive.",
    )
    working_dir: str | None = Field(
        default=None, description="Compose `working_dir` on `main`: where it starts, and where commands run by default."
    )
    platform: str | None = Field(
        default=None, description="Compose `platform` on `main`, as written: the platform its image is built for."
    )


class ComposeTask(BaseModel):
    """A task's compose file, merged with Harbor's base and interpolated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compose_path: str = Field(description="`<task>/environment/docker-compose.yaml`.")
    main_dockerfile: str | None = Field(
        description=(
            "`main`'s Dockerfile, relative to `environment/`: `Dockerfile` unless the task's `main.build` says "
            "otherwise. None when the task runs a prebuilt image."
        )
    )
    main_build: ComposeBuild | None = Field(
        default=None,
        description="What builds `main`: its context and Dockerfile. None when the task runs a prebuilt image.",
    )
    main_image: str | None = Field(
        default=None,
        description=(
            "The image `main` runs, for a prebuilt task: `task.toml`'s `docker_image` unless `main.image` replaces it."
        ),
    )
    services: dict[str, ComposeService] = Field(
        description="Every service other than `main`, by name, in start order (each after what it depends on)."
    )
    main: ComposeMainSettings = Field(description="What the compose file changes about `main`.")
    volumes: list[str] = Field(
        default_factory=list,
        description="The declared volumes something mounts, in file order. Each becomes an empty pod volume.",
    )
