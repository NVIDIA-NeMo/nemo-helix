# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose services for ``harbor_opensandbox`` trials, as an evaluation profile declares them.

A Compose task (one that ships ``docker-compose.yaml``) runs helper services next to ``main``.
OpenSandbox can't run Compose, so the evaluation profile lists the services with prebuilt
images, and the OpenSandbox server's NeMo extension (``plugins/_temporary-scaled-evals/opensandbox_ext``)
adds them to the sandbox pod as extra containers:

.. code-block:: yaml

    environment:
      kwargs:
        compose_services:
          volumes: [shared]
          services:
            - name: db
              image: docker.io/library/postgres:16
              env: {POSTGRES_HOST_AUTH_METHOD: trust}
              ports: [5432]
              readiness: {exec: [pg_isready, -U, postgres]}
            - name: migrate
              image: registry.example.com/task/migrate:1
              role: run_once
          main:
            entrypoint: [/app/entrypoint.sh]
            env: {DATABASE_URL: "postgresql://postgres@db:5432/app"}
            readiness: {exec: [curl, -sf, http://localhost:8000/health]}

Every container in the pod shares one network, so services reach each other and ``main`` by name
on 127.0.0.1. The same model is used by the dispatch backend, by ``NemoOpenSandboxEnvironment``
inside the Harbor runner, and by the server extension, which re-validates what it receives. It
only depends on pydantic and must stay importable on Python 3.10 (the server image's version).

The extension also delivers agent bundles (``SandboxAgentBundle``): an init container from the bundle
image copies a prebuilt agent into ``/installed-agent`` before ``main`` starts, so agents run on
images that can't install them, such as images with a non-root ``USER``.
"""

from __future__ import annotations

import posixpath
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

# The create request's extensions key. The stock server copies keys with this prefix onto the
# sandbox pod as the annotation below, which is how the extension reads the spec back.
EXTENSION_KEY = "opensandbox.extensions.nemo-compose-services"
POD_ANNOTATION = "opensandbox.io/extensions.nemo-compose-services"

MAIN_SERVICE = "main"
# File names, in the task's environment/ directory, that mark a Compose task.
COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")

# What the server extension can do, as its health route reports it. Clients check for the feature
# they need, since an older extension would ignore a key it doesn't know.
COMPOSE_SERVICES_FEATURE = "compose-services"
AGENT_BUNDLE_FEATURE = "agent-bundle"
# A spec with no services, and ``main``'s ``command`` and ``working_dir``: what a compose file that only changes
# ``main`` needs. An extension from before them refuses such a spec.
COMPOSE_MAIN_FEATURE = "compose-main"
# Each container's ``security`` and ``tmpfs``: the restrictions a compose file sets with ``user``, ``read_only``,
# ``cap_drop``, ``security_opt`` and ``tmpfs``. An older extension would ignore them and run the task unrestricted.
COMPOSE_SECURITY_FEATURE = "compose-security"
EXTENSION_FEATURES = (COMPOSE_SERVICES_FEATURE, AGENT_BUNDLE_FEATURE, COMPOSE_MAIN_FEATURE, COMPOSE_SECURITY_FEATURE)

# The agent bundle's extensions key and pod annotation, copied the same way as the services'.
AGENT_BUNDLE_EXTENSION_KEY = "opensandbox.extensions.nemo-agent-bundle"
AGENT_BUNDLE_POD_ANNOTATION = "opensandbox.io/extensions.nemo-agent-bundle"
# The init container that installs the bundle, and where it puts the agent in ``main``.
AGENT_BUNDLE_CONTAINER = "agent-bundle"
INSTALLED_AGENT_DIR = "/installed-agent"

# Names the sandbox pod already uses: its containers (sandbox, egress, execd-installer, and the
# agent bundle's), and ``main``, which every container resolves to 127.0.0.1.
RESERVED_NAMES = frozenset({MAIN_SERVICE, "sandbox", "egress", "execd-installer", AGENT_BUNDLE_CONTAINER})
# The most services one sandbox pod runs besides ``main``.
MAX_SERVICES = 16
# Ports the sandbox pod already listens on: execd (44772), the egress DNS proxy (15353) and the
# egress policy API (18080). Every container shares the pod network, so a service can't use them.
RESERVED_PORTS = frozenset({44772, 15353, 18080})

# A Kubernetes DNS label short enough that "nemo-vol-<name>", "nemo-shm-<name>" and "nemo-t<n>-<name>" still fit in 63.
Name = Annotated[str, Field(pattern=r"^[a-z0-9]([-a-z0-9]{0,52}[a-z0-9])?$")]
# A Linux capability as Kubernetes names it, without ``CAP_``: ``NET_RAW``, or ``ALL``.
Capability = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$", max_length=64)]
# The most tmpfs mounts one container takes, so each one's volume name, ``nemo-t<n>-<name>``, has a single digit.
MAX_TMPFS_MOUNTS = 10
Port = Annotated[int, Field(ge=1, le=65535)]


def _clean_path(path: str) -> str:
    """``path`` spelled one way for each folder: ``/tmp/`` becomes ``/tmp``, and ``//installed-agent`` ``/installed-agent``.

    Checks on mount paths compare strings, so two spellings of one folder would pass as two folders: a second mount at
    ``/tmp/`` would hide the one at ``/tmp``, and ``//installed-agent`` would get past the guard on the agent's install.
    ``normpath`` alone keeps a leading ``//``, which POSIX leaves to the system but Linux reads as ``/``.
    """
    return "/" + posixpath.normpath(path).lstrip("/")


AbsolutePath = Annotated[str, Field(pattern=r"^/", max_length=4096), AfterValidator(_clean_path)]
Quantity = Annotated[str, Field(pattern=r"^[0-9]+(\.[0-9]+)?(m|k|Ki|M|Mi|G|Gi|T|Ti)?$")]
EnvName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")]
# CRI-O resolves no short names, so a service image must name its registry.
QualifiedImage = Annotated[
    str, Field(pattern=r"^([a-z0-9-]+(\.[a-z0-9-]+)+(:[0-9]+)?|localhost(:[0-9]+)?)/\S+$", max_length=512)
]

# How a service starts relative to ``main``:
# * ``sidecar``: starts before ``main`` and keeps running. With ``readiness``, ``main`` waits
#   until the check passes (Compose ``service_healthy``); without it, only until the process
#   starts (``service_started``).
# * ``run_once``: runs to completion before ``main`` starts (``service_completed_successfully``).
# * ``after_main``: starts next to ``main``, for a service that depends on it.
Role = Literal["sidecar", "run_once", "after_main"]


class _Strict(BaseModel):
    """Base for every model: unknown fields (likely typos) and type coercion are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=True)


class Readiness(_Strict):
    """A command that exits 0 once the service is ready (Compose ``healthcheck.test``)."""

    exec_: list[str] = Field(alias="exec", min_length=1)


class VolumeMount(_Strict):
    """Mounts one of ``volumes`` into a container (a Compose named volume)."""

    name: Name
    mount_path: AbsolutePath
    read_only: bool = False


class Tmpfs(_Strict):
    """A memory-backed scratch directory in one container (Compose ``tmpfs``), as an ``emptyDir`` with medium ``Memory``.

    What's written to it counts against the container's memory limit, as a tmpfs does under Docker. A container with
    no memory limit, and no ``size_limit``, can fill it until the node is short of memory and the kubelet evicts the pod.
    """

    mount_path: AbsolutePath
    # The tmpfs's size; None leaves it bounded only by the container's memory limit, if it has one.
    size_limit: Quantity | None = None


class Security(_Strict):
    """Restrictions on one container (Compose ``user``, ``read_only``, ``cap_drop`` and ``security_opt``).

    None of them grants a privilege the task's image couldn't already have: capabilities are only dropped (the one
    ``main`` may add is ``Main.add_capabilities``), and a uid or gid, root included, is one the image could run as
    anyway. One the sandbox's template already sets can't be replaced (see the server extension).
    """

    # Compose `user`, as numbers: Kubernetes can't look a user name up in the image.
    run_as_user: int | None = Field(default=None, ge=0)
    run_as_group: int | None = Field(default=None, ge=0)
    # Compose `read_only`: the image's filesystem can't be written; only mounts can.
    read_only_root_filesystem: bool = False
    # Compose `cap_drop`.
    drop_capabilities: list[Capability] = Field(default_factory=list)
    # Compose `security_opt: [no-new-privileges]`: setuid programs and file capabilities don't raise privileges.
    no_new_privileges: bool = False

    @model_validator(mode="after")
    def _check(self) -> Security:
        """Reject a group without a user: Kubernetes would run the image's own user in that group, which Compose can't."""
        if self.run_as_group is not None and self.run_as_user is None:
            raise ValueError("run_as_group needs run_as_user")
        return self


def _check_tmpfs(owner: str, tmpfs: list[Tmpfs]) -> None:
    """Reject more tmpfs mounts than ``MAX_TMPFS_MOUNTS``, which their volume names can't number."""
    if len(tmpfs) > MAX_TMPFS_MOUNTS:
        raise ValueError(f"{owner}: at most {MAX_TMPFS_MOUNTS} tmpfs mounts per container")


# Multipliers for the suffixes ``Quantity`` allows.
_QUANTITY_SUFFIXES = {
    "": Decimal(1),
    "m": Decimal("0.001"),
    "k": Decimal(10) ** 3,
    "M": Decimal(10) ** 6,
    "G": Decimal(10) ** 9,
    "T": Decimal(10) ** 12,
    "Ki": Decimal(2) ** 10,
    "Mi": Decimal(2) ** 20,
    "Gi": Decimal(2) ** 30,
    "Ti": Decimal(2) ** 40,
}


def quantity_value(quantity: str) -> Decimal:
    """The numeric value of a ``Quantity`` string, so ``500m`` and ``1Gi`` can be compared."""
    number = quantity.rstrip("mkKMGTi")
    return Decimal(number) * _QUANTITY_SUFFIXES[quantity[len(number) :]]


class Resources(_Strict):
    """Kubernetes resource limits and requests for one service (Compose ``deploy.resources``)."""

    limits: dict[Literal["cpu", "memory", "ephemeral-storage"], Quantity] | None = None
    requests: dict[Literal["cpu", "memory", "ephemeral-storage"], Quantity] | None = None

    @model_validator(mode="after")
    def _check(self) -> Resources:
        """Reject a request above its limit, which Kubernetes refuses when it creates the container."""
        for resource, request in (self.requests or {}).items():
            limit = (self.limits or {}).get(resource)
            if limit is not None and quantity_value(request) > quantity_value(limit):
                raise ValueError(f"{resource} request {request!r} exceeds its limit {limit!r}")
        return self


class Service(_Strict):
    """One Compose service other than ``main``."""

    name: Name
    image: QualifiedImage
    role: Role = "sidecar"
    command: list[str] | None = None
    args: list[str] | None = None
    env: dict[EnvName, str] = Field(default_factory=dict)
    ports: list[Port] = Field(default_factory=list)
    readiness: Readiness | None = None
    resources: Resources | None = None
    volume_mounts: list[VolumeMount] = Field(default_factory=list)
    shm_size: Quantity | None = None
    security: Security | None = None
    tmpfs: list[Tmpfs] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Service:
        """Reject names and ports the sandbox pod already uses, and readiness on a ``run_once`` service."""
        _check_tmpfs(self.name, self.tmpfs)
        if self.name in RESERVED_NAMES:
            raise ValueError(f"service name {self.name!r} is reserved")
        reserved = RESERVED_PORTS.intersection(self.ports)
        if reserved:
            raise ValueError(f"service {self.name!r}: ports {sorted(reserved)} are used by the sandbox pod")
        if self.role == "run_once" and self.readiness is not None:
            raise ValueError(f"service {self.name!r}: a run_once service is done when it exits; drop readiness")
        return self


class MainReadiness(_Strict):
    """A command that exits 0 once ``main``'s own processes, started by its entrypoint, are up."""

    exec_: list[str] = Field(alias="exec", min_length=1)
    timeout_sec: int = Field(default=300, ge=1, le=3600)


class Main(_Strict):
    """The task's ``main`` service, which runs the task image as the sandbox container."""

    # The image's Dockerfile ENTRYPOINT. OpenSandbox replaces it with its own process, so the
    # client runs it explicitly, followed by "sleep infinity" to keep the container alive.
    entrypoint: list[str] | None = Field(default=None, min_length=1)
    # Compose `command`: runs after the entrypoint in place of "sleep infinity", as Compose passes it to the image's
    # ENTRYPOINT. Under Harbor it must keep running, or the container ends.
    command: list[str] | None = Field(default=None, min_length=1)
    # Compose `working_dir`: where main's entrypoint and command start, and where commands run by default.
    working_dir: AbsolutePath | None = None
    env: dict[EnvName, str] = Field(default_factory=dict)
    volume_mounts: list[VolumeMount] = Field(default_factory=list)
    # The only capability ``main`` may gain; anything broader (e.g. NET_ADMIN) could bypass egress.
    add_capabilities: list[Literal["SYS_PTRACE"]] = Field(default_factory=list)
    readiness: MainReadiness | None = None
    security: Security | None = None
    tmpfs: list[Tmpfs] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Main:
        """Reject more tmpfs mounts than the sandbox container takes, and a read-only ``main`` commands can't run in."""
        _check_tmpfs(MAIN_SERVICE, self.tmpfs)
        if self.security is not None and self.security.read_only_root_filesystem:
            # execd, which runs every command in main, first writes the command's output to files in its temp folder.
            # On a read-only root with nothing writable there, every command fails, even `true`.
            temp_dir = self.env.get("TMPDIR") or "/tmp"
            if not self._is_writable(temp_dir):
                raise ValueError(
                    f"main is read-only, so it needs a writable tmpfs or volume at {temp_dir}, "
                    "where commands write their output"
                )
        return self

    def _is_writable(self, path: str) -> bool:
        """Whether ``path`` can be written in a read-only ``main``: the closest mount over it is writable.

        A read-only mount inside a writable one wins over it, as it does in the container.
        """
        mounts = [(tmpfs.mount_path, True) for tmpfs in self.tmpfs]
        mounts += [(mount.mount_path, not mount.read_only) for mount in self.volume_mounts]
        covering_mounts = [
            (PurePosixPath(mount_path), writable)
            for mount_path, writable in mounts
            if PurePosixPath(path).is_relative_to(mount_path)
        ]
        if not covering_mounts:
            return False
        _closest_path, writable = max(covering_mounts, key=lambda mount: len(mount[0].parts))
        return writable


class ComposeServices(_Strict):
    """A profile's whole ``compose_services`` block: the services, the volumes they share, and ``main``."""

    volumes: list[Name] = Field(default_factory=list, max_length=16)
    # Empty for a compose file that only changes `main`.
    services: list[Service] = Field(default_factory=list, max_length=MAX_SERVICES)
    main: Main = Field(default_factory=Main)
    # How long ``NemoOpenSandboxEnvironment.start()`` waits for the services when the ``main`` container has no
    # readiness check to carry its own ``timeout_sec``.
    services_timeout_sec: int | None = Field(default=None, ge=1, le=3600)

    @model_validator(mode="after")
    def _check(self) -> ComposeServices:
        """Reject what only shows up across services: duplicate names, volumes or ports, and undeclared volumes."""
        names = self.names()
        if len(set(names)) != len(names):
            raise ValueError("service names must be unique")
        if len(set(self.volumes)) != len(self.volumes):
            raise ValueError("volume names must be unique")
        ports = [p for s in self.services for p in s.ports]
        if len(set(ports)) != len(ports):
            raise ValueError("two services claim the same port; all containers share one network")

        # Every mount, main's included, must name a declared volume.
        mounts = [(MAIN_SERVICE, m) for m in self.main.volume_mounts]
        mounts += [(s.name, m) for s in self.services for m in s.volume_mounts]
        for owner, mount in mounts:
            if mount.name not in self.volumes:
                raise ValueError(f"{owner}: volume {mount.name!r} is not declared in volumes")

        # Kubernetes requires a container's mount paths to be unique; separate containers may share one. A service
        # with `shm_size` already has a mount at /dev/shm, which the server adds for it.
        seen = {(s.name, "/dev/shm") for s in self.services if s.shm_size is not None}
        paths = [(owner, mount.mount_path) for owner, mount in mounts]
        paths += [(MAIN_SERVICE, t.mount_path) for t in self.main.tmpfs]
        paths += [(s.name, t.mount_path) for s in self.services for t in s.tmpfs]
        for owner, path in paths:
            if (owner, path) in seen:
                raise ValueError(f"{owner}: mount_path {path!r} is used by more than one mount")
            seen.add((owner, path))
        return self

    def names(self) -> list[str]:
        """The service names, in start order."""
        return [s.name for s in self.services]

    def sandbox_entrypoint(self, image_entrypoint: list[str] | None = None) -> list[str] | None:
        """The sandbox container's entrypoint: ``main``'s from the spec, else ``image_entrypoint``, then its command.

        ``image_entrypoint`` is the image's own ``ENTRYPOINT``, for a spec that doesn't carry it. See
        :func:`keep_alive_entrypoint`.
        """
        return keep_alive_entrypoint(self.main.entrypoint or image_entrypoint, self.main.command)

    def uses_main_settings(self) -> bool:
        """Whether the spec needs the extension's ``COMPOSE_MAIN_FEATURE``: no services, or ``main``'s command or working directory."""
        return not self.services or self.main.command is not None or self.main.working_dir is not None

    def uses_security_settings(self) -> bool:
        """Whether the spec needs the extension's ``COMPOSE_SECURITY_FEATURE``: any container's ``security`` or ``tmpfs``."""
        containers: list[Main | Service] = [self.main, *self.services]
        return any(container.security is not None or container.tmpfs for container in containers)

    def to_json(self) -> str:
        """The value sent in ``extensions[EXTENSION_KEY]``."""
        return self.model_dump_json(by_alias=True, exclude_defaults=True)


class SandboxAgentBundle(_Strict):
    """An agent bundle for one sandbox: the image, its installer, and the agent's executable.

    The extension runs ``installer /installed-agent`` from ``image`` in an init container, so the
    agent is in place before ``main`` starts; ``main`` mounts it read-only.
    """

    # Pinned by digest, or the signed tag hosted admission requires, which dispatch checks still
    # resolves to the registered digest at launch. Like services, it must name its registry.
    image: str = Field(
        pattern=(
            r"^([a-z0-9-]+(\.[a-z0-9-]+)+(:[0-9]+)?|localhost(:[0-9]+)?)/[^\s@]+"
            r"(@sha256:[0-9a-f]{64}|:[^\s@/:]+)$"
        ),
        max_length=512,
    )
    installer: AbsolutePath
    # Relative to /installed-agent, e.g. ``bin/claude``.
    entrypoint: str = Field(pattern=r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$", max_length=300)

    @model_validator(mode="after")
    def _check(self) -> SandboxAgentBundle:
        """Keep the executable inside ``/installed-agent``."""
        if ".." in self.entrypoint.split("/"):
            raise ValueError("entrypoint must not contain '..'")
        return self

    @property
    def executable(self) -> str:
        """The agent's executable in ``main``."""
        return f"{INSTALLED_AGENT_DIR}/{self.entrypoint}"

    def to_json(self) -> str:
        """The value sent in ``extensions[AGENT_BUNDLE_EXTENSION_KEY]``."""
        return self.model_dump_json(exclude_defaults=True)


def keep_alive_entrypoint(entrypoint: list[str] | None, command: list[str] | None = None) -> list[str] | None:
    """The sandbox container's entrypoint for an image that starts with ``entrypoint``; None to keep the default.

    OpenSandbox replaces the container's entrypoint with its own process, which would skip the
    image's ``ENTRYPOINT``, so it is restored here, followed by ``command`` as its arguments, as
    Docker passes a command to the image's ``ENTRYPOINT``. Without a command, Harbor's own
    ``sh -c "sleep infinity"`` follows it. An entrypoint or command that exits on its own therefore
    ends the container, as it would under Harbor, rather than being masked.

    ``["/bin/bash", "-l", "-c"]`` with command ``["/app/run.sh"]`` gives ``["/bin/bash", "-l", "-c", "/app/run.sh"]``.
    """
    if command:
        return [*(entrypoint or []), *command]
    if not entrypoint:
        return None
    return [*entrypoint, "sh", "-c", "sleep infinity"]


def parse_compose_services(raw: object) -> ComposeServices:
    """Validate a profile's ``environment.kwargs.compose_services`` (plain YAML/JSON data)."""
    if not isinstance(raw, dict):
        raise ValueError(f"compose_services must be a mapping, got {type(raw).__name__}")
    return ComposeServices.model_validate(raw)
