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
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The create request's extensions key. The stock server copies keys with this prefix onto the
# sandbox pod as the annotation below, which is how the extension reads the spec back.
EXTENSION_KEY = "opensandbox.extensions.nemo-compose-services"
POD_ANNOTATION = "opensandbox.io/extensions.nemo-compose-services"

MAIN_SERVICE = "main"
# File names, in the task's environment/ directory, that mark a Compose task.
COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")

# Names the sandbox pod already uses: its containers (sandbox, egress, execd-installer), and
# ``main``, which every container resolves to 127.0.0.1.
RESERVED_NAMES = frozenset({MAIN_SERVICE, "sandbox", "egress", "execd-installer"})
# Ports the sandbox pod already listens on: execd (44772), the egress DNS proxy (15353) and the
# egress policy API (18080). Every container shares the pod network, so a service can't use them.
RESERVED_PORTS = frozenset({44772, 15353, 18080})

# A Kubernetes DNS label short enough that "nemo-vol-<name>" and "nemo-shm-<name>" still fit in 63.
Name = Annotated[str, Field(pattern=r"^[a-z0-9]([-a-z0-9]{0,52}[a-z0-9])?$")]
Port = Annotated[int, Field(ge=1, le=65535)]
AbsolutePath = Annotated[str, Field(pattern=r"^/", max_length=4096)]
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


class Resources(_Strict):
    """Kubernetes resource limits and requests for one service (Compose ``deploy.resources``)."""

    limits: dict[Literal["cpu", "memory", "ephemeral-storage"], Quantity] | None = None
    requests: dict[Literal["cpu", "memory", "ephemeral-storage"], Quantity] | None = None


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

    @model_validator(mode="after")
    def _check(self) -> Service:
        """Reject names and ports the sandbox pod already uses, and readiness on a ``run_once`` service."""
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
    env: dict[EnvName, str] = Field(default_factory=dict)
    volume_mounts: list[VolumeMount] = Field(default_factory=list)
    # The only capability ``main`` may gain; anything broader (e.g. NET_ADMIN) could bypass egress.
    add_capabilities: list[Literal["SYS_PTRACE"]] = Field(default_factory=list)
    readiness: MainReadiness | None = None


class ComposeServices(_Strict):
    """A profile's whole ``compose_services`` block: the services, the volumes they share, and ``main``."""

    volumes: list[Name] = Field(default_factory=list, max_length=16)
    services: list[Service] = Field(min_length=1, max_length=16)
    main: Main = Field(default_factory=Main)

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
        return self

    def names(self) -> list[str]:
        """The service names, in start order."""
        return [s.name for s in self.services]

    def sandbox_entrypoint(self) -> list[str] | None:
        """The sandbox container's entrypoint: ``main``'s own entrypoint, kept alive; None to keep the default."""
        if self.main.entrypoint is None:
            return None
        return [*self.main.entrypoint, "sh", "-c", "sleep infinity"]

    def to_json(self) -> str:
        """The value sent in ``extensions[EXTENSION_KEY]``."""
        return self.model_dump_json(by_alias=True, exclude_defaults=True)


def parse_compose_services(raw: object) -> ComposeServices:
    """Validate a profile's ``environment.kwargs.compose_services`` (plain YAML/JSON data)."""
    if not isinstance(raw, dict):
        raise ValueError(f"compose_services must be a mapping, got {type(raw).__name__}")
    return ComposeServices.model_validate(raw)
