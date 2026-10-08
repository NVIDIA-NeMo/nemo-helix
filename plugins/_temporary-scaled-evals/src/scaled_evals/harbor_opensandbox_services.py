# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose services for ``harbor_opensandbox`` trials, as an evaluation profile declares them.

A Compose task (one that ships ``docker-compose.yaml``) runs its services next to ``main``.
OpenSandbox can't run Compose, so the evaluation profile lists the services with prebuilt
images, and the server's NeMo services extension (``k8s/helm/examples/opensandbox``) adds them
to the sandbox pod. This is the only part of the Harbor ``environment`` block a profile may set:

.. code-block:: yaml

    environment:
      kwargs:
        compose_services:
          volumes: [shared]
          services:
            - name: db
              image: docker.io/library/postgres:16
              env: {POSTGRES_PASSWORD: postgres}
              ports: [5432]
              readiness: {exec: [pg_isready, -U, postgres]}
            - name: migrate              # Compose "service_completed_successfully"
              image: registry.example.com/task/migrate:1
              run_once: true
          main:
            entrypoint: [/app/entrypoint.sh]   # the task Dockerfile's ENTRYPOINT
            env: {DATABASE_URL: "postgresql://postgres:postgres@db:5432/app"}
            readiness: {exec: [curl, -sf, http://localhost:8000/health]}

Services reach each other and ``main`` by name on 127.0.0.1. Fields mirror Compose; the server
validates the same spec again. Imported by the dispatch backend and, inside the Harbor runner,
by ``NemoOpenSandboxEnvironment``, so it only depends on pydantic.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Extension key the OpenSandbox server's services extension reads from a create request.
SERVICES_EXTENSION_KEY = "nemo.nvidia.com/services"
SERVICES_SPEC_VERSION = 1
MAIN_SERVICE = "main"
# File names, in the task's environment/ directory, that mark a Compose task.
COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")

Name = Annotated[str, Field(pattern=r"^[a-z0-9]([-a-z0-9]{0,52}[a-z0-9])?$")]
Port = Annotated[int, Field(ge=1, le=65535)]
AbsolutePath = Annotated[str, Field(pattern=r"^/", max_length=4096)]
Quantity = Annotated[str, Field(pattern=r"^[0-9]+(\.[0-9]+)?(m|k|Ki|M|Mi|G|Gi|T|Ti)?$")]
EnvName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")]
# CRI-O resolves no short names, so a service image must name its registry.
QualifiedImage = Annotated[
    str, Field(pattern=r"^([a-z0-9-]+(\.[a-z0-9-]+)+(:[0-9]+)?|localhost(:[0-9]+)?)/\S+$", max_length=512)
]


class _Strict(BaseModel):
    """Base for every profile model: unknown fields (likely typos) and type coercion are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=True)


class Readiness(_Strict):
    """A command that exits 0 once the service is ready (Compose ``healthcheck.test``)."""

    exec_: list[str] = Field(alias="exec", min_length=1)


class VolumeMount(_Strict):
    """Mounts one of ``compose_services.volumes`` into a container (a Compose named volume)."""

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
    command: list[str] | None = None
    args: list[str] | None = None
    env: dict[EnvName, str] = Field(default_factory=dict)
    ports: list[Port] = Field(default_factory=list)
    readiness: Readiness | None = None
    # Compose ``condition: service_completed_successfully``: runs to completion before what follows.
    run_once: bool = False
    # A service that depends on ``main``: starts with it instead of before it.
    after_main: bool = False
    resources: Resources | None = None
    volume_mounts: list[VolumeMount] = Field(default_factory=list)
    shm_size: Quantity | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Service:
        """Reject ``main`` as a service name, and flag combinations the server can't place in the pod."""
        if self.name == MAIN_SERVICE:
            raise ValueError("'main' is the task container; describe it under compose_services.main")
        if self.run_once and self.after_main:
            raise ValueError(f"service {self.name!r}: run_once and after_main are mutually exclusive")
        if self.run_once and self.readiness is not None:
            raise ValueError(f"service {self.name!r}: a run_once service is done when it exits; drop readiness")
        return self


class MainReadiness(_Strict):
    """A command that exits 0 once ``main``'s own processes (started by its entrypoint) are up."""

    exec_: list[str] = Field(alias="exec", min_length=1)
    timeout_sec: int = Field(default=300, ge=1, le=3600)


class Main(_Strict):
    """The task's ``main`` service, which runs the task image as the sandbox container."""

    # The image's Dockerfile ENTRYPOINT. OpenSandbox replaces it with its own process, so it
    # is run explicitly with "sleep infinity" as its command.
    entrypoint: list[str] | None = Field(default=None, min_length=1)
    env: dict[EnvName, str] = Field(default_factory=dict)
    volume_mounts: list[VolumeMount] = Field(default_factory=list)
    add_capabilities: list[Literal["SYS_PTRACE"]] = Field(default_factory=list)
    readiness: MainReadiness | None = None


class ComposeServices(_Strict):
    """A profile's whole ``compose_services`` block: the services, the volumes they share, and ``main``."""

    volumes: list[Name] = Field(default_factory=list, max_length=16)
    services: list[Service] = Field(min_length=1, max_length=16)
    main: Main = Field(default_factory=Main)

    @model_validator(mode="after")
    def _references(self) -> ComposeServices:
        """Reject duplicate service names and ports, and mounts of volumes the block does not declare."""
        names = [s.name for s in self.services]
        if len(set(names)) != len(names):
            raise ValueError("service names must be unique")
        ports = [p for s in self.services for p in s.ports]
        if len(set(ports)) != len(ports):
            raise ValueError("two services claim the same port; all containers share one network")

        mounts = [(MAIN_SERVICE, m) for m in self.main.volume_mounts]
        mounts += [(s.name, m) for s in self.services for m in s.volume_mounts]
        for owner, mount in mounts:
            if mount.name not in self.volumes:
                raise ValueError(f"{owner}: volume {mount.name!r} is not declared in compose_services.volumes")
        return self

    def names(self) -> list[str]:
        """The service names, in start order."""
        return [s.name for s in self.services]

    def sandbox_entrypoint(self) -> list[str] | None:
        """The sandbox container's entrypoint: ``main``'s own entrypoint, kept alive; None to keep the default."""
        if self.main.entrypoint is None:
            return None
        return [*self.main.entrypoint, "sh", "-c", "sleep infinity"]

    def server_spec(self) -> str:
        """The JSON the server extension reads from ``extensions[SERVICES_EXTENSION_KEY]``."""
        # main's env, entrypoint and readiness are applied client-side; only these reach the pod builder.
        sandbox: dict[str, object] = {}
        if self.main.volume_mounts:
            sandbox["volume_mounts"] = [m.model_dump(mode="json") for m in self.main.volume_mounts]
        if self.main.add_capabilities:
            sandbox["add_capabilities"] = list(self.main.add_capabilities)

        spec: dict[str, object] = {
            "version": SERVICES_SPEC_VERSION,
            "services": [s.model_dump(mode="json", by_alias=True, exclude_defaults=True) for s in self.services],
        }
        if self.volumes:
            spec["volumes"] = list(self.volumes)
        if sandbox:
            spec["sandbox"] = sandbox
        return json.dumps(spec, separators=(",", ":"), sort_keys=True)


def parse_compose_services(raw: object) -> ComposeServices:
    """Validate a profile's ``environment.kwargs.compose_services`` (plain YAML/JSON data)."""
    if not isinstance(raw, dict):
        raise ValueError(f"compose_services must be a mapping, got {type(raw).__name__}")
    return ComposeServices.model_validate(raw)
