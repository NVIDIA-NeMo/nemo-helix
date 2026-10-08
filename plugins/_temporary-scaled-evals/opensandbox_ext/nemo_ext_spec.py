# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The services spec a sandbox create request carries in ``extensions["nemo.nvidia.com/services"]``.

A Harbor task can ship a Docker Compose file that runs helper services (a database, a
cache, a mock API) next to the task's own container, ``main``. OpenSandbox runs a single
container per sandbox, so the client describes those services in this spec instead, and
the server extension adds them to the sandbox pod (see ``nemo_ext_pod``).

The spec arrives from any client that can reach the server, so it is not trusted. This
module validates everything the pod builder relies on: container and volume names, ports,
resource quantities, and the only extra capability the sandbox may get. Anything the pod
builder would otherwise have to defend against is rejected here.

Only depends on pydantic, so it can be loaded and tested without the server.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# The key in the create request's ``extensions`` map. The key must not start with
# ``opensandbox.extensions.``: the stock server copies such keys onto the pod as annotations.
SERVICES_KEY = "nemo.nvidia.com/services"

# Containers the stock BatchSandbox pod already has. ``sandbox`` runs the task image (Compose's
# ``main``), and ``egress`` enforces the sandbox's network policy.
MAIN_CONTAINER = "sandbox"
EGRESS_CONTAINER = "egress"

# Services may not take these names. Besides the stock containers above, the stock pod has an
# ``execd-installer`` init container, and services reach the sandbox container as ``main``
# through a host alias, so a service called ``main`` would shadow it.
RESERVED_NAMES = frozenset({MAIN_CONTAINER, EGRESS_CONTAINER, "execd-installer", "main"})

# Every container in the pod shares one network namespace, so a service listening on one of
# these ports would break the sandbox: execd (the in-sandbox command server) listens on 44772,
# and the egress filter's DNS proxy on 15353.
RESERVED_PORTS = frozenset({44772, 15353})

# A Kubernetes DNS label of at most 54 characters. The pod builder derives volume names by
# adding a 9-character prefix ("nemo-vol-" or "nemo-shm-"), and those must fit in 63.
Name = Annotated[str, Field(pattern=r"^[a-z0-9]([-a-z0-9]{0,52}[a-z0-9])?$")]
Port = Annotated[int, Field(ge=1, le=65535)]
AbsolutePath = Annotated[str, Field(pattern=r"^/", max_length=4096)]

# A Kubernetes resource quantity such as "500m", "2", or "1Gi".
Quantity = Annotated[str, Field(pattern=r"^[0-9]+(\.[0-9]+)?(m|k|Ki|M|Mi|G|Gi|T|Ti)?$")]
EnvName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")]
ResourceName = Literal["cpu", "memory", "ephemeral-storage"]

# The only capability the sandbox container may gain, for tasks whose Compose file gives ``main``
# ``cap_add: [SYS_PTRACE]`` to inspect running processes. Anything broader (e.g. NET_ADMIN) could
# let the task bypass the egress filter.
Capability = Literal["SYS_PTRACE"]

# Where a service runs in the pod; see ``Service.role``.
Role = Literal["sidecar", "run_once", "after_main"]


class ServicesSpecError(ValueError):
    """The services extension in a create request is invalid.

    Subclasses ``ValueError`` on purpose: the stock server turns a ``ValueError`` raised during
    sandbox creation into an HTTP 400, so the client sees the validation message.
    """


class _Strict(BaseModel):
    """Base for every spec model.

    Unknown fields are rejected, so a typo or a field this server version doesn't support
    (e.g. ``privileged``) fails loudly instead of being ignored. Types aren't coerced
    (``"5432"`` is not a port), and parsed instances are immutable.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Readiness(_Strict):
    """A command that exits 0 once the service is ready, like Compose's ``healthcheck.test``.

    It runs as an exec probe inside the service container; Kubernetes uses it to hold the
    containers that start after this one.
    """

    exec_: list[str] = Field(alias="exec", min_length=1)


class VolumeMount(_Strict):
    """Mounts one of the spec's shared ``volumes`` into a container.

    Compose tasks often share a named volume between ``main`` and a service, for example a
    directory the service writes results into that the verifier then reads from ``main``.
    """

    name: Name
    mount_path: AbsolutePath
    read_only: bool = False


class Resources(_Strict):
    """Kubernetes resource limits and requests for one service container."""

    limits: dict[ResourceName, Quantity] | None = None
    requests: dict[ResourceName, Quantity] | None = None


class Service(_Strict):
    """One Compose service to add to the sandbox pod as an extra container.

    ``run_once`` and ``after_main`` pick the service's ``role``, which decides where it goes
    in the pod and when the sandbox counts it as up.
    """

    name: Name
    image: str = Field(min_length=1, max_length=512)
    command: list[str] | None = None
    args: list[str] | None = None
    env: dict[EnvName, str] = Field(default_factory=dict)
    ports: list[Port] = Field(default_factory=list)
    readiness: Readiness | None = None
    run_once: bool = False
    after_main: bool = False
    resources: Resources | None = None
    volume_mounts: list[VolumeMount] = Field(default_factory=list)
    shm_size: Quantity | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Service:
        """Reject flag combinations that don't map to a single role, and names or ports the pod already uses."""
        # A service either finishes before main starts or runs next to it; it can't do both.
        if self.run_once and self.after_main:
            raise ValueError(f"service {self.name!r}: run_once and after_main are mutually exclusive")

        # A run_once service counts as done when it exits 0, so a readiness check would never apply.
        if self.run_once and self.readiness is not None:
            raise ValueError(f"service {self.name!r}: a run_once service is done when it exits; drop readiness")

        if self.name in RESERVED_NAMES:
            raise ValueError(f"service name {self.name!r} is reserved")

        reserved = RESERVED_PORTS.intersection(self.ports)
        if reserved:
            raise ValueError(f"service {self.name!r}: ports {sorted(reserved)} are used by the sandbox pod")

        return self

    @property
    def role(self) -> Role:
        """How the service runs relative to ``main``.

        * ``run_once``: runs to completion before ``main`` starts, like a Compose service other
          services wait on with ``condition: service_completed_successfully`` (e.g. a migration).
        * ``after_main``: starts next to ``main`` instead of before it, for services that depend
          on ``main`` being up.
        * ``sidecar`` (the default): starts before ``main`` and keeps running; ``main`` waits
          until its readiness check passes, like Compose's ``condition: service_healthy``.
        """
        if self.run_once:
            return "run_once"
        return "after_main" if self.after_main else "sidecar"


class SandboxOptions(_Strict):
    """Changes the task's Compose file makes to ``main`` itself: shared volume mounts and extra capabilities."""

    volume_mounts: list[VolumeMount] = Field(default_factory=list)
    add_capabilities: list[Capability] = Field(default_factory=list)


class ServicesSpec(_Strict):
    """The whole extension value: the services, the volumes they share, and the options for ``main``.

    ``version`` lets the format change later; a server only accepts the versions it knows.
    The service and volume limits keep a single request from building an unreasonably large pod.
    """

    version: Literal[1]
    services: list[Service] = Field(min_length=1, max_length=16)
    volumes: list[Name] = Field(default_factory=list, max_length=16)
    sandbox: SandboxOptions = Field(default_factory=SandboxOptions)

    @model_validator(mode="after")
    def _references(self) -> ServicesSpec:
        """Reject checks that span services: duplicate names or ports, and mounts of undeclared volumes."""
        # Service names become container names and host aliases, so each must be unique.
        names = [s.name for s in self.services]
        if len(set(names)) != len(names):
            raise ValueError("service names must be unique")

        if len(set(self.volumes)) != len(self.volumes):
            raise ValueError("volume names must be unique")

        # Under Compose each service has its own network namespace, so two services may listen
        # on the same port. In one pod they would collide.
        ports = [p for s in self.services for p in s.ports]
        if len(set(ports)) != len(ports):
            raise ValueError("two services claim the same port; all containers share one network namespace")

        # Every mount, by main or by a service, must name a volume declared in ``volumes``;
        # the pod builder only creates declared volumes.
        mounts = [("sandbox", m) for m in self.sandbox.volume_mounts]
        mounts += [(s.name, m) for s in self.services for m in s.volume_mounts]
        for owner, mount in mounts:
            if mount.name not in self.volumes:
                raise ValueError(f"{owner}: volume {mount.name!r} is not declared in volumes")

        return self

    def roles(self) -> dict[str, Role]:
        """Each service's role by name, in spec order, which is also the order the pod starts them in."""
        return {s.name: s.role for s in self.services}


def parse_spec(raw: str) -> ServicesSpec:
    """Parse and validate the extension's JSON string.

    Pydantic reports each problem separately; they are joined into one ``ServicesSpecError``
    that names every offending field (e.g. ``services.0.ports.0``), so the client's 400
    response lists everything wrong with the request at once.
    """
    try:
        return ServicesSpec.model_validate_json(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in exc.errors()
        )
        # ``from None``: the joined message already says everything; pydantic's own traceback is noise.
        raise ServicesSpecError(f"invalid extensions[{SERVICES_KEY!r}]: {problems}") from None
