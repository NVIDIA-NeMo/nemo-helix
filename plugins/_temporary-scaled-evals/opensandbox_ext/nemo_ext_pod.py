# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adds a ``ServicesSpec``'s services to a BatchSandbox pod, and reads their state back.

The stock OpenSandbox server builds a BatchSandbox manifest whose pod has the task container
(``sandbox``), an ``egress`` container that enforces the network policy, and an
``execd-installer`` init container. This module adds the Compose services to that pod so they
start in the order, and with the readiness gating, that Compose would have used:

* Long-running services become native sidecars: init containers with ``restartPolicy:
  Always``. Kubernetes starts init containers one at a time, in list order, and holds the
  next one (and the sandbox container) until a sidecar's startup probe passes. That gives
  Compose's ``depends_on: condition: service_healthy`` without any extra controller.
* ``run_once`` services become plain init containers, which must exit 0 before the next one
  starts (Compose's ``condition: service_completed_successfully``).
* ``after_main`` services become regular containers that start next to the sandbox container.
* Every container shares the pod's network namespace, so services reach each other and
  ``main`` on 127.0.0.1. Host aliases map each Compose service name, and ``main``, to
  127.0.0.1, so the task's configuration (``postgres://db:5432``) works unchanged.
* The egress filter is moved ahead of every service in the init order, so its network
  rules apply to every container from the first packet.

The second half reads a running pod back to tell the server which services it is still
waiting for, and which one failed and why.

Pure functions over manifest dicts and ``V1Pod`` objects; no Kubernetes API calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from nemo_ext_spec import (
    EGRESS_CONTAINER,
    MAIN_CONTAINER,
    Readiness,
    Role,
    Service,
    ServicesSpec,
    ServicesSpecError,
    VolumeMount,
)

# Under the server's logger tree, which is the only one its logging config enables.
logger = logging.getLogger("opensandbox_server.nemo_ext")

# Pod annotation that records each service's role. The exec route reads it back from the pod
# rather than from server memory, so it works for any sandbox, including after a server restart.
ROLES_ANNOTATION = "nemo.nvidia.com/service-roles"

# The egress container serves its policy API on 18080 by default, and reads an alternative listen
# address from this environment variable. It shares the pod network with the services.
EGRESS_DEFAULT_PORT = 18080
EGRESS_ALT_PORTS = range(18090, 18190)
EGRESS_ADDR_ENV = "OPENSANDBOX_EGRESS_HTTP_ADDR"

# Probes run every second, so these failure thresholds are also roughly seconds. Once its
# container runs, a service gets about 10 minutes to become ready, the egress filter 2 minutes.
# Once ready, three failed checks in a row mark a service unready again.
STARTUP_PROBE_FAILURES = 600
EGRESS_STARTUP_PROBE_FAILURES = 120
READINESS_PROBE_FAILURES = 3

# Kubernetes restarts a crashing sidecar forever. After this many restarts it counts as failed,
# so the create call fails fast instead of waiting for its timeout.
CRASH_RESTARTS = 3

# Waiting reasons that never clear on their own: the image can't be pulled or the container
# can't be configured. Without this the sandbox would wait until the create call times out.
FATAL_WAITING_REASONS = frozenset(
    {"ImagePullBackOff", "InvalidImageName", "ImageInspectError", "CreateContainerConfigError", "ErrImageNeverPull"}
)


# ---- pod spec ----------------------------------------------------------------------------


def _probe(readiness: Readiness, failure_threshold: int) -> dict[str, Any]:
    """An exec probe that runs the service's readiness command every second.

    The same command backs both the startup probe (which gates the containers after this one)
    and the readiness probe (which reports whether the service is still healthy); only the
    failure threshold differs.
    """
    return {
        "exec": {"command": list(readiness.exec_)},
        "periodSeconds": 1,
        "timeoutSeconds": 5,
        "failureThreshold": failure_threshold,
    }


def _volume_mounts(mounts: list[VolumeMount]) -> list[dict[str, Any]]:
    """Kubernetes ``volumeMounts`` for spec mounts.

    The pod volumes themselves are added once by ``_add_volumes``. The ``nemo-vol-`` prefix
    keeps them apart from the volumes the stock pod already has.
    """
    return [{"name": f"nemo-vol-{m.name}", "mountPath": m.mount_path, "readOnly": m.read_only} for m in mounts]


def build_container(svc: Service, image_pull_policy: str) -> dict[str, Any]:
    """The Kubernetes container manifest for one service, with the probes and restart policy its role needs.

    * A ``run_once`` service gets no probes: it is done when it exits 0.
    * A ``sidecar`` gets ``restartPolicy: Always``, which makes it a native sidecar, and a
      startup probe, which is what holds the containers after it until it is ready.
    * An ``after_main`` service only gets a readiness probe. Nothing waits on it, but the server
      still waits for it to be ready before reporting the sandbox up.
    """
    container: dict[str, Any] = {
        "name": svc.name,
        "image": svc.image,
        "imagePullPolicy": image_pull_policy,
        # Services share the pod network with the egress filter. Dropping these capabilities
        # stops a service from changing the firewall rules or sending raw packets around them.
        "securityContext": {"privileged": False, "capabilities": {"drop": ["NET_ADMIN", "NET_RAW"]}},
    }

    # Optional fields are only set when the spec has them, so the image's own defaults apply otherwise.
    if svc.command is not None:
        container["command"] = list(svc.command)
    if svc.args is not None:
        container["args"] = list(svc.args)
    if svc.resources is not None:
        container["resources"] = svc.resources.model_dump(exclude_none=True)
    if svc.env:
        container["env"] = [{"name": k, "value": v} for k, v in svc.env.items()]
    if svc.ports:
        container["ports"] = [{"containerPort": p} for p in svc.ports]

    # Shared volumes, plus a memory-backed /dev/shm when the service sets shm_size
    # (Compose's ``shm_size``; databases and browsers often need more than the 64 MiB default).
    mounts = _volume_mounts(svc.volume_mounts)
    if svc.shm_size is not None:
        mounts.append({"name": f"nemo-shm-{svc.name}", "mountPath": "/dev/shm"})
    if mounts:
        container["volumeMounts"] = mounts

    if svc.role == "run_once":
        return container

    if svc.readiness is not None:
        container["readinessProbe"] = _probe(svc.readiness, READINESS_PROBE_FAILURES)

    # Kubernetes only holds the next init container (and main) for a sidecar's startup probe,
    # not its readiness probe, so the readiness command is also the startup probe.
    if svc.role == "sidecar":
        container["restartPolicy"] = "Always"
        if svc.readiness is not None:
            container["startupProbe"] = _probe(svc.readiness, STARTUP_PROBE_FAILURES)

    return container


def _move_egress_port(egress: dict[str, Any], spec: ServicesSpec) -> None:
    """Move the egress policy API off port 18080 when a service listens on that port.

    Without this, the service and the egress container would both bind 18080 in the shared
    network namespace, and one of them would fail to start. The new port is the first free one
    in ``EGRESS_ALT_PORTS``.

    In the pod, OpenSandbox only refers to the port in this container's port list and readiness
    probe, so those are all that change. Clients that read the policy back after create still
    use 18080, so a sandbox whose service claims it fails that readback (known limitation).
    """
    claimed = {p for s in spec.services for p in s.ports}
    if EGRESS_DEFAULT_PORT not in claimed:
        return

    port = next(p for p in EGRESS_ALT_PORTS if p not in claimed)

    # Tell the egress process where to listen...
    egress.setdefault("env", []).append({"name": EGRESS_ADDR_ENV, "value": f":{port}"})

    # ...and point the container's declared port and readiness probe at it.
    for p in egress.get("ports") or []:
        if p.get("containerPort") == EGRESS_DEFAULT_PORT:
            p["containerPort"] = port
    http_get = (egress.get("readinessProbe") or {}).get("httpGet") or {}
    if http_get.get("port") == EGRESS_DEFAULT_PORT:
        http_get["port"] = port

    logger.info("nemo services: egress moved to port %s because a service claims %s", port, EGRESS_DEFAULT_PORT)


def _check_pod(containers: list[dict[str, Any]], init: list[dict[str, Any]], spec: ServicesSpec) -> None:
    """Fail if the stock pod isn't shaped the way this module expects.

    The rest of this module assumes the first container is the task container, and that no
    service has the same name as a container the stock pod already has. A different shape
    means the server version changed underneath the extension, so stop rather than build a
    broken pod.
    """
    if not containers or containers[0].get("name") != MAIN_CONTAINER:
        raise ServicesSpecError(f"unexpected BatchSandbox pod: first container is not {MAIN_CONTAINER!r}")

    # RESERVED_NAMES already covers the known stock containers; this also catches any new ones.
    taken = {c.get("name") for c in [*containers, *init]}
    clashes = taken.intersection(s.name for s in spec.services)
    if clashes:
        raise ServicesSpecError(f"service names clash with sandbox containers: {sorted(clashes)}")


def _move_egress_first(containers: list[dict[str, Any]], init: list[dict[str, Any]], spec: ServicesSpec) -> None:
    """Turn the egress filter into a native sidecar that starts before any service.

    The stock pod runs egress as a regular container, which starts at the same time as the
    sandbox. Services start earlier than that, as init containers, so with egress left in place
    they would run unfiltered until it came up. As the first sidecar, its startup probe holds
    every service until its rules are in place.

    A sandbox created without a network policy has no egress container; then there is nothing to move.
    """
    egress = next((c for c in containers if c.get("name") == EGRESS_CONTAINER), None)
    if egress is None:
        return

    _move_egress_port(egress, spec)

    containers.remove(egress)
    egress["restartPolicy"] = "Always"

    # Reuse its readiness probe as the startup probe, which is what holds the next init container.
    if egress.get("readinessProbe"):
        egress["startupProbe"] = {**egress["readinessProbe"], "failureThreshold": EGRESS_STARTUP_PROBE_FAILURES}

    init.append(egress)


def _add_volumes(pod_spec: dict[str, Any], spec: ServicesSpec) -> None:
    """Add the pod volumes the spec needs.

    * An ``emptyDir`` per shared volume. It lives as long as the pod, which is the life of the
      sandbox, like a Compose named volume for the length of one trial.
    * A memory-backed ``emptyDir`` per service that sets ``shm_size``, mounted on /dev/shm
      and sized to the requested limit.
    """
    volumes = pod_spec.setdefault("volumes", [])

    volumes.extend({"name": f"nemo-vol-{v}", "emptyDir": {}} for v in spec.volumes)
    volumes.extend(
        {"name": f"nemo-shm-{s.name}", "emptyDir": {"medium": "Memory", "sizeLimit": s.shm_size}}
        for s in spec.services
        if s.shm_size is not None
    )

    # Don't leave an empty list behind on a pod that had no volumes.
    if not volumes:
        del pod_spec["volumes"]


def _configure_main(main: dict[str, Any], spec: ServicesSpec) -> None:
    """Apply the spec's sandbox options to the task container: shared volume mounts and extra capabilities."""
    if spec.sandbox.volume_mounts:
        main.setdefault("volumeMounts", []).extend(_volume_mounts(spec.sandbox.volume_mounts))

    # Merge with any capabilities the stock container already adds, without duplicates.
    if spec.sandbox.add_capabilities:
        caps = main.setdefault("securityContext", {}).setdefault("capabilities", {})
        caps["add"] = sorted(set(caps.get("add") or []) | set(spec.sandbox.add_capabilities))


def inject_services(body: dict[str, Any], spec: ServicesSpec, image_pull_policy: str) -> None:
    """Add the spec's services to the BatchSandbox manifest the stock server built, in place.

    ``image_pull_policy`` is the server's setting for the sandbox container; services use the
    same one. Raises ``ServicesSpecError`` if the manifest isn't shaped as expected.
    """
    template = body["spec"]["template"]
    pod_spec = template["spec"]
    containers: list[dict[str, Any]] = pod_spec["containers"]
    init: list[dict[str, Any]] = pod_spec.setdefault("initContainers", [])

    _check_pod(containers, init, spec)

    # Init containers start in list order, after the stock execd installer: egress first, then
    # the services in spec order. after_main services join the regular containers instead.
    _move_egress_first(containers, init, spec)
    for svc in spec.services:
        target = containers if svc.role == "after_main" else init
        target.append(build_container(svc, image_pull_policy))

    _add_volumes(pod_spec, spec)
    _configure_main(containers[0], spec)

    # Resolve every Compose service name, and main, to the shared loopback address.
    names = [s.name for s in spec.services]
    pod_spec.setdefault("hostAliases", []).append({"ip": "127.0.0.1", "hostnames": [*names, "main"]})

    # Record the roles on the pod; see ROLES_ANNOTATION.
    annotations = template.setdefault("metadata", {}).setdefault("annotations", {})
    annotations[ROLES_ANNOTATION] = json.dumps(spec.roles(), separators=(",", ":"))


# ---- pod status --------------------------------------------------------------------------


def pod_roles(pod: Any) -> dict[str, Role]:
    """The service roles ``inject_services`` recorded on the pod, by container name.

    Empty for a pod created without services, or with an unreadable annotation, so callers
    treat that pod as having no services at all.
    """
    raw = (pod.metadata.annotations or {}).get(ROLES_ANNOTATION)
    if not raw:
        return {}

    try:
        roles = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return roles if isinstance(roles, dict) else {}


def _statuses(pod: Any) -> dict[str, Any]:
    """Every container status of the pod, init containers included, by container name."""
    status = pod.status
    return {s.name: s for s in [*(status.init_container_statuses or []), *(status.container_statuses or [])]}


@dataclass(frozen=True)
class ServiceFailure:
    """A container that won't come up, and why. Becomes the message of the 422 the create call returns."""

    container: str
    detail: str


def _container_failure(name: str, role: Role, st: Any, pod_failed: bool) -> tuple[Any, ServiceFailure] | None:
    """Decide whether one service container has failed for good.

    Returns ``(finished_at, failure)`` if it has, or None while it may still come up.
    ``finished_at`` is when the container exited; it is None for a container that never
    started (e.g. its image can't be pulled). ``service_failure`` uses it to find the first
    failure in the pod.
    """
    # A container stuck waiting for a reason that never clears has failed without ever running.
    waiting = st.state.waiting if st.state else None
    if waiting is not None and waiting.reason in FATAL_WAITING_REASONS:
        return None, ServiceFailure(name, f"{waiting.reason}: {waiting.message or ''}".strip())

    # Otherwise only an exited container can have failed. A sidecar that Kubernetes is
    # restarting is "waiting" again, so its last exit is only in last_state.
    term = (st.state.terminated if st.state else None) or (st.last_state.terminated if st.last_state else None)
    if term is None:
        return None

    if role == "run_once":
        # A job that exited 0 succeeded; anything else fails the pod.
        failed = term.exit_code != 0
    elif role == "after_main":
        # Regular containers aren't restarted in a BatchSandbox pod, so any exit is final.
        failed = True
    else:
        # Sidecars are restarted forever. Give up after CRASH_RESTARTS, or as soon as the pod
        # itself has failed and nothing will restart it.
        failed = (st.restart_count or 0) >= CRASH_RESTARTS or (pod_failed and term.exit_code != 0)

    if not failed:
        return None
    return term.finished_at, ServiceFailure(name, f"exit code {term.exit_code}, {term.reason}")


def service_failure(pod: Any, roles: dict[str, Role]) -> ServiceFailure | None:
    """Return why the sandbox's services can't come up, or None while they still may.

    BatchSandbox pods use ``restartPolicy: Never``, so a ``run_once`` service that exits
    non-zero fails the pod, and an ``after_main`` service that exits is gone for good.
    Kubernetes restarts a native sidecar forever regardless, so a sidecar counts as failed
    after ``CRASH_RESTARTS`` restarts, or when it exited non-zero in a pod that already failed.

    If the pod failed but no service is to blame, the failure is reported against ``main``.
    """
    statuses = _statuses(pod)
    pod_failed = pod.status.phase == "Failed"

    # Containers that haven't been created yet have no status, and can't have failed yet.
    failures = [
        failure
        for name, role in roles.items()
        if name in statuses and (failure := _container_failure(name, role, statuses[name], pod_failed)) is not None
    ]

    if failures:
        # One failing service fails the pod, and Kubernetes then stops the others, which then
        # look failed too. Report the earliest exit, the likely cause; containers that never
        # started sort after every exit.
        failures.sort(key=lambda f: (f[0] is None, f[0] or 0))
        return failures[0][1]

    if pod_failed:
        return ServiceFailure(MAIN_CONTAINER, f"pod failed: {pod.status.reason or pod.status.message or 'unknown'}")

    return None


def services_waiting(pod: Any, roles: dict[str, Role]) -> list[str]:
    """The containers the sandbox is still waiting for, in start order, with ``main`` last.

    * A ``run_once`` service is done once it exited 0.
    * Any other service is done once Kubernetes reports it ready, which means its readiness
      check passed, or that it is running when it has none.
    * ``main`` is done once the task container runs.

    The sandbox is up when this returns an empty list.
    """
    statuses = _statuses(pod)
    waiting = []

    for name, role in roles.items():
        st = statuses.get(name)
        if role == "run_once":
            done = st is not None and st.state is not None and st.state.terminated is not None
            if not (done and st.state.terminated.exit_code == 0):
                waiting.append(name)
        elif st is None or not st.ready:
            waiting.append(name)

    main = statuses.get(MAIN_CONTAINER)
    if main is None or main.state is None or main.state.running is None:
        waiting.append("main")

    return waiting
