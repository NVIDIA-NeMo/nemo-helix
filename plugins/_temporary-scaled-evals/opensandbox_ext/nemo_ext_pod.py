# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adds a task's Compose services to the BatchSandbox pod the stock server built, and reads their state back.

The stock pod has the task container (``sandbox``), an ``egress`` container that enforces the
network policy, and an ``execd-installer`` init container. Kubernetes does all the ordering:

* ``sidecar`` services become native sidecars (init containers with ``restartPolicy: Always``).
  Init containers start one at a time, and a sidecar's startup probe (its ``readiness`` command)
  holds the next one, and ``sandbox``, until it passes.
* ``run_once`` services become plain init containers, which must exit 0 before the next starts.
* ``after_main`` services become regular containers that start next to ``sandbox``.
* ``egress`` moves to the front of the init containers, so its rules cover every service.
* Host aliases resolve every service name, and ``main``, to 127.0.0.1.

Pure functions over manifest dicts and kubernetes ``V1Pod`` objects; no API calls.
"""

from __future__ import annotations

from typing import Any

from scaled_evals.harbor_opensandbox_services import ComposeServices, Readiness, Service, VolumeMount

SANDBOX_CONTAINER = "sandbox"
EGRESS_CONTAINER = "egress"

# Probes run every second, so failure thresholds are roughly seconds: a service gets ~10 minutes
# to become ready once it runs, egress 2 minutes.
STARTUP_PROBE_FAILURES = 600
EGRESS_STARTUP_PROBE_FAILURES = 120
READINESS_PROBE_FAILURES = 3

# Kubernetes restarts a crashing sidecar forever; after this many restarts it counts as failed.
CRASH_RESTARTS = 3
# Waiting reasons that never clear on their own.
FATAL_WAITING_REASONS = frozenset(
    {"ErrImagePull", "ImagePullBackOff", "InvalidImageName", "ErrImageNeverPull", "CreateContainerConfigError"}
)


def _probe(readiness: Readiness, failure_threshold: int) -> dict[str, Any]:
    """An exec probe that runs ``readiness`` every second and fails after ``failure_threshold`` misses in a row."""
    return {
        "exec": {"command": list(readiness.exec_)},
        "periodSeconds": 1,
        "timeoutSeconds": 5,
        "failureThreshold": failure_threshold,
    }


def _mounts(mounts: list[VolumeMount]) -> list[dict[str, Any]]:
    """Container ``volumeMounts`` for spec mounts; each spec volume is the pod volume ``nemo-vol-<name>``."""
    return [{"name": f"nemo-vol-{m.name}", "mountPath": m.mount_path, "readOnly": m.read_only} for m in mounts]


def service_container(svc: Service, image_pull_policy: str) -> dict[str, Any]:
    """The container manifest for one service.

    Its role decides where it goes and which probes it gets: a ``sidecar`` restarts with the pod
    and, with ``readiness``, gets a startup probe that holds the containers after it; ``run_once``
    gets no probes, since it's done when it exits 0; every role but ``run_once`` gets a readiness
    probe, so the pod's Ready condition and the services route both track it.
    """
    container: dict[str, Any] = {
        "name": svc.name,
        "image": svc.image,
        "imagePullPolicy": image_pull_policy,
        # Services share the pod network with egress; without these they could rewrite its rules.
        "securityContext": {"privileged": False, "capabilities": {"drop": ["NET_ADMIN", "NET_RAW"]}},
    }
    if svc.command is not None:
        container["command"] = list(svc.command)
    if svc.args is not None:
        container["args"] = list(svc.args)
    if svc.env:
        container["env"] = [{"name": k, "value": v} for k, v in svc.env.items()]
    if svc.ports:
        container["ports"] = [{"containerPort": p} for p in svc.ports]
    if svc.resources is not None:
        container["resources"] = svc.resources.model_dump(exclude_none=True)

    mounts = _mounts(svc.volume_mounts)
    if svc.shm_size is not None:
        mounts.append({"name": f"nemo-shm-{svc.name}", "mountPath": "/dev/shm"})
    if mounts:
        container["volumeMounts"] = mounts

    if svc.role == "sidecar":
        container["restartPolicy"] = "Always"
        if svc.readiness is not None:
            # Kubernetes holds the next init container on a sidecar's startup probe, not its readiness probe.
            container["startupProbe"] = _probe(svc.readiness, STARTUP_PROBE_FAILURES)
    if svc.role != "run_once" and svc.readiness is not None:
        container["readinessProbe"] = _probe(svc.readiness, READINESS_PROBE_FAILURES)
    return container


def add_services(pod_spec: dict[str, Any], services: ComposeServices, image_pull_policy: str) -> None:
    """Add ``services`` to the BatchSandbox pod spec in place; ``ValueError`` if the pod isn't shaped as expected.

    Init containers start in list order, after the stock ones (``execd-installer``), so the final
    order is: stock init containers, ``egress``, then the services in spec order. ``after_main``
    services join ``sandbox`` as regular containers.
    """
    # The stock pod must look as expected, or the ordering below means nothing.
    containers: list[dict[str, Any]] = pod_spec["containers"]
    init: list[dict[str, Any]] = pod_spec.setdefault("initContainers", [])
    if not containers or containers[0].get("name") != SANDBOX_CONTAINER:
        raise ValueError(f"unexpected BatchSandbox pod: the first container is not {SANDBOX_CONTAINER!r}")
    clashes = {c.get("name") for c in [*containers, *init]}.intersection(services.names())
    if clashes:
        raise ValueError(f"service names clash with sandbox pod containers: {sorted(clashes)}")

    # egress becomes the first sidecar, gated on its own readiness probe, so no service starts
    # before its rules are in place. A sandbox without a network policy has no egress container.
    egress = next((c for c in containers if c.get("name") == EGRESS_CONTAINER), None)
    if egress is not None:
        containers.remove(egress)
        egress["restartPolicy"] = "Always"
        if egress.get("readinessProbe"):
            egress["startupProbe"] = {**egress["readinessProbe"], "failureThreshold": EGRESS_STARTUP_PROBE_FAILURES}
        init.append(egress)

    # Services in spec order: init containers start one at a time, which gives depends_on order.
    for svc in services.services:
        (containers if svc.role == "after_main" else init).append(service_container(svc, image_pull_policy))

    # Shared volumes and per-service /dev/shm are emptyDirs, so they live and die with the pod.
    volumes = pod_spec.setdefault("volumes", [])
    volumes.extend({"name": f"nemo-vol-{v}", "emptyDir": {}} for v in services.volumes)
    volumes.extend(
        {"name": f"nemo-shm-{s.name}", "emptyDir": {"medium": "Memory", "sizeLimit": s.shm_size}}
        for s in services.services
        if s.shm_size is not None
    )

    # main is the stock sandbox container; it only gains mounts and the one allowed capability.
    sandbox = containers[0]
    if services.main.volume_mounts:
        sandbox.setdefault("volumeMounts", []).extend(_mounts(services.main.volume_mounts))
    if services.main.add_capabilities:
        caps = sandbox.setdefault("securityContext", {}).setdefault("capabilities", {})
        caps["add"] = sorted(set(caps.get("add") or []) | set(services.main.add_capabilities))

    # Every container shares the pod network, so each Compose name is 127.0.0.1.
    pod_spec.setdefault("hostAliases", []).append({"ip": "127.0.0.1", "hostnames": [*services.names(), "main"]})


def service_states(pod: Any, services: ComposeServices) -> tuple[tuple[str, str] | None, list[str]]:
    """``(failure, not_ready)`` for the pod's services.

    ``failure`` is ``(service, why)`` for the first service, in start order, that can't come up
    anymore; None while they all still may. ``not_ready`` lists the services that aren't up yet:
    a ``run_once`` service until it exited 0, any other until Kubernetes reports it ready.
    """
    # Init containers and regular containers report separately; services can be either.
    status = pod.status
    statuses = {s.name: s for s in [*(status.init_container_statuses or []), *(status.container_statuses or [])]}
    failure: tuple[str, str] | None = None
    not_ready = []
    for svc in services.services:
        # A container Kubernetes hasn't reached yet has no status at all.
        st = statuses.get(svc.name)
        state = st.state if st is not None else None
        term = state.terminated if state is not None else None

        # Readiness: run_once is done when it exited 0; the rest when Kubernetes says ready.
        if svc.role == "run_once":
            if term is None or term.exit_code != 0:
                not_ready.append(svc.name)
        elif st is None or not st.ready:
            not_ready.append(svc.name)

        # Failure: only the first one, in start order, is reported, since it holds up the rest.
        if st is None or failure is not None:
            continue
        waiting = state.waiting if state is not None else None
        # A restarting sidecar is running or waiting again; its exit is in last_state.
        last = term or (st.last_state.terminated if st.last_state is not None else None)
        if waiting is not None and waiting.reason in FATAL_WAITING_REASONS:
            failure = (svc.name, f"{waiting.reason}: {waiting.message or ''}".strip())
        elif last is not None and (
            # A run_once that exited non-zero has failed; the task's setup step didn't complete.
            (svc.role == "run_once" and last.exit_code != 0)
            # A BatchSandbox pod never restarts regular containers, so any exit is final.
            or svc.role == "after_main"
            or (svc.role == "sidecar" and (st.restart_count or 0) >= CRASH_RESTARTS)
        ):
            failure = (svc.name, f"exited with code {last.exit_code} ({last.reason})")
    return failure, not_ready
