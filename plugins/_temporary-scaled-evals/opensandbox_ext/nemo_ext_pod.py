# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adds a task's Compose services and agent bundle to the BatchSandbox pod the stock server built, and reads their state back.

The stock pod has the task container (``sandbox``), an ``egress`` container that enforces the
network policy, and an ``execd-installer`` init container. Kubernetes does all the ordering:

* ``egress`` moves to the front of the init containers, so its rules cover everything after it.
* The agent bundle becomes a plain init container that copies the agent into ``/installed-agent``.
* ``sidecar`` services become native sidecars (init containers with ``restartPolicy: Always``).
  Init containers start one at a time, and a sidecar's startup probe (its ``readiness`` command)
  holds the next one, and ``sandbox``, until it passes.
* ``run_once`` services become plain init containers, which must exit 0 before the next starts.
* ``after_main`` services become regular containers that start next to ``sandbox``.
* Host aliases resolve every service name, and ``main``, to 127.0.0.1.

Pure functions over manifest dicts and kubernetes ``V1Pod`` objects; no API calls.
"""

from __future__ import annotations

from typing import Any

from nhx_sandbox.harbor_opensandbox_services import (
    AGENT_BUNDLE_CONTAINER,
    INSTALLED_AGENT_DIR,
    ComposeServices,
    Readiness,
    SandboxAgentBundle,
    Service,
    VolumeMount,
)

SANDBOX_CONTAINER = "sandbox"
EGRESS_CONTAINER = "egress"
# The volume the bundle is installed into when the template doesn't already mount /installed-agent.
INSTALLED_AGENT_VOLUME = "nemo-installed-agent"

# Probes run every second, so failure thresholds are roughly seconds while checks answer quickly: a
# service gets ~10 minutes to become ready once it runs, egress 2 minutes.
STARTUP_PROBE_FAILURES = 600
EGRESS_STARTUP_PROBE_FAILURES = 120
READINESS_PROBE_FAILURES = 3
# How long one service check may run: Compose's default healthcheck timeout, which is what a check
# that doesn't set its own ``timeout`` gets under Harbor.
PROBE_TIMEOUT_SECONDS = 30

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
        "timeoutSeconds": PROBE_TIMEOUT_SECONDS,
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


def _pod_containers(pod_spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The pod's ``(containers, initContainers)``; ``ValueError`` if the stock pod isn't shaped as expected.

    The ordering the callers rely on means nothing on a pod of another shape.
    """
    containers: list[dict[str, Any]] = pod_spec["containers"]
    init: list[dict[str, Any]] = pod_spec.setdefault("initContainers", [])

    if not containers or containers[0].get("name") != SANDBOX_CONTAINER:
        raise ValueError(f"unexpected BatchSandbox pod: the first container is not {SANDBOX_CONTAINER!r}")

    return containers, init


def _egress_first(containers: list[dict[str, Any]], init: list[dict[str, Any]]) -> None:
    """Make ``egress`` the next init container, as a sidecar gated on its own readiness probe.

    Every init container added after it then starts only once the egress rules are in place.
    A no-op when it already moved, and for a sandbox without a network policy, which has none.
    """
    egress = next((c for c in containers if c.get("name") == EGRESS_CONTAINER), None)
    if egress is None:
        return
    containers.remove(egress)
    egress["restartPolicy"] = "Always"
    if egress.get("readinessProbe"):
        egress["startupProbe"] = {**egress["readinessProbe"], "failureThreshold": EGRESS_STARTUP_PROBE_FAILURES}
    init.append(egress)


def add_agent_bundle(pod_spec: dict[str, Any], bundle: SandboxAgentBundle, image_pull_policy: str) -> None:
    """Add the init container that installs ``bundle`` into ``/installed-agent``; ``ValueError`` on an unexpected pod.

    It runs after ``egress``, so the installer gets the sandbox's egress rules, and before any
    Compose service, which ``add_services`` appends after it. ``main`` starts only once it
    exits 0, and mounts ``/installed-agent`` read-only, so the agent can't change its own
    installation. The template's ``/installed-agent`` volume is reused when it has one.
    """
    containers, init = _pod_containers(pod_spec)
    if AGENT_BUNDLE_CONTAINER in {c.get("name") for c in [*containers, *init]}:
        raise ValueError(f"the sandbox pod already has a container named {AGENT_BUNDLE_CONTAINER!r}")

    # The installer runs under the sandbox's egress rules, like everything else in the pod.
    _egress_first(containers, init)

    # main reads the agent from /installed-agent: reuse the template's volume there, or add an emptyDir.
    sandbox = containers[0]
    mounts: list[dict[str, Any]] = sandbox.setdefault("volumeMounts", [])
    installed_agent_mount: dict[str, Any] | None = next(
        (m for m in mounts if m.get("mountPath") == INSTALLED_AGENT_DIR), None
    )
    if installed_agent_mount is None:
        installed_agent_mount = {"name": INSTALLED_AGENT_VOLUME, "mountPath": INSTALLED_AGENT_DIR}
        mounts.append(installed_agent_mount)
        pod_spec.setdefault("volumes", []).append({"name": INSTALLED_AGENT_VOLUME, "emptyDir": {}})
    installed_agent_mount["readOnly"] = True

    # The installer gets the same volume writable, and copies the agent into it.
    bundle_container: dict[str, Any] = {
        "name": AGENT_BUNDLE_CONTAINER,
        "image": bundle.image,
        # The operator's policy, as for the task image and the services.
        "imagePullPolicy": image_pull_policy,
        "command": [bundle.installer, INSTALLED_AGENT_DIR],
        "volumeMounts": [{"name": installed_agent_mount["name"], "mountPath": INSTALLED_AGENT_DIR}],
        # It shares the pod network with egress; without these it could rewrite egress's rules.
        "securityContext": {"privileged": False, "capabilities": {"drop": ["NET_ADMIN", "NET_RAW"]}},
    }
    init.append(bundle_container)


def agent_bundle_state(pod: Any) -> tuple[str | None, bool]:
    """``(failure, installed)`` for the bundle's init container: why it can't finish, and whether it exited 0."""
    statuses = pod.status.init_container_statuses or []
    st = next((s for s in statuses if s.name == AGENT_BUNDLE_CONTAINER), None)
    # Kubernetes hasn't reached it yet.
    if st is None or st.state is None:
        return None, False

    waiting, term = st.state.waiting, st.state.terminated
    # An image that can't be pulled never clears on its own.
    if waiting is not None and waiting.reason in FATAL_WAITING_REASONS:
        return f"{waiting.reason}: {waiting.message or ''}".strip(), False
    # Still pulling or copying.
    if term is None:
        return None, False

    # A BatchSandbox pod never restarts its containers, so a non-zero exit is final.
    if term.exit_code != 0:
        return f"exited with code {term.exit_code} ({term.reason})", False
    return None, True


def add_services(pod_spec: dict[str, Any], services: ComposeServices, image_pull_policy: str) -> None:
    """Add ``services`` to the BatchSandbox pod spec in place; ``ValueError`` if the pod isn't shaped as expected.

    Init containers start in list order, after the stock ones (``execd-installer``), so the final
    order is: stock init containers, ``egress``, the agent bundle if any, then the services in
    spec order. ``after_main`` services join ``sandbox`` as regular containers.
    """
    containers, init = _pod_containers(pod_spec)
    # Service names the pod's own containers already use.
    conflicting_names = set(services.names()).intersection(c.get("name") for c in [*containers, *init])
    if conflicting_names:
        raise ValueError(f"service names clash with sandbox pod containers: {sorted(conflicting_names)}")

    # egress becomes the first sidecar, so no service starts before its rules are in place.
    _egress_first(containers, init)

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

    # main is the stock sandbox container; it only gains mounts, the one allowed capability and its working directory.
    sandbox = containers[0]
    if services.main.working_dir is not None:
        # The stock server sets none, so the image's WORKDIR applied; Compose's `working_dir` replaces it.
        sandbox["workingDir"] = services.main.working_dir
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
