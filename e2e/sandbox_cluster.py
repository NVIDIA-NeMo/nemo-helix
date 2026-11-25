# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a cluster's OpenSandbox sandboxes with kubectl, for e2e tests whose work runs in them.

``SandboxCluster`` reads the cluster on a named kubectl context, never the current one. ``GymHostWatch``
records a job's Gym host while the job runs: the host is meant to be gone by the time the job ends,
so what the watch saw is then the only record of it.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

#: The label the OpenSandbox server puts on every sandbox pod it creates.
SANDBOX_POD_LABEL = "opensandbox.io/id"
#: The labels the chart gives the platform API's pods: the default ``api.componentLabel`` and the
#: legacy one that k8s/helm/README.md tells installations from before the rename to keep.
PLATFORM_API_POD_SELECTOR = "app.kubernetes.io/component in (nhx-api,nmp-api)"
GYM_HOST_POLL_INTERVAL_SECONDS = 2.0
#: Run in a platform API pod: connects to each ``host:port`` in argv[1] (a JSON list) and prints the
#: ones it could not reach, with their errors, as a JSON object.
_CONTROL_PROBE_SOURCE = """\
import json, socket, sys
failed = {}
for target in json.loads(sys.argv[1]):
    host, _, port = target.rpartition(":")
    try:
        socket.create_connection((host, int(port)), timeout=10).close()
    except OSError as error:
        failed[target] = f"{type(error).__name__}: {error}"
print(json.dumps(failed))
"""


@dataclass(frozen=True)
class SandboxCluster:
    """The cluster the platform runs on, read with kubectl on the named context only."""

    context: str
    namespace: str

    def _get(self, *resource: str, namespace: str | None = None) -> dict[str, Any]:
        """``kubectl get`` as JSON. However it fails, it raises RuntimeError, so callers handle one type."""
        cmd = ["kubectl", "--context", self.context, "--namespace", namespace or self.namespace, "get", *resource]
        try:
            result = subprocess.run(
                [*cmd, "-o", "json"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"`{' '.join(cmd)}` did not complete: {error}") from error
        if result.returncode != 0:
            raise RuntimeError(f"`{' '.join(cmd)}` failed: {result.stderr.strip()}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"`{' '.join(cmd)}` did not print JSON: {error}") from error

    def api_server_address(self) -> str:
        """The Kubernetes API's in-cluster ``address:port``, which any pod reaches unless something stops it."""
        spec = self._get("service", "kubernetes", namespace="default")["spec"]
        return f"{spec['clusterIP']}:{spec['ports'][0]['port']}"

    def runs_platform_api(self) -> bool:
        """Whether the namespace has a platform API pod, by the labels the chart gives them."""
        return bool(self._get("pods", "--selector", PLATFORM_API_POD_SELECTOR).get("items"))

    def unreachable_from_platform_api(self, targets: Sequence[str]) -> dict[str, str]:
        """The ``host:port`` targets a running platform API pod cannot connect to, each with its error.

        The chart restricts only the API's ingress, so the API pod is the cluster's control for a
        destination a sandbox is expected to be denied: a target it cannot reach either says nothing
        about the sandbox's egress rules.
        """
        pods = self._get("pods", "--selector", PLATFORM_API_POD_SELECTOR).get("items") or []
        running = [pod["metadata"]["name"] for pod in pods if (pod.get("status") or {}).get("phase") == "Running"]
        if not running:
            raise RuntimeError(f"no running platform API pod ({PLATFORM_API_POD_SELECTOR})")
        cmd = ["kubectl", "--context", self.context, "--namespace", self.namespace, "exec", running[0], "--"]
        cmd += ["python", "-c", _CONTROL_PROBE_SOURCE, json.dumps(list(targets))]
        try:
            result = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"`kubectl exec {running[0]}` did not complete: {error}") from error
        if result.returncode != 0:
            raise RuntimeError(f"`kubectl exec {running[0]}` failed: {result.stderr.strip()}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"`kubectl exec {running[0]}` did not print JSON: {error}") from error

    def gym_hosts(self, markers: frozenset[str], known: frozenset[str] = frozenset()) -> dict[str, dict[str, Any]]:
        """A job's Gym host: the BatchSandboxes that name the job and their pods, by ``kind/name``.

        A sandbox carries its job's name, which the platform hands a job as its id, in its metadata,
        its bootstrap environment and its PVC sub-paths. Matching on it is also how
        k8s/helm/examples/opensandbox/verify/lib.sh finds a sandbox. A pod is the host's when it has the
        label OpenSandbox gives the pods it creates and either belongs to one of those BatchSandboxes
        or names the job itself. The label matters: the job's own step pods name the job too. The
        server selects on it, so a busy namespace costs no more to read than an idle one.

        ``known`` holds the ``kind/name`` of objects already identified as the host. They count for
        as long as they exist, and a known BatchSandbox keeps claiming its pods after it is deleted:
        OpenSandbox removes it first, and its pod can outlive it by the termination grace period.
        """

        def names_job(item: Mapping[str, Any]) -> bool:
            manifest = json.dumps(item)
            return any(marker in manifest for marker in markers)

        found: dict[str, dict[str, Any]] = {}
        sandboxes = {key.removeprefix("batchsandboxes/") for key in known if key.startswith("batchsandboxes/")}
        for sandbox in self._get("batchsandboxes").get("items") or []:
            name = sandbox["metadata"]["name"]
            if name in sandboxes or names_job(sandbox):
                sandboxes.add(name)
                found[f"batchsandboxes/{name}"] = sandbox
        for pod in self._get("pods", "--selector", SANDBOX_POD_LABEL).get("items") or []:
            metadata = pod.get("metadata") or {}
            sandbox_id = (metadata.get("labels") or {}).get(SANDBOX_POD_LABEL)
            if sandbox_id is None:
                # The selector already excludes these. Counting a job's own step pod as its host would
                # report a leak that is not one, so do not rely on the selector alone.
                continue
            owners = {owner.get("name") for owner in metadata.get("ownerReferences") or []}
            key = f"pods/{metadata.get('name')}"
            if key in known or sandbox_id in sandboxes or owners & sandboxes or names_job(pod):
                found[key] = pod
        return found


class GymHostWatch:
    """Records a job's Gym host while the job runs, because the host is meant to be gone when it ends."""

    def __init__(self, cluster: SandboxCluster, markers: frozenset[str]) -> None:
        self.markers = markers
        #: Every host object seen, and the sandbox pods seen running, each as last seen.
        self.seen: dict[str, dict[str, Any]] = {}
        self.running_pods: dict[str, dict[str, Any]] = {}
        self.failed_polls: list[str] = []
        self._cluster = cluster
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, name="gym-host-watch", daemon=True)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                hosts = self._cluster.gym_hosts(self.markers, frozenset(self.seen))
            except Exception as error:
                # Nothing may end the watch: the next read may still see the host, and a thread that died
                # here would leave "no host was seen" as the only symptom. kubectl failures are RuntimeError.
                self.failed_polls.append(f"{type(error).__name__}: {error}")
            else:
                self.seen.update(hosts)
                self.running_pods.update(
                    {
                        name: host
                        for name, host in hosts.items()
                        if name.startswith("pods/") and (host.get("status") or {}).get("phase") == "Running"
                    }
                )
            self._stop.wait(GYM_HOST_POLL_INTERVAL_SECONDS)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=120)

    def report(self) -> str:
        """Each host object as last seen, for a failure message.

        The platform removes the host as the job ends, so after a failure this is the only record of
        how far the host got, and why it stopped there.
        """
        seen = dict(self.seen)  # The poll may still be updating it.
        if not seen:
            return f"no host object seen; failed polls: {self.failed_polls[-3:]}"
        states = [f"{key}: {_host_object_state(key, item)}" for key, item in sorted(seen.items())]
        return "; ".join(states) + (f"; failed polls: {self.failed_polls[-3:]}" if self.failed_polls else "")


def _host_object_state(key: str, item: Mapping[str, Any]) -> str:
    """A host pod's phase, conditions and containers, or a BatchSandbox's status, in one line."""
    status = item.get("status") or {}
    if not key.startswith("pods/"):
        return json.dumps(status, separators=(",", ":"))
    parts = [f"phase {status.get('phase')}"]
    if status.get("reason"):
        parts.append(f"{status['reason']}: {status.get('message')}")
    for condition in status.get("conditions") or []:
        if condition.get("status") != "True":
            parts.append(f"{condition.get('type')} {condition.get('status')} ({condition.get('reason')})")
    for container in [*(status.get("initContainerStatuses") or []), *(status.get("containerStatuses") or [])]:
        states = []
        for when, state in (("now", container.get("state")), ("last", container.get("lastState"))):
            for name, detail in (state or {}).items():
                exit_code = detail.get("exitCode")
                words = [name, detail.get("reason"), None if exit_code is None else f"exit {exit_code}"]
                states.append(f"{when} " + " ".join(word for word in words if word))
                if detail.get("message"):
                    states.append(detail["message"])
        parts.append(f"{container.get('name')} [{', '.join(states)}] {container.get('restartCount', 0)} restarts")
    return ", ".join(parts)
