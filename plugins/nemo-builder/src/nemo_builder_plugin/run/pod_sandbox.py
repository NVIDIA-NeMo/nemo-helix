# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``kubernetes_pod`` provider: each sandbox is a plain pod the build step creates, hardened by the builder.

The pod runs one script that builds its images in turn, and the build step learns each image's result from the
marker lines in the pod's log.
"""

from __future__ import annotations

import hashlib
import logging
import shlex
import time

from kubernetes import client as k8s
from kubernetes import watch as k8s_watch
from nemo_builder_plugin.run.sandbox import (
    BUILD_TIMEOUT_SECONDS,
    JOB_LABEL,
    SANDBOX_DEADLINE_SECONDS,
    clear_output,
    job_key,
    kaniko_command,
    mounts,
    sandbox_labels,
)
from nemo_builder_plugin.steps import SandboxGroup, SandboxSpec
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: All kaniko needs of the container runtime's default capabilities.
KANIKO_CAPABILITIES = ["CHOWN", "DAC_OVERRIDE", "FOWNER", "SETUID", "SETGID"]

RESULT_MARKER = "NHX_IMAGE_RESULT"

#: How long a sweep waits for an earlier attempt's pods to go: a pod of the same name can't be created until then.
_SWEEP_TIMEOUT_SECONDS = 120


def _build_script(group: SandboxGroup) -> str:
    """The shell the sandbox runs: one kaniko invocation per image, in turn, with ``--cleanup`` between them.

    No ``set -e``, so one failure doesn't stop the rest; each marker line's ``$?`` must be kaniko's status.
    """
    lines = ["set -u"]
    for image in group.images:
        lines.append(clear_output(image))
        lines.append(kaniko_command(group, image))
        # On a line of its own: kaniko passes a `RUN`'s output through as is, so it may not end in a newline.
        lines.append(f"printf '\\n%s %s %s\\n' {RESULT_MARKER} {shlex.quote(image.image)} \"$?\"")
    return "\n".join(lines)


def sandbox_pod_name(workspace: str, job_id: str, index: int) -> str:
    """A legal pod name, unique across workspaces: job names are unique only within one."""
    suffix = hashlib.sha256(f"{workspace}/{job_id}".encode()).hexdigest()[:10]
    return f"nhx-sbx-{job_id[:36].rstrip('-')}-{suffix}-g{index}"


def _pod_manifest(
    *,
    name: str,
    namespace: str,
    group: SandboxGroup,
    sandbox: SandboxSpec,
    labels: dict[str, str],
    job_sub_path: str,
) -> k8s.V1Pod:
    """The sandbox pod. It runs caller code, so it gets no ServiceAccount token, no secret, and no credential in its env."""
    return k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(name=name, namespace=namespace, labels=labels),
        spec=k8s.V1PodSpec(
            restart_policy="Never",
            active_deadline_seconds=SANDBOX_DEADLINE_SECONDS,
            automount_service_account_token=False,
            # Otherwise the kubelet injects env vars naming every Service in the namespace.
            enable_service_links=False,
            node_selector=sandbox.node_selector or None,
            # Public resolvers, not cluster DNS, so a NetworkPolicy can block every cluster address.
            dns_policy="None",
            dns_config=k8s.V1PodDNSConfig(nameservers=list(sandbox.dns_nameservers)),
            containers=[
                k8s.V1Container(
                    name="build",
                    image=sandbox.image,
                    command=["/busybox/sh", "-c", _build_script(group)],
                    security_context=k8s.V1SecurityContext(
                        # kaniko needs root to unpack layers whose files belong to many uids.
                        run_as_user=0,
                        allow_privilege_escalation=False,
                        # Not `Unconfined`, which the namespace's `baseline` standard refuses.
                        seccomp_profile=k8s.V1SeccompProfile(type="RuntimeDefault"),
                        capabilities=k8s.V1Capabilities(drop=["ALL"], add=KANIKO_CAPABILITIES),
                    ),
                    volume_mounts=[
                        k8s.V1VolumeMount(
                            name="work",
                            sub_path=mount.sub_path,
                            mount_path=mount.mount_path,
                            read_only=mount.read_only or None,
                        )
                        for mount in mounts(group, job_sub_path)
                    ],
                    resources=k8s.V1ResourceRequirements(
                        requests={"cpu": sandbox.cpu, "memory": sandbox.memory},
                    ),
                )
            ],
            volumes=[
                k8s.V1Volume(
                    name="work",
                    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=sandbox.work_pvc),
                )
            ],
        ),
    )


def _await_pod(api: k8s.CoreV1Api, *, name: str, namespace: str) -> str:
    watcher = k8s_watch.Watch()
    deadline = time.monotonic() + BUILD_TIMEOUT_SECONDS
    try:
        for event in watcher.stream(
            api.list_namespaced_pod,
            namespace=namespace,
            field_selector=f"metadata.name={name}",
            timeout_seconds=BUILD_TIMEOUT_SECONDS,
        ):
            phase = event["object"].status.phase
            logger.info("sandbox %s: %s", sanitize_for_log(name), sanitize_for_log(phase))
            if phase in ("Succeeded", "Failed"):
                watcher.stop()
                return phase
            if time.monotonic() > deadline:
                break
    finally:
        watcher.stop()
    return "Unknown"


def _read_pod_log(api: k8s.CoreV1Api, *, name: str, namespace: str) -> str:
    """The sandbox's log, decoded here: with ``_preload_content`` on, the client can return the repr of its bytes."""
    response = api.read_namespaced_pod_log(name=name, namespace=namespace, _preload_content=False)
    return response.data.decode("utf-8", errors="replace")


def _delete_pod(api: k8s.CoreV1Api, *, name: str, namespace: str) -> None:
    """Delete a sandbox, logging rather than raising if that fails: an error here would replace the one being
    handled, or cost the images a sandbox has already built."""
    try:
        api.delete_namespaced_pod(name=name, namespace=namespace, grace_period_seconds=0)
    except Exception:
        logger.warning("sandbox %s could not be deleted", sanitize_for_log(name), exc_info=True)


def _run_sandbox(api: k8s.CoreV1Api, *, name: str, manifest: k8s.V1Pod, namespace: str) -> tuple[str, str]:
    """Run one sandbox to its end, deleting it whatever happens. Returns its phase and its log."""
    api.create_namespaced_pod(namespace=namespace, body=manifest)
    try:
        phase = _await_pod(api, name=name, namespace=namespace)
        return phase, _read_pod_log(api, name=name, namespace=namespace)
    finally:
        _delete_pod(api, name=name, namespace=namespace)


def _results_from_log(log: str | bytes) -> dict[str, int]:
    """Each image's kaniko exit code, from the marker lines in the sandbox's log."""
    if isinstance(log, bytes):
        log = log.decode("utf-8", errors="replace")

    results: dict[str, int] = {}
    for line in log.splitlines():
        if not line.startswith(RESULT_MARKER):
            continue
        parts = line.split()
        if len(parts) == 3 and parts[2].isdigit():
            results[parts[1]] = int(parts[2])
    return results


class KubernetesPodProvider:
    """:class:`~nemo_builder_plugin.run.sandbox.SandboxProvider` creating plain pods in this step's namespace."""

    def __init__(
        self,
        api: k8s.CoreV1Api,
        *,
        namespace: str,
        sandbox: SandboxSpec,
        workspace: str,
        job_id: str,
        job_sub_path: str,
    ) -> None:
        self._api = api
        self._namespace = namespace
        self._sandbox = sandbox
        self._workspace = workspace
        self._job_id = job_id
        self._job_sub_path = job_sub_path

    def _leftovers(self) -> list[str]:
        selector = f"{JOB_LABEL}={job_key(self._workspace, self._job_id)}"
        pods = self._api.list_namespaced_pod(namespace=self._namespace, label_selector=selector)
        return [pod.metadata.name for pod in pods.items]

    def sweep(self) -> None:
        """An earlier attempt's pods have this attempt's names, so this waits until they're gone, not just deleted."""
        leftovers = self._leftovers()
        if not leftovers:
            return
        logger.info("deleting %d sandbox(es) an earlier attempt left", len(leftovers))
        for name in leftovers:
            _delete_pod(self._api, name=name, namespace=self._namespace)
        deadline = time.monotonic() + _SWEEP_TIMEOUT_SECONDS
        while leftovers := self._leftovers():
            if time.monotonic() > deadline:
                raise RuntimeError(f"an earlier attempt's sandboxes are still there: {', '.join(leftovers)}")
            time.sleep(1)

    def build(self, index: int, group: SandboxGroup) -> dict[str, int]:
        name = sandbox_pod_name(self._workspace, self._job_id, index)
        manifest = _pod_manifest(
            name=name,
            namespace=self._namespace,
            group=group,
            sandbox=self._sandbox,
            labels=sandbox_labels(self._workspace, self._job_id),
            job_sub_path=self._job_sub_path,
        )
        logger.info("creating sandbox %s for %d image(s)", sanitize_for_log(name), len(group.images))
        phase, log = _run_sandbox(self._api, name=name, manifest=manifest, namespace=self._namespace)
        # Line by line: sanitizing the whole log at once would run it into one line.
        logger.info("sandbox %s ended %s; its log:", sanitize_for_log(name), sanitize_for_log(phase))
        for line in log.splitlines():
            logger.info("  %s", sanitize_for_log(line))
        return _results_from_log(log)
