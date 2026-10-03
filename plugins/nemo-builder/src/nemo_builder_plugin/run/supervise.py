# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nhx-build supervise``: the build step. It runs the sandbox pods that build with kaniko.

It mounts no volume itself, so it learns each image's result from its sandbox's log.
"""

from __future__ import annotations

import hashlib
import logging
import shlex
import time
from pathlib import PurePosixPath

from kubernetes import client as k8s
from kubernetes import config as k8s_config
from kubernetes import watch as k8s_watch
from nemo_builder_plugin.run.context import job_identity, read_step_config
from nemo_builder_plugin.steps import SandboxGroup, SandboxSpec, SuperviseStepConfig, WorkLayout
from nemo_helix_plugin.jobs.constants import job_storage_subpath
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: The root of the job's `WorkLayout` in the sandbox. A `RUN` that writes here writes to the work
#: volume, so it is a name no Dockerfile would plausibly use.
SANDBOX_ROOT = PurePosixPath("/nhx-work")

#: All kaniko needs of the container runtime's default capabilities.
KANIKO_CAPABILITIES = ["CHOWN", "DAC_OVERRIDE", "FOWNER", "SETUID", "SETGID"]

RESULT_MARKER = "NHX_IMAGE_RESULT"

KANIKO_EXECUTOR = "/kaniko/executor"

_POD_TIMEOUT_SECONDS = 60 * 60

#: Enforced by the kubelet, so a sandbox ends even if this step is killed before it can delete it.
_POD_DEADLINE_SECONDS = _POD_TIMEOUT_SECONDS + 5 * 60


def _build_script(group: SandboxGroup, sandbox: SandboxSpec) -> str:
    """The shell the sandbox runs: one kaniko invocation per image, in turn, with ``--cleanup`` between them.

    No ``set -e``, so one failure doesn't stop the rest; each marker line's ``$?`` must be kaniko's status.
    """
    view = WorkLayout(SANDBOX_ROOT)
    lines = ["set -u"]
    for image in group.images:
        layout = view.output(image.image)
        # Empty the output first, or a failed build would leave an earlier attempt's layout for
        # `push` to publish. It is a mount point, so only its contents can go.
        quoted = shlex.quote(str(layout))
        lines.append(f"rm -rf {quoted}/* {quoted}/.[!.]* {quoted}/..?*")
        args = [
            KANIKO_EXECUTOR,
            f"--context=dir://{view.context(group.source)}",
            f"--dockerfile={image.dockerfile}",
            f"--custom-platform={image.platform}",
            "--no-push",
            "--no-push-cache",
            f"--oci-layout-path={layout}",
            "--cleanup",
            "--verbosity=info",
        ]
        lines.append(" ".join(shlex.quote(a) for a in args))
        lines.append(f'echo {RESULT_MARKER} {shlex.quote(image.image)} "$?"')
    return "\n".join(lines)


def _volume_mounts(group: SandboxGroup, job_sub_path: str) -> list[k8s.V1VolumeMount]:
    """The group's own context, read-only, and an output directory for each of its images.

    Never the whole output directory, where a later group's ``RUN`` could rewrite an earlier group's layout.
    """
    volume, view = WorkLayout(PurePosixPath(job_sub_path)), WorkLayout(SANDBOX_ROOT)
    return [
        k8s.V1VolumeMount(
            name="work",
            sub_path=str(volume.context(group.source)),
            mount_path=str(view.context(group.source)),
            read_only=True,
        ),
        *(
            k8s.V1VolumeMount(
                name="work", sub_path=str(volume.output(image.image)), mount_path=str(view.output(image.image))
            )
            for image in group.images
        ),
    ]


def sandbox_pod_name(workspace: str, job_id: str, index: int) -> str:
    """A legal pod name, unique across workspaces: job names are unique only within one."""
    suffix = hashlib.sha256(f"{workspace}/{job_id}".encode()).hexdigest()[:10]
    return f"nhx-sbx-{job_id[:36].rstrip('-')}-{suffix}-g{index}"


def _pod_manifest(
    *,
    name: str,
    group: SandboxGroup,
    sandbox: SandboxSpec,
    pvc: str,
    job_sub_path: str,
) -> k8s.V1Pod:
    """The sandbox pod. It runs caller code, so it gets no ServiceAccount token, env or secret."""
    return k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(
            name=name,
            namespace=sandbox.namespace,
            # What a NetworkPolicy selects the sandbox on.
            labels={"nhx.nvidia.com/sandbox": "true", "app.kubernetes.io/managed-by": "nemo-builder"},
        ),
        spec=k8s.V1PodSpec(
            restart_policy="Never",
            active_deadline_seconds=_POD_DEADLINE_SECONDS,
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
                    command=["/busybox/sh", "-c", _build_script(group, sandbox)],
                    security_context=k8s.V1SecurityContext(
                        # kaniko needs root to unpack layers whose files belong to many uids.
                        run_as_user=0,
                        allow_privilege_escalation=False,
                        # Not `Unconfined`, which the namespace's `baseline` standard refuses.
                        seccomp_profile=k8s.V1SeccompProfile(type="RuntimeDefault"),
                        capabilities=k8s.V1Capabilities(drop=["ALL"], add=KANIKO_CAPABILITIES),
                    ),
                    volume_mounts=_volume_mounts(group, job_sub_path),
                    resources=k8s.V1ResourceRequirements(
                        requests={"cpu": sandbox.cpu, "memory": sandbox.memory},
                    ),
                )
            ],
            volumes=[
                k8s.V1Volume(
                    name="work",
                    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=pvc),
                )
            ],
        ),
    )


def _await_pod(api: k8s.CoreV1Api, *, name: str, namespace: str) -> str:
    watcher = k8s_watch.Watch()
    deadline = time.monotonic() + _POD_TIMEOUT_SECONDS
    try:
        for event in watcher.stream(
            api.list_namespaced_pod,
            namespace=namespace,
            field_selector=f"metadata.name={name}",
            timeout_seconds=_POD_TIMEOUT_SECONDS,
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


def _exit_code(*, failures: int, total: int) -> int:
    """Non-zero only if no image built, since Jobs runs ``push`` only after this step succeeds.

    A partial failure still fails the job: ``push`` counts each missing layout as a failure.
    """
    if total and failures >= total:
        return 1
    return 0


def main() -> int:
    config = SuperviseStepConfig.model_validate(read_step_config())
    workspace, job_id = job_identity()
    job_sub_path = job_storage_subpath(workspace, job_id)

    k8s_config.load_incluster_config()
    api = k8s.CoreV1Api()

    total = sum(len(group.images) for group in config.groups)
    failures = 0
    for index, group in enumerate(config.groups):
        pod_name = sandbox_pod_name(workspace, job_id, index)
        manifest = _pod_manifest(
            name=pod_name,
            group=group,
            sandbox=config.sandbox,
            pvc=config.sandbox.work_pvc,
            job_sub_path=job_sub_path,
        )

        logger.info("creating sandbox %s for %d image(s)", sanitize_for_log(pod_name), len(group.images))
        api.create_namespaced_pod(namespace=config.sandbox.namespace, body=manifest)
        try:
            phase = _await_pod(api, name=pod_name, namespace=config.sandbox.namespace)
            log = _read_pod_log(api, name=pod_name, namespace=config.sandbox.namespace)
            logger.info("sandbox %s log:\n%s", sanitize_for_log(pod_name), sanitize_for_log(log))
            results = _results_from_log(log)

            for image in group.images:
                code = results.get(image.image)
                if code is None:
                    logger.error(
                        "image %s: no result recorded (sandbox phase %s)",
                        sanitize_for_log(image.image),
                        sanitize_for_log(phase),
                    )
                    failures += 1
                elif code != 0:
                    logger.error("image %s: kaniko exited %d", sanitize_for_log(image.image), code)
                    failures += 1
                else:
                    logger.info("image %s: built", sanitize_for_log(image.image))
        finally:
            api.delete_namespaced_pod(name=pod_name, namespace=config.sandbox.namespace, grace_period_seconds=0)

    if failures:
        logger.error("%d of %d image(s) failed to build", failures, total)
    return _exit_code(failures=failures, total=total)
