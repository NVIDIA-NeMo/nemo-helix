# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nmp-build supervise`` -- step 2. The trusted control plane for an untrusted pod.

It holds RBAC to create, watch and delete Pods in the build namespace, and **nothing else**: no
Files client, no registry credential, no signing key. It creates the sandbox, watches it, records
what happened, and deletes it.

**It mounts no volume at all.** That is not an oversight -- it is why this step cannot read a
build context even by mistake, and it is enforced by the compiler emitting no storage env var for
it rather than by anything here. The consequence is that everything this step learns about a
build, it learns from the pod's *log*, which is also all it is permitted to read.

**One sandbox per group, one kaniko invocation per image inside it, sequential.** The sequencing
is not a throughput choice: invocations share one container root and measurably cannot overlap.
Parallelism across a set comes from multiple groups, which are separate pods.

This is the first step binary in this repository to talk to the Kubernetes API. Everything else
that does lives in the jobs controller, which is why the Role backing it is written narrowly and
bound to this identity alone.
"""

from __future__ import annotations

import logging
import shlex
import time

from kubernetes import client as k8s
from kubernetes import config as k8s_config
from kubernetes import watch as k8s_watch
from nemo_builder_plugin.run.context import job_identity, read_step_config
from nemo_builder_plugin.steps import SandboxGroup, SandboxSpec, SuperviseStepConfig
from nemo_platform_plugin.jobs.constants import job_storage_subpath

logger = logging.getLogger(__name__)

SANDBOX_CONTEXT_MOUNT = "/ctx"
SANDBOX_OUTPUT_MOUNT = "/out"

#: The five capabilities kaniko needs out of containerd's default fourteen. Measured by ablation:
#: dropping all of them fails at `chown /etc/gshadow: operation not permitted`, because extracting
#: layers that own files across many uids IS the job. Keeping only these drops NET_RAW and MKNOD
#: -- raw sockets and device nodes, the two most useful to an attacker -- along with seven others.
KANIKO_CAPABILITIES = ["CHOWN", "DAC_OVERRIDE", "FOWNER", "SETUID", "SETGID"]

#: Printed by the sandbox after each build so this step can attribute a result without reading
#: the volume the sandbox wrote to.
RESULT_MARKER = "NMP_IMAGE_RESULT"

_POD_TIMEOUT_SECONDS = 60 * 60


def _build_script(group: SandboxGroup, sandbox: SandboxSpec) -> str:
    """The shell the sandbox runs: one kaniko invocation per image, in order.

    ``--cleanup`` between invocations is what makes a shared container root safe to reuse.
    ``--no-push`` is what makes this pod credential-free: it writes an OCI layout to the work
    volume and the trusted push step publishes it.

    **The set is not aborted on a failure.** Each image gets its own exit line, so one broken
    Dockerfile in a set of ten does not cost the other nine -- which is also why the reconciler
    must ask the registry even when the job as a whole exited non-zero.
    """
    lines = ["set -u"]
    for image in group.images:
        args = [
            "/kaniko/executor",
            f"--context=dir://{image.context}",
            f"--dockerfile={image.dockerfile}",
            f"--custom-platform={image.platform}",
            "--no-push",
            "--no-push-cache",
            f"--oci-layout-path={image.layout}",
            f"--digest-file={image.layout}.digest",
            "--cleanup",
            "--verbosity=info",
        ]
        if sandbox.registry_mirror:
            # With a mirror configured, refuse to silently fall back past it -- otherwise a
            # deployment believes every FROM is bounded when some are not.
            args.append(f"--registry-mirror={sandbox.registry_mirror}")
            args.append("--skip-default-registry-fallback")
        lines.append(" ".join(shlex.quote(a) for a in args) + " || true")
        lines.append(f'echo "{RESULT_MARKER} {image.image} $?"')
    return "\n".join(lines)


def _pod_manifest(
    *,
    name: str,
    group: SandboxGroup,
    sandbox: SandboxSpec,
    pvc: str,
    job_sub_path: str,
) -> k8s.V1Pod:
    """The untrusted pod, composed entirely from operator config.

    Two lines carry most of the security argument and both are one line each:
    ``automount_service_account_token=False`` removes the control-plane identity, and the absence
    of any ``env``, ``env_from`` or secret volume removes the credential. Together they make the
    threat model's "read the credential" and "spend the credential" paths unreachable rather than
    merely firewalled.
    """
    return k8s.V1Pod(
        metadata=k8s.V1ObjectMeta(
            name=name,
            namespace=sandbox.namespace,
            # What the NetworkPolicy selects on. Without this label the pod is unconstrained.
            labels={"nmp.nvidia.com/sandbox": "true", "app.kubernetes.io/managed-by": "nemo-builder"},
        ),
        spec=k8s.V1PodSpec(
            restart_policy="Never",
            automount_service_account_token=False,
            runtime_class_name=sandbox.runtime_class or None,
            node_selector=sandbox.node_selector or None,
            # Public resolvers, not cluster DNS. Cluster DNS here answers on a link-local address
            # that the egress policy denies, and re-allowing it reopens the Pod CIDR. A sandbox
            # needs to resolve pypi.org, not kubernetes.default.svc.
            dns_policy="None",
            dns_config=k8s.V1PodDNSConfig(nameservers=list(sandbox.dns_nameservers)),
            containers=[
                k8s.V1Container(
                    name="build",
                    image=sandbox.image,
                    command=["/busybox/sh", "-c", _build_script(group, sandbox)],
                    security_context=k8s.V1SecurityContext(
                        # Root, but not the stock root -- see KANIKO_CAPABILITIES. Running as
                        # uid 0 is load-bearing: unpacking layers means owning files across many
                        # uids.
                        run_as_user=0,
                        allow_privilege_escalation=False,
                        # `baseline`, not a relaxation of it. Earlier designs needed
                        # seccomp: Unconfined here, and that is exactly what this namespace
                        # refuses -- see deploy/negative-control.sh.
                        seccomp_profile=k8s.V1SeccompProfile(type="RuntimeDefault"),
                        capabilities=k8s.V1Capabilities(drop=["ALL"], add=KANIKO_CAPABILITIES),
                    ),
                    # No env, no env_from, no secret volume. There is nothing here to read.
                    volume_mounts=[
                        k8s.V1VolumeMount(
                            name="work",
                            mount_path=SANDBOX_CONTEXT_MOUNT,
                            sub_path=f"{job_sub_path}/{group.context_sub_path}",
                            read_only=True,
                        ),
                        k8s.V1VolumeMount(
                            name="work",
                            mount_path=SANDBOX_OUTPUT_MOUNT,
                            sub_path=f"{job_sub_path}/{group.output_sub_path}",
                        ),
                    ],
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
    """Block until the sandbox reaches a terminal phase. Returns that phase."""
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
            logger.info("sandbox %s: %s", name, phase)
            if phase in ("Succeeded", "Failed"):
                watcher.stop()
                return phase
            if time.monotonic() > deadline:
                break
    finally:
        watcher.stop()
    return "Unknown"


def _results_from_log(log: str) -> dict[str, int]:
    """Per-image exit codes, parsed out of the sandbox's own output.

    Reading the log is how this step learns anything at all, because it mounts no volume. The
    marker lines are appended by the script it generated, so the format is not a guess.
    """
    results: dict[str, int] = {}
    for line in log.splitlines():
        if not line.startswith(RESULT_MARKER):
            continue
        parts = line.split()
        if len(parts) == 3 and parts[2].isdigit():
            results[parts[1]] = int(parts[2])
    return results


def main() -> int:
    config = SuperviseStepConfig.model_validate(read_step_config())
    workspace, job_id = job_identity()
    job_sub_path = job_storage_subpath(workspace, job_id)

    k8s_config.load_incluster_config()
    api = k8s.CoreV1Api()

    failures = 0
    for index, group in enumerate(config.groups):
        pod_name = f"nmp-sbx-{job_id}-g{index}"[:63]
        manifest = _pod_manifest(
            name=pod_name,
            group=group,
            sandbox=config.sandbox,
            pvc=config.sandbox.work_pvc,
            job_sub_path=job_sub_path,
        )

        logger.info("creating sandbox %s for %d image(s)", pod_name, len(group.images))
        api.create_namespaced_pod(namespace=config.sandbox.namespace, body=manifest)
        try:
            phase = _await_pod(api, name=pod_name, namespace=config.sandbox.namespace)
            log = api.read_namespaced_pod_log(name=pod_name, namespace=config.sandbox.namespace)
            logger.info("sandbox %s log:\n%s", pod_name, log)
            results = _results_from_log(log)

            for image in group.images:
                code = results.get(image.image)
                if code is None:
                    logger.error("image %s: no result recorded (sandbox phase %s)", image.image, phase)
                    failures += 1
                elif code != 0:
                    logger.error("image %s: kaniko exited %d", image.image, code)
                    failures += 1
                else:
                    logger.info("image %s: built", image.image)
        finally:
            # Deleted by the step that made it, always -- including when the watch timed out or
            # the log read failed. A leaked sandbox is a pod holding a context mount.
            api.delete_namespaced_pod(name=pod_name, namespace=config.sandbox.namespace, grace_period_seconds=0)

    if failures:
        # Non-zero, but every group was still attempted. A partially failed set still has images
        # worth publishing, which is why push and the reconciler both continue from here.
        logger.error("%d image(s) failed to build", failures)
        return 1
    return 0
