#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover GPU nodes, run one NCCL worker pod per node, then delete workers."""

import json
import os
import sys
import time
import traceback

from kubernetes import client, config
from kubernetes.client import (
    V1Capabilities,
    V1ConfigMapVolumeSource,
    V1Container,
    V1ContainerPort,
    V1EnvVar,
    V1EnvVarSource,
    V1KeyToPath,
    V1LocalObjectReference,
    V1ObjectFieldSelector,
    V1ObjectMeta,
    V1OwnerReference,
    V1Pod,
    V1PodSpec,
    V1ResourceRequirements,
    V1SecurityContext,
    V1Volume,
    V1VolumeMount,
)
from kubernetes.client.rest import ApiException


def _truthy(val, default=False):
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "y")


def _env(name, default=None):
    v = os.environ.get(name, default)
    if v is None or v == "":
        if default is not None:
            return default
        raise RuntimeError(f"missing env {name}")
    return v


def _image_pull_secret_names():
    raw = os.environ.get("IMAGE_PULL_SECRETS", "[]")
    if not raw:
        return []
    try:
        names = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("IMAGE_PULL_SECRETS must be a JSON list of secret names") from exc
    if not isinstance(names, list):
        raise RuntimeError("IMAGE_PULL_SECRETS must be a JSON list of secret names")

    result = []
    for name in names:
        if not isinstance(name, str):
            raise RuntimeError("IMAGE_PULL_SECRETS must be a JSON list of secret names")
        if name:
            result.append(name)
    return result


def _node_ready(node):
    for c in node.status.conditions or []:
        if c.type == "Ready" and c.status == "True":
            return True
    return False


def _wait_pod_success(v1, ns, name, timeout_s):
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        pod = v1.read_namespaced_pod(name, ns)
        phase = pod.status.phase
        if phase == "Succeeded":
            return
        if phase == "Failed":
            reason = pod.status.reason or ""
            msg = f"pod {name} Failed: {reason}"
            raise RuntimeError(msg)
        time.sleep(3)
    _print_pod_status_snapshot(v1, ns, name)
    _dump_events(v1, ns, [name])
    raise TimeoutError(f"pod {name} did not succeed within {timeout_s}s")


def _wait_pod_running(v1, ns, name, timeout_s=300):
    """Wait until pod phase is Running (has IP; needed before reading leader podIP for followers)."""
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        pod = v1.read_namespaced_pod(name, ns)
        phase = pod.status.phase
        if phase == "Running":
            return pod
        if phase in ("Failed", "Succeeded"):
            reason = pod.status.reason or ""
            raise RuntimeError(f"pod {name} entered {phase} before Running: {reason}")
        time.sleep(1)
    raise TimeoutError(f"pod {name} did not reach Running within {timeout_s}s")


def _wait_pod_absent(v1, ns, name, timeout_s=120):
    """Wait until the pod is gone from the API (delete has finished)."""
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            v1.read_namespaced_pod(name, ns)
        except ApiException as e:
            if e.status == 404:
                return
            raise
        time.sleep(1)
    raise TimeoutError(f"pod {name} still present after {timeout_s}s")


def _dump_logs(v1, ns, names):
    for name in names:
        try:
            logs = v1.read_namespaced_pod_log(name, ns, container="nccl-test")
            print(f"---- logs {name} ----\n{logs}")
        except ApiException as e:
            print(f"(no logs for {name}) {e}")


def _dump_events(v1, ns, names):
    """Print Kubernetes Events for worker pods (scheduling / mount / OOM hints on failure)."""
    if not names:
        return
    print("---- Kubernetes events (worker pods) ----")
    for name in names:
        try:
            evs = v1.list_namespaced_event(
                ns,
                field_selector=f"involvedObject.name={name},involvedObject.kind=Pod",
            )
        except ApiException as e:
            print(f"(could not list events for {name}) {e}")
            continue
        items = evs.items or []
        if not items:
            print(f"(no events for pod {name})")
            continue
        for ev in sorted(
            items,
            key=lambda x: str(x.last_timestamp or x.first_timestamp or ""),
        ):
            typ = ev.type or ""
            reason = ev.reason or ""
            msg = (ev.message or "").replace("\n", " ")
            cnt = ev.count or 0
            print(f"  {name}: [{typ}] {reason} (x{cnt}) {msg}")


def _print_pod_status_snapshot(v1, ns, name):
    """Print latest Pod status (phase, conditions, container state) for debugging timeouts."""
    try:
        pod = v1.read_namespaced_pod(name, ns)
    except ApiException as e:
        print(f"(could not read pod {name} for status: {e})")
        return
    st = pod.status
    print(f"---- pod status snapshot: {name} ----")
    print(f"  phase: {st.phase}")
    if st.message:
        print(f"  message: {st.message}")
    if st.reason:
        print(f"  reason: {st.reason}")
    for c in st.conditions or []:
        print(
            f"  condition {c.type}: status={c.status} "
            f"reason={c.reason or ''} message={(c.message or '').replace(chr(10), ' ')}"
        )
    for cs in st.container_statuses or []:
        parts = [f"ready={cs.ready}", f"restart_count={cs.restart_count}"]
        state = cs.state
        if state is None:
            parts.append("state=unknown")
        elif state.waiting:
            parts.append(f"waiting reason={state.waiting.reason} msg={state.waiting.message or ''}")
        elif state.terminated:
            t = state.terminated
            parts.append(f"terminated exit={t.exit_code} reason={t.reason} msg={t.message or ''}")
        elif state.running:
            parts.append("running")
        print(f"  container {cs.name}: {'; '.join(parts)}")


def _delete_workers(v1, ns, names):
    for name in names:
        try:
            v1.delete_namespaced_pod(name, ns, grace_period_seconds=0)
            print(f"Delete issued for pod {name}")
        except ApiException as ae:
            if ae.status != 404:
                print(
                    f"warn: could not delete {name}: {ae}",
                    file=sys.stderr,
                )
                continue
        _wait_pod_absent(v1, ns, name)


def _log_node_interconnect_hints(nodes, rdma_resource):
    """Summarize labels/capacity for multinode networking (cloud-agnostic hints)."""
    for n in nodes:
        lb = n.metadata.labels or {}
        alloc = n.status.allocatable or {}
        primary = f"alloc[{rdma_resource}]={alloc.get(rdma_resource, '')} " if rdma_resource else ""
        print(
            f"Node {n.metadata.name}: mellanox={lb.get('nvidia.com/mellanox.present', '')} "
            f"rdma_capable={lb.get('feature.node.kubernetes.io/rdma.capable', '')} "
            f"{primary}"
            f"efa={alloc.get('vpc.amazonaws.com/efa', '')} "
            f"mlnxnics={alloc.get('nvidia.com/mlnxnics', '')}"
        )


def _print_leader_success_log(v1, ns, worker_pod_name):
    """Print full nccl-test container log from the leader worker (rank-0 node) after success."""
    try:
        logs = v1.read_namespaced_pod_log(worker_pod_name, ns, container="nccl-test")
    except ApiException as e:
        print(f"(could not load leader worker log for {worker_pod_name}: {e})")
        return
    print(f"================ {worker_pod_name} leader (nccl-test) ================================")
    print(logs.rstrip())
    print("================================================================================")


def _resource_qty_str(q):
    if q is None:
        return ""
    return str(q).strip()


def _assert_pod_interconnect_from_kyverno(pod, resource_name, want_req, want_lim):
    """Assert pod spec has interconnect resource requests/limits (Kyverno mutate policy)."""
    containers = pod.spec.containers or []
    if not containers:
        raise RuntimeError("pod has no containers")
    c0 = containers[0]
    res = c0.resources
    if not res:
        raise RuntimeError("pod container has no resources")
    req = res.requests or {}
    lim = res.limits or {}
    gr = req.get(resource_name)
    gl = lim.get(resource_name)
    if gr is None:
        raise RuntimeError(
            f"expected Kyverno to inject resources.requests[{resource_name!r}]={want_req!r}; got requests={dict(req)}"
        )
    if gl is None:
        raise RuntimeError(
            f"expected Kyverno to inject resources.limits[{resource_name!r}]={want_lim!r}; got limits={dict(lim)}"
        )
    if _resource_qty_str(gr) != _resource_qty_str(want_req):
        raise RuntimeError(f"Kyverno requests[{resource_name}]: got {gr!r}, want {want_req!r}")
    if _resource_qty_str(gl) != _resource_qty_str(want_lim):
        raise RuntimeError(f"Kyverno limits[{resource_name}]: got {gl!r}, want {want_lim!r}")


def _wait_assert_kyverno_interconnect(v1, namespace, pod_name, resource_name, want_req, want_lim, timeout_s=30):
    """Poll until pod spec shows Kyverno-injected interconnect resources (admission is usually sync)."""
    deadline = time.time() + float(timeout_s)
    last_err = None
    while time.time() < deadline:
        pod = v1.read_namespaced_pod(pod_name, namespace)
        try:
            _assert_pod_interconnect_from_kyverno(pod, resource_name, want_req, want_lim)
            print(f"Kyverno assertion ok: pod {pod_name} has {resource_name} requests={want_req} limits={want_lim}")
            return
        except RuntimeError as e:
            last_err = e
            time.sleep(0.5)
    raise last_err or RuntimeError("Kyverno interconnect assertion failed")


def _ensure_rdma_allocatable(nodes, resource_name):
    missing = []
    for n in nodes:
        alloc = n.status.allocatable or {}
        if not alloc.get(resource_name):
            missing.append(n.metadata.name)
    if missing:
        raise RuntimeError(
            f"requireRdmaAllocatable: nodes missing {resource_name} in allocatable: {missing}. "
            "See verify-multinode-setup.sh / network-operator docs."
        )


def config_test(iteration: int, print_logs: bool):
    namespace = _env("NAMESPACE")
    fullname = _env("TEST_FULLNAME")
    label_key = _env("GPU_NODE_LABEL_KEY")
    label_value = _env("GPU_NODE_LABEL_VALUE")
    worker_image = _env("WORKER_IMAGE")
    scripts_cm = _env("SCRIPTS_CONFIGMAP_NAME")
    master_port = _env("MASTER_PORT", "29500")
    timeout_s = int(_env("WAIT_TIMEOUT_SECONDS", "900"))

    image_pull_secret_names = _image_pull_secret_names()
    release_name = os.environ.get("RELEASE_NAME", "")

    gpu_req = _env("WORKER_GPU_REQUEST", "1")
    gpu_resource_key = os.environ.get("WORKER_GPU_RESOURCE_KEY", "nvidia.com/gpu")
    try:
        gpu_n = max(1, int(float(gpu_req)))
    except (ValueError, TypeError):
        gpu_n = 1

    cpu_req = _env("WORKER_CPU_REQUEST", "4")
    cpu_lim = _env("WORKER_CPU_LIMIT", "8")
    mem_req = _env("WORKER_MEMORY_REQUEST", "8Gi")
    mem_lim = _env("WORKER_MEMORY_LIMIT", "16Gi")

    config.load_incluster_config()
    v1 = client.CoreV1Api()

    pod_name = os.environ.get("POD_NAME") or os.environ.get("HOSTNAME", "")
    if not pod_name:
        print("ERROR: POD_NAME or HOSTNAME must be set", file=sys.stderr)
        return 1
    hook = os.environ.get("HELM_HOOK", "test")
    hook_delete = os.environ.get(
        "HELM_HOOK_DELETE_POLICY",
        "before-hook-creation,hook-succeeded,hook-failed",
    )

    try:
        orch_pod = v1.read_namespaced_pod(pod_name, namespace)
    except ApiException as e:
        print(
            f"ERROR: cannot read orchestrator pod {pod_name}: {e}",
            file=sys.stderr,
        )
        return 1

    job_owner = None
    for ref in orch_pod.metadata.owner_references or []:
        if ref.kind == "Job":
            job_owner = ref
            break

    owner_refs = None
    if job_owner is not None:
        owner_refs = [
            V1OwnerReference(
                api_version=job_owner.api_version,
                kind=job_owner.kind,
                name=job_owner.name,
                uid=job_owner.uid,
                controller=False,
                block_owner_deletion=False,
            )
        ]
    else:
        print(
            "WARN: orchestrator pod has no Job ownerReference; workers will not be tied to hook Job GC",
            file=sys.stderr,
        )

    selector = label_key + "=" + label_value
    all_nodes = v1.list_node(label_selector=selector).items
    nodes = sorted(
        [n for n in all_nodes if _node_ready(n)],
        key=lambda n: n.metadata.name,
    )
    if not nodes:
        print(
            f"ERROR: No Ready nodes match GPU label selector {selector}",
            file=sys.stderr,
        )
        return 1

    world_size = len(nodes)
    global_world = world_size * gpu_n
    print(
        f"Discovered {world_size} GPU node(s): {[n.metadata.name for n in nodes]}; "
        f"{gpu_n} GPU(s) per node → {global_world} global NCCL ranks"
    )

    rdma_alloc_res = os.environ.get("RDMA_ALLOCATABLE_RESOURCE", "").strip()

    _log_node_interconnect_hints(nodes, rdma_alloc_res)

    if _truthy(os.environ.get("REQUIRE_RDMA_ALLOCATABLE")):
        if not rdma_alloc_res:
            print(
                "ERROR: REQUIRE_RDMA_ALLOCATABLE is true but RDMA_ALLOCATABLE_RESOURCE is empty "
                "(template bug or unsupported cloud for allocatable checks).",
                file=sys.stderr,
            )
            return 1
        _ensure_rdma_allocatable(nodes, rdma_alloc_res)
        print(f"All {world_size} GPU nodes advertise {rdma_alloc_res} in allocatable.")

    min_bw = os.environ.get("NCCL_TEST_MIN_BANDWIDTH_MBPS", "0")
    exp_ib = _truthy(os.environ.get("NCCL_TEST_EXPECT_IB_TRANSPORT"))
    strict_ib = _truthy(os.environ.get("NCCL_TEST_STRICT_IB_PORT_ACTIVE"))
    print(
        f"Validation flags: minBandwidth1024MB={min_bw} MB/s expectIbTransport={exp_ib} strictIbPortActive={strict_ib}"
    )

    worker_net = _truthy(os.environ.get("WORKER_INTERCONNECT_RESOURCE_ENABLED"))
    wn_name = os.environ.get("WORKER_INTERCONNECT_RESOURCE_NAME", "").strip()
    wn_req = os.environ.get("WORKER_INTERCONNECT_RESOURCE_REQUEST", "0")
    wn_lim = os.environ.get("WORKER_INTERCONNECT_RESOURCE_LIMIT", wn_req)
    # When true, we annotate disable-rdma-injection and must set interconnect resources ourselves.
    injection_disabled = not _truthy(os.environ.get("NCCL_TEST_ALLOW_PLATFORM_INJECTION", "true"))

    worker_ann = {
        "helm.sh/hook": hook,
        "helm.sh/hook-delete-policy": hook_delete,
    }
    if _truthy(os.environ.get("NCCL_TEST_KYVERNO_ENABLE_MULTI_NODE", "true")):
        worker_ann["nhx.nvidia.com/enable-multi-node-networking"] = "true"
        worker_ann["nhx.nvidia.com/num-nodes"] = str(world_size)
    if not _truthy(os.environ.get("NCCL_TEST_ALLOW_PLATFORM_INJECTION", "true")):
        worker_ann["disable-rdma-injection"] = "true"

    created_names = []

    def worker_pod_name(rank):
        return f"{fullname}-w-{rank}"

    # Followers use rank-0 pod IP for LEADER_ADDR (PyTorch MASTER_ADDR). Headless DNS
    # can resolve on the leader pod but fail on other nodes (NodeLocal DNS / split views).
    leader_ip = None

    image_pull_secrets = [V1LocalObjectReference(name=name) for name in image_pull_secret_names]

    def _make_worker_pod(rank, hostname, leader_addr_from_field_ref):
        name = worker_pod_name(rank)
        meta_kwargs = dict(
            name=name,
            labels={
                "app.kubernetes.io/instance": release_name,
                "app.kubernetes.io/name": "nemo-helix",
                "nccl-helm-test-worker": "true",
                "iteration": str(iteration),
            },
            annotations=worker_ann,
        )
        if owner_refs is not None:
            meta_kwargs["owner_references"] = owner_refs
        meta = V1ObjectMeta(**meta_kwargs)
        if leader_addr_from_field_ref:
            leader_env = V1EnvVar(
                name="LEADER_ADDR",
                value_from=V1EnvVarSource(field_ref=V1ObjectFieldSelector(field_path="status.podIP")),
            )
        else:
            if not leader_ip:
                raise RuntimeError("leader_ip unset for follower worker")
            leader_env = V1EnvVar(name="LEADER_ADDR", value=leader_ip)
        env_vars = [
            V1EnvVar(name="PYTHONUNBUFFERED", value="1"),
            V1EnvVar(name="NVIDIA_VISIBLE_DEVICES", value="all"),
            V1EnvVar(
                name="NVIDIA_DRIVER_CAPABILITIES",
                value="compute,utility",
            ),
            V1EnvVar(name="NCCL_DEBUG", value="INFO"),
            leader_env,
            V1EnvVar(name="MASTER_PORT", value=master_port),
            V1EnvVar(name="NODE_RANK", value=str(rank)),
            V1EnvVar(name="NUM_NODES", value=str(world_size)),
            V1EnvVar(name="NPROC_PER_NODE", value=str(gpu_n)),
            V1EnvVar(
                name="NCCL_TEST_MIN_BANDWIDTH_MBPS",
                value=os.environ.get("NCCL_TEST_MIN_BANDWIDTH_MBPS", "0"),
            ),
            V1EnvVar(
                name="NCCL_TEST_EXPECT_IB_TRANSPORT",
                value=("true" if _truthy(os.environ.get("NCCL_TEST_EXPECT_IB_TRANSPORT")) else "false"),
            ),
            V1EnvVar(
                name="NCCL_TEST_STRICT_IB_PORT_ACTIVE",
                value=("true" if _truthy(os.environ.get("NCCL_TEST_STRICT_IB_PORT_ACTIVE")) else "false"),
            ),
        ]
        req = {
            gpu_resource_key: gpu_req,
            "cpu": cpu_req,
            "memory": mem_req,
        }
        lim = {
            gpu_resource_key: gpu_req,
            "cpu": cpu_lim,
            "memory": mem_lim,
        }
        if worker_net and wn_name and injection_disabled:
            req[wn_name] = wn_req
            lim[wn_name] = wn_lim

        container = V1Container(
            name="nccl-test",
            image=worker_image,
            image_pull_policy="IfNotPresent",
            command=["/bin/bash", "/scripts/entrypoint.sh"],
            env=env_vars,
            resources=V1ResourceRequirements(
                requests=req,
                limits=lim,
            ),
            volume_mounts=[
                V1VolumeMount(name="scripts", mount_path="/scripts"),
                V1VolumeMount(
                    name="platform-config",
                    mount_path="/platform-config",
                ),
            ],
            ports=[V1ContainerPort(container_port=int(master_port))],
            security_context=V1SecurityContext(
                run_as_user=0,
                capabilities=V1Capabilities(add=["IPC_LOCK", "SYS_NICE"]),
            ),
        )
        return V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=meta,
            spec=V1PodSpec(
                restart_policy="Never",
                node_selector={
                    "kubernetes.io/hostname": hostname,
                    label_key: label_value,
                },
                image_pull_secrets=image_pull_secrets or None,
                containers=[container],
                volumes=[
                    V1Volume(
                        name="scripts",
                        config_map=V1ConfigMapVolumeSource(
                            name=scripts_cm,
                            default_mode=493,
                        ),
                    ),
                    V1Volume(
                        name="platform-config",
                        config_map=V1ConfigMapVolumeSource(
                            name=scripts_cm,
                            default_mode=493,
                            items=[
                                V1KeyToPath(
                                    key="nccl-env.sh",
                                    path="nccl-env.sh",
                                    mode=493,
                                )
                            ],
                        ),
                    ),
                ],
            ),
        )

    try:
        n0 = nodes[0].metadata.name
        name0 = worker_pod_name(0)
        pod0 = _make_worker_pod(0, n0, leader_addr_from_field_ref=True)
        print(f"Creating pod {name0} on node {n0}")
        v1.create_namespaced_pod(namespace, body=pod0)
        if worker_net and wn_name and not injection_disabled:
            _wait_assert_kyverno_interconnect(v1, namespace, name0, wn_name, wn_req, wn_lim)
        p0 = _wait_pod_running(v1, namespace, name0, timeout_s=300)
        leader_ip = p0.status.pod_ip
        if not leader_ip:
            raise RuntimeError(f"pod {name0} has no status.pod_ip")
        print(f"Leader pod IP for MASTER_ADDR (followers): {leader_ip}")
        created_names.append(name0)

        for rank in range(1, world_size):
            hostname = nodes[rank].metadata.name
            name = worker_pod_name(rank)
            pod = _make_worker_pod(rank, hostname, leader_addr_from_field_ref=False)
            print(f"Creating pod {name} on node {hostname}")
            v1.create_namespaced_pod(namespace, body=pod)
            if worker_net and wn_name and not injection_disabled:
                _wait_assert_kyverno_interconnect(v1, namespace, name, wn_name, wn_req, wn_lim)
            created_names.append(name)

        for name in created_names:
            print(f"Waiting for pod {name} ...")
            _wait_pod_success(v1, namespace, name, timeout_s)
            print(f"Pod {name} succeeded")

        if created_names:
            if print_logs:
                _print_leader_success_log(v1, namespace, created_names[0])

        print(
            f"NCCL validation passed: {world_size} node(s) × {gpu_n} GPU(s)/node = "
            f"{global_world} processes in all_reduce."
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        _dump_logs(v1, namespace, created_names)
        _dump_events(v1, namespace, created_names)
        return 1
    finally:
        _delete_workers(v1, namespace, created_names)


def main():
    iteration = int(_env("NCCL_TEST_ITERATIONS", "10"))
    for i in range(iteration):
        print(f"Running NCCL test iteration {i + 1} of {iteration}")
        res = config_test(i, print_logs=i == iteration - 1)
        if res != 0:
            return res
    return 0


if __name__ == "__main__":
    sys.exit(main())
