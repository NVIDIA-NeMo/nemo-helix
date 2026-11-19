# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in lifecycle checks using disposable kind and the rendered controller Role."""

import datetime
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import nhx.core.jobs.controllers.backends.kubernetes.configmaps as configmaps
import pytest
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.types import HelixJobStepWithContext
from nhx.core.jobs.app.providers import ContainerSpec
from nhx.core.jobs.controllers.backends.base import JobUpdate
from nhx.core.jobs.controllers.backends.kubernetes import CPUKubernetesJobBackend, KubernetesJobExecutionProfileConfig
from nhx.core.jobs.controllers.backends.kubernetes.common import common_labels_for_step, create_configmap, name_for_step
from nhx.core.jobs.controllers.backends.kubernetes.configmaps import StepConfigMapSweeper

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture
def cluster_api() -> Iterator[client.ApiClient]:
    kubeconfig = os.environ.get("NHX_CONFIGMAP_GC_KUBECONFIG")
    if not kubeconfig:
        pytest.skip("Set NHX_CONFIGMAP_GC_KUBECONFIG for the disposable kind cluster")
    contexts, _ = config.list_kube_config_contexts(config_file=kubeconfig)
    assert any(context["name"] == "kind-step-configmap-gc" for context in contexts)
    with config.new_client_from_config(config_file=kubeconfig, context="kind-step-configmap-gc") as api:
        yield api


@pytest.fixture
def namespace(cluster_api: client.ApiClient) -> Iterator[str]:
    namespace = f"configmap-gc-{uuid4().hex[:10]}"
    core = client.CoreV1Api(cluster_api)
    core.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace)))
    try:
        yield namespace
    finally:
        core.delete_namespace(namespace)


@pytest.fixture
def controller_api(cluster_api: client.ApiClient, namespace: str) -> Iterator[client.ApiClient]:
    rendered = subprocess.run(
        [
            "helm",
            "template",
            "configmap-gc",
            str(ROOT / "k8s/helm"),
            "--namespace",
            namespace,
            "--set",
            "rbac.volcanoEnabled=false",
            "--show-only",
            "templates/core/controller-role.yaml",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    role = next(doc for doc in yaml.safe_load_all(rendered) if doc and doc["kind"] == "Role")
    rbac, core = client.RbacAuthorizationV1Api(cluster_api), client.CoreV1Api(cluster_api)
    name = role["metadata"]["name"]
    core.create_namespaced_service_account(namespace, client.V1ServiceAccount(metadata=client.V1ObjectMeta(name=name)))
    rbac.create_namespaced_role(
        namespace,
        client.V1Role(
            metadata=client.V1ObjectMeta(name=name),
            rules=[
                client.V1PolicyRule(api_groups=rule["apiGroups"], resources=rule["resources"], verbs=rule["verbs"])
                for rule in role["rules"]
            ],
        ),
    )
    rbac.create_namespaced_role_binding(
        namespace,
        client.V1RoleBinding(
            metadata=client.V1ObjectMeta(name=name),
            role_ref=client.V1RoleRef(api_group="rbac.authorization.k8s.io", kind="Role", name=name),
            subjects=[client.RbacV1Subject(kind="ServiceAccount", name=name, namespace=namespace)],
        ),
    )
    with client.ApiClient(
        configuration=cluster_api.configuration,
        header_name="Impersonate-User",
        header_value=f"system:serviceaccount:{namespace}:{name}",
    ) as api:
        yield api


def wait_until(check: Callable[[], bool], timeout: int = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(1)
    pytest.fail("Timed out waiting for Kubernetes lifecycle transition")


def missing(read: Callable[[], client.V1Job | client.V1ConfigMap]) -> bool:
    try:
        read()
        return False
    except ApiException as exc:
        if exc.status != 404:
            raise
        return True


@pytest.fixture
def fast_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(configmaps, "MIN_CONFIGMAP_AGE", datetime.timedelta(0))
    monkeypatch.setattr(configmaps, "SWEEP_INTERVAL_SECONDS", 0)


def make_backend(
    api: client.ApiClient,
    namespace: str,
    *,
    immediate: bool = False,
    terminal: bool = True,
) -> CPUKubernetesJobBackend:
    backend = object.__new__(CPUKubernetesJobBackend)
    backend.namespace = namespace
    backend._profile_name = "default"
    backend._cancelled_workload_cursor = ""
    backend._jobs = MagicMock(spec=JobsClient)
    backend._jobs.get_attempt_step.return_value.data.return_value = None
    backend._execution_profile_config = KubernetesJobExecutionProfileConfig(
        namespace=namespace,
        ttl_seconds_after_finished=3600,
        cleanup_completed_jobs_immediately=immediate,
    )
    backend._core_v1, backend._batch_v1 = client.CoreV1Api(api), client.BatchV1Api(api)
    backend._configmap_sweeper = StepConfigMapSweeper(
        backend._core_v1,
        backend._batch_v1,
        namespace,
        lambda job, step, workspace, attempt: terminal,
    )
    return backend


def prepare_step(step: HelixJobStepWithContext) -> None:
    step.id = "platform-job-step-" + uuid4().hex[:12]
    assert step.step_spec is not None
    step.step_spec.environment = []


def schedule_step(
    backend: CPUKubernetesJobBackend,
    step: HelixJobStepWithContext,
    successful: bool,
    command: list[str] | None = None,
) -> JobUpdate:
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels=common_labels_for_step(step)),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[
                client.V1Container(
                    name="task",
                    image="busybox:1.37",
                    command=command or ["sh", "-c", "exit 0" if successful else "exit 1"],
                    volume_mounts=[client.V1VolumeMount(name="step-config", mount_path="/step-config")],
                )
            ],
            volumes=[
                client.V1Volume(
                    name="step-config",
                    config_map=client.V1ConfigMapVolumeSource(name=name_for_step(step)),
                )
            ],
        ),
    )
    with (
        patch(
            "nhx.core.jobs.controllers.backends.kubernetes.kubernetes_job.create_pod_template_spec",
            return_value=template,
        ),
        patch.object(backend, "get_secrets_environment_variable_for_injection", return_value=[]),
        patch.object(backend, "_ensure_initial_workload_delegation"),
    ):
        return backend.schedule_job(ContainerSpec(image="busybox:1.37"), step)


@pytest.mark.parametrize("successful,immediate", [(False, True), (True, False), (True, True)])
def test_sweep_collects_configmap_after_job_deletion(
    controller_api,
    namespace,
    test_step_pending,
    fast_sweeps,
    successful,
    immediate,
):
    step = test_step_pending
    prepare_step(step)
    backend = make_backend(controller_api, namespace, immediate=immediate)
    schedule_step(backend, step, successful)
    core, batch = backend._core_v1, backend._batch_v1
    name = name_for_step(step)
    assert (
        core.read_namespaced_config_map(name, namespace).metadata.owner_references[0].uid
        == batch.read_namespaced_job(name, namespace).metadata.uid
    )
    wait_until(
        lambda: any(
            condition.type == ("Complete" if successful else "Failed") and condition.status == "True"
            for condition in (batch.read_namespaced_job(name, namespace).status.conditions or [])
        )
    )
    with (
        patch.object(backend, "check_step_is_terminal", return_value=True),
        patch.object(backend, "_workload_delegation_target_for_job", return_value=None),
    ):
        backend.cleanup_steps()
    if successful and immediate:
        wait_until(lambda: missing(lambda: batch.read_namespaced_job(name, namespace)))
        assert missing(lambda: core.read_namespaced_config_map(name, namespace))
    else:
        # Even a terminal platform step must retain its ConfigMap while the Job exists.
        assert core.read_namespaced_config_map(name, namespace)
        batch.patch_namespaced_job(name, namespace, {"spec": {"ttlSecondsAfterFinished": 1}})
        wait_until(lambda: missing(lambda: batch.read_namespaced_job(name, namespace)))
        wait_until(lambda: not core.list_namespaced_pod(namespace, label_selector=f"job-name={name}").items)
        # No controller sweep runs after TTL: Kubernetes alone collects the ConfigMap.
        wait_until(lambda: missing(lambda: core.read_namespaced_config_map(name, namespace)))
    backend.cleanup_steps()


def test_young_configmap_survives_then_restart_collects_orphan(
    controller_api,
    namespace,
    test_step_pending,
    monkeypatch,
):
    prepare_step(test_step_pending)
    backend = make_backend(controller_api, namespace)
    core = backend._core_v1
    name = create_configmap(core, namespace, test_step_pending)
    backend.cleanup_steps()
    assert core.read_namespaced_config_map(name, namespace)
    monkeypatch.setattr(configmaps, "MIN_CONFIGMAP_AGE", datetime.timedelta(0))
    # A fresh backend needs no remembered cleanup intent or saved plan.
    restarted = make_backend(controller_api, namespace)
    restarted.cleanup_steps()
    assert missing(lambda: core.read_namespaced_config_map(name, namespace))


def test_active_platform_step_is_preserved(controller_api, namespace, test_step_pending, fast_sweeps):
    prepare_step(test_step_pending)
    backend = make_backend(controller_api, namespace, terminal=False)
    name = create_configmap(backend._core_v1, namespace, test_step_pending)
    backend.cleanup_steps()
    assert backend._core_v1.read_namespaced_config_map(name, namespace)
    make_backend(controller_api, namespace, terminal=True).cleanup_steps()
    assert missing(lambda: backend._core_v1.read_namespaced_config_map(name, namespace))


def test_live_pod_preserves_orphan_configuration(
    cluster_api,
    controller_api,
    namespace,
    test_step_pending,
    fast_sweeps,
):
    prepare_step(test_step_pending)
    backend = make_backend(controller_api, namespace)
    core = backend._core_v1
    admin = client.CoreV1Api(cluster_api)
    name = create_configmap(core, namespace, test_step_pending)
    admin.create_namespaced_pod(
        namespace,
        client.V1Pod(
            metadata=client.V1ObjectMeta(name="config-consumer"),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name="task",
                        image="busybox:1.37",
                        command=["sh", "-c", "sleep 300"],
                        volume_mounts=[client.V1VolumeMount(name="config", mount_path="/config")],
                    )
                ],
                volumes=[client.V1Volume(name="config", config_map=client.V1ConfigMapVolumeSource(name=name))],
            ),
        ),
    )
    backend.cleanup_steps()
    assert core.read_namespaced_config_map(name, namespace)
    admin.delete_namespaced_pod("config-consumer", namespace, grace_period_seconds=0)
    wait_until(lambda: not core.list_namespaced_pod(namespace).items)
    backend.cleanup_steps()
    assert missing(lambda: core.read_namespaced_config_map(name, namespace))


def test_api_failure_skips_sweep_then_recovers(controller_api, namespace, test_step_pending, fast_sweeps):
    prepare_step(test_step_pending)
    backend = make_backend(controller_api, namespace)
    core = backend._core_v1
    name = create_configmap(core, namespace, test_step_pending)
    with patch.object(backend._batch_v1, "read_namespaced_job", side_effect=ApiException(status=503)):
        backend.cleanup_steps()
    assert core.read_namespaced_config_map(name, namespace)
    backend.cleanup_steps()
    assert missing(lambda: core.read_namespaced_config_map(name, namespace))


def test_lost_delete_response_is_idempotent(controller_api, namespace, test_step_pending, fast_sweeps):
    prepare_step(test_step_pending)
    backend = make_backend(controller_api, namespace)
    core = backend._core_v1
    name = create_configmap(core, namespace, test_step_pending)
    delete = core.delete_namespaced_config_map

    def delete_then_timeout(**kwargs):
        delete(**kwargs)
        raise ApiException(status=503)

    with patch.object(core, "delete_namespaced_config_map", side_effect=delete_then_timeout):
        backend.cleanup_steps()
    assert missing(lambda: core.read_namespaced_config_map(name, namespace))
    restarted = make_backend(controller_api, namespace)
    restarted.cleanup_steps()
    assert not restarted._batch_v1.list_namespaced_job(namespace).items


def test_concurrent_configmap_update_rejects_stale_delete(
    cluster_api,
    controller_api,
    namespace,
    test_step_pending,
    fast_sweeps,
):
    prepare_step(test_step_pending)
    backend = make_backend(controller_api, namespace)
    core = backend._core_v1
    admin = client.CoreV1Api(cluster_api)
    name = create_configmap(core, namespace, test_step_pending)
    delete = core.delete_namespaced_config_map

    def update_before_delete(**kwargs):
        admin.patch_namespaced_config_map(name, namespace, {"metadata": {"annotations": {"updated": "true"}}})
        return delete(**kwargs)

    with patch.object(core, "delete_namespaced_config_map", side_effect=update_before_delete):
        backend.cleanup_steps()
    assert core.read_namespaced_config_map(name, namespace)
    backend.cleanup_steps()
    assert missing(lambda: core.read_namespaced_config_map(name, namespace))


def test_live_volcano_job_is_preserved(
    cluster_api,
    controller_api,
    namespace,
    test_step_pending,
    fast_sweeps,
):
    extensions = client.ApiextensionsV1Api(cluster_api)
    crd_name = "jobs.batch.volcano.sh"
    extensions.create_custom_resource_definition(
        client.V1CustomResourceDefinition(
            metadata=client.V1ObjectMeta(name=crd_name),
            spec=client.V1CustomResourceDefinitionSpec(
                group="batch.volcano.sh",
                scope="Namespaced",
                names=client.V1CustomResourceDefinitionNames(plural="jobs", singular="job", kind="Job"),
                versions=[
                    client.V1CustomResourceDefinitionVersion(
                        name="v1alpha1",
                        served=True,
                        storage=True,
                        schema=client.V1CustomResourceValidation(
                            open_apiv3_schema=client.V1JSONSchemaProps(
                                type="object",
                                x_kubernetes_preserve_unknown_fields=True,
                            )
                        ),
                    )
                ],
            ),
        )
    )
    try:
        wait_until(
            lambda: any(
                condition.type == "Established" and condition.status == "True"
                for condition in (extensions.read_custom_resource_definition(crd_name).status.conditions or [])
            )
        )
        prepare_step(test_step_pending)
        backend = make_backend(controller_api, namespace)
        name = create_configmap(backend._core_v1, namespace, test_step_pending)
        custom = client.CustomObjectsApi(cluster_api)
        custom.create_namespaced_custom_object(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace=namespace,
            plural="jobs",
            body={"apiVersion": "batch.volcano.sh/v1alpha1", "kind": "Job", "metadata": {"name": name}},
        )
        # Scheduling remains disabled; legacy cleanup only needs get access.
        with pytest.raises(ApiException) as forbidden:
            client.CustomObjectsApi(controller_api).list_namespaced_custom_object(
                group="batch.volcano.sh", version="v1alpha1", namespace=namespace, plural="jobs"
            )
        assert forbidden.value.status == 403
        backend.cleanup_steps()
        owner = backend._core_v1.read_namespaced_config_map(name, namespace).metadata.owner_references[0]
        assert owner.api_version == "batch.volcano.sh/v1alpha1"
        assert (
            owner.uid
            == custom.get_namespaced_custom_object(
                group="batch.volcano.sh", version="v1alpha1", namespace=namespace, plural="jobs", name=name
            )["metadata"]["uid"]
        )
        custom.delete_namespaced_custom_object(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace=namespace,
            plural="jobs",
            name=name,
        )
        wait_until(lambda: missing(lambda: backend._core_v1.read_namespaced_config_map(name, namespace)))
        prepare_step(test_step_pending)
        orphan = create_configmap(backend._core_v1, namespace, test_step_pending)
        backend.cleanup_steps()
        assert missing(lambda: backend._core_v1.read_namespaced_config_map(orphan, namespace))
    finally:
        extensions.delete_custom_resource_definition(crd_name)
        wait_until(
            lambda: (
                not any(
                    group.name == "batch.volcano.sh" for group in client.ApisApi(cluster_api).get_api_versions().groups
                )
            )
        )


def test_partial_cleanup_retries_only_remaining_configmaps(
    controller_api,
    namespace,
    test_step_pending,
    fast_sweeps,
):
    backend = make_backend(controller_api, namespace)
    core = backend._core_v1
    prepare_step(test_step_pending)
    first = create_configmap(core, namespace, test_step_pending)
    prepare_step(test_step_pending)
    second = create_configmap(core, namespace, test_step_pending)
    delete = core.delete_namespaced_config_map

    def fail_one_delete(**kwargs):
        if kwargs["name"] == first:
            raise ApiException(status=503)
        return delete(**kwargs)

    with patch.object(core, "delete_namespaced_config_map", side_effect=fail_one_delete):
        backend.cleanup_steps()
    assert core.read_namespaced_config_map(first, namespace)
    assert missing(lambda: core.read_namespaced_config_map(second, namespace))
    backend.cleanup_steps()
    assert missing(lambda: core.read_namespaced_config_map(first, namespace))


def test_sweep_repairs_interrupted_adoption(controller_api, namespace, test_step_pending, fast_sweeps):
    prepare_step(test_step_pending)
    backend = make_backend(controller_api, namespace, terminal=False)
    with patch("nhx.core.jobs.controllers.backends.kubernetes.kubernetes_job.adopt_step_configmap"):
        schedule_step(backend, test_step_pending, successful=False)
    name = name_for_step(test_step_pending)
    core, batch = backend._core_v1, backend._batch_v1
    assert not core.read_namespaced_config_map(name, namespace).metadata.owner_references
    # Legacy Jobs remain protected even without current controller labels.
    batch.patch_namespaced_job(name, namespace, {"metadata": {"labels": None}})
    backend.cleanup_steps()
    assert (
        core.read_namespaced_config_map(name, namespace).metadata.owner_references[0].uid
        == batch.read_namespaced_job(name, namespace).metadata.uid
    )
    batch.delete_namespaced_job(name, namespace, propagation_policy="Foreground")
    wait_until(lambda: missing(lambda: core.read_namespaced_config_map(name, namespace)))


@pytest.mark.parametrize("owned", [False, True])
def test_cancelled_step_cleanup_terminates_unfinished_job(
    controller_api, namespace, test_step_pending, fast_sweeps, owned
):
    step = test_step_pending
    prepare_step(step)
    backend = make_backend(controller_api, namespace)
    with patch.object(backend, "get_step_safe") as platform_lookup:
        platform_lookup.return_value = step
        if owned:
            schedule_step(backend, step, False, command=["sh", "-c", "sleep 300"])
        else:
            with patch("nhx.core.jobs.controllers.backends.kubernetes.kubernetes_job.adopt_step_configmap"):
                schedule_step(backend, step, False, command=["sh", "-c", "sleep 300"])
        name = name_for_step(step)
        assert backend._batch_v1.read_namespaced_job(name, namespace).status.completion_time is None
        backend.cleanup_steps()
        assert backend._batch_v1.read_namespaced_job(name, namespace)
        platform_lookup.return_value = step.model_copy(update={"status": HelixJobStatus.CANCELLED})
        with patch.object(backend, "_workload_delegation_target_for_job", return_value=None):
            backend.cleanup_steps()
            backend.cleanup_steps()
        wait_until(lambda: missing(lambda: backend._batch_v1.read_namespaced_job(name, namespace)))
        wait_until(lambda: not backend._core_v1.list_namespaced_pod(namespace).items)
        assert missing(lambda: backend._core_v1.read_namespaced_config_map(name, namespace))


@pytest.mark.parametrize("status", [HelixJobStatus.ERROR, HelixJobStatus.COMPLETED])
def test_non_cancelled_terminal_step_keeps_unfinished_job(
    controller_api, namespace, test_step_pending, fast_sweeps, status
):
    step = test_step_pending
    prepare_step(step)
    backend = make_backend(controller_api, namespace)
    with patch.object(backend, "get_step_safe") as platform_lookup:
        schedule_step(backend, step, False, command=["sh", "-c", "sleep 300"])
        platform_lookup.return_value = step.model_copy(update={"status": status})
        backend.cleanup_steps()
        name = name_for_step(step)
        assert backend._batch_v1.read_namespaced_job(name, namespace)
        assert backend._core_v1.read_namespaced_config_map(name, namespace)


def test_next_cleanup_pass_reaps_job_created_during_orphan_delete(
    controller_api, namespace, test_step_pending, fast_sweeps
):
    step = test_step_pending
    prepare_step(step)
    backend = make_backend(controller_api, namespace)
    with patch.object(backend, "get_step_safe") as platform_lookup:
        platform_lookup.return_value = step.model_copy(update={"status": HelixJobStatus.CANCELLED})
        core, batch = backend._core_v1, backend._batch_v1
        name = create_configmap(core, namespace, step)
        create, delete = batch.create_namespaced_job, core.delete_namespaced_config_map
        accepted = {}

        def delayed_job_response(**kwargs):
            # The Job becomes visible only after the sweep's final Job lookup.
            kwargs["body"].spec.template.spec.scheduling_gates = [client.V1PodSchedulingGate(name="test/hold")]

            def create_before_delete(**delete_kwargs):
                accepted["job"] = create(**kwargs)
                wait_until(lambda: bool(core.list_namespaced_pod(namespace).items))
                return delete(**delete_kwargs)

            with patch.object(core, "delete_namespaced_config_map", side_effect=create_before_delete):
                backend._configmap_sweeper.sweep()
            return accepted["job"]

        with patch.object(batch, "create_namespaced_job", side_effect=delayed_job_response):
            schedule_step(backend, step, False, command=["sh", "-c", "sleep 300"])
        assert batch.read_namespaced_job(name, namespace)
        assert missing(lambda: core.read_namespaced_config_map(name, namespace))
        with patch.object(backend, "_workload_delegation_target_for_job", return_value=None):
            backend.cleanup_steps()
        wait_until(lambda: missing(lambda: batch.read_namespaced_job(name, namespace)))
        wait_until(lambda: not core.list_namespaced_pod(namespace).items)
        assert missing(lambda: core.read_namespaced_config_map(name, namespace))
