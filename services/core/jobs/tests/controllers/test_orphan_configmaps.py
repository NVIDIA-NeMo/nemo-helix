# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client
from kubernetes.client.rest import ApiException
from nhx.common.jobs.constants import NEMO_JOB_STEP_CONFIG_FILE_NAME
from nhx.core.jobs.app.constants import JOB_ID_LABEL, JOB_MANAGED_BY_LABEL, JOB_TYPE_LABEL
from nhx.core.jobs.controllers.backends.kubernetes.common import common_labels_for_step, name_for_step
from nhx.core.jobs.controllers.backends.kubernetes.configmaps import (
    IDENTITY_LABELS,
    REQUEST_TIMEOUT,
    SWEEP_MAX_PAGES,
    SWEEP_PAGE_SIZE,
    StepConfigMapSweeper,
    adopt_step_configmap,
)


@pytest.fixture
def cm(test_step_pending) -> client.V1ConfigMap:
    return client.V1ConfigMap(
        metadata=client.V1ObjectMeta(
            name=name_for_step(test_step_pending),
            namespace="test-namespace",
            uid="cm-uid",
            resource_version="17",
            creation_timestamp=datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
            labels=common_labels_for_step(test_step_pending),
        ),
        data={NEMO_JOB_STEP_CONFIG_FILE_NAME: "{}"},
    )


@pytest.fixture
def apis(cm):
    core, batch, custom = (
        MagicMock(spec=api)
        for api in (
            client.CoreV1Api,
            client.BatchV1Api,
            client.CustomObjectsApi,
        )
    )
    core.api_client = MagicMock(spec=client.ApiClient)
    core.list_namespaced_config_map.return_value = client.V1ConfigMapList(items=[cm])
    core.read_namespaced_config_map.return_value = cm
    core.list_namespaced_pod.return_value = client.V1PodList(items=[])
    batch.read_namespaced_job.side_effect = ApiException(status=404)
    custom.get_namespaced_custom_object.side_effect = ApiException(status=404)
    return core, batch, custom


@pytest.fixture
def sweeper(apis):
    core, batch, custom = apis
    terminal = MagicMock(return_value=True)
    with patch("nhx.core.jobs.controllers.backends.kubernetes.configmaps.client.CustomObjectsApi", return_value=custom):
        return StepConfigMapSweeper(core, batch, "test-namespace", terminal), terminal


def test_old_orphan_is_deleted_with_preconditions(sweeper, apis, cm):
    collector, terminal = sweeper
    core, batch, custom = apis
    collector.sweep()
    terminal.assert_called_once_with("test-job-id", "test-step", "default", "test-job-attempt-id")
    kwargs = core.delete_namespaced_config_map.call_args.kwargs
    assert kwargs["namespace"] == "test-namespace"
    assert kwargs["body"].preconditions.uid == cm.metadata.uid
    assert kwargs["body"].preconditions.resource_version == cm.metadata.resource_version
    assert batch.read_namespaced_job.call_count == 1
    assert custom.get_namespaced_custom_object.call_count == 1


@pytest.mark.parametrize(
    "state", ["young", "owned", "deleting", "missing-uid", "missing-version", "no-data", "unmanaged", "wrong-type"]
)
def test_ineligible_configmaps_are_preserved(sweeper, apis, cm, state):
    if state == "young":
        cm.metadata.creation_timestamp = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=4)
    elif state == "owned":
        cm.metadata.owner_references = [
            client.V1OwnerReference(api_version="batch/v1", kind="Job", name="job", uid="uid")
        ]
    elif state == "deleting":
        cm.metadata.deletion_timestamp = datetime.datetime.now(datetime.UTC)
    elif state == "missing-uid":
        cm.metadata.uid = None
    elif state == "missing-version":
        cm.metadata.resource_version = None
    elif state == "no-data":
        cm.data = {}
    elif state == "unmanaged":
        cm.metadata.labels[JOB_MANAGED_BY_LABEL] = "other-controller"
    else:
        cm.metadata.labels[JOB_TYPE_LABEL] = "storage-cleanup"
    sweeper[0].sweep()
    apis[0].delete_namespaced_config_map.assert_not_called()
    sweeper[1].assert_not_called()


@pytest.mark.parametrize("label", IDENTITY_LABELS)
def test_incomplete_identity_is_preserved(sweeper, apis, cm, label):
    del cm.metadata.labels[label]
    sweeper[0].sweep()
    apis[0].delete_namespaced_config_map.assert_not_called()


def test_live_kubernetes_job_preserves_configmap(sweeper, apis, cm):
    apis[1].read_namespaced_job.side_effect = None
    apis[1].read_namespaced_job.return_value = client.V1Job(
        metadata=client.V1ObjectMeta(name=cm.metadata.name, uid="job-uid")
    )
    sweeper[0].sweep()
    apis[0].delete_namespaced_config_map.assert_not_called()
    sweeper[1].assert_called_once()


def test_live_volcano_job_preserves_configmap(sweeper, apis, cm):
    apis[2].get_namespaced_custom_object.side_effect = None
    apis[2].get_namespaced_custom_object.return_value = {"metadata": {"name": cm.metadata.name, "uid": "vc-uid"}}
    sweeper[0].sweep()
    apis[0].delete_namespaced_config_map.assert_not_called()
    sweeper[1].assert_called_once()


def test_active_platform_step_is_preserved(sweeper, apis):
    sweeper[1].return_value = False
    sweeper[0].sweep()
    apis[0].delete_namespaced_config_map.assert_not_called()


@pytest.mark.parametrize("reference", ["volume", "projected", "env", "env-from", "init", "ephemeral"])
def test_pod_references_are_preserved(sweeper, apis, cm, reference):
    container = client.V1Container(name="task", image="busybox")
    spec = client.V1PodSpec(containers=[container])
    env = client.V1EnvVar(
        name="CONFIG",
        value_from=client.V1EnvVarSource(
            config_map_key_ref=client.V1ConfigMapKeySelector(name=cm.metadata.name, key="key"),
        ),
    )
    if reference == "volume":
        spec.volumes = [
            client.V1Volume(name="config", config_map=client.V1ConfigMapVolumeSource(name=cm.metadata.name))
        ]
    elif reference == "projected":
        spec.volumes = [
            client.V1Volume(
                name="config",
                projected=client.V1ProjectedVolumeSource(
                    sources=[
                        client.V1VolumeProjection(config_map=client.V1ConfigMapProjection(name=cm.metadata.name)),
                    ]
                ),
            )
        ]
    elif reference == "env":
        container.env = [env]
    elif reference == "env-from":
        container.env_from = [client.V1EnvFromSource(config_map_ref=client.V1ConfigMapEnvSource(name=cm.metadata.name))]
    elif reference == "init":
        spec.init_containers = [client.V1Container(name="init", image="busybox", env=[env])]
    else:
        spec.ephemeral_containers = [client.V1EphemeralContainer(name="debug", image="busybox", env=[env])]
    apis[0].list_namespaced_pod.return_value = client.V1PodList(items=[client.V1Pod(spec=spec)])
    sweeper[0].sweep()
    apis[0].delete_namespaced_config_map.assert_not_called()


@pytest.mark.parametrize("field", ["uid", "resource-version"])
def test_changed_object_is_preserved(sweeper, apis, cm, field):
    changed = deepcopy(cm)
    if field == "uid":
        changed.metadata.uid = "replacement"
    else:
        changed.metadata.resource_version = "18"
    apis[0].read_namespaced_config_map.return_value = changed
    sweeper[0].sweep()
    apis[0].delete_namespaced_config_map.assert_not_called()


@pytest.mark.parametrize("status", [403, 500])
@pytest.mark.parametrize("operation", ["configmaps", "jobs", "volcano", "pods", "recheck", "platform"])
def test_lookup_errors_never_establish_absence(sweeper, apis, status, operation):
    core, batch, custom = apis
    error = ApiException(status=status)
    if operation == "configmaps":
        core.list_namespaced_config_map.side_effect = error
    elif operation == "jobs":
        batch.read_namespaced_job.side_effect = error
    elif operation == "volcano":
        custom.get_namespaced_custom_object.side_effect = error
    elif operation == "pods":
        core.list_namespaced_pod.side_effect = error
    elif operation == "recheck":
        core.read_namespaced_config_map.side_effect = error
    else:
        sweeper[1].side_effect = RuntimeError("platform lookup unavailable")
    sweeper[0].sweep()
    core.delete_namespaced_config_map.assert_not_called()


@pytest.mark.parametrize("volcano", [False, True])
def test_job_created_during_platform_lookup_is_preserved(sweeper, apis, volcano):
    def platform_lookup(*args):
        if volcano:
            apis[2].get_namespaced_custom_object.side_effect = None
            apis[2].get_namespaced_custom_object.return_value = {"metadata": {"uid": "new-uid"}}
        else:
            apis[1].read_namespaced_job.side_effect = None
            apis[1].read_namespaced_job.return_value = client.V1Job(metadata=client.V1ObjectMeta(uid="new-uid"))
        return True

    sweeper[1].side_effect = platform_lookup
    sweeper[0].sweep()
    apis[0].delete_namespaced_config_map.assert_not_called()


def test_sweep_is_throttled(sweeper, apis):
    with patch("nhx.core.jobs.controllers.backends.kubernetes.configmaps.time.monotonic", return_value=100.0) as clock:
        sweeper[0].sweep()
        clock.return_value = 110.0
        sweeper[0].sweep()
        assert apis[0].list_namespaced_config_map.call_count == 1
        clock.return_value = 161.0
        sweeper[0].sweep()
    assert apis[0].list_namespaced_config_map.call_count == 2


def test_failed_deletion_is_retried_on_next_sweep(sweeper, apis):
    apis[0].delete_namespaced_config_map.side_effect = [ApiException(status=503), None]
    sweeper[0].sweep()
    sweeper[0]._next_sweep = 0
    sweeper[0].sweep()
    assert apis[0].delete_namespaced_config_map.call_count == 2


def test_lost_delete_response_needs_no_cleanup_state(sweeper, apis):
    core = apis[0]

    def delete_then_timeout(**kwargs):
        core.list_namespaced_config_map.return_value = client.V1ConfigMapList(items=[])
        raise ApiException(status=503)

    core.delete_namespaced_config_map.side_effect = delete_then_timeout
    sweeper[0].sweep()
    sweeper[0]._next_sweep = 0
    sweeper[0].sweep()
    core.delete_namespaced_config_map.assert_called_once()


def test_partial_failure_does_not_block_other_orphans(sweeper, apis, cm):
    other = deepcopy(cm)
    other.metadata.name = "another-step"
    other.metadata.labels[JOB_ID_LABEL] = "another-job"
    core = apis[0]
    core.list_namespaced_config_map.return_value = client.V1ConfigMapList(items=[cm, other])
    core.read_namespaced_config_map.side_effect = [cm, other]
    core.delete_namespaced_config_map.side_effect = [ApiException(status=503), None]
    sweeper[0].sweep()
    assert [call.kwargs["name"] for call in core.delete_namespaced_config_map.call_args_list] == [
        cm.metadata.name,
        other.metadata.name,
    ]


def test_backlog_is_paged_with_one_pod_scan_per_batch(sweeper, apis, cm):
    core = apis[0]
    candidates = []
    for index in range(SWEEP_PAGE_SIZE * 2):
        other = deepcopy(cm)
        other.metadata.name = f"orphan-{index}"
        candidates.append(other)
    core.list_namespaced_config_map.side_effect = [
        client.V1ConfigMapList(items=candidates[:SWEEP_PAGE_SIZE], metadata=client.V1ListMeta(_continue="next-page")),
        client.V1ConfigMapList(items=candidates[SWEEP_PAGE_SIZE:], metadata=client.V1ListMeta()),
    ]
    core.read_namespaced_config_map.side_effect = candidates
    progress = MagicMock()
    sweeper[0].sweep(progress)
    assert core.delete_namespaced_config_map.call_count == SWEEP_PAGE_SIZE
    assert core.list_namespaced_pod.call_count == 1
    assert progress.call_count >= SWEEP_PAGE_SIZE
    sweeper[0]._next_sweep = 0
    sweeper[0].sweep(progress)
    assert core.delete_namespaced_config_map.call_count == 2 * SWEEP_PAGE_SIZE
    assert core.list_namespaced_pod.call_count == 2
    assert core.list_namespaced_config_map.call_args.kwargs["_continue"] == "next-page"
    assert core.list_namespaced_config_map.call_args.kwargs["limit"] == SWEEP_PAGE_SIZE
    # These two batches each find candidates on their first page.
    assert len(core.method_calls) + len(apis[1].method_calls) + len(apis[2].method_calls) <= 2 * (
        2 + 4 * SWEEP_PAGE_SIZE
    )
    for api in apis[:3]:
        for call in api.method_calls:
            assert call.kwargs["_request_timeout"] == REQUEST_TIMEOUT


def test_slow_batch_keeps_unfinished_candidates_and_reports_progress(sweeper, apis, cm):
    other = deepcopy(cm)
    other.metadata.name = "second-orphan"
    core = apis[0]
    core.list_namespaced_config_map.return_value = client.V1ConfigMapList(items=[cm, other])
    core.read_namespaced_config_map.side_effect = [cm, other]
    progress = MagicMock()
    with patch("nhx.core.jobs.controllers.backends.kubernetes.configmaps.time.monotonic", return_value=100.0) as clock:

        def slow_delete(**kwargs):
            clock.return_value += 3

        core.delete_namespaced_config_map.side_effect = slow_delete
        sweeper[0].sweep(progress)
        core.delete_namespaced_config_map.assert_called_once()
        assert progress.call_count >= 6
        sweeper[0]._next_sweep = 0
        sweeper[0].sweep(progress)
    assert core.delete_namespaced_config_map.call_count == 2
    core.list_namespaced_config_map.assert_called_once()
    assert not sweeper[0]._pending


def test_expired_page_token_restarts_scan(sweeper, apis):
    sweeper[0]._continue = "expired"
    apis[0].list_namespaced_config_map.side_effect = ApiException(status=410)
    sweeper[0].sweep()
    assert sweeper[0]._continue == ""
    apis[0].delete_namespaced_config_map.assert_not_called()


def test_live_job_repairs_interrupted_ownership(sweeper, apis, cm):
    batch = apis[1]
    batch.read_namespaced_job.side_effect = None
    batch.read_namespaced_job.return_value = client.V1Job(metadata=client.V1ObjectMeta(uid="actual-job-uid"))
    apis[0].api_client.sanitize_for_serialization.side_effect = client.ApiClient().sanitize_for_serialization
    sweeper[0].sweep()
    metadata = apis[0].patch_namespaced_config_map.call_args.kwargs["body"]["metadata"]
    assert metadata["uid"] == cm.metadata.uid
    assert metadata["resourceVersion"] == cm.metadata.resource_version
    assert metadata["ownerReferences"][0]["uid"] == "actual-job-uid"
    apis[0].delete_namespaced_config_map.assert_not_called()
    sweeper[1].assert_called_once()


@pytest.mark.parametrize("owned", [False, True])
def test_adoption_preserves_existing_owner(apis, cm, owned):
    owner = client.V1OwnerReference(api_version="batch/v1", kind="Job", name="job", uid="job-uid")
    cm.metadata.owner_references = [owner] if owned else None
    adopt_step_configmap(apis[0], "test-namespace", cm.metadata.name, owner)
    assert apis[0].patch_namespaced_config_map.call_count == (0 if owned else 1)


def test_adoption_failure_does_not_fail_scheduling(apis, cm):
    apis[0].patch_namespaced_config_map.side_effect = ApiException(status=409)
    adopt_step_configmap(
        apis[0],
        "test-namespace",
        cm.metadata.name,
        client.V1OwnerReference(api_version="batch/v1", kind="Job", name="job", uid="job-uid"),
    )
    apis[0].patch_namespaced_config_map.assert_called_once()


def test_projected_volume_without_sources_does_not_stop_cleanup(sweeper, apis):
    projected = client.V1ProjectedVolumeSource(sources=[])
    projected.sources = None
    apis[0].list_namespaced_pod.return_value = client.V1PodList(
        items=[
            client.V1Pod(
                spec=client.V1PodSpec(
                    containers=[client.V1Container(name="task")],
                    volumes=[client.V1Volume(name="empty", projected=projected)],
                )
            )
        ]
    )
    sweeper[0].sweep()
    apis[0].delete_namespaced_config_map.assert_called_once()


def test_delayed_requests_keep_progress_within_health_window(sweeper, apis, cm):
    core, batch, custom = apis
    candidates = [deepcopy(cm) for _ in range(SWEEP_PAGE_SIZE)]
    for index, candidate in enumerate(candidates):
        candidate.metadata.name = f"delayed-{index}"
    ticks = [100.0]
    heartbeats = [ticks[0]]

    def delay(result):
        ticks[0] += 1
        return result

    core.list_namespaced_config_map.side_effect = lambda **kw: delay(client.V1ConfigMapList(items=candidates))
    core.list_namespaced_pod.side_effect = lambda **kw: delay(client.V1PodList(items=[]))
    core.read_namespaced_config_map.side_effect = lambda **kw: delay(candidates[0])

    def absent(**kwargs):
        ticks[0] += 1
        raise ApiException(status=404)

    batch.read_namespaced_job.side_effect = absent
    custom.get_namespaced_custom_object.side_effect = absent
    sweeper[1].side_effect = lambda *args: delay(True)
    core.delete_namespaced_config_map.side_effect = lambda **kw: delay(None)
    with patch("nhx.core.jobs.controllers.backends.kubernetes.configmaps.time.monotonic", side_effect=lambda: ticks[0]):
        sweeper[0].sweep(lambda: heartbeats.append(ticks[0]))
    assert max(right - left for left, right in zip(heartbeats, heartbeats[1:])) < 6
    assert len(sweeper[0]._pending) == SWEEP_PAGE_SIZE - 1
    core.delete_namespaced_config_map.assert_called_once()


@pytest.mark.parametrize("page_contents", ["young", "owned", "empty"])
def test_scan_advances_past_ineligible_pages(sweeper, apis, cm, page_contents):
    core = apis[0]
    ineligible = deepcopy(cm)
    if page_contents == "young":
        ineligible.metadata.creation_timestamp = datetime.datetime.now(datetime.UTC)
    elif page_contents == "owned":
        ineligible.metadata.owner_references = [
            client.V1OwnerReference(api_version="batch/v1", kind="Job", name="retained", uid="job-uid")
        ]
    core.list_namespaced_config_map.side_effect = [
        client.V1ConfigMapList(
            items=[] if page_contents == "empty" else [ineligible],
            metadata=client.V1ListMeta(_continue="candidate-page"),
        ),
        client.V1ConfigMapList(items=[cm], metadata=client.V1ListMeta()),
    ]
    progress = MagicMock()
    sweeper[0].sweep(progress)
    assert [call.kwargs["_continue"] for call in core.list_namespaced_config_map.call_args_list] == [
        "",
        "candidate-page",
    ]
    core.list_namespaced_pod.assert_called_once()
    core.delete_namespaced_config_map.assert_called_once()
    assert progress.call_count >= 2


def test_empty_page_request_cap_preserves_continuation(sweeper, apis, cm):
    core = apis[0]
    core.list_namespaced_config_map.side_effect = [
        *[
            client.V1ConfigMapList(items=[], metadata=client.V1ListMeta(_continue=f"page-{index + 1}"))
            for index in range(SWEEP_MAX_PAGES)
        ],
        client.V1ConfigMapList(items=[cm], metadata=client.V1ListMeta()),
    ]
    with patch("nhx.core.jobs.controllers.backends.kubernetes.configmaps.time.monotonic", return_value=100.0):
        sweeper[0].sweep()
        assert core.list_namespaced_config_map.call_count == SWEEP_MAX_PAGES
        core.list_namespaced_pod.assert_not_called()
        assert sweeper[0]._continue == f"page-{SWEEP_MAX_PAGES}"
        sweeper[0]._next_sweep = 0
        sweeper[0].sweep()
    assert core.list_namespaced_config_map.call_args.kwargs["_continue"] == f"page-{SWEEP_MAX_PAGES}"
    core.delete_namespaced_config_map.assert_called_once()


def test_empty_page_deadline_preserves_continuation(sweeper, apis, cm):
    core = apis[0]
    with patch("nhx.core.jobs.controllers.backends.kubernetes.configmaps.time.monotonic", return_value=100.0) as clock:

        def slow_page(**kwargs):
            clock.return_value = 103.0
            return client.V1ConfigMapList(items=[], metadata=client.V1ListMeta(_continue="after-deadline"))

        core.list_namespaced_config_map.side_effect = slow_page
        sweeper[0].sweep()
        core.list_namespaced_config_map.assert_called_once()
        core.list_namespaced_pod.assert_not_called()
        assert sweeper[0]._continue == "after-deadline"
        core.list_namespaced_config_map.side_effect = None
        core.list_namespaced_config_map.return_value = client.V1ConfigMapList(items=[cm], metadata=client.V1ListMeta())
        clock.return_value = 200.0
        sweeper[0].sweep()
    assert core.list_namespaced_config_map.call_args.kwargs["_continue"] == "after-deadline"
    core.delete_namespaced_config_map.assert_called_once()


def test_empty_final_page_stops_without_restarting_scan(sweeper, apis):
    core = apis[0]
    core.list_namespaced_config_map.return_value = client.V1ConfigMapList(items=[], metadata=client.V1ListMeta())
    sweeper[0].sweep()
    core.list_namespaced_config_map.assert_called_once()
    core.list_namespaced_pod.assert_not_called()
    assert sweeper[0]._continue == ""


def test_expired_token_after_empty_page_restarts_scan(sweeper, apis, cm):
    core = apis[0]
    core.list_namespaced_config_map.side_effect = [
        client.V1ConfigMapList(items=[], metadata=client.V1ListMeta(_continue="expired")),
        ApiException(status=410),
        client.V1ConfigMapList(items=[cm], metadata=client.V1ListMeta()),
    ]
    sweeper[0].sweep()
    assert core.list_namespaced_config_map.call_count == 2
    assert sweeper[0]._continue == ""
    core.delete_namespaced_config_map.assert_not_called()
    sweeper[0]._next_sweep = 0
    sweeper[0].sweep()
    assert core.list_namespaced_config_map.call_args.kwargs["_continue"] == ""
    core.delete_namespaced_config_map.assert_called_once()


@pytest.mark.parametrize("expires_after", [300, 3600])
def test_expiring_inventory_reaches_orphan_after_retained_prefix(sweeper, apis, cm, expires_after):
    core = apis[0]
    owned = deepcopy(cm)
    owned.metadata.owner_references = [
        client.V1OwnerReference(api_version="batch/v1", kind="Job", name="retained", uid="job-uid")
    ]
    inventory = [owned] * 400 + [cm]
    clock = [0.0]
    expirations = []

    def list_page(**kwargs):
        token = kwargs["_continue"]
        snapshot, offset = tuple(map(int, token.split(":"))) if token else (int(clock[0]), 0)
        if clock[0] - snapshot >= expires_after:
            expirations.append(offset)
            expired = ApiException(status=410, reason="ResourceExpired")
            expired.body = json.dumps({"metadata": {"continue": f"{int(clock[0])}:{offset}"}})
            raise expired
        end = offset + kwargs["limit"]
        return client.V1ConfigMapList(
            items=inventory[offset:end],
            metadata=client.V1ListMeta(_continue=f"{snapshot}:{end}" if end < len(inventory) else ""),
        )

    core.list_namespaced_config_map.side_effect = list_page
    with patch("nhx.core.jobs.controllers.backends.kubernetes.configmaps.time.monotonic", side_effect=lambda: clock[0]):
        for _ in range(30):
            before = core.list_namespaced_config_map.call_count
            sweeper[0].sweep()
            assert core.list_namespaced_config_map.call_count - before <= SWEEP_MAX_PAGES
            if core.delete_namespaced_config_map.called:
                break
            clock[0] += 60
    core.delete_namespaced_config_map.assert_called_once()
    if expires_after == 300:
        assert len(expirations) >= 2
        assert all(right > left for left, right in zip(expirations, expirations[1:]))
    else:
        assert not expirations


@pytest.mark.parametrize("body", [None, "not-json", "[]", '{"metadata": null}', '{"metadata": {"continue": 7}}'])
def test_invalid_expiration_response_restarts_safely(sweeper, apis, body):
    error = ApiException(status=410)
    error.body = body
    sweeper[0]._continue = "expired"
    apis[0].list_namespaced_config_map.side_effect = error
    sweeper[0].sweep()
    assert sweeper[0]._continue == ""
    apis[0].delete_namespaced_config_map.assert_not_called()


def test_pod_list_expiration_does_not_replace_configmap_cursor(sweeper, apis, cm):
    apis[0].list_namespaced_config_map.return_value = client.V1ConfigMapList(
        items=[cm], metadata=client.V1ListMeta(_continue="configmap-next-page")
    )
    error = ApiException(status=410)
    error.body = json.dumps({"metadata": {"continue": "pod-page"}})
    apis[0].list_namespaced_pod.side_effect = error
    sweeper[0].sweep()
    assert sweeper[0]._continue == "configmap-next-page"
    apis[0].delete_namespaced_config_map.assert_not_called()
