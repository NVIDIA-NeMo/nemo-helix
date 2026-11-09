# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect old step ConfigMaps after workloads disappear and platform steps become terminal."""

import datetime
import json
import logging
from collections import deque
from collections.abc import Callable
from time import monotonic

from kubernetes import client
from kubernetes.client import CustomObjectsApi
from kubernetes.client.rest import ApiException
from nhx.common.jobs.constants import NEMO_JOB_STEP_CONFIG_FILE_NAME
from nhx.core.jobs.app.constants import (
    JOB_ATTEMPT_ID_LABEL,
    JOB_ID_LABEL,
    JOB_MANAGED_BY_JOBS_CONTROLLER,
    JOB_MANAGED_BY_LABEL,
    JOB_STEP_ID_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_TYPE_JOB,
    JOB_TYPE_LABEL,
    JOB_WORKSPACE_ID_LABEL,
)
from urllib3.exceptions import HTTPError

logger = logging.getLogger(__name__)
SWEEP_INTERVAL_SECONDS = 60
SWEEP_PAGE_SIZE = 5
SWEEP_MAX_PAGES = 5
SWEEP_BUDGET_SECONDS = 2
REQUEST_TIMEOUT = (1, 2)
MIN_CONFIGMAP_AGE = datetime.timedelta(minutes=5)
SELECTOR = f"app=nemo-job,{JOB_MANAGED_BY_LABEL}={JOB_MANAGED_BY_JOBS_CONTROLLER},{JOB_TYPE_LABEL}={JOB_TYPE_JOB}"
IDENTITY_LABELS = (JOB_ID_LABEL, JOB_WORKSPACE_ID_LABEL, JOB_ATTEMPT_ID_LABEL, JOB_STEP_NAME_LABEL, JOB_STEP_ID_LABEL)


def _set_owner(
    core_v1: client.CoreV1Api, namespace: str, cm: client.V1ConfigMap, owner: client.V1OwnerReference
) -> None:
    """Attach only an unowned, managed object, rejecting concurrent changes."""
    metadata = cm.metadata
    if (
        not metadata
        or not metadata.uid
        or not metadata.resource_version
        or metadata.owner_references
        or metadata.deletion_timestamp
        or (metadata.labels or {}).get(JOB_MANAGED_BY_LABEL) != JOB_MANAGED_BY_JOBS_CONTROLLER
    ):
        return
    core_v1.patch_namespaced_config_map(
        name=metadata.name,
        namespace=namespace,
        body={
            "metadata": {
                "uid": metadata.uid,
                "resourceVersion": metadata.resource_version,
                "ownerReferences": [core_v1.api_client.sanitize_for_serialization(owner)],
            }
        },
        _request_timeout=REQUEST_TIMEOUT,
    )


def adopt_step_configmap(core_v1: client.CoreV1Api, namespace: str, name: str, owner: client.V1OwnerReference) -> None:
    """Let Kubernetes collect the configuration with its Job; the sweep repairs failures."""
    try:
        cm = core_v1.read_namespaced_config_map(name=name, namespace=namespace, _request_timeout=REQUEST_TIMEOUT)
        _set_owner(core_v1, namespace, cm, owner)
    except (ApiException, HTTPError):
        logger.warning(
            "Could not set step ConfigMap owner; the sweep will retry", extra={"configmap": name}, exc_info=True
        )


def _replacement_continue(error: ApiException) -> str:
    """Read the server's forward cursor without trusting the error response shape."""
    try:
        status = json.loads(error.body or "{}")
    except json.JSONDecodeError:
        return ""
    if not isinstance(status, dict):
        return ""
    metadata = status.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    token = metadata.get("continue")
    return token if isinstance(token, str) else ""


class StepConfigMapSweeper:
    def __init__(
        self,
        core_v1: client.CoreV1Api,
        batch_v1: client.BatchV1Api,
        namespace: str,
        step_is_terminal: Callable[[str, str, str, str], bool],
    ) -> None:
        self._core_v1 = core_v1
        self._batch_v1 = batch_v1
        self._custom_v1 = CustomObjectsApi(core_v1.api_client)
        self._namespace = namespace
        self._step_is_terminal = step_is_terminal
        self._next_sweep = 0.0
        self._continue = ""
        self._pending: deque[client.V1ConfigMap] = deque()

    def sweep(self, on_progress: Callable[[], None] | None = None) -> None:
        """Scan a bounded number of pages and retain unfinished candidates for the next pass."""
        now = monotonic()
        if now < self._next_sweep:
            return
        self._next_sweep = now + SWEEP_INTERVAL_SECONDS
        deadline = now + SWEEP_BUDGET_SECONDS
        progress = on_progress or (lambda: None)
        try:
            pages = 0
            while not self._pending and pages < SWEEP_MAX_PAGES:
                page = self._list_page()
                if page is None:
                    progress()
                    return
                self._continue = (page.metadata._continue if page.metadata else None) or ""
                self._pending.extend(cm for cm in page.items if self._is_candidate(cm))
                pages += 1
                progress()
                if not self._continue or monotonic() >= deadline:
                    break
            if not self._pending:
                return
            pods = self._core_v1.list_namespaced_pod(namespace=self._namespace, _request_timeout=REQUEST_TIMEOUT).items
            referenced = _referenced_configmaps(pods)
            progress()
        except (ApiException, HTTPError):
            logger.warning("Could not list resources for step ConfigMap cleanup; will retry", exc_info=True)
            return

        processed = 0
        while self._pending and processed < SWEEP_PAGE_SIZE and (processed == 0 or monotonic() < deadline):
            cm = self._pending.popleft()
            processed += 1
            try:
                self._cleanup(cm, referenced, progress)
            except (ApiException, HTTPError, RuntimeError):
                logger.warning(
                    "Could not clean up step ConfigMap; will retry",
                    extra={"configmap": cm.metadata.name, "namespace": self._namespace},
                    exc_info=True,
                )
            finally:
                progress()

    def _list_page(self) -> client.V1ConfigMapList | None:
        try:
            return self._core_v1.list_namespaced_config_map(
                namespace=self._namespace,
                label_selector=SELECTOR,
                limit=SWEEP_PAGE_SIZE,
                _continue=self._continue,
                _request_timeout=REQUEST_TIMEOUT,
            )
        except ApiException as exc:
            if exc.status != 410:
                raise
            # A consistent inventory is unnecessary: every mutation rechecks the
            # object and workload. Keep the next key when switching snapshots.
            self._continue = _replacement_continue(exc)
            logger.warning(
                "Step ConfigMap inventory expired; will retry",
                extra={"namespace": self._namespace, "resuming": bool(self._continue)},
            )
            return None

    def _is_candidate(self, cm: client.V1ConfigMap) -> bool:
        metadata = cm.metadata
        if (
            not metadata
            or not metadata.name
            or not metadata.uid
            or not metadata.resource_version
            or not metadata.creation_timestamp
            or metadata.deletion_timestamp
            or metadata.owner_references
        ):
            return False
        labels = metadata.labels or {}
        if (
            labels.get("app") != "nemo-job"
            or labels.get(JOB_MANAGED_BY_LABEL) != JOB_MANAGED_BY_JOBS_CONTROLLER
            or labels.get(JOB_TYPE_LABEL) != JOB_TYPE_JOB
            or not all(labels.get(key) for key in IDENTITY_LABELS)
            or not cm.data
            or NEMO_JOB_STEP_CONFIG_FILE_NAME not in cm.data
        ):
            return False
        created_at = metadata.creation_timestamp
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=datetime.timezone.utc)
        return datetime.datetime.now(datetime.timezone.utc) - created_at >= MIN_CONFIGMAP_AGE

    def _cleanup(self, cm: client.V1ConfigMap, referenced: set[str], progress: Callable[[], None]) -> None:
        labels = cm.metadata.labels
        terminal = self._step_is_terminal(
            labels[JOB_ID_LABEL],
            labels[JOB_STEP_NAME_LABEL],
            labels[JOB_WORKSPACE_ID_LABEL],
            labels[JOB_ATTEMPT_ID_LABEL],
        )
        progress()
        current = self._core_v1.read_namespaced_config_map(
            name=cm.metadata.name, namespace=self._namespace, _request_timeout=REQUEST_TIMEOUT
        )
        progress()
        if (current.metadata.uid, current.metadata.resource_version) != (cm.metadata.uid, cm.metadata.resource_version):
            return
        # Check Jobs after platform state and identity, just before mutation.
        owner = self._job_owner(cm.metadata.name, progress)
        if owner:
            _set_owner(self._core_v1, self._namespace, current, owner)
            return
        if not terminal or cm.metadata.name in referenced:
            return
        try:
            self._core_v1.delete_namespaced_config_map(
                name=cm.metadata.name,
                namespace=self._namespace,
                body=client.V1DeleteOptions(
                    preconditions=client.V1Preconditions(
                        uid=cm.metadata.uid,
                        resource_version=cm.metadata.resource_version,
                    )
                ),
                _request_timeout=REQUEST_TIMEOUT,
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
        logger.info(
            "Cleaned up orphan step ConfigMap", extra={"configmap": cm.metadata.name, "namespace": self._namespace}
        )

    def _job_owner(self, name: str, progress: Callable[[], None]) -> client.V1OwnerReference | None:
        try:
            job = self._batch_v1.read_namespaced_job(
                name=name, namespace=self._namespace, _request_timeout=REQUEST_TIMEOUT
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
        else:
            return client.V1OwnerReference(api_version="batch/v1", kind="Job", name=name, uid=job.metadata.uid)
        finally:
            progress()
        try:
            volcano_job = self._custom_v1.get_namespaced_custom_object(
                group="batch.volcano.sh",
                version="v1alpha1",
                namespace=self._namespace,
                plural="jobs",
                name=name,
                _request_timeout=REQUEST_TIMEOUT,
            )
        except ApiException as exc:
            if exc.status == 404:
                return None
            raise
        finally:
            progress()
        return client.V1OwnerReference(
            api_version="batch.volcano.sh/v1alpha1", kind="Job", name=name, uid=volcano_job["metadata"]["uid"]
        )


def _referenced_configmaps(pods: list[client.V1Pod]) -> set[str]:
    names: set[str] = set()
    for pod in pods:
        spec = pod.spec
        if spec is None:
            continue
        for volume in spec.volumes or []:
            if volume.config_map:
                names.add(volume.config_map.name)
            if volume.projected:
                names.update(source.config_map.name for source in volume.projected.sources or [] if source.config_map)
        for container in [*spec.containers, *(spec.init_containers or []), *(spec.ephemeral_containers or [])]:
            names.update(
                env.value_from.config_map_key_ref.name
                for env in (container.env or [])
                if env.value_from and env.value_from.config_map_key_ref
            )
            names.update(env.config_map_ref.name for env in (container.env_from or []) if env.config_map_ref)
    return names
