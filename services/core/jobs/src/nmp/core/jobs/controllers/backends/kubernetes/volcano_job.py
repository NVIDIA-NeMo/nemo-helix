# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
from typing import Any, Literal

from kubernetes import client
from kubernetes.client.rest import ApiException
from nemo_platform_plugin.jobs.types import PlatformJobStepWithContext
from nmp.common.auth import get_workload_delegation_audience
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.core.jobs.app.constants import (
    JOB_EXECUTION_BACKEND_LABEL,
    JOB_EXECUTION_PROFILE_LABEL,
    JOB_ID_LABEL,
    JOB_MANAGED_BY_JOBS_CONTROLLER,
    JOB_MANAGED_BY_LABEL,
    JOB_MULTINODE_NETWORKING_ANNOTATION,
    JOB_NUM_NODES_ANNOTATION,
    JOB_STEP_NAME_LABEL,
    JOB_TYPE_JOB,
    JOB_TYPE_LABEL,
    JOB_USES_PERSISTENT_STORAGE_LABEL,
    JOB_WORKSPACE_ID_LABEL,
    KUBE_JOB_SELECTOR_LABELS,
)
from nmp.core.jobs.app.providers import DistributedGPUExecutionProvider
from nmp.core.jobs.app.schemas import BaseExecutionProfile
from nmp.core.jobs.controllers.backends.base import (
    JobBackend,
    JobUpdate,
    require_staleness_timeout_seconds,
    staleness_error_message,
)
from nmp.core.jobs.controllers.backends.kubernetes.common import (
    BaseKubernetesExecutionProfileConfig,
    build_event_field_selector,
    build_metadata,
    cleanup_job_persistent_storage,
    common_labels_for_step,
    create_configmap,
    create_pod_template_spec,
    delete_configmap,
    get_namespace_from_environment,
    load_kubernetes_config,
    name_for_step,
    update_all_tasks,
)
from nmp.core.jobs.controllers.backends.kubernetes.workload_delegations import (
    KubernetesPodBoundWorkloadDelegationManager,
    KubernetesPodBoundWorkloadDelegationTarget,
)
from nmp.core.jobs.controllers.backends.workload_tokens import create_authenticated_workload_delegation_store
from pydantic import Field

logger = logging.getLogger(__name__)


class VolcanoJobExecutionProfileConfig(BaseKubernetesExecutionProfileConfig):
    """Configuration for Volcano Job Execution Profile"""

    queue: str = Field(
        default="default",
        description="The Volcano queue to submit the job to.",
    )
    scheduler_name: str = Field(
        default="volcano",
        description="The scheduler name to use for the Volcano job.",
    )

    max_retry: int = Field(default=0, description="maxRetry indicates the maximum number of retries allowed by the job")

    plugins: dict[str, Any] = Field(
        default_factory=dict,
        description="plugins indicates the plugins used by Volcano when the job is scheduled. We always add the pytorch plugin if more than one node.",
    )

    enable_multi_node_networking: bool = Field(
        default=True,
        description="Enable multi-node networking injection. Sets annotations to trigger Kyverno policy mutations.",
    )


class VolcanoJobExecutionProfile(BaseExecutionProfile):
    """Volcano Job Execution Profile"""

    backend: Literal["volcano_job"] = "volcano_job"
    config: VolcanoJobExecutionProfileConfig = Field(
        description="Additional configuration for the kubernetes executor",
    )

    @property
    def supports_persistent_storage(self) -> bool:
        """Indicates if the execution profile supports persistent storage."""
        return self.config.storage is not None and self.config.storage.pvc_name is not None


class VolcanoJobBackend(
    JobBackend[DistributedGPUExecutionProvider, VolcanoJobExecutionProfileConfig],
):
    BACKEND_NAME: str = "volcano_job"

    def init(self):
        load_kubernetes_config()
        self._core_v1 = client.CoreV1Api()
        self._custom_v1 = client.CustomObjectsApi()
        self._batch_v1 = client.BatchV1Api()
        self.namespace = self._execution_profile_config.namespace or get_namespace_from_environment()
        self._workload_delegation_store = create_authenticated_workload_delegation_store(self._nmp_sdk)
        self._workload_delegations = KubernetesPodBoundWorkloadDelegationManager(
            core_v1=self._core_v1,
            namespace=self.namespace,
            ttl_seconds_active=lambda: self._execution_profile_config.ttl_seconds_active,
            workload_audience=get_workload_delegation_audience,
            register_workload_delegation=self._workload_delegation_store.register,
            revoke_workload_delegation=self._workload_delegation_store.revoke,
        )

    def shutdown(self):
        self._core_v1.api_client.close()
        self._custom_v1.api_client.close()
        return

    def _workload_delegation_target_for_volcano_job(
        self, job: dict
    ) -> KubernetesPodBoundWorkloadDelegationTarget | None:
        metadata = job.get("metadata", {}) or {}
        job_name = metadata.get("name")
        if not isinstance(job_name, str) or not job_name:
            logger.warning("Skipping Volcano workload delegation reconciliation for job with missing name")
            return None

        service_account_name = self._execution_profile_config.service_account_name
        for task in job.get("spec", {}).get("tasks", []) or []:
            task_spec = task.get("template", {}).get("spec", {}) or {}
            task_service_account_name = task_spec.get("serviceAccountName") or task_spec.get("service_account_name")
            if isinstance(task_service_account_name, str) and task_service_account_name:
                service_account_name = task_service_account_name
                break

        return KubernetesPodBoundWorkloadDelegationTarget(
            namespace=metadata.get("namespace") or self.namespace,
            name=job_name,
            labels=metadata.get("labels", {}) or {},
            service_account_name=service_account_name,
        )

    def get_volcano_job_by_labels(self, labels: dict[str, str]) -> dict | None:
        jobs = self.get_volcano_job_list_by_labels(labels)
        if len(jobs) > 0:
            return jobs[0]
        return None

    def get_volcano_job_by_name(self, name: str) -> dict | None:
        try:
            return self._custom_v1.get_namespaced_custom_object(
                group="batch.volcano.sh",
                version="v1alpha1",
                namespace=self.namespace,
                plural="jobs",
                name=name,
            )
        except ApiException as e:
            if e.status == 404:
                return None
            else:
                logger.warning("Error API fetching Volcano job by name: %s", e.reason)
        except Exception:
            logger.exception("Unexpected error fetching Volcano job by name")
        return None

    def get_volcano_job_list_by_labels(self, labels: dict[str, str]) -> list[dict]:
        label_selector = ",".join([f"{k}={v}" for k, v in labels.items()])
        try:
            job_list = self._custom_v1.list_namespaced_custom_object(
                group="batch.volcano.sh",
                version="v1alpha1",
                namespace=self.namespace,
                plural="jobs",
                label_selector=label_selector,
            )
            if "items" in job_list and len(job_list["items"]) > 0:
                return job_list["items"]
        except ApiException as e:
            # 404 means the Volcano CRD (batch.volcano.sh/v1alpha1) is not installed.
            # This is distinct from "no matching jobs" which returns 200 with empty items.
            if e.status == 404:
                logger.debug("Volcano CRD not found (batch.volcano.sh/v1alpha1) - Volcano may not be installed")
            else:
                logger.warning("Error API fetching Volcano jobs: %s", e.reason)
        except Exception:
            logger.exception("Unexpected error fetching Volcano jobs")
        return []

    def schedule(
        self,
        executor_config: DistributedGPUExecutionProvider,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        """
        Schedule a Volcano Job using the Volcano Job CRD.
        """
        job_name = name_for_step(step)

        # Basic labels for tracking
        common_labels = common_labels_for_step(step)
        common_labels[JOB_EXECUTION_BACKEND_LABEL] = self.BACKEND_NAME
        common_labels[JOB_EXECUTION_PROFILE_LABEL] = self._profile_name

        if executor_config.resources is not None and executor_config.resources.num_gpus is not None:
            num_gpus = executor_config.resources.num_gpus
        elif self._execution_profile_config.num_gpus is not None:
            num_gpus = self._execution_profile_config.num_gpus
        else:
            num_gpus = 1

        # Create a configmap storing the job's step config
        configmap_name = create_configmap(
            core_v1=self._core_v1,
            namespace=self.namespace,
            step=step,
        )

        pod_template = create_pod_template_spec(
            step,
            self.namespace,
            executor_config.container,
            common_labels,
            self.get_secrets_environment_variable_for_injection(step),
            configmap_name,
            self._execution_profile_config,
            self._core_v1,
            num_gpus=num_gpus,
            executor_resources=executor_config.resources,
        )

        job_metadata = build_metadata(
            labels=common_labels,
            metadata=self._execution_profile_config.job_metadata,
        )
        job_metadata.name = job_name

        # Volcano job policies
        job_policies = [
            {
                "event": "PodFailed",
                "action": "RestartJob",
            }
        ]
        leader_policies = [
            {
                "event": "TaskCompleted",
                "action": "CompleteJob",
            }
        ]

        # Volcano Job spec
        if executor_config.resources is not None and executor_config.resources.num_nodes is not None:
            num_nodes = executor_config.resources.num_nodes
        else:
            num_nodes = 1

        # Add multi-node networking annotations if enabled and multi-node
        if self._execution_profile_config.enable_multi_node_networking and num_nodes > 1:
            # Create a copy of annotations to avoid mutating the shared metadata object
            if pod_template.metadata.annotations is None:
                pod_template.metadata.annotations = {}
            else:
                pod_template.metadata.annotations = dict(pod_template.metadata.annotations)

            pod_template.metadata.annotations[JOB_MULTINODE_NETWORKING_ANNOTATION] = "true"
            pod_template.metadata.annotations[JOB_NUM_NODES_ANNOTATION] = str(num_nodes)

        if num_nodes <= 1:
            # (single task, minimal)
            tasks = [
                {
                    "name": "worker",
                    "replicas": 1,
                    "policies": leader_policies,
                    "template": to_full_dict(pod_template),
                }
            ]
        else:
            leader_task = {
                "name": "leader",
                "replicas": 1,
                "policies": leader_policies,
                "template": to_full_dict(pod_template),
            }
            worker_task = {
                "name": "worker",
                "replicas": num_nodes - 1,
                "template": to_full_dict(pod_template),
            }
            tasks = [leader_task, worker_task]

        volcano_job_spec: dict[str, Any] = {
            "minAvailable": num_nodes,
            "queue": self._execution_profile_config.queue,
            "schedulerName": self._execution_profile_config.scheduler_name,
            "policies": job_policies,
            "tasks": tasks,
        }
        volcano_job: dict[str, Any] = {
            "apiVersion": "batch.volcano.sh/v1alpha1",
            "kind": "Job",
            "metadata": job_metadata.to_dict(),
            "spec": volcano_job_spec,
        }

        plugins = dict(self._execution_profile_config.plugins)
        # We always want the pytorch plugin in more than one node.
        if num_nodes > 1 and "pytorch" not in plugins:
            plugins["pytorch"] = ["--master=leader", "--worker=worker", "--port=23456"]
        if plugins:
            volcano_job_spec["plugins"] = plugins

        if self._execution_profile_config.max_retry is not None:
            volcano_job_spec["maxRetry"] = self._execution_profile_config.max_retry

        try:
            self._custom_v1.create_namespaced_custom_object(
                group="batch.volcano.sh",
                version="v1alpha1",
                namespace=self.namespace,
                plural="jobs",
                body=volcano_job,
            )
            logger.info(
                "Scheduled job step with Volcano job", extra={"job_name": job_name, "namespace": self.namespace}
            )
        except ApiException as e:
            if e.status == 404:
                # Volcano CRD not found - Volcano is not installed in the cluster
                error_message = (
                    "Volcano is not available in this Kubernetes cluster. "
                    "Distributed GPU jobs require Volcano to be installed. "
                    "Please contact your platform administrator to install Volcano or use a different execution profile."
                )
                logger.error(
                    "Failed to schedule Volcano job",
                    extra={"job_name": job_name, "namespace": self.namespace, "error_message": error_message},
                )
                # Clean up the configmap since we won't be creating the job
                delete_configmap(
                    core_v1=self._core_v1,
                    namespace=self.namespace,
                    name=configmap_name,
                )
                return JobUpdate(
                    status=PlatformJobStatus.ERROR,
                    error_details={
                        "message": error_message,
                        "reason": "VolcanoNotInstalled",
                    },
                )
            else:
                logger.exception(
                    "Failed to create Volcano Job",
                    extra={"job_name": job_name, "namespace": self.namespace, "error_message": e.reason},
                )
                raise
        except Exception:
            logger.exception("Failed to create Volcano Job", extra={"job_name": job_name, "namespace": self.namespace})
            raise
        return JobUpdate(
            status=PlatformJobStatus.PENDING, status_details={"message": "Job scheduled with Volcano backend"}
        )

    def sync(
        self,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        """Sync the Volcano job with the current step status."""

        volcano_job: dict | None = self.get_volcano_job_by_name(name_for_step(step))
        if step.status == PlatformJobStatus.ACTIVE:
            if volcano_job is not None and (
                result := self.enforce_sync_ttl(
                    step, self._execution_profile_config.ttl_seconds_active, volcano_job, before_active=False
                )
            ):
                return result
            if volcano_job is not None and self.check_step_is_stale(step):
                message = staleness_error_message(require_staleness_timeout_seconds(step))
                return self.sync_remove_job_with_status(
                    step,
                    PlatformJobStatus.ERROR,
                    volcano_job,
                    status_details={"message": message},
                    error_details={"message": message},
                )
            return self.sync_active(step, volcano_job)
        elif step.status == PlatformJobStatus.PENDING:
            if volcano_job is not None and (
                result := self.enforce_sync_ttl(
                    step,
                    self._execution_profile_config.ttl_seconds_before_active,
                    volcano_job,
                    before_active=True,
                )
            ):
                return result
            return self.sync_pending(step, volcano_job)
        elif step.status == PlatformJobStatus.CANCELLING:
            return self.sync_remove_job_with_status(step, PlatformJobStatus.CANCELLED, volcano_job)
        elif step.status == PlatformJobStatus.PAUSING:
            raise NotImplementedError("Pausing not yet implemented for volcano backend.")
        else:
            raise ValueError(f"Unhandled step status for sync: {step.status}")

    def cleanup_steps(self):
        jobs = self.get_volcano_job_list_by_labels(labels=KUBE_JOB_SELECTOR_LABELS)
        for job in jobs:
            # Extract job_id and step_name from labels
            job_id = job.get("metadata", {}).get("labels", {}).get(JOB_ID_LABEL)
            step_name = job.get("metadata", {}).get("labels", {}).get(JOB_STEP_NAME_LABEL)
            workspace_id = job.get("metadata", {}).get("labels", {}).get(JOB_WORKSPACE_ID_LABEL)

            # Cleanup jobs have similar labels but are of a different type, and cleanup on their own.
            job_type = job.get("metadata", {}).get("labels", {}).get(JOB_TYPE_LABEL)
            if job_type is not None and job_type != JOB_TYPE_JOB:
                continue

            # Skip jobs missing required labels (e.g. old or manually created); avoid calling SDK with None.
            if not job_id or not step_name or not workspace_id:
                logger.warning(
                    "Skipping cleanup for Volcano job with missing labels (job_id, step_name, or workspace_id)",
                    extra={
                        "job_name": job.get("metadata", {}).get("name"),
                        "job_id": job_id,
                        "step_name": step_name,
                        "workspace_id": workspace_id,
                    },
                )
                continue

            # Verify the step is terminal before cleaning up, or that the job entity is gone (e.g. workspace deleted).
            # check_step_is_terminal returns True when terminal or when step entity is not found (404).
            if not self.check_step_is_terminal(job=job_id, step_name=step_name, workspace=workspace_id):
                logger.debug(
                    "Skipping cleanup for job as step is not in terminal state",
                    extra={"job_id": job_id, "step_name": step_name, "workspace_id": workspace_id},
                )
                continue

            status, _ = map_volcano_job_status_to_platform_status(job["status"])

            if (
                self._execution_profile_config.cleanup_completed_jobs_immediately
                and status == PlatformJobStatus.COMPLETED
            ):
                uses_persistent_storage = (
                    job.get("metadata", {}).get("labels", {}).get(JOB_USES_PERSISTENT_STORAGE_LABEL) == "true"
                )
                if uses_persistent_storage and self._execution_profile_config.storage:
                    if self.check_job_persistent_storage_cleanup_allowed(
                        job=job_id, step_name=step_name, workspace=workspace_id
                    ):
                        logger.info(
                            "Cleaning up persistent storage for successful job",
                            extra={"workspace_id": workspace_id, "job_id": job_id},
                        )
                        cleanup_job_persistent_storage(
                            namespace=self.namespace,
                            batch_v1=self._batch_v1,
                            pvc_name=self._execution_profile_config.storage.pvc_name,
                            workspace=workspace_id,
                            job_id=job_id,
                            step_name=step_name,
                            permissions_image=self._execution_profile_config.storage.volume_permissions_image,
                            execution_backend=self.BACKEND_NAME,
                            execution_profile=self._profile_name,
                            job_metadata=self._execution_profile_config.job_metadata,
                            pod_metadata=self._execution_profile_config.pod_metadata,
                            pod_security_context=self._execution_profile_config.pod_security_context,
                        )
                    else:
                        logger.debug(
                            "Skipping persistent storage cleanup for job",
                            extra={"workspace_id": workspace_id, "job_id": job_id, "step_name": step_name},
                        )

                logger.debug(
                    "Garbage collecting volcano job",
                    extra={"volcano_job_name": job["metadata"]["name"], "namespace": self.namespace},
                )
                self.terminate_job(job)
                continue

            if (
                self._execution_profile_config.cleanup_completed_jobs_immediately is False
                and status == PlatformJobStatus.COMPLETED
            ) or status == PlatformJobStatus.ERROR:
                # Check that the job has exceeded the configured TTL before being deleted from the kube cluster.
                last_transition_time_str = job["status"]["state"]["lastTransitionTime"]
                if (
                    datetime.datetime.fromisoformat(last_transition_time_str)
                    + datetime.timedelta(seconds=self._execution_profile_config.ttl_seconds_after_finished)
                ) < datetime.datetime.now(datetime.UTC):
                    if status == PlatformJobStatus.COMPLETED:
                        uses_persistent_storage = (
                            job.get("metadata", {}).get("labels", {}).get(JOB_USES_PERSISTENT_STORAGE_LABEL) == "true"
                        )
                        if uses_persistent_storage and self._execution_profile_config.storage:
                            if self.check_job_persistent_storage_cleanup_allowed(
                                job=job_id, step_name=step_name, workspace=workspace_id
                            ):
                                logger.info(
                                    "Cleaning up persistent storage for successful volcano job",
                                    extra={"workspace_id": workspace_id, "job_id": job_id},
                                )
                                cleanup_job_persistent_storage(
                                    namespace=self.namespace,
                                    batch_v1=self._batch_v1,
                                    pvc_name=self._execution_profile_config.storage.pvc_name,
                                    workspace=workspace_id,
                                    job_id=job_id,
                                    step_name=step_name,
                                    permissions_image=self._execution_profile_config.storage.volume_permissions_image,
                                    execution_backend=self.BACKEND_NAME,
                                    execution_profile=self._profile_name,
                                    job_metadata=self._execution_profile_config.job_metadata,
                                    pod_metadata=self._execution_profile_config.pod_metadata,
                                    pod_security_context=self._execution_profile_config.pod_security_context,
                                )
                            else:
                                logger.debug(
                                    "Skipping persistent storage cleanup for volcano job",
                                    extra={"workspace_id": workspace_id, "job_id": job_id, "step_name": step_name},
                                )

                    logger.debug(
                        "Garbage collecting volcano job",
                        extra={"volcano_job_name": job["metadata"]["name"], "namespace": self.namespace},
                    )
                    self.terminate_job(job)

    def get_volcano_job_events(self, job: dict) -> list[dict[str, Any]]:
        """Get events related to a Volcano Job."""
        job_name = job.get("metadata", {}).get("name", "")
        try:
            event_list = self._core_v1.list_namespaced_event(
                namespace=self.namespace,
                field_selector=build_event_field_selector(
                    kind="Job",
                    api_version="batch.volcano.sh/v1alpha1",
                    name=job_name,
                ),
            )
            events = []
            for event in event_list.items:
                events.append(
                    {
                        "type": event.type,
                        "reason": event.reason,
                        "message": event.message,
                        "first_timestamp": str(event.first_timestamp),
                        "last_timestamp": str(event.last_timestamp),
                        "count": event.count,
                    }
                )
            return events
        except ApiException:
            logger.exception(
                "Error API fetching events for Volcano Job", extra={"job_name": job_name, "namespace": self.namespace}
            )
        except Exception:
            logger.exception(
                "Error fetching events for Volcano Job", extra={"job_name": job_name, "namespace": self.namespace}
            )
        return []

    def get_volcano_pod_group_events(self, job: dict) -> list[dict[str, Any]]:
        """Get events related to a Volcano PodGroup."""
        pod_group_name_prefix = job.get("metadata", {}).get("name", "")
        try:
            event_list = self._core_v1.list_namespaced_event(
                namespace=self.namespace,
                field_selector=build_event_field_selector(
                    kind="PodGroup",
                    api_version="scheduling.volcano.sh/v1beta1",
                ),
            )
            events = []
            for event in event_list.items:
                if event.involved_object.name.startswith(pod_group_name_prefix):
                    events.append(
                        {
                            "type": event.type,
                            "reason": event.reason,
                            "message": event.message,
                            "first_timestamp": str(event.first_timestamp),
                            "last_timestamp": str(event.last_timestamp),
                            "count": event.count,
                        }
                    )
            return events
        except ApiException:
            logger.exception("Error API fetching events for Volcano PodGroups")
        except Exception:
            logger.exception("Error fetching events for Volcano PodGroups")
        return []

    def create_step_update(self, step: PlatformJobStepWithContext, job: dict) -> JobUpdate:
        status, status_details = map_volcano_job_status_to_platform_status(job["status"])
        status_details["events"] = self.get_volcano_job_events(job)
        status_details["events"].extend(self.get_volcano_pod_group_events(job))
        tasks_has_error = update_all_tasks(self._nmp_sdk, self._core_v1, self.namespace, step)
        if tasks_has_error:
            status = PlatformJobStatus.ERROR
            if "message" not in status_details:
                status_details["message"] = "One or more tasks are in error state"
            else:
                status_details["message"] += "; One or more tasks are in error state"
        if self._workload_delegations.should_manage_step(step):
            target = self._workload_delegation_target_for_volcano_job(job)
            if target is not None:
                if status in PlatformJobStatus.terminals():
                    self._workload_delegations.revoke_for_target(target)
                else:
                    self._workload_delegations.ensure_for_target(step, target)
        return JobUpdate(status=status, status_details=status_details)

    def enforce_sync_ttl(
        self,
        step: PlatformJobStepWithContext,
        ttl_seconds: int,
        volcano_job: dict,
        *,
        before_active: bool = False,
    ) -> JobUpdate | None:
        ttl_exceeded = (
            self.check_step_ttl_before_active(step, ttl_seconds)
            if before_active
            else self.check_step_ttl(step, ttl_seconds)
        )
        if not ttl_exceeded:
            return None
        return self.sync_remove_job_with_status(
            step,
            PlatformJobStatus.ERROR,
            volcano_job,
            status_details={"message": f"Job timed out after reaching max TTL of {ttl_seconds} seconds"},
            error_details={"message": f"Job timed out after reaching max TTL of {ttl_seconds} seconds"},
        )

    def sync_active(self, step: PlatformJobStepWithContext, job: dict | None) -> JobUpdate:
        job_name = name_for_step(step)
        if job is None:
            logger.error("Job not found: %s", job_name)
            self._workload_delegations.revoke_by_key(namespace=self.namespace, name=job_name)
            # Job was deleted
            return JobUpdate(status=PlatformJobStatus.ERROR, error_details={"message": "Job not found"})
        else:
            return self.create_step_update(step, job)

    def sync_pending(self, step: PlatformJobStepWithContext, job: dict | None) -> JobUpdate:
        if job is None:
            # Job doesn't exist yet
            return JobUpdate(status=PlatformJobStatus.PENDING, status_details={"message": "Job not yet created"})
        else:
            return self.create_step_update(step, job)

    def sync_remove_job_with_status(
        self,
        step: PlatformJobStepWithContext,
        stop_status: PlatformJobStatus,
        job: dict | None,
        status_details: dict | None = None,
        error_details: dict | None = None,
    ) -> JobUpdate:
        if job is not None:
            # job has not yet been terminated
            update_all_tasks(self._nmp_sdk, self._core_v1, self.namespace, step)
            if status_details is None:
                status_details = {}
            events = self.get_volcano_job_events(job)
            events.extend(self.get_volcano_pod_group_events(job))
            status_details["events"] = events
            self.terminate_job(job)
            if error_details is not None:
                status = PlatformJobStatus.ERROR
            else:
                status = step.status
            return JobUpdate(status=status, status_details=status_details, error_details=error_details)
        else:
            self._workload_delegations.revoke_by_key(namespace=self.namespace, name=name_for_step(step))
            # If the job was not found, then it has been successfully stopped and removed
            # Transition into terminal state
            return JobUpdate(status=stop_status)

    def terminate_job(self, job: dict):
        labels = job.get("metadata", {}).get("labels", {}) or {}
        if labels.get(JOB_MANAGED_BY_LABEL) != JOB_MANAGED_BY_JOBS_CONTROLLER:
            job_name = job.get("metadata", {}).get("name", "unknown")
            logger.warning(
                "Skipping delete of Volcano job (not managed by jobs-controller)",
                extra=dict(job_name=job_name, namespace=self.namespace),
            )
            return
        if job_name := job.get("metadata", {}).get("name"):
            target = self._workload_delegation_target_for_volcano_job(job)
            if target is not None:
                self._workload_delegations.revoke_for_target(target)
            try:
                self._custom_v1.delete_namespaced_custom_object(
                    group="batch.volcano.sh",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="jobs",
                    name=job_name,
                    propagation_policy="Foreground",
                )
            except ApiException as e:
                if e.status == 404:
                    # Job or Volcano CRD not found - either already deleted or Volcano not installed
                    # This is acceptable since the goal is to remove the job
                    logger.debug(
                        "Volcano job not found during termination as already deleted or Volcano not available",
                        extra={"job_name": job_name, "namespace": self.namespace},
                    )
                else:
                    logger.error(
                        "Failed to delete Volcano job",
                        extra={"job_name": job_name, "namespace": self.namespace, "error_message": e.reason},
                    )
                    raise
            except Exception:
                logger.exception(
                    "Unexpected error deleting Volcano job", extra={"job_name": job_name, "namespace": self.namespace}
                )
                raise

            delete_configmap(
                core_v1=self._core_v1,
                namespace=self.namespace,
                name=job_name,
            )


def map_volcano_job_status_to_platform_status(status: dict) -> tuple[PlatformJobStatus, dict]:
    """Map Kubernetes job status to PlatformJobStatus."""
    if status["state"]["phase"] == "Completed":
        return PlatformJobStatus.COMPLETED, {"message": "Job completed successfully"}
    elif status["state"]["phase"] == "Failed":
        return PlatformJobStatus.ERROR, {"message": "Job failed"}
    elif status["state"]["phase"] == "Running":
        return PlatformJobStatus.ACTIVE, {"message": "Job is currently running"}
    else:
        return PlatformJobStatus.PENDING, {"message": "Job is pending (not yet started)"}


def to_full_dict(obj: Any):
    """
    Recursively serialize a Kubernetes model object to a dict,
    including unset fields as None.

    This resolves issues where lists of objects do not have a .to_dict() method
    and thus are not serialized properly.
    """
    if obj is None:
        return None

    # If it's a list, recurse on each element
    if isinstance(obj, list):
        return [to_full_dict(x) for x in obj]

    # If it's a dict, recurse on values
    if isinstance(obj, dict):
        return {k: to_full_dict(v) for k, v in obj.items()}

    # If it's a Kubernetes model object (has attribute_map)
    if hasattr(obj, "attribute_map"):
        result = {}
        for attr, json_key in obj.attribute_map.items():
            value = getattr(obj, attr, None)
            result[json_key] = to_full_dict(value)
        return result

    # Primitive type
    return obj
