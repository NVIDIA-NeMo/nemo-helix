# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generic, Literal, TypeVar

from kubernetes import client
from kubernetes.client.models import V1Job, V1JobStatus
from kubernetes.client.rest import ApiException
from nemo_platform_plugin.jobs.types import PlatformJobStepWithContext, PlatformJobTaskUpdate
from nmp.common.auth import get_workload_delegation_audience
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.core.jobs.app.constants import (
    JOB_EXECUTION_BACKEND_LABEL,
    JOB_EXECUTION_PROFILE_LABEL,
    JOB_ID_LABEL,
    JOB_MANAGED_BY_JOBS_CONTROLLER,
    JOB_MANAGED_BY_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_TYPE_JOB,
    JOB_TYPE_LABEL,
    JOB_USES_PERSISTENT_STORAGE_LABEL,
    JOB_WORKSPACE_ID_LABEL,
    KUBE_JOB_SELECTOR_LABELS,
)
from nmp.core.jobs.app.providers import (
    ComputeResources,
    ContainerSpec,
    CPUExecutionProvider,
    ExecutionProviderT,
    GPUExecutionProvider,
)
from nmp.core.jobs.app.schemas import BaseExecutionProfile
from nmp.core.jobs.controllers.backends.base import (
    JobBackend,
    JobUpdate,
    require_staleness_timeout_seconds,
    staleness_error_message,
)
from nmp.core.jobs.controllers.backends.kubernetes.common import (
    BaseKubernetesExecutionProfileConfig,
    aggregate_pod_statuses_for_job_step,
    build_event_field_selector,
    build_metadata,
    cleanup_job_persistent_storage,
    common_labels_for_step,
    create_configmap,
    create_pod_template_spec,
    delete_configmap,
    get_namespace_from_environment,
    list_pod_status,
    load_kubernetes_config,
    name_for_step,
    update_all_tasks,
)
from nmp.core.jobs.controllers.backends.kubernetes.workload_delegations import (
    KubernetesPodBoundWorkloadDelegationManager,
    KubernetesPodBoundWorkloadDelegationTarget,
)
from nmp.core.jobs.controllers.backends.workload_tokens import (
    create_authenticated_workload_delegation_store,
)
from pydantic import Field

logger = logging.getLogger(__name__)

ProviderT = TypeVar("ProviderT", bound=ExecutionProviderT)
KubernetesEventDetails = dict[str, object]
KubernetesStatusDetails = dict[str, object]


class KubernetesJobExecutionProfileConfig(BaseKubernetesExecutionProfileConfig):
    """Configuration for Kubernetes execution environment."""


class KubernetesJobExecutionProfile(BaseExecutionProfile):
    """
    Execution configuration for a Kubernetes Job.
    This is used to define the executor type, provider, profile, and any additional configuration
    required for the executor to run the job on Kubernetes
    """

    backend: Literal["kubernetes_job"] = "kubernetes_job"
    config: KubernetesJobExecutionProfileConfig = Field(
        description="Additional configuration for the kubernetes executor",
    )

    @property
    def supports_persistent_storage(self) -> bool:
        """Indicates if the execution profile supports persistent storage."""
        return self.config.storage is not None and self.config.storage.pvc_name != ""


class KubernetesJobBackend(JobBackend[ProviderT, KubernetesJobExecutionProfileConfig], Generic[ProviderT]):
    """Kubernetes backend for running jobs as Kubernetes Jobs."""

    BACKEND_NAME: str = "kubernetes_job"

    def init(self) -> None:
        """Initialize Kubernetes client and determine namespace."""
        load_kubernetes_config()
        self._batch_v1 = client.BatchV1Api()
        self._core_v1 = client.CoreV1Api()
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
        self._batch_v1.api_client.close()
        self._core_v1.api_client.close()
        return

    def _workload_delegation_target_for_job(self, job: V1Job) -> KubernetesPodBoundWorkloadDelegationTarget | None:
        metadata = job.metadata
        if metadata is None:
            logger.warning("Skipping Kubernetes workload delegation reconciliation for job with missing metadata")
            return None
        job_name = metadata.name
        if not isinstance(job_name, str) or not job_name:
            logger.warning("Skipping Kubernetes workload delegation reconciliation for job with missing name")
            return None
        namespace = metadata.namespace or self.namespace
        service_account_name = self._execution_profile_config.service_account_name
        if job.spec and job.spec.template and job.spec.template.spec:
            service_account_name = job.spec.template.spec.service_account_name or service_account_name
        return KubernetesPodBoundWorkloadDelegationTarget(
            namespace=namespace,
            name=job_name,
            labels=metadata.labels or {},
            service_account_name=service_account_name,
        )

    def get_job_by_name(self, name: str) -> V1Job | None:
        try:
            return self._batch_v1.read_namespaced_job(name=name, namespace=self.namespace)
        except client.ApiException as e:
            if e.status == 404:
                return None

            logger.exception(
                "Error API fetching Kubernetes Job by name", extra={"job_name": name, "namespace": self.namespace}
            )
        except Exception:
            logger.exception(
                "Error fetching Kubernetes Job by name", extra={"job_name": name, "namespace": self.namespace}
            )
        return None

    def get_kubernetes_job_list_by_labels(self, labels: dict[str, str]) -> list[V1Job]:
        label_selector = ",".join([f"{k}={v}" for k, v in labels.items()])
        try:
            job_list = self._batch_v1.list_namespaced_job(namespace=self.namespace, label_selector=label_selector)
            return job_list.items
        except client.ApiException:
            logger.exception("Error API fetching jobs")
        except Exception:
            logger.exception("Error fetching jobs")
        return []

    def schedule_job(
        self,
        container: ContainerSpec,
        step: PlatformJobStepWithContext,
        num_gpus=None,
        executor_resources: ComputeResources | None = None,
    ) -> JobUpdate:
        """Schedule a job as a Kubernetes Job."""

        job_name = name_for_step(step)

        # If the job is being resumed, first try unsuspending the job.
        # If it cannot be found, it means either the job was paused before being scheduled on the k8s cluster,
        # or the job was deleted out of band. In either case, we create a new job.
        if step.status == PlatformJobStatus.RESUMING:
            k8s_job: V1Job | None = self.get_job_by_name(job_name)
            if k8s_job is not None:
                self.resume_job(k8s_job)
                return JobUpdate(
                    status=PlatformJobStatus.PENDING,
                    status_details={"message": "Job resumed with Kubernetes Job backend"},
                )
            else:
                logger.warning(
                    "Kubernetes job not found for step, creating a new Kubernetes job",
                    extra={"job_name": job_name, "namespace": self.namespace},
                )

        # Create a configmap storing the job's step config
        configmap_name = create_configmap(
            core_v1=self._core_v1,
            namespace=self.namespace,
            step=step,
        )

        # Otherwise, create a new job
        common_labels = common_labels_for_step(step)
        common_labels[JOB_EXECUTION_BACKEND_LABEL] = self.BACKEND_NAME
        common_labels[JOB_EXECUTION_PROFILE_LABEL] = self._profile_name
        pod_template = create_pod_template_spec(
            step,
            self.namespace,
            container,
            common_labels,
            self.get_secrets_environment_variable_for_injection(step),
            configmap_name,
            self._execution_profile_config,
            self._core_v1,
            num_gpus=num_gpus,
            executor_resources=executor_resources,
        )

        # Create job spec
        job_spec = client.V1JobSpec(
            template=pod_template,
            backoff_limit=0,  # Don't restart failed jobs
            ttl_seconds_after_finished=self._execution_profile_config.ttl_seconds_after_finished,
        )

        # Create the job metadata
        job_metadata = build_metadata(
            labels=common_labels,
            metadata=self._execution_profile_config.job_metadata,
        )
        job_metadata.name = job_name

        # Create job object
        k8s_job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=job_metadata,
            spec=job_spec,
        )

        try:
            # Create the job in Kubernetes
            self._batch_v1.create_namespaced_job(namespace=self.namespace, body=k8s_job)
            logger.info(
                "Scheduled job step with Kubernetes job", extra={"job_name": job_name, "namespace": self.namespace}
            )
        except ApiException:
            logger.exception(
                "Failed to create Kubernetes job", extra={"job_name": job_name, "namespace": self.namespace}
            )
            raise

        return JobUpdate(
            status=PlatformJobStatus.PENDING,
            status_details={"message": "Job scheduled with Kubernetes Job backend"},
        )

    def _sync(self, step: PlatformJobStepWithContext) -> JobUpdate:
        """Sync job status from Kubernetes."""
        k8s_job: V1Job | None = self.get_job_by_name(name_for_step(step))

        if step.status == PlatformJobStatus.ACTIVE:
            if k8s_job is not None and (
                result := self.enforce_sync_ttl(
                    step, self._execution_profile_config.ttl_seconds_active, k8s_job, before_active=False
                )
            ):
                return result
            if k8s_job is not None and self.check_step_is_stale(step):
                update_all_tasks(self._nmp_sdk, self._core_v1, self.namespace, step)
                self.terminate_job(k8s_job)
                message = staleness_error_message(require_staleness_timeout_seconds(step))
                return JobUpdate(
                    status=PlatformJobStatus.ERROR,
                    status_details={
                        "message": message,
                        "events": self.get_kube_job_events(k8s_job),
                    },
                    error_details={
                        "message": message,
                    },
                )
            return self.sync_active(step, k8s_job)
        elif step.status == PlatformJobStatus.PENDING:
            if k8s_job is not None and (
                result := self.enforce_sync_ttl(
                    step,
                    self._execution_profile_config.ttl_seconds_before_active,
                    k8s_job,
                    before_active=True,
                )
            ):
                return result
            return self.sync_pending(step, k8s_job)
        elif step.status == PlatformJobStatus.CANCELLING:
            if k8s_job is None:
                # If the job doesn't exist, we consider it already cancelled since there are no resources to clean up.
                return JobUpdate(
                    status=PlatformJobStatus.CANCELLED,
                    status_details={"message": "Job is cancelled"},
                )
            return self.sync_terminate_job(step, k8s_job)
        elif step.status == PlatformJobStatus.PAUSING:
            # If the job doesn't exist, we can't suspend it, so we will
            # move it immediately to cancelled. When it is resumed, a new job will be created.
            # This can happen if a job is created and then immediately paused before it is scheduled on the k8s cluster.
            if k8s_job is None:
                return JobUpdate(
                    status=PlatformJobStatus.PAUSED,
                    status_details={"message": "Job is paused"},
                )
            return self.sync_suspend_job(step, k8s_job)
        else:
            raise ValueError(f"Unhandled job step status: {step.status}")

    def get_kube_job_events(self, job: V1Job) -> list[KubernetesEventDetails]:
        """Get events related to a Kubernetes Job."""
        job_name = job.metadata.name
        try:
            events_list = self._core_v1.list_namespaced_event(
                namespace=self.namespace,
                field_selector=build_event_field_selector(
                    kind="Job",
                    api_version="batch/v1",
                    name=job_name,
                ),
            )
            events = []
            for event in events_list.items:
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
            logger.debug(
                "Got events for Kubernetes Job",
                extra={"job_name": job_name, "namespace": self.namespace, "events": events},
            )
            return events
        except ApiException:
            logger.exception(
                "Failed to get events for Kubernetes Job", extra={"job_name": job_name, "namespace": self.namespace}
            )
            return []

    def create_step_update(self, step: PlatformJobStepWithContext, job: V1Job) -> JobUpdate:
        status, status_details = map_kubernetes_job_status_to_step_status(job, self._core_v1, step)
        error_message: str | None = None
        if status == PlatformJobStatus.ERROR:
            error_message = _status_details_message(status_details, "Job encountered an error")
        status_details["events"] = self.get_kube_job_events(job)
        task_has_error = update_all_tasks(self._nmp_sdk, self._core_v1, self.namespace, step)
        teardown_lifecycle_statuses = {
            PlatformJobStatus.PAUSING,
            PlatformJobStatus.PAUSED,
            PlatformJobStatus.CANCELLING,
            PlatformJobStatus.CANCELLED,
        }
        # Task-level errors can appear while Kubernetes is tearing down pods for
        # a requested pause or cancel. Preserve those user-requested lifecycle
        # states so the dispatcher can finish transitioning to PAUSED/CANCELLED.
        if task_has_error and status in teardown_lifecycle_statuses:
            logger.debug(
                "Task error observed during container teardown",
                extra={
                    "workspace": step.workspace,
                    "job": step.job,
                    "step": step.name,
                    "status": status,
                },
            )
        elif task_has_error:
            status = PlatformJobStatus.ERROR
            task_error_message = "One or more tasks are in error state"
            if error_message is None:
                error_message = task_error_message
            else:
                error_message = f"{error_message}; {task_error_message}"
        if self._workload_delegations.should_manage_step(step):
            target = self._workload_delegation_target_for_job(job)
            if target is not None:
                if status in PlatformJobStatus.terminals():
                    self._workload_delegations.revoke_for_target(target)
                else:
                    self._workload_delegations.ensure_for_target(step, target)
        error_details = {"message": error_message} if error_message is not None else {}
        return JobUpdate(status=status, status_details=status_details, error_details=error_details)

    def enforce_sync_ttl(
        self,
        step: PlatformJobStepWithContext,
        ttl_seconds: int,
        k8s_job: V1Job,
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
        self.terminate_job(k8s_job)
        status = PlatformJobStatus.ERROR
        status_details = {"message": f"Job timed out after reaching max TTL of {ttl_seconds} seconds"}
        error_details = {"message": f"Job timed out after reaching max TTL of {ttl_seconds} seconds"}
        status_details["events"] = self.get_kube_job_events(k8s_job)
        update_all_tasks(self._nmp_sdk, self._core_v1, self.namespace, step)
        return JobUpdate(status=status, status_details=status_details, error_details=error_details)

    def sync_active(self, step: PlatformJobStepWithContext, job: V1Job | None) -> JobUpdate:
        job_name = name_for_step(step)
        if job is None:
            logger.error("Job not found: %s", job_name)
            self._workload_delegations.revoke_by_key(namespace=self.namespace, name=job_name)
            # Job was deleted
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                error_details={"message": "Job not found when checking for active status"},
            )
        else:
            return self.create_step_update(step, job)

    def sync_pending(self, step: PlatformJobStepWithContext, job: V1Job | None) -> JobUpdate:
        if job is None:
            logger.error("Job not found: %s", name_for_step(step))
            # Job doesn't exist whien it should, so we consider it an error
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                error_details={"message": "Job not found when checking for pending status"},
            )
        else:
            return self.create_step_update(step, job)

    def sync_terminate_job(self, step: PlatformJobStepWithContext, job: V1Job | None) -> JobUpdate:
        if job is None:
            self._workload_delegations.revoke_by_key(namespace=self.namespace, name=name_for_step(step))
            # Job already deleted
            # List all the tasks on the step that are ACTIVE and mark them as CANCELLED too,
            # since at this point all those pods should be deleted.
            tasks = self._jobs.list_job_step_tasks(
                name=step.name,
                job=step.job,
                workspace=step.workspace,
            ).data()
            for task in tasks.data:
                if task.status == PlatformJobStatus.ACTIVE:
                    if task.name is None:
                        logger.warning(
                            "Skipping cancellation update for Kubernetes task with missing name",
                            extra={"workspace": step.workspace, "job": step.job, "step": step.name},
                        )
                        continue
                    self._jobs.update_job_step_task(
                        name=task.name,
                        workspace=step.workspace,
                        job=step.job,
                        step=step.name,
                        body=PlatformJobTaskUpdate(
                            status=PlatformJobStatus.CANCELLED,
                            status_details={"message": "Task cancelled as part of job cancellation"},
                        ),
                    )

            return JobUpdate(
                status=PlatformJobStatus.CANCELLED,
                status_details={"message": "Job is cancelled"},
            )
        else:
            self.terminate_job(job)
            return self.create_step_update(step, job)

    def sync_suspend_job(self, step: PlatformJobStepWithContext, job: V1Job | None) -> JobUpdate:
        if job is None:
            # Job not found
            return JobUpdate(
                status=PlatformJobStatus.ERROR,
                error_details={"message": "Job not found while suspending"},
            )

        if not self.is_suspended(job):
            self.suspend_job(job)
            # After suspending, get the updated job to pass for status updates
            job = self.get_job_by_name(name_for_step(step))
            if job is None:
                return JobUpdate(
                    status=PlatformJobStatus.ERROR,
                    error_details={"message": "Job not found after suspending"},
                )
        return self.create_step_update(step, job)

    def is_suspended(self, job: V1Job) -> bool:
        if job.spec and job.spec.suspend is not None:
            if job.spec.suspend:
                return True
        return False

    def suspend_job(self, job: V1Job):
        job_name = job.metadata.name
        logger.info("Suspending Kubernetes Job", extra={"job_name": job_name, "namespace": self.namespace})
        self._batch_v1.patch_namespaced_job(name=job_name, namespace=self.namespace, body={"spec": {"suspend": True}})

    def resume_job(self, job: V1Job):
        job_name = job.metadata.name
        logger.info("Resuming Kubernetes Job", extra={"job_name": job_name, "namespace": self.namespace})
        self._batch_v1.patch_namespaced_job(name=job_name, namespace=self.namespace, body={"spec": {"suspend": False}})

    def terminate_job(self, job: V1Job):
        job_name = job.metadata.name
        if job.metadata.labels.get(JOB_MANAGED_BY_LABEL) != JOB_MANAGED_BY_JOBS_CONTROLLER:  # type: ignore[union-attr]
            logger.warning(
                "Skipping delete of Kubernetes job (not managed by jobs-controller)",
                extra=dict(job_name=job_name, namespace=self.namespace),
            )
            return
        logger.info("Deleting Kubernetes Job", extra={"job_name": job_name, "namespace": self.namespace})
        target = self._workload_delegation_target_for_job(job)
        if target is not None:
            self._workload_delegations.revoke_for_target(target)
        try:
            self._batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground",
                ),
            )
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    "Kubernetes job not found during termination (already deleted or not managed by jobs-controller)",
                    extra=dict(job_name=job_name, namespace=self.namespace),
                )
            else:
                raise
        delete_configmap(
            core_v1=self._core_v1,
            namespace=self.namespace,
            name=job_name,
        )

    def cleanup_steps(self):
        jobs = self.get_kubernetes_job_list_by_labels(labels=KUBE_JOB_SELECTOR_LABELS)
        for job in jobs:
            # Cleanup completed jobs immediately if the config says to. Otherwise rely on k8s job TTL.
            if (
                self._execution_profile_config.cleanup_completed_jobs_immediately
                and job.status.completion_time is not None
            ):
                # Extract job_id and step_name from labels
                job_id = job.metadata.labels.get(JOB_ID_LABEL)
                step_name = job.metadata.labels.get(JOB_STEP_NAME_LABEL)
                workspace_id = job.metadata.labels.get(JOB_WORKSPACE_ID_LABEL)
                job_type = job.metadata.labels.get(JOB_TYPE_LABEL)

                # Cleanup jobs have similar labels but are of a different type, and cleanup on their own.
                if job_type is not None and job_type != JOB_TYPE_JOB:
                    continue

                # Skip jobs missing required labels (e.g. old or manually created); avoid calling SDK with None.
                if not job_id or not step_name or not workspace_id:
                    logger.warning(
                        "Skipping cleanup for Kubernetes job with missing labels (job_id, step_name, or workspace_id)",
                        extra={
                            "job_name": job.metadata.name,
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
                        "Skipping cleanup for kubernetes job as step is not in terminal state",
                        extra={
                            "job_id": job_id,
                            "step_name": step_name,
                            "workspace_id": workspace_id,
                        },
                    )
                    continue

                uses_persistent_storage = job.metadata.labels.get(JOB_USES_PERSISTENT_STORAGE_LABEL) == "true"
                if uses_persistent_storage and self._execution_profile_config.storage:
                    if self.check_job_persistent_storage_cleanup_allowed(
                        job=job_id, step_name=step_name, workspace=workspace_id
                    ):
                        logger.info(
                            "Cleaning up persistent storage for successful job",
                            extra={
                                "workspace_id": workspace_id,
                                "job_id": job_id,
                            },
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
                            extra={
                                "workspace_id": workspace_id,
                                "job_id": job_id,
                                "step_name": step_name,
                            },
                        )

                logger.debug(
                    "Garbage collecting kubernetes job",
                    extra={
                        "kubernetes_job_name": job.metadata.name,
                        "namespace": self.namespace,
                        "job_id": job_id,
                        "step_name": step_name,
                        "workspace_id": workspace_id,
                    },
                )
                self.terminate_job(job)


class CPUKubernetesJobBackend(KubernetesJobBackend[CPUExecutionProvider]):
    """Kubernetes job backend for CPU execution."""

    def schedule(
        self,
        executor_config: CPUExecutionProvider,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        return self.schedule_job(executor_config.container, step)

    def sync(
        self,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        return self._sync(step)


class GPUKubernetesJobBackend(KubernetesJobBackend[GPUExecutionProvider]):
    """Kubernetes job backend for GPU execution."""

    def schedule(
        self,
        executor_config: GPUExecutionProvider,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        if executor_config.resources is not None and executor_config.resources.num_gpus is not None:
            num_gpus = executor_config.resources.num_gpus
        elif self._execution_profile_config.num_gpus is not None:
            num_gpus = self._execution_profile_config.num_gpus
        else:
            num_gpus = 1

        return self.schedule_job(
            container=executor_config.container,
            step=step,
            num_gpus=num_gpus,
            executor_resources=executor_config.resources,
        )

    def sync(
        self,
        step: PlatformJobStepWithContext,
    ) -> JobUpdate:
        return self._sync(step)


def _status_details_message(status_details: KubernetesStatusDetails, default: str) -> str:
    message = status_details.get("message")
    if isinstance(message, str):
        return message
    return default


def _job_name(job: V1Job) -> str | None:
    metadata = job.metadata
    if metadata is None:
        return None
    return metadata.name


def _job_namespace(job: V1Job) -> str | None:
    metadata = job.metadata
    if metadata is None:
        return None
    return metadata.namespace


def map_kubernetes_job_status_to_step_status(
    job: V1Job, core_v1: client.CoreV1Api, step: PlatformJobStepWithContext
) -> tuple[PlatformJobStatus, KubernetesStatusDetails]:
    status: V1JobStatus = job.status

    is_cancelling = step.status == PlatformJobStatus.CANCELLING

    if status is None:
        raise ValueError("Kubernetes Job status is None")

    # Completed jobs always have a completion time set
    if status.completion_time is not None:
        # Job is completed
        return PlatformJobStatus.COMPLETED, {"message": f"Job has completed successfully at {status.completion_time}"}

    # Failed jobs will have at least one failure recorded.
    # We want to keep the job active until the job has fully terminated however.
    if status.failed is not None and status.failed > 0:
        # Check the conditions for failure condition and job has stopped
        if status.conditions is not None:
            for condition in status.conditions:
                if condition.type == "Failed" and condition.status == "True":
                    return PlatformJobStatus.ERROR, {"message": f"Job failed: {condition.message}"}
            cond_summary = [
                {"type": condition.type, "status": condition.status, "message": condition.message}
                for condition in status.conditions
            ]
            return PlatformJobStatus.ERROR, {
                "message": (f"Job reports {status.failed} failure(s) but no Failed condition is True yet"),
                "kubernetes_conditions": cond_summary,
            }
        else:
            # No conditions available, but job has failed. Something went wrong.
            raise ValueError("Kubernetes Job has failed but no conditions are available to determine failure reason")

    pods = list_pod_status(core_v1, job.metadata.namespace, common_labels_for_step(step))
    errored_pods = [pod for pod in pods if pod.errors]
    running_pods = [pod for pod in pods if pod.phase == "Running"]
    pending_pods = [pod for pod in pods if pod.phase == "Pending"]

    # Check suspended/cancelling state BEFORE errored pods. When K8s suspends or
    # cancels a Job it sends SIGTERM to running pods, which causes them to appear
    # as "errored" (non-zero exit code). These terminated pods are expected — they
    # must not cause the reconciler to report ERROR.
    is_suspended = job.spec and job.spec.suspend is not None and job.spec.suspend
    is_active = len(running_pods) > 0
    if is_cancelling:
        if is_active:
            return PlatformJobStatus.CANCELLING, {"message": "Job is being cancelled"}
        else:
            return PlatformJobStatus.CANCELLED, {"message": "Job is cancelled"}
    if is_suspended:
        if is_active:
            return PlatformJobStatus.PAUSING, {"message": "Job is being paused"}
        else:
            return PlatformJobStatus.PAUSED, {"message": "Job is paused"}

    if errored_pods:
        return PlatformJobStatus.ERROR, {"message": "Job has errored pods, check tasks for error details"}
    if is_active:
        return PlatformJobStatus.ACTIVE, {"message": f"Job is active with {len(running_pods)}/{len(pods)} running pods"}
    # In some cases after resuming from suspension, the job status may not reflect active pods immediately.
    # Check for pods owned by the job and if there are any active pods, consider the job active.
    # If the pods are pending, the job is  pending.
    # If there are no pods yet, the job is pending.
    if pending_pods:
        return PlatformJobStatus.PENDING, {
            "message": f"Job is pending with {len(pending_pods)}/{len(pods)} pending pods"
        }

    if not pods:
        logger.warning(
            "No pods yet for Job step; treating as pending",
            extra={
                "job_name": _job_name(job),
                "namespace": _job_namespace(job),
                "step_id": step.id,
            },
        )
        return PlatformJobStatus.PENDING, {"message": "Waiting for pods to be created for this job"}

    mapped_status, details = aggregate_pod_statuses_for_job_step(pods)
    if mapped_status == PlatformJobStatus.PENDING:
        logger.warning(
            "Job step status inferred as pending from pod states without Running/Pending phases",
            extra={
                "job_name": _job_name(job),
                "namespace": _job_namespace(job),
                "step_id": step.id,
                "job_status_summary": {
                    "active": status.active,
                    "succeeded": status.succeeded,
                    "failed": status.failed,
                },
                "pod_phases": [p.phase for p in pods],
            },
        )
    elif mapped_status == PlatformJobStatus.COMPLETED:
        logger.info(
            "Job pods report success before batch Job completion_time was set",
            extra={
                "job_name": _job_name(job),
                "namespace": _job_namespace(job),
                "step_id": step.id,
            },
        )
    return mapped_status, details
