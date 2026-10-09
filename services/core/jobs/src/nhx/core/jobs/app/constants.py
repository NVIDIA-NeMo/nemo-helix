# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Constants
JOB_WORKSPACE_ID_LABEL = "nhx.nvidia.com/job_workspace_id"
JOB_ID_LABEL = "nhx.nvidia.com/job_id"
JOB_ATTEMPT_ID_LABEL = "nhx.nvidia.com/job_attempt_id"
JOB_STEP_NAME_LABEL = "nhx.nvidia.com/job_step_name"
JOB_STEP_ID_LABEL = "nhx.nvidia.com/job_step_id"
JOB_TASK_ID_LABEL = "nhx.nvidia.com/job_task_id"
JOB_USES_PERSISTENT_STORAGE_LABEL = "nhx.nvidia.com/uses_persistent_storage"
JOB_CONTROLLER_INSTANCE_ID_LABEL = "nhx.nvidia.com/jobs_controller_instance_id"

JOB_TYPE_LABEL = "nhx.nvidia.com/job_type"
JOB_TYPE_JOB = "job"
JOB_TYPE_STORAGE_CLEANUP = "storage-cleanup"

JOB_MANAGED_BY_LABEL = "nhx.nvidia.com/managed_by"
JOB_MANAGED_BY_JOBS_CONTROLLER = "jobs-controller"

JOB_EXECUTION_BACKEND_LABEL = "nhx.nvidia.com/job_execution_backend"
JOB_EXECUTION_PROFILE_LABEL = "nhx.nvidia.com/job_execution_profile"

NEMO_JOB_TASK_CONTAINER_NAME = "nemo-job-task"
DEFAULT_VOLUME_PERMISSIONS_IMAGE = "docker.io/library/busybox:stable"

KUBE_JOB_SELECTOR_LABELS = {
    "app": "nemo-job",
    JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
}

JOB_MULTINODE_NETWORKING_ANNOTATION = "nhx.nvidia.com/enable-multi-node-networking"
JOB_NUM_NODES_ANNOTATION = "nhx.nvidia.com/num-nodes"

PAUSE_REQUESTED_AT = "pause_requested_at"
RESUMED_AT = "resumed_at"
STOPPED_AT = "stopped_at"
STORAGE_RECLAIMED_AT = "storage_reclaimed_at"
IMAGE_DIGEST = "image_digest"
IMAGE_DIGEST_AT_SAVE = "image_digest_at_save"
IMAGE_DIGEST_WARNING = "image_digest_warning"
IMAGE_DIGEST_COMPARISON_SKIPPED = "image_digest_comparison_skipped"
RERUN_WARNING = "rerun_warning"
RESUMABLE = "resumable"
NON_RESUMABLE_REASON = "non_resumable_reason"

# sysexits EX_TEMPFAIL. The training process exits with this after the pause checkpoint is written.
PAUSE_EXIT_CODE = 75

# Progress copied onto a rerun. Pod status, pause clocks, and storage flags stay on the failed attempt.
RERUN_EXCLUDED_STATUS_DETAILS = frozenset(
    {
        RESUMABLE,
        NON_RESUMABLE_REASON,
        STORAGE_RECLAIMED_AT,
        PAUSE_REQUESTED_AT,
        RESUMED_AT,
        STOPPED_AT,
        IMAGE_DIGEST,
        IMAGE_DIGEST_WARNING,
        IMAGE_DIGEST_COMPARISON_SKIPPED,
        RERUN_WARNING,
        "events",
        "containers",
        "conditions",
        "exit_code",
        "message",
        "pid",
        "pgid",
        "subprocess_work_dir",
        "subprocess_persistent_storage_path",
        "name",
        "restart_count",
        "ready",
        "error",
    }
)
POD_PHASES = frozenset({"Pending", "Running", "Succeeded", "Failed", "Unknown"})
