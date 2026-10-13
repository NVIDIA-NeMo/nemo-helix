# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for the deployments plugin."""

MANAGED_BY_LABEL = "nemo-deployments"
"""Label value backends use to tag backend resources for orphan cleanup."""

ENTITY_TYPE_DEPLOYMENT_CONFIG = "deployments_deployment_config"
ENTITY_TYPE_DEPLOYMENT = "deployments_deployment"
ENTITY_TYPE_VOLUME = "deployments_volume"

DEFAULT_JOB_TTL_SECONDS_AFTER_FINISHED = 60
"""Default ttlSecondsAfterFinished for finite (Never/OnFailure) Jobs such as the
weight-puller. Completed puller pods hold their ReadWriteOnce weights volume
attachment until reaped; without a TTL the cluster default (effectively never)
leaves the pod lingering, so the serving Deployment cannot mount the RWO volume
and fails with MultiAttachError. The serving Deployment starts only after the
puller Job reaches SUCCEEDED (Prerequisite condition="succeeded"), so reaping a
completed puller promptly is safe. 60s clears the attachment far faster than the
cluster default while leaving the reconciler ample time to observe completion and
read the pod exit code before the pod is garbage-collected."""

MIN_JOB_TTL_SECONDS_AFTER_FINISHED = 10
"""Lower bound for a *set* ttlSecondsAfterFinished. The reconciler observes a
finite Job's completion + pod exit code via a status read shortly after the Job
finishes; if the TTL reaps the Job (and its pod) before that read lands, the read
404s and the puller is recorded as FAILED (or SUCCEEDED without an exit code),
stalling the puller->server prerequisite. A very small or zero TTL makes that race
likely, so a set TTL must be >= this floor. ``None`` still opts out entirely
(defer to the cluster default); the floor only constrains explicit values."""
