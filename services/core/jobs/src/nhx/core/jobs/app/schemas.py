# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared schemas for the Jobs service.

The job specification types now live in :mod:`nemo_helix_plugin.jobs.spec`
and the ``BaseExecutionProfile`` base in the same package, so that both the
server and the typed HTTP client (``JobsClient``) share one source of truth.
This module re-exports them for backward compatibility.
"""

from nemo_helix_plugin.jobs import spec as _spec

BackendRef = _spec.BackendRef
BaseExecutionProfile = _spec.BaseExecutionProfile
HelixJobEnvironmentVariable = _spec.HelixJobEnvironmentVariable
HelixJobSecret = _spec.HelixJobSecret
HelixJobSecretEnvironmentVariableRef = _spec.HelixJobSecretEnvironmentVariableRef
HelixJobSpec = _spec.HelixJobSpec
HelixJobStepSpec = _spec.HelixJobStepSpec
ProfileRef = _spec.ProfileRef
ProviderRef = _spec.ProviderRef
StepLifecycle = _spec.StepLifecycle
