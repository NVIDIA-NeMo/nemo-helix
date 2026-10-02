# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution provider types for the Jobs service.

The definitions now live in :mod:`nemo_helix_plugin.jobs.providers` so that
both the server and the typed HTTP client (``JobsClient``) share one source of
truth.  This module re-exports them for backward compatibility.
"""

from nemo_helix_plugin.jobs import providers as _providers

ComputeResources = _providers.ComputeResources
ComputeResourceSpec = _providers.ComputeResourceSpec
ContainerSpec = _providers.ContainerSpec
CPUExecutionProvider = _providers.CPUExecutionProvider
DistributedGPUExecutionProvider = _providers.DistributedGPUExecutionProvider
ExecutionProviderT = _providers.ExecutionProviderT
GPUExecutionProvider = _providers.GPUExecutionProvider
Provider = _providers.Provider
SubprocessExecutionProvider = _providers.SubprocessExecutionProvider
TaskSpec = _providers.TaskSpec
