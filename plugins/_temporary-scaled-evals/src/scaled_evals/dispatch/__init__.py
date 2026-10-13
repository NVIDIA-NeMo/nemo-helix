# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation dispatch.

Turns a queued evaluation row into a running sandbox by handing it to a
pluggable :class:`~scaled_evals.dispatch.runtime_backend.RuntimeBackend`. The
control-plane never talks to a cluster directly — every cluster-specific
call lives behind the backend interface, which keeps dispatch unit-testable
with an injected fake.

Layout: :mod:`scaled_evals.dispatch.runtime_backend` holds the generic contract
(the ``RuntimeBackend`` protocol and ``LaunchSpec``/``LaunchHandle``/
``RuntimeStatus``); concrete backends live in their own modules and expose
``register_runtime_backends`` plugin hooks. The registry always loads
:mod:`scaled_evals.dispatch.sandbox_k8s` (agent-sandbox / Harbor on K8s) through
that hook, then loads additional plugins such as
:mod:`scaled_evals.dispatch.gym.plugin`.
:mod:`scaled_evals.dispatch.registry` maps ``runtime`` to backend factories and
runtime capabilities. :mod:`scaled_evals.dispatch.worker` owns the generic
dispatch lifecycle.

Trigger model: ``POST /evaluations`` durably inserts ``status='queued'`` and
returns. The API submits a Platform Job for the row (the Jobs controller
resubmits it if that fails), and the Job calls :class:`Dispatcher` to launch,
poll, resume, and write terminal status.
"""

from scaled_evals.dispatch.gym import (
    GymDaytonaBackend,
    GymSandboxDaytonaBackend,
    GymSandboxOpenSandboxBackend,
    build_gym_argv,
    build_gym_daytona_backend,
    build_gym_sandbox_daytona_backend,
    build_gym_sandbox_opensandbox_backend,
    make_gym_daytona_submitter,
    make_gym_sandbox_daytona_submitter,
    make_gym_sandbox_opensandbox_submitter,
)
from scaled_evals.dispatch.registry import (
    RuntimeBackendRegistry,
    build_runtime_backend_registry,
    get_backend,
    get_backend_capabilities,
    load_runtime_backend_plugin,
    load_runtime_backend_plugins,
    registered_runtime_names,
    validate_backend_config,
)
from scaled_evals.dispatch.runtime_backend import (
    CallableRuntimeBackend,
    LaunchHandle,
    LaunchSpec,
    ResultSummary,
    RuntimeBackend,
    RuntimeBackendCapabilities,
    RuntimeBackendRegistration,
    RuntimeStatus,
)
from scaled_evals.dispatch.sandbox_k8s import (
    SandboxK8sBackend,
    build_backend,
    make_sandbox_k8s_status_reader,
    make_sandbox_k8s_submitter,
    render_harbor_config,
    summarize_harbor_result,
)
from scaled_evals.dispatch.sandbox_k8s import (
    build_backend as build_sandbox_k8s_backend,
)
from scaled_evals.dispatch.worker import Dispatcher, get_dispatcher

__all__ = [
    "CallableRuntimeBackend",
    "Dispatcher",
    "GymDaytonaBackend",
    "GymSandboxDaytonaBackend",
    "GymSandboxOpenSandboxBackend",
    "LaunchHandle",
    "LaunchSpec",
    "ResultSummary",
    "RuntimeBackend",
    "RuntimeBackendCapabilities",
    "RuntimeBackendRegistration",
    "RuntimeBackendRegistry",
    "RuntimeStatus",
    "SandboxK8sBackend",
    "build_gym_argv",
    "build_gym_daytona_backend",
    "build_runtime_backend_registry",
    "build_gym_sandbox_daytona_backend",
    "build_gym_sandbox_opensandbox_backend",
    "build_sandbox_k8s_backend",
    "build_backend",
    "get_backend",
    "get_backend_capabilities",
    "get_dispatcher",
    "load_runtime_backend_plugin",
    "load_runtime_backend_plugins",
    "make_gym_daytona_submitter",
    "make_gym_sandbox_daytona_submitter",
    "make_gym_sandbox_opensandbox_submitter",
    "make_sandbox_k8s_status_reader",
    "make_sandbox_k8s_submitter",
    "registered_runtime_names",
    "render_harbor_config",
    "summarize_harbor_result",
    "validate_backend_config",
]
