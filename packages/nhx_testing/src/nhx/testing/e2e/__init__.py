# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E test harness for NeMo Helix.

This package provides backends for running E2E tests against the NeMo Helix
using either Docker containers or Kubernetes.

Usage:
    from nhx.testing.e2e import Docker, Kubernetes, E2EBackend

    # Docker backend (in-memory)
    with Docker() as backend:
        client = backend.get_client()
        # Run tests...

    # Kubernetes backend (K3s cluster)
    with Kubernetes() as backend:
        kubeconfig = backend.get_kubeconfig_path()
        # Deploy NeMo Helix via Helm, then:
        backend.set_base_url("http://localhost:8080")
        client = backend.get_client()
        # Run tests...

Config-driven approach:
    from nhx.testing.e2e.config import E2EConfig, discover_configs, load_config

    # Discover all configs in e2e/configs/
    configs = discover_configs(Path("e2e/configs"))

    # Load a config (file used as-is; platform applies merge at runtime)
    config = load_config(Path("e2e/configs/docker.yaml"), repo_root)
"""

from .base import DEFAULT_REGISTRY, DEFAULT_TAG, E2EBackend
from .config import (
    AUTH_QUICKSTART_CONFIG,
    DEFAULT_QUICKSTART_CONFIG,
    E2EConfig,
    deep_merge,
    discover_configs,
    infer_backend,
    load_config,
)
from .docker import Docker
from .jobs import cleanup_platform_job, wait_for_job_logs, wait_for_platform_job
from .kubernetes import Kubernetes

__all__ = [
    "AUTH_QUICKSTART_CONFIG",
    "DEFAULT_QUICKSTART_CONFIG",
    "DEFAULT_REGISTRY",
    "DEFAULT_TAG",
    "E2EBackend",
    "E2EConfig",
    "Docker",
    "Kubernetes",
    "deep_merge",
    "discover_configs",
    "infer_backend",
    "load_config",
    "cleanup_platform_job",
    "wait_for_job_logs",
    "wait_for_platform_job",
]
