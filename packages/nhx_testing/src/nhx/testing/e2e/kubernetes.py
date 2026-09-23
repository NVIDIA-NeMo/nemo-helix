# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes-based E2E test backend using K3s testcontainers.

This backend starts a K3s cluster via testcontainers but does NOT deploy the
NeMo Helix stack. Deployment is handled separately (e.g., via Helm chart), and the
base URL must be set after deployment before the SDK can be used.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import IO, TYPE_CHECKING

from nemo_helix import NeMoHelix
from testcontainers.k3s import K3SContainer

from .base import E2EBackend

if TYPE_CHECKING:
    from .config import E2EConfig

logger = logging.getLogger(__name__)


class Kubernetes(E2EBackend):
    """Kubernetes-based test backend using K3s.

    This backend starts the cluster and provides access to it.
    NeMo Helix stack deployment is handled separately (e.g., via Helm chart).
    Use self.config_path and self.image for Helm values.

    Args:
        config_path: Path to the NeMo Helix configuration YAML file, or an E2EConfig object.
        **kwargs: Additional arguments passed to E2EBackend (registry, tag).

    Usage:
        with Kubernetes(config_path="e2e/configs/k8s.yaml") as backend:
            # Get kubeconfig for kubectl/helm
            kubeconfig_path = backend.get_kubeconfig_path()

            # Deploy NeMo Helix via Helm using backend.config_path
            # helm install nhx charts/nhx --kubeconfig $kubeconfig_path ...

            # Set the NeMo Helix API URL after deployment
            backend.set_base_url("http://localhost:8080")

            # Now you can get the SDK
            sdk = backend.get_sdk()
    """

    def __init__(self, config_path: str | Path | E2EConfig, **kwargs):
        super().__init__(config_path=config_path, **kwargs)
        self.k3s: K3SContainer | None = None
        self._base_url: str | None = None
        self._kubeconfig_tmpfile: IO[str] | None = None

    def start(self) -> None:
        """Start K3s cluster."""
        logger.info("Starting K3s cluster...")
        self.k3s = K3SContainer()
        self.k3s.start()
        logger.info("K3s cluster started")

    def stop(self) -> None:
        """Stop K3s cluster and cleanup resources."""
        if self._kubeconfig_tmpfile:
            try:
                self._kubeconfig_tmpfile.close()
            except Exception as e:
                logger.warning(f"Error closing kubeconfig temp file: {e}")
            self._kubeconfig_tmpfile = None

        if self.k3s:
            try:
                self.k3s.stop()
            except Exception as e:
                logger.warning(f"Error stopping K3s cluster: {e}")
            self.k3s = None

        self._base_url = None

    def get_kubeconfig(self) -> str:
        """Get kubeconfig content for kubectl/helm access.

        Returns:
            YAML string containing the kubeconfig.

        Raises:
            RuntimeError: If the cluster is not running.
        """
        if self.k3s is None:
            raise RuntimeError("K3s cluster is not running. Call start() first.")
        return self.k3s.config_yaml()

    def get_kubeconfig_path(self) -> str:
        """Write kubeconfig to temp file and return path.

        The temp file is managed by this backend and will be cleaned up
        when stop() is called.

        Returns:
            Path to the temporary kubeconfig file.

        Raises:
            RuntimeError: If the cluster is not running.
        """
        if self.k3s is None:
            raise RuntimeError("K3s cluster is not running. Call start() first.")

        if self._kubeconfig_tmpfile is None:
            self._kubeconfig_tmpfile = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                delete=False,
            )
            self._kubeconfig_tmpfile.write(self.get_kubeconfig())
            self._kubeconfig_tmpfile.flush()

        return self._kubeconfig_tmpfile.name

    def set_base_url(self, url: str) -> None:
        """Set the NeMo Helix API base URL after Helm deployment.

        This must be called after deploying NeMo Helix to the cluster via Helm
        and before calling get_sdk().

        Args:
            url: The base URL of the deployed NeMo Helix API
                 (e.g., "http://localhost:8080" after port-forwarding).
        """
        self._base_url = url

    def get_sdk(self, principal_id: str | None = None) -> NeMoHelix:
        """Get SDK client. Must call set_base_url() after Helm deploy.

        Args:
            principal_id: Optional principal ID for authentication (X-NHX-Principal-Id header).

        Returns:
            Configured NeMoHelix SDK client.

        Raises:
            RuntimeError: If base URL is not set (NeMo Helix not deployed).
        """
        if self._base_url is None:
            raise RuntimeError("Base URL not set. Deploy NeMo Helix via Helm first, then call set_base_url()")
        headers = {"X-NHX-Principal-Id": principal_id} if principal_id else None
        return NeMoHelix(base_url=self._base_url, default_headers=headers)

    @property
    def base_url(self) -> str | None:
        """Get the base URL (if set).

        Returns:
            The base URL or None if not yet set.
        """
        return self._base_url
