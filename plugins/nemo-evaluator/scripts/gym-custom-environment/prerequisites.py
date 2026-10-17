# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only prerequisite checks used by ``workflow.py``.

``PrerequisiteChecker`` validates local tools, Kubernetes resources, active
Platform configuration, deployed image references, and registry visibility
before the workflow creates anything. Keeping those checks here makes their
read-only boundary explicit and keeps the lifecycle orchestration readable.
This module is an implementation detail; developers invoke ``run.py``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any

import yaml
from artifacts import write_json
from commands import CommandRunner
from config import Settings
from console import Console


class PrerequisiteError(RuntimeError):
    """Raised when the existing workstation or cluster is not ready."""


def require_prerequisite(condition: bool, message: str) -> None:
    """Raise a focused prerequisite error when a required condition is false."""
    if not condition:
        raise PrerequisiteError(message)


@dataclass(frozen=True, slots=True)
class DeploymentImages:
    """Release-matched images required by the evaluation workflow."""

    registry: str
    tag: str

    @property
    def api(self) -> str:
        """Return the API/controller image reference."""
        return f"{self.registry}/nmp-api:{self.tag}"

    @property
    def cpu_tasks(self) -> str:
        """Return the Evaluator task image reference."""
        return f"{self.registry}/nmp-cpu-tasks:{self.tag}"

    @property
    def gym_host(self) -> str:
        """Return the OpenSandbox Gym runtime image reference."""
        return f"{self.registry}/nmp-gym-host:{self.tag}"


class PrerequisiteChecker:
    """Validate the preconfigured cluster and published images without changing them."""

    def __init__(
        self,
        settings: Settings,
        runner: CommandRunner,
        console: Console,
    ) -> None:
        """Store workflow collaborators used by every prerequisite check."""
        self.settings = settings
        self.runner = runner
        self.console = console

    def _kubectl(self, *arguments: str) -> str:
        """Run a read-only kubectl command and return stdout."""
        return self.runner.run(["kubectl", *arguments])

    def _load_platform_config(self) -> dict[str, Any]:
        """Load the active Platform YAML configuration from its ConfigMap."""
        rendered_config = self._kubectl(
            "get",
            "configmap",
            "nemo-platform-config",
            "-n",
            self.settings.namespace,
            "-o",
            r"jsonpath={.data.config\.yaml}",
        )
        config = yaml.safe_load(rendered_config)
        require_prerequisite(
            isinstance(config, dict),
            "Platform ConfigMap does not contain a YAML mapping",
        )
        return config

    def _check_workstation_tools(self) -> None:
        """Require each command-line tool used later in the workflow."""
        for command in ("docker", "helm", "kubectl", "uv"):
            require_prerequisite(
                shutil.which(command) is not None,
                f"required command not found: {command}",
            )

    def _check_cluster_resources(self) -> None:
        """Require the installed release and cluster resources used by the job."""
        # Confirm the selected namespace contains the expected Helm release.
        self._kubectl("get", "namespace", self.settings.namespace)
        self.runner.run(
            [
                "helm",
                "status",
                self.settings.release,
                "-n",
                self.settings.namespace,
                "-o",
                "json",
            ],
            output_path=self.settings.paths.evidence / "helm-status.json",
        )

        # The evaluator reaches OpenSandbox through its service and authenticates
        # with a secret copied into the Platform namespace.
        self._kubectl(
            "get",
            "service",
            self.settings.opensandbox_service,
            "-n",
            self.settings.opensandbox_namespace,
        )
        self._kubectl(
            "get",
            "secret",
            self.settings.opensandbox_api_secret,
            "-n",
            self.settings.namespace,
        )

        # Sandbox pods need the same registry credentials as the Platform jobs.
        self._kubectl(
            "get",
            "secret",
            self.settings.registry_secret,
            "-n",
            self.settings.namespace,
        )

        # Environment staging and sandbox execution share this volume, so the
        # claim must support simultaneous mounts across pods.
        persistent_volume_claim = self.runner.run_json(
            [
                "kubectl",
                "get",
                "pvc",
                self.settings.job_pvc,
                "-n",
                self.settings.namespace,
                "-o",
                "json",
            ]
        )
        access_modes = persistent_volume_claim.get("spec", {}).get("accessModes", [])
        require_prerequisite(
            "ReadWriteMany" in access_modes,
            f"PVC {self.settings.job_pvc} must support ReadWriteMany",
        )
        write_json(
            self.settings.paths.evidence / "job-pvc.json",
            persistent_volume_claim,
        )

    def _resolve_images(self, config: dict[str, Any]) -> DeploymentImages:
        """Resolve and validate the immutable image release from Platform config."""
        platform_config = config.get("platform", {})
        image_registry = platform_config.get("image_registry")
        image_tag = platform_config.get("image_tag")
        require_prerequisite(
            isinstance(image_registry, str) and bool(image_registry),
            "platform.image_registry is missing",
        )
        require_prerequisite(
            isinstance(image_tag, str) and bool(image_tag),
            "platform.image_tag is missing",
        )
        require_prerequisite(
            image_tag != "latest",
            "platform.image_tag must be immutable; latest is not supported",
        )
        return DeploymentImages(registry=image_registry, tag=image_tag)

    def _check_sandbox_config(
        self,
        config: dict[str, Any],
        images: DeploymentImages,
    ) -> None:
        """Require Platform and Evaluator settings needed by sandboxed Gym."""
        platform_config = config.get("platform", {})
        evaluator_config = config.get("evaluator", {})

        # First verify that sandboxing is enabled end to end.
        require_prerequisite(
            platform_config.get("sandbox_cluster_capable") is True,
            "Platform sandboxing is disabled",
        )
        require_prerequisite(
            evaluator_config.get("sandbox_cluster_capable") is True,
            "Evaluator sandboxing is disabled",
        )
        require_prerequisite(
            evaluator_config.get("sandboxed_gym_default") is True,
            "sandboxed Gym is not the default",
        )
        require_prerequisite(
            evaluator_config.get("sandbox_host_provider") == "opensandbox",
            "Evaluator is not configured to use OpenSandbox",
        )

        # An explicit runtime image can silently run stale code instead of the
        # release-matched image.
        require_prerequisite(
            not evaluator_config.get("sandbox_runtime_image"),
            "evaluator.sandbox_runtime_image must be absent or empty",
        )

        # The configured storage and network policy must match resources used by
        # this workflow, otherwise environment staging or inference will fail.
        require_prerequisite(
            evaluator_config.get("sandbox_job_storage_pvc_claim") == self.settings.job_pvc,
            "Evaluator uses the wrong sandbox storage PVC",
        )
        require_prerequisite(
            self.settings.internal_api in evaluator_config.get("sandbox_policy_base_urls", []),
            "the internal Platform API is absent from sandbox_policy_base_urls",
        )

        selected_config = {
            "platform": {
                "image_registry": images.registry,
                "image_tag": images.tag,
                "sandbox_cluster_capable": platform_config.get("sandbox_cluster_capable"),
            },
            "evaluator": {
                key: evaluator_config.get(key)
                for key in (
                    "sandbox_cluster_capable",
                    "sandboxed_gym_default",
                    "sandbox_host_provider",
                    "sandbox_runtime_image",
                    "sandbox_job_storage_pvc_claim",
                    "sandbox_policy_base_urls",
                )
            },
        }
        write_json(self.settings.paths.evidence / "platform-config.json", selected_config)

    def _check_deployment_images(self, images: DeploymentImages) -> None:
        """Require API and controller deployments to use the configured API image."""
        deployments = self.runner.run_json(
            [
                "kubectl",
                "get",
                "deployments",
                "-n",
                self.settings.namespace,
                "-l",
                f"app.kubernetes.io/instance={self.settings.release}",
                "-o",
                "json",
            ]
        )
        deployed_api_images = [
            container["image"]
            for deployment in deployments.get("items", [])
            for container in deployment["spec"]["template"]["spec"]["containers"]
            if "/nmp-api:" in container["image"]
        ]
        require_prerequisite(
            bool(deployed_api_images),
            "no nmp-api API/controller deployment was found",
        )
        require_prerequisite(
            all(image == images.api for image in deployed_api_images),
            "an API/controller deployment uses an image that differs from Platform config",
        )
        write_json(self.settings.paths.evidence / "deployments.json", deployments)

    def _check_published_images(self, images: DeploymentImages) -> None:
        """Require all release-matched image manifests to be readable locally."""
        for image_name, image_reference in (
            ("nmp-api", images.api),
            ("nmp-cpu-tasks", images.cpu_tasks),
            ("nmp-gym-host", images.gym_host),
        ):
            self.runner.run(
                ["docker", "buildx", "imagetools", "inspect", image_reference],
                output_path=self.settings.paths.evidence / f"{image_name}-image.txt",
            )

    def check(self) -> DeploymentImages:
        """Fail fast through local, cluster, config, deployment, and image checks."""
        self.settings.paths.evidence.mkdir(parents=True, exist_ok=True)

        self._check_workstation_tools()
        self._check_cluster_resources()

        platform_config = self._load_platform_config()
        images = self._resolve_images(platform_config)

        self._check_sandbox_config(platform_config, images)
        self._check_deployment_images(images)
        self._check_published_images(images)

        self.console.detail("Image registry", images.registry)
        self.console.detail("Image tag", images.tag)
        self.console.detail("Platform namespace", self.settings.namespace)
        return images
