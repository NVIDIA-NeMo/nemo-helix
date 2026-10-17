# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration shared by the custom Gym environment implementation modules.

``run.py`` creates one ``Settings`` object from documented environment
variables and passes it into ``workflow.py``. The dataclasses here centralize
cluster names, temporary Platform resource names, and evidence paths so those
values are derived consistently throughout a run.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ENVIRONMENT_PREFIX = "NMP_GYM_CUSTOM_"


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem locations generated during one workflow run."""

    root: Path
    evidence: Path
    converter_environment: Path
    converter_dataset: Path
    environment: Path
    dataset: Path

    @classmethod
    def create(cls, root: Path) -> RunPaths:
        """Derive every generated path from one run directory."""
        return cls(
            root=root,
            evidence=root / "evidence",
            converter_environment=root / "converter-environment",
            converter_dataset=root / "converter-dataset",
            environment=root / "environment",
            dataset=root / "dataset.jsonl",
        )


@dataclass(frozen=True, slots=True)
class Settings:
    """Workflow inputs, cluster prerequisites, and run-scoped resource names."""

    repo_root: Path
    asset_root: Path
    paths: RunPaths
    environment_source: Path | None
    dataset_source: Path | None
    requested_resources_server: str | None
    namespace: str
    release: str
    workspace: str
    job_pvc: str
    registry_secret: str
    opensandbox_namespace: str
    opensandbox_service: str
    opensandbox_api_secret: str
    platform_api_service: str
    local_port: int
    model_entity_id: str
    fileset: str
    inference_secret: str
    inference_provider: str

    @classmethod
    def from_environment(
        cls,
        *,
        run_dir: Path | None = None,
        environment_dir: Path | None = None,
        dataset: Path | None = None,
        resources_server: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> Settings:
        """Resolve CLI inputs, environment overrides, and unique run resources.

        Custom environments require both a package directory and dataset. When
        neither is provided, later preparation stages use the bundled example.
        """
        if (environment_dir is None) != (dataset is None):
            raise ValueError("--environment-dir and --dataset must be supplied together")
        if resources_server is not None and not resources_server.strip():
            raise ValueError("--resources-server cannot be empty")

        env = dict(os.environ if environment is None else environment)

        def override(name: str, default: str) -> str:
            return env.get(f"{ENVIRONMENT_PREFIX}{name}", default)

        # Derive repository paths from this file so the script works from any
        # current working directory.
        asset_root = Path(__file__).resolve().parent
        repo_root = asset_root.parents[3]

        # One identifier ties together local evidence and every temporary
        # Platform resource, which also makes cleanup ownership unambiguous.
        run_id = f"{datetime.now(UTC):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
        default_run_dir = Path(f"/tmp/nmp-gym-custom-environment-{run_id}")
        workspace = override("WORKSPACE", "default")

        return cls(
            repo_root=repo_root,
            asset_root=asset_root,
            paths=RunPaths.create((run_dir or default_run_dir).resolve()),
            environment_source=environment_dir.resolve() if environment_dir is not None else None,
            dataset_source=dataset.resolve() if dataset is not None else None,
            requested_resources_server=resources_server,
            namespace=override("NAMESPACE", "nmp-temp1"),
            release=override("RELEASE", "nemo-platform"),
            workspace=workspace,
            job_pvc=override("JOB_PVC", "nemo-platform-core-storage"),
            registry_secret=override("REGISTRY_SECRET", "nvcrimagepullsecret"),
            opensandbox_namespace=override("OPENSANDBOX_NAMESPACE", "opensandbox-system"),
            opensandbox_service=override("OPENSANDBOX_SERVICE", "opensandbox-server-crun"),
            opensandbox_api_secret=override("OPENSANDBOX_API_SECRET", "opensandbox-server-crun-api-key"),
            platform_api_service=override("PLATFORM_API_SERVICE", "nemo-platform-api"),
            local_port=int(override("LOCAL_PORT", "18080")),
            model_entity_id=override(
                "MODEL_ENTITY_ID",
                f"{workspace}/nvidia-meta-llama-3-3-70b-instruct",
            ),
            fileset=f"gym-custom-environment-{run_id}",
            inference_secret=f"gym-custom-inference-key-{run_id}",
            inference_provider=f"gym-custom-inference-hub-{run_id}",
        )

    @property
    def base_url(self) -> str:
        """Return the workstation URL for the temporary API port-forward."""
        return f"http://127.0.0.1:{self.local_port}"

    @property
    def internal_api(self) -> str:
        """Return the Platform API URL reachable from sandbox pods."""
        return f"http://{self.platform_api_service}.{self.namespace}.svc.cluster.local:8080"
