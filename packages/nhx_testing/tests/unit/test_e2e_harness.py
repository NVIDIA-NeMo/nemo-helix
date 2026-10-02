# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the local pytest E2E harness helpers."""

from pathlib import Path
from typing import Any, cast

import e2e.services_pool as services_pool


def test_e2e_services_env_sets_isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NHX_DATA_DIR", "/shell/value/should/not/leak")

    config_path = tmp_path / "platform.yaml"
    data_dir = tmp_path / "isolated-data"

    env = services_pool.e2e_services_env(config_path, data_dir)

    assert env["NHX_CONFIG_FILE_PATH"] == str(config_path)
    assert env["NHX_DATA_DIR"] == str(data_dir)
    assert env["NHX_SEED_ON_STARTUP"] == "true"
    assert env["NHX_INFERENCE_GATEWAY_MOCK_PROVIDER_PREFIX"] == "igw-mock-"
    assert env["NHX_CONFIG_WARNINGS_DISABLED"] == "1"


def test_e2e_services_data_dir_is_stable_per_hash(tmp_path):
    log_dir = tmp_path / "logs"

    path = services_pool.e2e_services_data_dir(log_dir, "abc123def456")

    assert path == Path(tmp_path / "logs-data" / "data-abc123def456")
    assert not path.is_relative_to(log_dir)


def test_with_e2e_instance_paths_namespaces_local_filesystem_paths(tmp_path):
    data_dir = tmp_path / "data-abc123def456"
    config_data: dict[str, Any] = {
        "jobs": {
            "executors": [
                {
                    "provider": "subprocess",
                    "profile": "default",
                    "backend": "subprocess",
                    "config": {"working_directory": ".tmp/e2e/subprocess-jobs"},
                }
            ],
            "executor_defaults": {
                "subprocess": {"working_directory": ".tmp/e2e/subprocess-jobs"},
            },
        },
        "files": {
            "default_storage_config": {
                "type": "local",
                "path": ".tmp/e2e/files",
            }
        },
    }

    rendered = services_pool.with_e2e_instance_paths(config_data, data_dir)

    assert rendered["jobs"]["executors"][0]["config"]["working_directory"] == str(data_dir / "subprocess-jobs")
    assert rendered["jobs"]["executor_defaults"]["subprocess"]["working_directory"] == str(data_dir / "subprocess-jobs")
    assert rendered["files"]["default_storage_config"]["path"] == str(data_dir / "files")
    jobs_config = cast(dict[str, Any], config_data["jobs"])
    executors = cast(list[dict[str, Any]], jobs_config["executors"])
    assert executors[0]["config"]["working_directory"] == ".tmp/e2e/subprocess-jobs"


def test_with_e2e_instance_paths_scopes_docker_deployments_executor(tmp_path):
    data_dir = tmp_path / "data-abc123def456"
    config_data: dict[str, Any] = {
        "deployments": {
            "executors": [
                {"name": "local-docker", "backend": "docker", "config": {"pull_images": False}},
                {"name": "local-k8s", "backend": "k8s", "config": {}},
            ],
        },
    }

    rendered = services_pool.with_e2e_instance_paths(config_data, data_dir, resource_scope="e2e-abc123def456")

    deployments = cast(dict[str, Any], rendered["deployments"])
    executors = cast(list[dict[str, Any]], deployments["executors"])
    assert executors[0]["config"]["resource_scope"] == "e2e-abc123def456"
    assert "resource_scope" not in executors[1]["config"]
