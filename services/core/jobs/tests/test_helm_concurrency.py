# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import shutil
import subprocess

import pytest
import yaml
from nhx.common.config import Configuration
from nhx.core.jobs.config import JobsServiceConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
CHART = REPO_ROOT / "k8s" / "helm"

pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")


def _rendered_jobs_config(*values_files: pathlib.Path) -> JobsServiceConfig:
    command = ["helm", "template", "t", str(CHART), "--show-only", "templates/platform-configmap.yaml"]
    for values_file in values_files:
        command += ["-f", str(values_file)]
    rendered = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    config_yaml = yaml.safe_load(rendered)["data"]["config.yaml"]
    return Configuration.global_settings_to_service_config(yaml.safe_load(config_yaml), JobsServiceConfig)


def test_chart_defaults_render_the_documented_limits():
    config = _rendered_jobs_config()

    assert config.executor_defaults.kubernetes_job.max_active_jobs == 500
    assert config.executor_defaults.volcano_job.max_active_jobs == 500
    assert config.max_active_jobs_per_user == 50
    assert config.max_active_jobs_per_workspace == 200


def test_chart_values_override_the_limits():
    config = _rendered_jobs_config(CHART / "ci" / "20-jobs-concurrency.yaml")

    assert config.executor_defaults.kubernetes_job.max_active_jobs == 100
    assert config.executor_defaults.volcano_job.max_active_jobs == 100
    assert config.max_active_jobs_per_user == 10
    assert config.max_active_jobs_per_workspace == 40
