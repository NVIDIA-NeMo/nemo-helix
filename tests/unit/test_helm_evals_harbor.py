# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from nemo_evals.config import HarborTaskJobsConfig

ROOT = Path(__file__).parent.parent.parent
HELM_DIR = ROOT / "k8s" / "helm"
HELM_TEMPLATE_TIMEOUT_SECONDS = 60
API_HOST = "nemo-helix-api.default.svc.cluster.local"
RUNS_ON = ("--set", "sandboxClusterCapable=true", "--set", "builder.enabled=true")


def _helm_template(*args: str) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")

    completed = subprocess.run(
        ["helm", "template", "nemo-helix", str(HELM_DIR), "--namespace", "default", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=HELM_TEMPLATE_TIMEOUT_SECONDS,
    )
    return [document for document in yaml.safe_load_all(completed.stdout) if document]


def _platform_config(documents: list[dict]) -> dict:
    config_map = next(
        document
        for document in documents
        if document["kind"] == "ConfigMap" and "config.yaml" in document.get("data", {})
    )
    return yaml.safe_load(config_map["data"]["config.yaml"])


def _harbor_config(*args: str) -> dict | None:
    return _platform_config(_helm_template(*args))["evals"].get("runs", {}).get("harbor")


def _with_platform_config(platform_config: dict) -> tuple[str, ...]:
    return (*RUNS_ON, "--set-json", f"platformConfig={json.dumps(platform_config)}")


@pytest.mark.parametrize(
    ("sandbox_cluster_capable", "builder_enabled", "runs_enabled"),
    [(False, False, False), (True, False, False), (False, True, False), (True, True, True)],
)
def test_harbor_runs_need_opensandbox_and_builder(
    sandbox_cluster_capable: bool, builder_enabled: bool, runs_enabled: bool
) -> None:
    harbor = _harbor_config(
        "--set",
        f"sandboxClusterCapable={str(sandbox_cluster_capable).lower()}",
        "--set",
        f"builder.enabled={str(builder_enabled).lower()}",
    )

    assert (harbor is not None) is runs_enabled


def test_default_harbor_config_is_valid_with_harbor_standard_only() -> None:
    config = HarborTaskJobsConfig.model_validate(_harbor_config(*RUNS_ON))

    assert config.dispatcher_profile == "default"
    assert config.default_sandbox_profile == "harbor-standard"
    assert list(config.sandbox_profiles) == ["harbor-standard"]
    standard = config.sandbox_profiles["harbor-standard"]
    assert standard.agent_channel == (API_HOST,)
    assert standard.egress_allow == (API_HOST,)
    assert {"registry.npmjs.org"} <= set(standard.agent_setup_egress)


def test_platform_config_list_replaces_the_chart_list() -> None:
    harbor = _harbor_config(
        *_with_platform_config(
            {"evals": {"runs": {"harbor": {"sandbox_profiles": {"harbor-standard": {"egress_allow": ["igw.local"]}}}}}}
        )
    )

    standard = HarborTaskJobsConfig.model_validate(harbor).sandbox_profiles["harbor-standard"]
    assert standard.egress_allow == ("igw.local",)
    assert standard.agent_channel == (API_HOST,)


def test_platform_config_profile_is_added_beside_harbor_standard() -> None:
    harbor = _harbor_config(
        *_with_platform_config(
            {"evals": {"runs": {"harbor": {"sandbox_profiles": {"harbor-open": {"egress_allow": ["*.com"]}}}}}}
        )
    )

    config = HarborTaskJobsConfig.model_validate(harbor)
    assert sorted(config.sandbox_profiles) == ["harbor-open", "harbor-standard"]
    # Helm renders profiles alphabetically, so the first one is harbor-open; the chart's default keeps harbor-standard.
    assert config.default_sandbox_profile == "harbor-standard"


def test_platform_config_null_turns_harbor_runs_off() -> None:
    assert _harbor_config(*_with_platform_config({"evals": {"runs": {"harbor": None}}})) is None
