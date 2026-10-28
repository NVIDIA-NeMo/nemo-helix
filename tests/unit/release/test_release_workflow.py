# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the workflow's version and dispatch adapters without publishing."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/release.yaml").read_text())
TIMESTAMP = "20261026010203"


def step(job: str, name: str) -> dict:
    return next(item for item in WORKFLOW["jobs"][job]["steps"] if item.get("name") == name)


@pytest.mark.parametrize(
    ("release_type", "version", "prerelease", "expected_wheel", "expected_chart"),
    [
        ("rc", "", "false", f"0.9.0.dev{TIMESTAMP}", f"0.9.0-rc-{TIMESTAMP}"),
        ("stable", "0.9.0", "false", "0.9.0", "0.9.0"),
        ("stable", "0.9.0-rc1", "true", "0.9.0rc1", "0.9.0-rc1"),
    ],
)
def test_workflow_versions(tmp_path, release_type, version, prerelease, expected_wheel, expected_chart):
    output = tmp_path / "outputs"
    # Only the preserved workflow helper exists; selected source has no new helper.
    (tmp_path / "stamp_sdk_version.py").write_bytes((ROOT / ".github/scripts/stamp_sdk_version.py").read_bytes())
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
        "RUNNER_TEMP": str(tmp_path),
        "RELEASE_TYPE": release_type,
        "RELEASE_VERSION": version,
        "IS_PRERELEASE": prerelease,
        "RC_TIMESTAMP": TIMESTAMP,
        "RC_BASE_VERSION": "0.9.0" if release_type == "rc" else "",
        "WHEEL_IDS": '["nemo-helix-plugin"]',
        "HELM_VERSION_OVERRIDE": "",
    }
    for name in ["Resolve wheel version", "Resolve Helm version"]:
        subprocess.run(["bash", "-e", "-c", step("plan-release", name)["run"]], cwd=tmp_path, env=env, check=True)
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs == {"wheel_version": expected_wheel, "helm_version": expected_chart}


@pytest.mark.parametrize("act", ["false", "true"])
def test_rc_dispatch_contract(act):
    env = {
        **os.environ,
        "ACT": act,
        "DISPATCH_REPO": "example/deploy",
        "SOURCE_SHA": "a" * 40,
        "RELEASE_TYPE": "rc",
        "IS_PRERELEASE": "false",
        "RELEASE_LABEL": f"rc-{TIMESTAMP}",
        "RC_TIMESTAMP": TIMESTAMP,
        "WHEEL_VERSION": f"0.9.0.dev{TIMESTAMP}",
        "WHEEL_IDS": '["nemo-helix-plugin"]',
        "CONTAINER_IDS": '["nhx-api"]',
        "PUBLISH_RC_WHEELS": "true",
        "HELM_VERSION": f"0.9.0-rc-{TIMESTAMP}",
    }
    adapter = """
        const writes = [], logs = [];
        const core = { info: (text) => logs.push(text) };
        const github = {rest: {repos: {createDispatchEvent: async (payload) => writes.push(payload)}}};
        (async () => { SCRIPT })().then(() => console.log(JSON.stringify({writes, logs})));
    """
    payloads = []
    for job, name in [
        ("dispatch-wheel-stage", "Dispatch wheel builds"),
        ("dispatch-container-stage", "Dispatch container builds"),
        ("signal-deployment", "Dispatch deployment creation"),
    ]:
        script = step(job, name)["with"]["script"]
        result = subprocess.run(
            ["node", "-e", adapter.replace("SCRIPT", script)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(result.stdout)
        if act == "true":
            assert result["writes"] == []
            payload = json.loads(result["logs"][0].split("\n", 1)[1])
        else:
            [payload] = result["writes"]
        payloads.append(payload["client_payload"])
        assert payload["client_payload"]["cadence"] == "rc"
        assert payload["client_payload"]["ref"] == "a" * 40
    wheel, container, deployment = payloads
    assert wheel["rc_timestamp"] == TIMESTAMP
    assert wheel["publish_rc_wheels"] is True
    assert wheel["wheel_version"] == env["WHEEL_VERSION"]
    assert wheel["wheels"] == ["nemo-helix-plugin"]
    assert container["version"] == wheel["release_label"] == deployment["release_label"] == f"rc-{TIMESTAMP}"
    assert deployment["helm_version"] == env["HELM_VERSION"]
    assert "nightly" not in json.dumps(payloads)
