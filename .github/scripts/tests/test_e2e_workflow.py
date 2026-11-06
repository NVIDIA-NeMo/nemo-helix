# /// script
# requires-python = ">=3.11"
# dependencies = ["pytest>=9.0.3,<10", "pyyaml>=6,<7"]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the independent, opt-in E2E workflow's trigger and dependency contract."""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / "workflows"
# BaseLoader preserves the GitHub Actions `on` key (not a YAML 1.1 boolean).
CI = yaml.load((WORKFLOWS / "ci.yaml").read_text(), Loader=yaml.BaseLoader)
E2E = yaml.load((WORKFLOWS / "e2e.yaml").read_text(), Loader=yaml.BaseLoader)
SUITES = {
    "kind-cpu-e2e",
    "kind-gym-e2e",
    "kind-customizer-auth-e2e",
    "kind-network-policy-smoke",
    "python-auth-idp-e2e-test",
    "python-e2e-test",
    "python-e2e-image-test",
    "web-studio-e2e",
}


def test_e2e_is_separate_from_required_ci():
    assert SUITES <= E2E["jobs"].keys()
    assert not SUITES & CI["jobs"].keys()
    assert "build-cpu-smoke-images" not in CI["jobs"]
    assert "pr-e2e" not in CI["jobs"]["changes"]["outputs"]
    assert "labeled" not in CI["on"]["pull_request"]["types"]
    assert not SUITES & set(CI["jobs"]["ci-status"]["needs"])


def test_triggers_and_trust_boundary():
    assert set(E2E["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert E2E["on"]["push"]["branches"] == ["main"]
    assert E2E["on"]["pull_request"]["branches"] == ["main", "release/*"]
    assert set(E2E["on"]["pull_request"]["types"]) == {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "labeled",
    }
    gate = E2E["jobs"]["eligibility"]["if"]
    assert "github.event_name == 'pull_request' &&" in gate
    assert "github.event.pull_request.head.repo.full_name == github.repository &&" in gate
    assert "toJSON(github.event.pull_request.draft) == 'false' &&" in gate
    assert "contains(github.event.pull_request.labels.*.name, 'e2e')" in gate
    assert E2E["concurrency"]["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
    assert E2E["permissions"] == {"contents": "read"}


@pytest.mark.parametrize("workflow", [CI, E2E], ids=["ci", "e2e"])
def test_dependencies_are_local(workflow):
    for job in workflow["jobs"].values():
        assert set(job.get("needs", [])) <= workflow["jobs"].keys()


def test_e2e_notification_covers_all_jobs_on_main_pushes_only():
    notification = E2E["jobs"]["post-merge-e2e-status"]
    assert set(notification["needs"]) == E2E["jobs"].keys() - {"post-merge-e2e-status"}
    assert notification["if"].strip() == ("always() && github.event_name == 'push' && github.ref == 'refs/heads/main'")
    report = notification["steps"][-1]
    assert report["env"]["CHECKS_NAME"] == "E2E"
    assert report["env"]["SLACK_ALERTS_WEBHOOK"] == "${{ secrets.SLACK_ALERTS_WEBHOOK }}"
    assert 'require("./.github/scripts/ci-post-merge-notification.cjs")' in report["with"]["script"]


def test_every_e2e_job_is_gated():
    for name, job in E2E["jobs"].items():
        if name != "eligibility":
            assert "eligibility" in job["needs"], name
            assert "needs.changes" not in job.get("if", ""), name
    for name in ["python-e2e-test", "python-e2e-image-test", "python-auth-idp-e2e-test"]:
        assert "policy-wasm" in E2E["jobs"][name]["needs"]
    for name in SUITES:
        job = E2E["jobs"][name]
        if "build-cpu-smoke-images" in job.get("needs", []):
            assert job["if"] == "needs.build-cpu-smoke-images.outputs.publish_images == 'true'"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--confcutdir", str(Path(__file__).parent), "-o", "addopts="]))
