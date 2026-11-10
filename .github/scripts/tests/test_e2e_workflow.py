# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract for persistent, label-gated PR E2E runs."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/e2e.yaml"


def test_pr_e2e_runs_on_label_and_new_commits() -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    assert workflow["on"]["pull_request"]["types"] == ["labeled", "synchronize"]
    assert workflow["on"]["pull_request"]["branches"] == ["main", "release/*"]
    assert "pull_request_target" not in workflow["on"]


def test_plugin_build_uses_the_web_package_manager_pin() -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["web-studio-e2e"]
    step = next(step for step in job["steps"] if step.get("name") == "Build Agent Hardener plugin UI")
    # The standalone plugin has no packageManager field. Starting Corepack there
    # selects latest pnpm, which may not work with our pinned Node/Corepack.
    assert job["defaults"]["run"]["working-directory"] == "web"
    assert "working-directory" not in step
    assert step["run"].splitlines() == [
        "pnpm --dir ../plugins/nemo-agent-hardener/web install --frozen-lockfile",
        "pnpm --dir ../plugins/nemo-agent-hardener/web build",
    ]


def test_pr_e2e_preserves_eligibility_guards() -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    eligibility = " ".join(workflow["jobs"]["eligibility"]["if"].split())
    assert "(github.event.action == 'synchronize' || github.event.label.name == 'e2e') &&" in eligibility
    assert "github.event.pull_request.head.repo.full_name == github.repository &&" in eligibility
    assert "toJSON(github.event.pull_request.draft) == 'false' &&" in eligibility
    assert "contains(github.event.pull_request.labels.*.name, 'e2e')" in eligibility


def test_unrelated_labels_cannot_cancel_active_e2e() -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    concurrency = workflow["concurrency"]
    # Only unrelated label additions get a unique group; synchronize and explicit
    # e2e opt-in share the PR group so newer runs cancel outdated commits.
    assert concurrency["group"] == (
        "${{ github.workflow }}-${{ github.event_name == 'pull_request' && "
        "github.event.action == 'labeled' && github.event.label.name != 'e2e' && "
        "format('ignored-{0}', github.run_id) || github.event.pull_request.number || github.ref }}"
    )
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
