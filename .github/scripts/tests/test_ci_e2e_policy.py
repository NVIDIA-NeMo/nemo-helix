# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual CI YAML predicates, not a second production policy helper."""

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/ci.yaml"
CI = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
JOBS = CI["jobs"]
E2E_JOBS = (
    "kind-cpu-e2e",
    "kind-gym-e2e",
    "kind-customizer-auth-e2e",
    "kind-network-policy-smoke",
    "python-auth-idp-e2e-test",
    "python-e2e-test",
    "python-e2e-image-test",
)
IMAGE_JOBS = tuple(name for name in E2E_JOBS if name != "python-e2e-test")
OPT_IN_JOBS = ("build-cpu-smoke-images", "python-auth-idp-static-test", *E2E_JOBS)


def _lookup(value, components):
    if not components:
        return value
    head, *tail = components
    if head == "*":
        return [_lookup(item, tail) for item in value] if isinstance(value, list) else []
    return _lookup(value.get(head), tail) if isinstance(value, dict) else None


def _equal(left, right):
    # GitHub string equality ignores case; unlike Python, null == false is true.
    if isinstance(left, str) and isinstance(right, str):
        return left.casefold() == right.casefold()
    if type(left) is not type(right):

        def number(value):
            if value is None or value == "":
                return 0
            try:
                return float(value)
            except (ValueError, TypeError):
                return math.nan

        return number(left) == number(right)
    return left == right


class Expression:
    """Test-only interpreter for the explicit boolean subset used by these gates.

    Reject unsupported syntax/functions rather than silently accepting changed
    workflow conditions. No eval(), runtime helper, token, or API call is used.
    """

    def __init__(self, expression, context):
        expression = expression.strip().removeprefix("${{").removesuffix("}}").strip()
        pattern = re.compile(r"\s*('(?:[^']|'')*'|&&|\|\||==|!=|[!(),]|[A-Za-z_][\w.*-]*)")
        self.tokens = []
        position = 0
        while position < len(expression):
            match = pattern.match(expression, position)
            assert match, f"Unsupported condition syntax: {expression[position:]}"
            self.tokens.append(match[1])
            position = match.end()
        self.position = 0
        self.context = context

    def take(self, token):
        if self.position < len(self.tokens) and self.tokens[self.position] == token:
            self.position += 1
            return True
        return False

    def parse(self):
        result = self.disjunction()
        assert self.position == len(self.tokens), self.tokens[self.position :]
        return bool(result)

    def disjunction(self):
        result = self.conjunction()
        while self.take("||"):
            right = self.conjunction()
            result = result or right
        return result

    def conjunction(self):
        result = self.comparison()
        while self.take("&&"):
            right = self.comparison()
            result = result and right
        return result

    def comparison(self):
        result = self.atom()
        if self.take("=="):
            return _equal(result, self.atom())
        if self.take("!="):
            return not _equal(result, self.atom())
        return result

    def atom(self):
        if self.take("!"):
            return not self.atom()
        if self.take("("):
            result = self.disjunction()
            assert self.take(")")
            return result
        token = self.tokens[self.position]
        self.position += 1
        if token.startswith("'"):
            return token[1:-1].replace("''", "'")
        if token in {"true", "false", "null"}:
            return {"true": True, "false": False, "null": None}[token]
        if self.take("("):
            arguments = []
            if not self.take(")"):
                arguments.append(self.disjunction())
                while self.take(","):
                    arguments.append(self.disjunction())
                assert self.take(")")
            functions = {
                "contains": lambda values, value: any(_equal(item, value) for item in (values or [])),
                "toJSON": lambda value: json.dumps(value, separators=(",", ":")),
                "cancelled": lambda: self.context.get("cancelled", False),
            }
            assert token in functions, f"Unsupported condition function: {token}"
            return functions[token](*arguments)
        return _lookup(self.context, token.split("."))


def evaluate(expression, context):
    return Expression(expression, context).parse()


def context(*, event="pull_request", action="synchronize", draft=False, labels=("e2e",), fork=False):
    result = {
        "github": {
            "event_name": event,
            "repository": "NVIDIA-NeMo/nemo-helix",
            "ref": "refs/heads/main" if event == "push" else "refs/pull/42/merge",
            "event": {
                "action": action,
                "label": {"name": "unrelated"},
                "pull_request": {
                    "draft": draft,
                    "head": {"repo": {"full_name": "other/fork" if fork else "NVIDIA-NeMo/nemo-helix"}},
                    "labels": [{"name": name} for name in labels],
                },
            },
        },
        "needs": {
            "build-cpu-smoke-images": {"result": "success", "outputs": {"publish_images": "true"}},
            "python-auth-idp-static-test": {"result": "success"},
        },
    }
    outputs = {key: "false" for key in ("cpu-smoke", "auth-idp", "customization-auth", "helm", "e2e")}
    for name in ("post-merge", "pr-e2e"):
        outputs[name] = str(evaluate(JOBS["changes"]["outputs"][name], result)).lower()
    result["needs"]["changes"] = {"outputs": outputs}
    return result


@pytest.mark.parametrize("action", ["opened", "synchronize", "reopened", "ready_for_review", "labeled", "unlabeled"])
def test_current_opt_in_labels_enable_every_e2e_job_without_path_changes(action):
    event = context(action=action)
    for name in OPT_IN_JOBS:
        assert evaluate(JOBS[name]["if"], event), name


@pytest.mark.parametrize("labels", [(), ("e2e-other",), ("run-e2e",), ("documentation",)])
def test_no_opt_in_label_means_no_privileged_pr_e2e(labels):
    event = context(labels=labels)
    for name in ("build-cpu-smoke-images", *E2E_JOBS):
        assert not evaluate(JOBS[name]["if"], event), name


@pytest.mark.parametrize("override", [{"draft": True}, {"fork": True}, {"draft": "false"}, {"draft": None}])
def test_drafts_forks_and_nonboolean_draft_context_fail_closed(override):
    event = context(**override)
    assert not evaluate(JOBS["changes"]["outputs"]["pr-e2e"], event)
    for name in OPT_IN_JOBS:
        assert not evaluate(JOBS[name]["if"], event), name


@pytest.mark.parametrize("missing", ["pull_request", "head", "draft", "labels"])
def test_missing_pr_context_fails_closed(missing):
    event = context()
    payload = event["github"]["event"]
    if missing == "pull_request":
        payload.pop(missing)
    else:
        payload["pull_request"].pop(missing)
    assert not evaluate(JOBS["changes"]["outputs"]["pr-e2e"], event)


@pytest.mark.parametrize("event_name", ["merge_group", "push", "workflow_dispatch"])
def test_non_pr_events_never_gain_pr_opt_in_from_labels(event_name):
    event = context(event=event_name)
    assert not evaluate(JOBS["changes"]["outputs"]["pr-e2e"], event)


@pytest.mark.parametrize("event_name", ["workflow_dispatch", "push"])
def test_existing_manual_and_postmerge_paths_remain_enabled(event_name):
    event = context(event=event_name, labels=())
    # Main pushes retain the existing change filter, unlike explicit PR opt-in.
    event["needs"]["changes"]["outputs"].update({"cpu-smoke": "true", "customization-auth": "true", "helm": "true"})
    for name in OPT_IN_JOBS:
        assert evaluate(JOBS[name]["if"], event), name


def test_main_push_retains_existing_path_filters():
    event = context(event="push", labels=())
    for name in (
        "build-cpu-smoke-images",
        "kind-customizer-auth-e2e",
        "kind-network-policy-smoke",
        "python-auth-idp-e2e-test",
    ):
        assert not evaluate(JOBS[name]["if"], event), name


def test_release_push_and_merge_queue_do_not_enable_e2e():
    for event_name in ("push", "merge_group"):
        event = context(event=event_name)
        event["github"]["ref"] = "refs/heads/release/0.10"
        postmerge = evaluate(JOBS["changes"]["outputs"]["post-merge"], event)
        event["needs"]["changes"]["outputs"]["post-merge"] = str(postmerge).lower()
        assert not postmerge
        for name in ("build-cpu-smoke-images", *E2E_JOBS):
            assert not evaluate(JOBS[name]["if"], event), name


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_image_consumers_require_successful_cpu_build(result):
    event = context()
    event["needs"]["build-cpu-smoke-images"]["result"] = result
    for name in IMAGE_JOBS:
        assert "build-cpu-smoke-images" in JOBS[name]["needs"]
        assert not evaluate(JOBS[name]["if"], event), name


def test_unpublished_images_and_failed_static_prerequisite_block_dependents():
    event = context()
    event["needs"]["build-cpu-smoke-images"]["outputs"]["publish_images"] = "false"
    for name in IMAGE_JOBS:
        assert not evaluate(JOBS[name]["if"], event), name
    event = context()
    event["needs"]["python-auth-idp-static-test"]["result"] = "failure"
    assert not evaluate(JOBS["python-auth-idp-e2e-test"]["if"], event)


def test_cancelled_runs_cannot_start_e2e_or_prerequisites():
    event = context()
    event["cancelled"] = True
    for name in OPT_IN_JOBS:
        assert not evaluate(JOBS[name]["if"], event), name


def test_event_types_concurrency_permissions_and_required_aggregate():
    triggers = CI["on"]
    assert "pull_request_target" not in triggers
    assert triggers["pull_request"]["branches"] == ["main", "release/*"]
    assert set(triggers["pull_request"]["types"]) == {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "converted_to_draft",
        "labeled",
        "unlabeled",
    }
    assert CI["concurrency"]["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"
    assert "github.event.pull_request.number" in CI["concurrency"]["group"]
    assert CI["permissions"] == {"actions": "read", "contents": "read", "pull-requests": "read"}
    assert JOBS["build-cpu-smoke-images"]["permissions"] == {"contents": "read", "packages": "write"}
    for name in IMAGE_JOBS:
        assert JOBS[name]["permissions"] == {"contents": "read", "packages": "read"}
    assert set(OPT_IN_JOBS) - {"python-auth-idp-static-test"} <= set(JOBS["ci-status"]["needs"])
    # Native Rego removed policy-wasm; no consumer may retain a dangling job dependency.
    for name, job in JOBS.items():
        needs = job.get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        assert set(needs) <= JOBS.keys(), name


def test_policy_regressions_run_unconditionally_in_small_isolated_ci_environment():
    job = JOBS["test-gates"]
    assert "if" not in job
    step = next(step for step in job["steps"] if step.get("name") == "Test e2e opt-in policy")
    command = " ".join(step["run"].split())
    assert command == (
        "uv run --isolated --no-project --with pytest==9.1.1 --with pyyaml==6.0.3 "
        "python -m pytest -c /dev/null -p no:cacheprovider --confcutdir=.github/scripts/tests "
        ".github/scripts/tests/test_ci_e2e_policy.py -q"
    )


def test_ui_stays_manual_and_unrelated_image_builds_and_notifications_stay_postmerge():
    assert JOBS["web-studio-e2e"]["if"] == "github.event_name == 'workflow_dispatch'"
    for name in ("build-garak-test-image", "build-safe-synthesizer-test-image"):
        assert not evaluate(JOBS[name]["if"], context()), name
    assert JOBS["post-merge-image-e2e-status"]["if"].strip() == (
        "always() && github.event_name == 'push' && github.ref == 'refs/heads/main'"
    )


@pytest.mark.parametrize(
    ("event_name", "fork", "publish", "expected"),
    [
        ("pull_request", False, "", "true"),
        ("pull_request", True, "", "false"),
        ("workflow_dispatch", False, "true", "true"),
        ("workflow_dispatch", False, "false", "false"),
    ],
)
def test_actual_cpu_publish_script_preserves_pr_and_manual_publishing(tmp_path, event_name, fork, publish, expected):
    step = next(step for step in JOBS["build-cpu-smoke-images"]["steps"] if step.get("id") == "bake-vars")
    output = tmp_path / "output"
    env = {
        **os.environ,
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_REPOSITORY": "NVIDIA-NeMo/nemo-helix",
        "GITHUB_REF": "refs/pull/42/merge",
        "GITHUB_SHA": "merge-sha",
        "HEAD_REPOSITORY": "other/fork" if fork else "NVIDIA-NeMo/nemo-helix",
        "SOURCE_SHA": "head-sha",
        "INPUT_PUBLISH": publish,
        "INPUT_IMAGE_TAG": "",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_ENV": str(tmp_path / "environment"),
        "GITHUB_OUTPUT": str(output),
    }
    subprocess.run(["bash", "-c", step["run"]], env=env, check=True, timeout=10)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["publish_images"] == expected
    assert values["image_tag"] == "head-sha"
