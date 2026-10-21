# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submission snapshot and public projection tests."""

from __future__ import annotations

import json
import os

import pytest

try:
    from scaled_evals.dispatch.credentials import injected_credential_env
    from scaled_evals.models.execution_snapshot import (
        EXECUTION_SNAPSHOT_SCHEMA_VERSION,
        build_execution_snapshot,
        public_execution_projection,
        snapshot_profile_config,
        validate_execution_snapshot,
    )
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def _snapshot() -> dict:
    return build_execution_snapshot(
        captured_at="2026-07-10T12:00:00+00:00",
        evaluation={
            "id": "ev_1",
            "runtime": "sandbox_k8s",
            "instruction_prefix": "use sk-private-value",
            "initial_user_turns": ["private prompt"],
            "extra_skill_object_keys": ["skills/archive.tar.gz"],
        },
        task={
            "id": "task_1",
            "revision": 2,
            "image_digest": "sha256:" + "a" * 64,
            "build_payload": {
                "source_project": "aire/evals",
                "source_ref": "abc123",
                "access_token": "never-public",
            },
        },
        profiles={
            "harbor": {
                "id": "cfg_1",
                "name": "profile",
                "type": "harbor",
                "config": {
                    "environment": {"image_pull_secrets": ["pull-secret"]},
                    "api_key": "sk-profile-secret",
                },
                "updated_at": "2026-07-10T11:00:00+00:00",
            }
        },
        credentials={"openai": "team-a/openai-key"},
        submission_identity={"source_revision": "control-plane-sha"},
    )


def test_private_snapshot_drives_dispatch_and_public_projection_is_hash_only() -> None:
    snapshot = _snapshot()

    assert snapshot["schema_version"] == EXECUTION_SNAPSHOT_SCHEMA_VERSION
    assert snapshot_profile_config(snapshot, "harbor")["environment"] == {"image_pull_secrets": ["pull-secret"]}

    public = public_execution_projection(snapshot)
    payload = json.dumps(public)
    assert "sk-private-value" not in payload
    assert "private prompt" not in payload
    assert "sk-profile-secret" not in payload
    assert "never-public" not in payload
    assert public["profiles"]["harbor"]["config_sha256"].startswith("sha256:")
    assert public["evaluation"]["instruction_prefix_sha256"].startswith("sha256:")
    assert public["credentials"] == {"openai": "team-a/openai-key"}


def test_unknown_or_empty_snapshot_fails_closed() -> None:
    assert validate_execution_snapshot(None) is None
    with pytest.raises(ValueError, match="schema_version"):
        validate_execution_snapshot({})
    with pytest.raises(ValueError, match="schema_version"):
        validate_execution_snapshot({"schema_version": "future-v99"})


def test_dispatch_takes_injected_secrets_out_of_the_process_env(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("SCALED_EVALS_CREDENTIAL_ANTHROPIC", "sk-ant-value")
    monkeypatch.setenv("SCALED_EVALS_CREDENTIAL_OPENSHIFT", "oc-token")

    env = injected_credential_env({"anthropic": "team-a/ant", "openshift": "team-a/oc"})

    assert env["ANTHROPIC_API_KEY"] == "sk-ant-value"
    assert env["ANTHROPIC_BASE_URL"]
    assert env["SANDBOX_OC_TOKEN"] == "oc-token"
    # Backend subprocesses inherit os.environ; the raw injected values must not ride along.
    assert "SCALED_EVALS_CREDENTIAL_ANTHROPIC" not in os.environ
    # A retry in the same process still finds the value it already took.
    assert injected_credential_env({"anthropic": "team-a/ant"})["ANTHROPIC_API_KEY"] == "sk-ant-value"
    with pytest.raises(ValueError, match="openai was not injected"):
        injected_credential_env({"openai": "team-a/missing"})
