# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for authenticated Intake snapshot exports."""

from pathlib import Path

import pytest
from evaluation import artifact
from evaluation.registry import Subject


def _glamr_subject(**overrides: object) -> Subject:
    return Subject(
        "glamr",
        "intake",
        {
            "agent": "glamr",
            "workspace": "default",
            "base_url": "https://agenthub.aire.nvidia.com",
            "auth": "basic",
            "intake_path_prefix": "/glamr/intake",
            "auth_user_env": "GLAMR_INTAKE_USER",
            "auth_password_env": "GLAMR_INTAKE_PASSWORD",
            **overrides,
        },
    )


def test_snapshot_basic_auth_client_uses_named_credentials_and_normalized_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GLAMR_INTAKE_USER", "intake-user")
    monkeypatch.setenv("GLAMR_INTAKE_PASSWORD", "secret")
    built: dict[str, object] = {}
    sentinel = object()

    def fake_builder(**kwargs: object) -> object:
        built.update(kwargs)
        return sentinel

    monkeypatch.setattr(artifact, "build_basic_auth_intake_client", fake_builder)

    client = artifact._basic_auth_intake_client_for(_glamr_subject(), "https://snapshot.example")

    assert client is sentinel
    assert built == {
        "base_url": "https://snapshot.example",
        "username": "intake-user",
        "password": "secret",
        "real_prefix": "/glamr/intake/",
    }


@pytest.mark.parametrize(
    ("missing_env", "credential"),
    [
        ("GLAMR_INTAKE_USER", "username"),
        ("GLAMR_INTAKE_PASSWORD", "password"),
    ],
)
def test_snapshot_basic_auth_client_exits_for_missing_credential(
    monkeypatch: pytest.MonkeyPatch,
    missing_env: str,
    credential: str,
) -> None:
    monkeypatch.setenv("GLAMR_INTAKE_USER", "intake-user")
    monkeypatch.setenv("GLAMR_INTAKE_PASSWORD", "secret")
    monkeypatch.delenv(missing_env)

    with pytest.raises(SystemExit, match=f"glamr.*{credential}.*{missing_env}"):
        artifact._basic_auth_intake_client_for(_glamr_subject(), "https://snapshot.example")


def test_snapshot_non_basic_subject_does_not_build_authenticated_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        artifact,
        "build_basic_auth_intake_client",
        lambda **kwargs: pytest.fail(f"unexpected basic-auth builder call: {kwargs}"),
    )

    assert (
        artifact._basic_auth_intake_client_for(
            Subject("plain", "intake", {"workspace": "default", "base_url": "https://snapshot.example"}),
            "https://snapshot.example",
        )
        is None
    )


def test_snapshot_setup_failure_does_not_construct_authenticated_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    constructed = False

    def fake_client(subject: Subject, source_url: str) -> object:
        nonlocal constructed
        constructed = True
        return object()

    def fail_pick_records(tmp_dir: Path, names: list[str]) -> list[Path]:
        raise RuntimeError("setup failed")

    monkeypatch.setattr(artifact, "_basic_auth_intake_client_for", fake_client)
    monkeypatch.setattr(artifact, "pick_records", fail_pick_records)

    with pytest.raises(RuntimeError, match="setup failed"):
        artifact.snapshot_export([_glamr_subject()], tmp_path / "snapshot.tar.zst", tmp_path / "tmp", since=None)

    assert not constructed


def test_snapshot_export_scopes_authentication_to_each_subject(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GLAMR_INTAKE_USER", "glamr-user")
    monkeypatch.setenv("GLAMR_INTAKE_PASSWORD", "glamr-secret")
    monkeypatch.setenv("OTHER_INTAKE_USER", "other-user")
    monkeypatch.setenv("OTHER_INTAKE_PASSWORD", "other-secret")
    source_url = "https://snapshot.example"
    other = Subject(
        "other",
        "intake",
        {
            "workspace": "other-workspace",
            "base_url": source_url,
            "auth": "basic",
            "intake_path_prefix": "/other/intake",
            "auth_user_env": "OTHER_INTAKE_USER",
            "auth_password_env": "OTHER_INTAKE_PASSWORD",
        },
    )
    plain = Subject("plain", "intake", {"workspace": "plain-workspace", "base_url": source_url})
    duplicate_workspace = Subject("duplicate", "intake", {"workspace": "plain-workspace", "base_url": source_url})
    built: list[tuple[dict[str, object], object]] = []
    exports: list[tuple[list[str], object | None]] = []
    manifest_stats: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> object:
        client = object()
        built.append((kwargs, client))
        return client

    def fake_export(
        base_url: str,
        workspaces: list[str],
        out_dir: Path,
        *,
        since: object,
        client: object | None,
        experiment: str | None = None,
    ) -> dict:
        exports.append((workspaces, client))
        return {
            "workspaces": {
                workspace: {"spans": 0, "annotations": 0, "evaluator_results": 0} for workspace in workspaces
            },
            "min_start_time": f"2026-07-0{len(exports)}T00:00:00+00:00",
            "max_start_time": f"2026-07-0{len(exports)}T01:00:00+00:00",
        }

    def fake_manifest(
        subjects: list[str],
        records: list[Path],
        stats: dict,
        *,
        source_url: str,
        platform_info: dict | None,
        env: object,
    ) -> dict:
        manifest_stats.update(stats)
        return {}

    monkeypatch.setattr(artifact, "build_basic_auth_intake_client", fake_builder)
    monkeypatch.setattr(artifact.export, "export_workspaces", fake_export)
    monkeypatch.setattr(artifact, "build_export_manifest", fake_manifest)
    monkeypatch.setattr(artifact, "fetch_platform_info", lambda base_url: None)
    monkeypatch.setattr(artifact.subprocess, "run", lambda *args, **kwargs: None)

    artifact.snapshot_export(
        [_glamr_subject(base_url=source_url), other, plain, duplicate_workspace],
        tmp_path / "snapshot.tar.zst",
        tmp_path / "tmp",
        since=None,
    )

    assert [kwargs for kwargs, _ in built] == [
        {
            "base_url": source_url,
            "username": "glamr-user",
            "password": "glamr-secret",
            "real_prefix": "/glamr/intake/",
        },
        {
            "base_url": source_url,
            "username": "other-user",
            "password": "other-secret",
            "real_prefix": "/other/intake/",
        },
    ]
    assert exports == [
        (["default"], built[0][1]),
        (["other-workspace"], built[1][1]),
        (["plain-workspace"], None),
    ]
    workspaces = manifest_stats["workspaces"]
    assert isinstance(workspaces, dict)
    assert set(workspaces) == {"default", "other-workspace", "plain-workspace"}
    assert manifest_stats["min_start_time"] == "2026-07-01T00:00:00+00:00"
    assert manifest_stats["max_start_time"] == "2026-07-03T01:00:00+00:00"
