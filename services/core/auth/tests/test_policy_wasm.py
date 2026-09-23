# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest
from nhx.core.auth.app.embedded_pdp.policy_wasm import (
    PolicyWasmError,
    ensure_embedded_policy_wasm,
    policy_wasm_needs_build,
)


def test_policy_wasm_needs_build_when_missing(tmp_path: Path):
    assert policy_wasm_needs_build(wasm_path=tmp_path / "policy.wasm")


def test_policy_wasm_does_not_need_build_when_artifact_exists(tmp_path: Path):
    wasm = tmp_path / "policy.wasm"
    wasm.write_bytes(b"wasm")

    assert not policy_wasm_needs_build(wasm_path=wasm)


def test_ensure_builds_missing_wasm_from_source_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    build_script = repo_root / "script" / "build_policy_wasm.sh"
    wasm = tmp_path / "out" / "policy.wasm"
    build_script.parent.mkdir(parents=True)
    build_script.write_text("#!/usr/bin/env sh\n")

    def fake_run(*args, **kwargs):
        assert args[0] == [str(build_script)]
        assert kwargs["cwd"] == repo_root
        assert kwargs["env"]["REPO_ROOT"] == str(repo_root)
        assert kwargs["env"]["OUTPUT_DIR"] == str(wasm.parent)
        wasm.parent.mkdir(parents=True)
        wasm.write_bytes(b"wasm")
        return subprocess.CompletedProcess(args[0], 0, stdout="built", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        ensure_embedded_policy_wasm(
            wasm_path=wasm,
            repo_root=repo_root,
            discover_source_checkout=False,
        )
        == wasm
    )


def test_ensure_raises_when_auto_build_disabled(tmp_path: Path):
    with pytest.raises(PolicyWasmError, match="make build-policy"):
        ensure_embedded_policy_wasm(
            auto_build=False,
            wasm_path=tmp_path / "policy.wasm",
            discover_source_checkout=False,
        )


def test_ensure_raises_when_missing_outside_source_checkout(tmp_path: Path):
    with pytest.raises(PolicyWasmError, match="source checkout"):
        ensure_embedded_policy_wasm(
            wasm_path=tmp_path / "policy.wasm",
            discover_source_checkout=False,
        )


def test_ensure_reports_build_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    build_script = repo_root / "script" / "build_policy_wasm.sh"
    build_script.parent.mkdir(parents=True)
    build_script.write_text("#!/usr/bin/env sh\n")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 2, stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(PolicyWasmError, match="OPA_BIN"):
        ensure_embedded_policy_wasm(
            wasm_path=tmp_path / "out" / "policy.wasm",
            repo_root=repo_root,
            discover_source_checkout=False,
        )
