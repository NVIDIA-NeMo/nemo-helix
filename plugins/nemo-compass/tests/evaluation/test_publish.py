# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`evaluation publish` — laptop-first mint and CSS S3 upload."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from evaluation import cli, publish, release

MANIFEST = {
    "kind": "evaluation-export",
    "subjects": ["tau2-airline"],
    "workspaces": ["tau2-airline", "tau2-airline-oracle"],
    "counts": {
        "tau2-airline": {"spans": 450, "annotations": 3, "evaluator_results": 30},
        "tau2-airline-oracle": {"spans": 456, "annotations": 1, "evaluator_results": 30},
    },
    "min_start_time": "2026-07-01T00:00:00+00:00",
    "max_start_time": "2026-07-02T00:00:00+00:00",
    "source_url": "http://localhost:8080",
}


def _make_bundle(path: Path, manifest: dict = MANIFEST) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        state = Path(tmp) / "state"
        state.mkdir(parents=True)
        (state / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        subprocess.run(["tar", "--zstd", "-cf", str(path), "-C", tmp, "state"], check=True)
    return path


@pytest.fixture
def fake_store(monkeypatch):
    uploads = []
    monkeypatch.setattr(release, "object_names", lambda: ["state-v6.tar.zst"])
    monkeypatch.setattr(
        release,
        "upload_ref",
        lambda ref, bundle, **kwargs: uploads.append((ref, bundle, kwargs)),
    )
    return uploads


def test_publish_mints_next_ref_and_uploads_metadata(fake_store, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("getpass.getuser", lambda: "ada")
    candidate = _make_bundle(tmp_path / "candidate.tar.zst")
    ref = publish.publish(candidate, reason="fresh\ncorpus", env={})

    assert ref == "state-v7"
    tarball = tmp_path / "state-v7.tar.zst"
    assert tarball.read_bytes() == candidate.read_bytes()
    assert fake_store == [
        (
            "state-v7",
            tarball,
            {
                "metadata": {
                    "reason": "fresh corpus",
                    "published-by": "ada",
                    "sha256": hashlib.sha256(tarball.read_bytes()).hexdigest(),
                }
            },
        )
    ]
    assert "published state-v7" in capsys.readouterr().out


def test_publish_reason_falls_back_to_env(fake_store, tmp_path):
    publish.publish(_make_bundle(tmp_path / "candidate.tar.zst"), reason=None, env={"REASON": "from env"})
    assert fake_store[0][2]["metadata"]["reason"] == "from env"


def test_publish_reason_falls_back_to_manifest(fake_store, tmp_path):
    manifest = {**MANIFEST, "reason": "captured during produce"}
    publish.publish(_make_bundle(tmp_path / "candidate.tar.zst", manifest), reason=None, env={})
    assert fake_store[0][2]["metadata"]["reason"] == "captured during produce"


def test_publish_retries_with_next_ref_after_concurrent_create(tmp_path, monkeypatch):
    candidate = _make_bundle(tmp_path / "candidate.tar.zst")
    uploads: list[str] = []
    monkeypatch.setattr(release, "object_names", lambda: ["state-v6.tar.zst"])

    def upload(ref, bundle, **kwargs):
        uploads.append(ref)
        if len(uploads) == 1:
            raise release.StateRefConflict(ref)

    monkeypatch.setattr(release, "upload_ref", upload)

    assert publish.publish(candidate, reason="", env={}) == "state-v8"
    assert uploads == ["state-v7", "state-v8"]
    assert not (tmp_path / "state-v7.tar.zst").exists()
    assert (tmp_path / "state-v8.tar.zst").read_bytes() == candidate.read_bytes()


def test_publish_rejects_non_export_bundle(fake_store, tmp_path):
    bundle = _make_bundle(tmp_path / "legacy.tar.zst", {"created_at": "2026-01-01"})
    with pytest.raises(SystemExit) as exc:
        publish.publish(bundle, reason="", env={})
    assert "evaluation export bundle" in str(exc.value)
    assert fake_store == []


def test_publish_missing_bundle_exits(fake_store, tmp_path):
    with pytest.raises(SystemExit) as exc:
        publish.publish(tmp_path / "nope.tar.zst", reason="", env={})
    assert "no such bundle" in str(exc.value)


def test_cli_publish_reason_flag_wins_over_env(fake_store, tmp_path, monkeypatch):
    monkeypatch.setenv("REASON", "from-env")
    bundle = _make_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "publish", str(bundle), "--no-verify", "--reason", "from-flag"])
    cli.main()
    assert fake_store[0][2]["metadata"]["reason"] == "from-flag"


def test_cli_publish_without_verify_flags_exits(fake_store, tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "publish", str(bundle)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "--base" in str(exc.value) and "--no-verify" in str(exc.value)
    assert fake_store == []


def test_cli_publish_no_verify_skips_guard(fake_store, tmp_path, monkeypatch, capsys):
    guard_calls = []
    monkeypatch.setattr(cli, "_run_roundtrip", lambda *args, **kwargs: guard_calls.append(args))
    bundle = _make_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "publish", str(bundle), "--no-verify"])
    cli.main()
    assert guard_calls == []
    assert len(fake_store) == 1
    assert "round-trip guard" in capsys.readouterr().out


def test_cli_publish_base_runs_guard_before_upload(fake_store, tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(cli, "_run_roundtrip", lambda *args, **kwargs: order.append("guard"))
    monkeypatch.setattr(release, "upload_ref", lambda *args, **kwargs: order.append("upload"))
    bundle = _make_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "publish", str(bundle), "--base", "http://local:8080"])
    cli.main()
    assert order == ["guard", "upload"]


def test_cli_publish_guard_failure_prevents_upload(fake_store, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_run_roundtrip", lambda *args, **kwargs: sys.exit("round-trip mismatch"))
    bundle = _make_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(sys, "argv", ["evaluation", "publish", str(bundle), "--base", "http://local:8080"])
    with pytest.raises(SystemExit):
        cli.main()
    assert fake_store == []


def test_cli_publish_base_and_no_verify_conflict(fake_store, tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path / "candidate.tar.zst")
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation", "publish", str(bundle), "--base", "http://local:8080", "--no-verify"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert fake_store == []
