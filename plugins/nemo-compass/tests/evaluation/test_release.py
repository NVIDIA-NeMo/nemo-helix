# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from evaluation import release
from evaluation.registry import Subject

STORE = release.StateStore(endpoint="https://pdx.s8k.io", region="us-east-1", bucket="fixtures")
CREDS = {
    release.ACCESS_KEY_ENV: "team-test",
    release.SECRET_KEY_ENV: "secret",
}


def _subject(name: str, state: str | None = None) -> Subject:
    config = {"type": "intake"}
    if state is not None:
        config["state"] = state
    return Subject(name=name, type="intake", config=config)


def test_latest_ref_numeric_sort():
    names = ["state-v2.tar.zst", "state-v10.tar.zst", "junk.txt", "state-v3-failed.tar.zst"]
    assert release.latest_ref(names) == "state-v10"


def test_latest_ref_empty():
    assert release.latest_ref([]) is None


def test_next_ref():
    assert release.next_ref("state-v10") == "state-v11"
    assert release.next_ref(None) == "state-v1"


def test_state_store_reads_committed_non_secret_config(tmp_path):
    registry = tmp_path / "evaluations.toml"
    registry.write_text(
        'state_s3_endpoint = "https://pdx.s8k.io"\nstate_s3_region = "us-east-1"\nstate_s3_bucket = "fixtures"\n',
        encoding="utf-8",
    )
    assert release.state_store(registry) == STORE


def test_state_store_requires_every_key(tmp_path):
    registry = tmp_path / "evaluations.toml"
    registry.write_text('state_s3_endpoint = "https://pdx.s8k.io"\n', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        release.state_store(registry)
    assert "state_s3_region" in str(exc.value)
    assert "state_s3_bucket" in str(exc.value)


def test_pinned_ref_reads_subject_config():
    assert release.pinned_ref(_subject("tau2-airline", "state-v6")) == "state-v6"
    assert release.pinned_ref(_subject("tau2-retail")) is None


def test_aws_passes_css_credentials_and_endpoint(monkeypatch):
    seen = {}
    monkeypatch.setenv("AWS_SESSION_TOKEN", "unrelated-aws-session")

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert release._aws(STORE, "s3api", "list-buckets", env=CREDS) == "ok"
    assert seen["command"] == [
        "aws",
        "--endpoint-url",
        STORE.endpoint,
        "--region",
        STORE.region,
        "s3api",
        "list-buckets",
    ]
    assert seen["env"]["AWS_ACCESS_KEY_ID"] == "team-test"
    assert seen["env"]["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert "AWS_SESSION_TOKEN" not in seen["env"]


def test_aws_requires_dedicated_credentials(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("missing credentials invoked AWS"))
    with pytest.raises(SystemExit) as exc:
        release._aws(STORE, "s3api", "list-buckets", env={})
    assert release.ACCESS_KEY_ENV in str(exc.value)
    assert release.SECRET_KEY_ENV in str(exc.value)


def test_aws_prints_stderr_on_failure(monkeypatch, capsys):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="s3: some auth error\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        release._aws(STORE, "s3api", "list-buckets", env=CREDS)
    assert "s3: some auth error" in capsys.readouterr().err


def test_aws_missing_binary_exits_with_install_pointer(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "aws")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        release._aws(STORE, "s3api", "list-buckets", env=CREDS)
    assert str(exc.value) == "AWS CLI not found — install it with `brew install awscli`"


def test_object_names_parses_s3_response(monkeypatch):
    monkeypatch.setattr(
        release,
        "_aws",
        lambda *args, **kwargs: json.dumps({"Contents": [{"Key": "state-v6.tar.zst"}, {"Key": "notes.txt"}]}),
    )
    assert release.object_names(store=STORE) == ["state-v6.tar.zst", "notes.txt"]


def test_object_names_handles_empty_bucket(monkeypatch):
    monkeypatch.setattr(release, "_aws", lambda *args, **kwargs: "{}")
    assert release.object_names(store=STORE) == []


def test_resolve_state_explicit_ref_bypasses_subject_pin():
    assert release.resolve_state("state-v2", subject=_subject("tau2-airline", "state-v6")) == "state-v2"
    assert release.resolve_state("state-v2", subject=_subject("tau2-retail")) == "state-v2"
    assert release.resolve_state("state-v2", subject=None) == "state-v2"


@pytest.mark.parametrize("bad", ["state-6", "latest", "v6", "state-v6.tar.zst", ""])
def test_resolve_state_rejects_malformed_ref(tmp_path, bad):
    with pytest.raises(SystemExit) as exc:
        release.resolve_state(bad, subject=_subject("tau2-airline", "state-v6"))
    message = str(exc.value)
    assert bad in message  # names the offender
    assert "state-v<N>" in message  # names the expected pattern
    # the file hint must match each command's real surface: analyze takes --state FILE,
    # restore takes the positional FILE (its --state is refs-only)
    assert "--state FILE" in message and "positional FILE" in message


def test_resolve_state_none_uses_subject_stanza():
    assert release.resolve_state(None, subject=_subject("tau2-airline", "state-v6")) == "state-v6"
    assert release.resolve_state(None, subject=_subject("nvq", "state-v9")) == "state-v9"


def test_resolve_state_rejects_malformed_stanza_pin():
    with pytest.raises(SystemExit) as exc:
        release.resolve_state(None, subject=_subject("tau2-airline", "v6"))
    assert str(exc.value) == ("evaluations.toml state for 'tau2-airline' is 'v6' — expected state-v<N> (e.g. state-v6)")


def test_resolve_state_none_missing_entry_exits_with_guidance():
    with pytest.raises(SystemExit) as exc:
        release.resolve_state(None, subject=_subject("tau2-retail"))
    assert str(exc.value) == (
        "no state configured in evaluations.toml for subject 'tau2-retail' — add state = \"state-vN\" "
        "to its stanza after publishing a fixture, or pass an explicit state "
        "(analyze: --live / --state <state-vN|FILE>; restore: FILE / --state state-vN)"
    )


def test_resolve_state_none_without_subject_exits():
    with pytest.raises(SystemExit) as exc:
        release.resolve_state(None, subject=None)
    assert "--state" in str(exc.value)


def test_download_ref_aws_args_and_return_path(tmp_path, monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_aws(store, *args):
        calls.append(args)
        Path(args[-1]).write_bytes(b"downloaded")
        return ""

    monkeypatch.setattr(release, "_aws", fake_aws)
    dest = tmp_path / "dl"
    result = release.download_ref("state-v4", dest, store=STORE)
    partial = Path(calls[0][-1])
    assert calls == [
        (
            "s3",
            "cp",
            f"s3://{STORE.bucket}/state-v4.tar.zst",
            str(partial),
        )
    ]
    assert partial.parent == result.parent
    assert partial.name.startswith(f".{result.name}.")
    assert partial.suffix == ".partial"
    assert not partial.exists()
    assert result.read_bytes() == b"downloaded"


def test_download_ref_uses_distinct_partial_files_concurrently(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    partials: list[Path] = []

    def download(store, *args):
        partial = Path(args[-1])
        partials.append(partial)
        partial.write_bytes(b"same immutable state")
        barrier.wait(timeout=5)
        return ""

    monkeypatch.setattr(release, "_aws", download)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: release.download_ref("state-v4", tmp_path / "dl", store=STORE), range(2)))

    assert results[0] == results[1]
    assert len(set(partials)) == 2
    assert all(not partial.exists() for partial in partials)
    assert results[0].read_bytes() == b"same immutable state"


def test_download_ref_reuses_cached_file_without_aws(tmp_path, monkeypatch, capsys):
    first = release.download_ref

    def download(store, *args):
        Path(args[-1]).write_bytes(b"cached bytes")
        return ""

    monkeypatch.setattr(release, "_aws", download)
    result = first("state-v4", tmp_path / "dl", store=STORE)
    monkeypatch.setattr(release, "_aws", lambda *args: pytest.fail("cached ref must not invoke aws"))
    assert first("state-v4", tmp_path / "dl", store=STORE) == result
    assert result.read_bytes() == b"cached bytes"
    assert "using cached state-v4.tar.zst" in capsys.readouterr().out


def test_upload_ref_uses_key_and_metadata(tmp_path, monkeypatch):
    bundle = tmp_path / "state-v11.tar.zst"
    bundle.write_bytes(b"fixture")
    calls = []
    monkeypatch.setattr(
        release,
        "_aws",
        lambda store, *args: calls.append(args) or '{"Metadata":{"sha256":"abc"}}',
    )
    release.upload_ref("state-v11", bundle, metadata={"reason": "new data", "sha256": "abc"}, store=STORE)
    assert calls == [
        (
            "s3",
            "cp",
            str(bundle),
            f"s3://{STORE.bucket}/state-v11.tar.zst",
            "--no-overwrite",
            "--metadata",
            '{"reason":"new data","sha256":"abc"}',
        ),
        ("s3api", "head-object", "--bucket", STORE.bucket, "--key", "state-v11.tar.zst", "--output", "json"),
    ]


def test_upload_ref_detects_no_overwrite_conflict(tmp_path, monkeypatch):
    bundle = tmp_path / "state-v11.tar.zst"
    bundle.write_bytes(b"fixture")
    monkeypatch.setattr(release, "_aws", lambda *args: '{"Metadata":{"sha256":"different"}}')
    with pytest.raises(release.StateRefConflict):
        release.upload_ref("state-v11", bundle, metadata={"sha256": "expected"}, store=STORE)


@pytest.mark.parametrize("stderr", ["PreconditionFailed", "ConditionalRequestConflict", "HTTP 412", "HTTP 409"])
def test_upload_ref_reports_conditional_conflict(tmp_path, monkeypatch, stderr):
    bundle = tmp_path / "state-v11.tar.zst"
    bundle.write_bytes(b"fixture")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr=stderr)

    monkeypatch.setattr(release, "_aws", fail)
    with pytest.raises(release.StateRefConflict):
        release.upload_ref("state-v11", bundle, store=STORE)
