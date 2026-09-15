# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import httpx
import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.cli.main import cli
from test_cli import runner_with


def test_benchmark_download_queues_without_polling(monkeypatch, tmp_path) -> None:
    calls = []

    def handler(request):
        calls.append(request.method)
        assert request.url.path == "/v1/benchmark-runs/bmr_1/archive"
        return httpx.Response(200, json={"status": "missing" if request.method == "GET" else "queued"})

    runner = runner_with(monkeypatch, handler)
    result = runner.invoke(cli, ["benchmark-run", "download", "bmr_1", "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert calls == ["GET", "POST"]
    assert "--wait" in result.output
    assert not (tmp_path / "out").exists()


def test_benchmark_download_waits_and_extracts_one_experiment(monkeypatch, tmp_path) -> None:
    import io
    import tarfile

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in ("bmr_1/result.json", "bmr_1/ev_1__trial/result.json"):
            info = tarfile.TarInfo(name)
            info.size = 2
            archive.addfile(info, io.BytesIO(b"{}"))
    states = iter(["building", "ready"])
    sleeps = []
    monkeypatch.setattr("scaled_evals.cli.main.time.sleep", sleeps.append)

    def handler(request):
        if request.url.path.endswith("/download"):
            return httpx.Response(200, content=stream.getvalue())
        return httpx.Response(200, json={"status": next(states), "download": "/benchmark-runs/bmr_1/archive/download"})

    runner = runner_with(monkeypatch, handler)
    dest = tmp_path / "out"
    result = runner.invoke(cli, ["benchmark-run", "download", "bmr_1", "--wait", "-o", str(dest)])
    assert result.exit_code == 0, result.output
    assert sleeps == [5.0]
    assert (dest / "result.json").read_text() == "{}"
    assert (dest / "ev_1__trial/result.json").is_file()
    assert not (dest / "bmr_1").exists()


def test_benchmark_download_refuses_existing_directory(monkeypatch, tmp_path) -> None:
    def handler(request):
        raise AssertionError("no API requests needed")

    runner = runner_with(monkeypatch, handler)
    result = runner.invoke(cli, ["benchmark-run", "download", "bmr_1", "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_benchmark_download_reports_archive_failure(monkeypatch, tmp_path) -> None:
    runner = runner_with(
        monkeypatch,
        lambda request: httpx.Response(200, json={"status": "failed", "error": "member executions changed"}),
    )
    result = runner.invoke(cli, ["benchmark-run", "download", "bmr_1", "-o", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "member executions changed" in result.output
    assert "--build" in result.output


def test_benchmark_download_rejects_symlinks_without_leaving_output(monkeypatch, tmp_path) -> None:
    import io
    import tarfile

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("bmr_1/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)

    def handler(request):
        if request.url.path.endswith("/download"):
            return httpx.Response(200, content=stream.getvalue())
        return httpx.Response(200, json={"status": "ready", "download": "/benchmark-runs/bmr_1/archive/download"})

    runner = runner_with(monkeypatch, handler)
    result = runner.invoke(cli, ["benchmark-run", "download", "bmr_1", "-o", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "unsafe" in result.output
    assert list(tmp_path.iterdir()) == []


def test_benchmark_download_archive_only_and_partial_notice(monkeypatch, tmp_path) -> None:
    def handler(request):
        if request.url.path.endswith("/download"):
            return httpx.Response(200, content=b"complete tarball")
        return httpx.Response(
            200,
            json={
                "status": "ready",
                "partial": True,
                "download": "/benchmark-runs/bmr_1/archive/download",
            },
        )

    runner = runner_with(monkeypatch, handler)
    dest = tmp_path / "results.tar.gz"
    result = runner.invoke(cli, ["benchmark-run", "download", "bmr_1", "--archive-only", "-o", str(dest)])
    assert result.exit_code == 0, result.output
    assert dest.read_bytes() == b"complete tarball"
    assert list(tmp_path.iterdir()) == [dest]
    assert "Some member trial data is missing" in result.output


def test_benchmark_download_timeout_does_not_cancel_build(monkeypatch, tmp_path) -> None:
    requests = []
    from types import SimpleNamespace

    clock = iter([0, 2])
    monkeypatch.setattr("scaled_evals.cli.main.time", SimpleNamespace(monotonic=lambda: next(clock)))

    def handler(request):
        requests.append(request.method)
        return httpx.Response(200, json={"status": "building"})

    runner = runner_with(monkeypatch, handler)
    result = runner.invoke(
        cli,
        [
            "benchmark-run",
            "download",
            "bmr_1",
            "--wait",
            "--timeout",
            "1",
            "-o",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "server build continues" in result.output
    assert requests == ["GET"]


@pytest.mark.parametrize(
    "flags, method, force", [([], "GET", None), (["--build"], "POST", False), (["--force"], "POST", True)]
)
def test_benchmark_archive_command(monkeypatch, flags, method, force) -> None:
    import json

    def handler(request):
        assert request.method == method
        assert request.url.path == "/v1/benchmark-runs/bmr_1/archive"
        if method == "POST":
            assert json.loads(request.content) == {"force": force}
        return httpx.Response(200, json={"status": "queued"})

    result = runner_with(monkeypatch, handler).invoke(cli, ["benchmark-run", "archive", "bmr_1", *flags])
    assert result.exit_code == 0, result.output


def test_download_keeps_platform_mount_prefix_and_auth(monkeypatch, tmp_path) -> None:
    seen = []

    def handler(request):
        seen.append(request.url.path)
        assert request.headers["authorization"] == "Bearer test-token"
        if request.url.path.endswith("/download"):
            return httpx.Response(200, content=b"archive")
        return httpx.Response(200, json={"status": "ready", "download": "/benchmark-runs/bmr_1/archive/download"})

    result = runner_with(monkeypatch, handler).invoke(
        cli,
        [
            "--base-url",
            "https://platform.example/apis/scaled-evals",
            "--token",
            "test-token",
            "benchmark-run",
            "download",
            "bmr_1",
            "--archive-only",
            "-o",
            str(tmp_path / "archive.tar.gz"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen == [
        "/apis/scaled-evals/v1/benchmark-runs/bmr_1/archive",
        "/apis/scaled-evals/v1/benchmark-runs/bmr_1/archive/download",
    ]


@pytest.mark.parametrize("archive_only", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
def test_download_checks_server_digest_before_publishing(monkeypatch, tmp_path, archive_only, corrupt) -> None:
    import hashlib
    import io
    import tarfile

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("bmr_1/result.json")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"{}"))
    original = stream.getvalue()
    # Change bytes without changing the length; size checks alone cannot detect this.
    downloaded = bytes([original[0] ^ 1]) + original[1:] if corrupt else original

    def handler(request):
        if request.url.path.endswith("/download"):
            return httpx.Response(200, content=downloaded)
        return httpx.Response(
            200,
            json={
                "status": "ready",
                "download": "/benchmark-runs/bmr_1/archive/download",
                "size_bytes": len(original),
                "sha256": hashlib.sha256(original).hexdigest(),
            },
        )

    dest = tmp_path / ("archive.tar.gz" if archive_only else "experiment")
    args = ["benchmark-run", "download", "bmr_1", "-o", str(dest)]
    if archive_only:
        args.append("--archive-only")
    result = runner_with(monkeypatch, handler).invoke(cli, args)
    if corrupt:
        assert result.exit_code == 1
        assert "checksum" in result.output
        assert list(tmp_path.iterdir()) == []
    else:
        assert result.exit_code == 0, result.output
        assert dest.exists()
        assert "Legacy" not in result.output


def test_download_rejects_size_mismatch(monkeypatch, tmp_path) -> None:
    def handler(request):
        if request.url.path.endswith("/download"):
            return httpx.Response(200, content=b"short")
        return httpx.Response(
            200, json={"status": "ready", "size_bytes": 123, "download": "/benchmark-runs/bmr_1/archive/download"}
        )

    dest = tmp_path / "archive.tar.gz"
    result = runner_with(monkeypatch, handler).invoke(
        cli, ["benchmark-run", "download", "bmr_1", "--archive-only", "-o", str(dest)]
    )
    assert result.exit_code == 1
    assert "size" in result.output
    assert list(tmp_path.iterdir()) == []
