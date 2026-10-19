# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer tests for the Docker Compose sandbox provider."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import SandboxCreateError, SandboxStatus
from nhx_evals_sdk.agent_eval.runtimes.sandbox.providers import _compose_transfer as compose_transfer

from packages.nhx_evals_sdk.tests.agent_eval._compose_testkit import _compose_suffix, _create, _provider, _Runner

_FILE_TARGET_OPERATION = "nemo-compose-file-target"
_FILE_REPAIR_OPERATION = "nemo-compose-file-repair"
_DIRECTORY_TARGET_OPERATION = "nemo-compose-directory-target"


def _assert_file_target_operation(command: tuple[str, ...], parent: str, target: str) -> None:
    assert command[:5] == ("exec", "--no-TTY", "agent", "sh", "-c")
    script, operation, parent_arg, target_arg = command[5:]
    assert operation == _FILE_TARGET_OPERATION
    assert parent_arg == parent
    assert target_arg == target
    assert parent not in script
    assert target not in script
    assert 'mkdir -p "$parent"' in script
    assert '[ -L "$target" ]' in script


def _assert_file_repair_operation(command: tuple[str, ...], target: str) -> None:
    assert command[:7] == ("exec", "--no-TTY", "--user", "0", "agent", "sh", "-c")
    script, operation, target_arg, identity = command[7:]
    assert operation == _FILE_REPAIR_OPERATION
    assert target_arg == target
    assert identity == "1001:1002"
    assert target not in script
    assert identity not in script
    assert '[ -f "$target" ] && [ ! -L "$target" ]' in script
    assert 'chown -h "$identity" "$target"' in script
    assert 'chmod u+w "$target"' in script


def _assert_directory_target_operation(command: tuple[str, ...], target: str) -> None:
    assert command[:7] == ("exec", "--no-TTY", "--user", "0", "agent", "sh", "-c")
    script, operation, target_arg = command[7:]
    assert operation == _DIRECTORY_TARGET_OPERATION
    assert target_arg == target
    assert target not in script
    assert '[ ! -L "$target" ]' in script
    assert 'mkdir -p "$target"' in script


@pytest.mark.parametrize(
    ("target", "directory", "message"),
    [
        ("", False, "Container path cannot be empty"),
        ("/", False, "File upload target cannot be the container root"),
        (".", False, "File upload target cannot be the container root"),
        ("work/", False, "File upload target must name an exact file"),
        ("/", True, "Directory upload target cannot be the container root"),
        (".", True, "Directory upload target cannot be the container root"),
    ],
)
def test_upload_target_validation_rejects_unsafe_paths(
    target: str,
    directory: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compose_transfer._normalized_upload_target(target, directory=directory)


def test_upload_target_validation_normalizes_once() -> None:
    assert (
        compose_transfer._normalized_upload_target(
            "ignored/../work/seed.txt",
            directory=False,
        )
        == "/work/seed.txt"
    )
    assert compose_transfer._normalized_upload_target("ignored/../work", directory=True) == "/work"


async def test_upload_file_prepares_parent_as_runtime_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)

    source = tmp_path / "seed.txt"
    source.write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)
    await provider.upload_file(handle, source, "/missing/parent/seed.txt")

    transfer_suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    _assert_file_target_operation(
        transfer_suffixes[0],
        "/missing/parent",
        "/missing/parent/seed.txt",
    )
    assert transfer_suffixes[1:3] == [
        ("cp", str(source), "agent:/missing/parent/seed.txt"),
        (
            "exec",
            "--no-TTY",
            "agent",
            "sh",
            "-c",
            'printf "%s:%s" "$(id -u)" "$(id -g)"',
        ),
    ]
    _assert_file_repair_operation(transfer_suffixes[3], "/missing/parent/seed.txt")
    assert not any(
        args[:7] == ("exec", "--no-TTY", "--user", "0", "agent", "sh", "-c") and "mkdir" in args[7]
        for args in transfer_suffixes
    )
    await provider.close(handle)


async def test_exec_download_and_status_target_configured_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)

    result = await provider.exec(handle, "echo ok", cwd="/work", env={"A": "b"}, stdin=b"x")
    await provider.download_file(handle, "/work/out.txt", tmp_path / "out" / "out.txt")

    assert result.ok
    assert await provider.status(handle) == SandboxStatus.RUNNING
    suffixes = [_compose_suffix(argv) for argv, _, _ in runner.calls if argv[:2] == ("docker", "compose")]
    assert any(args[-4:] == ("agent", "sh", "-lc", "echo ok") for args in suffixes)
    assert ("cp", "agent:/work/out.txt", str(tmp_path / "out" / "out.txt")) in suffixes
    await provider.close(handle)


async def test_directory_transfers_copy_contents_into_prepared_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "source"
    source.mkdir()
    (source / "seed.txt").write_text("seed", encoding="utf-8")
    existing_download = tmp_path / "existing-download"
    existing_download.mkdir()
    absent_download = tmp_path / "absent" / "download"

    transfer_start = len(runner.calls)
    await provider.upload_dir(handle, source, "/work/existing")
    await provider.download_dir(handle, "/out", existing_download)
    await provider.download_dir(handle, "/out", absent_download)

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    _assert_directory_target_operation(suffixes[0], "/work/existing")
    assert suffixes[1:4] == [
        ("cp", f"{source}{os.sep}.", "agent:/work/existing"),
        ("exec", "--no-TTY", "agent", "sh", "-c", 'printf "%s:%s" "$(id -u)" "$(id -g)"'),
        ("exec", "--no-TTY", "--user", "0", "agent", "chown", "-R", "1001:1002", "--", "/work/existing"),
    ]
    assert ("cp", "agent:/out/.", str(existing_download)) in suffixes
    assert ("cp", "agent:/out/.", str(absent_download)) in suffixes
    assert existing_download.is_dir()
    assert absent_download.is_dir()
    await provider.close(handle)


@pytest.mark.parametrize(
    ("target", "target_kind", "exception_type", "message"),
    [
        ("/", None, ValueError, "Directory upload target cannot be the container root"),
        ("/work/link", "symlink", RuntimeError, "Compose upload target preparation failed"),
    ],
)
async def test_upload_dir_rejects_root_and_exact_symlink_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
    target_kind: str | None,
    exception_type: type[Exception],
    message: str,
) -> None:
    runner = _Runner()
    if target_kind is not None:
        runner.directory_target_kinds[target] = target_kind
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "source"
    source.mkdir()
    (source / "seed.txt").write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)

    with pytest.raises(exception_type, match=message):
        await provider.upload_dir(handle, source, target)

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    assert not any(args[:1] == ("cp",) for args in suffixes)
    assert not any(args[:6] == ("exec", "--no-TTY", "--user", "0", "agent", "chown") for args in suffixes)
    await provider.close(handle)


async def test_upload_dir_preparation_failure_stops_before_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    runner.failures.add("directory_prepare")
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "source"
    source.mkdir()
    (source / "seed.txt").write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)

    with pytest.raises(RuntimeError, match="Compose upload target preparation failed"):
        await provider.upload_dir(handle, source, "/work/upload")

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    _assert_directory_target_operation(suffixes[0], "/work/upload")
    assert not any(args[:1] == ("cp",) for args in suffixes)
    await provider.close(handle)


async def test_relative_upload_targets_use_the_same_root_for_every_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source_file = tmp_path / "seed.txt"
    source_file.write_text("seed", encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "nested.txt").write_text("nested", encoding="utf-8")

    transfer_start = len(runner.calls)
    await provider.upload_file(handle, source_file, "missing/parent/seed.txt")
    await provider.upload_file(handle, source_file, "root-seed.txt")
    await provider.upload_dir(handle, source_dir, "workspace")

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    _assert_file_target_operation(
        suffixes[0],
        "/missing/parent",
        "/missing/parent/seed.txt",
    )
    assert suffixes[1:3] == [
        ("cp", str(source_file), "agent:/missing/parent/seed.txt"),
        (
            "exec",
            "--no-TTY",
            "agent",
            "sh",
            "-c",
            'printf "%s:%s" "$(id -u)" "$(id -g)"',
        ),
    ]
    _assert_file_repair_operation(suffixes[3], "/missing/parent/seed.txt")
    _assert_file_target_operation(suffixes[4], "/", "/root-seed.txt")
    assert ("cp", str(source_file), "agent:/root-seed.txt") in suffixes
    _assert_file_repair_operation(suffixes[6], "/root-seed.txt")
    _assert_directory_target_operation(suffixes[7], "/workspace")
    assert ("cp", f"{source_dir}{os.sep}.", "agent:/workspace") in suffixes
    await provider.close(handle)


async def test_upload_file_derives_parent_from_normalized_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "seed.txt"
    source.write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)

    await provider.upload_file(handle, source, "ignored/../root-seed.txt")

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    _assert_file_target_operation(suffixes[0], "/", "/root-seed.txt")
    assert ("cp", str(source), "agent:/root-seed.txt") in suffixes
    _assert_file_repair_operation(suffixes[-1], "/root-seed.txt")
    await provider.close(handle)


async def test_upload_file_does_not_repair_a_preexisting_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    runner.directories.add("/existing/parent")
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "seed.txt"
    source.write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)

    await provider.upload_file(handle, source, "/existing/parent/seed.txt")

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    _assert_file_target_operation(
        suffixes[0],
        "/existing/parent",
        "/existing/parent/seed.txt",
    )
    assert suffixes[1:3] == [
        ("cp", str(source), "agent:/existing/parent/seed.txt"),
        (
            "exec",
            "--no-TTY",
            "agent",
            "sh",
            "-c",
            'printf "%s:%s" "$(id -u)" "$(id -g)"',
        ),
    ]
    _assert_file_repair_operation(suffixes[3], "/existing/parent/seed.txt")
    await provider.close(handle)


@pytest.mark.parametrize(
    ("return_code", "stderr", "timed_out"),
    [
        (1, "token=sensitive-value", False),
        (2, "", False),
        (0, "", True),
    ],
)
async def test_upload_file_preparation_failure_stops_before_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    return_code: int,
    stderr: str,
    timed_out: bool,
) -> None:
    runner = _Runner()
    runner.file_prepare_result = (return_code, "", stderr, timed_out)
    provider = _provider(tmp_path, environment_defaults={"TEST_TOKEN": "sensitive-value"})
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "seed.txt"
    source.write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)

    with pytest.raises(RuntimeError, match="Compose upload target preparation failed") as caught:
        await provider.upload_file(handle, source, "/existing/parent/seed.txt")

    assert "sensitive-value" not in str(caught.value)
    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    _assert_file_target_operation(
        suffixes[0],
        "/existing/parent",
        "/existing/parent/seed.txt",
    )
    assert len(suffixes) == 1
    await provider.close(handle)


@pytest.mark.parametrize("target_kind", ["directory", "symlink", "other"])
async def test_upload_file_rejects_unsafe_exact_target_before_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_kind: str,
) -> None:
    runner = _Runner()
    runner.file_target_kinds["/existing/parent/seed.txt"] = target_kind
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "seed.txt"
    source.write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)

    with pytest.raises(RuntimeError, match="Compose upload target preparation failed"):
        await provider.upload_file(handle, source, "/existing/parent/seed.txt")

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    assert not any(args[:1] == ("cp",) for args in suffixes)
    assert not any(
        args[:5] == ("exec", "--no-TTY", "--user", "0", "agent")
        and any("chown" in arg or "chmod" in arg for arg in args[5:])
        for args in suffixes
    )
    await provider.close(handle)


async def test_repeated_file_upload_never_repairs_the_parent_as_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    first_source = tmp_path / "first.txt"
    first_source.write_text("first", encoding="utf-8")
    second_source = tmp_path / "second.txt"
    second_source.write_text("second", encoding="utf-8")
    transfer_start = len(runner.calls)

    await provider.upload_file(handle, first_source, "/shared/parent/first.txt")
    await provider.upload_file(handle, second_source, "/shared/parent/second.txt")

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    parent_operations = [
        args
        for args in suffixes
        if args[:5] == ("exec", "--no-TTY", "agent", "sh", "-c") and args[-3] == _FILE_TARGET_OPERATION
    ]
    assert len(parent_operations) == 2
    assert runner.directories == {"/shared/parent"}
    assert runner.prepared_file_targets == [
        ("/shared/parent", "/shared/parent/first.txt"),
        ("/shared/parent", "/shared/parent/second.txt"),
    ]
    assert not any(
        args[:5] == ("exec", "--no-TTY", "--user", "0", "agent")
        and "/shared/parent" == args[-1]
        and any("chown" in arg or "chmod" in arg for arg in args[5:])
        for args in suffixes
    )
    repair_operations = [
        args
        for args in suffixes
        if args[:7] == ("exec", "--no-TTY", "--user", "0", "agent", "sh", "-c") and args[-3] == _FILE_REPAIR_OPERATION
    ]
    assert len(repair_operations) == 2
    _assert_file_repair_operation(repair_operations[0], "/shared/parent/first.txt")
    _assert_file_repair_operation(repair_operations[1], "/shared/parent/second.txt")
    await provider.close(handle)


async def test_upload_file_repairs_only_a_writable_regular_leaf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "readonly.txt"
    source.write_text("seed", encoding="utf-8")
    source.chmod(0o444)
    transfer_start = len(runner.calls)

    await provider.upload_file(handle, source, "/work/readonly.txt")

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    assert source.stat().st_mode & 0o777 == 0o444
    _assert_file_repair_operation(suffixes[-1], "/work/readonly.txt")
    assert runner.file_repairs == [("/work/readonly.txt", "1001:1002")]
    assert runner.directories == {"/work"}
    await provider.close(handle)


@pytest.mark.parametrize("target_kind", ["directory", "symlink", "other"])
async def test_upload_file_repair_rejects_unsafe_post_copy_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_kind: str,
) -> None:
    runner = _Runner()
    target = "/work/seed.txt"
    runner.file_target_kinds_after_copy[target] = target_kind
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "seed.txt"
    source.write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)

    with pytest.raises(RuntimeError, match="Compose upload ownership repair failed"):
        await provider.upload_file(handle, source, target)

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    assert any(args[:1] == ("cp",) for args in suffixes)
    assert target in runner.retained_file_targets
    _assert_file_repair_operation(suffixes[-1], target)
    assert runner.file_repairs == []
    await provider.close(handle)


@pytest.mark.parametrize("failure", ["chown", "chmod"])
async def test_upload_file_repair_failure_is_non_atomic_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    runner = _Runner()
    runner.failures.add(failure)
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "seed.txt"
    source.write_text("seed", encoding="utf-8")
    target = "/work/seed.txt"
    transfer_start = len(runner.calls)

    with pytest.raises(RuntimeError, match="Compose upload ownership repair failed"):
        await provider.upload_file(handle, source, target)

    assert target in runner.retained_file_targets
    runner.failures.remove(failure)
    await provider.upload_file(handle, source, target)

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    assert sum(args[:1] == ("cp",) for args in suffixes) == 2
    assert runner.file_repairs == [(target, "1001:1002")]
    await provider.close(handle)


async def test_upload_file_uses_non_login_shell_for_identity_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path)
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "seed.txt"
    source.write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)

    await provider.upload_file(handle, source, "/seed.txt")

    suffixes = [
        _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
    ]
    assert ("exec", "--no-TTY", "agent", "sh", "-c", 'printf "%s:%s" "$(id -u)" "$(id -g)"') in suffixes
    await provider.close(handle)


@pytest.mark.parametrize(
    ("failure", "exception_type", "message"),
    [
        ("mkdir", RuntimeError, "Compose upload target preparation failed"),
        ("copy", RuntimeError, "Compose upload failed"),
        ("identity", SandboxCreateError, "Could not determine target service identity"),
        ("chown", RuntimeError, "Compose upload ownership repair failed"),
    ],
)
async def test_upload_failures_are_ordered_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    exception_type: type[Exception],
    message: str,
) -> None:
    runner = _Runner()
    provider = _provider(tmp_path, environment_defaults={"TEST_TOKEN": "sensitive-value"})
    handle = await _create(monkeypatch, provider, runner)
    source = tmp_path / "seed.txt"
    source.write_text("seed", encoding="utf-8")
    transfer_start = len(runner.calls)
    runner.failures.add(failure)

    try:
        with pytest.raises(exception_type, match=message) as caught:
            await provider.upload_file(handle, source, "/missing/parent/seed.txt")
        assert "sensitive-value" not in str(caught.value)
        suffixes = [
            _compose_suffix(argv) for argv, _, _ in runner.calls[transfer_start:] if argv[:2] == ("docker", "compose")
        ]
        if failure == "mkdir":
            assert not any(args[:1] == ("cp",) for args in suffixes)
        elif failure == "copy":
            assert suffixes[-1][:1] == ("cp",)
            assert not any(args[:6] == ("exec", "--no-TTY", "--user", "0", "agent", "chown") for args in suffixes)
        elif failure == "identity":
            assert any(args[:1] == ("cp",) for args in suffixes)
            assert not any("chown" in args for args in suffixes)
    finally:
        runner.failures.clear()
        await provider.close(handle)
