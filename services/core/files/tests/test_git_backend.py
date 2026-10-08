# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Git storage backend, run against a real repository over the file protocol."""

import asyncio
import functools
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from nemo_helix_plugin.files.storage_config import SshRemote
from nhx.common.api.common import SecretRef
from nhx.common.files.storage_config import LocalStorageConfig
from nhx.core.files.app.backends import git as git_backend
from nhx.core.files.app.backends.base import ByteRange
from nhx.core.files.app.backends.factory import storage_impl_factory
from nhx.core.files.app.backends.git import (
    GitAccessError,
    GitBackendError,
    GitConfigError,
    GitServerFault,
    GitStorageConfig,
    GitStorageImpl,
    GitUnavailableError,
    _classify_failure,
    _fetched_commits,
    _last_touched,
    _listings,
    _pick_ref,
    _prune_fetch_repositories,
    _repo_lock,
    _tree_entries,
    communicate_within,
    normalize_private_key,
    stop_process_group,
)
from nhx.core.files.app.backends.local import LocalStorageImpl
from nhx.core.files.app.external_hosts import ExternalHostNotAllowedError, validate_external_host
from nhx.core.files.exceptions import NotFoundError
from pydantic import ValidationError

KNOWN_HOSTS = "gitlab.example.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample"
SECRETS = {"ssh_key": "-----BEGIN OPENSSH PRIVATE KEY-----\nkey\n-----END OPENSSH PRIVATE KEY-----"}

# Signing, hooks and default-branch settings in a developer's git config would change these repositories.
_ISOLATED_GIT_ENV = {
    **{name: value for name, value in os.environ.items() if not name.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        env=_ISOLATED_GIT_ENV,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def remote(tmp_path: Path) -> dict[str, str]:
    repo = tmp_path / "remote"
    repo.mkdir()
    _git(repo, "init", "--quiet", "--initial-branch=main")
    _git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    (repo / "agents" / "support").mkdir(parents=True)
    (repo / "agents" / "support" / "agent.yaml").write_text("name: support\n")
    (repo / "agents" / "support" / "prompt.md").write_text("0123456789")
    (repo / "README.md").write_text("readme\n")
    os.symlink("README.md", repo / "link.md")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "first")
    first = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "-a", "v1", "-m", "v1")
    (repo / "README.md").write_text("changed\n")
    _git(repo, "commit", "--quiet", "-am", "second")
    return {"url": f"file://{repo}", "dir": str(repo), "first": first, "main": _git(repo, "rev-parse", "HEAD")}


def _config(url: str, **overrides) -> GitStorageConfig:
    # file:// fails the SSH-only validator; tests reach a local repository instead.
    return GitStorageConfig.model_construct(
        url=url,
        revision=overrides.pop("revision", "main"),
        original_revision=overrides.pop("original_revision", None),
        path=overrides.pop("path", ""),
        ssh_key_secret=SecretRef("key"),
        known_hosts=KNOWN_HOSTS,
        read_chunk_size=overrides.pop("read_chunk_size", 4),
    )


def _durable(root: Path) -> LocalStorageImpl:
    return LocalStorageImpl(LocalStorageConfig(path=str(root)))


def _impl(config: GitStorageConfig, cache_root: Path, secrets: dict[str, str] | None = None) -> GitStorageImpl:
    return GitStorageImpl(
        config,
        SECRETS if secrets is None else secrets,
        cache_root=cache_root,
        durable=_durable(cache_root.parent / "durable"),
        allowed_protocols="file",
    )


async def _read(impl: GitStorageImpl, path: str, byte_range: ByteRange | None = None) -> bytes:
    return b"".join([chunk async for chunk in await impl.download(path, byte_range)])


@pytest.fixture(autouse=True)
def _forget_process_state():
    yield
    _listings.clear()
    _fetched_commits.clear()
    _last_touched.clear()


def _durable_files(tmp_path: Path) -> list[Path]:
    return [path for path in (tmp_path / "durable").rglob("*") if path.is_file()]


def _fetch_repository(cache: Path) -> Path:
    (repository,) = [entry for entry in cache.iterdir() if entry.is_dir()]
    return repository


async def _until_background_work_is_done() -> None:
    while git_backend._background_work:
        await asyncio.sleep(0.05)


class TestGitStorageConfig:
    @pytest.mark.parametrize(
        ("url", "host_url"),
        [
            ("git@gitlab.example.com:org/repo.git", "ssh://gitlab.example.com"),
            ("gitlab.example.com:org/repo.git", "ssh://gitlab.example.com"),
            ("ssh://git@Git.Example.com:2222/org/repo.git", "ssh://git.example.com:2222"),
            ("ssh://git.example.com/org/repo.git", "ssh://git.example.com"),
            ("git@my_host:org/repo.git", "ssh://my_host"),
            ("git@gitlab.example.com.:org/repo.git", "ssh://gitlab.example.com"),
            ("git@host.example:/srv/git/repo.git", "ssh://host.example"),
        ],
    )
    def test_accepts_ssh_remotes(self, url, host_url):
        config = GitStorageConfig(url=url, ssh_key_secret=SecretRef("key"), known_hosts=KNOWN_HOSTS)
        assert config.remote.host_url == host_url
        assert config.get_secret_references() == {"ssh_key": SecretRef("key")}

    @pytest.mark.parametrize(
        "url",
        [
            "https://gitlab.example.com/org/repo.git",
            "file:///etc/repo",
            "ext::sh -c touch% /tmp/pwned",
            "-oProxyCommand=evil:org/repo",
            "ssh://-oProxyCommand=evil/org/repo",
            "/local/path",
            "ssh://evil.example%2F@allowed.example/org/repo.git",
            "ssh://allowed.example?.evil.example/org/repo.git",
            "ssh://allowed.example#.evil.example/org/repo.git",
            "allowed.example?.evil.example:org/repo.git",
            "ssh://git@host.example/org/repo\x00.git",
            "ssh://host.example:٢٢/org/repo.git",
            "ssh://host.example:70000/org/repo.git",
            "ssh://host.example:0/org/repo.git",
            "ssh://host.example:/org/repo.git",
            "ssh://host.example",
            "ssh://host.example/",
            "ftp://host.example/org/repo.git",
            "@host.example:org/repo.git",
            "a@b@host.example:org/repo.git",
            "host..example:org/repo.git",
            "host-.example:org/repo.git",
            "[::1]:org/repo.git",
            "git@host.example:org/my repo.git",
            "dir/host.example:org/repo.git",
        ],
    )
    def test_rejects_urls_that_are_not_plain_ssh_remotes(self, url):
        with pytest.raises(ValidationError):
            GitStorageConfig(url=url, ssh_key_secret=SecretRef("key"), known_hosts=KNOWN_HOSTS)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("revision", "--upload-pack=evil"),
            ("revision", "main:refs/nhx/x"),
            ("revision", "main\n"),
            ("revision", "ma\x00in"),
            ("original_revision", "+refs/*:refs/*"),
            ("original_revision", "-upload-pack=evil"),
            ("original_revision", "main\n"),
            ("path", "agents\x00/calc"),
            ("known_hosts", " "),
            ("known_hosts", "host ssh-ed25519 AAAA\x00"),
        ],
    )
    def test_rejects_values_git_would_reinterpret(self, field, value):
        with pytest.raises(ValidationError):
            GitStorageConfig.model_validate(
                {"url": "git@host.example:org/repo.git", "ssh_key_secret": "key", "known_hosts": KNOWN_HOSTS}
                | {field: value}
            )

    def test_commit_ids_are_lowercased_and_sha256_is_refused(self):
        payload = {"url": "git@host.example:o/r.git", "ssh_key_secret": "key", "known_hosts": KNOWN_HOSTS}
        upper = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        assert GitStorageConfig.model_validate(payload | {"revision": upper}).revision == upper.lower()
        with pytest.raises(ValidationError, match="SHA-256"):
            GitStorageConfig.model_validate(payload | {"revision": "a" * 64})

    @pytest.mark.parametrize(
        ("url", "user", "path"),
        [
            ("git@host.example:org/a@b.git", "git", "org/a@b.git"),
            ("ssh://deploy@host.example/org/a@b.git", "deploy", "org/a@b.git"),
            ("host.example:org/repo.git", None, "org/repo.git"),
            ("git@host.example:22:org/repo.git", "git", "22:org/repo.git"),
        ],
    )
    def test_only_the_authority_is_split_on_at_and_colon(self, url, user, path):
        remote = GitStorageConfig(url=url, ssh_key_secret=SecretRef("key"), known_hosts=KNOWN_HOSTS).remote
        assert (remote.user, remote.path) == (user, path)

    def test_factory_builds_the_git_backend(self):
        config = GitStorageConfig(url="git@host:org/repo.git", ssh_key_secret=SecretRef("key"), known_hosts=KNOWN_HOSTS)
        assert isinstance(storage_impl_factory(config, SECRETS), GitStorageImpl)

    @pytest.mark.parametrize(
        ("url", "canonical"),
        [
            ("git@GitLab.Example.com.:org/repo.git", "git@gitlab.example.com:org/repo.git"),
            ("ssh://git@host.example/org/repo.git", "ssh://git@host.example/org/repo.git"),
            ("ssh://host.example/srv/git/repo.git", "ssh://host.example/srv/git/repo.git"),
            ("host.example:srv/git/repo.git", "host.example:srv/git/repo.git"),
            ("ssh://host.example:022/org/repo.git", "ssh://host.example/org/repo.git"),
            ("ssh://git@host.example:22/org/repo.git", "ssh://git@host.example/org/repo.git"),
            ("ssh://git@host.example:12051/org/repo.git", "ssh://git@host.example:12051/org/repo.git"),
            ("  git@host.example:/srv/git/repo.git ", "git@host.example:/srv/git/repo.git"),
        ],
    )
    def test_url_is_stored_in_the_form_ssh_and_known_hosts_agree_on(self, url, canonical):
        config = GitStorageConfig(url=url, ssh_key_secret=SecretRef("key"), known_hosts=KNOWN_HOSTS)
        assert config.url == canonical
        assert (
            GitStorageConfig(url=canonical, ssh_key_secret=SecretRef("key"), known_hosts=KNOWN_HOSTS).url == canonical
        )

    @pytest.mark.parametrize("size", [0, -1, 16 * 1024 * 1024 + 1])
    def test_read_chunk_size_is_bounded(self, size):
        with pytest.raises(ValidationError, match="read_chunk_size"):
            GitStorageConfig(
                url="git@host:o/r.git", ssh_key_secret=SecretRef("key"), known_hosts=KNOWN_HOSTS, read_chunk_size=size
            )

    def test_default_ssh_port_matches_the_allowlist(self):
        validate_external_host("ssh://gitlab.example.com:22", allowed_hosts=["ssh://gitlab.example.com"])


class TestSshCredentials:
    def test_key_and_known_hosts_are_written_privately_and_removed(self, tmp_path):
        impl = GitStorageImpl(
            GitStorageConfig(
                url="git@gitlab.example.com:org/repo.git", ssh_key_secret=SecretRef("key"), known_hosts=KNOWN_HOSTS
            ),
            {"ssh_key": "line1\r\nline2"},
            cache_root=tmp_path,
        )
        with impl._ssh_env() as env:
            command = env["GIT_SSH_COMMAND"]
            key_path = Path(command.split(" -i ")[1].split(" ")[0])
            assert key_path.read_text() == "line1\nline2\n"
            assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
            assert "StrictHostKeyChecking=yes" in command
            assert "ServerAliveInterval=15" in command
            assert env["GIT_ALLOW_PROTOCOL"] == "ssh"
            assert env["GIT_LITERAL_PATHSPECS"] == "1"
            assert (env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_VALUE_0"]) == ("gc.auto", "0")
        assert not key_path.exists()

    def test_repr_leaves_out_the_private_key(self, tmp_path):
        impl = GitStorageImpl(_config("file:///x"), {"ssh_key": "SECRET-KEY-MATERIAL"}, cache_root=tmp_path)
        assert "SECRET-KEY-MATERIAL" not in repr(impl)

    def test_a_key_pasted_into_a_single_line_input_is_rewrapped(self):
        body = "A" * 100
        flattened = f"-----BEGIN OPENSSH PRIVATE KEY-----{body}-----END OPENSSH PRIVATE KEY-----"
        assert normalize_private_key(flattened) == (
            f"-----BEGIN OPENSSH PRIVATE KEY-----\n{body[:64]}\n{body[64:]}\n-----END OPENSSH PRIVATE KEY-----\n"
        )

    def test_spaces_left_where_newlines_were_are_dropped(self):
        assert normalize_private_key("-----BEGIN RSA PRIVATE KEY----- AB CD -----END RSA PRIVATE KEY-----") == (
            "-----BEGIN RSA PRIVATE KEY-----\nABCD\n-----END RSA PRIVATE KEY-----\n"
        )


class TestClassifyFailure:
    @pytest.mark.parametrize(
        ("stderr", "error"),
        [
            ("Host key verification failed.\nfatal: Could not read from remote repository.", GitAccessError),
            ("git@host: Permission denied (publickey).", GitAccessError),
            ("ssh: Could not resolve hostname nope: nodename nor servname provided", GitUnavailableError),
            ("ERROR: Repository not found.", GitConfigError),
            ("fatal: Server does not allow request for unadvertised object abc123", GitConfigError),
            ("error: unable to create file objects/ab: Permission denied", GitServerFault),
            ("fatal: unable to write new index file: No space left on device", GitServerFault),
            ("error: cannot run ssh: No such file or directory", GitServerFault),
            ("No user exists for uid 10001\nfatal: Could not read from remote repository.", GitServerFault),
            ("fatal: the remote end hung up unexpectedly", GitUnavailableError),
            ("Timeout, server gitlab.example.com not responding.", GitUnavailableError),
            ("ssh: connect to host gitlab.example.com port 22: Host is down", GitUnavailableError),
            (
                "remote: ERROR: The project you were looking for could not be found or you don't have permission "
                "to view it.\nfatal: Could not read from remote repository.",
                GitConfigError,
            ),
            ("fatal: Unable to create '/cache/r/shallow.lock': File exists.", GitBackendError),
            ("git@host: Permission denied (password,keyboard-interactive).", GitAccessError),
            ("git@host: Permission denied (gssapi-with-mic).", GitAccessError),
            ("warning: unrelated\nerror: inflate: data stream error (incorrect header check)", GitBackendError),
            ("fatal: something new", GitBackendError),
            ("Connection closed by 10.0.0.1 port 22", GitUnavailableError),
            ("fetch-pack: unexpected disconnect while reading sideband packet\nfatal: early EOF", GitUnavailableError),
        ],
    )
    def test_maps_stderr_to_storage_errors(self, stderr, error):
        assert type(_classify_failure(stderr, "repo")) is error

    def test_names_the_cause_not_gits_closing_advice(self):
        stderr = (
            "ssh: Could not resolve hostname nope.invalid: nodename nor servname provided\r\n"
            "fatal: Could not read from remote repository.\n\n"
            "Please make sure you have the correct access rights\nand the repository exists.\n"
        )
        message = str(_classify_failure(stderr, "ssh://nope.invalid/o/r.git"))
        assert "Could not resolve hostname nope.invalid" in message
        assert "repository exists" not in message


class TestProcessGroups:
    async def test_stopping_a_group_lets_it_clean_up_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(git_backend, "_GRACE_SECONDS", 30.0)
        marker = tmp_path / "cleaned"
        ready = tmp_path / "ready"
        cleanup = (
            "import pathlib, signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path(sys.argv[1]).touch(), sys.exit(0)))\n"
            "pathlib.Path(sys.argv[2]).touch()\n"
            "time.sleep(30)\n"
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", cleanup, str(marker), str(ready), start_new_session=True
        )
        while not ready.exists():
            await asyncio.sleep(0.05)
        await stop_process_group(proc)
        assert marker.exists()

    async def test_a_timeout_stops_children_holding_the_pipes(self):
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "sleep 30 & sleep 30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await communicate_within(proc, 0.5)
        assert time.monotonic() - started < 10

    async def test_a_cancelled_request_stops_its_process(self):
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "sleep 30 & sleep 30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        task = asyncio.create_task(communicate_within(proc, 60))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert proc.returncode is not None


class TestRefResolution:
    async def test_branch_pins_to_its_head(self, remote, tmp_path):
        resolved = await _impl(_config(remote["url"]), tmp_path / "cache").resolve_config()
        assert resolved.revision == remote["main"]
        assert resolved.original_revision == "main"
        assert resolved.tracked_revision == "main"

    async def test_a_tag_wins_over_a_branch_of_the_same_name_as_in_git(self, remote, tmp_path):
        _git(Path(remote["dir"]), "branch", "v1", "main")
        resolved = await _impl(_config(remote["url"], revision="v1"), tmp_path / "cache").resolve_config()
        assert resolved.revision == remote["first"]

    @pytest.mark.parametrize(
        ("revision", "expected"),
        [("tags/v1", "first"), ("heads/main", "main"), ("refs/heads/v1", "main"), ("pull/5/head", "first")],
    )
    async def test_names_under_refs_resolve_as_in_git(self, remote, tmp_path, revision, expected):
        _git(Path(remote["dir"]), "branch", "v1", "main")
        _git(Path(remote["dir"]), "update-ref", "refs/pull/5/head", remote["first"])
        resolved = await _impl(_config(remote["url"], revision=revision), tmp_path / "cache").resolve_config()
        assert resolved.revision == remote[expected]

    async def test_annotated_tag_pins_to_the_commit(self, remote, tmp_path):
        for revision in ("v1", "refs/tags/v1"):
            resolved = await _impl(_config(remote["url"], revision=revision), tmp_path / "cache").resolve_config()
            assert resolved.revision == remote["first"]

    def test_peeled_tag_line_wins_over_the_tag_object(self):
        refs = {"refs/tags/v1": "tag-object", "refs/tags/v1^{}": "commit"}
        assert _pick_ref(refs, "refs/tags/v1") == "commit"
        assert _pick_ref(refs, "v1") == "commit"

    async def test_head_pins_to_the_default_branch(self, remote, tmp_path):
        resolved = await _impl(_config(remote["url"], revision="HEAD"), tmp_path / "cache").resolve_config()
        assert resolved.revision == remote["main"]

    async def test_full_sha_is_kept(self, remote, tmp_path):
        resolved = await _impl(_config(remote["url"], revision=remote["first"]), tmp_path / "cache").resolve_config()
        assert resolved.revision == remote["first"]
        assert resolved.tracked_revision is None

    async def test_a_sha256_repository_is_refused(self, remote, tmp_path, monkeypatch):
        async def sha256_refs(self, *_args, **_kwargs):
            return f"{'a' * 64}\trefs/heads/main\n".encode()

        monkeypatch.setattr(GitStorageImpl, "_remote_git", sha256_refs)
        with pytest.raises(GitConfigError, match="SHA-256"):
            await _impl(_config(remote["url"], revision="main"), tmp_path / "cache").resolve_config()

    async def test_unknown_ref_is_a_config_error(self, remote, tmp_path):
        with pytest.raises(GitConfigError, match="no branch or tag named 'nope'"):
            await _impl(_config(remote["url"], revision="nope"), tmp_path / "cache").resolve_config()

    async def test_missing_repository_is_a_config_error(self, tmp_path):
        with pytest.raises(GitConfigError):
            await _impl(_config(f"file://{tmp_path}/missing"), tmp_path / "cache").resolve_config()

    async def test_empty_key_secret_is_a_config_error(self, remote, tmp_path):
        with pytest.raises(GitConfigError, match="SSH key secret"):
            await _impl(_config(remote["url"]), tmp_path / "cache", secrets={"ssh_key": "  "}).resolve_config()


class TestReads:
    async def test_lists_regular_files_and_skips_symlinks(self, remote, tmp_path):
        files = await _impl(_config(remote["url"]), tmp_path / "cache").list_files()
        assert {(f.path, f.size) for f in files} == {
            ("README.md", 8),
            ("agents/support/agent.yaml", 14),
            ("agents/support/prompt.md", 10),
        }

    async def test_paths_are_relative_to_the_configured_directory(self, remote, tmp_path):
        impl = _impl(_config(remote["url"], path="agents"), tmp_path / "cache")
        assert sorted(f.path for f in await impl.list_files()) == ["support/agent.yaml", "support/prompt.md"]
        assert [f.path for f in await impl.list_files("support/agent.yaml")] == ["support/agent.yaml"]
        file = await impl.get_file("support/prompt.md")
        assert (file.path, file.size) == ("support/prompt.md", 10)

    @pytest.mark.parametrize(
        "path", ["support", "support/missing.md", "../README.md", "support/./prompt.md", "", ":!README.md", "*.md"]
    )
    async def test_only_an_exact_file_path_is_found(self, remote, tmp_path, path):
        with pytest.raises(NotFoundError):
            await _impl(_config(remote["url"], path="agents"), tmp_path / "cache").get_file(path)

    async def test_missing_directory_is_a_config_error(self, remote, tmp_path):
        with pytest.raises(GitConfigError):
            await _impl(_config(remote["url"], path="agents/nope"), tmp_path / "cache").list_files()

    async def test_streams_a_whole_file_and_a_byte_range(self, remote, tmp_path):
        impl = _impl(_config(remote["url"], path="agents/support"), tmp_path / "cache")
        assert await _read(impl, "agent.yaml") == b"name: support\n"
        assert await _read(impl, "prompt.md", ByteRange(start=3, end=8)) == b"345678"

    async def test_symlink_is_not_served(self, remote, tmp_path):
        with pytest.raises(NotFoundError):
            await _read(_impl(_config(remote["url"]), tmp_path / "cache"), "link.md")

    def test_tree_listing_skips_names_that_are_not_utf8(self):
        output = b"100644 blob abc       5\tok.txt\x00100644 blob def       5\tbad-\xff.txt\x00"
        assert [entry.path for entry in _tree_entries(output)] == ["ok.txt"]


class TestListings:
    async def test_a_listed_commit_is_listed_again_without_git_or_the_remote(self, remote, tmp_path, monkeypatch):
        config = _config(remote["url"], revision=remote["first"])
        await _impl(config, tmp_path / "cache").list_files()
        _listings.clear()
        shutil.rmtree(remote["dir"])
        shutil.rmtree(tmp_path / "cache")

        async def no_git(*_args, **_kwargs):
            raise AssertionError("listing a materialized commit ran git")

        monkeypatch.setattr(GitStorageImpl, "_run", no_git)
        assert len(await _impl(config, tmp_path / "cache").list_files()) == 3

    async def test_listing_stores_no_file_contents(self, remote, tmp_path):
        await _impl(_config(remote["url"]), tmp_path / "cache").list_files()
        assert [path.suffix for path in _durable_files(tmp_path)] == [".json"]

    async def test_a_listing_is_read_from_memory_after_the_first_request(self, remote, tmp_path, monkeypatch):
        config = _config(remote["url"], revision=remote["main"])
        await _impl(config, tmp_path / "cache").list_files()
        downloads = 0
        download = LocalStorageImpl.download

        async def counting(self, path, byte_range):
            nonlocal downloads
            downloads += 1
            return await download(self, path, byte_range)

        monkeypatch.setattr(LocalStorageImpl, "download", counting)
        for _ in range(3):
            assert len(await _impl(config, tmp_path / "cache").list_files()) == 3
        assert downloads == 0

    async def test_concurrent_reads_of_a_new_commit_fetch_it_once(self, remote, tmp_path, monkeypatch):
        fetches = 0
        fetch = GitStorageImpl._fetch

        async def counting(self, repo, sha):
            nonlocal fetches
            fetches += 1
            await fetch(self, repo, sha)

        monkeypatch.setattr(GitStorageImpl, "_fetch", counting)
        config = _config(remote["url"], revision=remote["main"])
        results = await asyncio.gather(*[_impl(config, tmp_path / "cache").list_files() for _ in range(4)])
        assert all(len(files) == 3 for files in results)
        assert fetches == 1

    async def test_a_listing_finishes_after_its_request_is_cancelled(self, remote, tmp_path, monkeypatch):
        fetches = 0
        released = asyncio.Event()
        fetch = GitStorageImpl._fetch

        async def held(self, repo, sha):
            nonlocal fetches
            fetches += 1
            await released.wait()
            await fetch(self, repo, sha)

        monkeypatch.setattr(GitStorageImpl, "_fetch", held)
        config = _config(remote["url"], revision=remote["main"])
        request = asyncio.create_task(_impl(config, tmp_path / "cache").list_files())
        await asyncio.sleep(0.2)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        released.set()
        await _until_background_work_is_done()

        assert len(await _impl(config, tmp_path / "cache").list_files()) == 3
        assert fetches == 1

    async def test_an_unreadable_listing_is_rebuilt(self, remote, tmp_path):
        config = _config(remote["url"], revision=remote["main"])
        await _impl(config, tmp_path / "cache").list_files()
        _listings.clear()
        for listing in _durable_files(tmp_path):
            listing.write_text("{not json")
        assert len(await _impl(config, tmp_path / "cache").list_files()) == 3

    async def test_a_storage_failure_reading_a_listing_is_not_hidden(self, remote, tmp_path, monkeypatch):
        config = _config(remote["url"], revision=remote["main"])
        await _impl(config, tmp_path / "cache").list_files()
        _listings.clear()

        async def broken(*_args, **_kwargs):
            raise RuntimeError("storage is down")

        monkeypatch.setattr(LocalStorageImpl, "download", broken)
        with pytest.raises(RuntimeError, match="storage is down"):
            await _impl(config, tmp_path / "cache").list_files()

    @pytest.mark.parametrize(
        "damage",
        [
            {"commit": "b" * 40},
            {"files": [["README.md", "a" * 40, -1]]},
            {"files": [["README.md", "a" * 40, True]]},
            {"files": [["README.md", "not-a-blob", 1]]},
            {"files": [["README.md", "a" * 40]]},
            {"files": "README.md"},
        ],
    )
    async def test_a_listing_of_the_wrong_commit_or_shape_is_rebuilt(self, remote, tmp_path, damage):
        config = _config(remote["url"], revision=remote["main"])
        await _impl(config, tmp_path / "cache").list_files()
        _listings.clear()
        (listing,) = _durable_files(tmp_path)
        listing.write_text(json.dumps(json.loads(listing.read_text()) | damage))

        files = await _impl(config, tmp_path / "cache").list_files()
        assert {(file.path, file.size) for file in files} == {
            ("README.md", 8),
            ("agents/support/agent.yaml", 14),
            ("agents/support/prompt.md", 10),
        }

    async def test_listings_in_memory_are_bounded_by_their_files(self, remote, tmp_path, monkeypatch):
        monkeypatch.setattr(git_backend, "_LISTED_FILES_IN_MEMORY", 4)
        await _impl(_config(remote["url"]), tmp_path / "cache").list_files()
        await _impl(_config(remote["url"], path="agents"), tmp_path / "cache").list_files()
        assert [len(listing) for listing in _listings.values()] == [2]

    async def test_a_listing_that_cannot_be_stored_is_still_served(self, remote, tmp_path, monkeypatch):
        async def full(*_args, **_kwargs):
            raise OSError("No space left on device")

        monkeypatch.setattr(LocalStorageImpl, "upload", full)
        assert len(await _impl(_config(remote["url"]), tmp_path / "cache").list_files()) == 3

    async def test_a_commit_with_too_many_files_is_refused(self, remote, tmp_path, monkeypatch):
        monkeypatch.setattr(git_backend, "_MAX_LISTED_FILES", 2)
        with pytest.raises(GitConfigError, match="more than 2 files"):
            await _impl(_config(remote["url"]), tmp_path / "cache").list_files()


class TestDownloads:
    async def test_cache_keys_name_blobs_so_unchanged_files_share_them(self, remote, tmp_path):
        first = _impl(_config(remote["url"], revision=remote["first"]), tmp_path / "cache")
        main = _impl(_config(remote["url"], revision=remote["main"]), tmp_path / "cache")
        prefix = await first.get_cache_path_key()
        assert prefix == await main.get_cache_path_key()
        for impl in (first, main):
            assert (await impl.get_cache_path_key("agents/support/agent.yaml") or "").startswith(f"{prefix}/")
        assert await first.get_cache_path_key("agents/support/agent.yaml") == await main.get_cache_path_key(
            "agents/support/agent.yaml"
        )
        assert await first.get_cache_path_key("README.md") != await main.get_cache_path_key("README.md")
        assert await main.get_cache_path_key("missing.md") is None

    @pytest.mark.parametrize(("start", "end"), [(0, 0), (3, 8), (4, 7), (9, 9), (0, 9)])
    async def test_byte_ranges_across_chunk_boundaries(self, remote, tmp_path, start, end):
        impl = _impl(_config(remote["url"], path="agents/support"), tmp_path / "cache")
        assert await _read(impl, "prompt.md", ByteRange(start=start, end=end)) == b"0123456789"[start : end + 1]

    async def test_a_read_runs_one_git_process_without_waiting_for_the_repository(self, remote, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        impl = _impl(_config(remote["url"], revision=remote["main"]), cache)
        await impl.list_files()
        started: list[tuple[str, ...]] = []
        spawn = asyncio.create_subprocess_exec

        async def recording(*args, **kwargs):
            started.append(args)
            return await spawn(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", recording)
        async with _repo_lock(_fetch_repository(cache)):
            content = await asyncio.wait_for(_read(impl, "README.md"), 10)
        assert content == b"changed\n"
        assert [args[3] for args in started] == ["cat-file"]

    async def test_a_removed_fetch_repository_is_fetched_again_on_read(self, remote, tmp_path):
        impl = _impl(_config(remote["url"], revision=remote["main"]), tmp_path / "cache")
        await impl.list_files()
        shutil.rmtree(tmp_path / "cache")
        assert await _read(impl, "README.md") == b"changed\n"


class TestColdRepositories:
    async def test_looking_a_file_up_fetches_its_commit_before_the_download_starts(self, remote, tmp_path, monkeypatch):
        config = _config(remote["url"], revision=remote["main"])
        await _impl(config, tmp_path / "cache").list_files()
        shutil.rmtree(tmp_path / "cache")
        _fetched_commits.clear()

        impl = _impl(config, tmp_path / "cache")
        await impl.get_file("README.md")

        async def no_fetch(*_args, **_kwargs):
            raise AssertionError("the download fetched")

        monkeypatch.setattr(GitStorageImpl, "_fetch", no_fetch)
        assert await _read(impl, "README.md") == b"changed\n"

    async def test_reads_keep_a_repository_from_looking_idle(self, remote, tmp_path):
        impl = _impl(_config(remote["url"], revision=remote["main"]), tmp_path / "cache")
        await impl.list_files()
        repository = _fetch_repository(tmp_path / "cache")
        week_ago = time.time() - 8 * 24 * 60 * 60
        os.utime(repository, (week_ago, week_ago))
        _last_touched.clear()

        await _read(impl, "README.md")
        assert repository.stat().st_mtime > week_ago + 24 * 60 * 60

    async def test_finding_a_commit_already_fetched_counts_as_use(self, remote, tmp_path):
        impl = _impl(_config(remote["url"], revision=remote["main"]), tmp_path / "cache")
        await impl.get_file("README.md")
        repository = _fetch_repository(tmp_path / "cache")
        week_ago = time.time() - 8 * 24 * 60 * 60
        os.utime(repository, (week_ago, week_ago))
        _last_touched.clear()

        await _impl(_config(remote["url"], revision=remote["main"]), tmp_path / "cache").get_file("README.md")
        assert repository.stat().st_mtime > week_ago + 24 * 60 * 60

    async def test_use_is_recorded_at_most_once_a_minute(self, tmp_path):
        repository = tmp_path / "repo"
        repository.mkdir()
        await git_backend._touch(repository)
        week_ago = time.time() - 8 * 24 * 60 * 60
        os.utime(repository, (week_ago, week_ago))
        await git_backend._touch(repository)
        assert repository.stat().st_mtime == pytest.approx(week_ago)

    async def test_fetching_again_prunes_other_repositories(self, remote, tmp_path):
        cache = tmp_path / "cache"
        impl = _impl(_config(remote["url"], revision=remote["main"]), cache)
        await impl.list_files()
        shutil.rmtree(_fetch_repository(cache))
        _fetched_commits.clear()
        idle = cache / "idle"
        idle.mkdir()
        week_ago = time.time() - 8 * 24 * 60 * 60
        os.utime(idle, (week_ago, week_ago))

        await impl.get_file("README.md")
        assert not idle.exists()


class TestServerFaults:
    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    async def test_an_unwritable_cache_is_a_server_fault(self, remote, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            with pytest.raises(GitServerFault, match="not writable"):
                await _impl(_config(remote["url"]), locked / "cache").list_files()
        finally:
            locked.chmod(0o700)


class TestAccess:
    async def test_a_cached_commit_is_read_without_contacting_the_remote(self, remote, tmp_path):
        config = _config(remote["url"], revision=remote["main"])
        await _impl(config, tmp_path / "cache").list_files()
        shutil.rmtree(remote["dir"])
        assert len(await _impl(config, tmp_path / "cache").list_files()) == 3

    @pytest.mark.parametrize("pinned", [True, False])
    async def test_registering_a_cached_commit_asks_the_remote_with_the_new_key(
        self, remote, tmp_path, monkeypatch, pinned
    ):
        monkeypatch.setattr("nhx.core.files.app.backends.git.validate_external_host", lambda _url: None)
        monkeypatch.setattr(GitStorageConfig, "remote", property(lambda _self: SshRemote(None, "h", None, "p")))
        revision = remote["main"] if pinned else "main"
        await _impl(_config(remote["url"], revision=remote["main"]), tmp_path / "cache").list_files()
        keys: list[str] = []
        fetch = GitStorageImpl._fetch

        async def recording(self, repo, sha):
            keys.append(self.secrets["ssh_key"])
            await fetch(self, repo, sha)

        monkeypatch.setattr(GitStorageImpl, "_fetch", recording)
        await _impl(
            _config(remote["url"], revision=revision), tmp_path / "cache", {"ssh_key": "second"}
        ).validate_storage()
        assert keys == (["second"] if pinned else [])

    async def test_registering_a_cold_pinned_commit_checks_for_it_only_around_the_fetch(
        self, remote, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("nhx.core.files.app.backends.git.validate_external_host", lambda _url: None)
        monkeypatch.setattr(GitStorageConfig, "remote", property(lambda _self: SshRemote(None, "h", None, "p")))
        checks = 0
        has_commit = GitStorageImpl._has_commit

        async def counting(self, repo, sha):
            nonlocal checks
            checks += 1
            return await has_commit(self, repo, sha)

        monkeypatch.setattr(GitStorageImpl, "_has_commit", counting)
        impl = _impl(_config(remote["url"], revision=remote["main"], path="agents"), tmp_path / "cache")
        await impl.validate_storage()
        assert checks == 2

    async def test_registering_a_pinned_commit_skips_ls_remote(self, remote, tmp_path, monkeypatch):
        monkeypatch.setattr("nhx.core.files.app.backends.git.validate_external_host", lambda _url: None)
        monkeypatch.setattr(GitStorageConfig, "remote", property(lambda _self: SshRemote(None, "h", None, "p")))

        async def no_ls_remote(self):
            raise AssertionError("ls-remote ran for a pinned commit")

        monkeypatch.setattr(GitStorageImpl, "_ls_remote", no_ls_remote)
        await _impl(_config(remote["url"], revision=remote["main"]), tmp_path / "cache").validate_storage()

    async def test_a_host_outside_the_allowlist_is_refused_before_git_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "nhx.core.files.app.backends.git.validate_external_host",
            functools.partial(validate_external_host, allowed_hosts=["ssh://allowed.example"]),
        )

        async def no_git(*_args, **_kwargs):
            raise AssertionError("git ran for a host outside the allowlist")

        monkeypatch.setattr(GitStorageImpl, "_run", no_git)
        config = GitStorageConfig(
            url="git@elsewhere.example:org/repo.git", ssh_key_secret=SecretRef("key"), known_hosts=KNOWN_HOSTS
        )
        with pytest.raises(ExternalHostNotAllowedError):
            await _impl(config, tmp_path / "cache").validate_storage()

    async def test_a_key_the_remote_refuses_cannot_register_a_cached_commit(self, remote, tmp_path, monkeypatch):
        monkeypatch.setattr("nhx.core.files.app.backends.git.validate_external_host", lambda _url: None)
        monkeypatch.setattr(GitStorageConfig, "remote", property(lambda _self: SshRemote(None, "h", None, "p")))
        config = _config(remote["url"], revision=remote["main"])
        await _impl(config, tmp_path / "cache").list_files()

        async def refuse(self, repo, sha):
            raise GitAccessError("The SSH key was rejected")

        monkeypatch.setattr(GitStorageImpl, "_fetch", refuse)
        with pytest.raises(GitAccessError):
            await _impl(config, tmp_path / "cache", {"ssh_key": "second"}).validate_storage()


class TestRegistration:
    async def test_a_client_supplied_original_revision_cannot_repoint_a_pin(self, remote, tmp_path):
        config = _config(remote["url"], revision=remote["first"], original_revision="main")
        resolved = await _impl(config, tmp_path / "cache").resolve_config()
        assert (resolved.revision, resolved.original_revision) == (remote["first"], remote["first"])
        assert resolved.tracked_revision is None

    async def test_validate_storage_refuses_a_pinned_commit_the_remote_lacks(self, remote, tmp_path, monkeypatch):
        monkeypatch.setattr("nhx.core.files.app.backends.git.validate_external_host", lambda _url: None)
        monkeypatch.setattr(GitStorageConfig, "remote", property(lambda _self: SshRemote(None, "h", None, "p")))
        with pytest.raises(GitConfigError):
            await _impl(_config(remote["url"], revision="a" * 40), tmp_path / "cache").validate_storage()

    async def test_validate_storage_does_not_fetch_a_branch(self, remote, tmp_path, monkeypatch):
        monkeypatch.setattr("nhx.core.files.app.backends.git.validate_external_host", lambda _url: None)
        monkeypatch.setattr(GitStorageConfig, "remote", property(lambda _self: SshRemote(None, "h", None, "p")))
        await _impl(_config(remote["url"], revision="main"), tmp_path / "cache").validate_storage()
        assert not (tmp_path / "cache").exists()


class TestFetchRepositoryPruning:
    async def test_idle_repositories_are_removed_and_busy_ones_kept(self, tmp_path):
        week_ago = time.time() - 8 * 24 * 60 * 60
        idle, busy, recent = (tmp_path / name for name in ("idle", "busy", "recent"))
        for directory in (idle, busy, recent):
            directory.mkdir()
        for directory in (idle, busy):
            os.utime(directory, (week_ago, week_ago))

        async with _repo_lock(busy):
            await _prune_fetch_repositories(tmp_path, tmp_path / "fetching")

        assert (idle.exists(), busy.exists(), recent.exists()) == (False, True, True)
