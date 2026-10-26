# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "update_python_trust", Path(__file__).parents[1] / "update-python-trust.py"
)
assert SPEC is not None and SPEC.loader is not None
trust = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trust)


@pytest.mark.parametrize(
    ("value", "enabled"),
    [("1", True), (" TRUE ", True), ("Yes", True), (" oN ", True), (None, False), ("0", False), ("false", False)],
)
def test_source_collection_covers_archive_and_go_sources(tmp_path: Path, monkeypatch, value, enabled: bool) -> None:
    if value is None:
        monkeypatch.delenv("NHX_COLLECT_SOURCES", raising=False)
    else:
        monkeypatch.setenv("NHX_COLLECT_SOURCES", value)
    output = tmp_path / "sources"
    output.mkdir()
    monkeypatch.setattr(trust, "SOURCE_OUTPUT", output)
    buffer = io.BytesIO()
    hook = b'def build_wandb_core(*, output_path, **kwargs):\n    output_path.write_bytes(b"rebuilt")\n'
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("wandb-0.31.0/core/hatch.py")
        member.size = len(hook)
        archive.addfile(member, io.BytesIO(hook))
    data = buffer.getvalue()
    metadata = {
        "urls": [
            {
                "packagetype": "sdist",
                "filename": "wandb.tar.gz",
                "url": "https://example.invalid/source",
                "digests": {"sha256": hashlib.sha256(data).hexdigest()},
            }
        ]
    }
    responses = iter([json.dumps(metadata).encode(), data])
    monkeypatch.setattr(trust.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(next(responses)))
    monkeypatch.setattr(trust.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(trust.subprocess, "check_output", lambda *args, **kwargs: "wandb-core: go1.27.3\n")
    collected = []
    monkeypatch.setattr(trust.subprocess, "run", lambda command, **kwargs: collected.append(command))

    binary = trust.rebuild_wandb("0.31.0", tmp_path)

    assert binary.read_bytes() == b"rebuilt"
    assert (output / "wandb.tar.gz").exists() is enabled
    assert bool(collected) is enabled
    if enabled:
        assert collected[0][1] == "/tmp/collect-go-module-sources.py"


def test_update_bundles_skips_unnamed_distributions(tmp_path: Path, monkeypatch) -> None:
    for index, metadata in enumerate(("", "Version: 1.0\n", "Name: CeRtIfI\nVersion: 1.0\n")):
        directory = tmp_path / f"package{index}-1.0.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_text(metadata)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    original_run = trust.subprocess.run
    installs = []

    def probe(command, **kwargs):
        return original_run([command[0], "-S", *command[1:]], stdout=trust.subprocess.PIPE, check=True, **kwargs).stdout

    monkeypatch.setattr(trust.subprocess, "check_output", probe)
    monkeypatch.setattr(trust.subprocess, "run", lambda command, **kwargs: installs.append(command))

    trust.update_bundles(Path(sys.executable))

    assert len(installs) == 1
    assert installs[0][-1] == "certifi==2026.10.19"


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_replace_binary_detaches_shared_files_and_updates_record(tmp_path: Path, link: str) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"old Go executable")
    binary = tmp_path / "wandb-core"
    if link == "symlink":
        binary.symlink_to(original)
    else:
        os.link(original, binary)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"patched Go executable")
    record = tmp_path / "RECORD"
    record.write_text("wandb/bin/wandb-core,old-hash,17\nwandb/__init__.py,unchanged,123\n")

    trust.replace_binary(binary, replacement, record)

    assert original.read_bytes() == b"old Go executable"
    assert binary.read_bytes() == replacement.read_bytes()
    assert not binary.is_symlink()
    assert binary.stat().st_mode & 0o111 == 0o111
    digest = base64.urlsafe_b64encode(hashlib.sha256(replacement.read_bytes()).digest()).decode().rstrip("=")
    with record.open() as handle:
        assert list(csv.reader(handle)) == [
            ["wandb/bin/wandb-core", f"sha256={digest}", str(replacement.stat().st_size)],
            ["wandb/__init__.py", "unchanged", "123"],
        ]


def test_discovers_nested_gym_environments_without_resolving_python_symlinks(tmp_path: Path, monkeypatch) -> None:
    interpreter = tmp_path / "system-python"
    interpreter.touch()
    monkeypatch.setattr(trust.shutil, "which", lambda _: str(interpreter))
    configs = [tmp_path / "gym/agents/a/.venv", tmp_path / "gym/resources/b/.venv"]
    for directory in configs:
        (directory / "bin").mkdir(parents=True)
        (directory / "pyvenv.cfg").touch()
        (directory / "bin/python").symlink_to(interpreter)

    found = trust.python_environments([tmp_path / "gym"])

    assert found == sorted([interpreter, *(directory / "bin/python" for directory in configs)])


def test_updates_bootstrap_wheel_as_well_as_ensurepip_version(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "ensurepip/__init__.py"
    module.parent.mkdir()
    module.write_text('_PIP_VERSION = "26.2.1"\n')
    bundled = module.parent / "_bundled"
    bundled.mkdir()
    old = bundled / "pip-26.2.1-py3-none-any.whl"
    old.write_bytes(b"old certificate bundle")
    monkeypatch.setattr(trust.subprocess, "check_output", lambda *args, **kwargs: str(module))
    monkeypatch.setattr(trust, "fixed_pip_wheel", lambda: b"verified fixed wheel")

    trust.update_ensurepip(tmp_path / "python")

    assert module.read_text() == '_PIP_VERSION = "26.2.2"\n'
    assert not old.exists()
    assert (bundled / "pip-26.2.2-py3-none-any.whl").read_bytes() == b"verified fixed wheel"


@pytest.mark.parametrize("referenced", [False, True])
@pytest.mark.parametrize("index_version", ["wheels-v6", "wheels-v7"])
def test_archive_cleanup_removes_only_obsolete_index_links(
    tmp_path: Path, monkeypatch, referenced: bool, index_version: str
) -> None:
    cache = tmp_path / "cache"
    index = cache / index_version / "pypi/certifi"
    index.mkdir(parents=True)
    archives = {}
    for key, version in (("old", "2026.7.22"), ("fixed", "2026.10.19")):
        archive = cache / "archive-v0" / key
        metadata = archive / f"certifi-{version}.dist-info/METADATA"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(f"Name: certifi\nVersion: {version}\n")
        (index / key).symlink_to(archive)
        archives[key] = archive
    # Index links can also point inside an archive, rather than at its root.
    nested = index / "old-metadata"
    nested.symlink_to(archives["old"] / "certifi-2026.7.22.dist-info/METADATA")
    unrelated = cache / "archive-v0/unrelated"
    unrelated.mkdir()
    (index / "unrelated").symlink_to(unrelated)
    environment = tmp_path / "environment"
    environment.mkdir()
    if referenced:
        (environment / "certifi").symlink_to(archives["old"])
    monkeypatch.setattr(sys, "argv", ["update-python-trust.py", "--root", str(environment)])
    monkeypatch.setattr(trust, "CACHE_ROOTS", (cache,))
    monkeypatch.setattr(trust, "SOURCE_OUTPUT", tmp_path / "sources")
    monkeypatch.setattr(trust, "python_environments", lambda _: [])

    if referenced:
        with pytest.raises(RuntimeError, match="An environment still references obsolete certifi"):
            trust.main()
        assert archives["old"].exists()
        assert (index / "old").exists()
        assert nested.exists()
    else:
        trust.main()
        assert not archives["old"].exists()
        assert not (index / "old").is_symlink()
        assert not nested.is_symlink()
    assert archives["fixed"].exists()
    assert (index / "fixed").exists()
    assert unrelated.exists()
    assert (index / "unrelated").exists()
