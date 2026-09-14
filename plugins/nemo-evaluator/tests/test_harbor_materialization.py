# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("harbor")
from nemo_evaluator.api.schemas import HarborTaskDefinition
from nemo_evaluator.api.task_definitions.harbor import HarborTreeSource
from nemo_evaluator.harbor.manifest import inspect_tree
from nemo_evaluator.harbor.materialization import materialize_harbor_tasks, materialize_harbor_tasks_sync
from nemo_evaluator.harbor.tasks import StoredHarborTask
from nemo_evaluator_sdk.agent_eval.taskset_sources import digest_harbor_tree


def snapshot(root, folder="wrapped", entity="stored"):
    manifest = inspect_tree(root, task_dir=folder)
    data = manifest.to_bytes()
    prefix = entity
    objects = {f"{prefix}/manifest.json": data}
    objects.update(
        {f"{prefix}/files/{p.relative_to(root).as_posix()}": p.read_bytes() for p in root.rglob("*") if p.is_file()}
    )
    return StoredHarborTask(
        entity_name=f"default/{entity}",
        revision_digest="a" * 64,
        definition=HarborTaskDefinition(
            kind="harbor",
            tree=HarborTreeSource(
                root_ref=f"default/files#{prefix}/files",
                manifest_ref=f"default/files#{prefix}/manifest.json",
                manifest_digest=hashlib.sha256(data).hexdigest(),
                tree_digest=digest_harbor_tree(root),
            ),
        ),
    ), objects


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "source"
    for name, content in {
        "task.toml": '[task]\nname = "commerce/checkout"\n',
        "instruction.md": "Do it",
        "environment/Dockerfile": "FROM ubuntu",
        "tests/test.sh": "exit 0",
        ".hidden": "included",
        "root-data.txt": "included too",
    }.items():
        path = root / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(content)
    (root / "empty").mkdir()
    (root / "tests/test.sh").chmod(0o751)
    return root


def run(members, objects, destination, transport):
    client = Mock()

    def download(*, workspace, name, path):
        assert workspace == "default" and name == "files"
        if path not in objects:
            raise ValueError("Missing object")
        data = objects[path]

        @contextmanager
        def stream(**kwargs):
            yield iter([data])

        @asynccontextmanager
        async def astream(**kwargs):
            async def chunks():
                await asyncio.sleep(0)
                yield data

            yield chunks()

        return Mock(stream=astream if transport == "async" else stream)

    client.download_file = AsyncMock(side_effect=download) if transport == "async" else Mock(side_effect=download)
    if transport == "async":
        result = asyncio.run(materialize_harbor_tasks(members, files_client=client, destination_root=destination))
    else:
        result = materialize_harbor_tasks_sync(members, files_client=client, destination_root=destination)
    return result


@pytest.mark.parametrize("transport", ["sync", "async"])
def test_complete_tree_round_trip(tree, tmp_path, transport):
    member, objects = snapshot(tree)
    objects["stored/files/unlisted"] = b"must not execute"
    result = run([member], objects, tmp_path / "destination", transport)
    task = result.members[0]
    assert task.task_id == "commerce/checkout" and task.task_dir.name == "wrapped"
    assert task.source.entity_name == "default/stored"
    assert digest_harbor_tree(task.task_dir) == member.definition.tree.tree_digest
    assert result.materialization_digest == digest_harbor_tree(result.dataset_root)
    assert (task.task_dir / "tests/test.sh").stat().st_mode & 0o7777 == 0o755
    assert (task.task_dir / "empty").is_dir()
    assert (task.task_dir / ".hidden").read_text() == "included"
    assert not (task.task_dir / "unlisted").exists()
    again = run([member], objects, tmp_path / "destination", transport)
    assert again.dataset_root != result.dataset_root


@pytest.mark.parametrize("transport", ["sync", "async"])
@pytest.mark.parametrize(
    "failure", ["manifest", "tree", "file", "size", "truncated", "missing", "duplicate", "later-member"]
)
def test_failure_is_atomic(tree, tmp_path, transport, failure):
    member, objects = snapshot(tree)
    members = [member]
    if failure == "manifest":
        objects["stored/manifest.json"] += b" "
    elif failure == "tree":
        member.definition.tree.tree_digest = "0" * 64
    elif failure == "file":
        objects["stored/files/instruction.md"] = b"Xo it"
    elif failure == "size":
        objects["stored/files/instruction.md"] += b"extra"
    elif failure == "truncated":
        objects["stored/files/instruction.md"] = b""
    elif failure == "missing":
        del objects["stored/files/instruction.md"]
    elif failure == "duplicate":
        members.append(member)
    else:
        second, other = snapshot(tree, folder="second", entity="second")
        second.definition.tree.tree_digest = "0" * 64
        objects.update(other)
        members.append(second)
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_text("caller-owned")
    with pytest.raises(ValueError):
        run(members, objects, destination, transport)
    assert list(destination.iterdir()) == [sentinel]


@pytest.mark.parametrize("transport", ["sync", "async"])
@pytest.mark.parametrize("failure", ["instruction", "environment", "config", "identity", "verifier"])
def test_native_invalid_tree_never_published(tree, tmp_path, transport, failure):
    if failure == "environment":
        (tree / "environment/Dockerfile").unlink()
        (tree / "environment").rmdir()
    elif failure == "config":
        (tree / "task.toml").write_text("[invalid TOML")
    elif failure == "identity":
        (tree / "task.toml").write_text("[task]\n")
    else:
        (tree / ("instruction.md" if failure == "instruction" else "tests/test.sh")).unlink()
    member, objects = snapshot(tree)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError):
        run([member], objects, destination, transport)
    assert not list(destination.iterdir())


@pytest.mark.parametrize("transport", ["sync", "async"])
@pytest.mark.parametrize("layout", ["multistep", "separate-verifier"])
def test_native_config_controls_required_artifacts(tree, tmp_path, transport, layout):
    (tree / "tests/test.sh").unlink()
    if layout == "multistep":
        (tree / "instruction.md").unlink()
        (tree / "task.toml").write_text('[task]\nname = "commerce/checkout"\n[[steps]]\nname = "first"\n')
        step = tree / "steps/first"
        (step / "tests").mkdir(parents=True)
        (step / "instruction.md").write_text("Complete step")
        (step / "tests/test.sh").write_text("exit 0")
    else:
        (tree / "task.toml").write_text(
            '[task]\nname = "commerce/checkout"\n[verifier]\nenvironment_mode = "separate"\n'
        )
    member, objects = snapshot(tree)
    result = run([member], objects, tmp_path / "destination", transport)
    assert result.members[0].task_id == "commerce/checkout"


@pytest.mark.parametrize("transport", ["sync", "async"])
def test_duplicate_ids_with_different_folders(tree, tmp_path, transport):
    first, objects = snapshot(tree)
    second, other = snapshot(tree, folder="second", entity="second")
    with pytest.raises(ValueError, match="Duplicate Harbor task IDs"):
        run([first, second], objects | other, tmp_path / "destination", transport)


@pytest.mark.parametrize("transport", ["sync", "async"])
@pytest.mark.parametrize("failure", ["unsafe", "duplicate", "file-ancestor", "reserved", "mode", "symlink"])
def test_untrusted_manifest_rejected_before_files(tree, tmp_path, transport, failure):
    member, objects = snapshot(tree)
    manifest = json.loads(objects["stored/manifest.json"])
    if failure == "unsafe":
        manifest["entries"][0]["path"] = "../outside"
    elif failure == "duplicate":
        manifest["entries"].append(manifest["entries"][0])
    elif failure == "file-ancestor":
        manifest["entries"] = [e for e in manifest["entries"] if e["path"] != "environment"]
    elif failure == "reserved":
        manifest["task_dir"] = "task_template"
    elif failure == "mode":
        manifest["entries"][0]["executable"] = 0o4755
    else:
        manifest["entries"][0]["kind"] = "symlink"
    data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    member.definition.tree.manifest_digest = hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):
        # No file objects: the manifest must fail first.
        run([member], {"stored/manifest.json": data}, tmp_path / "destination", transport)
    assert not list((tmp_path / "destination").iterdir())


@pytest.mark.parametrize("transport", ["sync", "async"])
def test_manifest_stream_limit(tree, tmp_path, transport, monkeypatch):
    member, objects = snapshot(tree)
    monkeypatch.setattr("nemo_evaluator.harbor.materialization.MAX_MANIFEST_BYTES", 10)
    with pytest.raises(ValueError, match="byte limit"):
        run([member], objects, tmp_path / "destination", transport)


@pytest.mark.parametrize("transport", ["sync", "async"])
def test_empty_input_rejected_before_allocation(tmp_path, transport):
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="at least one"):
        run([], {}, destination, transport)
    assert not destination.exists()


@pytest.mark.parametrize("fail", [False, True])
async def test_async_concurrency_is_bounded_and_drained(tree, tmp_path, fail):
    for i in range(20):
        (tree / f"data-{i:02}").write_text("data")
    member, objects = snapshot(tree)
    active = 0
    peak = 0
    completed = 0

    async def download(*, workspace, name, path):
        @asynccontextmanager
        async def stream(**kwargs):
            nonlocal active, peak, completed
            active += 1
            peak = max(peak, active)
            try:

                async def chunks():
                    await asyncio.sleep(0.001)
                    if fail and path.endswith("data-00"):
                        raise ValueError("interrupted file")
                    yield objects[path]

                yield chunks()
                completed += 1
            finally:
                active -= 1

        return Mock(stream=stream)

    client = Mock(download_file=AsyncMock(side_effect=download))
    destination = tmp_path / "destination"
    if fail:
        with pytest.raises(ValueError, match="interrupted file"):
            await materialize_harbor_tasks([member], files_client=client, destination_root=destination)
        assert not list(destination.iterdir())
        assert completed < len(objects)
    else:
        await materialize_harbor_tasks([member], files_client=client, destination_root=destination)
        assert completed == len(objects)
    assert 1 < peak <= 8
    assert active == 0


@pytest.mark.parametrize("transport", ["sync", "async"])
def test_many_small_files_suite(tree, tmp_path, transport):
    import shutil
    import time

    members = []
    objects = {}
    for index in range(10):
        root = tmp_path / f"task-{index}"
        shutil.copytree(tree, root)
        (root / "task.toml").write_text(f'[task]\nname = "suite/task-{index}"\n')
        for file_index in range(100):
            (root / f"file-{file_index:03}.txt").write_bytes(b"x" * 256)
        member, files = snapshot(root, folder=f"task-{index}", entity=f"task-{index}")
        members.append(member)
        objects.update(files)
    start = time.monotonic()
    result = run(members, objects, tmp_path / "destination", transport)
    elapsed = time.monotonic() - start
    assert len(result.members) == 10
    assert [m.task_id for m in result.members] == [f"suite/task-{i}" for i in range(10)]
    print(
        f"Synthetic {transport} suite: tasks=10, requests={len(objects)}, bytes={sum(map(len, objects.values()))}, preparation_seconds={elapsed:.3f}"
    )
