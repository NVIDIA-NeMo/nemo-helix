# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from uuid import UUID

import httpx
import pytest
from nemo_evaluator.api.schemas import HarborTaskDefinition
from nemo_evaluator.api.task_definitions.harbor import HarborTreeSource
from nemo_evaluator.harbor.manifest import HarborTreeManifest, TreeFile, inspect_tree
from nemo_evaluator.harbor.publication import publish_harbor_task_tree
from nemo_evaluator_sdk.agent_eval.taskset_sources import digest_harbor_tree
from nemo_platform_plugin.client.errors import InternalServerError
from nemo_platform_plugin.files.client import FilesClient


def source(**overrides):
    return HarborTreeSource.model_validate(
        {
            "root_ref": "default/files#task/files",
            "manifest_ref": "default/files#task/manifest.json",
            "tree_digest": "a" * 64,
            "manifest_digest": "b" * 64,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "path", ["", "/a", "a/../b", "a/./b", "a//b", "a/", "a\\b", "%2e%2e", "a?b", "a#b", "a\x00b", "a."]
)
def test_unsafe_paths_rejected_in_schema_and_manifest(path):
    with pytest.raises(ValueError):
        source(root_ref=f"default/files#{path}")
    with pytest.raises(ValueError):
        TreeFile(path=path, size=0, sha256="a" * 64, executable=0)


@pytest.mark.parametrize(
    "ref",
    [
        "files#path",
        "fileset://default/files#path",
        "default/files/path",
        "default/files#/path",
        "../files#path",
        "default/..#path",
    ],
)
def test_noncanonical_refs_rejected(ref):
    with pytest.raises(ValueError):
        source(root_ref=ref)


def test_manifest_must_be_outside_root():
    with pytest.raises(ValueError, match="outside"):
        source(manifest_ref="default/files#task/files/manifest.json")


def test_schema_replaces_archive_fields():
    definition = HarborTaskDefinition(kind="harbor", tree=source())
    assert HarborTaskDefinition.model_validate_json(definition.model_dump_json()) == definition
    for payload in [
        {"kind": "harbor"},
        {"kind": "harbor", "archive_ref": "default/files#a", "archive_digest": "a" * 64},
        {**definition.model_dump(), "archive_ref": "default/files#a"},
    ]:
        with pytest.raises(ValueError):
            HarborTaskDefinition.model_validate(payload)


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "task"
    for name, content in {
        "task.toml": '[task]\nname = "test/task"\n',
        "instruction.md": "Do it",
        "environment/Dockerfile": "FROM ubuntu",
        "tests/test.sh": "exit 0",
        ".gitignore": "*.txt",
        "ignored.txt": "still included",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (root / "empty").mkdir()
    (root / "tests/test.sh").chmod(0o751)
    return root


def test_manifest_codec_and_whole_tree_semantics(root, tmp_path):
    manifest = inspect_tree(root)
    data = manifest.to_bytes()
    assert HarborTreeManifest.from_bytes(data, hashlib.sha256(data).hexdigest()) == manifest
    assert "ignored.txt" in [entry.path for entry in manifest.entries]
    relocated = tmp_path / "other-folder"
    shutil.copytree(root, relocated)
    os.utime(relocated / "task.toml", (1, 1))
    assert digest_harbor_tree(root) == digest_harbor_tree(relocated)
    assert inspect_tree(relocated).task_dir != manifest.task_dir
    for variant in [data + b" ", json.dumps(manifest.model_dump(mode="json"), indent=2).encode()]:
        with pytest.raises(ValueError, match="canonical"):
            HarborTreeManifest.from_bytes(variant, hashlib.sha256(variant).hexdigest())


@pytest.mark.parametrize("change", ["bytes", "rename", "execute", "empty-dir"])
def test_tree_changes_change_integrity(root, change):
    before = digest_harbor_tree(root)
    if change == "bytes":
        (root / "instruction.md").write_text("changed")
    elif change == "rename":
        (root / "ignored.txt").rename(root / "renamed.txt")
    elif change == "execute":
        (root / "tests/test.sh").chmod(0o644)
    else:
        (root / "empty").rmdir()
    assert digest_harbor_tree(root) != before


@pytest.mark.parametrize("entry", ["relative-link", "absolute-link", "fifo"])
def test_publisher_rejects_links_and_special_files(root, entry):
    if entry == "fifo":
        os.mkfifo(root / "unsafe")
    else:
        (root / "unsafe").symlink_to("instruction.md" if entry == "relative-link" else root / "instruction.md")
    with pytest.raises(ValueError, match="Unsupported"):
        inspect_tree(root)


def test_manifest_limits_and_aliases():
    file = {"kind": "file", "path": "a", "size": 0, "sha256": "a" * 64, "executable": 0}
    for entries in [
        [file, file],
        [{**file, "path": "A"}, file],
        [{**file, "path": "a/b"}],
        [{**file, "executable": 0o200}],
        [{**file, "size": 4 * 1024**3 + 1}],
        [{**file, "size": True}],
    ]:
        with pytest.raises(ValueError):
            HarborTreeManifest.model_validate({"task_dir": "task", "entries": entries})
    with pytest.raises(ValueError, match="total byte"):
        HarborTreeManifest.model_validate(
            {"task_dir": "task", "entries": [{**file, "path": str(i), "size": 4 * 1024**3} for i in range(5)]}
        )


@pytest.mark.parametrize("failure", [None, "interrupted", "corrupt", "bad-native"])
def test_publication_verifies_before_manifest_and_definition(root, failure):
    pytest.importorskip("harbor")
    objects = {}
    writes = []
    reads = []
    if failure == "bad-native":
        (root / "tests/test.sh").unlink()

    def handler(request):
        assert request.headers["authorization"] == "Bearer synthetic"
        path = request.url.path.split("/-/", 1)[1]
        if request.method == "PUT":
            writes.append(path)
            if failure == "interrupted" and len(writes) == 2:
                return httpx.Response(500)
            data = request.read()
            objects[path] = b"x" * len(data) if failure == "corrupt" else data
            return httpx.Response(
                200,
                json={
                    "path": path,
                    "size": len(data),
                    "file_ref": f"default/files#{path}",
                    "file_url": str(request.url),
                },
            )
        reads.append(path)
        return httpx.Response(200, stream=httpx.ByteStream(objects[path]))

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = FilesClient(
            base_url="http://platform.test",
            workspace="default",
            http_client=http_client,
            default_headers={"Authorization": "Bearer synthetic"},
        )
        if failure:
            with pytest.raises(
                {"corrupt": ValueError, "bad-native": FileNotFoundError, "interrupted": InternalServerError}[failure]
            ):
                publish_harbor_task_tree(root, files_client=client, fileset_ref="default/files")
            assert not any(path.endswith("manifest.json") for path in writes)
            if failure == "bad-native":
                assert not writes
            return
        definition = publish_harbor_task_tree(root, files_client=client, fileset_ref="default/files")
        assert writes[-1].endswith("manifest.json") and reads == writes
        assert definition.tree.tree_digest == digest_harbor_tree(root)
        assert definition.config["task"]["name"] == "test/task"
        prefix, files = definition.tree.root_ref.split("#", 1)[1].rsplit("/", 1)
        publication_uuid = UUID(hex=prefix)
        assert publication_uuid.version == 4
        assert publication_uuid.hex == prefix
        assert files == "files"
        assert definition.tree.manifest_ref == f"default/files#{prefix}/manifest.json"
        assert writes == [
            *(f"{prefix}/files/{entry.path}" for entry in inspect_tree(root).entries if isinstance(entry, TreeFile)),
            f"{prefix}/manifest.json",
        ]
        original_objects = objects.copy()
        with ThreadPoolExecutor(max_workers=2) as executor:
            repeated = list(
                executor.map(
                    lambda _: publish_harbor_task_tree(root, files_client=client, fileset_ref="default/files"),
                    range(2),
                )
            )
        assert len({definition.tree.root_ref, *(item.tree.root_ref for item in repeated)}) == 3
        assert all(item.tree.tree_digest == definition.tree.tree_digest for item in repeated)
        assert all(item.tree.manifest_digest == definition.tree.manifest_digest for item in repeated)
        assert all(objects[path] == content for path, content in original_objects.items())
        renamed = root.with_name("renamed")
        shutil.copytree(root, renamed)
        other = publish_harbor_task_tree(renamed, files_client=client, fileset_ref="default/files")
        assert definition.tree.root_ref != other.tree.root_ref
        assert definition.tree.tree_digest == other.tree.tree_digest
        assert definition.tree.manifest_digest != other.tree.manifest_digest

        previous_objects = objects.copy()
        (root / "instruction.md").write_text("A different task instruction\n")
        changed = publish_harbor_task_tree(root, files_client=client, fileset_ref="default/files")
        assert changed.tree.tree_digest != definition.tree.tree_digest
        assert changed.tree.root_ref not in {item.tree.root_ref for item in [definition, *repeated, other]}
        assert all(objects[path] == content for path, content in previous_objects.items())


def test_invalid_fileset_rejected_before_upload(root):
    client = Mock()
    with pytest.raises(ValueError):
        publish_harbor_task_tree(root, files_client=client, fileset_ref="default/files#path")
    client.upload_file.assert_not_called()
