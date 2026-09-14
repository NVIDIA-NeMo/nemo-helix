# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish plain Harbor task files, returning a definition only after verified upload."""

import hashlib
import tempfile
import tomllib
import uuid
from pathlib import Path

from nemo_evaluator.api.task_definitions.harbor import HarborTaskDefinition, HarborTreeSource
from nemo_evaluator.harbor.manifest import CHUNK_BYTES, TreeDirectory, TreeFile, inspect_tree
from nemo_evaluator.harbor.tree_io import download_verified
from nemo_evaluator_sdk.agent_eval.taskset_sources import digest_harbor_tree
from nemo_platform_plugin.files.client import FilesClient


def publish_harbor_task_tree(root: Path, *, files_client: FilesClient, fileset_ref: str) -> HarborTaskDefinition:
    """Upload to an existing workspace/fileset; caller publishes the returned task definition.

    The input must be quiescent while copying. Each call uses a fresh prefix, so interrupted
    uploads cannot alter an existing publication. Unreferenced partial uploads may be removed
    by the fileset owner; this helper never deletes or rewrites another publication.
    """
    from harbor.models.task.task import Task

    # Validate the reference before local copying or remote writes.
    if "#" in fileset_ref:
        raise ValueError("fileset_ref must be workspace/fileset without a path")
    HarborTreeSource(
        root_ref=f"{fileset_ref}#files",
        manifest_ref=f"{fileset_ref}#manifest.json",
        manifest_digest="0" * 64,
        tree_digest="0" * 64,
    )
    workspace, name = fileset_ref.split("/", 1)
    manifest = inspect_tree(root)
    with tempfile.TemporaryDirectory(prefix="harbor-publish-") as temporary:
        snapshot = Path(temporary) / manifest.task_dir
        snapshot.mkdir()
        for entry in manifest.entries:
            destination = snapshot / entry.path
            if isinstance(entry, TreeDirectory):
                destination.mkdir(mode=0o755)
            else:
                size = 0
                with (root / entry.path).open("rb") as source, destination.open("xb") as output:
                    while chunk := source.read(CHUNK_BYTES):
                        size += len(chunk)
                        if size > entry.size:
                            raise ValueError("Task grew while preparing publication")
                        output.write(chunk)
                if size != entry.size:
                    raise ValueError("Task changed while preparing publication")
                destination.chmod(0o644 | entry.executable)
        if inspect_tree(snapshot) != manifest:
            raise ValueError("Task changed while preparing publication")
        Task(snapshot)
        if not (snapshot / "environment").is_dir() or not Task.is_valid_dir(snapshot):
            raise ValueError("Invalid native Harbor package or required artifact")
        tree_digest = digest_harbor_tree(snapshot)
        manifest_bytes = manifest.to_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        # A fresh prefix also distinguishes identical contents with different task folder names.
        prefix = uuid.uuid4().hex
        tree = HarborTreeSource(
            root_ref=f"{fileset_ref}#{prefix}/files",
            manifest_ref=f"{fileset_ref}#{prefix}/manifest.json",
            manifest_digest=manifest_digest,
            tree_digest=tree_digest,
        )
        for entry in manifest.entries:
            if not isinstance(entry, TreeFile):
                continue
            with (snapshot / entry.path).open("rb") as stream:
                files_client.upload_file(
                    workspace=workspace,
                    name=name,
                    path=f"{prefix}/files/{entry.path}",
                    content=iter(lambda: stream.read(CHUNK_BYTES), b""),
                ).data()
            # Files upload metadata has no checksum. Verify actual stored bytes before the
            # manifest makes this complete tree available for task revision publication.
            with tempfile.TemporaryFile() as output:
                download_verified(
                    files_client,
                    tree.root_ref + "/" + entry.path,
                    output,
                    limit=entry.size,
                    expected_size=entry.size,
                    expected_digest=entry.sha256,
                )
        files_client.upload_file(
            workspace=workspace,
            name=name,
            path=f"{prefix}/manifest.json",
            content=manifest_bytes,
        ).data()
        with tempfile.TemporaryFile() as output:
            download_verified(
                files_client,
                tree.manifest_ref,
                output,
                limit=len(manifest_bytes),
                expected_size=len(manifest_bytes),
                expected_digest=manifest_digest,
            )
        instruction = snapshot / "instruction.md"
        return HarborTaskDefinition(
            kind="harbor",
            tree=tree,
            instruction=instruction.read_text() if instruction.is_file() else None,
            config=tomllib.loads((snapshot / "task.toml").read_text()),
        )
