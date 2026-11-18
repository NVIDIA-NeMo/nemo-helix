#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify fileset commit markers through the Helix SDK and the configured context.

Run from the repository root against a server containing the commit marker change:
    uv run services/core/files/script/demo_fileset_revision.py --workspace default
"""

import argparse
from uuid import uuid4

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput, UpdateFilesetRequest


def require_commit_revision(value: str | None) -> str:
    if not value:
        raise RuntimeError("No commit revision returned. Use an SDK and server containing the fileset commit change.")
    return value


def verify_unchanged(files: FilesClient, fileset: FilesetOutput, operation: str) -> None:
    observed = files.get_fileset_by_id(workspace=fileset.workspace, fileset_id=fileset.id).data()
    if observed.commit_revision != fileset.commit_revision:
        raise RuntimeError(f"{operation}: an ordinary mutation changed the commit marker")
    print(f"{operation} verified: commit marker unchanged")


def verify_commit(files: FilesClient, fileset: FilesetOutput) -> str:
    committed = files.commit_fileset(workspace=fileset.workspace, name=fileset.name).data()
    token = require_commit_revision(committed.commit_revision)
    observed = files.get_fileset_by_id(workspace=fileset.workspace, fileset_id=fileset.id).data()
    if token == fileset.commit_revision or observed.commit_revision != token:
        raise RuntimeError("Commit: marker did not advance or GET does not match its receipt")
    print(f"Commit verified: {fileset.commit_revision} -> {token}")
    return token


def run_demo(files: FilesClient, *, workspace: str, name: str, keep: bool = False) -> None:
    """Create a dedicated fileset, verify mutations, and remove it unless keep is set."""
    fileset = files.create_fileset(
        workspace=workspace, body=CreateFilesetRequest(name=name, description="SDK revision demo")
    ).data()
    try:
        fileset.commit_revision = require_commit_revision(fileset.commit_revision)
        print(f"Created {workspace}/{name}: commit={fileset.commit_revision}")
        fileset.commit_revision = verify_commit(files, fileset)
        for operation, content in (("Upload", b"first version\n"), ("Overwrite", b"second version\n")):
            files.upload_file(workspace=workspace, name=name, path="demo.txt", content=content).data()
            verify_unchanged(files, fileset, operation)
            downloaded = files.download_file(workspace=workspace, name=name, path="demo.txt").read()
            if downloaded != content:
                raise RuntimeError(f"{operation}: downloaded content differs from uploaded content")

        description = "SDK revision demo: metadata updated"
        files.update_fileset(workspace=workspace, name=name, body=UpdateFilesetRequest(description=description)).data()
        verify_unchanged(files, fileset, "Metadata update")
        if files.get_fileset(workspace=workspace, name=name).data().description != description:
            raise RuntimeError("Metadata update: description was not saved")

        files.delete_file(workspace=workspace, name=name, path="demo.txt").data()
        verify_unchanged(files, fileset, "Delete")
        fileset.commit_revision = verify_commit(files, fileset)
        verify_commit(files, fileset)
        if files.list_files(workspace=workspace, name=name).data().data:
            raise RuntimeError("Delete: fileset is not empty")
        print("PASS: SDK commit markers and file contents verified.")
    finally:
        if keep:
            print(f"Kept fileset {workspace}/{name}")
        else:
            files.delete_fileset_by_id(workspace=workspace, fileset_id=fileset.id)
            print(f"Cleaned up fileset {workspace}/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", help="Helix context name; defaults to the active context")
    parser.add_argument("--workspace", help="Workspace; defaults to the configured context's workspace")
    parser.add_argument(
        "--name", default=f"revision-demo-{uuid4().hex[:8]}", help="New fileset name; must not already exist"
    )
    parser.add_argument("--keep", action="store_true", help="Keep the demo fileset after running")
    args = parser.parse_args()
    with NemoClient.from_config(context=args.context) as client:
        workspace = args.workspace or client.workspace
        if not workspace:
            parser.error("Specify --workspace or select a context with a workspace")
        run_demo(FilesClient.from_client(client), workspace=workspace, name=args.name, keep=args.keep)


if __name__ == "__main__":
    main()
