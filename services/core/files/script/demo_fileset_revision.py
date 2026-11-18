#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify fileset revision receipts through the Helix SDK and the configured context.

Run from the repository root against a server containing the revision change:
    uv run services/core/files/script/demo_fileset_revision.py --workspace default
"""

import argparse
from uuid import uuid4

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput, UpdateFilesetRequest


def require_revision(value: str | None) -> str:
    if not value:
        raise RuntimeError("No revision returned. Use an SDK and server containing the fileset revision change.")
    return value


def verify_change(
    files: FilesClient, fileset: FilesetOutput, previous: str, receipt: str | None, operation: str
) -> str:
    assigned = require_revision(receipt)
    if assigned == previous:
        raise RuntimeError(f"{operation}: revision did not change")
    observed = files.get_fileset_by_id(workspace=fileset.workspace, fileset_id=fileset.id).data()
    if observed.revision != assigned:
        raise RuntimeError(f"{operation}: GET revision {observed.revision!r} differs from receipt {assigned!r}")
    print(f"{operation} verified: {previous} -> {assigned}")
    return assigned


def run_demo(files: FilesClient, *, workspace: str, name: str, keep: bool = False) -> None:
    """Create a dedicated fileset, verify mutations, and remove it unless keep is set."""
    fileset = files.create_fileset(
        workspace=workspace, body=CreateFilesetRequest(name=name, description="SDK revision demo")
    ).data()
    try:
        previous = require_revision(fileset.revision)
        print(f"Created {workspace}/{name}: {previous}")
        for operation, content in (("Upload", b"first version\n"), ("Overwrite", b"second version\n")):
            uploaded = files.upload_file(workspace=workspace, name=name, path="demo.txt", content=content).data()
            previous = verify_change(files, fileset, previous, uploaded.fileset_revision, operation)
            downloaded = files.download_file(workspace=workspace, name=name, path="demo.txt").read()
            if downloaded != content:
                raise RuntimeError(f"{operation}: downloaded content differs from uploaded content")

        description = "SDK revision demo: metadata updated"
        updated = files.update_fileset(
            workspace=workspace, name=name, body=UpdateFilesetRequest(description=description)
        ).data()
        previous = verify_change(files, fileset, previous, updated.revision, "Metadata update")
        if files.get_fileset(workspace=workspace, name=name).data().description != description:
            raise RuntimeError("Metadata update: description was not saved")

        deleted = files.delete_file(workspace=workspace, name=name, path="demo.txt").data()
        verify_change(files, fileset, previous, deleted.fileset_revision, "Delete")
        if files.list_files(workspace=workspace, name=name).data().data:
            raise RuntimeError("Delete: fileset is not empty")
        print("PASS: SDK revision receipts and file contents verified.")
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
