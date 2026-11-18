# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the demo entry point against real Files/Entity services via the SDK."""

import runpy
import sys
from pathlib import Path

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import FilesetOutput

DEMO = Path(__file__).parents[2] / "script" / "demo_fileset_revision.py"


@pytest.mark.parametrize("keep", [False, True])
def test_demo_verifies_sdk_receipts_and_cleanup(
    client: NemoClient, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], keep: bool
):
    def configured_client(context: str | None = None) -> NemoClient:
        return client.with_options()

    monkeypatch.setattr(NemoClient, "from_config", configured_client)
    argv = [str(DEMO), "--workspace", "default", "--name", "revision-demo-test"]
    if keep:
        argv.append("--keep")
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(DEMO), run_name="__main__")
    output = capsys.readouterr().out
    for operation in ("Upload", "Overwrite", "Metadata update", "Delete"):
        assert f"{operation} verified:" in output
    assert "PASS: SDK revision receipts and file contents verified." in output
    files = FilesClient.from_client(client)
    remaining = list(files.list_filesets(workspace="default").items())
    assert len(remaining) == int(keep)
    if keep:
        assert remaining[0].description == "SDK revision demo: metadata updated"
        assert not files.list_files(workspace="default", name=remaining[0].name).data().data
        files.delete_fileset_by_id(workspace="default", fileset_id=remaining[0].id)


def test_demo_does_not_remove_an_existing_name(
    client: NemoClient, fileset: FilesetOutput, monkeypatch: pytest.MonkeyPatch
):
    def configured_client(context: str | None = None) -> NemoClient:
        return client.with_options()

    monkeypatch.setattr(NemoClient, "from_config", configured_client)
    monkeypatch.setattr(sys, "argv", [str(DEMO), "--workspace", "default", "--name", fileset.name])
    with pytest.raises(ConflictError):
        runpy.run_path(str(DEMO), run_name="__main__")
    observed = FilesClient.from_client(client).get_fileset(workspace="default", name=fileset.name).data()
    assert observed.id == fileset.id
    assert observed.revision == fileset.revision
