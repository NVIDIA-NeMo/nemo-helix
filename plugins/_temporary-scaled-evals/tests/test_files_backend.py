# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Files-backed equivalence tests for the artifact-storage migration.

Drives ``scaled_evals.api._files_backend`` against a REAL, in-memory Files service
(``create_test_client(FilesService, SecretsService)`` on a ``local`` tmp_path backend —
no cluster, no external S3/GCS). Asserts the same transport-level behaviors the current
boto3 path provides, so once ``scaled_evals.api.artifacts`` delegates to this backend the
characterization suite (test_artifact_storage_contract.py) proves end-to-end equivalence.

The fileset key mapping under test (see ``_files_backend.split_key``):
    evaluations/{id}/...      -> se-eval-{id}
    {task_id}/rev/{n}/...     -> se-task-{task_id}
    switchyard-contexts/...   -> se-build-context
    (anything else)           -> se-shared
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib.util import find_spec
from pathlib import Path

import pytest

# Skip the whole module only when the optional plugin/platform packages are genuinely
# absent (default sync). Gate on find_spec so a BROKEN import inside an installed package
# still fails collection loudly instead of being masked as "not installed".
if any(find_spec(pkg) is None for pkg in ("filesets", "nemo_helix_plugin", "nhx", "scaled_evals")):
    pytest.skip("scaled-evals plugin or platform not installed", allow_module_level=True)

from filesets.resources import FilesResource
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nhx.core.files.service import FilesService
from nhx.core.secrets.service import SecretsService
from nhx.testing import create_test_client
from scaled_evals.api import _files_backend as fb


@pytest.fixture
def files_backend() -> Iterator[None]:
    """Point the backend at an in-memory Files service for the duration of a test."""
    with create_test_client(FilesService, SecretsService) as sdk:
        resource = FilesResource(sdk, files_client=client_from_platform(sdk, FilesClient))
        sdk.__dict__["files"] = resource
        fb.set_files_resource(resource)
        # The in-memory platform ships a "default" workspace; align the backend with it.
        fb.set_workspace_override(sdk.workspace or "default")
        try:
            yield
        finally:
            fb.set_files_resource(None)
            fb.set_workspace_override(None)


# ---------------------------------------------------------------------------
# key -> (fileset, path) mapping (pure, no service needed)
# ---------------------------------------------------------------------------


def test_split_key_maps_each_namespace() -> None:
    assert fb.split_key("evaluations/ev1/artifacts/score.json") == fb.FilesetRef("se-eval-ev1", "artifacts/score.json")
    assert fb.split_key("evaluations/ev1/live/1/runner.log") == fb.FilesetRef("se-eval-ev1", "live/1/runner.log")
    assert fb.split_key("task_abc/rev/2/tarball.tar.gz") == fb.FilesetRef("se-task-task_abc", "rev/2/tarball.tar.gz")
    assert fb.split_key("switchyard-contexts/refs/x.tar.gz") == fb.FilesetRef("se-build-context", "refs/x.tar.gz")
    assert fb.split_key("benchmark-archives/a.tar.gz") == fb.FilesetRef("se-shared", "benchmark-archives/a.tar.gz")


def test_fileset_id_is_normalized_to_a_valid_entity_name() -> None:
    # Files entity names must be lowercase, no doubled/trailing dashes. Uppercase and other
    # characters are normalized deterministically so any id maps to a valid fileset.
    assert fb.split_key("evaluations/Ev-Mixed/artifacts/x").fileset == "se-eval-ev-mixed"
    # Spaces in the id component collapse to a single dash (the id here is "a  b").
    assert fb.split_key("evaluations/a  b/artifacts/x").fileset == "se-eval-a-b"


def test_fileset_id_is_clamped_to_the_entity_length_limit() -> None:
    # Files entity names cap at 63 chars. A pathologically long id must be clamped so
    # ``se-eval-<id>`` still fits, with a deterministic digest suffix so two long ids
    # sharing a prefix can't collide onto the same fileset.
    long_id = "a" * 120
    name = fb.split_key(f"evaluations/{long_id}/artifacts/x").fileset
    assert name.startswith("se-eval-")
    assert len(name) <= 63
    # Deterministic: same id -> same fileset.
    assert name == fb.split_key(f"evaluations/{long_id}/artifacts/y").fileset
    # Two long ids that share the clamped prefix still map to distinct filesets.
    other = fb.split_key(f"evaluations/{'a' * 119}b/artifacts/x").fileset
    assert other != name


def test_keys_with_hash_are_rejected() -> None:
    # The Files path parser treats '#' as the fileset#path delimiter even with explicit
    # fileset kwargs, so a '#' in a key would silently mis-route. Reject it up front.
    with pytest.raises(ValueError, match="#"):
        fb.split_key("evaluations/ev1/artifacts/a#b.txt")
    with pytest.raises(ValueError, match="#"):
        fb.fileset_prefix("evaluations/ev1/artifacts/a#")


def test_keys_with_fileset_scheme_are_rejected() -> None:
    # A leading 'fileset://' is parsed as a full workspace/fileset reference by the Files
    # SDK, which would let a key escape the configured workspace. Reject it up front.
    with pytest.raises(ValueError, match="fileset://"):
        fb.split_key("fileset://other-workspace/name/file")
    with pytest.raises(ValueError, match="fileset://"):
        fb.fileset_prefix("fileset://other-workspace/name/")


def test_fileset_prefix_maps_listing_scope() -> None:
    assert fb.fileset_prefix("evaluations/ev1/artifacts/") == ("se-eval-ev1", "artifacts/")
    assert fb.fileset_prefix("evaluations/ev1/") == ("se-eval-ev1", "")


# ---------------------------------------------------------------------------
# Transport primitives against real in-memory Files
# ---------------------------------------------------------------------------


def test_put_get_exists_size_delete_roundtrip(files_backend: None) -> None:
    key = "evaluations/eva1/artifacts/score.json"
    assert fb.object_exists(key) is False
    assert fb.object_size(key) is None

    fb.put_bytes(key, b'{"metric": 1}', content_type="application/json")
    assert fb.object_exists(key) is True
    assert fb.object_size(key) == len(b'{"metric": 1}')
    assert fb.get_bytes(key) == b'{"metric": 1}'

    fb.delete_object(key)
    assert fb.object_exists(key) is False
    # Deleting a now-missing key is a no-op (S3 DeleteObject semantics).
    fb.delete_object(key)


def test_get_missing_raises(files_backend: None) -> None:
    with pytest.raises(fb.MissingObjectError):
        fb.get_bytes("evaluations/evMissing/artifacts/none.json")


def test_upload_and_download_file(files_backend: None, tmp_path: Path) -> None:
    src = tmp_path / "tarball.tar.gz"
    src.write_bytes(b"x" * 2048)
    key = "task_pack1/rev/1/tarball.tar.gz"

    size = fb.upload_file(src, key, content_type="application/gzip")
    assert size == 2048
    assert fb.object_size(key) == 2048

    dest = tmp_path / "roundtrip.tar.gz"
    fb.download_object(key, str(dest))
    assert dest.read_bytes() == b"x" * 2048


def test_list_objects_reconstructs_full_keys(files_backend: None) -> None:
    prefix = "evaluations/evb1/artifacts/"
    fb.put_bytes(f"{prefix}a.txt", b"aa")
    fb.put_bytes(f"{prefix}sub/b.txt", b"bbb")
    # A live-log file in the same fileset but outside the artifacts/ prefix.
    fb.put_bytes("evaluations/evb1/live/1/runner.log", b"log")

    listed = {item["key"]: item["size_bytes"] for item in fb.list_objects(prefix)}
    assert listed == {f"{prefix}a.txt": 2, f"{prefix}sub/b.txt": 3}


def test_stream_object_yields_full_content(files_backend: None) -> None:
    key = "evaluations/evc1/results.tar.gz"
    payload = b"y" * (1024 * 1024 + 7)  # spans multiple 1 MiB chunks
    fb.put_bytes(key, payload)
    streamed = b"".join(fb.stream_object(key))
    assert streamed == payload


def test_stream_missing_raises(files_backend: None) -> None:
    with pytest.raises(fb.MissingObjectError):
        list(fb.stream_object("evaluations/evNope/results.tar.gz"))


def test_separate_evaluations_use_separate_filesets(files_backend: None) -> None:
    fb.put_bytes("evaluations/ev1/artifacts/x.txt", b"1")
    fb.put_bytes("evaluations/ev2/artifacts/x.txt", b"2")
    assert fb.get_bytes("evaluations/ev1/artifacts/x.txt") == b"1"
    assert fb.get_bytes("evaluations/ev2/artifacts/x.txt") == b"2"
    # Listing one eval's prefix never leaks the other's fileset.
    ev1 = {item["key"] for item in fb.list_objects("evaluations/ev1/artifacts/")}
    assert ev1 == {"evaluations/ev1/artifacts/x.txt"}


def test_readiness_probe_ok_on_live_service(files_backend: None) -> None:
    # Must not raise against a healthy Files service, even with no shared fileset yet.
    fb.readiness_probe()
