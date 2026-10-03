# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download customizer E2E assets from S3 and stage them into platform filesets.

CI and local GPU tests must not pull datasets or model weights from Hugging Face.
Assets are pre-published to ``s3://<bucket>/`` (see ``publish_assets_to_s3.sh``);
this module syncs the needed prefixes locally, then uploads into dataset/model
filesets through the typed platform client.
"""

import logging
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from filesets import transfer
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelEntityRequest
from nhx.testing.e2e.customizer import wait_for_model_spec

from e2e.customizer.assets_manifest import load as load_manifest
from e2e.customizer.assets_manifest import s3_bucket
from e2e.customizer.customization_helpers import assert_fileset_has_files, unique_name

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(__file__).parent / ".asset-cache"
_UPLOAD_SKIP_NAMES = {".cache", ".gitattributes"}
# Fail fast on a hung transfer or an interactive credential prompt rather than
# blocking the whole GPU E2E run. Overridable for very large/slow syncs.
_SYNC_TIMEOUT_S = int(os.environ.get("E2E_ASSETS_SYNC_TIMEOUT", "1800"))


def cache_dir() -> Path:
    return Path(os.environ.get("E2E_ASSETS_CACHE_DIR", _DEFAULT_CACHE_DIR))


def _aws_sync(s3_uri: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["aws"]
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if endpoint:
        cmd.extend(["--endpoint-url", endpoint])
    cmd.extend(["s3", "sync", s3_uri, str(local_dir), "--only-show-errors"])
    logger.info("Syncing %s -> %s", s3_uri, local_dir)
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=_SYNC_TIMEOUT_S)
    except FileNotFoundError:
        pytest.fail("aws CLI is required to stage E2E assets from S3")
    except subprocess.TimeoutExpired:
        pytest.fail(f"S3 sync for {s3_uri} timed out after {_SYNC_TIMEOUT_S}s")
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"S3 sync failed for {s3_uri}: {exc}")


def sync_dataset_format(format_name: str, *, manifest: dict | None = None) -> Path:
    """Sync ``s3://<bucket>/datasets/<format_name>/`` into the local cache."""
    manifest = manifest or load_manifest()
    bucket = s3_bucket(manifest)
    local_dir = cache_dir() / "datasets" / format_name
    _aws_sync(f"s3://{bucket}/datasets/{format_name}/", local_dir)
    return local_dir


def sync_model(s3_folder: str, *, manifest: dict | None = None) -> Path:
    """Sync ``s3://<bucket>/models/<s3_folder>/`` into the local cache."""
    manifest = manifest or load_manifest()
    bucket = s3_bucket(manifest)
    local_dir = cache_dir() / "models" / s3_folder
    _aws_sync(f"s3://{bucket}/models/{s3_folder}/", local_dir)
    return local_dir


def stage_dataset_fileset(
    client: NemoClient,
    workspace: str,
    *,
    local_dir: Path,
    name_prefix: str,
    files: Mapping[str, str],
    purpose: FilesetPurpose = FilesetPurpose.DATASET,
) -> str:
    """Upload ``files`` (remote_name -> local filename) from ``local_dir`` into a new fileset."""
    files_client = FilesClient.from_client(client)
    fileset_name = unique_name(name_prefix)
    files_client.create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=fileset_name,
            purpose=purpose,
            description=f"E2E dataset staged from {local_dir.name}",
        ),
    )
    for remote_name, local_name in files.items():
        source = local_dir / local_name
        if not source.is_file():
            pytest.fail(f"Staged dataset file missing after S3 sync: {source}")
        logger.info("Uploading %s -> %s/%s#%s", source, workspace, fileset_name, remote_name)
        transfer.upload(
            files_client,
            local_path=str(source),
            remote_path=remote_name,
            fileset=fileset_name,
            workspace=workspace,
        )
    assert_fileset_has_files(client, workspace, fileset_name)
    return fileset_name


def stage_model_entity(
    client: NemoClient,
    workspace: str,
    *,
    entity_name: str,
    snapshot_dir: Path,
    hf_repo: str | None = None,
    wait_timeout: int = 600,
) -> str:
    """Upload a model snapshot directory into a weights fileset and register a model entity."""
    files_client = FilesClient.from_client(client)
    weights_fileset = f"{entity_name}-weights"
    files_client.create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=weights_fileset,
            purpose=FilesetPurpose.MODEL,
            description=f"E2E model weights staged from {snapshot_dir.name}",
        ),
        exist_ok=True,
    )

    uploaded = 0
    for path in sorted(snapshot_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(snapshot_dir)
        if rel.parts and rel.parts[0] in _UPLOAD_SKIP_NAMES:
            continue
        if rel.name == ".gitattributes":
            continue
        transfer.upload(
            files_client,
            local_path=str(path),
            remote_path=str(rel),
            fileset=weights_fileset,
            workspace=workspace,
        )
        uploaded += 1

    if uploaded == 0:
        pytest.fail(f"No model files found to upload under {snapshot_dir}")

    custom_fields = {"hf_model_id": hf_repo} if hf_repo else None
    ModelsClient.from_client(client).create_model(
        workspace=workspace,
        body=CreateModelEntityRequest(
            name=entity_name,
            fileset=f"{workspace}/{weights_fileset}",
            custom_fields=custom_fields,
        ),
        exist_ok=True,
    )
    wait_for_model_spec(client, workspace, entity_name, timeout=wait_timeout)
    logger.info(
        "Staged model entity %s/%s from %s (%d files)",
        workspace,
        entity_name,
        snapshot_dir,
        uploaded,
    )
    return entity_name
