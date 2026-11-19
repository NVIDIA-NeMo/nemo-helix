# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real S3/HTTP/disk-cache regression.

Run with RUN_EXTERNAL_STORAGE_TESTS=1; see the Files README. The Files API runs
in-process. S3 uses S3_TEST_ENDPOINT when supplied, otherwise disposable MinIO
(requires Docker, no cloud credentials).
"""

import os
import uuid
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import boto3
import httpx
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.common.api.common import SecretRef
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.common.files.storage_config import S3StorageConfig
from nhx.core.files.app.backends.local import LocalStorageConfig
from nhx.core.files.config import FilesConfig, files_config
from nhx.core.files.service import FilesService
from nhx.core.secrets.service import SecretsService
from nhx.testing import ClientContext, create_test_client
from pydantic import SecretStr
from tenacity import Retrying, stop_after_delay, wait_fixed

import docker

pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("files_s3_cache")]
MINIO_IMAGE = (
    "minio/minio:RELEASE.2025-04-22T22-12-26Z@sha256:a1ea29fa28355559ef137d71fc570e508a214ec84ff8083e39bc5428980b015e"
)
# Match the external-storage suite defaults; these are disposable test credentials.
ACCESS_KEY = os.environ.get("S3_TEST_ACCESS_KEY", "rustfsadmin")
SECRET_KEY = os.environ.get("S3_TEST_SECRET_KEY", "rustfsadmin")
REGION = os.environ.get("S3_TEST_REGION", "us-east-1")


@pytest.fixture(autouse=True)
def isolated_files_config() -> Iterator[None]:
    """Each in-process app owns its config, rather than reusing another app's cached getter."""
    files_config.cache_clear()
    yield
    files_config.cache_clear()


@pytest.fixture(scope="module")
def s3_endpoint() -> Iterator[str]:
    """Use external S3 or start MinIO; missing prerequisites fail after explicit opt-in."""
    if endpoint := os.environ.get("S3_TEST_ENDPOINT"):
        yield endpoint
        return
    with closing(docker.from_env()) as engine:
        engine.ping()
        container = engine.containers.run(
            MINIO_IMAGE,
            ["server", "/data", "--address", ":9000"],
            environment={
                "MINIO_ROOT_USER": ACCESS_KEY,
                "MINIO_ROOT_PASSWORD": SECRET_KEY,
                "MINIO_REGION_NAME": REGION,
            },
            ports={"9000/tcp": ("127.0.0.1", None)},
            detach=True,
        )
        try:
            container.reload()
            port = container.attrs["NetworkSettings"]["Ports"]["9000/tcp"][0]["HostPort"]
            endpoint = f"http://127.0.0.1:{port}"
            for attempt in Retrying(stop=stop_after_delay(30), wait=wait_fixed(0.2), reraise=True):
                with attempt:
                    httpx.get(f"{endpoint}/minio/health/ready", timeout=2).raise_for_status()
            yield endpoint
        finally:
            container.remove(force=True, v=True)


@pytest.fixture
def s3_bucket(s3_endpoint: str) -> Iterator[tuple[Any, str]]:
    """Own a unique bucket, including its historical versions and delete markers."""
    with closing(
        boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            region_name=REGION,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
    ) as s3:
        bucket = f"cache-test-{uuid.uuid4().hex[:12]}"
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": REGION})
        try:
            yield s3, bucket
        finally:
            for page in s3.get_paginator("list_object_versions").paginate(Bucket=bucket):
                objects = [
                    {"Key": obj["Key"], "VersionId": obj["VersionId"]}
                    for kind in ("Versions", "DeleteMarkers")
                    for obj in page.get(kind, [])
                ]
                if objects:
                    s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            s3.delete_bucket(Bucket=bucket)


@pytest.mark.parametrize("versioned,retained", [(False, 1), (True, 2)])
def test_s3_cache_freshness_and_retention(
    s3_endpoint: str,
    s3_bucket: tuple[Any, str],
    tmp_path: Path,
    versioned: bool,
    retained: int,
):
    """API/bucket mutations never serve obsolete bytes; pruning preserves active readers."""
    s3, bucket = s3_bucket
    cache_root = tmp_path / "cache"
    auth = AuthClient(principal=Principal(id="test@example.com"), config=AuthConfig(enabled=False))
    config = FilesConfig(
        default_storage_config=LocalStorageConfig(path=str(cache_root)),
        s3_cache_max_versions_per_object=retained,
    )
    with (
        create_test_client(
            FilesService,
            SecretsService,
            client_type=ClientContext,
            tmp_dir=tmp_path,
            service_configs={FilesService: config},
            dependency_overrides={get_auth_client: lambda: auth},
        ) as ctx,
    ):
        if versioned:
            s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
        secrets_client = SecretsClient.from_client(ctx.client)
        for name, value in (("s3-access", ACCESS_KEY), ("s3-secret", SECRET_KEY)):
            secrets_client.create_secret(
                workspace="default", body=HelixSecretCreateRequest(name=name, value=SecretStr(value))
            )
        files_client = FilesClient.from_client(ctx.client)
        files_client.create_fileset(
            workspace="default",
            body=CreateFilesetRequest(
                name="dataset",
                storage=S3StorageConfig(
                    bucket=bucket,
                    prefix="dataset",
                    endpoint_url=s3_endpoint,
                    region=REGION,
                    access_key_id_secret=SecretRef("s3-access"),
                    secret_access_key_secret=SecretRef("s3-secret"),
                    use_sdk_auth=False,
                ),
            ),
        )
        target = {"workspace": "default", "name": "dataset", "path": "data.txt"}
        url = "/apis/files/v2/workspaces/default/filesets/dataset/-/data.txt"
        original, replacement = b"0123456789A", b"0123456789B"  # Equal size, different ETag.
        legacy = cache_root / "cache/s3" / bucket / "dataset/data.txt"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"stale legacy cache")

        files_client.upload_file(**target, content=original)
        assert files_client.download_file(**target).read() == original
        assert files_client.download_file(**target).read() == original
        cached = [path for path in (cache_root / "cache/s3-v2").rglob("*") if path.is_file()]
        assert len(cached) == 1
        first = cached[0]
        assert first.read_bytes() == original
        identity = s3.head_object(Bucket=bucket, Key="dataset/data.txt")
        initial_etag = identity["ETag"]
        if versioned:
            assert identity.get("VersionId") not in (None, "null")

        with first.open("rb", buffering=0) as reader:
            assert reader.read(1) == original[:1]
            files_client.upload_file(**target, content=replacement)
            assert files_client.download_file(**target).read() == replacement
            ranged = files_client.with_headers({"Range": "bytes=2-5"}).download_file(**target)
            assert ranged.read() == replacement[2:6]
            assert ranged.http_response.status_code == 206
            head = ctx.test_client.head(url)
            assert head.status_code == 200 and head.headers["content-length"] == str(len(replacement))

            # Verify the real S3 server enforces the same signed If-Match contract as the backend.
            signed = s3.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": "dataset/data.txt", "IfMatch": initial_etag}
            )
            assert httpx.get(signed, headers={"If-Match": initial_etag}).status_code == 412

            files_client.delete_file(**target)
            with pytest.raises(ClientError) as error:
                s3.head_object(Bucket=bucket, Key="dataset/data.txt")
            assert error.value.response["ResponseMetadata"]["HTTPStatusCode"] == 404
            with pytest.raises(NotFoundError):
                files_client.download_file(**target).read()
            assert ctx.test_client.head(url).status_code == 404

            # API mutations above use the default TTL. Direct writes below use strict
            # revalidation, avoiding a sleep or mocked clock; no platform invalidation runs.
            config.s3_metadata_cache_ttl_seconds = 0
            for content in (b"bucket-side recreation", b"next version", b"latest version"):
                s3.put_object(Bucket=bucket, Key="dataset/data.txt", Body=content)
                response = files_client.download_file(**target)
                assert response.read() == content
                assert response.http_response.headers["content-length"] == str(len(content))
                versions = [path for path in first.parent.iterdir() if path.is_file()]
                assert len(versions) <= retained
                assert content in [path.read_bytes() for path in versions]
            assert len(versions) == retained
            assert not first.exists()
            assert reader.read() == original[1:]  # POSIX reader survives eviction of its pathname.

        s3.delete_object(Bucket=bucket, Key="dataset/data.txt")
        with pytest.raises(NotFoundError):
            files_client.download_file(**target).read()
        assert legacy.read_bytes() == b"stale legacy cache"  # Ignored, not migrated or served.
