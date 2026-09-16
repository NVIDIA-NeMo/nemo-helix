# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object-store access for task packs and evaluation artifacts.

The default backend is S3-compatible storage through boto3: RustFS in compose and
Kubernetes dev, AWS S3/MinIO, or GCS's XML API when HMAC credentials are
available. The GCS backend is intentionally separate because GKE Workload
Identity gives pods OAuth bearer tokens, not S3 HMAC credentials that boto3 can
use. It talks to GCS APIs with OAuth directly, preferring V4 signed URLs through
IAM Credentials signBlob when configured and falling back to resumable-upload
session URIs for task-pack upload.
"""

import base64
import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import boto3
import httpx
from botocore.client import Config
from botocore.exceptions import ClientError

from scaled_evals.api.redaction import redact_json_text, redact_secret_text
from scaled_evals.api.settings import settings

# s3v4: universal modern signing standard (boto3's default; pinned defensively).
# path-style: backend-specific — self-hosted stores (RustFS/MinIO) need it, AWS
# prefers virtual-hosted. Wrong combo → vague 403s.
_CONFIG = Config(signature_version="s3v4", s3={"addressing_style": "path"})
# Fail-fast variant for the readiness probe — short timeouts, single attempt.
_READYZ_CONFIG = _CONFIG.merge(Config(connect_timeout=3, read_timeout=3, retries={"max_attempts": 1}))
_GCS_OBJECT_KEY_RE = re.compile(r"[^\x00-\x1f\x7f\\?#]+")
ARTIFACT_MANIFEST_PATH = "scaled-evals-manifest.json"
ARCHIVE_FILE_NAME = "results.tar.gz"
HARBOR_VIEWER_ARCHIVE_FILE_NAME = "harbor-viewer.tar.gz"
_SECRET_FILE_NAMES = frozenset({".env", "target.env", "daytona.env"})
_TEXT_ARTIFACT_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".txt", ".yaml", ".yml", ".toml", ".md"})
_GCS_TOKEN_CACHE: tuple[str, float] | None = None


def _client(endpoint_url: str, config: Config = _CONFIG) -> Any:
    """Build an S3 client for `endpoint_url`. Cheap; built per call so we never
    share a client across the threadpool that runs sync routes."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=config,
    )


def _bucket() -> str:
    return settings.resolved_object_store_bucket()


# A concurrent replica won the create, or the bucket is pre-provisioned elsewhere.
_BUCKET_EXISTS_CODES = frozenset({"BucketAlreadyExists", "BucketAlreadyOwnedByYou"})


def _using_gcs() -> bool:
    return settings.object_store_backend == "gcs"


def _gcs_client_error(operation: str, response: httpx.Response) -> ClientError:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else {}
    message = response.text
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        message = error["message"]
    code = str(response.status_code)
    if response.status_code == 404:
        code = "NoSuchKey"
    elif response.status_code == 403:
        code = "AccessDenied"
    elif isinstance(error, dict) and isinstance(error.get("status"), str):
        code = error["status"]
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": response.status_code},
        },
        operation,
    )


def _raise_for_gcs(operation: str, response: httpx.Response) -> None:
    if response.status_code >= 400:
        response.read()
        raise _gcs_client_error(operation, response)


def _gcs_access_token() -> str:
    global _GCS_TOKEN_CACHE  # noqa: PLW0603
    if settings.gcs_access_token:
        return settings.gcs_access_token

    now = time.time()
    if _GCS_TOKEN_CACHE is not None:
        token, expires_at = _GCS_TOKEN_CACHE
        if expires_at - now > 60:
            return token

    with httpx.Client(timeout=settings.gcs_request_timeout_seconds) as client:
        response = client.get(
            settings.gcs_token_url,
            headers={"Metadata-Flavor": "Google"},
        )
    _raise_for_gcs("GetAccessToken", response)
    payload = response.json()
    token = payload["access_token"]
    expires_in = int(payload.get("expires_in") or 300)
    _GCS_TOKEN_CACHE = (token, now + expires_in)
    return token


def _gcs_headers(*, content_type: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {_gcs_access_token()}"}
    if content_type is not None:
        headers["Content-Type"] = content_type
    return headers


def _gcs_url(path: str) -> str:
    return f"{settings.gcs_api_base_url.rstrip('/')}{path}"


def _validated_gcs_object_key(object_key: str) -> str:
    """Return a GCS object key that is safe to place in a JSON API URL.

    Object keys are data, not URLs. Keep them relative and free of URL/control
    syntax before percent-encoding the complete value as one path component.
    """
    if not object_key or len(object_key.encode("utf-8")) > 1024:
        raise ValueError("GCS object key must contain between 1 and 1024 UTF-8 bytes")
    if _GCS_OBJECT_KEY_RE.fullmatch(object_key) is None:
        raise ValueError("GCS object key contains unsafe URL characters")
    parts = object_key.split("/")
    if any(part in {"", ".", ".."} for part in parts) or ":" in parts[0]:
        raise ValueError("GCS object key must be a relative path without traversal")
    return object_key


def _gcs_object_metadata_url(object_key: str) -> str:
    safe_key = _validated_gcs_object_key(object_key)
    return _gcs_url(f"/storage/v1/b/{quote(_bucket(), safe='')}/o/{quote(safe_key, safe='')}")


def _gcs_media_url(object_key: str) -> str:
    safe_key = _validated_gcs_object_key(object_key)
    return _gcs_url(f"/download/storage/v1/b/{quote(_bucket(), safe='')}/o/{quote(safe_key, safe='')}")


def _gcs_can_sign_urls() -> bool:
    return bool(settings.gcs_signing_service_account.strip())


def _gcs_use_signed_upload_url() -> bool:
    if settings.gcs_upload_mode == "signed_url":
        return True
    if settings.gcs_upload_mode == "resumable_session":
        return False
    return _gcs_can_sign_urls()


def _gcs_sign_blob(payload: bytes) -> bytes:
    service_account = settings.gcs_signing_service_account.strip()
    if not service_account:
        raise RuntimeError("GCS signed URLs require GCS_SIGNING_SERVICE_ACCOUNT")
    encoded_account = quote(service_account, safe="")
    with httpx.Client(timeout=settings.gcs_request_timeout_seconds) as client:
        response = client.post(
            f"{settings.gcs_iam_credentials_base_url.rstrip('/')}/v1/projects/-/"
            f"serviceAccounts/{encoded_account}:signBlob",
            headers=_gcs_headers(content_type="application/json"),
            json={"payload": base64.b64encode(payload).decode("ascii")},
        )
    _raise_for_gcs("SignBlob", response)
    signed_blob = response.json().get("signedBlob")
    if not isinstance(signed_blob, str):
        raise RuntimeError("IAM Credentials signBlob response did not include signedBlob")
    return base64.b64decode(signed_blob)


def _gcs_xml_base_url() -> tuple[str, str]:
    parts = urlsplit(settings.gcs_api_base_url.rstrip("/"))
    if not parts.scheme or not parts.netloc:
        raise RuntimeError("GCS_API_BASE_URL must include scheme and host")
    return f"{parts.scheme}://{parts.netloc}", parts.netloc


def _canonical_query(params: dict[str, str]) -> str:
    return "&".join(f"{quote(key, safe='')}={quote(value, safe='~')}" for key, value in sorted(params.items()))


def _gcs_signed_url(
    method: str,
    object_key: str,
    *,
    expires_in: int,
    content_type: str | None = None,
) -> str:
    if expires_in > 604800:
        raise ValueError("GCS signed URLs cannot expire later than 604800 seconds")
    base_url, host = _gcs_xml_base_url()
    timestamp = datetime.now(tz=UTC)
    datestamp = timestamp.strftime("%Y%m%d")
    active_datetime = timestamp.strftime("%Y%m%dT%H%M%SZ")
    credential_scope = f"{datestamp}/{settings.gcs_signed_url_region}/storage/goog4_request"
    service_account = settings.gcs_signing_service_account.strip()
    credential = f"{service_account}/{credential_scope}"
    canonical_uri = f"/{quote(_bucket(), safe='')}/{quote(object_key, safe='/~')}"
    headers = {"host": host}
    if content_type is not None:
        headers["content-type"] = content_type
    canonical_headers = "".join(f"{name}:{value.strip()}\n" for name, value in sorted(headers.items()))
    signed_headers = ";".join(sorted(headers))
    query_params = {
        "X-Goog-Algorithm": "GOOG4-RSA-SHA256",
        "X-Goog-Credential": credential,
        "X-Goog-Date": active_datetime,
        "X-Goog-Expires": str(expires_in),
        "X-Goog-SignedHeaders": signed_headers,
    }
    canonical_query = _canonical_query(query_params)
    canonical_request = "\n".join(
        [
            method,
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_headers,
            "UNSIGNED-PAYLOAD",
        ]
    )
    string_to_sign = "\n".join(
        [
            "GOOG4-RSA-SHA256",
            active_datetime,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signature = _gcs_sign_blob(string_to_sign.encode("utf-8")).hex()
    return f"{base_url}{canonical_uri}?{canonical_query}&X-Goog-Signature={signature}"


def _gcs_create_resumable_upload(
    object_key: str,
    *,
    content_type: str,
    content_length: int | None = None,
) -> str:
    # A GCS resumable session URI acts like our S3 presigned PUT URL: the API
    # authenticates once with Workload Identity, then the client can upload
    # directly to that opaque HTTPS URI without receiving cloud credentials.
    headers = _gcs_headers(content_type="application/json")
    headers["X-Upload-Content-Type"] = content_type
    if content_length is not None:
        headers["X-Upload-Content-Length"] = str(content_length)
    with httpx.Client(timeout=settings.gcs_request_timeout_seconds) as client:
        response = client.post(
            _gcs_url(f"/upload/storage/v1/b/{quote(_bucket(), safe='')}/o"),
            params={"uploadType": "resumable", "name": object_key},
            headers=headers,
            content=b"{}",
        )
    _raise_for_gcs("CreateResumableUpload", response)
    location = response.headers.get("location")
    if not location:
        raise RuntimeError("GCS resumable upload did not return a Location header")
    return location


def _content_range(size_bytes: int) -> str:
    if size_bytes == 0:
        return "bytes */0"
    return f"bytes 0-{size_bytes - 1}/{size_bytes}"


def _iter_fileobj(fileobj: Any, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    while True:
        chunk = fileobj.read(chunk_size)
        if not chunk:
            break
        yield chunk


def _gcs_upload_fileobj(
    fileobj: Any,
    object_key: str,
    *,
    content_type: str,
    size_bytes: int,
) -> None:
    session_url = _gcs_create_resumable_upload(
        object_key,
        content_type=content_type,
        content_length=size_bytes,
    )
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(size_bytes),
        "Content-Range": _content_range(size_bytes),
    }
    with httpx.Client(timeout=None) as client:
        response = client.put(session_url, headers=headers, content=_iter_fileobj(fileobj))
    _raise_for_gcs("UploadObject", response)


def _gcs_upload_bytes(object_key: str, body: bytes, *, content_type: str) -> None:
    with httpx.Client(timeout=settings.gcs_request_timeout_seconds) as client:
        response = client.post(
            _gcs_url(f"/upload/storage/v1/b/{quote(_bucket(), safe='')}/o"),
            params={"uploadType": "media", "name": object_key},
            headers=_gcs_headers(content_type=content_type),
            content=body,
        )
    _raise_for_gcs("PutObject", response)


def _gcs_upload_file(path: Path, object_key: str) -> None:
    size_bytes = path.stat().st_size
    with path.open("rb") as source:
        _gcs_upload_fileobj(
            source,
            object_key,
            content_type="application/octet-stream",
            size_bytes=size_bytes,
        )


def _gcs_read_object_bytes(object_key: str) -> bytes:
    with httpx.Client(timeout=settings.gcs_request_timeout_seconds) as client:
        response = client.get(
            _gcs_media_url(object_key),
            params={"alt": "media"},
            headers=_gcs_headers(),
        )
    _raise_for_gcs("GetObject", response)
    return response.content


def _read_object_bytes(object_key: str) -> bytes:
    if _using_gcs():
        return _gcs_read_object_bytes(object_key)
    response = _client(settings.s3_endpoint).get_object(Bucket=_bucket(), Key=object_key)
    with closing(response["Body"]) as body:
        return body.read()


def _gcs_stream_object(object_key: str) -> Iterator[bytes]:
    if _GCS_OBJECT_KEY_RE.fullmatch(object_key):
        with (
            httpx.Client(timeout=None) as client,
            client.stream(
                "GET",
                _gcs_media_url(object_key),
                params={"alt": "media"},
                headers=_gcs_headers(),
            ) as response,
        ):
            _raise_for_gcs("GetObject", response)
            yield from response.iter_bytes()
        return
    raise ValueError("GCS object key contains unsafe URL characters")


def can_presign_get() -> bool:
    return not _using_gcs() or _gcs_can_sign_urls()


def presign_put(object_key: str, expires_in: int = 900) -> dict[str, str]:
    """Return the method, URL, and headers for a presigned tarball PUT.

    The client PUTs the gzip tarball bytes to `url` with `headers`. Signed
    against the public endpoint so the URL's host matches what the client hits.
    """
    if _using_gcs():
        if _gcs_use_signed_upload_url():
            return {
                "method": "PUT",
                "url": _gcs_signed_url(
                    "PUT",
                    object_key,
                    expires_in=expires_in,
                    content_type="application/gzip",
                ),
                "headers": {"Content-Type": "application/gzip"},
                "mode": "put",
            }
        return {
            "method": "PUT",
            "url": _gcs_create_resumable_upload(object_key, content_type="application/gzip"),
            "headers": {"Content-Type": "application/gzip"},
            "mode": "gcs_resumable",
        }
    url = _client(settings.resolved_s3_public_endpoint()).generate_presigned_url(
        "put_object",
        Params={"Bucket": _bucket(), "Key": object_key},
        ExpiresIn=expires_in,
    )
    return {
        "method": "PUT",
        "url": url,
        "headers": {"Content-Type": "application/gzip"},
        "mode": "put",
    }


def object_size(object_key: str) -> int | None:
    """Return object size in bytes from the internal endpoint.

    ``None`` means the object store did not provide a usable Content-Length,
    which is unsafe for task-pack quota enforcement.
    """
    if _using_gcs():
        with httpx.Client(timeout=settings.gcs_request_timeout_seconds) as client:
            response = client.get(
                _gcs_object_metadata_url(object_key),
                params={"fields": "size"},
                headers=_gcs_headers(),
            )
        _raise_for_gcs("HeadObject", response)
        content_length = response.json().get("size")
        if content_length is None:
            return None
        return int(content_length)
    response = _client(settings.s3_endpoint).head_object(
        Bucket=_bucket(),
        Key=object_key,
    )
    content_length = response.get("ContentLength")
    if content_length is None:
        return None
    return int(content_length)


def is_missing_object_error(exc: ClientError) -> bool:
    """Return True for backend-specific object-not-found errors."""
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return code in {"404", "NoSuchKey", "NotFound"}


def object_exists(object_key: str) -> bool:
    """Return whether an object can be read from the configured object store."""
    try:
        object_size(object_key)
    except ClientError as exc:
        if is_missing_object_error(exc):
            return False
        raise
    return True


def delete_object(object_key: str) -> None:
    """Delete one object through the internal endpoint."""
    if _using_gcs():
        with httpx.Client(timeout=settings.gcs_request_timeout_seconds) as client:
            response = client.delete(_gcs_object_metadata_url(object_key), headers=_gcs_headers())
        _raise_for_gcs("DeleteObject", response)
        return
    _client(settings.s3_endpoint).delete_object(Bucket=_bucket(), Key=object_key)


def presign_get(object_key: str, expires_in: int = 900) -> str:
    """Return a presigned download URL for one object.

    Signed against the public endpoint so the URL's host matches what API
    clients can reach, while object listing/upload use the internal endpoint.
    """
    if _using_gcs():
        return _gcs_signed_url("GET", object_key, expires_in=expires_in)
    return _client(settings.resolved_s3_public_endpoint()).generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": object_key},
        ExpiresIn=expires_in,
    )


def stream_object(object_key: str) -> Any:
    """Return the streaming body of an S3 object for server-side proxying.

    Uses the internal endpoint — callers must be running in the same network as
    the object store. The returned body is a file-like iterator suitable for
    FastAPI's StreamingResponse.
    """
    if _using_gcs():
        return _gcs_stream_object(object_key)
    return _client(settings.s3_endpoint).get_object(Bucket=_bucket(), Key=object_key)["Body"]


def read_json_object(object_key: str) -> dict[str, Any]:
    """Read one internal JSON object for worker-side evidence composition."""
    if _using_gcs():
        value = json.loads(_gcs_read_object_bytes(object_key))
        if not isinstance(value, dict):
            raise ValueError(f"JSON object expected at {object_key}")
        return value
    response = _client(settings.s3_endpoint).get_object(
        Bucket=_bucket(),
        Key=object_key,
    )
    with closing(response["Body"]) as body:
        value = json.loads(body.read())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected at {object_key}")
    return value


def put_json_object(object_key: str, value: dict[str, Any]) -> None:
    """Write one JSON object through the internal S3 endpoint."""
    body = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    if _using_gcs():
        _gcs_upload_bytes(object_key, body, content_type="application/json")
        return
    _client(settings.s3_endpoint).put_object(
        Bucket=_bucket(),
        Key=object_key,
        Body=body,
        ContentType="application/json",
    )


def put_text_object(object_key: str, value: str) -> None:
    """Write UTF-8 text through the internal object-store endpoint."""
    body = value.encode()
    if _using_gcs():
        _gcs_upload_bytes(object_key, body, content_type="text/plain; charset=utf-8")
        return
    _client(settings.s3_endpoint).put_object(
        Bucket=_bucket(),
        Key=object_key,
        Body=body,
        ContentType="text/plain; charset=utf-8",
    )


def read_text_object_if_exists(object_key: str) -> str | None:
    """Read UTF-8 text, returning ``None`` when the object is not present."""
    try:
        return _read_object_bytes(object_key).decode("utf-8", errors="replace")
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def evaluation_live_log_key(evaluation_id: str, execution_number: int) -> str:
    """Stable live-log snapshot key for one evaluation execution."""
    return f"evaluations/{evaluation_id}/live/{execution_number}/runner.log"


def evaluation_artifact_prefix(evaluation_id: str) -> str:
    """Stable object prefix for all artifacts produced by one evaluation."""
    return f"evaluations/{evaluation_id}/artifacts/"


def _is_unsafe_artifact_path(relative_path: str) -> bool:
    # Raw split, not PurePosixPath.parts — the latter normalizes away "." and
    # "//" segments, which would let them slip through undetected. Empty parts
    # also reject leading/trailing/doubled slashes.
    if not relative_path or "\\" in relative_path:
        return True
    return any(part in {"", ".", ".."} for part in relative_path.split("/"))


def evaluation_artifact_key(evaluation_id: str, path: str) -> str:
    """Object key for an artifact path relative to an evaluation.

    Rejects traversal/absolute paths: S3-compatible gateways may normalize
    `..` segments in request URLs, so an unchecked caller-supplied path could
    resolve outside the evaluation's artifact prefix.
    """
    relative = path.lstrip("/")
    if _is_unsafe_artifact_path(relative):
        raise ValueError(f"unsafe artifact path: {path!r}")
    return f"{evaluation_artifact_prefix(evaluation_id)}{relative}"


def evaluation_archive_key(evaluation_id: str) -> str:
    """Stable object key for one evaluation's downloadable artifact bundle."""
    return f"evaluations/{evaluation_id}/{ARCHIVE_FILE_NAME}"


def evaluation_harbor_viewer_archive_key(evaluation_id: str) -> str:
    """Stable object key for one Harbor Viewer-compatible job archive."""
    return f"evaluations/{evaluation_id}/{HARBOR_VIEWER_ARCHIVE_FILE_NAME}"


def upload_file(path: Path, object_key: str, *, content_type: str) -> int:
    """Upload one local file to a stable object key and return its size."""
    size_bytes = path.stat().st_size
    if _using_gcs():
        with path.open("rb") as source:
            _gcs_upload_fileobj(
                source,
                object_key,
                content_type=content_type,
                size_bytes=size_bytes,
            )
        return size_bytes
    _client(settings.s3_endpoint).upload_file(
        str(path),
        _bucket(),
        object_key,
        ExtraArgs={"ContentType": content_type},
    )
    return size_bytes


def _is_secret_file(root: Path, path: Path) -> bool:
    rel = path.relative_to(root)
    return any(part in _SECRET_FILE_NAMES or part.endswith(".env") for part in rel.parts)


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.relative_to(root).as_posix() != ARTIFACT_MANIFEST_PATH
            and not _is_secret_file(root, path)
        ):
            yield path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _bytes_sha256(body: bytes) -> str:
    return f"sha256:{hashlib.sha256(body).hexdigest()}"


def _artifact_manifest(root: Path, files: list[Path]) -> bytes:
    entries = [
        {
            "path": source.relative_to(root).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": _file_sha256(source),
        }
        for source in files
    ]
    return _artifact_manifest_entries(entries)


def _artifact_manifest_entries(entries: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {
            "schema_version": "scaled-evals-artifacts-v1",
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "files": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _redacted_text_body(path: Path) -> bytes | None:
    if path.suffix.lower() not in _TEXT_ARTIFACT_SUFFIXES:
        return None
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    redacted = (
        redact_json_text(text, lines=path.suffix.lower() == ".jsonl")
        if path.suffix.lower() in {".json", ".jsonl"}
        else redact_secret_text(text)
    )
    if redacted == text:
        return None
    return redacted.encode("utf-8")


@contextmanager
def _staged_artifacts(root: Path) -> Iterator[tuple[Path, list[Path]]]:
    """Materialize immutable, already-redacted bytes for hashing and persistence."""
    with tempfile.TemporaryDirectory(prefix="scaled-evals-artifacts-") as tmp:
        staged_root = Path(tmp)
        for source in _iter_files(root):
            relative = source.relative_to(root)
            destination = staged_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            redacted = _redacted_text_body(source)
            if redacted is None:
                shutil.copy2(source, destination)
            else:
                destination.write_bytes(redacted)
                shutil.copystat(source, destination)
        yield staged_root, list(_iter_files(staged_root))


def sync_directory_to_prefix(root: Path | str, prefix: str) -> int:
    """Upload all regular files under ``root`` to ``prefix``.

    Returns the number of uploaded files. Missing roots are a no-op so failed
    evaluations with no materialized job dir can still be marked terminal.
    """
    root_path = Path(root)
    if not root_path.exists():
        return 0
    if _using_gcs():
        with _staged_artifacts(root_path) as (staged_root, files):
            _gcs_upload_bytes(
                f"{prefix.rstrip('/')}/{ARTIFACT_MANIFEST_PATH}",
                _artifact_manifest(staged_root, files),
                content_type="application/json",
            )
            for source in files:
                object_key = f"{prefix.rstrip('/')}/{source.relative_to(staged_root).as_posix()}"
                _gcs_upload_file(source, object_key)
        return len(files) + 1
    client = _client(settings.s3_endpoint)
    with _staged_artifacts(root_path) as (staged_root, files):
        client.put_object(
            Bucket=_bucket(),
            Key=f"{prefix.rstrip('/')}/{ARTIFACT_MANIFEST_PATH}",
            Body=_artifact_manifest(staged_root, files),
            ContentType="application/json",
        )
        for source in files:
            object_key = f"{prefix.rstrip('/')}/{source.relative_to(staged_root).as_posix()}"
            client.upload_file(str(source), _bucket(), object_key)
    return len(files) + 1


def replace_directory_at_prefix(root: Path | str, prefix: str) -> int:
    """Replace one stable object prefix with the files currently under ``root``.

    Deleting first means a partial upload can leave an incomplete newest result,
    but it cannot mix files from different evaluation executions.
    """
    normalized = f"{prefix.rstrip('/')}/"
    for item in list_objects(normalized):
        object_key = str(item.get("key") or "")
        if object_key.startswith(normalized):
            delete_object(object_key)
    return sync_directory_to_prefix(root, normalized)


def sync_evidence_files(root: Path | str, prefix: str) -> int:
    """Upload terminal provenance/SBOM bytes and rebuild the remote file manifest."""
    from scaled_evals.models.provenance import MANIFEST_FILE_NAME
    from scaled_evals.models.sbom import SBOM_FILE_NAME

    root_path = Path(root)
    if _using_gcs():
        for name in (SBOM_FILE_NAME, MANIFEST_FILE_NAME):
            source = root_path / name
            if not source.is_file():
                raise FileNotFoundError(source)
            _gcs_upload_bytes(
                f"{prefix.rstrip('/')}/{name}",
                source.read_bytes(),
                content_type="application/json",
            )
        _rebuild_remote_artifact_manifest(prefix)
        return 3
    client = _client(settings.s3_endpoint)
    for name in (SBOM_FILE_NAME, MANIFEST_FILE_NAME):
        source = root_path / name
        if not source.is_file():
            raise FileNotFoundError(source)
        client.put_object(
            Bucket=_bucket(),
            Key=f"{prefix.rstrip('/')}/{name}",
            Body=source.read_bytes(),
            ContentType="application/json",
        )
    _rebuild_remote_artifact_manifest(prefix)
    return 3


def _rebuild_remote_artifact_manifest(prefix: str) -> None:
    normalized = f"{prefix.rstrip('/')}/"
    entries: list[dict[str, Any]] = []
    for item in sorted(list_objects(normalized), key=lambda value: value["key"]):
        key = str(item["key"])
        relative = key[len(normalized) :]
        if not relative or relative == ARTIFACT_MANIFEST_PATH:
            continue
        body = _read_object_bytes(key)
        entries.append({"path": relative, "size_bytes": len(body), "sha256": _bytes_sha256(body)})
    if _using_gcs():
        _gcs_upload_bytes(
            f"{normalized}{ARTIFACT_MANIFEST_PATH}",
            _artifact_manifest_entries(entries),
            content_type="application/json",
        )
        return
    client = _client(settings.s3_endpoint)
    client.put_object(
        Bucket=_bucket(),
        Key=f"{normalized}{ARTIFACT_MANIFEST_PATH}",
        Body=_artifact_manifest_entries(entries),
        ContentType="application/json",
    )


def list_objects(prefix: str) -> list[dict[str, Any]]:
    """List object metadata below ``prefix`` using the internal endpoint."""
    if _using_gcs():
        objects: list[dict[str, Any]] = []
        params = {
            "prefix": prefix,
            "fields": "items(name,size,updated),nextPageToken",
        }
        while True:
            with httpx.Client(timeout=settings.gcs_request_timeout_seconds) as client:
                response = client.get(
                    _gcs_url(f"/storage/v1/b/{quote(_bucket(), safe='')}/o"),
                    params=params,
                    headers=_gcs_headers(),
                )
            _raise_for_gcs("ListObjects", response)
            page = response.json()
            for item in page.get("items", []):
                key = item.get("name")
                if not key:
                    continue
                objects.append(
                    {
                        "key": key,
                        "size_bytes": int(item.get("size") or 0),
                        "updated_at": item.get("updated"),
                    }
                )
            token = page.get("nextPageToken")
            if not token:
                return objects
            params["pageToken"] = token
    client = _client(settings.s3_endpoint)
    paginator = client.get_paginator("list_objects_v2")
    objects: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if not key:
                continue
            last_modified = item.get("LastModified")
            objects.append(
                {
                    "key": key,
                    "size_bytes": item.get("Size", 0),
                    "updated_at": last_modified.isoformat() if isinstance(last_modified, datetime) else last_modified,
                }
            )
    return objects


class ArchiveBuildError(RuntimeError):
    """Raised when a results archive cannot be built from synced artifacts."""


def _safe_archive_member_name(relative_path: str) -> str:
    if _is_unsafe_artifact_path(relative_path):
        raise ArchiveBuildError(f"unsafe artifact path: {relative_path!r}")
    return f"artifacts/{relative_path}"


def _add_local_file_to_archive(tar: tarfile.TarFile, root: Path, source: Path) -> int:
    relative_path = source.relative_to(root).as_posix()
    body = _redacted_text_body(source)
    if body is not None:
        info = tarfile.TarInfo(_safe_archive_member_name(relative_path))
        info.size = len(body)
        info.mode = 0o644
        info.mtime = int(source.stat().st_mtime)
        tar.addfile(info, io.BytesIO(body))
        return len(body)

    tar.add(
        source,
        arcname=_safe_archive_member_name(relative_path),
        recursive=False,
        filter=lambda info: _normalize_archive_info(info, source.stat().st_mtime),
    )
    return source.stat().st_size


def _normalize_archive_info(info: tarfile.TarInfo, mtime: float) -> tarfile.TarInfo:
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = int(mtime)
    return info


def _upload_archive(evaluation_id: str, tmp: Any, size_bytes: int) -> dict[str, Any]:
    dest_key = evaluation_archive_key(evaluation_id)
    tmp.seek(0)
    if _using_gcs():
        _gcs_upload_fileobj(
            tmp,
            dest_key,
            content_type="application/gzip",
            size_bytes=size_bytes,
        )
        return {"object_key": dest_key, "size_bytes": size_bytes}
    _client(settings.s3_endpoint).upload_fileobj(
        tmp,
        _bucket(),
        dest_key,
        ExtraArgs={"ContentType": "application/gzip"},
    )
    return {"object_key": dest_key, "size_bytes": size_bytes}


def build_evaluation_archive_from_directory(evaluation_id: str, root: Path | str) -> dict[str, Any]:
    """Create ``results.tar.gz`` directly from local artifacts, then upload it.

    This is the dispatch-worker happy path after an evaluation finishes: artifacts
    are still on the local worker filesystem, so we avoid listing and downloading
    the just-synced S3 objects only to re-compress them.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise ArchiveBuildError("no local artifacts found")
    with _staged_artifacts(root_path) as (staged_root, files):
        manifest = _artifact_manifest(staged_root, files)
        archive_file_count = len(files) + 1
        if archive_file_count > settings.evaluation_archive_max_files:
            raise ArchiveBuildError(
                f"archive would include {archive_file_count} files; limit is {settings.evaluation_archive_max_files}"
            )

        with tempfile.TemporaryFile() as tmp:
            source_bytes = len(manifest)
            with tarfile.open(fileobj=tmp, mode="w:gz") as tar:
                info = tarfile.TarInfo(_safe_archive_member_name(ARTIFACT_MANIFEST_PATH))
                info.size = len(manifest)
                info.mode = 0o644
                info.mtime = int(datetime.now(tz=UTC).timestamp())
                tar.addfile(info, io.BytesIO(manifest))
                for source in files:
                    source_bytes += _add_local_file_to_archive(tar, staged_root, source)
                    if source_bytes > settings.evaluation_archive_max_source_bytes:
                        raise ArchiveBuildError(
                            f"archive source bytes {source_bytes} exceed "
                            f"limit {settings.evaluation_archive_max_source_bytes}"
                        )
            size_bytes = tmp.tell()
            archive = _upload_archive(evaluation_id, tmp, size_bytes)

    return {
        **archive,
        "file_count": archive_file_count,
        "source_bytes": source_bytes,
    }


def build_evaluation_archive(evaluation_id: str) -> dict[str, Any]:
    """Create ``results.tar.gz`` from an evaluation's synced artifact objects.

    The API request path never calls this. It is intended for the dispatch /
    archive worker so compression and object-store reads do not occupy a FastAPI
    request thread.
    """
    source_prefix = evaluation_artifact_prefix(evaluation_id)
    objects = [item for item in list_objects(source_prefix) if item.get("key", "").startswith(source_prefix)]
    if not objects:
        raise ArchiveBuildError("no synced artifacts found")
    if len(objects) > settings.evaluation_archive_max_files:
        raise ArchiveBuildError(
            f"archive would include {len(objects)} files; limit is {settings.evaluation_archive_max_files}"
        )
    total_source_bytes = sum(int(item.get("size_bytes") or 0) for item in objects)
    if total_source_bytes > settings.evaluation_archive_max_source_bytes:
        raise ArchiveBuildError(
            f"archive source bytes {total_source_bytes} exceed limit {settings.evaluation_archive_max_source_bytes}"
        )

    client = None if _using_gcs() else _client(settings.s3_endpoint)
    with tempfile.TemporaryFile() as tmp:
        with tarfile.open(fileobj=tmp, mode="w:gz") as tar:
            for item in sorted(objects, key=lambda obj: obj["key"]):
                key = item["key"]
                relative_path = key[len(source_prefix) :]
                info = tarfile.TarInfo(_safe_archive_member_name(relative_path))
                info.size = int(item.get("size_bytes") or 0)
                info.mode = 0o644
                updated_at = item.get("updated_at")
                if isinstance(updated_at, str):
                    try:
                        info.mtime = int(datetime.fromisoformat(updated_at).timestamp())
                    except ValueError:
                        info.mtime = int(datetime.now(tz=UTC).timestamp())
                else:
                    info.mtime = int(datetime.now(tz=UTC).timestamp())
                if _using_gcs():
                    body = io.BytesIO(_gcs_read_object_bytes(key))
                    tar.addfile(info, body)
                else:
                    assert client is not None
                    with closing(client.get_object(Bucket=_bucket(), Key=key)["Body"]) as body:
                        tar.addfile(info, body)
        size_bytes = tmp.tell()
        archive = _upload_archive(evaluation_id, tmp, size_bytes)
    return {
        **archive,
        "file_count": len(objects),
        "source_bytes": total_source_bytes,
    }


def upload_context_archive(archive_path: Path, object_key: str) -> None:
    """Upload a local build-context archive to the object store.

    Server-side put (e.g. staging a Switchyard context so Cloud Build can
    consume it as a GCS ``storageSource``), so it targets the internal endpoint.
    """
    if _using_gcs():
        _gcs_upload_file(archive_path, object_key)
        return
    _client(settings.s3_endpoint).upload_file(str(archive_path), _bucket(), object_key)


def download_object(object_key: str, dest_path: str) -> None:
    """Download an object to a local file, using the internal endpoint.

    Server-side fetch (e.g. the task build downloading a tarball), so it
    targets `s3_endpoint` — the in-cluster/in-compose address — not the public
    one baked into presigned URLs.
    """
    if _using_gcs():
        with (
            httpx.Client(timeout=None) as client,
            client.stream(
                "GET",
                _gcs_media_url(object_key),
                params={"alt": "media"},
                headers=_gcs_headers(),
            ) as response,
        ):
            _raise_for_gcs("GetObject", response)
            with Path(dest_path).open("wb") as output:
                for chunk in response.iter_bytes():
                    output.write(chunk)
        return
    _client(settings.s3_endpoint).download_file(_bucket(), object_key, dest_path)


def ensure_bucket() -> str:
    """Create the bucket when it is missing, so a fresh object store works unattended.

    A new RustFS or MinIO volume starts empty, and every write path here assumes the
    bucket already exists — without this, `/v1/readyz` reports `object_store: fail`
    until someone runs `s3 mb` out of band.

    Deliberately tolerant of a refusal: a managed store commonly denies `CreateBucket`
    to the runtime principal and expects the bucket to be pre-provisioned, which is
    also the platform's own posture for Files. Losing a race with another replica is
    equally benign. `check_bucket` stays the single source of truth for whether
    storage is actually usable; this only removes the manual step where it can.

    Returns:
        str: the bucket name, once it exists or was already present.

    """
    bucket = _bucket()
    if _using_gcs():
        # GCS buckets carry location/class/IAM policy that we must not invent.
        return bucket
    client = _client(settings.s3_endpoint, _READYZ_CONFIG)
    try:
        client.head_bucket(Bucket=bucket)
        return bucket
    except ClientError:
        pass
    try:
        client.create_bucket(Bucket=bucket)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in _BUCKET_EXISTS_CODES:
            raise
    return bucket


def check_bucket() -> None:
    """Confirm the object store is reachable and the bucket exists, using the
    internal endpoint and our credentials. Raises on failure.

    Backend-agnostic: `head_bucket` is a standard S3 operation, so this works
    against RustFS in dev and a managed S3-compatible store in prod — no
    dependency on RustFS's `/health` path.
    """
    if _using_gcs():
        with httpx.Client(timeout=settings.gcs_request_timeout_seconds) as client:
            response = client.get(
                _gcs_url(f"/storage/v1/b/{quote(_bucket(), safe='')}/o"),
                params={"maxResults": "1", "fields": "items(name)"},
                headers=_gcs_headers(),
            )
        _raise_for_gcs("HeadBucket", response)
        return
    _client(settings.s3_endpoint, _READYZ_CONFIG).head_bucket(Bucket=_bucket())
