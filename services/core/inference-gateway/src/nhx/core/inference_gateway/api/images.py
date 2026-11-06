# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve fileset image references in chat requests to inline ``data:`` URLs.

A chat completions image part may name a file in a fileset instead of inlining it::

    {"type": "image_url", "image_url": {"url": "fileset://my-ws/my-images#cats/1.png"}}

The gateway downloads the file with the caller's own client (so Files enforces the
caller's permissions) and replaces the URL with a base64 ``data:`` URL before any
middleware or the model sees the request. The gateway never reads a host path.
"""

import asyncio
import base64
import logging
import re
from typing import Any, NoReturn
from urllib.parse import urlsplit

from fastapi import HTTPException
from fastapi import status as http_status
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import (
    AuthenticationError,
    NemoClientError,
    NotFoundError,
    PermissionDeniedError,
)
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.refs import FILESET_REF_PATTERN

logger = logging.getLogger(__name__)

FILESET_URL_PREFIX = "fileset://"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_CONCURRENT_DOWNLOADS = 4
IMAGE_MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

_FILESET_REF = re.compile(FILESET_REF_PATTERN)
_ACCEPTED_FORMS = (
    "Image URLs must be a data: URL, an http(s) URL, or a fileset reference "
    "'fileset://<workspace>/<fileset>#<path>' (the workspace defaults to the request's)."
)
_CHUNK_BYTES = 1024 * 1024


def _image_parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``image_url`` objects of every image content part in ``body["messages"]``."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    parts = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                parts.append(image_url)
    return parts


def _is_host_path(url: str) -> bool:
    # ``C:\...`` parses with a one-letter scheme, so treat that as a path too.
    scheme = urlsplit(url).scheme
    return scheme in ("", "file") or len(scheme) == 1


def _parse_fileset_url(url: str, request_workspace: str) -> tuple[str, str, str, str]:
    """Split ``fileset://[workspace/]fileset#path`` into ``(workspace, fileset, path, mime)``."""
    ref = url.removeprefix(FILESET_URL_PREFIX)
    location = ref.partition("#")[0]
    if "/" not in location:
        ref = f"{request_workspace}/{ref}"
    if not _FILESET_REF.fullmatch(ref):
        raise HTTPException(
            http_status.HTTP_400_BAD_REQUEST, detail=f"Invalid fileset image reference {url!r}. {_ACCEPTED_FORMS}"
        )
    location, _, path = ref.partition("#")
    workspace, _, fileset = location.partition("/")
    basename = path.rpartition("/")[2]
    mime = IMAGE_MIME_TYPES.get(basename.rpartition(".")[2].lower()) if "." in basename else None
    if mime is None:
        raise HTTPException(
            http_status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image file type for {url!r}. Supported extensions: {', '.join(sorted(IMAGE_MIME_TYPES))}."
            ),
        )
    return workspace, fileset, path, mime


def _raise_download_error(source: str, exc: NemoClientError) -> NoReturn:
    if isinstance(exc, NotFoundError):
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail=f"Image file {source} not found.") from exc
    if isinstance(exc, (PermissionDeniedError, AuthenticationError)):
        raise HTTPException(exc.status_code, detail=f"Not permitted to read image file {source}: {exc.detail}") from exc
    raise HTTPException(
        http_status.HTTP_502_BAD_GATEWAY, detail=f"Could not download image file {source} from the Files service."
    ) from exc


async def _download_image(files_client: AsyncFilesClient, workspace: str, fileset: str, path: str) -> bytes:
    source = f"{workspace}/{fileset}#{path}"
    data = bytearray()
    try:
        response = await files_client.download_file(workspace=workspace, name=fileset, path=path)
        async with response.stream(_CHUNK_BYTES) as chunks:
            async for chunk in chunks:
                data += chunk
                if len(data) > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        http_status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Image file {source} is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MiB.",
                    )
    except NemoClientError as exc:
        logger.warning("Fileset image download failed for %s: %s", source, exc)
        _raise_download_error(source, exc)
    return bytes(data)


async def resolve_fileset_images(
    body: dict[str, Any], *, workspace: str, request_nemo_client: AsyncNemoClient | None
) -> None:
    """Replace every ``fileset://`` image URL in ``body`` with a base64 ``data:`` URL, in place.

    ``data:`` and ``http(s)`` URLs are left as they are; ``file://`` URLs and bare paths
    are rejected. Files are downloaded with ``request_nemo_client`` so the caller's
    permissions apply.
    """
    pending: list[tuple[dict[str, Any], str, str, str, str]] = []
    for image_url in _image_parts(body):
        url = image_url["url"]
        if url.startswith(FILESET_URL_PREFIX):
            pending.append((image_url, *_parse_fileset_url(url, workspace)))
        elif not url.startswith("data:") and _is_host_path(url):
            raise HTTPException(http_status.HTTP_400_BAD_REQUEST, detail=f"Unsupported image URL. {_ACCEPTED_FORMS}")
    if not pending:
        return
    if request_nemo_client is None:
        # Never fall back to the gateway's own credentials: that would bypass the caller's permissions.
        raise HTTPException(
            http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fileset image references cannot be resolved: no caller-scoped client for this request.",
        )

    files_client = AsyncFilesClient.from_client(request_nemo_client)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async def resolve(image_url: dict[str, Any], file_workspace: str, fileset: str, path: str, mime: str) -> None:
        async with semaphore:
            data = await _download_image(files_client, file_workspace, fileset, path)
        image_url["url"] = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    try:
        async with asyncio.TaskGroup() as task_group:
            for item in pending:
                task_group.create_task(resolve(*item))
    except ExceptionGroup as group:
        # Surface the first failure as the response; the rest were cancelled or failed the same way.
        raise group.exceptions[0]
