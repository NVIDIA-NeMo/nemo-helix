# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP range request handling utilities."""

import logging
import re

from fastapi import APIRouter
from nhx.core.files.app.backends import ByteRange
from nhx.core.files.exceptions import InvalidRangeError
from starlette.status import HTTP_200_OK, HTTP_206_PARTIAL_CONTENT

logger = logging.getLogger(__name__)
router = APIRouter()


# `bytes=start-end` format
RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)")


def parse_range_header(range_header: str | None, file_size: int) -> ByteRange | None:
    """Parse HTTP Range header"""

    if not range_header:
        return None

    match = RANGE_PATTERN.match(range_header)
    if not match:
        raise InvalidRangeError(f"Could not match range header provided: {range_header}")

    start_str, end_str = match.groups()

    if start_str and end_str:
        # bytes=100-200
        start = int(start_str)
        end = int(end_str)
    elif start_str and not end_str:
        # bytes=100-
        start = int(start_str)
        end = file_size - 1
    elif not start_str and end_str:
        # bytes=-500 (last 500 bytes)
        start = max(0, file_size - int(end_str))
        end = file_size - 1
    else:
        # bytes=-
        raise InvalidRangeError(f"Invalid range header provided: {range_header}")

    if start < 0 or end >= file_size or start > end:
        raise InvalidRangeError(f"Invalid range for {start=} and {end=}")

    return ByteRange(start, end)


def download_response_status_and_headers(byte_range: ByteRange | None, file_size: int) -> tuple[int, dict[str, str]]:
    """Generate appropriate headers for file download responses."""

    if byte_range is not None:
        content_length = byte_range.end - byte_range.start + 1
        headers = {
            "accept-ranges": "bytes",
            "content-range": f"bytes {byte_range.start}-{byte_range.end}/{file_size}",
            "content-length": str(content_length),
        }
        return HTTP_206_PARTIAL_CONTENT, headers
    else:
        return HTTP_200_OK, {"accept-ranges": "bytes", "content-length": str(file_size)}
