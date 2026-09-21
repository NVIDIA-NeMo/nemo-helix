# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Files service.

Wraps the endpoint functions from ``files.endpoints`` as direct methods
using the ``method()`` descriptor, following the example-plugin pattern.

Usage::

    client = FilesClient(base_url="...", workspace="default")
    resp = client.create_fileset(body=CreateFilesetRequest(name="my-fs"))
    fileset = resp.data()

    files_resp = client.upload_file(name="my-fs", path="data.txt", content=b"hello")
"""

from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method
from nemo_platform_plugin.files import endpoints


class _FilesMethods:
    create_fileset = method(endpoints.create_fileset)
    list_filesets = method(endpoints.list_filesets)
    get_fileset = method(endpoints.get_fileset)
    update_fileset = method(endpoints.update_fileset)
    delete_fileset = method(endpoints.delete_fileset)
    refresh_fileset = method(endpoints.refresh_fileset)
    list_files = method(endpoints.list_files)
    upload_file = method(endpoints.upload_file)
    download_file = method(endpoints.download_file)
    delete_file = method(endpoints.delete_file)
    upload_otlp_logs = method(endpoints.upload_otlp_logs)
    query_otlp_logs = method(endpoints.query_otlp_logs)


class FilesClient(_FilesMethods, NemoClient):
    """Sync client for the Files service API."""


class AsyncFilesClient(_FilesMethods, AsyncNemoClient):
    """Async client for the Files service API."""
