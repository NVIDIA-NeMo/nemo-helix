# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static return-type contracts for the typed client.

The functions in this module are checked by ``ty`` but intentionally not
collected by pytest. They ensure endpoint annotations flow through prepared
requests and both client implementations without being erased to ``Any``.
"""

from collections.abc import AsyncIterator, Iterator
from typing import assert_type

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient, _parse_json_body
from nemo_helix_plugin.client.endpoint import get
from nemo_helix_plugin.client.response import (
    AsyncNemoPaginatedResponse,
    AsyncNemoStreamResponse,
    NemoPaginatedResponse,
    NemoResponse,
    NemoStreamResponse,
    PageResult,
)
from nemo_helix_plugin.client.types import (
    CursorPagination,
    CursorPaginationMetadata,
    OffsetPagination,
    OffsetPaginationMetadata,
    Paginated,
    PreparedRequest,
    Stream,
)
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.files.types import FilesetOutput
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobLog
from nemo_helix_plugin.jobs.types import HelixJobResponse, HelixJobStepWithContext
from nemo_helix_plugin.secrets.client import AsyncSecretsClient, SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretResponse
from pydantic import BaseModel


class Item(BaseModel):
    name: str


@get("/items/{name}")
def get_item(*, name: str) -> Item:
    raise NotImplementedError


@get("/items")
def get_item_list() -> list[Item]:
    raise NotImplementedError


@get("/items/pages")
def get_item_pages() -> Paginated[Item]:
    raise NotImplementedError


@get("/items/cursor-pages")
def get_cursor_item_pages() -> Paginated[Item, CursorPagination]:
    raise NotImplementedError


@get("/items/stream")
def get_item_stream() -> Stream[Item]:
    raise NotImplementedError


def _check_sync_return_types(client: NemoClient) -> None:
    item_request = get_item(name="one")
    assert_type(item_request, PreparedRequest[Item])
    assert_type(client.send(item_request), NemoResponse[Item])

    list_request = get_item_list()
    assert_type(list_request, PreparedRequest[list[Item]])
    assert_type(client.send(list_request), NemoResponse[list[Item]])
    pages = client.send(get_item_pages())
    assert_type(pages, NemoPaginatedResponse[Item, OffsetPagination])
    assert_type(pages.page(), PageResult[Item, OffsetPaginationMetadata])
    assert_type(pages.items(), Iterator[Item])
    assert_type(pages.pages(), Iterator[PageResult[Item, OffsetPaginationMetadata]])
    cursor_pages = client.send(get_cursor_item_pages())
    assert_type(cursor_pages, NemoPaginatedResponse[Item, CursorPagination])
    assert_type(cursor_pages.page(), PageResult[Item, CursorPaginationMetadata])
    assert_type(cursor_pages.items(), Iterator[Item])
    assert_type(cursor_pages.pages(), Iterator[PageResult[Item, CursorPaginationMetadata]])
    assert_type(client.send(get_item_stream()), NemoStreamResponse[Item])

    assert_type(_parse_json_body(Item, {"name": "one"}), Item)
    assert_type(_parse_json_body(list[Item], [{"name": "one"}]), list[Item])


async def _check_async_return_types(client: AsyncNemoClient) -> None:
    assert_type(await client.send(get_item(name="one")), NemoResponse[Item])
    assert_type(await client.send(get_item_list()), NemoResponse[list[Item]])
    pages = await client.send(get_item_pages())
    assert_type(pages, AsyncNemoPaginatedResponse[Item, OffsetPagination])
    assert_type(pages.page(), PageResult[Item, OffsetPaginationMetadata])
    assert_type(pages.items(), AsyncIterator[Item])
    assert_type(pages.pages(), AsyncIterator[PageResult[Item, OffsetPaginationMetadata]])
    cursor_pages = await client.send(get_cursor_item_pages())
    assert_type(cursor_pages, AsyncNemoPaginatedResponse[Item, CursorPagination])
    assert_type(cursor_pages.page(), PageResult[Item, CursorPaginationMetadata])
    assert_type(cursor_pages.items(), AsyncIterator[Item])
    assert_type(cursor_pages.pages(), AsyncIterator[PageResult[Item, CursorPaginationMetadata]])
    assert_type(await client.send(get_item_stream()), AsyncNemoStreamResponse[Item])


def _check_jobs_return_types(client: JobsClient) -> None:
    logs = client.list_job_logs(name="job")
    assert_type(logs, NemoPaginatedResponse[HelixJobLog, CursorPagination])
    assert_type(logs.page(), PageResult[HelixJobLog, CursorPaginationMetadata])


async def _check_async_jobs_return_types(client: AsyncJobsClient) -> None:
    logs = await client.list_job_logs(name="job")
    assert_type(logs, AsyncNemoPaginatedResponse[HelixJobLog, CursorPagination])
    assert_type(logs.page(), PageResult[HelixJobLog, CursorPaginationMetadata])


def _check_offset_client_return_types(
    files: FilesClient,
    secrets: SecretsClient,
    jobs: JobsClient,
) -> None:
    filesets = files.list_filesets()
    assert_type(filesets, NemoPaginatedResponse[FilesetOutput, OffsetPagination])
    assert_type(filesets.page(), PageResult[FilesetOutput, OffsetPaginationMetadata])

    secret_pages = secrets.list_secrets()
    assert_type(secret_pages, NemoPaginatedResponse[HelixSecretResponse, OffsetPagination])
    assert_type(secret_pages.page(), PageResult[HelixSecretResponse, OffsetPaginationMetadata])

    job_pages = jobs.list_jobs()
    assert_type(job_pages, NemoPaginatedResponse[HelixJobResponse, OffsetPagination])
    assert_type(job_pages.page(), PageResult[HelixJobResponse, OffsetPaginationMetadata])

    step_pages = jobs.list_steps(name="job")
    assert_type(step_pages, NemoPaginatedResponse[HelixJobStepWithContext, OffsetPagination])
    assert_type(step_pages.page(), PageResult[HelixJobStepWithContext, OffsetPaginationMetadata])


async def _check_async_offset_client_return_types(
    files: AsyncFilesClient,
    secrets: AsyncSecretsClient,
    jobs: AsyncJobsClient,
) -> None:
    filesets = await files.list_filesets()
    assert_type(filesets, AsyncNemoPaginatedResponse[FilesetOutput, OffsetPagination])
    assert_type(filesets.page(), PageResult[FilesetOutput, OffsetPaginationMetadata])

    secret_pages = await secrets.list_secrets()
    assert_type(secret_pages, AsyncNemoPaginatedResponse[HelixSecretResponse, OffsetPagination])
    assert_type(secret_pages.page(), PageResult[HelixSecretResponse, OffsetPaginationMetadata])

    job_pages = await jobs.list_jobs()
    assert_type(job_pages, AsyncNemoPaginatedResponse[HelixJobResponse, OffsetPagination])
    assert_type(job_pages.page(), PageResult[HelixJobResponse, OffsetPaginationMetadata])

    step_pages = await jobs.list_steps(name="job")
    assert_type(step_pages, AsyncNemoPaginatedResponse[HelixJobStepWithContext, OffsetPagination])
    assert_type(step_pages.page(), PageResult[HelixJobStepWithContext, OffsetPaginationMetadata])
