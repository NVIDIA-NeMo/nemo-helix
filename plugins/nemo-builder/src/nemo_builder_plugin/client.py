# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A typed client for the builder's routes, which the push step calls."""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, TypedDict

from nemo_builder_plugin.completion import CompleteRequest
from nemo_builder_plugin.entities import ContainerImage
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.endpoint import get, post
from nemo_helix_plugin.client.method import method

_IMAGES = "/apis/builder/v2/workspaces/{workspace}/container-images"


class ListContainerImagesQueryParams(TypedDict, total=False):
    job: str
    status: Literal["pending", "ready", "failed"]
    page: int
    page_size: int


@get(_IMAGES)
@abstractmethod
def list_container_images(
    *, workspace: str | None = None, query_params: ListContainerImagesQueryParams | None = None
) -> list[ContainerImage]: ...


@get(_IMAGES + "/{name}")
@abstractmethod
def get_container_image(*, workspace: str | None = None, name: str) -> ContainerImage: ...


@post(_IMAGES + "/{name}/complete")
@abstractmethod
def complete_container_image(*, workspace: str | None = None, name: str, body: CompleteRequest) -> ContainerImage: ...


class _BuilderMethods:
    list_container_images = method(list_container_images)
    get_container_image = method(get_container_image)
    complete_container_image = method(complete_container_image)


class BuilderClient(_BuilderMethods, NemoClient):
    """Sync client for the builder's routes."""


class AsyncBuilderClient(_BuilderMethods, AsyncNemoClient):
    """Async client for the builder's routes."""
