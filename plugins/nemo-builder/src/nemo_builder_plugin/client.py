# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A typed client for the builder's own routes -- the ones the build plane and the broker call.

Two callers, and both call *as the submitter* or anonymously, never as a platform identity of
their own:

- the push step reads its rows and delivers signatures through its Jobs delegation, the way
  ``fetch`` reads Files;
- the credential broker reads a job's ``pending`` rows with the token it exchanged the step's own
  token for, or anonymously where platform auth is off.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, TypedDict

from nemo_builder_plugin.completion import SignatureDelivery
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


@post(_IMAGES + "/{name}/signature")
@abstractmethod
def deliver_signature(*, workspace: str | None = None, name: str, body: SignatureDelivery) -> ContainerImage:
    """Hand the control plane the signature for one row; it verifies it, and makes the row ``ready``."""
    ...


class _BuilderMethods:
    list_container_images = method(list_container_images)
    get_container_image = method(get_container_image)
    deliver_signature = method(deliver_signature)


class BuilderClient(_BuilderMethods, NemoClient):
    """Sync client for the builder's routes."""


class AsyncBuilderClient(_BuilderMethods, AsyncNemoClient):
    """Async client for the builder's routes."""
