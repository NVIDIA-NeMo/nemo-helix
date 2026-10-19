# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the in-process Data Designer integration tests.

Every fixture hangs off one ``client_context`` (the in-process platform), so a
test asks for the client it needs and never builds one from a raw ``NemoClient``.
"""

from collections.abc import Generator

import nemo_data_designer_plugin.testing.utils as u
import pytest
from nemo_data_designer_plugin.sdk.resources import AsyncDataDesignerResource, DataDesignerResource
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.secrets.client import SecretsClient
from nhx.testing import ClientContext


@pytest.fixture
def client_context() -> Generator[ClientContext]:
    with u.make_mock_client_context() as client_context:
        yield client_context


@pytest.fixture
def client(client_context: ClientContext) -> NemoClient:
    return client_context.client


@pytest.fixture
def async_client(client_context: ClientContext) -> AsyncNemoClient:
    return client_context.async_client


@pytest.fixture
def files_client(client: NemoClient) -> FilesClient:
    return FilesClient.from_client(client)


@pytest.fixture
def secrets_client(client: NemoClient) -> SecretsClient:
    return SecretsClient.from_client(client)


@pytest.fixture
def data_designer(client: NemoClient) -> DataDesignerResource:
    return DataDesignerResource(client)


@pytest.fixture
def async_data_designer(async_client: AsyncNemoClient) -> AsyncDataDesignerResource:
    return AsyncDataDesignerResource(async_client)


@pytest.fixture
def mock_providers(client_context: ClientContext) -> Generator[None]:
    with u.setup_mock_providers(client_context):
        yield


@pytest.fixture
def mock_secret(client_context: ClientContext) -> Generator[None]:
    with u.setup_mock_secret(client_context):
        yield


@pytest.fixture
def mock_file(client_context: ClientContext) -> Generator[None]:
    with u.setup_mock_file(client_context):
        yield
