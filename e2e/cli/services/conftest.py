# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared fixtures for CLI service E2E tests."""

from collections.abc import Generator

import pytest
from docker.errors import DockerException
from nhx.testing import ensure_mock_nim_image
from nhx.testing.docker import MOCK_NIM_IMAGE_TAG

import docker

MOCK_NIM_IMAGE_NAME = f"mock-nim-e2e:{MOCK_NIM_IMAGE_TAG}"


@pytest.fixture(scope="session")
def mock_nim_image() -> Generator[str, None, None]:
    """Build or retrieve the nginx-based mock NIM image for E2E tests."""
    try:
        docker_client = docker.from_env()
        docker_client.ping()
    except DockerException as e:
        pytest.skip(f"Docker is required for model deployment E2E tests: {e}")
    try:
        yield ensure_mock_nim_image(docker_client, MOCK_NIM_IMAGE_NAME)
    finally:
        docker_client.close()
