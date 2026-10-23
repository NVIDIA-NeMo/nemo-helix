# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request
from nhx.core.jobs.api.dependencies import dep_system_entity_client


def _request(**state: object) -> Request:
    app = FastAPI()
    for name, value in state.items():
        setattr(app.state, name, value)
    return Request({"type": "http", "app": app})


def _jobs_service() -> SimpleNamespace:
    return SimpleNamespace(name="jobs", dependency_provider=Mock())


def test_system_entity_client_acts_as_jobs_in_platform_runner_app() -> None:
    """The platform runner merges services into one app, which keeps no ``app.state.service``."""
    service = _jobs_service()

    client = dep_system_entity_client(_request(jobs_service=service))

    service.dependency_provider.get_service_entity_client.assert_called_once_with("jobs")
    assert client is service.dependency_provider.get_service_entity_client.return_value


def test_system_entity_client_acts_as_jobs_in_standalone_app() -> None:
    service = _jobs_service()

    client = dep_system_entity_client(_request(service=service))

    service.dependency_provider.get_service_entity_client.assert_called_once_with("jobs")
    assert client is service.dependency_provider.get_service_entity_client.return_value


def test_system_entity_client_refuses_app_without_jobs_service() -> None:
    """Falling back to the caller's client would check the system workspace as the caller."""
    with pytest.raises(RuntimeError, match="not registered"):
        dep_system_entity_client(_request())
