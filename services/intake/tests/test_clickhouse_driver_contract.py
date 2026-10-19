# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signature contract for the clickhouse-connect surfaces Intake and its examples use."""

import importlib
import inspect

import clickhouse_connect
from aiohttp import ClientSession
from clickhouse_connect.driver import AsyncClient
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.exceptions import ClickHouseError
from clickhouse_connect.driver.external import ExternalData


def test_sync_factory_keeps_example_and_ping_parameters() -> None:
    parameters = inspect.signature(clickhouse_connect.get_client).parameters

    assert inspect.isclass(Client)
    for name in ("dsn", "host", "port", "username", "password", "database"):
        assert name in parameters
    accepts_connect_timeout = "connect_timeout" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    assert accepts_connect_timeout
    importlib.import_module("clickhouse_connect.driver.httpclient")


def test_async_factory_is_native_and_accepts_intake_parameters() -> None:
    parameters = inspect.signature(clickhouse_connect.get_async_client).parameters

    assert inspect.iscoroutinefunction(clickhouse_connect.get_async_client)
    for name in ("host", "port", "secure", "username", "password", "database"):
        assert name in parameters
    assert "executor_threads" not in parameters
    assert "executor" not in parameters


def test_async_methods_keep_intake_call_shape() -> None:
    query = inspect.signature(AsyncClient.query).parameters
    command = inspect.signature(AsyncClient.command).parameters
    insert = inspect.signature(AsyncClient.insert).parameters
    close = inspect.signature(AsyncClient.close)

    assert {"parameters", "settings", "external_data"} <= set(query)
    assert {"parameters", "settings"} <= set(command)
    assert {"column_names", "database"} <= set(insert)
    assert inspect.iscoroutinefunction(AsyncClient.close)
    assert list(close.parameters) == ["self"]
    external = inspect.signature(ExternalData).parameters
    assert {"file_name", "data", "fmt", "structure"} <= set(external)
    assert issubclass(ClickHouseError, Exception)


def test_lifecycle_probe_can_see_the_driver_session() -> None:
    assert "_session" in inspect.getsource(AsyncClient)
    assert "_loop" in inspect.getsource(ClientSession.__init__)
    assert isinstance(inspect.getattr_static(ClientSession, "closed"), property)
